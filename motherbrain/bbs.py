"""MotherBrain as a bulletin board system, answering telnet on port 23.

A BBS was one machine with a modem on it, and you were either connected or
you were not. It had a menu, message bases, file areas, doors to play, and
a teleconference where whoever else happened to be dialled in was there
with you. Everything was ANSI: sixteen colours and the IBM PC's block
glyphs, drawn at whatever speed your modem managed.

This is that, with a language model behind the sysop's chair. The main
menu is the same menu `mb console` shows - the four things MotherBrain does
plus the window - and around it is the board: chat rooms across nodes, five
doors, a file area that really transfers files over XMODEM, a message base
MotherBrain will answer threads in, and a gallery where you hand it a
picture and it tells you what it thinks it is looking at.

What it is not is a way in. Telnet is plaintext and always was: it is
bound to the loopback interface unless a password is set, everything that
touches the filesystem or a shell is refused outright to a caller, and
training the model is the sysop's key alone. The BBS aesthetic is 1985.
The exposure model is not.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from motherbrain import ansi as A
from motherbrain import doors, xmodem
from motherbrain.telnet import Telnet, escape, prefers_cp437

DEFAULT_PORT = 23
BOARD_NAME = "MOTHERBRAIN"
SYSOP = "MotherBrain"

# Massively multiuser means the model is the bottleneck, not the sockets.
# asyncio holds thousands of idle telnet connections on one thread without
# noticing; what it cannot do is run two forward passes at once. So the node
# count is generous, the model sits behind one lock with a visible queue, and
# the limits that matter are the ones stopping a single caller monopolising
# either of them.
MAX_CALLERS = 512
MAX_PER_ADDRESS = 8
IDLE_SECONDS = 30 * 60

# A caller can feed the corpus, and everything fed can be generated back out
# later. Both halves of that are capped: one contribution, and a session's
# worth of them.
MAX_FEED = 64 * 1024
MAX_FEED_PER_CALL = 256 * 1024

# Nothing that touches the filesystem or a shell is reachable from a caller.
# `commands.LOCAL_ONLY` is the same set the HTTP server refuses, for the same
# reason: over a wire it is remote code execution, not convenience.
from motherbrain.commands import LOCAL_ONLY  # noqa: E402


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---- one connection ---------------------------------------------------------

class Hangup(Exception):
    """The caller dropped carrier."""


class Caller:
    """One node: a socket, a screen, and whoever is sitting at the far end."""

    def __init__(self, reader, writer, board, node: int) -> None:
        self.reader, self.writer = reader, writer
        self.board, self.node = board, node
        self.tn = Telnet()
        self.handle = ""
        self.sysop = False
        self.columns, self.rows = 80, 24
        self.encoding = "utf-8"
        self.baud = 0                       # 0 = as fast as the socket goes
        self.room = ""
        self.connected_at = time.time()
        self.idle_timeout = IDLE_SECONDS
        self.fed = 0
        self.keys: deque[str] = deque()
        self.inbox: asyncio.Queue = asyncio.Queue()
        self._read_task: asyncio.Task | None = None
        self._inbox_task: asyncio.Task | None = None
        self._raw = bytearray()
        # A transfer is bytes, not text. In raw mode incoming data skips the
        # decoder entirely - putting a .pt file through a UTF-8 decode is how
        # you get a download that arrives corrupted and blames the network.
        self.raw_mode = False
        self.rawbuf = bytearray()
        self._prompt = ""                   # redrawn when a message arrives
        self._buffer = ""
        peer = writer.get_extra_info("peername")
        self.address = peer[0] if peer else "unknown"
        self.local = self.address in ("127.0.0.1", "::1")

    # -- output ---------------------------------------------------------------

    async def send(self, text: str) -> None:
        """Write to the wire, in the caller's encoding, at the caller's baud."""
        if not text:
            return
        data = escape(text.encode(self.encoding, "replace"))
        try:
            if self.baud:
                # A modem moved ten bits per character: eight data, one start,
                # one stop. Nobody needs this. Everybody who had one wants it.
                per_second = self.baud / 10
                step = max(1, int(per_second / 20))
                for i in range(0, len(data), step):
                    self.writer.write(data[i:i + step])
                    await self.writer.drain()
                    await asyncio.sleep(step / per_second)
            else:
                self.writer.write(data)
                await self.writer.drain()
        except (ConnectionError, RuntimeError) as exc:
            raise Hangup(str(exc)) from exc

    async def line(self, text: str = "") -> None:
        await self.send(text + "\r\n")

    async def art(self, text: str) -> None:
        await self.send(text.replace("\r\n", "\n").replace("\n", "\r\n") + "\r\n")

    async def cls(self) -> None:
        await self.send(A.CLS)

    async def pause(self, text: str = "press any key") -> None:
        await self.send(f"{A.GREY}  [{A.HW}{text}{A.GREY}]{A.RESET}")
        await self.key()
        await self.send("\r" + A.CLEAR_LINE)

    def tell(self, text: str) -> None:
        """Deliver a line from elsewhere - another node, or the sysop."""
        self.inbox.put_nowait(text)

    # -- input ----------------------------------------------------------------

    async def key(self) -> str:
        """One keystroke. Arrow keys arrive as UP/DOWN/LEFT/RIGHT.

        A caller who has wandered off is disconnected rather than left
        holding a node: on a board sized for hundreds, the nodes lost to
        people who closed the window without logging off are the ones that
        run out first.
        """
        while not self.keys:
            try:
                await asyncio.wait_for(self._wait(), self.idle_timeout)
            except asyncio.TimeoutError:
                raise Hangup("idle too long") from None
        return self.keys.popleft()

    async def ask(self, prompt: str, limit: int = 200, mask: bool = False,
                  default: str = "") -> str:
        """A line of input, edited on the server. Ctrl-C gives up on it."""
        await self.send(prompt)
        self._prompt, self._buffer = prompt, ""
        try:
            while True:
                key = await self.key()
                if key in ("\r", "\n"):
                    await self.send("\r\n")
                    return self._buffer or default
                if key == "\x03":                       # ctrl-c
                    await self.send("\r\n")
                    return ""
                if key in ("\x08", "\x7f"):
                    if self._buffer:
                        self._buffer = self._buffer[:-1]
                        await self.send("\b \b")
                    continue
                if len(key) > 1 or key < " ":           # arrows, other controls
                    continue
                if len(self._buffer) >= limit:
                    continue
                self._buffer += key
                await self.send("*" if mask else key)
        finally:
            self._prompt = ""
            self._buffer = ""

    async def send_raw(self, data: bytes) -> None:
        """Bytes to the wire, untouched but for the protocol's own escaping."""
        try:
            self.writer.write(escape(data))
            await self.writer.drain()
        except (ConnectionError, RuntimeError) as exc:
            raise Hangup(str(exc)) from exc

    async def read_raw(self, timeout: float = 30.0) -> bytes:
        """Whatever bytes have arrived, waiting up to `timeout` for the first."""
        while not self.rawbuf:
            await asyncio.wait_for(self._wait(), timeout)
        out = bytes(self.rawbuf)
        self.rawbuf.clear()
        return out

    async def _wait(self) -> None:
        """Block until the wire says something or another node does."""
        if self._read_task is None:
            self._read_task = asyncio.create_task(self.reader.read(4096))
        if self._inbox_task is None:
            self._inbox_task = asyncio.create_task(self.inbox.get())
        done, _ = await asyncio.wait({self._read_task, self._inbox_task},
                                     return_when=asyncio.FIRST_COMPLETED)
        if self._read_task in done:
            task, self._read_task = self._read_task, None
            chunk = task.result()
            if not chunk:
                raise Hangup("carrier lost")
            await self._consume(chunk)
        if self._inbox_task in done:
            task, self._inbox_task = self._inbox_task, None
            await self._interject(task.result())

    async def _consume(self, chunk: bytes) -> None:
        result = self.tn.feed(chunk)
        if result.reply:
            self.writer.write(result.reply)
            await self.writer.drain()
        if result.size:
            self.columns, self.rows = result.size
        if result.terminal and prefers_cp437(result.terminal):
            self.encoding = "cp437"
        if result.interrupt:
            self.keys.append("\x03")
        if result.data:
            if self.raw_mode:
                self.rawbuf += result.data
            else:
                self._decode(result.data)

    def _decode(self, data: bytes) -> None:
        """Bytes to keystrokes, holding back anything cut in half."""
        self._raw += data
        if self.encoding == "cp437":
            text = bytes(self._raw).decode("cp437")
            self._raw.clear()
        else:
            try:
                text = bytes(self._raw).decode("utf-8")
                self._raw.clear()
            except UnicodeDecodeError as exc:
                text = bytes(self._raw[:exc.start]).decode("utf-8")
                del self._raw[:exc.start]
                if len(self._raw) > 8:               # not a truncation: junk
                    self._raw.clear()

        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\x1b" and text[i + 1:i + 2] in ("[", "O"):
                final = text[i + 2:i + 3]
                name = {"A": "UP", "B": "DOWN", "C": "RIGHT",
                        "D": "LEFT"}.get(final)
                if name:
                    self.keys.append(name)
                    i += 3
                    continue
                i += 3
                continue
            if ch == "\x00":                          # telnet CR NUL is CR
                i += 1
                continue
            if ch == "\n" and self.keys and self.keys[-1] == "\r":
                i += 1                                # CRLF is one Enter
                continue
            self.keys.append(ch)
            i += 1

    async def _interject(self, text: str) -> None:
        """Print something that arrived while the caller was typing."""
        await self.send("\r" + A.CLEAR_LINE + text + "\r\n")
        if self._prompt:
            await self.send(self._prompt + self._buffer)

    async def close(self) -> None:
        for task in (self._read_task, self._inbox_task):
            if task is not None:
                task.cancel()
        try:
            self.writer.close()
        except Exception:                             # noqa: BLE001
            pass


# ---- the board itself -------------------------------------------------------

class Board:
    """Everything shared between nodes: the model, the rooms, the message base."""

    def __init__(self, run_dir: str, corpus_dir: str, device: str = "auto",
                 password: str | None = None,
                 sysop_password: str | None = None,
                 max_callers: int = 512, max_per_address: int = 8,
                 max_tokens: int = 120,
                 steps: int = 100, grow: int = 1) -> None:
        self.run_dir, self.corpus_dir, self.device = run_dir, corpus_dir, device
        self.password = password
        self.sysop_password = sysop_password
        self.max_callers = max_callers
        self.max_per_address = max_per_address
        self.max_tokens, self.steps, self.grow = max_tokens, steps, grow

        self.dir = Path(run_dir) / "bbs"
        self.model = self.tok = self.torch_device = None
        self.version = 0
        # One model, many callers. The lock is what makes that safe; the
        # counter beside it is what makes it honest - a board that is simply
        # slow and says nothing is indistinguishable from a broken one.
        self.gpu_lock = asyncio.Lock()
        self.waiting = 0
        self.callers: dict[int, Caller] = {}
        self.rooms: dict[str, set[int]] = {"LOBBY": set()}
        self.started = time.time()
        self._node = 0
        self._stats: dict = {}
        self._stats_at = 0.0

    # -- storage --------------------------------------------------------------

    def _path(self, name: str) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir / name

    def read(self, name: str, default):
        try:
            return json.loads(self._path(name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return default

    def write(self, name: str, value) -> None:
        target = self._path(name)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=1), encoding="utf-8")
        tmp.replace(target)

    # -- the model ------------------------------------------------------------

    def load(self) -> None:
        """Load once, at startup, on the thread that starts the server.

        A board that loads the model on first call makes the first caller
        wait a minute for a menu and every later caller wonder why it was
        fast for them.
        """
        from motherbrain.cli import load_current

        from motherbrain.stats import gather

        self.model, self.tok, self.torch_device, self.version = load_current(
            self.run_dir, self.device)
        self._stats = gather(self.run_dir, self.corpus_dir, self.model,
                             self.torch_device)
        self._stats_at = time.monotonic()

    def stats(self) -> dict:
        """The cached stats block. Never blocks, because every screen wants it.

        Gathering counts a quarter of a billion characters of corpus and takes
        five seconds. Doing that per screen draw would stall every node on the
        board at once - one caller opening a menu would freeze the other
        five hundred - so it is gathered once at startup and refreshed on a
        timer, off the event loop.
        """
        return self._stats

    async def refresh_stats(self) -> dict:
        from motherbrain.stats import gather

        self._stats = await asyncio.to_thread(
            gather, self.run_dir, self.corpus_dir, self.model,
            self.torch_device)
        self._stats_at = time.monotonic()
        return self._stats

    async def stats_keeper(self, every: float = 120.0) -> None:
        """Keep the cache warm for as long as the board is up."""
        while True:
            await asyncio.sleep(every)
            with contextlib.suppress(Exception):
                await self.refresh_stats()

    async def generate(self, prompt: str, max_tokens: int | None = None,
                       temperature: float = 0.8) -> str:
        """Continue a prompt, off the event loop so other nodes keep moving."""
        if self.model is None:
            return ""

        def work() -> str:
            import torch

            from motherbrain.tokenizer import EOS_ID

            ids = torch.tensor([self.tok.encode(prompt, bos=True)],
                               device=self.torch_device)
            out = []
            for token in self.model.generate(
                    ids, max_new_tokens=max_tokens or self.max_tokens,
                    temperature=temperature, top_k=40, top_p=0.95,
                    repetition_penalty=1.15, eos_id=EOS_ID):
                out.append(self.tok.decode([token]))
            return "".join(out)

        async with self.busy():
            return await asyncio.to_thread(work)

    async def answer(self, text: str, caller: Caller) -> tuple[str, str]:
        """The honest answering order: computed, known, self, then generated.

        Exactly what `mb console` does, in the same order, for the same
        reason - a definite answer is never generated when it can be worked
        out - and the caller is told which of the four this was.
        """
        from motherbrain.chat import CONTINUATION_NOTE, consider, respond
        from motherbrain.logic import solve

        exact = await asyncio.to_thread(solve, text)
        if exact is not None:
            return "exact", exact.render()

        try:
            considered = await asyncio.to_thread(consider, text, self.run_dir)
        except Exception:                                 # noqa: BLE001
            considered = None
        if considered is not None:
            return "known", considered[1]

        kind, said = respond(text, self.stats())
        if kind == "fact":
            return "self", said

        produced = await self.generate(text)
        return "generated", (produced.strip() or "(nothing)") + \
            f"\n\n{CONTINUATION_NOTE}"

    async def name_image(self, tensor) -> list[tuple[float, str]]:
        """What the perception tower makes of one image, best first.

        Scores, not statements. It was trained on a closed world of captions
        and will pick the nearest one whatever it is shown, so the loss is
        reported beside the name and the screen says as much.
        """
        if self.model is None or getattr(self.model, "vision", None) is None:
            return []

        def work() -> list[tuple[float, str]]:
            import torch

            from motherbrain.sight import all_captions, encode_batch

            image = tensor.to(self.torch_device)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            out = []
            for caption in all_captions():
                idx, targets = encode_batch(self.tok, [caption],
                                            self.torch_device)
                with torch.no_grad():
                    logits, _ = self.model(idx, targets=targets, images=image)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1), ignore_index=-100)
                out.append((float(loss), caption))
            out.sort()
            return out

        async with self.busy():
            return await asyncio.to_thread(work)

    @contextlib.asynccontextmanager
    async def busy(self):
        """Take the model, counting everyone waiting for it while you have it."""
        self.waiting += 1
        try:
            async with self.gpu_lock:
                yield
        finally:
            self.waiting -= 1

    def queued(self) -> int:
        """How many callers are ahead of you for the model. Nought, usually."""
        return max(0, self.waiting - 1)

    async def two_answers(self, text: str, caller: Caller):
        """(what it knows, what it would generate) - for the TURING door."""
        from motherbrain.chat import classify, respond

        if classify(text) is None:
            return None, ""
        _, honest = respond(text, self.stats())
        made_up = (await self.generate(text, max_tokens=60)).strip()
        return honest, made_up or "..."

    # -- nodes and rooms ------------------------------------------------------

    def next_node(self) -> int:
        self._node += 1
        return self._node

    def join(self, caller: Caller, room: str) -> None:
        self.leave(caller)
        room = room.upper()[:16] or "LOBBY"
        self.rooms.setdefault(room, set()).add(caller.node)
        caller.room = room

    def leave(self, caller: Caller) -> None:
        if caller.room and caller.node in self.rooms.get(caller.room, ()):
            self.rooms[caller.room].discard(caller.node)
            if not self.rooms[caller.room] and caller.room != "LOBBY":
                self.rooms.pop(caller.room, None)
        caller.room = ""

    def broadcast(self, room: str, text: str, skip: int | None = None) -> None:
        for node in list(self.rooms.get(room, ())):
            if node == skip:
                continue
            other = self.callers.get(node)
            if other is not None:
                other.tell(text)

    def page_all(self, text: str, skip: int | None = None) -> None:
        for node, other in list(self.callers.items()):
            if node != skip:
                other.tell(text)

    # -- the message base -----------------------------------------------------

    def threads(self) -> list[dict]:
        return self.read("messages.json", [])

    def post(self, author: str, subject: str, body: str,
             parent: int | None = None) -> int:
        threads = self.threads()
        entry = {"id": len(threads) + 1, "author": author[:24],
                 "subject": subject[:60], "body": body[:4000],
                 "when": now(), "parent": parent}
        threads.append(entry)
        self.write("messages.json", threads)
        return entry["id"]

    def oneliners(self) -> list[dict]:
        return self.read("oneliners.json", [])

    def add_oneliner(self, author: str, text: str) -> None:
        lines = self.oneliners()
        lines.append({"who": author[:24], "text": text[:78], "when": now()})
        self.write("oneliners.json", lines[-200:])

    def callers_log(self) -> list[dict]:
        return self.read("callers.json", [])

    def log_call(self, caller: Caller, actions: int) -> None:
        log = self.callers_log()
        log.append({"who": caller.handle, "when": now(),
                    "from": "local" if caller.local else "network",
                    "minutes": round((time.time() - caller.connected_at) / 60, 1),
                    "did": actions})
        self.write("callers.json", log[-200:])


# ---- the file area ----------------------------------------------------------

def file_sections(board: Board) -> list[dict]:
    """What can be downloaded, and from where.

    Roots are named here and nowhere else. A caller never supplies a path -
    they pick a number off a list this function built - which is what keeps
    a file area from being an arbitrary-file-read primitive with a menu in
    front of it.
    """
    root = Path(board.run_dir).resolve()
    repo = Path(__file__).resolve().parent.parent
    user = (Path(board.run_dir) / "bbs" / "files").resolve()

    return [
        {"key": "1", "name": "WAREZ",
         "blurb": "The software this board runs on. All of it. Free, and "
                  "always was - the only warez here is MotherBrain itself.",
         "roots": [root / "models", repo / "models", root / "patches",
                   root / "bbs" / "export"],
         "globs": ["*.pt", "*.json", "*.bin"]},
        {"key": "2", "name": "SOURCE",
         "blurb": "Every line of the board, the model and the doors.",
         "roots": [repo / "motherbrain", repo / "scripts"],
         "globs": ["*.py", "*.sh", "*.ps1"]},
        {"key": "3", "name": "TEXTFILES",
         "blurb": "Documentation, in the finest tradition of the g-files.",
         "roots": [repo, repo / "docs"], "globs": ["*.md", "*.txt"],
         "flat": True},
        {"key": "4", "name": "USER",
         "blurb": "Programs MotherBrain wrote, for the callers who asked.",
         "roots": [user], "globs": ["*"]},
    ]


def listing(section: dict, limit: int = 200) -> list[Path]:
    """The files in a section: real, readable, not secrets, sorted."""
    from motherbrain.security import SENSITIVE_NAMES, SENSITIVE_SUFFIXES

    found: list[Path] = []
    for root in section["roots"]:
        root = Path(root)
        if not root.is_dir():
            continue
        root = root.resolve()
        walk = root.glob if section.get("flat") else root.rglob
        for pattern in section["globs"]:
            for path in sorted(walk(pattern)):
                try:
                    real = path.resolve()
                    real.relative_to(root)          # no symlink out of the root
                except (OSError, ValueError):
                    continue
                if not real.is_file():
                    continue
                if any(p in SENSITIVE_NAMES for p in real.parts):
                    continue
                if real.suffix.lower() in SENSITIVE_SUFFIXES:
                    continue
                # Hidden directories are hidden for a reason: .git holds the
                # whole history, .venv holds somebody else's package, and
                # neither is a file this board is offering.
                if any(part.startswith(".")
                       for part in real.relative_to(root).parts):
                    continue
                if real not in found:
                    found.append(real)
                if len(found) >= limit:
                    return found
    return found


def size_of(path: Path) -> str:
    try:
        n = path.stat().st_size
    except OSError:
        return "?"
    for unit in ("b", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:,.0f}{unit}" if unit == "b" else f"{n:.1f}{unit}"
        n /= 1024
    return "?"


# ---- the screens ------------------------------------------------------------

# The main menu. The first five are the console's, word for word and in the
# same order - a board that quietly renumbered them would be a different
# program wearing the same name. Everything after is the board.
CONSOLE_OPTIONS = [
    ("1", "Tell MotherBrain what kind of program to make"),
    ("2", "Tell MotherBrain what to do"),
    ("3", "Teach MotherBrain something new"),
    ("4", "Apply new knowledge as a patch (update)"),
    ("5", "Run the GUI"),
]

BOARD_OPTIONS = [
    ("C", "Chat with MotherBrain", "T", "Teleconference (chat rooms)"),
    ("D", "Doors - games", "F", "File area / downloads"),
    ("M", "Message base", "O", "One-liners"),
    ("A", "ANSI gallery - see what it sees", "S", "System info"),
    ("W", "Who is online", "L", "Last callers"),
    ("P", "Page the sysop", "!", "Settings (baud, ANSI)"),
    ("G", "Goodbye - log off", "?", "Help"),
]


async def header(caller: Caller) -> None:
    board = caller.board
    stats = board.stats()
    left = f"{BOARD_NAME} BBS"
    right = (f"node {caller.node}  v{stats.get('version', 0)}  "
             f"{stats.get('total_params_human', '?')}  "
             f"{len(board.callers)} online")
    width = min(max(caller.columns, 40), 100)
    bar = A.pad(f" {A.HW}{left}{A.RESET}", width - A.width_of(right) - 2)
    await caller.line(f"{A.bg('blue')}{bar}{A.HY}{right} {A.RESET}")


async def login(caller: Caller) -> bool:
    """Logo, handle, password if the sysop set one. False means hang up."""
    board = caller.board
    await caller.cls()
    await caller.art(A.logo())
    await caller.line(A.shaded_bar(min(caller.columns, 78)))
    await caller.line(A.centre(f"{A.HW}a {A.HC}{board.version and 'v' or ''}"
                               f"{board.version}{A.HW} language model, "
                               f"answering its own telephone{A.RESET}",
                               min(caller.columns, 78)))
    await caller.line("")

    if board.password:
        for _ in range(3):
            given = await caller.ask(f"  {A.HY}board password: {A.RESET}",
                                     limit=128, mask=True)
            from motherbrain.security import constant_time_eq
            if constant_time_eq(given, board.password):
                break
            await caller.line(f"  {A.HR}no.{A.RESET}")
        else:
            await caller.line("  goodbye.")
            return False

    handle = ""
    while not handle:
        handle = (await caller.ask(f"  {A.HG}handle: {A.RESET}", limit=24)).strip()
        handle = "".join(c for c in handle if c.isprintable())[:24]
        if handle.lower() in ("sysop", SYSOP.lower(), "motherbrain"):
            await caller.line(f"  {A.HR}that name is taken by the machine."
                              f"{A.RESET}")
            handle = ""
    caller.handle = handle

    if board.sysop_password:
        given = await caller.ask(f"  {A.GREY}sysop key (blank if you are not "
                                 f"the sysop): {A.RESET}", limit=128, mask=True)
        from motherbrain.security import constant_time_eq
        caller.sysop = constant_time_eq(given, board.sysop_password)
        if caller.sysop:
            await caller.line(f"  {A.HG}sysop.{A.RESET}")
    elif caller.local:
        caller.sysop = True

    await caller.line("")
    recent = board.callers_log()[-5:]
    if recent:
        await caller.art("\n".join(A.box(
            "LAST CALLERS",
            [f"{A.HW}{A.pad(c['who'], 20)}{A.GREY}{c['when']}  "
             f"{c['minutes']} min  {c['from']}" for c in reversed(recent)],
            width=min(caller.columns, 70))))
    lines = board.oneliners()
    if lines:
        last = lines[-1]
        await caller.line(f"  {A.HM}\"{last['text']}\"{A.GREY} - "
                          f"{last['who']}{A.RESET}")
    await caller.line("")
    await caller.line(f"  {A.HC}Welcome, {A.HW}{handle}{A.HC}. "
                      f"{len(board.callers)} node(s) in use.{A.RESET}")
    board.page_all(f"{A.HG}*** {handle} has logged on to node "
                   f"{caller.node} ***{A.RESET}", skip=caller.node)
    await caller.pause()
    return True


async def main_menu(caller: Caller) -> None:
    """Draw the menu and run whatever is chosen, until the caller leaves."""
    board = caller.board
    actions = 0
    while True:
        await caller.cls()
        await header(caller)
        await caller.line("")
        width = min(max(caller.columns, 60), 76)

        await caller.art("\n".join(A.box(
            "WHAT WOULD YOU LIKE TO DO",
            [f"  {A.HY}[{key}]{A.RESET}  {A.HW}{label}"
             for key, label in CONSOLE_OPTIONS],
            width=width, frame=A.HC)))
        rows = []
        for a_key, a_label, b_key, b_label in BOARD_OPTIONS:
            half = (width - 4) // 2
            # Truncate to two short of the column, so a label that exactly
            # fills its half still leaves a gap before the next one rather
            # than running into it.
            left = A.truncate(f"{A.HY}[{a_key}]{A.RESET} {A.HW}{a_label}",
                              half - 2)
            right = A.truncate(f"{A.HY}[{b_key}]{A.RESET} {A.HW}{b_label}",
                               half - 2)
            rows.append(f" {A.pad(left, half)}{right}")
        await caller.art("\n".join(A.box("THE BOARD", rows, width=width,
                                         frame=A.HB)))
        await caller.line("")
        choice = (await caller.ask(
            f"  {A.HG}{caller.handle}{A.RESET}@{A.HC}{BOARD_NAME}{A.RESET} "
            f"{A.HY}command: {A.RESET}", limit=4)).strip().upper()

        if not choice:
            continue
        if choice in ("G", "Q", "OFF", "BYE"):
            break
        handler = HANDLERS.get(choice)
        if handler is None:
            await caller.line(f"  {A.HR}no such command. [?] for help."
                              f"{A.RESET}")
            await caller.pause()
            continue
        actions += 1
        try:
            await handler(caller)
        except Hangup:
            raise
        except Exception as exc:                          # noqa: BLE001
            await caller.line(f"  {A.HR}that went wrong: {exc}{A.RESET}")
            await caller.pause()

    board.log_call(caller, actions)
    await goodbye(caller)


async def goodbye(caller: Caller) -> None:
    await caller.cls()
    await caller.art(A.logo())
    await caller.line("")
    minutes = (time.time() - caller.connected_at) / 60
    await caller.line(f"  {A.HC}You were connected for {minutes:.1f} minutes."
                      f"{A.RESET}")
    await caller.line(f"  {A.HW}NO CARRIER{A.RESET}")
    await caller.line("")


# ---- the five console options ----------------------------------------------

async def option_make(caller: Caller) -> None:
    """1 - describe a program; it writes one and files it under your handle."""
    board = caller.board
    await caller.cls()
    await header(caller)
    await caller.line("")
    await caller.line(f"  {A.HY}TELL MOTHERBRAIN WHAT KIND OF PROGRAM TO MAKE"
                      f"{A.RESET}")
    await caller.line(f"  {A.GREY}It writes Python. It is a 52M base model: "
                      f"what comes back is the{A.RESET}")
    await caller.line(f"  {A.GREY}shape of a program, not a working one. "
                      f"Read it before you run it.{A.RESET}")
    await caller.line("")
    want = (await caller.ask(f"  {A.HW}what kind of program? {A.RESET}",
                             limit=200)).strip()
    if not want:
        return

    import re
    words = [w for w in re.findall(r"[A-Za-z]+", want.lower())
             if w not in {"a", "an", "the", "that", "to", "and", "of", "for",
                          "in", "it", "with", "program", "script"}]
    slug = "_".join(words[:4]) or "main"
    head = f'"""{want}"""\n\n\ndef {slug}('

    await caller.line("")
    await caller.line(f"  {A.GREY}thinking...{A.RESET}")
    text = await board.generate(head, max_tokens=board.max_tokens)
    code = f'"""{want}"""\n\n\ndef {slug}({text}\n'

    await caller.line(A.rule(min(caller.columns, 78), "─", A.GREY))
    for line in code.split("\n"):
        await caller.line(f"{A.HG}{line}{A.RESET}")
    await caller.line(A.rule(min(caller.columns, 78), "─", A.GREY))
    await caller.line("")

    save = (await caller.ask(f"  save it to your file area? [Y/n] ",
                             limit=4)).strip().lower()
    if save.startswith("n"):
        return
    area = Path(board.run_dir) / "bbs" / "files" / _safe_name(caller.handle)
    area.mkdir(parents=True, exist_ok=True)
    target = area / f"{slug}.py"
    n = 1
    while target.exists():
        n += 1
        target = area / f"{slug}_{n}.py"
    target.write_text(code, encoding="utf-8")
    await caller.line(f"  {A.HG}filed as {target.name} in "
                      f"[F]ile area / USER.{A.RESET}")
    await caller.pause()


def _safe_name(handle: str) -> str:
    """A handle as a directory name: letters, digits, dash, and nothing else."""
    cleaned = "".join(c if (c.isalnum() or c in "-_") else "_"
                      for c in handle)[:24]
    return cleaned or "caller"


async def option_do(caller: Caller) -> None:
    """2 - tell it what to do. Everything that touches this machine is refused."""
    from motherbrain.commands import parse

    await caller.cls()
    await header(caller)
    await caller.line("")
    await caller.line(f"  {A.HY}TELL MOTHERBRAIN WHAT TO DO{A.RESET}")
    await caller.line(f"  {A.GREY}Ask it things. Blank line to go back."
                      f"{A.RESET}")
    await caller.line(f"  {A.GREY}Running programs, reading files and shell "
                      f"commands are refused here:{A.RESET}")
    await caller.line(f"  {A.GREY}over a wire that is not convenience, it is "
                      f"a shell on the sysop's machine.{A.RESET}")
    await caller.line("")

    while True:
        line = (await caller.ask(f"  {A.HG}>{A.RESET} ", limit=400)).strip()
        if not line:
            return
        command = parse(line)
        if command.name in LOCAL_ONLY:
            await caller.line(f"  {A.HR}`{command.name}` is refused to a "
                              f"caller.{A.RESET} It runs code or reads files "
                              f"on the machine this board is on.")
            await caller.line(f"  {A.GREY}The same command works in "
                              f"`mb console`, where you are already the one "
                              f"with the shell.{A.RESET}\n")
            continue
        source, text = await caller.board.answer(line, caller)
        await caller.line("")
        await caller.line(f"  {doors._label(source)}")
        for out in A.wrap(text, caller.columns - 6):
            await caller.line(f"    {A.HW}{out}{A.RESET}")
        await caller.line("")


async def option_teach(caller: Caller) -> None:
    """3 - feed the corpus. Stored, not learned; the difference is stated."""
    from motherbrain.data import Corpus

    board = caller.board
    await caller.cls()
    await header(caller)
    await caller.line("")
    await caller.line(f"  {A.HY}TEACH MOTHERBRAIN SOMETHING NEW{A.RESET}")
    await caller.line(f"  {A.GREY}Type or paste. A blank line ends it. "
                      f"This stores what you write - the{A.RESET}")
    await caller.line(f"  {A.GREY}model does not change until somebody "
                      f"applies a patch, option 4.{A.RESET}")
    await caller.line(f"  {A.GREY}Everything fed can come back out of the "
                      f"model later. Post accordingly.{A.RESET}")
    await caller.line("")

    lines: list[str] = []
    while True:
        got = await caller.ask("  ", limit=400)
        if not got.strip():
            break
        lines.append(got)
        if sum(len(x) for x in lines) > MAX_FEED:
            await caller.line(f"  {A.HY}that is enough for one contribution."
                              f"{A.RESET}")
            break
    text = "\n".join(lines).strip()
    if not text:
        return
    if caller.fed + len(text) > MAX_FEED_PER_CALL:
        await caller.line(f"  {A.HR}you have fed the limit for one call."
                          f"{A.RESET}")
        await caller.pause()
        return

    corpus = Corpus(board.corpus_dir)
    await asyncio.to_thread(corpus.add_text, text,
                            f"bbs-{_safe_name(caller.handle)}")
    caller.fed += len(text)
    await caller.line("")
    await caller.line(f"  {A.HG}{len(text):,} characters stored{A.RESET}, "
                      f"credited to {caller.handle}.")
    await caller.line(f"  {A.GREY}The model is unchanged. It waits for a "
                      f"patch.{A.RESET}")
    board.page_all(f"{A.HC}*** {caller.handle} fed the corpus "
                   f"{len(text):,} characters ***{A.RESET}", skip=caller.node)
    await caller.pause()


async def option_patch(caller: Caller) -> None:
    """4 - learn what was fed and ascend. The sysop's key, and only theirs."""
    board = caller.board
    await caller.cls()
    await header(caller)
    await caller.line("")
    await caller.line(f"  {A.HY}APPLY NEW KNOWLEDGE AS A PATCH (UPDATE)"
                      f"{A.RESET}")
    await caller.line("")

    if not caller.sysop:
        await caller.line(f"  {A.HR}Sysop only.{A.RESET}")
        await caller.line(f"  {A.GREY}Applying a patch trains the model on "
                          f"everything fed since the last{A.RESET}")
        await caller.line(f"  {A.GREY}version, adds parameters and ascends. "
                          f"It takes the machine for minutes{A.RESET}")
        await caller.line(f"  {A.GREY}and changes what every other caller is "
                          f"talking to, so it is not a{A.RESET}")
        await caller.line(f"  {A.GREY}button a caller gets. Feed it under [3] "
                          f"and the sysop will apply it.{A.RESET}")
        await caller.pause()
        return

    from motherbrain.data import Corpus
    from motherbrain.patches import PatchStore

    corpus = Corpus(board.corpus_dir)
    store = PatchStore(board.run_dir, create=False)
    pending = corpus.n_documents - store.consumed_docs()
    await caller.line(f"  {pending} document(s) waiting to be learned.")
    if pending <= 0:
        await caller.line(f"  {A.GREY}nothing to learn. Feed it first."
                          f"{A.RESET}")
        await caller.pause()
        return
    go = (await caller.ask(f"  train now? every node waits. [y/N] ",
                           limit=4)).strip().lower()
    if not go.startswith("y"):
        return

    board.page_all(f"{A.HY}*** the sysop is applying a patch. MotherBrain "
                   f"will be busy. ***{A.RESET}")
    await caller.line(f"  {A.HC}training...{A.RESET}")

    def work():
        from motherbrain.patches import PatchConfig, create_patch

        return create_patch(board.run_dir, board.corpus_dir,
                            PatchConfig(mode="grow", grow_experts=board.grow,
                                        steps=board.steps),
                            note=f"bbs/{caller.handle}", device=board.device)

    async with board.busy():
        version = await asyncio.to_thread(work)
    if version is None:
        await caller.line(f"  {A.HR}nothing was learned.{A.RESET}")
    else:
        await asyncio.to_thread(board.load)
        await board.refresh_stats()
        from motherbrain.stats import human
        await caller.line(f"  {A.HG}v{version.parent} -> v{version.version}"
                          f"{A.RESET}: {human(version.params_before)} -> "
                          f"{human(version.params_after)} parameters, "
                          f"loss {version.loss_before:.3f} -> "
                          f"{version.loss_after:.3f}")
        board.page_all(f"{A.HG}*** MotherBrain is now v{version.version}: "
                       f"{human(version.params_after)} parameters ***"
                       f"{A.RESET}")
    await caller.pause()


async def option_gui(caller: Caller) -> None:
    """5 - the window. It opens where the board runs, which is not here."""
    await caller.cls()
    await header(caller)
    await caller.line("")
    await caller.line(f"  {A.HY}RUN THE GUI{A.RESET}")
    await caller.line("")
    for line in A.wrap(
            "A window opens on the machine it is asked to open on. This "
            "board is a telephone line: whatever it did with a window, you "
            "would not see it. So it will not pretend to.", 70):
        await caller.line(f"  {A.HW}{line}{A.RESET}")
    await caller.line("")
    await caller.line(f"  {A.HC}Where MotherBrain is installed, either of "
                      f"these gives you the window:{A.RESET}")
    await caller.line(f"    {A.HG}mb gui{A.RESET}                 "
                      f"{A.GREY}a real window{A.RESET}")
    await caller.line(f"    {A.HG}mb console{A.RESET}             "
                      f"{A.GREY}the same menu, option 5{A.RESET}")
    await caller.line("")
    await caller.line(f"  {A.HC}And the browser console, which does reach "
                      f"across a network:{A.RESET}")
    await caller.line(f"    {A.HG}mb gui --network{A.RESET}       "
                      f"{A.GREY}prints a link and a key{A.RESET}")
    await caller.line("")
    await caller.line(f"  {A.GREY}Everything the window does, this board "
                      f"does. That is what the menu above is.{A.RESET}")
    await caller.pause()


# ---- chat with MotherBrain --------------------------------------------------

async def chat_with_motherbrain(caller: Caller) -> None:
    """C - a conversation, with every reply labelled by where it came from."""
    await caller.cls()
    await header(caller)
    await caller.line("")
    await caller.art(A.gradient("  CHAT WITH MOTHERBRAIN  ",
                                (A.HM, A.M, A.HB, A.B)))
    await caller.line("")
    for line in A.wrap(
            "It answers four ways, and says which every time: arithmetic it "
            "computes, things it was told it deduces, questions about itself "
            "it reads off its own state, and everything else it continues - "
            "fluently, and about nothing. Blank line to leave.", 72):
        await caller.line(f"  {A.GREY}{line}{A.RESET}")
    await caller.line("")

    while True:
        said = (await caller.ask(f"  {A.HG}{caller.handle}>{A.RESET} ",
                                 limit=400)).strip()
        if not said:
            return
        waiting = caller.board.queued()
        if waiting:
            await caller.line(f"  {A.GREY}{waiting} ahead of you in the "
                              f"queue...{A.RESET}")
        source, text = await caller.board.answer(said, caller)
        await caller.line("")
        await caller.line(f"  {A.HC}MotherBrain{A.RESET}  "
                          f"{doors._label(source)}")
        for line in A.wrap(text, caller.columns - 6):
            await caller.line(f"    {A.HW}{line}{A.RESET}")
        await caller.line("")


# ---- teleconference ---------------------------------------------------------

async def teleconference(caller: Caller) -> None:
    """T - the chat rooms. Whoever else is dialled in is in here with you."""
    board = caller.board
    board.join(caller, "LOBBY")
    await caller.cls()
    await header(caller)
    await caller.line("")
    await caller.art(A.gradient("  T E L E C O N F E R E N C E  ",
                                (A.HG, A.G, A.HC, A.C)))
    await caller.line("")
    await caller.line(f"  {A.GREY}/join <room>   /rooms   /who   /me <does>   "
                      f"/ask <MotherBrain>   /quit{A.RESET}")
    await caller.line("")
    board.broadcast("LOBBY", f"{A.HG}*** {caller.handle} joins LOBBY ***"
                             f"{A.RESET}", skip=caller.node)
    await _room_header(caller)

    try:
        while True:
            said = (await caller.ask(f"{A.HG}[{caller.room}]{A.RESET} ",
                                     limit=400)).strip()
            if not said:
                continue
            if said.startswith("/"):
                if await _room_command(caller, said):
                    return
                continue
            board.broadcast(caller.room,
                            f"{A.HC}<{caller.handle}>{A.RESET} {said}",
                            skip=caller.node)
            await caller.line(f"{A.GREY}<{caller.handle}>{A.RESET} {said}")
    finally:
        board.broadcast(caller.room, f"{A.HY}*** {caller.handle} leaves ***"
                                     f"{A.RESET}", skip=caller.node)
        board.leave(caller)


async def _room_header(caller: Caller) -> None:
    board = caller.board
    here = [board.callers[n].handle for n in board.rooms.get(caller.room, ())
            if n in board.callers]
    await caller.line(f"  {A.HY}{caller.room}{A.RESET}: "
                      f"{A.HW}{', '.join(here) or 'just you'}{A.RESET}")


async def _room_command(caller: Caller, said: str) -> bool:
    """One slash command. True means leave the teleconference."""
    board = caller.board
    verb, _, rest = said[1:].partition(" ")
    verb, rest = verb.lower(), rest.strip()

    if verb in ("quit", "exit", "leave", "q"):
        return True
    if verb == "join":
        old = caller.room
        board.broadcast(old, f"{A.HY}*** {caller.handle} leaves ***{A.RESET}",
                        skip=caller.node)
        board.join(caller, rest or "LOBBY")
        board.broadcast(caller.room, f"{A.HG}*** {caller.handle} joins "
                                     f"{caller.room} ***{A.RESET}",
                        skip=caller.node)
        await _room_header(caller)
        return False
    if verb == "rooms":
        for name, members in sorted(board.rooms.items()):
            await caller.line(f"  {A.HY}{A.pad(name, 18)}{A.RESET}"
                              f"{len(members)} here")
        return False
    if verb == "who":
        for node, other in sorted(board.callers.items()):
            where = other.room or "the menu"
            await caller.line(f"  {A.HW}node {node:>3}  "
                              f"{A.pad(other.handle, 20)}{A.GREY}{where}"
                              f"{A.RESET}")
        return False
    if verb == "me":
        board.broadcast(caller.room, f"{A.HM}* {caller.handle} {rest}"
                                     f"{A.RESET}")
        return False
    if verb == "ask":
        if not rest:
            await caller.line(f"  {A.GREY}/ask what?{A.RESET}")
            return False
        board.broadcast(caller.room,
                        f"{A.HC}<{caller.handle}>{A.RESET} {A.GREY}asks "
                        f"MotherBrain:{A.RESET} {rest}", skip=caller.node)
        source, text = await board.answer(rest, caller)
        head = f"{A.HM}<{SYSOP}>{A.RESET} {doors._label(source)}"
        board.broadcast(caller.room, head)
        await caller.line(head)
        for line in A.wrap(text, caller.columns - 4):
            board.broadcast(caller.room, f"  {A.HW}{line}{A.RESET}")
            await caller.line(f"  {A.HW}{line}{A.RESET}")
        return False
    await caller.line(f"  {A.GREY}/join /rooms /who /me /ask /quit{A.RESET}")
    return False


# ---- doors ------------------------------------------------------------------

async def door_menu(caller: Caller) -> None:
    """D - pick a door."""
    while True:
        await caller.cls()
        await header(caller)
        await caller.line("")
        rows = [f"  {A.HY}[{key}]{A.RESET}  {A.HW}{A.pad(name, 12)}"
                f"{A.GREY}{blurb}{A.RESET}"
                for key, name, blurb in doors.CATALOGUE]
        rows.append("")
        rows.append(f"  {A.HY}[Q]{A.RESET}  {A.HW}back to the main menu")
        await caller.art("\n".join(A.box("D O O R S", rows,
                                         width=min(caller.columns, 76),
                                         frame=A.HM)))
        await caller.line("")
        key = (await caller.ask(f"  {A.HY}door: {A.RESET}", limit=2)).strip()
        if not key or key.upper() == "Q":
            return
        caller.board.page_all(
            f"{A.GREY}*** {caller.handle} is in the doors ***{A.RESET}",
            skip=caller.node)
        if not await doors.play(caller, key):
            await caller.line(f"  {A.HR}no such door.{A.RESET}")
            await caller.pause()


# ---- message base -----------------------------------------------------------

async def message_base(caller: Caller) -> None:
    """M - threads, and MotherBrain will answer one if asked."""
    board = caller.board
    while True:
        threads = board.threads()
        roots = [t for t in threads if t.get("parent") is None]
        await caller.cls()
        await header(caller)
        await caller.line("")
        rows = []
        for t in roots[-14:]:
            replies = sum(1 for r in threads if r.get("parent") == t["id"])
            rows.append(f" {A.HY}{t['id']:>3}{A.RESET} "
                        f"{A.HW}{A.pad(t['subject'], 40)}{A.RESET}"
                        f"{A.GREY}{A.pad(t['author'], 14)}"
                        f"{replies} repl{'y' if replies == 1 else 'ies'}"
                        f"{A.RESET}")
        if not rows:
            rows = [f"  {A.GREY}nothing posted yet. [P] starts it."
                    f"{A.RESET}"]
        await caller.art("\n".join(A.box("M E S S A G E   B A S E", rows,
                                         width=min(caller.columns, 76),
                                         frame=A.HG)))
        await caller.line(f"  {A.HY}[P]{A.RESET} post   "
                          f"{A.HY}[number]{A.RESET} read   "
                          f"{A.HY}[Q]{A.RESET} back")
        choice = (await caller.ask(f"  {A.HY}> {A.RESET}", limit=8)).strip()
        if not choice or choice.upper() == "Q":
            return
        if choice.upper() == "P":
            subject = (await caller.ask(f"  subject: ", limit=60)).strip()
            if not subject:
                continue
            await caller.line(f"  {A.GREY}body, blank line to end:{A.RESET}")
            body = await _multiline(caller)
            if body:
                board.post(caller.handle, subject, body)
                board.page_all(f"{A.HG}*** {caller.handle} posted "
                               f"\"{subject}\" ***{A.RESET}", skip=caller.node)
            continue
        if choice.isdigit():
            await _read_thread(caller, int(choice))


async def _multiline(caller: Caller, limit: int = 4000) -> str:
    lines: list[str] = []
    while True:
        got = await caller.ask("  ", limit=400)
        if not got.strip():
            break
        lines.append(got)
        if sum(len(x) for x in lines) > limit:
            break
    return "\n".join(lines).strip()


async def _read_thread(caller: Caller, ident: int) -> None:
    board = caller.board
    threads = board.threads()
    root = next((t for t in threads if t["id"] == ident), None)
    if root is None:
        return
    chain = [root] + [t for t in threads if t.get("parent") == ident]

    await caller.cls()
    await header(caller)
    await caller.line("")
    for msg in chain:
        await caller.line(f"{A.HY}{msg['subject']}{A.RESET}  "
                          f"{A.GREY}{msg['author']}  {msg['when']}{A.RESET}")
        for line in A.wrap(msg["body"], caller.columns - 4):
            await caller.line(f"  {A.HW}{line}{A.RESET}")
        await caller.line("")
    await caller.line(f"  {A.HY}[R]{A.RESET} reply   "
                      f"{A.HY}[M]{A.RESET} ask MotherBrain to reply   "
                      f"{A.HY}[Q]{A.RESET} back")
    key = (await caller.ask(f"  > ", limit=2)).strip().upper()
    if key == "R":
        body = await _multiline(caller)
        if body:
            board.post(caller.handle, f"Re: {root['subject']}", body, ident)
    elif key == "M":
        await caller.line(f"  {A.GREY}thinking...{A.RESET}")
        source, text = await board.answer(root["body"], caller)
        board.post(SYSOP, f"Re: {root['subject']}",
                   f"[{source}] {text}", ident)
        await caller.line(f"  {doors._label(source)}")
        for line in A.wrap(text, caller.columns - 4):
            await caller.line(f"  {A.HW}{line}{A.RESET}")
        await caller.pause()


# ---- one-liners -------------------------------------------------------------

async def oneliner_wall(caller: Caller) -> None:
    """O - the graffiti wall every board had."""
    board = caller.board
    await caller.cls()
    await header(caller)
    await caller.line("")
    lines = board.oneliners()[-15:]
    rows = [f"{A.HM}\"{A.pad(x['text'], 52)}\"{A.RESET}{A.GREY} - "
            f"{x['who']}{A.RESET}" for x in reversed(lines)] or \
           [f"{A.GREY}nobody has written on the wall yet.{A.RESET}"]
    await caller.art("\n".join(A.box("O N E - L I N E R S", rows,
                                     width=min(caller.columns, 76),
                                     frame=A.HM)))
    said = (await caller.ask(f"  {A.HY}add one (blank to skip): {A.RESET}",
                             limit=78)).strip()
    if said:
        board.add_oneliner(caller.handle, said)
        board.page_all(f"{A.HM}*** {caller.handle} wrote on the wall: "
                       f"\"{said}\" ***{A.RESET}", skip=caller.node)


# ---- transfers --------------------------------------------------------------

async def xmodem_send(caller: Caller, data: bytes, name: str) -> bool:
    """Send a file down the line the way every board did. True if it landed."""
    await caller.line(f"  {A.HY}Start your XMODEM receive now "
                      f"({len(data):,} bytes, {name}).{A.RESET}")
    await caller.line(f"  {A.GREY}Ctrl-C, or wait 60 seconds, to abandon it."
                      f"{A.RESET}")
    sender = xmodem.Sender(data)
    was_raw, was_baud = caller.raw_mode, caller.baud
    caller.raw_mode, caller.baud = True, 0
    caller.rawbuf.clear()
    try:
        chunk = await caller.read_raw(timeout=60.0)
        out = sender.begin(chunk[-1])
        while out is not None and not sender.cancelled:
            await caller.send_raw(out)
            if sender.done:
                reply = await caller.read_raw(timeout=30.0)
                return xmodem.ACK in reply
            reply = await caller.read_raw(timeout=30.0)
            for byte in reply:
                out = sender.answer(byte)
                if out is None or sender.done or sender.cancelled:
                    break
        return False
    except (asyncio.TimeoutError, Hangup):
        return False
    finally:
        caller.raw_mode, caller.baud = was_raw, was_baud
        caller.rawbuf.clear()
        caller.keys.clear()


async def xmodem_receive(caller: Caller, limit: int = 4 * 1024 * 1024) -> bytes:
    """Take a file from the caller. Capped, because a socket is not a disk."""
    await caller.line(f"  {A.HY}Start your XMODEM send now.{A.RESET} "
                      f"{A.GREY}(up to {limit // 1024}K){A.RESET}")
    receiver = xmodem.Receiver(crc=True)
    was_raw, was_baud = caller.raw_mode, caller.baud
    caller.raw_mode, caller.baud = True, 0
    caller.rawbuf.clear()
    try:
        for _ in range(20):                       # the receiver opens, once a
            await caller.send_raw(receiver.start())   # second, until answered
            try:
                chunk = await caller.read_raw(timeout=3.0)
            except asyncio.TimeoutError:
                continue
            break
        else:
            return b""

        while not receiver.done:
            acks = receiver.feed(chunk)
            if acks:
                await caller.send_raw(acks)
            if receiver.done or len(receiver.data) > limit:
                break
            chunk = await caller.read_raw(timeout=30.0)
        return receiver.file()
    except (asyncio.TimeoutError, Hangup):
        return bytes(receiver.data)
    finally:
        caller.raw_mode, caller.baud = was_raw, was_baud
        caller.rawbuf.clear()
        caller.keys.clear()


# ---- the file area ----------------------------------------------------------

async def file_area(caller: Caller) -> None:
    """F - four sections, real transfers, and no path a caller ever types."""
    board = caller.board
    while True:
        sections = file_sections(board)
        await caller.cls()
        await header(caller)
        await caller.line("")
        rows = []
        for section in sections:
            count = len(listing(section, 400))
            rows.append(f"  {A.HY}[{section['key']}]{A.RESET} "
                        f"{A.HW}{A.pad(section['name'], 12)}{A.RESET}"
                        f"{A.GREY}{count:>4} files{A.RESET}")
            for line in A.wrap(section["blurb"], min(caller.columns, 70) - 8):
                rows.append(f"       {A.GREY}{line}{A.RESET}")
        rows.append("")
        rows.append(f"  {A.HY}[U]{A.RESET} {A.HW}upload{A.RESET}"
                    f"{A.GREY}   send the board a file over XMODEM{A.RESET}")
        rows.append(f"  {A.HY}[Q]{A.RESET} {A.HW}back{A.RESET}")
        await caller.art("\n".join(A.box("F I L E   A R E A", rows,
                                         width=min(caller.columns, 76),
                                         frame=A.HY)))
        key = (await caller.ask(f"  {A.HY}area: {A.RESET}", limit=2)).strip()
        if not key or key.upper() == "Q":
            return
        if key.upper() == "U":
            await _upload(caller)
            continue
        section = next((s for s in sections if s["key"] == key), None)
        if section is not None:
            await _browse(caller, section)


async def _browse(caller: Caller, section: dict) -> None:
    files = listing(section, 400)
    page = 0
    per_page = max(8, min(18, caller.rows - 10))
    while True:
        await caller.cls()
        await header(caller)
        await caller.line("")
        chunk = files[page * per_page:(page + 1) * per_page]
        rows = []
        for i, path in enumerate(chunk, start=page * per_page + 1):
            rows.append(f" {A.HY}{i:>3}{A.RESET} "
                        f"{A.HW}{A.pad(path.name, 40)}{A.RESET}"
                        f"{A.GREY}{size_of(path):>9}{A.RESET}")
        if not rows:
            rows = [f"  {A.GREY}empty.{A.RESET}"]
        await caller.art("\n".join(A.box(
            f"{section['name']}  ({len(files)} files)", rows,
            width=min(caller.columns, 76), frame=A.HY)))
        pages = max(1, (len(files) + per_page - 1) // per_page)
        await caller.line(f"  page {page + 1}/{pages}   "
                          f"{A.HY}[N]{A.RESET}ext  {A.HY}[B]{A.RESET}ack  "
                          f"{A.HY}[number]{A.RESET} to take it  "
                          f"{A.HY}[Q]{A.RESET}uit")
        choice = (await caller.ask(f"  {A.HY}> {A.RESET}", limit=6)).strip()
        if not choice or choice.upper() == "Q":
            return
        if choice.upper() == "N":
            page = min(page + 1, pages - 1)
            continue
        if choice.upper() == "B":
            page = max(0, page - 1)
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            await _download(caller, files[int(choice) - 1])


async def _download(caller: Caller, path: Path) -> None:
    await caller.cls()
    await caller.line("")
    await caller.line(f"  {A.HW}{path.name}{A.RESET}  "
                      f"{A.GREY}{size_of(path)}{A.RESET}")
    await caller.line("")
    await caller.line(f"  {A.HY}[X]{A.RESET} XMODEM   "
                      f"{A.GREY}a real transfer; your client must support it"
                      f"{A.RESET}")
    await caller.line(f"  {A.HY}[B]{A.RESET} base64   "
                      f"{A.GREY}printed to the screen, for a plain telnet"
                      f"{A.RESET}")
    await caller.line(f"  {A.HY}[V]{A.RESET} view     "
                      f"{A.GREY}read it here, if it is text{A.RESET}")
    await caller.line(f"  {A.HY}[Q]{A.RESET} back")
    key = (await caller.ask(f"  > ", limit=2)).strip().upper()
    if key not in ("X", "B", "V"):
        return

    try:
        data = await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        await caller.line(f"  {A.HR}cannot read it: {exc}{A.RESET}")
        await caller.pause()
        return

    if key == "X":
        ok = await xmodem_send(caller, data, path.name)
        await caller.line("")
        await caller.line(f"  {A.HG}transfer complete.{A.RESET}" if ok
                          else f"  {A.HR}transfer failed or was abandoned."
                               f"{A.RESET}")
    elif key == "B":
        import base64
        if len(data) > 512 * 1024:
            await caller.line(f"  {A.HR}too big to print. Use XMODEM."
                              f"{A.RESET}")
        else:
            text = base64.b64encode(data).decode()
            await caller.line(f"{A.GREY}--- begin {path.name} "
                              f"(base64) ---{A.RESET}")
            for i in range(0, len(text), 76):
                await caller.line(text[i:i + 76])
            await caller.line(f"{A.GREY}--- end {path.name} ---{A.RESET}")
            await caller.line(f"  {A.GREY}save it and: "
                              f"base64 -d in > {path.name}{A.RESET}")
    else:
        try:
            body = data.decode("utf-8")
        except UnicodeDecodeError:
            await caller.line(f"  {A.HR}not text.{A.RESET}")
        else:
            for line in body.split("\n")[:400]:
                await caller.line(f"{A.HW}{line}{A.RESET}")
    await caller.pause()


async def _upload(caller: Caller) -> None:
    """Uploads land in the caller's own directory and nowhere else."""
    name = (await caller.ask(f"  filename: ", limit=48)).strip()
    if not name:
        return
    safe = _safe_name(Path(name).name)
    suffix = Path(name).suffix.lower()
    if suffix in (".pt", ".exe", ".dll", ".so", ".sh", ".bat", ".ps1"):
        await caller.line(f"  {A.HR}not that kind of file.{A.RESET}")
        await caller.pause()
        return
    data = await xmodem_receive(caller)
    if not data:
        await caller.line(f"  {A.HR}nothing arrived.{A.RESET}")
        await caller.pause()
        return
    area = Path(caller.board.run_dir) / "bbs" / "files" / \
        _safe_name(caller.handle)
    area.mkdir(parents=True, exist_ok=True)
    target = area / safe
    target.write_bytes(data)
    await caller.line(f"  {A.HG}{len(data):,} bytes received as "
                      f"{target.name}.{A.RESET}")
    caller.board.page_all(f"{A.HC}*** {caller.handle} uploaded "
                          f"{target.name} ***{A.RESET}", skip=caller.node)
    await caller.pause()


# ---- the gallery ------------------------------------------------------------

async def gallery(caller: Caller) -> None:
    """A - a picture on the screen, and what MotherBrain makes of it.

    The board is running a model with a perception tower on it, and the
    tower's whole job is naming what it is shown. Rendering the picture in
    ANSI beside the naming is the only way to see whether the answer had
    anything to do with the image.
    """
    board = caller.board
    while True:
        await caller.cls()
        await header(caller)
        await caller.line("")
        await caller.art("\n".join(A.box("A N S I   G A L L E R Y", [
            f"  {A.HY}[S]{A.RESET} {A.HW}show me one{A.RESET}"
            f"{A.GREY}      a shape it was trained on; can it name it?"
            f"{A.RESET}",
            f"  {A.HY}[F]{A.RESET} {A.HW}from the files{A.RESET}"
            f"{A.GREY}   pick an image out of the file area{A.RESET}",
            f"  {A.HY}[U]{A.RESET} {A.HW}upload one{A.RESET}"
            f"{A.GREY}       XMODEM it up and it will look at it{A.RESET}",
            f"  {A.HY}[Q]{A.RESET} {A.HW}back{A.RESET}",
        ], width=min(caller.columns, 76), frame=A.HM)))
        key = (await caller.ask(f"  {A.HY}> {A.RESET}", limit=2)).strip().upper()
        if not key or key == "Q":
            return
        if key == "S":
            await _gallery_generated(caller)
        elif key == "F":
            await _gallery_file(caller)
        elif key == "U":
            data = await xmodem_receive(caller, limit=2 * 1024 * 1024)
            if not data:
                await caller.line(f"  {A.HR}nothing arrived.{A.RESET}")
                await caller.pause()
                continue
            await _look_at_bytes(caller, data, "your upload")


def _picture_width(caller: Caller) -> int:
    """Wide enough to see, short enough to fit above the caption.

    A square picture is half as many rows as it is columns, so the height of
    the caller's screen sets the width - a picture that scrolls its own
    verdict off the top is not a gallery.
    """
    return max(16, min(56, caller.columns - 4, 2 * max(8, caller.rows - 14)))


async def _gallery_generated(caller: Caller) -> None:
    """One of the shapes the tower was trained on, with the truth beside it."""
    import random

    board = caller.board
    try:
        from motherbrain.imagedata import pairs

        tensor, truth = (await asyncio.to_thread(
            pairs, 1, getattr(board.model.cfg, "image_size", 64),
            random.randint(1, 10 ** 6)))[0]
    except ImportError:
        await _no_imaging(caller)
        return

    await caller.cls()
    await caller.line("")
    # Straight from the tensor: the drawing needs no imaging library, so the
    # gallery works on any install that can run the model at all.
    picture = await asyncio.to_thread(
        A.picture_tensor, tensor, _picture_width(caller),
        16 if caller.encoding == "cp437" else 256)
    await caller.art(picture)
    await caller.line("")
    guesses = await board.name_image(tensor)
    await _report_guesses(caller, guesses, truth)


async def _gallery_file(caller: Caller) -> None:
    section = {"name": "IMAGES", "roots": [Path(caller.board.run_dir),
                                           Path(__file__).resolve().parent.parent],
               "globs": ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]}
    files = listing(section, 60)
    if not files:
        await caller.line(f"  {A.GREY}no images anywhere the board can see."
                          f"{A.RESET}")
        await caller.pause()
        return
    for i, path in enumerate(files[:20], start=1):
        await caller.line(f"  {A.HY}{i:>2}{A.RESET} {A.HW}{path.name}"
                          f"{A.RESET} {A.GREY}{size_of(path)}{A.RESET}")
    choice = (await caller.ask(f"  which? ", limit=4)).strip()
    if not choice.isdigit() or not 1 <= int(choice) <= min(20, len(files)):
        return
    path = files[int(choice) - 1]
    await _look_at_bytes(caller, path.read_bytes(), path.name)


async def _look_at_bytes(caller: Caller, data: bytes, name: str) -> None:
    """Draw it, then ask the tower what it is."""
    import io

    board = caller.board
    try:
        from PIL import Image
    except ImportError:
        await _no_imaging(caller)
        return
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:                              # noqa: BLE001
        await caller.line(f"  {A.HR}cannot read {name} as an image: {exc}"
                          f"{A.RESET}")
        await caller.pause()
        return

    await caller.cls()
    await caller.line("")
    art = await asyncio.to_thread(A.picture, image, _picture_width(caller),
                                  16 if caller.encoding == "cp437" else 256)
    await caller.art(art)
    await caller.line(f"  {A.GREY}{name}{A.RESET}")
    await caller.line("")

    size = getattr(board.model.cfg, "image_size", 64) if board.model else 64
    from motherbrain.perception import _to_tensor

    tensor = await asyncio.to_thread(_to_tensor, image.convert("RGB"), size)
    guesses = await board.name_image(tensor)
    await _report_guesses(caller, guesses, None)


async def _no_imaging(caller: Caller) -> None:
    """Pillow is missing. Say which command fixes it, rather than falling over."""
    await caller.line(f"  {A.HR}This board cannot handle pictures: Pillow is "
                      f"not installed.{A.RESET}")
    await caller.line(f"  {A.GREY}The sysop fixes it with:  "
                      f"{A.HW}pip install Pillow{A.RESET}")
    await caller.line(f"  {A.GREY}Everything else on the board works without "
                      f"it.{A.RESET}")
    await caller.pause()


async def _report_guesses(caller: Caller, guesses, truth) -> None:
    if not guesses:
        await caller.line(f"  {A.HR}This version has no perception tower, so "
                          f"it did not look at that.{A.RESET}")
        await caller.pause()
        return
    await caller.line(f"  {A.HC}MotherBrain says:{A.RESET}")
    for i, (loss, caption) in enumerate(guesses[:3]):
        mark = f"{A.HG}>" if i == 0 else f"{A.GREY} "
        await caller.line(f"  {mark} {A.pad(caption, 30)}{A.RESET}"
                          f"{A.GREY}loss {loss:.3f}{A.RESET}")
    if truth is not None:
        right = guesses[0][1] == truth
        await caller.line("")
        await caller.line(f"  it was {A.HW}{truth}{A.RESET} - "
                          + (f"{A.HG}right.{A.RESET}" if right
                             else f"{A.HR}wrong.{A.RESET}"))
    await caller.line("")
    await caller.line(f"  {A.GREY}These are scores over the captions it was "
                      f"trained on, not statements.{A.RESET}")
    await caller.line(f"  {A.GREY}It has never seen anything else, and will "
                      f"pick one of these whatever you show it.{A.RESET}")
    await caller.pause()


# ---- the informational screens ---------------------------------------------

async def system_info(caller: Caller) -> None:
    """S - the same stats block the terminal and the window draw."""
    from motherbrain.stats import render

    board = caller.board
    await caller.cls()
    await header(caller)
    await caller.line("")
    stats = await board.refresh_stats()
    await caller.art(f"{A.HC}" + render(stats, width=min(caller.columns, 70))
                     + A.RESET)
    await caller.line("")
    up = (time.time() - board.started) / 3600
    await caller.art("\n".join(A.box("THIS BOARD", [
        f"  nodes in use     {len(board.callers)} of {board.max_callers}",
        f"  rooms            {', '.join(sorted(board.rooms)) or 'none'}",
        f"  uptime           {up:.1f} hours",
        f"  calls logged     {len(board.callers_log())}",
        f"  messages         {len(board.threads())}",
        f"  your node        {caller.node} ({caller.address})",
        f"  your terminal    {caller.tn.terminal or 'unannounced'}, "
        f"{caller.columns}x{caller.rows}, {caller.encoding}",
    ], width=min(caller.columns, 70), frame=A.HB)))
    await caller.pause()


async def who_is_online(caller: Caller) -> None:
    board = caller.board
    await caller.cls()
    await header(caller)
    await caller.line("")
    rows = []
    for node, other in sorted(board.callers.items()):
        minutes = (time.time() - other.connected_at) / 60
        rows.append(f"  {A.HY}{node:>3}{A.RESET} "
                    f"{A.HW}{A.pad(other.handle, 22)}{A.RESET}"
                    f"{A.GREY}{A.pad(other.room or 'the menu', 14)}"
                    f"{minutes:>6.1f} min{A.RESET}")
    await caller.art("\n".join(A.box("W H O   I S   O N L I N E", rows,
                                     width=min(caller.columns, 76),
                                     frame=A.HG)))
    await caller.pause()


async def last_callers(caller: Caller) -> None:
    board = caller.board
    await caller.cls()
    await header(caller)
    await caller.line("")
    rows = [f"  {A.HW}{A.pad(c['who'], 22)}{A.RESET}{A.GREY}{c['when']}  "
            f"{c['minutes']:>5} min  {c['did']} things  {c['from']}{A.RESET}"
            for c in reversed(board.callers_log()[-18:])]
    await caller.art("\n".join(A.box("L A S T   C A L L E R S",
                                     rows or [f"  {A.GREY}you are the first."
                                              f"{A.RESET}"],
                                     width=min(caller.columns, 76))))
    await caller.pause()


async def page_sysop(caller: Caller) -> None:
    """P - page the sysop. The sysop is a language model, and answers."""
    board = caller.board
    await caller.cls()
    await header(caller)
    await caller.line("")
    await caller.line(f"  {A.HY}PAGING THE SYSOP{A.RESET}")
    await caller.line(f"  {A.GREY}The sysop here is the model. Any human "
                      f"sysop on a node will see this too.{A.RESET}")
    await caller.line("")
    said = (await caller.ask(f"  what do you want to say? ", limit=400)).strip()
    if not said:
        return
    for node, other in list(board.callers.items()):
        if other.sysop and node != caller.node:
            other.tell(f"{A.BLINK}{A.HR}*** PAGE from {caller.handle} "
                       f"(node {caller.node}) ***{A.RESET} {said}")
    await caller.line("")
    for i in range(3):
        await caller.send(f"\r{A.HY}  paging{'.' * (i + 1)}   {A.RESET}")
        await asyncio.sleep(0.4)
    await caller.line("")
    source, text = await board.answer(said, caller)
    await caller.line(f"  {A.HM}<{SYSOP}>{A.RESET} {doors._label(source)}")
    for line in A.wrap(text, caller.columns - 6):
        await caller.line(f"    {A.HW}{line}{A.RESET}")
    await caller.pause()


async def settings(caller: Caller) -> None:
    """! - baud rate, encoding, width. The baud rate is not a joke."""
    await caller.cls()
    await header(caller)
    await caller.line("")
    await caller.art("\n".join(A.box("S E T T I N G S", [
        f"  {A.HY}[1]{A.RESET} 300 baud    {A.GREY}as slow as it really was"
        f"{A.RESET}",
        f"  {A.HY}[2]{A.RESET} 1200 baud   {A.GREY}1983{A.RESET}",
        f"  {A.HY}[3]{A.RESET} 2400 baud   {A.GREY}1987, and readable"
        f"{A.RESET}",
        f"  {A.HY}[4]{A.RESET} 9600 baud   {A.GREY}1991{A.RESET}",
        f"  {A.HY}[5]{A.RESET} no limit    {A.GREY}as fast as the socket goes"
        f"{A.RESET}",
        "",
        f"  {A.HY}[C]{A.RESET} code page 437  {A.GREY}for a period client"
        f"{A.RESET}",
        f"  {A.HY}[8]{A.RESET} UTF-8          {A.GREY}for a modern one"
        f"{A.RESET}",
        f"  {A.HY}[Q]{A.RESET} back",
    ], width=min(caller.columns, 70), frame=A.HB)))
    await caller.line(f"  {A.GREY}now: "
                      f"{caller.baud or 'unlimited'} baud, {caller.encoding}, "
                      f"{caller.columns}x{caller.rows}{A.RESET}")
    key = (await caller.ask(f"  > ", limit=2)).strip().upper()
    rates = {"1": 300, "2": 1200, "3": 2400, "4": 9600, "5": 0}
    if key in rates:
        caller.baud = rates[key]
        await caller.line(f"  {A.HG}{caller.baud or 'unlimited'}.{A.RESET}")
    elif key == "C":
        caller.encoding = "cp437"
    elif key == "8":
        caller.encoding = "utf-8"
    await asyncio.sleep(0.3)


async def help_screen(caller: Caller) -> None:
    await caller.cls()
    await header(caller)
    await caller.line("")
    body = [
        f"  {A.HW}1-5{A.RESET}  the five things MotherBrain does. The same "
        f"five `mb console` shows,",
        f"       in the same order. Option 4 is the sysop's; option 5 opens "
        f"a window,",
        f"       and a window opens where the program runs, not where you "
        f"are calling from.",
        "",
        f"  {A.HW}C{A.RESET}    talk to it. Every reply is labelled with "
        f"where the answer came from.",
        f"  {A.HW}T{A.RESET}    the other callers. /join /who /me /ask.",
        f"  {A.HW}D{A.RESET}    five doors. TURING is the one to play.",
        f"  {A.HW}F{A.RESET}    files, over XMODEM or printed as base64.",
        f"  {A.HW}A{A.RESET}    show it a picture and watch it guess.",
        "",
        f"  {A.GREY}Refused to a caller: running programs, reading the "
        f"sysop's files, shell{A.RESET}",
        f"  {A.GREY}commands. Over a wire those are not features."
        f"{A.RESET}",
    ]
    await caller.art("\n".join(A.box("H E L P", body,
                                     width=min(caller.columns, 76))))
    await caller.pause()


HANDLERS = {
    "1": option_make, "2": option_do, "3": option_teach,
    "4": option_patch, "5": option_gui,
    "C": chat_with_motherbrain, "T": teleconference, "D": door_menu,
    "F": file_area, "M": message_base, "O": oneliner_wall,
    "A": gallery, "S": system_info, "W": who_is_online,
    "L": last_callers, "P": page_sysop, "!": settings, "?": help_screen,
}


# ---- the server -------------------------------------------------------------

async def session(reader, writer, board: Board) -> None:
    """One caller, start to finish. Every exit path frees the node."""
    caller = Caller(reader, writer, board, board.next_node())
    address = caller.address

    same_host = sum(1 for c in board.callers.values() if c.address == address)
    if len(board.callers) >= board.max_callers:
        writer.write(f"\r\n{A.HR}All {board.max_callers} nodes are busy. "
                     f"Call back.{A.RESET}\r\n".encode())
        await writer.drain()
        writer.close()
        return
    if same_host >= board.max_per_address and not caller.local:
        writer.write(f"\r\n{A.HR}Too many connections from {address}."
                     f"{A.RESET}\r\n".encode())
        await writer.drain()
        writer.close()
        return

    board.callers[caller.node] = caller
    try:
        writer.write(caller.tn.start())
        await writer.drain()
        # Give the client a moment to answer before the first screen, so the
        # window size and terminal type are known while drawing it.
        with contextlib.suppress(asyncio.TimeoutError, Hangup):
            await asyncio.wait_for(caller._wait(), 1.0)
        if await asyncio.wait_for(login(caller), IDLE_SECONDS):
            await main_menu(caller)
    except (Hangup, ConnectionError, asyncio.TimeoutError):
        pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:                                  # noqa: BLE001
        with contextlib.suppress(Exception):
            await caller.line(f"\r\n{A.HR}the board fell over: {exc}{A.RESET}")
    finally:
        board.leave(caller)
        board.callers.pop(caller.node, None)
        if caller.handle:                     # someone who never logged in
            board.page_all(f"{A.GREY}*** {caller.handle} hung up ***"
                           f"{A.RESET}")        # is not news
        await caller.close()


async def run_board(board: Board, host: str = "127.0.0.1",
                    port: int = DEFAULT_PORT) -> None:
    """Bind, then answer the telephone until something stops us."""
    async def handle(reader, writer):
        await session(reader, writer, board)

    keeper = asyncio.create_task(board.stats_keeper())
    server = await asyncio.start_server(handle, host, port)
    where = ", ".join(str(s.getsockname()[:2]) for s in server.sockets or [])
    print(f"  answering telnet on {where}")
    try:
        async with server:
            await server.serve_forever()
    finally:
        keeper.cancel()


def port_advice(port: int, exc: OSError) -> str:
    """Why the bind failed, and what to do about it, per platform."""
    import sys as _sys

    if getattr(exc, "errno", None) not in (13, 10013):        # EACCES, WSAEACCES
        return f"could not bind port {port}: {exc}"
    if _sys.platform == "win32":
        return (f"port {port} needs an elevated prompt on Windows.\n"
                f"  Right-click PowerShell, Run as administrator, and try "
                f"again -\n"
                f"  or use a high port that needs no privilege:\n"
                f"      mb bbs --port 2323\n"
                f"  then call it with:  telnet 127.0.0.1 2323")
    return (f"port {port} is privileged: on Unix only root may bind below "
            f"1024.\n"
            f"  Three ways, best first:\n"
            f"      mb bbs --port 2323                 no privilege needed\n"
            f"      sudo setcap 'cap_net_bind_service=+ep' $(readlink -f "
            f"$(which python3))\n"
            f"      sudo $(which mb) bbs               runs the board as root; "
            f"least good\n"
            f"  Callers reach a high port with:  telnet <host> 2323")


def serve(run_dir: str, corpus_dir: str, device: str = "auto",
          host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          password: str | None = None, sysop_password: str | None = None,
          insecure: bool = False, max_callers: int = MAX_CALLERS,
          max_tokens: int = 120, steps: int = 100, grow: int = 1) -> int:
    """Start the board. Blocking, and the same on every platform."""
    from motherbrain.security import check_exposure

    # Telnet has no encryption and never did. The board password is what
    # `--api-key` is to the HTTP server, and the same gate decides whether
    # facing a network without one is allowed.
    for warning in check_exposure(host, password, tls=False, insecure=insecure):
        print(f"warning: {warning}")
    if host not in ("127.0.0.1", "::1", "localhost"):
        print("warning: telnet is plaintext. The password, everything typed "
              "and everything\n         generated cross the network in the "
              "clear, and always will.")

    board = Board(run_dir, corpus_dir, device, password=password,
                  sysop_password=sysop_password, max_callers=max_callers,
                  max_per_address=MAX_PER_ADDRESS,
                  max_tokens=max_tokens, steps=steps, grow=grow)
    print(f"  loading MotherBrain from {run_dir} ...")
    try:
        board.load()
    except FileNotFoundError:
        print(f"no model in {run_dir}; run `mb bootstrap` first")
        return 1
    stats = board.stats()
    print(f"  v{stats.get('version', 0)}, "
          f"{stats.get('total_params_human', '?')} parameters, "
          f"on {board.torch_device}")
    print(f"  up to {max_callers} callers, {MAX_PER_ADDRESS} per address")
    print(f"  telnet {host if host != '0.0.0.0' else '<this machine>'} {port}")
    print()

    try:
        asyncio.run(run_board(board, host, port))
    except KeyboardInterrupt:
        print("\n  NO CARRIER")
    except OSError as exc:
        print(port_advice(port, exc))
        return 1
    return 0
