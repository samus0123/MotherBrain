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
from motherbrain import doors, warez, xmodem
from motherbrain import wwiv as W
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


DEFAULT_BULLETINS = [
    {"key": "1", "title": "What this board is",
     "body": "MotherBrain is a language model that grows by patching "
             "itself. It started at 18.9M parameters and is 52.2M now, "
             "across five patches, every one of them committed.\n\n"
             "This board is one of its four faces. The other three are a "
             "terminal, a window and a browser, and all four offer the "
             "same five things, in the same order.\n\n"
             "The model behind it is a base model. It completes text. It "
             "does not follow instructions, and nothing here pretends it "
             "does: every answer it gives is labelled with where the "
             "answer came from."},
    {"key": "2", "title": "Rules, such as they are",
     "body": "Upload what you like to GAMES and WHATEVERWARE. Do not "
             "upload anything you do not have the right to give away - "
             "the sysop deletes those and does not argue about it.\n\n"
             "Anything you feed the model under option 3 can come back "
             "out of it later, in front of somebody else. Post "
             "accordingly.\n\n"
             "Telnet is plaintext. Your password crosses the wire in the "
             "clear. Do not reuse one that matters."},
    {"key": "3", "title": "How the file area works",
     "body": "Every archive on the shelf carries a FILE_ID.DIZ, and the "
             "listing shows you what it says. Downloads go over XMODEM, "
             "which every terminal program has, or as base64 printed to "
             "the screen for anyone on a plain telnet.\n\n"
             "MBRAIN.ZIP is the whole program. MBDOORS.ZIP is the five "
             "door games on their own - they run on Python and nothing "
             "else. MBANSI.ZIP is the art."},
]

DEFAULT_POLLS = [
    {"key": "1", "question": "Should MotherBrain keep growing, or get better "
                             "at what it has?",
     "options": ["Grow - more parameters", "Improve - same size, more "
                 "training", "Learn to see properly first"],
     "votes": {}},
    {"key": "2", "question": "What should the sysop build next?",
     "options": ["More doors", "WWIVnet-style mail between boards",
                 "A better perception tower", "Leave it alone"],
     "votes": {}},
]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


# ---- one connection ---------------------------------------------------------

class Hangup(Exception):
    """The caller dropped carrier."""


class Goodbye(Exception):
    """The caller asked to leave, from wherever they were.

    Raised rather than returned because a caller who wants to log off from
    four screens deep should not have to press Q four times, and because
    every screen returning a "and also they want to leave" flag would put
    that check in fifty places instead of one.
    """


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
        self.user: W.User | None = None
        self.expert = False
        self.sub = "1"                      # the current message sub
        self.dir = "1"                      # the current file directory
        self.watchers: set[int] = set()     # sysop nodes reading over a shoulder
        # Where things were drawn, so a click can be turned into the key a
        # caller would have typed. A menu that can be pointed at is the same
        # menu; it just knows its own geometry.
        self.mouse = False
        # Where the caller is, screen by screen. The call stack already
        # takes them back one level; this is what lets the screen say so
        # by name, and what puts a breadcrumb at the top of every one.
        self.trail: list[str] = ["Main menu"]
        self.row = 0
        self.hotspots: list[tuple[int, int, int, str]] = []
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
        """Write to the wire, in the caller's encoding, at the caller's baud.

        Heart codes are resolved here and nowhere else. Every screen on the
        board is written in WWIV's notation - a heart and a digit - and this
        is the single place it becomes colour, which is what makes //COLORS
        change all of them at once. Text that is already ANSI passes through
        untouched, because there is no heart left in it to resolve.
        """
        if not text:
            return
        if W.HEART in text:
            text = W.render(text, tuple(self.board.colours))
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
        if self.watchers:
            self._echo(text)
        self.row += 1
        await self.send(text + "\r\n")

    @contextlib.contextmanager
    def at(self, name: str):
        """Mark a screen as entered, and leave the trail as it was found."""
        self.trail.append(name)
        try:
            yield self
        finally:
            if len(self.trail) > 1:
                self.trail.pop()

    def whence(self) -> str:
        """The screen a "go back" returns to."""
        return self.trail[-2] if len(self.trail) > 1 else "Main menu"

    def breadcrumb(self) -> str:
        return f"{A.GREY}" + f" {A.HB}»{A.GREY} ".join(self.trail) + A.RESET

    def hotspot(self, key: str, width: int | None = None) -> None:
        """Mark the line just written as clickable, standing for `key`."""
        self.hotspots.append((self.row, 1, width or self.columns, key))

    async def screen(self) -> None:
        """Start a new screen: clear it, and forget where everything was."""
        self.row = 0
        self.hotspots.clear()
        await self.send(A.CLS)

    def clicked(self, row: int, column: int) -> str:
        """The key a click at (row, column) stands for, or empty."""
        for at, first, last, key in self.hotspots:
            if at == row and first <= column <= last:
                return key
        return ""

    def _echo(self, text: str) -> None:
        """Copy a line to any sysop watching this node with //SPY."""
        for node in list(self.watchers):
            other = self.board.callers.get(node)
            if other is None:
                self.watchers.discard(node)
            else:
                other.tell(f"{A.GREY}[{self.node}]{A.RESET} {text}")

    async def art(self, text: str) -> None:
        body = text.replace("\r\n", "\n")
        self.row += body.count("\n") + 1
        await self.send(body.replace("\n", "\r\n") + "\r\n")

    async def cls(self) -> None:
        self.row = 0
        self.hotspots.clear()
        await self.send(A.CLS)

    async def pause(self, text: str = "press any key") -> None:
        """Wait for a key. G leaves the board, from this screen like any other.

        Screens that only show something and wait have no menu to put an
        exit on, so the exit is on the pause itself - which means there is
        no screen anywhere without a way out.
        """
        await self.send(f"{A.GREY}  [{A.HW}{text}{A.GREY}, or "
                        f"{A.HW}G{A.GREY} to log off]{A.RESET}")
        key = await self.key()
        await self.send("\r" + A.CLEAR_LINE)
        if key.upper() == "G":
            raise Goodbye()

    def tell(self, text: str) -> None:
        """Deliver a line from elsewhere - another node, or the sysop."""
        self.inbox.put_nowait(text)

    def minutes(self) -> float:
        """How long this call has lasted. WWIV counted in minutes, so do we."""
        return (time.time() - self.connected_at) / 60.0

    async def say(self, text: str) -> None:
        """A line written in heart codes, as every WWIV screen was."""
        await self.line(text)

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
                if key in ("\x03", "\x1b"):            # ctrl-c, escape
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
            if ch == "\x1b" and text[i + 1:i + 2] == "[" \
                    and text[i + 2:i + 3] == "<":
                # An SGR mouse report: ESC [ < button ; column ; row M or m.
                end = i + 3
                while end < len(text) and text[end] not in "Mm":
                    end += 1
                if end >= len(text):
                    break                       # cut in half; wait for more
                self._mouse(text[i + 3:end], text[end])
                i = end + 1
                continue
            if ch == "\x1b" and i + 1 >= len(text):
                self.keys.append("\x1b")            # a bare Escape: go back
                i += 1
                continue
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

    def _mouse(self, body: str, final: str) -> None:
        """One mouse report. A press on something clickable is that key.

        Only the press, and only button one: a menu that fired on release
        as well would run every command twice, and a menu that fired on
        scroll would run them at random.
        """
        if final != "M":
            return
        try:
            button, column, row = (int(part) for part in body.split(";"))
        except ValueError:
            return
        if button & 0b11 != 0 or button & 0x40:           # not a plain click
            return
        key = self.clicked(row, column)
        if key:
            for char in key:
                self.keys.append(char)
            self.keys.append("\r")

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
        self._engine = None
        self.max_batch = 8
        self.indexed = 0
        self.name = BOARD_NAME
        self.new_user_password: str | None = None
        # What somebody who has never called before can do. The default is
        # WWIV's: a new caller is unvalidated until the sysop says otherwise.
        # A board meant to be open to everyone raises this once, here or in
        # //CONFIG, rather than validating people one at a time forever.
        self.new_user_sl = W.NEW_USER_SL
        self.new_user_dsl = W.NEW_USER_DSL
        self.new_user_flags = ""
        # A caller from the machine the board runs on is the sysop. On a
        # board bound to loopback that is simply true; the flag exists so a
        # board facing a network can turn it off.
        self.trust_local = True
        self.closing = False
        self.colours = list(W.DEFAULT_COLOURS)
        self.users = W.Users(self.dir / "users.json")
        self.subs = [
            W.Sub("1", "General", "Anything at all."),
            W.Sub("2", "MotherBrain", "The model: what it is doing, what it "
                                      "got wrong."),
            W.Sub("3", "Programming", "Code, and what it wrote for you."),
            W.Sub("4", "Sysop", "For the sysop's attention.", post_sl=0),
        ]

    # -- storage --------------------------------------------------------------

    def load_settings(self) -> None:
        """Whatever the sysop changed with //CONFIG, //COLORS, //BOARDEDIT."""
        settings = self.read("config.json", {})
        self.name = settings.get("name", self.name)
        self.new_user_password = settings.get("new_user_password",
                                              self.new_user_password)
        self.max_callers = settings.get("max_callers", self.max_callers)
        self.max_per_address = settings.get("max_per_address",
                                            self.max_per_address)
        self.trust_local = settings.get("trust_local", self.trust_local)
        self.max_batch = settings.get("max_batch", self.max_batch)
        self.max_tokens = settings.get("max_tokens", self.max_tokens)
        self.new_user_sl = settings.get("new_user_sl", self.new_user_sl)
        self.new_user_dsl = settings.get("new_user_dsl", self.new_user_dsl)
        self.new_user_flags = settings.get("new_user_flags",
                                           self.new_user_flags)

        table = self.read("colours.json", None)
        if isinstance(table, list) and len(table) == len(W.DEFAULT_COLOURS):
            self.colours = [int(c) & 0xFF for c in table]

        subs = self.read("subs.json", None)
        if isinstance(subs, list) and subs:
            try:
                self.subs = [W.Sub(**entry) for entry in subs]
            except TypeError:
                pass

    def save_subs(self) -> None:
        from dataclasses import asdict

        self.write("subs.json", [asdict(sub) for sub in self.subs])

    def set_colours(self, table: list) -> None:
        self.colours = [int(c) & 0xFF for c in table]
        self.write("colours.json", self.colours)

    # -- users and their passwords --

    def sub(self, key: str) -> W.Sub:
        for entry in self.subs:
            if entry.key == key:
                return entry
        return self.subs[0]

    def _hash(self, password: str, salt: str) -> str:
        import hashlib

        return hashlib.pbkdf2_hmac("sha256", password.encode(),
                                   salt.encode(), 120_000).hex()

    def passwords(self) -> dict:
        return self.read("passwords.json", {})

    def has_password(self, user: W.User) -> bool:
        return str(user.number) in self.passwords()

    def set_password(self, user: W.User, password: str) -> None:
        import secrets as _secrets

        table = self.passwords()
        salt = _secrets.token_hex(16)
        table[str(user.number)] = {"salt": salt,
                                   "hash": self._hash(password, salt)}
        self.write("passwords.json", table)

    def clear_password(self, user: W.User) -> None:
        table = self.passwords()
        table.pop(str(user.number), None)
        self.write("passwords.json", table)

    def check_password(self, user: W.User, given: str) -> bool:
        from motherbrain.security import constant_time_eq

        entry = self.passwords().get(str(user.number))
        if not entry:
            return True
        return constant_time_eq(self._hash(given, entry["salt"]),
                                entry["hash"])

    # -- the small persistent things a board had --

    # -- what each file is, and how often it has been taken --

    def _file_key(self, path: Path) -> str:
        try:
            return str(Path(path).resolve())
        except OSError:
            return str(path)

    def descriptions(self) -> dict:
        return self.read("filedesc.json", {})

    def describe_file(self, path: Path, text: str, who: str) -> None:
        table = self.descriptions()
        table[self._file_key(path)] = {"text": text[:70], "by": who[:24],
                                       "when": now()}
        self.write("filedesc.json", table)

    def description_of(self, path: Path) -> str:
        """What a caller said about it, else what the file says about itself."""
        entry = self.descriptions().get(self._file_key(path))
        if entry:
            return entry["text"]
        return warez.describe(Path(path))

    def downloads(self) -> dict:
        return self.read("downloads.json", {})

    def bump_download(self, path: Path) -> int:
        table = self.downloads()
        key = self._file_key(path)
        table[key] = int(table.get(key, 0)) + 1
        self.write("downloads.json", table)
        return table[key]

    def download_count(self, path: Path) -> int:
        return int(self.downloads().get(self._file_key(path), 0))

    def motd(self) -> dict | None:
        """The message of the day. The sysop's, and only the sysop's."""
        return self.read("motd.json", None)

    def set_motd(self, text: str, who: str) -> None:
        if text.strip():
            self.write("motd.json", {"text": text[:2000], "by": who[:24],
                                     "when": now()})
        else:
            self.write("motd.json", None)

    def auto_message(self) -> dict | None:
        return self.read("automessage.json", None)

    def set_auto_message(self, who: str, text: str) -> None:
        self.write("automessage.json",
                   {"who": who[:30], "text": text[:200], "when": now()})

    def bulletins(self) -> list[dict]:
        return self.read("bulletins.json", DEFAULT_BULLETINS)

    def polls(self) -> list[dict]:
        return self.read("polls.json", DEFAULT_POLLS)

    def mail(self) -> list[dict]:
        return self.read("email.json", [])

    def send_mail(self, sender: str, to: int, subject: str,
                  body: str) -> None:
        box = self.mail()
        box.append({"from": sender[:30], "to": int(to),
                    "subject": subject[:60], "body": body[:4000],
                    "when": now(), "read": False})
        self.write("email.json", box[-2000:])

    def new_since(self, when_last: str) -> str:
        """One line on what has happened since a caller was last on."""
        threads = [t for t in self.threads() if t["when"][:10] > when_last]
        files = 0
        for section in file_sections(self):
            files += sum(1 for p in listing(section, 400)
                         if warez.when(p) > when_last)
        parts = []
        if threads:
            parts.append(f"{len(threads)} new message(s)")
        if files:
            parts.append(f"{files} new file(s)")
        return ("Since you were last on: " + ", ".join(parts) + "."
                if parts else "")

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

    async def read_the_corpus(self) -> None:
        """Index what it has read, so it can quote it. Off the event loop.

        A board that did this on the first question would make that caller
        wait eight seconds and nobody else, which is the kind of thing that
        looks like a fault rather than a cost.
        """
        from motherbrain import nlp

        index = await asyncio.to_thread(nlp.corpus_index, self.corpus_dir)
        self.indexed = len(index)

    async def stats_keeper(self, every: float = 120.0) -> None:
        """Keep the cache warm for as long as the board is up."""
        while True:
            await asyncio.sleep(every)
            with contextlib.suppress(Exception):
                await self.refresh_stats()

    async def generate(self, prompt: str, max_tokens: int | None = None,
                       temperature: float = 0.8) -> str:
        """Continue a prompt, through the batching engine.

        Callers arrive independently, so on a busy board several are waiting
        on the model at any moment. Running them one after another costs one
        generation each; running them in one batch costs about one
        generation for the lot, because the arithmetic is the same matrices
        either way. That is the difference between a board that seats eight
        and one that seats hundreds.
        """
        if self.model is None:
            return ""

        from motherbrain.inference import Request

        self.waiting += 1
        try:
            return await self.engine.submit(Request(
                prompt, max_new_tokens=max_tokens or self.max_tokens,
                temperature=temperature, top_k=40, top_p=0.95,
                repetition_penalty=1.15))
        finally:
            self.waiting -= 1

    async def answer(self, text: str, caller: Caller,
                     generate: bool = False) -> tuple[str, str]:
        """Computed, told, read off its own state, quoted - then, and only
        then, generated.

        Exactly what the terminal, the window and the browser do, in the
        same order, for the same reason: a definite answer is never
        generated when it can be worked out, and a caller is always told
        which of the five this was.
        """
        from motherbrain import nlp

        # One pipeline, off the event loop: it reads the sentence, works out
        # what is being asked, and composes a reply out of something that is
        # actually true - computed, told, read off its own state, or quoted
        # from what it has read. The sampler is not in that list.
        found = await asyncio.to_thread(
            nlp.answer, text, run_dir=self.run_dir,
            corpus_dir=self.corpus_dir, stats=self.stats())
        if found.source != "none":
            return found.source, found.render()
        if not generate:
            # "I do not know" is the answer. Following it with fluent prose
            # about nothing takes it back, and the prose is the part people
            # remember.
            return "none", found.text

        # Asked for explicitly - by the Oracle door, whose whole purpose is
        # showing what the model does on its own.
        from motherbrain.chat import CONTINUATION_NOTE

        produced = (await self.generate(text)).strip()
        if not produced:
            return "none", found.text
        return "generated", f"{found.text}\n\n{produced}\n\n" \
                            f"{CONTINUATION_NOTE}"

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

    @property
    def engine(self):
        """The batching engine, made on first use so it binds this loop."""
        from motherbrain.inference import Engine

        if self._engine is None:
            from motherbrain.tokenizer import EOS_ID

            self._engine = Engine(self.model, self.tok, self.torch_device,
                                  eos_id=EOS_ID, max_batch=self.max_batch)
            self._engine.start()
        return self._engine

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
             parent: int | None = None, sub: str = "1") -> int:
        threads = self.threads()
        entry = {"id": len(threads) + 1, "author": author[:24],
                 "subject": subject[:60], "body": body[:4000],
                 "when": now(), "parent": parent, "sub": sub}
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
    """The file directories, in WWIV's shape: gated by DSL and DAR.

    Roots are named here and nowhere else. A caller never supplies a path -
    they pick a number off a list this function built - which is what keeps
    a file area from being an arbitrary-file-read primitive with a menu in
    front of it.
    """
    run = Path(board.run_dir).resolve()
    repo = Path(__file__).resolve().parent.parent
    shelf = (run / "bbs" / "warez").resolve()
    user = (run / "bbs" / "files").resolve()
    for public in ("games", "whateverware"):
        (run / "bbs" / "public" / public).mkdir(parents=True, exist_ok=True)

    return [
        {"key": "1", "name": "WAREZ", "dsl": 0, "dar": "",
         "blurb": "Releases. The whole program, the doors on their own, and "
                  "an ANSI pack - each a real archive with a FILE_ID.DIZ in "
                  "it. All of it MotherBrain, all of it free.",
         "roots": [shelf], "globs": ["*"]},
        {"key": "2", "name": "MODELS", "dsl": 10, "dar": "",
         "blurb": "The weights themselves. Large, and the reason the board "
                  "has a file transfer protocol at all.",
         "roots": [run / "models", repo / "models", run / "patches"],
         "globs": ["*.pt", "tokenizer.json"]},
        {"key": "3", "name": "SOURCE", "dsl": 0, "dar": "",
         "blurb": "Every line of the board, the model and the doors, loose.",
         "roots": [repo / "motherbrain", repo / "scripts"],
         "globs": ["*.py", "*.sh", "*.ps1"]},
        {"key": "4", "name": "TEXTFILES", "dsl": 0, "dar": "",
         "blurb": "Documentation, in the finest tradition of the g-files.",
         "roots": [repo, repo / "docs"], "globs": ["*.md", "*.txt"],
         "flat": True},
        {"key": "5", "name": "GAMES", "dsl": 0, "dar": "", "upload": True,
         "public": True,
         "blurb": "Games. Anyone may upload, anyone may download. Doors, "
                  "BASIC listings, whatever you have got.",
         "roots": [run / "bbs" / "public" / "games"], "globs": ["*"]},
        {"key": "6", "name": "WHATEVERWARE", "dsl": 0, "dar": "",
         "public": True, "upload": True,
         "blurb": "Whatever. Open to everyone, both ways. Put something in "
                  "it and somebody will take it out.",
         "roots": [run / "bbs" / "public" / "whateverware"], "globs": ["*"]},
        {"key": "7", "name": "USER", "dsl": 0, "dar": "", "upload": True,
         "blurb": "Your own directory: what you uploaded, and what "
                  "MotherBrain wrote for you when you asked.",
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


# ---- the screens, in WWIV's shape -------------------------------------------

# The main menu. The first five are the console's, word for word and in the
# same order - a board that quietly renumbered them would be a different
# program wearing the same name. The letters around them are WWIV's own,
# with WWIV's meanings: T is the transfer section, A is the auto-message, F
# is feedback to the sysop, X toggles expert mode.
CONSOLE_OPTIONS = [
    ("1", "Tell MotherBrain what kind of program to make"),
    ("2", "Tell MotherBrain what to do"),
    ("3", "Teach MotherBrain something new"),
    ("4", "Apply new knowledge as a patch (update)"),
    ("5", "Run the GUI"),
]

# (letter, label, what an SL below this cannot reach)
MESSAGE_COMMANDS = [
    ("R", "Read messages", 0),
    ("P", "Post", 10),
    ("Q", "Quick-scan new", 0),
    ("E", "E-mail someone", 10),
    ("F", "Feedback to sysop", 0),
    ("A", "Auto-message", 0),
    ("B", "Bulletins", 0),
    ("V", "Voting booth", 0),
]

# The console's wording is what a caller sees wherever there is room for
# it. On a phone there is not, and a truncated label - "Tell MotherBrain
# what kind of pr" - is worse than a short one that says the same thing.
CONSOLE_SHORT = {
    "1": "Make me a program",
    "2": "Tell me what to do",
    "3": "Teach me something",
    "4": "Apply a patch",
    "5": "Run the GUI",
}

BOARD_COMMANDS = [
    ("T", "Transfer section", 0),
    ("N", "New-file scan", 0),
    ("C", "Chat with MotherBrain", 0),
    ("M", "Multi-node chat", 0),
    ("D", "Doors", 0),
    ("I", "System information", 0),
    ("Y", "Your statistics", 0),
    ("W", "Who is online", 0),
]

# Short on purpose: these four share one row, and a label that has to be
# truncated to fit is a label nobody can read on a phone.
MINOR_COMMANDS = [
    ("L", "Callers", 0),
    ("X", "Expert", 0),
    ("?", "Menu", 0),
    ("G", "Good-bye", 0),
]


# Below this the two-column menu stops being two columns. A phone held
# upright is about forty characters wide, and a board that insists on
# sixty-four there just runs off the side of the screen.
NARROW = 60


def wide(caller: Caller) -> int:
    """The width to draw at: what the caller has, within reason.

    The floor used to be sixty-four, which is fine for a terminal and wrong
    for a phone - the screens were drawn wider than the screen and the
    right-hand half was simply gone.
    """
    return min(max(caller.columns, 38), 79)


async def header(caller: Caller) -> None:
    """The shaded header a board put at the top of every screen."""
    board = caller.board
    width = wide(caller)
    await caller.art("\n".join(A.banner(
        f"{board.name}  B B S L L M", width,
        "Bulletin Board System / Large Language Model")))


async def footer(caller: Caller) -> None:
    """The status bar along the bottom: who, where, and how long left."""
    board = caller.board
    user = caller.user
    left = user.minutes_left() - caller.minutes() if user else 0.0
    stats = board.stats()
    await caller.line(A.status(
        f"{user.name if user else '?'} #{user.number if user else 0}"
        f"   {board.sub(caller.sub).name}",
        f"Node {caller.node}  v{stats.get('version', 0)} "
        f"{stats.get('total_params_human', '?')}  {left:.0f} min left",
        wide(caller)))


async def login(caller: Caller) -> bool:
    """WWIV's login: user number or name, then a password. False hangs up."""
    board = caller.board
    await caller.cls()
    await caller.art(A.logo())
    await caller.line(A.shaded_bar(min(caller.columns, 78)))
    await caller.line(W.render(A.centre(
        f"\x039{board.name} \x031- \x032a BBSLLM",
        min(caller.columns, 78))))
    await caller.line(W.render(A.centre(
        "\x030a Bulletin Board System with a Large Language Model in it",
        min(caller.columns, 78))))
    await caller.line(W.render(A.centre(
        f"\x030a v{board.version} language model, answering its own "
        f"telephone", min(caller.columns, 78))))
    await caller.line("")

    if board.password:
        for _ in range(3):
            given = await caller.ask("\x032System password: \x030",
                                     limit=128, mask=True)
            from motherbrain.security import constant_time_eq
            if constant_time_eq(given, board.password):
                break
            await caller.line("\x036No.\x030")
        else:
            await caller.line("\x030Good-bye.")
            return False

    for _ in range(4):
        who = (await caller.ask(
            "\x035Enter your user number or name \x030(NEW for a new "
            "account)\x035: \x030", limit=32)).strip()
        if not who:
            continue
        if who.upper() in ("NEW", "NEWUSER"):
            user = await new_user(caller)
            if user is None:
                return False
            caller.user = user
            break
        user = board.users.find(who)
        if user is None:
            await caller.line("\x036That user is not on this system.\x030")
            continue
        if board.has_password(user):
            given = await caller.ask("\x032Password: \x030", limit=64,
                                     mask=True)
            if not board.check_password(user, given):
                await caller.line("\x036Incorrect.\x030")
                continue
        caller.user = user
        break
    else:
        await caller.line("\x030Good-bye.")
        return False

    user = caller.user
    caller.handle = user.name
    caller.expert = user.expert
    # The sysop is the sysop. Calling from the machine the board runs on is
    # proof enough of that; from anywhere else it takes the sysop password.
    if caller.local and board.trust_local:
        user.sl = max(user.sl, W.SYSOP_SL)
        user.dsl = max(user.dsl, W.SYSOP_SL)
        user.ar = user.dar = "ABCDEFGHIJKLMNOP"
    elif board.sysop_password:
        given = await caller.ask("\x033Sysop password \x030(blank if you are "
                                 "not)\x033: \x030", limit=128, mask=True)
        from motherbrain.security import constant_time_eq
        if given and constant_time_eq(given, board.sysop_password):
            user.sl = user.dsl = W.SYSOP_SL
            user.ar = user.dar = "ABCDEFGHIJKLMNOP"
    caller.sysop = user.sysop
    board.users.begin_call(user)

    if user.minutes_left() <= 0:
        await caller.line("\x036You have used all of today's time.\x030")
        return False

    await caller.line("")
    await caller.line(W.render(
        f"\x031Welcome, \x039{user.name} \x031#\x032{user.number}"
        f"\x031. This is call \x032{user.calls}\x031, and you have "
        f"\x032{user.minutes_left():.0f}\x031 minutes."))
    if user.sysop:
        await caller.line("\x032You are the sysop. \x030// gets you the "
                          "internals; //? lists them.\x030")
    await caller.line("")

    motd = board.motd()
    if motd:
        lines = []
        for paragraph in motd["text"].split("\n"):
            for line in (A.wrap(paragraph, wide(caller) - 6) or [""]):
                lines.append(W.render(f"\x032{line}"))
        lines.append("")
        lines.append(W.render(f"\x030  - {motd['by']}, {motd['when']}"))
        await caller.art("\n".join(A.box("MESSAGE OF THE DAY", lines,
                                          width=wide(caller), frame=A.HY,
                                          head=A.HR)))
        await caller.line("")

    auto = board.auto_message()
    if auto:
        await caller.art("\n".join(A.box(
            "AUTO-MESSAGE", [W.render(f"\x032{auto['text']}"),
                             W.render(f"\x030          - {auto['who']}, "
                                      f"{auto['when']}")],
            width=wide(caller), frame=A.HB)))
    new = board.new_since(user.last_on)
    if new:
        await caller.line(W.render(f"\x032{new}\x030"))
    board.page_all(W.render(f"\x032*** {user.name} is on node "
                            f"{caller.node} ***\x030"), skip=caller.node)
    await caller.pause()
    return True


async def new_user(caller: Caller):
    """WWIV's new-user application, cut to what this board actually needs."""
    board = caller.board
    if board.new_user_password:
        for _ in range(3):
            given = await caller.ask("\x032New-user password: \x030",
                                     limit=64, mask=True)
            from motherbrain.security import constant_time_eq
            if constant_time_eq(given, board.new_user_password):
                break
            await caller.line("\x036No.\x030")
        else:
            return None

    name = ""
    while not name:
        name = (await caller.ask("\x035Handle: \x030", limit=30)).strip()
        name = "".join(c for c in name if c.isprintable())[:30]
        if not name:
            continue
        if name.upper() in ("SYSOP", "NEW", "NEWUSER", SYSOP.upper()):
            await caller.line("\x036That name is taken by the machine."
                              "\x030")
            name = ""
        elif board.users.find(name) is not None:
            await caller.line("\x036Somebody already calls themselves that."
                              "\x030")
            name = ""

    real = (await caller.ask("\x035Real name \x030(optional)\x035: \x030",
                             limit=40)).strip()
    password = await caller.ask("\x035Choose a password \x030(blank for "
                                "none)\x035: \x030", limit=64, mask=True)

    user = board.users.create(name, sl=board.new_user_sl,
                              dsl=board.new_user_dsl,
                              flags=board.new_user_flags)
    user.real_name = real[:40]
    if password:
        board.set_password(user, password)
    board.users.save()
    await caller.line("")
    await caller.line(W.render(
        f"\x031You are user \x032#{user.number}\x031, at security level "
        f"\x032{user.sl}\x031."))
    await caller.line(W.render(
        "\x030The sysop raises that with //UEDIT once they know who you "
        "are.\x030"))
    board.page_all(W.render(f"\x032*** {user.name} is a new user "
                            f"(#{user.number}) ***\x030"))
    return user


async def draw_menu(caller: Caller) -> None:
    """The main menu, drawn the way a 1987 board drew one.

    Shaded header, panels with drop shadows, keys in brackets so the eye
    finds them, and a status bar along the bottom. Every entry registers
    where it was drawn, so a click on it is the same as typing it.
    """
    user = caller.user
    width = wide(caller)
    narrow = width < NARROW
    await caller.screen()
    await header(caller)

    # -- what MotherBrain itself does --
    top = A.panel("MOTHERBRAIN", ["" for _ in CONSOLE_OPTIONS],
                  width=width - 2, frame=A.HC, head=A.HY)
    await caller.line(top[0])
    for (key, label), _ in zip(CONSOLE_OPTIONS, top[1:]):
        if width < NARROW:
            label = CONSOLE_SHORT.get(key, label)
        await caller.line(_framed(A.entry(key, label), width - 2, A.HC))
        caller.hotspot(key, width)
    await caller.line(top[-2])
    await caller.line(top[-1])

    # -- the board: two columns when there is room, one when there is not --
    columns = ([MESSAGE_COMMANDS + BOARD_COMMANDS] if narrow
               else [MESSAGE_COMMANDS, BOARD_COMMANDS])
    half = (width - 6) if narrow else (width - 6) // 2
    rows = max(len(table) for table in columns)
    frame = A.panel("THE BOARD", [""] * (rows + 4), width=width - 2,
                    frame=A.HB, head=A.HY)
    await caller.line(frame[0])
    if not narrow:
        await caller.line(_framed(
            f"{A.HC}{A.pad('MESSAGES', half)}{A.HC}THE BOARD", width - 2,
            A.HB))
    for i in range(rows):
        line, keys = "", []
        for table in columns:
            if i < len(table):
                key, label, needs = table[i]
                ok = user is None or user.sl >= needs
                cell = A.entry(key, label, available=ok)
                keys.append((key, ok))
            else:
                cell = ""
                keys.append(("", False))
            line += A.pad(cell, half)
        await caller.line(_framed(line, width - 2, A.HB))
        # Left half and right half are separately clickable.
        for column, (key, ok) in enumerate(keys):
            if key and ok:
                first = 2 + column * half
                caller.hotspots.append((caller.row, first,
                                        first + half, key))
    await caller.line(_framed("", width - 2, A.HB))

    per_row = 2 if narrow else len(MINOR_COMMANDS)
    cell = (width - 8) // per_row
    for start in range(0, len(MINOR_COMMANDS), per_row):
        chunk = MINOR_COMMANDS[start:start + per_row]
        await caller.line(_framed(
            "".join(A.pad(A.entry(key, label), cell)
                    for key, label, _needs in chunk), width - 2, A.HB))
        for column, (key, _label, _needs) in enumerate(chunk):
            first = 2 + column * cell
            caller.hotspots.append((caller.row, first, first + cell, key))
    await caller.line(_framed(
        f"{A.GREY}/A change sub   /D change directory"
        + ("   // sysop" if user is not None and user.sysop else ""),
        width - 2, A.HB))
    await caller.line(frame[-2])
    await caller.line(frame[-1])
    await footer(caller)


def _framed(body: str, width: int, frame: str) -> str:
    """One row inside a double-ruled panel, with the drop shadow on the end."""
    inner = width - 2
    return (f"{frame}║{A.RESET} {A.pad(body, inner - 1)}{frame}║"
            f"{A.RESET}{A.SHADOW}██{A.RESET}")


def _command_cell(user, item) -> str:
    """One menu entry, dimmed when the caller's SL cannot reach it."""
    key, label, needs = item
    if user is not None and user.sl < needs:
        return f"\x030{key}) {label}"
    return f"\x032{key}\x030) \x039{label}"


async def options(caller: Caller, extra: str = "") -> None:
    """The two lines every menu ends with: go back, and leave the board.

    Every screen has both. A caller four screens deep should be able to
    log off without pressing Q four times, and a caller who does not know
    where they are should be told where "back" goes.
    """
    await caller.line("")
    if extra:
        await caller.line(f"  {extra}")
    await caller.line(
        f"  {A.entry('Q', f'Go back to {caller.whence()}')}"
        f"    {A.entry('G', 'Exit and log off')}")
    caller.hotspots.append((caller.row, 1, 40, "Q"))
    caller.hotspots.append((caller.row, 41, caller.columns, "G"))


async def menu_choice(caller: Caller, prompt: str = "", limit: int = 8) -> str:
    """Ask a menu question. Q goes back, G leaves the board, from anywhere.

    Free-text screens - chat, teaching, writing a message - do not use
    this, because there "G" is a letter somebody meant to type.
    """
    answer = (await caller.ask(
        prompt or f"  {A.HY}Choice{A.HB}:{A.RESET} ", limit=limit)).strip()
    if answer.upper() in ("G", "O", "OFF", "BYE", "GOODBYE", "EXIT", "QUIT"):
        raise Goodbye()
    return answer


async def go_back(caller: Caller, key: str = "Q") -> None:
    """The line every screen ends with, naming the screen it returns to.

    "Back" on its own makes a caller guess. Naming the destination costs a
    dozen characters and means nobody has to.
    """
    await caller.line("")
    await caller.line(
        f"  {A.entry(key, f'Go back to {caller.whence()}')}"
        f"   {A.GREY}(Q, escape, or an empty line){A.RESET}")
    caller.hotspot(key)


async def crumbs(caller: Caller) -> None:
    """Where you are, at the top of the screen, as a board did it."""
    await caller.line(f"  {caller.breadcrumb()}")


def leaving(answer: str) -> bool:
    """Every screen agrees on what "go back" looks like when typed."""
    stripped = answer.strip().upper()
    return stripped in ("", "Q", "X", "BACK", "\x1b")


async def prompt(caller: Caller) -> str:
    """WWIV's command prompt: where you are, and how long you have left."""
    board = caller.board
    user = caller.user
    left = (user.minutes_left() - caller.minutes()) if user else 0.0
    sub = board.sub(caller.sub)
    return await caller.ask(
        f"{A.HB}[{A.HW}{sub.name}{A.HB}]{A.RESET} "
        f"{A.GREY}{left:.0f} min{A.RESET} {A.HY}Command{A.RESET}"
        f"{A.HB}:{A.RESET} ", limit=64)


async def main_menu(caller: Caller) -> None:
    """Draw, read a command, run it. Expert mode skips the drawing."""
    board = caller.board
    user = caller.user
    actions = 0

    while True:
        if user is not None and user.minutes_left() - caller.minutes() <= 0:
            await caller.line(W.render("\x036Your time is up for today."
                                       "\x030"))
            break
        if not caller.expert:
            await draw_menu(caller)
        command = (await prompt(caller)).strip()
        if not command:
            continue

        if command.startswith("//"):
            if not caller.sysop:
                await caller.line(W.render("\x036Sysop only.\x030"))
                await caller.pause()
                continue
            if await sysop_command(caller, command[2:].strip()):
                break
            continue
        if command.startswith("/"):
            await slash_command(caller, command[1:].strip())
            continue

        key = command[0].upper()
        if key in ("G", "O"):
            break
        handler = HANDLERS.get(key)
        if handler is None:
            await caller.line(W.render(f"\x036{key} is not a command. "
                                       f"\x030? for the menu.\x030"))
            await caller.pause()
            continue
        needs = _needs(key)
        if user is not None and user.sl < needs:
            await caller.line(W.render(
                f"\x036Your security level is {user.sl}; that needs "
                f"{needs}.\x030"))
            await caller.pause()
            continue
        actions += 1
        name = _screen_name(key)
        try:
            with caller.at(name):
                await handler(caller)
        except Goodbye:
            break
        except Hangup:
            raise
        except Exception as exc:                          # noqa: BLE001
            await caller.line(W.render(f"\x036That went wrong: {exc}\x030"))
            await caller.pause()

    board.log_call(caller, actions)
    if caller.user is not None:
        caller.user.expert = caller.expert
        board.users.end_call(caller.user, caller.minutes())
    await goodbye(caller)


def _screen_name(key: str) -> str:
    """What to call the screen a command opens, for the trail and the crumb."""
    for table in (CONSOLE_OPTIONS,):
        for entry, label in table:
            if entry == key:
                return label.split("(")[0].strip()
    for table in (MESSAGE_COMMANDS, BOARD_COMMANDS, MINOR_COMMANDS):
        for entry, label, _needs in table:
            if entry == key:
                return label
    return "a screen"


def _needs(key: str) -> int:
    for table in (MESSAGE_COMMANDS, BOARD_COMMANDS, MINOR_COMMANDS):
        for entry, _label, needs in table:
            if entry == key:
                return needs
    return 0


async def goodbye(caller: Caller) -> None:
    await caller.cls()
    await caller.art(A.logo())
    await caller.line("")
    user = caller.user
    if user is not None:
        await caller.line(W.render(
            f"\x031You were on for \x032{caller.minutes():.1f}\x031 "
            f"minutes. That is call \x032{user.calls}\x031, and "
            f"\x032{user.minutes:.0f}\x031 minutes in all."))
    await caller.line(W.render("\x039NO CARRIER\x030"))
    await caller.line("")


# ---- the five console options ----------------------------------------------

async def option_make(caller: Caller) -> None:
    """1 - describe a program; it writes one and files it under your handle."""
    board = caller.board
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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

    await caller.screen()
    await header(caller)
    await crumbs(caller)
    await caller.line("")
    await caller.line(f"  {A.HY}TELL MOTHERBRAIN WHAT TO DO{A.RESET}")
    await caller.line(f"  {A.GREY}Ask it things. Blank line to go back."
                      f"{A.RESET}")
    await caller.line(f"  {A.GREY}Running programs, reading files and shell "
                      f"commands are refused here:{A.RESET}")
    await caller.line(f"  {A.GREY}over a wire that is not convenience, it is "
                      f"a shell on the sysop's machine.{A.RESET}")
    await caller.line("")

    await options(caller)
    await caller.line("")

    while True:
        line = (await caller.ask(f"  {A.HG}>{A.RESET} ", limit=400)).strip()
        if not line or line.upper() == "Q":
            return
        if line.upper() in ("G", "OFF", "BYE"):
            raise Goodbye()
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
        from motherbrain import nlp

        nlp.forget_index()                        # the corpus grew
        asyncio.create_task(board.read_the_corpus())
        if board._engine is not None:
            await board._engine.stop()
            board._engine = None                  # the next call builds one
        await board.refresh_stats()               # around the new weights
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
    await options(caller, A.entry("?", "what it can answer", note=""))
    await caller.line("")

    while True:
        said = (await caller.ask(f"  {A.HG}{caller.handle}>{A.RESET} ",
                                 limit=400)).strip()
        if not said or said.upper() == "Q":
            return
        if said.upper() in ("G", "OFF", "BYE"):
            raise Goodbye()
        if said == "?":
            await caller.say("\x030Arithmetic it computes. Things you tell "
                             "it, it keeps. Questions about\r\n  itself it "
                             "reads off disk. Anything in its corpus it "
                             "quotes. Anything\r\n  else it says it does "
                             "not know.")
            continue
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
        await caller.screen()
        await header(caller)
        await crumbs(caller)
        await caller.line("")
        rows = [f"  {A.HY}[{key}]{A.RESET}  {A.HW}{A.pad(name, 12)}"
                f"{A.GREY}{blurb}{A.RESET}"
                for key, name, blurb in doors.CATALOGUE]
        rows.append("")
        rows.append(A.entry("Q", f"Go back to {caller.whence()}"))
        rows.append(A.entry("G", "Exit and log off"))
        await caller.art("\n".join(A.box("D O O R S", rows,
                                         width=min(caller.columns, 76),
                                         frame=A.HM)))
        await caller.line("")
        key = await menu_choice(caller, f"  {A.HY}Door{A.HB}:{A.RESET} ", 2)
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
    """R - read the current sub. /A moves you to another one."""
    board = caller.board
    while True:
        here = caller.sub
        threads = [t for t in board.threads() if t.get("sub", "1") == here]
        roots = [t for t in threads if t.get("parent") is None]
        await caller.screen()
        await header(caller)
        await crumbs(caller)
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
        await caller.art("\n".join(A.box(
            f"{board.sub(here).name.upper()}  -  {board.sub(here).blurb}",
            rows, width=wide(caller), frame=A.HG)))
        await caller.line(f"  {A.HY}[P]{A.RESET} post   "
                          f"{A.HY}[number]{A.RESET} read   "
                          f"{A.HY}[Q]{A.RESET} back")
        choice = await menu_choice(caller, f"  {A.HY}Message{A.HB}:{A.RESET} ", 8)
        if not choice or choice.upper() == "Q":
            return
        if choice.upper() == "P":
            subject = (await caller.ask(f"  subject: ", limit=60)).strip()
            if not subject:
                continue
            await caller.line(f"  {A.GREY}body, blank line to end:{A.RESET}")
            body = await _multiline(caller)
            if body:
                board.post(caller.handle, subject, body, sub=here)
                if caller.user is not None:
                    caller.user.posts += 1
                    board.users.save()
                board.page_all(W.render(
                    f"\x032*** {caller.handle} posted \"{subject}\" in "
                    f"{board.sub(here).name} ***\x030"), skip=caller.node)
            continue
        if choice.isdigit():
            await _read_thread(caller, int(choice))


async def post_message(caller: Caller) -> None:
    """P - post to the sub you are standing in."""
    board = caller.board
    here = board.sub(caller.sub)
    user = caller.user
    if user is not None and user.sl < here.post_sl:
        await caller.say(f"\x036Posting in {here.name} needs SL "
                         f"{here.post_sl}; yours is {user.sl}.\x030")
        await caller.pause()
        return
    await caller.screen()
    await header(caller)
    await crumbs(caller)
    await caller.line("")
    await caller.say(f"\x031Posting in \x039{here.name}\x030")
    subject = (await caller.ask(W.render("\x035Subject: \x030"),
                                limit=60)).strip()
    if not subject:
        return
    await caller.say("\x030Body, blank line to end:")
    body = await _multiline(caller)
    if not body:
        return
    board.post(caller.handle, subject, body, sub=caller.sub)
    if user is not None:
        user.posts += 1
        board.users.save()
    board.page_all(W.render(f"\x032*** {caller.handle} posted "
                            f"\"{subject}\" in {here.name} ***\x030"),
                   skip=caller.node)
    await caller.say("\x032Posted.\x030")
    await caller.pause()


async def quick_scan(caller: Caller) -> None:
    """Q - WWIV's quick-scan: what is new everywhere, in one screen."""
    board = caller.board
    user = caller.user
    since = user.last_on if user else ""
    await caller.screen()
    await header(caller)
    await crumbs(caller)
    await caller.line("")
    threads = board.threads()
    rows = []
    for sub in board.subs:
        here = [t for t in threads if t.get("sub", "1") == sub.key]
        fresh = [t for t in here if t["when"][:10] >= since]
        mark = "\x032" if fresh else "\x030"
        rows.append(W.render(
            f"  {mark}{A.pad(sub.key, 3)}\x039{A.pad(sub.name, 18)}"
            f"\x030{len(here):>4} message(s), \x032{len(fresh)}\x030 new"))
    rows.append("")
    rows.append(W.render("  \x030/A moves you to a sub, R reads it."))
    await caller.art("\n".join(A.box("Q U I C K   S C A N", rows,
                                      width=wide(caller), frame=A.HG)))
    await caller.pause()


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

    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
            board.post(caller.handle, f"Re: {root['subject']}", body, ident,
                       sub=root.get("sub", "1"))
            if caller.user is not None:
                caller.user.posts += 1
                board.users.save()
    elif key == "M":
        await caller.line(f"  {A.GREY}thinking...{A.RESET}")
        source, text = await board.answer(root["body"], caller)
        board.post(SYSOP, f"Re: {root['subject']}",
                   f"[{source}] {text}", ident, sub=root.get("sub", "1"))
        await caller.line(f"  {doors._label(source)}")
        for line in A.wrap(text, caller.columns - 4):
            await caller.line(f"  {A.HW}{line}{A.RESET}")
        await caller.pause()


# ---- one-liners -------------------------------------------------------------

async def oneliner_wall(caller: Caller) -> None:
    """O - the graffiti wall every board had."""
    board = caller.board
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
        await caller.screen()
        await header(caller)
        await crumbs(caller)
        await caller.line("")
        rows = []
        user = caller.user
        for section in sections:
            if user is not None and not user.may_download(
                    section.get("dar", ""), section.get("dsl", 0)):
                rows.append(W.render(
                    f"  \x030[{section['key']}] {A.pad(section['name'], 12)}"
                    f"  needs DSL {section.get('dsl', 0)}"))
                continue
            count = len(listing(section, 400))
            rows.append(f"  {A.HY}[{section['key']}]{A.RESET} "
                        f"{A.HW}{A.pad(section['name'], 12)}{A.RESET}"
                        f"{A.GREY}{count:>4} files{A.RESET}")
            for line in A.wrap(section["blurb"], min(caller.columns, 70) - 8):
                rows.append(f"       {A.GREY}{line}{A.RESET}")
        rows.append("")
        rows.append(f"  {A.HY}[U]{A.RESET} {A.HW}upload{A.RESET}"
                    f"{A.GREY}   send the board a file over XMODEM{A.RESET}")
        rows.append(A.entry("Q", f"Go back to {caller.whence()}"))
        rows.append(A.entry("G", "Exit and log off"))
        await caller.art("\n".join(A.box("F I L E   A R E A", rows,
                                         width=min(caller.columns, 76),
                                         frame=A.HY)))
        key = await menu_choice(caller, f"  {A.HY}Area{A.HB}:{A.RESET} ", 2)
        if not key or key.upper() == "Q":
            return
        if key.upper() == "U":
            await _upload(caller)
            continue
        section = next((s for s in sections if s["key"] == key), None)
        if section is None:
            continue
        user = caller.user
        if user is not None and not user.may_download(section.get("dar", ""),
                                                      section.get("dsl", 0)):
            await caller.say(f"\x036{section['name']} needs DSL "
                             f"{section.get('dsl', 0)}; yours is "
                             f"{user.dsl}.\x030")
            await caller.pause()
            continue
        caller.dir = section["key"]
        with caller.at(section['name']):
            await _browse(caller, section)


async def _browse(caller: Caller, section: dict) -> None:
    files = listing(section, 400)
    page = 0
    per_page = max(8, min(18, caller.rows - 10))
    while True:
        await caller.screen()
        await header(caller)
        await crumbs(caller)
        await caller.line("")
        per_page = max(4, min(9, (caller.rows - 12) // 2))
        chunk = files[page * per_page:(page + 1) * per_page]
        rows = []
        for i, path in enumerate(chunk, start=page * per_page + 1):
            taken = caller.board.download_count(path)
            rows.append(W.render(
                f" \x032{i:>3}\x030 \x039{A.pad(path.name, 30)}"
                f"\x030{size_of(path):>9}  {warez.when(path)}"
                f"  \x030{taken:>3}dl"))
            # The description under the name is the whole reason a file area
            # reads as a shelf rather than a directory listing.
            for line in A.wrap(caller.board.description_of(path),
                               wide(caller) - 10):
                rows.append(W.render(f"      \x030{line}"))
        if not rows:
            rows = [W.render("  \x030Empty. Somebody has to be first.")]
        await caller.art("\n".join(A.box(
            f"{section['name']}  ({len(files)} files)", rows,
            width=wide(caller), frame=A.HY)))
        pages = max(1, (len(files) + per_page - 1) // per_page)
        takes = section.get("upload")
        await caller.say(
            f"  \x030page {page + 1}/{pages}   \x032N\x030)ext  "
            f"\x032B\x030)ack  \x032number\x030) take it"
            + ("   \x032U\x030)pload" if takes else ""))
        await options(caller)
        choice = await menu_choice(caller, W.render("\x035File\x030: "), 6)
        if not choice or choice.upper() == "Q":
            return
        if choice.upper() == "N":
            page = min(page + 1, pages - 1)
            continue
        if choice.upper() == "B":
            page = max(0, page - 1)
            continue
        if choice.upper() == "U" and takes:
            await _upload(caller, section)
            files = listing(section, 400)
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            with caller.at(files[int(choice) - 1].name):
                await _download(caller, files[int(choice) - 1])


async def _download(caller: Caller, path: Path) -> None:
    board = caller.board
    await caller.cls()
    await caller.line("")
    await caller.say(f"  \x039{path.name}\x030  {size_of(path)}  "
                     f"{warez.when(path)}  "
                     f"{board.download_count(path)} download(s)")
    for line in A.wrap(board.description_of(path), wide(caller) - 4):
        await caller.say(f"  \x030{line}")
    inside = warez.contents(path, 12) if path.suffix.lower() == ".zip" else []
    if inside:
        await caller.say(f"  \x030contains: {', '.join(inside)}")
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
    key = (await menu_choice(caller, f"  {A.HY}Take it{A.HB}:{A.RESET} ",
                             2)).upper()
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
        if ok:
            board.bump_download(path)
            if caller.user is not None:
                caller.user.downloads += 1
                board.users.save()
            await caller.say("  \x032Transfer complete.\x030")
        else:
            await caller.say("  \x036Transfer failed or was abandoned."
                             "\x030")
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


def upload_target(caller: Caller, section: dict) -> Path:
    """Where an upload lands. A public directory takes it as it is; anything
    else files it under the caller's own name."""
    run = Path(caller.board.run_dir)
    if section.get("public"):
        return Path(section["roots"][0])
    return run / "bbs" / "files" / _safe_name(caller.handle)


async def _upload(caller: Caller, section: dict | None = None) -> None:
    """Take a file from a caller into a directory that accepts uploads."""
    board = caller.board
    user = caller.user
    if section is None:
        sections = [d for d in file_sections(board) if d.get("upload")]
        rows = [W.render(f"  \x032{d['key']}\x030) \x039{d['name']}")
                for d in sections]
        await caller.art("\n".join(A.box("UPLOAD TO", rows,
                                          width=wide(caller), frame=A.HY)))
        which = (await caller.ask(W.render("\x035Directory: \x030"),
                                  limit=4)).strip()
        section = next((d for d in sections if d["key"] == which), None)
        if section is None:
            return
    if not section.get("upload"):
        await caller.say(f"\x036{section['name']} does not take uploads."
                         f"\x030")
        await caller.pause()
        return
    if user is not None and not user.rules().can_upload:
        await caller.say(f"\x036Uploading needs a higher security level "
                         f"than {user.sl}. Leave feedback and ask.\x030")
        await caller.pause()
        return

    name = (await caller.ask(W.render("\x035Filename: \x030"),
                             limit=48)).strip()
    if not name:
        return
    safe = _safe_name(Path(name).name)
    suffix = Path(name).suffix.lower()
    # Nothing that runs itself on the sysop's machine, and nothing that
    # could be mistaken for the model. A file area is for files.
    if suffix in (".pt", ".exe", ".dll", ".so", ".bat", ".cmd", ".ps1",
                  ".scr", ".msi", ".com"):
        await caller.say("\x036Not that kind of file.\x030")
        await caller.pause()
        return
    if not Path(safe).suffix:
        safe = f"{safe}.dat"

    description = (await caller.ask(W.render(
        "\x035One line about it: \x030"), limit=70)).strip()
    data = await xmodem_receive(caller)
    if not data:
        await caller.say("\x036Nothing arrived.\x030")
        await caller.pause()
        return

    area = upload_target(caller, section)
    area.mkdir(parents=True, exist_ok=True)
    target = area / safe
    n = 1
    while target.exists():
        n += 1
        target = area / f"{Path(safe).stem}_{n}{Path(safe).suffix}"
    target.write_bytes(data)
    if description:
        board.describe_file(target, description, caller.handle)
    if user is not None:
        user.uploads += 1
        board.users.save()
    await caller.say(f"\x032{len(data):,} bytes received as "
                     f"{target.name}\x030 in {section['name']}.")
    board.page_all(W.render(f"\x032*** {caller.handle} uploaded "
                            f"{target.name} to {section['name']} ***\x030"),
                   skip=caller.node)
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
        await caller.screen()
        await header(caller)
        await crumbs(caller)
        await caller.line("")
        await caller.art("\n".join(A.box("A N S I   G A L L E R Y", [
            f"  {A.HY}[S]{A.RESET} {A.HW}show me one{A.RESET}"
            f"{A.GREY}      a shape it was trained on; can it name it?"
            f"{A.RESET}",
            f"  {A.HY}[F]{A.RESET} {A.HW}from the files{A.RESET}"
            f"{A.GREY}   pick an image out of the file area{A.RESET}",
            f"  {A.HY}[U]{A.RESET} {A.HW}upload one{A.RESET}"
            f"{A.GREY}       XMODEM it up and it will look at it{A.RESET}",
            A.entry("Q", "Go back"),
            A.entry("G", "Exit and log off"),
        ], width=min(caller.columns, 76), frame=A.HM)))
        key = (await menu_choice(caller, f"  {A.HY}Gallery{A.HB}:{A.RESET} ",
                                 2)).upper()
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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
    await caller.screen()
    await header(caller)
    await crumbs(caller)
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


# ---- WWIV's own screens -----------------------------------------------------

async def bulletins(caller: Caller) -> None:
    """B - the bulletins. Every board had them and nobody read them."""
    board = caller.board
    while True:
        items = board.bulletins()
        await caller.screen()
        await header(caller)
        await crumbs(caller)
        await caller.line("")
        rows = [W.render(f"  \x032{b['key']}\x030) \x039{b['title']}")
                for b in items] or [W.render("  \x030None.")]
        rows.append("")
        rows.append(A.entry("Q", f"Go back to {caller.whence()}"))
        rows.append(A.entry("G", "Exit and log off"))
        await caller.art("\n".join(A.box("B U L L E T I N S", rows,
                                          width=wide(caller), frame=A.HB)))
        choice = await menu_choice(
            caller, W.render("\x035Bulletin\x030: "), 4)
        if not choice or choice.upper() == "Q":
            return
        found = next((b for b in items if b["key"] == choice), None)
        if found is None:
            continue
        await caller.cls()
        await caller.say(f"\x031{found['title']}\x030")
        await caller.line("")
        for line in A.wrap(found["body"], wide(caller) - 4):
            await caller.say(f"  \x030{line}")
        await caller.line("")
        await caller.pause()


async def auto_message(caller: Caller) -> None:
    """A - the auto-message: one line, shown to everyone who logs on next."""
    board = caller.board
    await caller.screen()
    await header(caller)
    await crumbs(caller)
    await caller.line("")
    current = board.auto_message()
    if current:
        await caller.art("\n".join(A.box(
            "AUTO-MESSAGE",
            [W.render(f"\x032{current['text']}"),
             W.render(f"\x030          - {current['who']}, "
                      f"{current['when']}")],
            width=wide(caller), frame=A.HB)))
    else:
        await caller.say("  \x030Nobody has left one.")
    await caller.line("")
    text = (await caller.ask(W.render(
        "\x035Leave a new one \x030(blank to keep it)\x035: \x030"),
        limit=200)).strip()
    if text:
        board.set_auto_message(caller.user.name if caller.user else "?", text)
        board.page_all(W.render(f"\x032*** New auto-message from "
                                f"{caller.handle} ***\x030"),
                       skip=caller.node)
        await caller.say("  \x032Left.\x030")
        await caller.pause()


async def voting_booth(caller: Caller) -> None:
    """V - the voting booth. WWIV had one and sysops actually used it."""
    board = caller.board
    user = caller.user
    while True:
        polls = board.polls()
        await caller.screen()
        await header(caller)
        await crumbs(caller)
        await caller.line("")
        rows = []
        for poll in polls:
            mine = poll["votes"].get(str(user.number)) if user else None
            mark = "\x032*" if mine is not None else " "
            rows.append(W.render(f"  {mark}\x032{poll['key']}\x030) "
                                 f"\x039{poll['question']}"))
        rows.append("")
        rows.append(W.render("  \x030* = you have voted.    \x032Q\x030) "
                             "back"))
        await caller.art("\n".join(A.box("V O T I N G   B O O T H", rows,
                                          width=wide(caller), frame=A.HB)))
        choice = await menu_choice(
            caller, W.render("\x035Question\x030: "), 4)
        if not choice or choice.upper() == "Q":
            return
        poll = next((p for p in polls if p["key"] == choice), None)
        if poll is None:
            continue
        await _one_poll(caller, polls, poll)


async def _one_poll(caller: Caller, polls: list, poll: dict) -> None:
    board, user = caller.board, caller.user
    await caller.cls()
    await caller.say(f"\x031{poll['question']}\x030")
    await caller.line("")
    total = len(poll["votes"]) or 1
    for i, option in enumerate(poll["options"], start=1):
        count = sum(1 for v in poll["votes"].values() if v == i)
        share = count / total
        bar = "█" * int(share * 30)
        await caller.say(f"  \x032{i}\x030) \x039{A.pad(option, 40)}"
                         f"\x031{A.pad(bar, 30)}\x030{count}")
    await caller.line("")
    mine = poll["votes"].get(str(user.number)) if user else None
    if mine:
        await caller.say(f"  \x030You voted for "
                         f"{poll['options'][mine - 1]}.")
    answer = (await caller.ask(W.render(
        "\x035Your vote \x030(blank to leave it)\x035: \x030"),
        limit=4)).strip()
    if answer.isdigit() and 1 <= int(answer) <= len(poll["options"]) and user:
        poll["votes"][str(user.number)] = int(answer)
        board.write("polls.json", polls)
        await caller.say("  \x032Counted.\x030")
    await caller.pause()


async def feedback(caller: Caller) -> None:
    """F - feedback to the sysop. Here the sysop reads it and answers."""
    board = caller.board
    await caller.screen()
    await header(caller)
    await crumbs(caller)
    await caller.line("")
    await caller.say("\x031FEEDBACK TO THE SYSOP\x030")
    await caller.say("\x030The sysop is the model. It will answer, and any "
                     "human sysop on a node will see this too.")
    await caller.line("")
    body = await _multiline(caller)
    if not body:
        return
    board.post(caller.handle, "Feedback", body)
    board.send_mail(caller.handle, 1, "Feedback", body)
    for node, other in list(board.callers.items()):
        if other.sysop and node != caller.node:
            other.tell(W.render(f"\x036*** FEEDBACK from {caller.handle} "
                                f"***\x030 {body[:200]}"))
    source, text = await board.answer(body, caller)
    await caller.line("")
    await caller.say(f"\x031<{SYSOP}>\x030 {doors._label(source)}")
    for line in A.wrap(text, wide(caller) - 6):
        await caller.say(f"    \x039{line}")
    await caller.pause()


async def email(caller: Caller) -> None:
    """E - mail between users, and your own inbox."""
    board = caller.board
    user = caller.user
    while True:
        box = [m for m in board.mail()
               if user is not None and m["to"] == user.number]
        await caller.screen()
        await header(caller)
        await crumbs(caller)
        await caller.line("")
        rows = []
        for i, item in enumerate(box[-15:], start=1):
            flag = " " if item.get("read") else "\x032*"
            rows.append(W.render(f"  {flag}\x032{i:>2}\x030) "
                                 f"\x039{A.pad(item['subject'], 34)}"
                                 f"\x030{A.pad(item['from'], 16)}"
                                 f"{item['when']}"))
        if not rows:
            rows = [W.render("  \x030No mail.")]
        rows.append("")
        rows.append(W.render("  \x032S\x030) send    \x032number\x030) "
                             "read    \x032Q\x030) back"))
        await caller.art("\n".join(A.box("E - M A I L", rows,
                                          width=wide(caller), frame=A.HG)))
        choice = await menu_choice(caller, W.render("\x035Mail\x030: "), 4)
        if not choice or choice.upper() == "Q":
            return
        if choice.upper() == "S":
            await _send_mail(caller)
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(box[-15:]):
            item = box[-15:][int(choice) - 1]
            await caller.cls()
            await caller.say(f"\x031{item['subject']}\x030   "
                             f"\x030from {item['from']}, {item['when']}")
            await caller.line("")
            for line in A.wrap(item["body"], wide(caller) - 4):
                await caller.say(f"  \x039{line}")
            item["read"] = True
            board.write("email.json", board.mail()[:0] + _mark(board, item))
            await caller.line("")
            await caller.pause()


def _mark(board: Board, item: dict) -> list:
    box = board.mail()
    for entry in box:
        if (entry["when"] == item["when"] and entry["from"] == item["from"]
                and entry["subject"] == item["subject"]):
            entry["read"] = True
    return box


async def _send_mail(caller: Caller) -> None:
    board = caller.board
    who = (await caller.ask(W.render("\x035To \x030(user number or name)"
                                     "\x035: \x030"), limit=32)).strip()
    target = board.users.find(who)
    if target is None:
        await caller.say("\x036No such user.\x030")
        await caller.pause()
        return
    subject = (await caller.ask(W.render("\x035Subject: \x030"),
                                limit=60)).strip()
    if not subject:
        return
    await caller.say("\x030Body, blank line to end:")
    body = await _multiline(caller)
    if not body:
        return
    board.send_mail(caller.handle, target.number, subject, body)
    for other in board.callers.values():
        if other.user is not None and other.user.number == target.number:
            other.tell(W.render(f"\x032*** Mail from {caller.handle}: "
                                f"{subject} ***\x030"))
    await caller.say(f"\x032Sent to {target.name} #{target.number}.\x030")
    await caller.pause()


async def your_info(caller: Caller) -> None:
    """Y - your own user record, as WWIV showed it."""
    user = caller.user
    if user is None:
        return
    rules = user.rules()
    await caller.screen()
    await header(caller)
    await crumbs(caller)
    await caller.line("")
    await caller.art("\n".join(A.box(f"USER #{user.number}", [
        W.render(f"  \x030Handle          \x039{user.name}"),
        W.render(f"  \x030Real name       \x039{user.real_name or '-'}"),
        W.render(f"  \x030Security        \x032SL {user.sl}\x030 / "
                 f"\x032DSL {user.dsl}\x030"
                 f"   flags {user.ar or '-'} / {user.dar or '-'}"),
        W.render(f"  \x030Calls           \x039{user.calls}"),
        W.render(f"  \x030Posts           \x039{user.posts}"),
        W.render(f"  \x030Uploads         \x039{user.uploads}"
                 f"\x030   downloads \x039{user.downloads}"),
        W.render(f"  \x030Time, all calls \x039{user.minutes:.0f}\x030 min"),
        W.render(f"  \x030Today           \x039{user.today:.0f}\x030 of "
                 f"\x039{rules.minutes_per_day}\x030 min"),
        W.render(f"  \x030This call       \x039{caller.minutes():.1f}"
                 f"\x030 min, \x039{user.minutes_left() - caller.minutes():.0f}"
                 f"\x030 left"),
        W.render(f"  \x030First on        \x039{user.first_on}"),
        W.render(f"  \x030Expert mode     \x039"
                 f"{'on' if caller.expert else 'off'}"),
    ], width=wide(caller), frame=A.HC)))
    await caller.line("")
    if (await caller.ask(W.render("\x035Change your password? [y/N] \x030"),
                         limit=4)).strip().lower().startswith("y"):
        new = await caller.ask(W.render("\x035New password: \x030"),
                               limit=64, mask=True)
        if new:
            caller.board.set_password(user, new)
            await caller.say("\x032Changed.\x030")
        else:
            caller.board.clear_password(user)
            await caller.say("\x032Cleared.\x030")
    await caller.pause()


async def expert_toggle(caller: Caller) -> None:
    """X - WWIV's expert mode: stop drawing the menu, just take commands."""
    caller.expert = not caller.expert
    if caller.user is not None:
        caller.user.expert = caller.expert
        caller.board.users.save()
    await caller.say(f"\x032Expert mode "
                     f"{'on' if caller.expert else 'off'}.\x030 "
                     f"\x030? redraws the menu either way.")
    await asyncio.sleep(0.4)


async def show_menu(caller: Caller) -> None:
    """? - draw the menu even in expert mode."""
    was, caller.expert = caller.expert, False
    await draw_menu(caller)
    caller.expert = was
    await caller.pause()


async def new_file_scan(caller: Caller) -> None:
    """N - what has arrived in the file directories since you last called."""
    board = caller.board
    user = caller.user
    since = user.last_on if user else ""
    await caller.screen()
    await header(caller)
    await crumbs(caller)
    await caller.line("")
    rows = []
    for section in file_sections(board):
        if user is not None and not user.may_download(section.get("dar", ""),
                                                      section.get("dsl", 0)):
            continue
        fresh = [p for p in listing(section, 400) if warez.when(p) >= since]
        for path in fresh[:8]:
            rows.append(W.render(
                f"  \x030{A.pad(section['name'], 14)}"
                f"\x039{A.pad(path.name, 30)}"
                f"\x030{size_of(path):>9}  {warez.when(path)}"))
    if not rows:
        rows = [W.render("  \x030Nothing new since you were last on.")]
    await caller.art("\n".join(A.box(f"NEW FILES SINCE {since or 'ever'}",
                                      rows, width=wide(caller), frame=A.HY)))
    await caller.pause()


# ---- /A, /D and the sysop's // ----------------------------------------------

async def slash_command(caller: Caller, rest: str) -> None:
    """WWIV's single-slash commands: change sub, change directory."""
    board = caller.board
    verb, _, argument = rest.partition(" ")
    verb = verb.upper()

    if verb == "A":
        rows = [W.render(f"  \x032{sub.key}\x030) \x039{A.pad(sub.name, 16)}"
                         f"\x030{sub.blurb}") for sub in board.subs]
        await caller.art("\n".join(A.box("SUB-BOARDS", rows,
                                          width=wide(caller), frame=A.HG)))
        which = argument or await caller.ask(W.render("\x035Sub: \x030"),
                                             limit=4)
        chosen = next((s for s in board.subs if s.key == which.strip()), None)
        if chosen is not None:
            caller.sub = chosen.key
            await caller.say(f"\x032Now in {chosen.name}.\x030")
        await caller.pause()
        return

    if verb == "D":
        sections = file_sections(board)
        rows = [W.render(f"  \x032{d['key']}\x030) \x039{A.pad(d['name'], 16)}"
                         f"\x030{len(listing(d, 400))} files")
                for d in sections]
        await caller.art("\n".join(A.box("FILE DIRECTORIES", rows,
                                          width=wide(caller), frame=A.HY)))
        which = argument or await caller.ask(W.render("\x035Directory: "
                                                      "\x030"), limit=4)
        if any(d["key"] == which.strip() for d in sections):
            caller.dir = which.strip()
            await caller.say(f"\x032Now in directory {caller.dir}.\x030")
        await caller.pause()
        return

    await caller.say("\x030/A changes sub, /D changes file directory.\x030")
    await caller.pause()


SYSOP_HELP = [
    ("//?", "this list"),
    ("//UEDIT [user]", "the user editor: SL, DSL, flags, passwords"),
    ("//BOARDEDIT", "the sub-boards: add, rename, set the levels"),
    ("//DIREDIT", "the file directories, as the board sees them"),
    ("//CONFIG", "board name, new-user password, node limits"),
    ("//COLORS", "the ten heart-code colours, as WWIV let you"),
    ("//WHO", "every node, with what it is doing"),
    ("//SPY <node>", "watch a node's screen until you press a key"),
    ("//BROADCAST <text>", "to every node at once"),
    ("//STATS", "calls, posts, files, model, engine"),
    ("//AUTOVAL <user>", "raise a new user to SL 50 / DSL 50"),
    ("//PURGE <n>", "delete user #n"),
    ("//MOTD", "write the message of the day; blank line ends it"),
    ("//WAREZ", "rebuild the release archives from source"),
    ("//SHUTDOWN", "close the board to new callers"),
]


async def sysop_command(caller: Caller, rest: str) -> bool:
    """WWIV's // commands. True means the board is coming down.

    Everything WWIV let a sysop reach from the keyboard is here, with one
    deliberate exception: there is no //DOS. A shell on the far end of a
    plaintext telnet session is not a sysop feature, it is the hole, and
    the sysop already has a shell on the machine the board is running on.
    """
    board = caller.board
    verb, _, argument = rest.partition(" ")
    verb, argument = verb.upper(), argument.strip()

    if verb in ("", "?", "HELP"):
        rows = [W.render(f"  \x032{A.pad(name, 22)}\x030{what}")
                for name, what in SYSOP_HELP]
        await caller.art("\n".join(A.box("SYSOP COMMANDS", rows,
                                          width=wide(caller), frame=A.HR)))
        await caller.pause()
        return False

    if verb == "UEDIT":
        await user_editor(caller, argument)
        return False
    if verb == "BOARDEDIT":
        await board_editor(caller)
        return False
    if verb == "DIREDIT":
        await dir_editor(caller)
        return False
    if verb == "CONFIG":
        await config_editor(caller)
        return False
    if verb in ("COLORS", "COLOURS"):
        await colour_editor(caller)
        return False
    if verb == "WHO":
        await who_is_online(caller)
        return False
    if verb == "SPY":
        await spy(caller, argument)
        return False
    if verb == "BROADCAST":
        if argument:
            board.page_all(W.render(f"\x036*** SYSOP: {argument} ***\x030"))
            await caller.say("\x032Sent.\x030")
        await caller.pause()
        return False
    if verb == "STATS":
        await system_info(caller)
        return False
    if verb == "AUTOVAL":
        target = board.users.find(argument)
        if target is None:
            await caller.say("\x036No such user.\x030")
        else:
            target.sl, target.dsl = 50, 50
            target.ar = target.dar = "A"
            board.users.save()
            await caller.say(f"\x032{target.name} is validated: SL 50, "
                             f"DSL 50.\x030")
        await caller.pause()
        return False
    if verb == "PURGE":
        target = board.users.find(argument)
        if target is None or target.number == 1:
            await caller.say("\x036No.\x030")
        else:
            board.users.records.pop(target.number, None)
            board.clear_password(target)
            board.users.save()
            await caller.say(f"\x032#{target.number} is gone.\x030")
        await caller.pause()
        return False
    if verb == "MOTD":
        current = board.motd()
        if current:
            await caller.say("\x030The message of the day is now:")
            for line in current["text"].split("\n"):
                await caller.say(f"  \x032{line}")
            await caller.line("")
        await caller.say("\x030Type the new one. A blank line ends it; end "
                         "it immediately to clear it.")
        text = await _multiline(caller, limit=2000)
        board.set_motd(text, caller.handle)
        if text:
            board.page_all(W.render("\x032*** The sysop has posted a new "
                                    "message of the day. ***\x030"),
                           skip=caller.node)
            await caller.say("\x032Posted. Every caller sees it as they log "
                             "on.\x030")
        else:
            await caller.say("\x032Cleared.\x030")
        await caller.pause()
        return False

    if verb == "WAREZ":
        await caller.say("\x030Rebuilding...")
        built = await asyncio.to_thread(warez.build, board.run_dir, True)
        for path in built:
            await caller.say(f"  \x039{path.name}\x030 "
                             f"{size_of(path)}")
        await caller.pause()
        return False
    if verb == "SHUTDOWN":
        answer = await caller.ask(W.render("\x036Close the board? [y/N] "
                                           "\x030"), limit=4)
        if answer.strip().lower().startswith("y"):
            board.page_all(W.render("\x036*** The sysop is closing the "
                                    "board. ***\x030"))
            board.closing = True
            return True
        return False

    await caller.say(f"\x036// {verb} is not a command. //? lists them."
                     f"\x030")
    await caller.pause()
    return False


async def user_editor(caller: Caller, who: str) -> None:
    """//UEDIT - every field WWIV let a sysop change, and it saves."""
    board = caller.board
    while True:
        if not who:
            rows = [W.render(
                f"  \x032#{u.number:<4}\x039{A.pad(u.name, 18)}"
                f"\x030SL {u.sl:<4}DSL {u.dsl:<4}"
                f"{A.pad(u.ar or '-', 6)}{u.calls:>4} calls")
                for u in board.users.sorted()[:40]]
            await caller.art("\n".join(A.box("USERS", rows or [
                W.render("  \x030Nobody yet.")],
                width=wide(caller), frame=A.HR)))
            who = (await caller.ask(W.render(
                "\x035User number or name \x030(blank to leave)\x035: "
                "\x030"), limit=32)).strip()
            if not who:
                return
        target = board.users.find(who)
        who = ""
        if target is None:
            await caller.say("\x036No such user.\x030")
            continue

        await caller.cls()
        await caller.art("\n".join(A.box(f"USER #{target.number}", [
            W.render(f"  \x032A\x030) Handle       \x039{target.name}"),
            W.render(f"  \x032B\x030) Real name    "
                     f"\x039{target.real_name or '-'}"),
            W.render(f"  \x032C\x030) SL           \x039{target.sl}"),
            W.render(f"  \x032D\x030) DSL          \x039{target.dsl}"),
            W.render(f"  \x032E\x030) AR flags     "
                     f"\x039{target.ar or '-'}"),
            W.render(f"  \x032F\x030) DAR flags    "
                     f"\x039{target.dar or '-'}"),
            W.render(f"  \x032G\x030) Note         "
                     f"\x039{target.note or '-'}"),
            W.render(f"  \x032H\x030) Clear password"),
            W.render(f"  \x030   calls {target.calls}, posts {target.posts}, "
                     f"{target.minutes:.0f} minutes"),
            "",
            A.entry("Q", "Go back"),
            A.entry("G", "Exit and log off"),
        ], width=wide(caller), frame=A.HR)))
        field = (await menu_choice(caller, W.render("\x035Field\x030: "),
                                   4)).upper()
        if not field or field == "Q":
            return
        if field == "H":
            board.clear_password(target)
            await caller.say("\x032Password cleared.\x030")
            await caller.pause()
            continue
        value = (await caller.ask(W.render("\x035New value: \x030"),
                                  limit=60)).strip()
        try:
            if field == "A" and value:
                target.name = value[:30]
            elif field == "B":
                target.real_name = value[:40]
            elif field == "C":
                target.sl = max(0, min(255, int(value)))
            elif field == "D":
                target.dsl = max(0, min(255, int(value)))
            elif field == "E":
                target.ar = "".join(c for c in value.upper()
                                    if c in "ABCDEFGHIJKLMNOP")
            elif field == "F":
                target.dar = "".join(c for c in value.upper()
                                     if c in "ABCDEFGHIJKLMNOP")
            elif field == "G":
                target.note = value[:80]
        except ValueError:
            await caller.say("\x036That is not a number.\x030")
            await caller.pause()
            continue
        board.users.save()
        # A user who is on right now sees the change immediately.
        for other in board.callers.values():
            if other.user is not None and other.user.number == target.number:
                other.user = target
                other.sysop = target.sysop
        who = str(target.number)


async def board_editor(caller: Caller) -> None:
    """//BOARDEDIT - the sub-boards, added and edited and saved."""
    board = caller.board
    while True:
        rows = [W.render(
            f"  \x032{s.key}\x030) \x039{A.pad(s.name, 16)}"
            f"\x030read SL {s.read_sl:<4}post SL {s.post_sl:<4}"
            f"{s.blurb[:24]}") for s in board.subs]
        rows.append("")
        rows.append(W.render("  \x032N\x030) new sub   \x032number\x030) "
                             "edit   \x032Q\x030) back"))
        await caller.cls()
        await caller.art("\n".join(A.box("SUB-BOARDS", rows,
                                          width=wide(caller), frame=A.HR)))
        choice = (await menu_choice(caller, W.render("\x035Sub\x030: "),
                                    4)).upper()
        if not choice or choice == "Q":
            board.save_subs()
            return
        if choice == "N":
            name = (await caller.ask(W.render("\x035Name: \x030"),
                                     limit=24)).strip()
            if name:
                key = str(max((int(s.key) for s in board.subs
                               if s.key.isdigit()), default=0) + 1)
                blurb = (await caller.ask(W.render("\x035Description: "
                                                   "\x030"), limit=60)).strip()
                board.subs.append(W.Sub(key, name[:24], blurb[:60]))
                board.save_subs()
            continue
        target = next((s for s in board.subs if s.key == choice), None)
        if target is None:
            continue
        await caller.say(f"\x030Editing \x039{target.name}\x030. Blank "
                         f"keeps the current value.")
        name = (await caller.ask(W.render(f"\x035Name [{target.name}]: "
                                          f"\x030"), limit=24)).strip()
        blurb = (await caller.ask(W.render(f"\x035Description "
                                           f"[{target.blurb[:20]}]: \x030"),
                                  limit=60)).strip()
        read_sl = (await caller.ask(W.render(f"\x035Read SL "
                                             f"[{target.read_sl}]: \x030"),
                                    limit=4)).strip()
        post_sl = (await caller.ask(W.render(f"\x035Post SL "
                                             f"[{target.post_sl}]: \x030"),
                                    limit=4)).strip()
        index = board.subs.index(target)
        board.subs[index] = W.Sub(
            target.key, name or target.name, blurb or target.blurb,
            int(read_sl) if read_sl.isdigit() else target.read_sl,
            int(post_sl) if post_sl.isdigit() else target.post_sl,
            target.ar, target.anonymous)
        board.save_subs()


async def dir_editor(caller: Caller) -> None:
    """//DIREDIT - the file directories, their levels and their roots."""
    board = caller.board
    sections = file_sections(board)
    overrides = board.read("dirs.json", {})
    while True:
        rows = []
        for section in sections:
            over = overrides.get(section["key"], {})
            dsl = over.get("dsl", section.get("dsl", 0))
            rows.append(W.render(
                f"  \x032{section['key']}\x030) "
                f"\x039{A.pad(section['name'], 14)}"
                f"\x030DSL {dsl:<5}"
                f"{'upload  ' if section.get('upload') else '        '}"
                f"{len(listing(section, 400))} files"))
        for section in sections:
            for root in section["roots"]:
                pass
        rows.append("")
        rows.append(W.render("  \x032number\x030) set its DSL   "
                             "\x032R\x030) show roots   \x032Q\x030) back"))
        await caller.cls()
        await caller.art("\n".join(A.box("FILE DIRECTORIES", rows,
                                          width=wide(caller), frame=A.HR)))
        choice = (await menu_choice(caller,
                                    W.render("\x035Directory\x030: "),
                                    4)).upper()
        if not choice or choice == "Q":
            return
        if choice == "R":
            rows = []
            for section in sections:
                rows.append(W.render(f"  \x039{section['name']}"))
                for root in section["roots"]:
                    rows.append(W.render(f"    \x030{root}"))
            await caller.cls()
            await caller.art("\n".join(A.box("ROOTS", rows,
                                              width=wide(caller), frame=A.HR)))
            await caller.say("\x030Roots are fixed in the source: a "
                             "directory a caller could point anywhere is an "
                             "arbitrary-file-read primitive with a menu on "
                             "it.")
            await caller.pause()
            continue
        section = next((d for d in sections if d["key"] == choice), None)
        if section is None:
            continue
        value = (await caller.ask(W.render(
            f"\x035DSL for {section['name']} "
            f"[{section.get('dsl', 0)}]: \x030"), limit=4)).strip()
        if value.isdigit():
            overrides.setdefault(section["key"], {})["dsl"] = int(value)
            board.write("dirs.json", overrides)
            await caller.say("\x032Saved.\x030")
            await caller.pause()


async def config_editor(caller: Caller) -> None:
    """//CONFIG - the board's own settings, saved and applied at once."""
    board = caller.board
    settings = board.read("config.json", {})
    while True:
        await caller.cls()
        await caller.art("\n".join(A.box("BOARD CONFIGURATION", [
            W.render(f"  \x032A\x030) Board name        \x039{board.name}"),
            W.render(f"  \x032B\x030) New-user password \x039"
                     f"{board.new_user_password or '(none)'}"),
            W.render(f"  \x032C\x030) Nodes             "
                     f"\x039{board.max_callers}"),
            W.render(f"  \x032D\x030) Per address       "
                     f"\x039{board.max_per_address}"),
            W.render(f"  \x032E\x030) Idle minutes      "
                     f"\x039{IDLE_SECONDS // 60}"),
            W.render(f"  \x032F\x030) Local caller is sysop  \x039"
                     f"{'yes' if board.trust_local else 'no'}"),
            W.render(f"  \x032G\x030) Batch size        "
                     f"\x039{board.max_batch}"),
            W.render(f"  \x032H\x030) Tokens per reply  "
                     f"\x039{board.max_tokens}"),
            W.render(f"  \x032I\x030) New caller SL     "
                     f"\x039{board.new_user_sl}\x030 "
                     f"(10 = unvalidated, 50 = can post and upload)"),
            W.render(f"  \x032J\x030) New caller DSL    "
                     f"\x039{board.new_user_dsl}"),
            W.render(f"  \x032K\x030) New caller flags  "
                     f"\x039{board.new_user_flags or '(none)'}"),
            "",
            A.entry("Q", "Go back"),
            A.entry("G", "Exit and log off"),
        ], width=wide(caller), frame=A.HR)))
        field = (await menu_choice(caller, W.render("\x035Field\x030: "),
                                   4)).upper()
        if not field or field == "Q":
            return
        value = (await caller.ask(W.render("\x035New value: \x030"),
                                  limit=60)).strip()
        if field == "A" and value:
            board.name = settings["name"] = value[:24]
        elif field == "B":
            board.new_user_password = settings["new_user_password"] = \
                value or None
        elif field == "C" and value.isdigit():
            board.max_callers = settings["max_callers"] = int(value)
        elif field == "D" and value.isdigit():
            board.max_per_address = settings["max_per_address"] = int(value)
        elif field == "F":
            board.trust_local = settings["trust_local"] = \
                value.lower().startswith("y")
        elif field == "G" and value.isdigit():
            board.max_batch = settings["max_batch"] = max(1, int(value))
            if board._engine is not None:
                board._engine.max_batch = board.max_batch
        elif field == "H" and value.isdigit():
            board.max_tokens = settings["max_tokens"] = max(8, int(value))
        elif field == "I" and value.isdigit():
            board.new_user_sl = settings["new_user_sl"] = \
                max(0, min(254, int(value)))
        elif field == "J" and value.isdigit():
            board.new_user_dsl = settings["new_user_dsl"] = \
                max(0, min(254, int(value)))
        elif field == "K":
            board.new_user_flags = settings["new_user_flags"] = "".join(
                c for c in value.upper() if c in "ABCDEFGHIJKLMNOP")
        board.write("config.json", settings)


async def colour_editor(caller: Caller) -> None:
    """//COLORS - the ten heart-code colours, exactly as WWIV let you.

    This is the one piece of WWIV nobody else copied: the colours are data,
    the screens name them by number, and changing one here changes every
    screen on the board at once.
    """
    board = caller.board
    while True:
        await caller.cls()
        rows = []
        for i, byte in enumerate(board.colours):
            sample = W.attribute(byte)
            rows.append(f"  {A.RESET}{i}) {sample}"
                        f"the quick brown fox jumps{A.RESET}  "
                        f"{A.GREY}0x{byte:02X}{A.RESET}")
        rows.append("")
        rows.append(W.render("  \x030number to change, \x032R\x030) reset, "
                             "\x032Q\x030) back"))
        await caller.art("\n".join(A.box("COLOURS", rows,
                                          width=wide(caller), frame=A.HR)))
        await caller.say("\x030Low nibble is the foreground, high nibble "
                         "the background, bit 7 blinks - an IBM attribute "
                         "byte, as it always was.")
        choice = (await menu_choice(caller, W.render("\x035Colour\x030: "),
                                    4)).upper()
        if not choice or choice == "Q":
            return
        if choice == "R":
            board.set_colours(list(W.DEFAULT_COLOURS))
            continue
        if not choice.isdigit() or not 0 <= int(choice) <= 9:
            continue
        value = (await caller.ask(W.render(
            "\x035Attribute byte, hex or decimal: \x030"), limit=8)).strip()
        try:
            byte = int(value, 16) if value.lower().startswith("0x") \
                else int(value, 0)
        except ValueError:
            continue
        table = list(board.colours)
        table[int(choice)] = byte & 0xFF
        board.set_colours(table)


async def spy(caller: Caller, argument: str) -> None:
    """//SPY - watch another node, as WWIV's sysop could.

    The node being watched is told. A board that let the sysop read over a
    caller's shoulder silently would be a different kind of program.
    """
    board = caller.board
    if not argument.isdigit():
        await caller.say("\x030//SPY <node>. //WHO lists them.\x030")
        await caller.pause()
        return
    target = board.callers.get(int(argument))
    if target is None or target is caller:
        await caller.say("\x036No such node.\x030")
        await caller.pause()
        return

    target.watchers.add(caller.node)
    target.tell(W.render(f"\x036*** The sysop is watching this node. ***"
                         f"\x030"))
    await caller.say(f"\x032Watching node {target.node} "
                     f"({target.handle}). Any key stops.\x030")
    try:
        await caller.key()
    finally:
        target.watchers.discard(caller.node)
        target.tell(W.render("\x030*** The sysop has stopped watching. ***"
                             "\x030"))


HANDLERS = {
    "1": option_make, "2": option_do, "3": option_teach,
    "4": option_patch, "5": option_gui,
    "R": message_base, "P": post_message, "Q": quick_scan,
    "E": email, "F": feedback, "A": auto_message, "B": bulletins,
    "V": voting_booth,
    "T": file_area, "N": new_file_scan, "C": chat_with_motherbrain,
    "M": teleconference, "D": door_menu, "I": system_info, "Y": your_info,
    "W": who_is_online, "L": last_callers, "X": expert_toggle,
    "?": show_menu,
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
        # SGR mouse reporting: 1000 turns clicks on, 1006 asks for them in
        # the extended form that works past column 95. A terminal that does
        # not do mice ignores both and nothing is lost.
        writer.write(b"\x1b[?1000h\x1b[?1006h")
        await writer.drain()
        caller.mouse = True
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
        if caller.mouse:
            with contextlib.suppress(Exception):
                await caller.send("\x1b[?1006l\x1b[?1000l")
        board.leave(caller)
        board.callers.pop(caller.node, None)
        if caller.handle:                     # someone who never logged in
            board.page_all(f"{A.GREY}*** {caller.handle} hung up ***"
                           f"{A.RESET}")        # is not news
        await caller.close()


async def run_board(board: Board, host: str = "127.0.0.1",
                    port: int = DEFAULT_PORT, web_port: int = 0) -> None:
    """Bind, then answer the telephone until something stops us."""
    async def handle(reader, writer):
        await session(reader, writer, board)

    keeper = asyncio.create_task(board.stats_keeper())
    reader = asyncio.create_task(board.read_the_corpus())
    web = None
    if web_port:
        from motherbrain import webterm

        web = asyncio.create_task(
            webterm.serve("127.0.0.1", port, host, web_port))
        print(f"  a browser can reach it on http://{host}:{web_port}/  "
              f"(tap the menus)")
    server = await asyncio.start_server(handle, host, port)
    where = ", ".join(str(s.getsockname()[:2]) for s in server.sockets or [])
    print(f"  answering telnet on {where}")
    try:
        async with server:
            await server.serve_forever()
    finally:
        keeper.cancel()
        reader.cancel()
        if web is not None:
            web.cancel()


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
          max_tokens: int = 120, steps: int = 100, grow: int = 1,
          web_port: int = 0, new_user_sl: int | None = None) -> int:
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
    board.load_settings()
    if new_user_sl is not None:
        board.new_user_sl = board.new_user_dsl = new_user_sl
        board.new_user_flags = "A"
        print(f"  a caller who has never called before starts at SL "
              f"{new_user_sl}: they can read, post, download and upload.")
    print("  building the file area ...")
    for path in warez.build(run_dir):
        print(f"    {path.name}")

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
        asyncio.run(run_board(board, host, port, web_port))
    except KeyboardInterrupt:
        print("\n  NO CARRIER")
    except OSError as exc:
        print(port_advice(port, exc))
        return 1
    return 0
