"""The doors, played at your own keyboard instead of over a modem.

A door game was written against a serial port, and the board handed it
one. Nothing about HAMURABI needs a telephone line - what it needs is a
screen that takes ANSI and a keyboard that gives up one keypress at a
time. This provides both, locally, so the same five doors run as an
ordinary program: `mb doors`, or the standalone release from the board's
file area.

It is deliberately the same interface `Caller` presents, method for
method, so the doors do not know or care which one they are talking to.
That is the whole reason the games are worth shipping separately: they
were never coupled to the board.
"""

from __future__ import annotations

import asyncio
import os
import sys

from motherbrain import ansi as A


class LocalCaller:
    """A caller who is sitting at this keyboard.

    The same surface as `bbs.Caller` - send, line, art, cls, key, ask,
    pause, columns, rows - over stdin and stdout.
    """

    def __init__(self, board=None) -> None:
        self.board = board
        self.handle = os.environ.get("USER") or os.environ.get(
            "USERNAME") or "player"
        self.node = 0
        self.sysop = True
        self.encoding = "utf-8"
        self.baud = 0
        size = _size()
        self.columns, self.rows = size

    # -- output ---------------------------------------------------------------

    async def send(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    async def line(self, text: str = "") -> None:
        await self.send(text + "\n")

    async def art(self, text: str) -> None:
        await self.send(text.replace("\r\n", "\n") + "\n")

    async def cls(self) -> None:
        await self.send(A.CLS)

    async def pause(self, text: str = "press any key") -> None:
        await self.send(f"{A.GREY}  [{A.HW}{text}{A.GREY}]{A.RESET}")
        await self.key()
        await self.send("\r" + A.CLEAR_LINE)

    def tell(self, text: str) -> None:
        """Nobody else is here, so there is nothing to deliver."""

    # -- input ----------------------------------------------------------------

    async def key(self) -> str:
        return await asyncio.to_thread(_read_key)

    async def ask(self, prompt: str, limit: int = 200, mask: bool = False,
                  default: str = "") -> str:
        await self.send(prompt)
        try:
            got = await asyncio.to_thread(sys.stdin.readline)
        except (EOFError, KeyboardInterrupt):
            return ""
        if not got:
            return ""
        return got.strip()[:limit] or default


def _size() -> tuple[int, int]:
    try:
        size = os.get_terminal_size()
        return max(40, size.columns), max(10, size.lines)
    except OSError:
        return 80, 24


def _read_key() -> str:
    """One keypress, without waiting for Enter. Arrows come back named."""
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):                 # the two-byte arrow keys
            return {"H": "UP", "P": "DOWN", "K": "LEFT",
                    "M": "RIGHT"}.get(msvcrt.getwch(), "")
        return ch

    try:
        import termios
        import tty
    except ImportError:                            # no tty at all
        return sys.stdin.read(1)

    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:                          # piped input, not a terminal
        return sys.stdin.read(1) or "\r"
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # An escape sequence, or a lone Escape. Read only what is already
            # waiting so a bare Escape does not block for the rest of one.
            import select

            if select.select([fd], [], [], 0.05)[0]:
                rest = sys.stdin.read(2)
                return {"[A": "UP", "[B": "DOWN", "[C": "RIGHT",
                        "[D": "LEFT", "OA": "UP", "OB": "DOWN",
                        "OC": "RIGHT", "OD": "LEFT"}.get(rest, "")
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


class OfflineBoard:
    """What a door asks the board for, when there is no board.

    The two doors that use the model - the Oracle and Turing - need one.
    Rather than hiding them, they are offered and say plainly that they
    need a model, because a menu that lists five doors and plays three is
    worse than one that explains itself.
    """

    def __init__(self, run_dir: str | None = None,
                 corpus_dir: str | None = None,
                 device: str = "auto") -> None:
        self.run_dir, self.corpus_dir, self.device = run_dir, corpus_dir, device
        self.model = self.tok = self.torch_device = None
        self.loaded = False

    def load(self) -> bool:
        if self.loaded or not self.run_dir:
            return self.loaded
        try:
            from motherbrain.cli import load_current

            self.model, self.tok, self.torch_device, _ = load_current(
                self.run_dir, self.device)
            self.loaded = True
        except Exception:                                  # noqa: BLE001
            self.loaded = False
        return self.loaded

    def stats(self) -> dict:
        if not self.loaded:
            return {}
        from motherbrain.stats import gather

        return gather(self.run_dir, self.corpus_dir, model=self.model,
                      device=self.torch_device)

    def queued(self) -> int:
        return 0

    async def generate(self, prompt: str, max_tokens: int = 120,
                       temperature: float = 0.8) -> str:
        if not self.load():
            return ""
        from motherbrain.inference import Request, generate_batch
        from motherbrain.tokenizer import EOS_ID

        request = Request(prompt, max_new_tokens=max_tokens,
                          temperature=temperature, repetition_penalty=1.15)
        await asyncio.to_thread(generate_batch, self.model, self.tok,
                                [request], self.torch_device, EOS_ID)
        return request.text

    async def answer(self, text: str, caller) -> tuple[str, str]:
        from motherbrain.logic import solve

        exact = solve(text)
        if exact is not None:
            return "exact", exact.render()
        if not self.load():
            return "generated", (
                "There is no model here - this is the doors on their own. "
                "Point them at one with:  mb doors --run runs/default")
        from motherbrain.chat import CONTINUATION_NOTE, consider, respond

        try:
            considered = consider(text, self.run_dir)
        except Exception:                                  # noqa: BLE001
            considered = None
        if considered is not None:
            return "known", considered[1]
        kind, said = respond(text, self.stats())
        if kind == "fact":
            return "self", said
        produced = await self.generate(text)
        return "generated", (produced.strip() or "(nothing)") + \
            f"\n\n{CONTINUATION_NOTE}"

    async def two_answers(self, text: str, caller):
        from motherbrain.chat import classify, respond

        if not self.load() or classify(text) is None:
            return None, ""
        _, honest = respond(text, self.stats())
        made_up = (await self.generate(text, max_tokens=60)).strip()
        return honest, made_up or "..."


async def _menu(caller: LocalCaller) -> None:
    from motherbrain import doors

    while True:
        await caller.cls()
        await caller.art(A.gradient("  M O T H E R B R A I N   D O O R S  ",
                                    (A.HC, A.C, A.HB, A.B)))
        rows = [f"  {A.HY}[{key}]{A.RESET}  {A.HW}{A.pad(name, 12)}"
                f"{A.GREY}{blurb}{A.RESET}"
                for key, name, blurb in doors.CATALOGUE]
        rows.append("")
        rows.append(f"  {A.HY}[Q]{A.RESET}  {A.HW}quit")
        await caller.art("\n".join(A.box("D O O R S", rows,
                                         width=min(caller.columns, 76),
                                         frame=A.HM)))
        if not getattr(caller.board, "loaded", False):
            await caller.line(f"  {A.GREY}THE ORACLE and TURING want a model; "
                              f"the other three do not.{A.RESET}")
        await caller.line("")
        key = (await caller.ask(f"  {A.HY}door: {A.RESET}", limit=2)).strip()
        if not key or key.upper() == "Q":
            await caller.line("")
            return
        if not await doors.play(caller, key):
            await caller.line(f"  {A.HR}no such door.{A.RESET}")
            await caller.pause()


def play(run_dir: str | None = None, corpus_dir: str | None = None,
         device: str = "auto") -> int:
    """Open the door menu at this keyboard. Returns an exit status."""
    board = OfflineBoard(run_dir, corpus_dir, device)
    caller = LocalCaller(board)
    try:
        asyncio.run(_menu(caller))
    except KeyboardInterrupt:
        print()
    finally:
        sys.stdout.write(A.RESET + A.SHOW_CURSOR)
        sys.stdout.flush()
    return 0
