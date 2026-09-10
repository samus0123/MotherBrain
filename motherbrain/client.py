"""A telnet client, so the board can be called from a machine without one.

Windows has shipped without a telnet client since Vista. macOS dropped
theirs in High Sierra. The board is a telnet board, and telling somebody to
go and install a client before they can see their own software is a poor
way to hand it to them - especially on a USB stick, where the whole promise
is that nothing has to be installed.

So this is the client. It negotiates the same options the board offers,
puts the local terminal into raw mode so single keypresses reach the far
end, and decodes CP437 or UTF-8 depending on what was agreed. It is about
two hundred lines because that is all a telnet client is.

It is not a general-purpose terminal. It does what a caller needs: type,
see, and hang up.
"""

from __future__ import annotations

import selectors
import socket
import sys

from motherbrain.telnet import (IAC, OPT_BINARY, OPT_ECHO, OPT_NAWS,
                                OPT_SGA, OPT_TTYPE, SB, SE, DO, DONT, WILL,
                                WONT, prefers_cp437)

# What the client takes on itself, and what it agrees to when offered.
_WE_WILL = (OPT_NAWS, OPT_TTYPE, OPT_SGA, OPT_BINARY)
_WE_DO = (OPT_ECHO, OPT_SGA, OPT_BINARY)

DEFAULT_TERMINAL = "xterm-256color"


class Session:
    """One call. Feed it socket bytes; it hands back what to print."""

    def __init__(self, columns: int = 80, rows: int = 24,
                 terminal: str = DEFAULT_TERMINAL) -> None:
        self.columns, self.rows, self.terminal = columns, rows, terminal
        self.encoding = "utf-8"
        self._state = "data"
        self._verb = 0
        self._sub = bytearray()
        self._will: dict[int, bool] = {}
        self._do: dict[int, bool] = {}
        self._partial = bytearray()

    def size(self) -> bytes:
        """A NAWS subnegotiation, sent whenever the window changes."""
        return bytes([IAC, SB, OPT_NAWS,
                      (self.columns >> 8) & 0xFF, self.columns & 0xFF,
                      (self.rows >> 8) & 0xFF, self.rows & 0xFF, IAC, SE])

    def feed(self, chunk: bytes) -> tuple[str, bytes]:
        """(text to print, bytes to send back)."""
        text = bytearray()
        reply = bytearray()

        for byte in chunk:
            if self._state == "data":
                if byte == IAC:
                    self._state = "iac"
                else:
                    text.append(byte)
            elif self._state == "iac":
                if byte == IAC:
                    text.append(IAC)
                    self._state = "data"
                elif byte in (DO, DONT, WILL, WONT):
                    self._verb, self._state = byte, "opt"
                elif byte == SB:
                    self._sub.clear()
                    self._state = "sub"
                else:
                    self._state = "data"
            elif self._state == "opt":
                reply += self._answer(self._verb, byte)
                self._state = "data"
            elif self._state == "sub":
                if byte == IAC:
                    self._state = "sub-iac"
                else:
                    self._sub.append(byte)
            elif self._state == "sub-iac":
                if byte == IAC:
                    self._sub.append(IAC)
                    self._state = "sub"
                elif byte == SE:
                    reply += self._subnegotiation(bytes(self._sub))
                    self._state = "data"
                else:
                    self._state = "sub"

        return self._decode(bytes(text)), bytes(reply)

    def _answer(self, verb: int, opt: int) -> bytes:
        """The same never-answer-twice rule the board uses, for the same reason."""
        if verb in (DO, DONT):
            wanted = verb == DO
            if opt in _WE_WILL:
                if self._will.get(opt) == wanted:
                    return b""
                self._will[opt] = wanted
                out = bytes([IAC, WILL if wanted else WONT, opt])
                if wanted and opt == OPT_NAWS:
                    out += self.size()
                return out
            if self._will.get(opt) is False:
                return b""
            self._will[opt] = False
            return bytes([IAC, WONT, opt])

        if verb in (WILL, WONT):
            wanted = verb == WILL
            if opt in _WE_DO:
                if self._do.get(opt) == wanted:
                    return b""
                self._do[opt] = wanted
                return bytes([IAC, DO if wanted else DONT, opt])
            if self._do.get(opt) is False:
                return b""
            self._do[opt] = False
            return bytes([IAC, DONT, opt])
        return b""

    def _subnegotiation(self, payload: bytes) -> bytes:
        if len(payload) >= 2 and payload[0] == OPT_TTYPE and payload[1] == 1:
            # The board asked what kind of terminal this is. The answer
            # decides whether it sends CP437 bytes or UTF-8, so it also
            # decides how what comes back has to be decoded.
            self.encoding = "cp437" if prefers_cp437(self.terminal) else "utf-8"
            return (bytes([IAC, SB, OPT_TTYPE, 0])
                    + self.terminal.encode("ascii", "replace")
                    + bytes([IAC, SE]))
        return b""

    def _decode(self, data: bytes) -> str:
        """Bytes to text, holding back anything cut in half by a packet."""
        if not data:
            return ""
        self._partial += data
        if self.encoding == "cp437":
            out = bytes(self._partial).decode("cp437")
            self._partial.clear()
            return out
        try:
            out = bytes(self._partial).decode("utf-8")
            self._partial.clear()
            return out
        except UnicodeDecodeError as exc:
            out = bytes(self._partial[:exc.start]).decode("utf-8")
            del self._partial[:exc.start]
            if len(self._partial) > 8:                # junk, not a truncation
                self._partial.clear()
            return out


def call(host: str = "127.0.0.1", port: int = 23,
         terminal: str = DEFAULT_TERMINAL, timeout: float = 10.0) -> int:
    """Connect, and hand the terminal over until the far end hangs up."""
    columns, rows = _terminal_size()
    session = Session(columns, rows, terminal)

    try:
        sock = socket.create_connection((host, port), timeout)
    except OSError as exc:
        print(f"cannot reach {host}:{port} - {exc}", file=sys.stderr)
        print("is the board running? start it with:  mb bbs", file=sys.stderr)
        return 1
    sock.settimeout(None)

    print(f"Connected to {host}:{port}. Ctrl-] to hang up.")
    with _raw_terminal():
        try:
            _pump(sock, session)
        except (OSError, KeyboardInterrupt):
            pass
        finally:
            try:
                sock.close()
            except OSError:
                pass
    print("\r\nNO CARRIER")
    return 0


def _pump(sock: socket.socket, session: Session) -> None:
    """Both directions at once, on one thread, with no polling loop."""
    picker = selectors.DefaultSelector()
    picker.register(sock, selectors.EVENT_READ, "socket")
    try:
        picker.register(sys.stdin, selectors.EVENT_READ, "keyboard")
    except (ValueError, OSError):
        # No selectable stdin - a pipe on Windows. Fall back to blocking
        # reads on the socket only, which is enough to watch a board.
        picker = None

    if picker is None:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                return
            text, reply = session.feed(chunk)
            if reply:
                sock.sendall(reply)
            if text:
                sys.stdout.write(text)
                sys.stdout.flush()

    while True:
        for key, _mask in picker.select():
            if key.data == "socket":
                chunk = sock.recv(4096)
                if not chunk:
                    return
                text, reply = session.feed(chunk)
                if reply:
                    sock.sendall(reply)
                if text:
                    sys.stdout.write(text)
                    sys.stdout.flush()
            else:
                data = sys.stdin.buffer.raw.read(1024)
                if not data:
                    return
                if b"\x1d" in data:                   # Ctrl-] hangs up
                    return
                sock.sendall(data.replace(bytes([IAC]), bytes([IAC, IAC])))


def _terminal_size() -> tuple[int, int]:
    import os

    try:
        size = os.get_terminal_size()
        return max(40, size.columns), max(10, size.lines)
    except OSError:
        return 80, 24


class _raw_terminal:
    """Put the local terminal in raw mode, and always put it back."""

    def __enter__(self):
        self.saved = None
        if sys.platform == "win32":
            self._windows_ansi()
            return self
        try:
            import termios
            import tty

            fd = sys.stdin.fileno()
            self.saved = (fd, termios.tcgetattr(fd))
            tty.setraw(fd)
        except Exception:                                 # noqa: BLE001
            self.saved = None
        return self

    def __exit__(self, *_exc) -> None:
        if self.saved is not None:
            import termios

            fd, attributes = self.saved
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, attributes)
            except Exception:                             # noqa: BLE001
                pass

    @staticmethod
    def _windows_ansi() -> None:
        """Turn on ANSI in a Windows console, which is off by default."""
        try:
            import ctypes

            kernel = ctypes.windll.kernel32
            for handle, flag in ((-11, 0x0004), (-10, 0x0200)):
                mode = ctypes.c_uint32()
                if kernel.GetConsoleMode(kernel.GetStdHandle(handle),
                                         ctypes.byref(mode)):
                    kernel.SetConsoleMode(kernel.GetStdHandle(handle),
                                          mode.value | flag)
        except Exception:                                 # noqa: BLE001
            pass
