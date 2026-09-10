"""The telnet protocol, enough of it to run a bulletin board.

Telnet is not a raw socket. Inside the stream are in-band commands, all of
them introduced by the byte 255 (IAC, "interpret as command"), and a client
that is never answered will sit in line mode - buffering until Enter,
echoing locally, and refusing to send the single keypresses a BBS menu is
driven by.

So this negotiates. The server offers to echo and to suppress go-ahead,
which together put a client into character-at-a-time mode, and asks for the
window size and terminal type, which are what let the board draw to the
right width and pick an encoding. Everything else offered by the client is
refused politely, because an unanswered option is worse than a declined one:
some clients wait forever for the answer.

RFCs, for the record: 854 (telnet), 857 (echo), 858 (go-ahead), 1073
(window size), 1091 (terminal type).
"""

from __future__ import annotations

from dataclasses import dataclass, field

IAC = 255
DONT, DO, WONT, WILL = 254, 253, 252, 251
SB, SE = 250, 240
GA, NOP = 249, 241
BRK, IP, AO, AYT, EC, EL = 243, 244, 245, 246, 247, 248

OPT_BINARY = 0
OPT_ECHO = 1
OPT_SGA = 3
OPT_TTYPE = 24
OPT_NAWS = 31

IS, SEND = 0, 1

# What the server takes on itself, and what it asks the client for.
_WE_WILL = (OPT_ECHO, OPT_SGA, OPT_BINARY)
_WE_DO = (OPT_NAWS, OPT_TTYPE, OPT_SGA, OPT_BINARY)


@dataclass
class Negotiation:
    """What a chunk of incoming bytes turned out to contain."""

    data: bytes = b""                    # the caller's actual keystrokes
    reply: bytes = b""                   # what must be sent back, now
    size: tuple[int, int] | None = None  # (columns, rows) if NAWS arrived
    terminal: str = ""                   # terminal type if TTYPE arrived
    interrupt: bool = False              # the caller pressed the break key


@dataclass
class Telnet:
    """A streaming IAC parser. Feed it bytes; it hands back keystrokes.

    Stateful because a command can be split across TCP segments - a two-byte
    IAC DO can easily arrive one byte at a time, and a parser that assumes
    whole commands drops options at random under load.
    """

    columns: int = 80
    rows: int = 24
    terminal: str = ""
    _state: str = "data"
    _verb: int = 0
    _sub: bytearray = field(default_factory=bytearray)
    # What has already been settled, so nothing is ever re-offered. Two
    # machines that treat every confirmation as a fresh request shout IAC at
    # each other until the socket fills; this is what stops that.
    _will: dict = field(default_factory=dict)
    _do: dict = field(default_factory=dict)

    def start(self) -> bytes:
        """The negotiation a board sends the instant a caller connects."""
        out = bytearray()
        for opt in _WE_WILL:
            self._will[opt] = True
            out += bytes([IAC, WILL, opt])
        for opt in _WE_DO:
            self._do[opt] = True
            out += bytes([IAC, DO, opt])
        out += bytes([IAC, SB, OPT_TTYPE, SEND, IAC, SE])
        return bytes(out)

    def feed(self, chunk: bytes) -> Negotiation:
        out = Negotiation()
        data = bytearray()
        reply = bytearray()

        for byte in chunk:
            if self._state == "data":
                if byte == IAC:
                    self._state = "iac"
                else:
                    data.append(byte)
            elif self._state == "iac":
                if byte == IAC:                       # 255 255 is a literal 255
                    data.append(IAC)
                    self._state = "data"
                elif byte in (DO, DONT, WILL, WONT):
                    self._verb, self._state = byte, "opt"
                elif byte == SB:
                    self._sub.clear()
                    self._state = "sub"
                elif byte in (IP, BRK):
                    out.interrupt = True
                    self._state = "data"
                elif byte == AYT:
                    reply += b"\r\n[MotherBrain is here]\r\n"
                    self._state = "data"
                else:                                  # NOP, GA, DM and friends
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
                    self._subnegotiation(bytes(self._sub), out)
                    self._state = "data"
                else:
                    self._state = "sub"

        out.data = bytes(data)
        out.reply = bytes(reply)
        return out

    def _answer(self, verb: int, opt: int) -> bytes:
        """Reply to one option, without ever answering twice.

        Two rules, and they are the whole of it. An option already settled
        the way the far end is asking for gets no answer - repeating a
        confirmation is what starts two machines shouting IAC at each other
        until the socket fills. And an option this board does not implement
        is refused exactly once, not once per time it is offered.
        """
        if verb in (DO, DONT):
            wanted = verb == DO
            if opt in _WE_WILL:
                if self._will.get(opt) == wanted:
                    return b""
                self._will[opt] = wanted
                return bytes([IAC, WILL if wanted else WONT, opt])
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

    def _subnegotiation(self, payload: bytes, out: Negotiation) -> None:
        if not payload:
            return
        opt = payload[0]
        if opt == OPT_NAWS and len(payload) >= 5:
            cols = (payload[1] << 8) | payload[2]
            rows = (payload[3] << 8) | payload[4]
            if 20 <= cols <= 500 and 5 <= rows <= 200:
                self.columns, self.rows = cols, rows
                out.size = (cols, rows)
        elif opt == OPT_TTYPE and len(payload) >= 2 and payload[1] == IS:
            name = payload[2:].decode("ascii", "replace").strip()
            if name:
                self.terminal = name
                out.terminal = name


# Terminal types that mean "this is a period client and it speaks code page
# 437". Everything else is assumed to be a modern UTF-8 terminal, which is
# the safer guess: mojibake on a modern terminal is ugly, but CP437 bytes
# sent to one produce nothing legible at all.
CP437_TERMINALS = ("syncterm", "ansi-bbs", "pcansi", "qansi", "netrunner",
                   "mtelnet", "ansi")


def prefers_cp437(terminal: str) -> bool:
    name = (terminal or "").strip().lower()
    return any(name == t or name.startswith(t) for t in CP437_TERMINALS)


def escape(data: bytes) -> bytes:
    """Double any literal 255 on the way out, as the protocol requires."""
    return data.replace(bytes([IAC]), bytes([IAC, IAC]))
