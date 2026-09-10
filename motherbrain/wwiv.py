"""WWIV: the shape Wayne Bell gave a bulletin board in 1984.

WWIV ran a large fraction of the boards anyone remembers, and it had a
character that was entirely its own. This module holds the parts of that
character that are mechanism rather than decoration, so the board can be
built on them instead of merely dressed up as them.

**Heart codes.** WWIV did not write ANSI escapes into its screens. It wrote
Ctrl-C - ASCII 3, which the IBM PC drew as a heart - followed by a digit,
and looked the colour up in a table the sysop could edit. `♥1` meant
"colour one", whatever the sysop had decided colour one was. Screens here
are written the same way and converted on the way out, which is why they
read as WWIV screens rather than as ANSI with WWIV's words in it.

**Security levels.** Every user had an SL and a DSL, both 0-255. SL gated
commands, DSL gated file directories, and AR/DAR flags gated individual
subs and dirs on top of that. A board's whole access policy was those four
numbers per user, and it is a genuinely good design: there is exactly one
place to look to know what somebody can do.

**A time bank.** You had minutes per call and minutes per day, both set by
your SL, and the prompt told you how many were left. It is the reason a
WWIV board never had the problem this one would otherwise have: a caller
who wanders off does not hold a node all night.

**User numbers.** You were #1 through #n, forever, and the sysop was #1.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---- heart codes ------------------------------------------------------------

HEART = "\x03"

# WWIV's default colour table, as it shipped. The numbers are IBM PC
# attribute bytes: low nibble foreground, high nibble background, bit 7
# blink. Kept in that form because that is what a sysop editing colours saw.
DEFAULT_COLOURS = (
    0x07,      # 0  normal text - light grey
    0x0B,      # 1  bright cyan - headings
    0x0E,      # 2  yellow - things that matter
    0x05,      # 3  magenta - quiet detail
    0x9F,      # 4  bright white on blue, blinking - warnings
    0x02,      # 5  green - prompts
    0x8C,      # 6  bright red, blinking - errors
    0x09,      # 7  bright blue - frames
    0x03,      # 8  cyan - secondary
    0x0F,      # 9  bright white - emphasis
)

_FG = (30, 34, 32, 36, 31, 35, 33, 37)          # IBM order to ANSI order


def attribute(byte: int) -> str:
    """One IBM attribute byte as an ANSI escape sequence."""
    blink = bool(byte & 0x80)
    bg = (byte >> 4) & 0x07
    fg = byte & 0x0F
    parts = ["0"]
    if fg & 0x08:
        parts.append("1")
    if blink:
        parts.append("5")
    parts.append(str(_FG[fg & 0x07]))
    if bg:
        parts.append(str(_FG[bg] + 10))
    return "\x1b[" + ";".join(parts) + "m"


def render(text: str, colours: tuple[int, ...] = DEFAULT_COLOURS) -> str:
    """Turn heart codes into ANSI. `♥3` becomes colour three.

    A heart followed by anything that is not a digit is left alone, because
    a caller who types a heart into a message meant to type a heart.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == HEART and i + 1 < len(text) and text[i + 1].isdigit():
            out.append(attribute(colours[int(text[i + 1]) % len(colours)]))
            i += 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def strip(text: str) -> str:
    """The same text with the colour taken out - for widths and for logs."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == HEART and i + 1 < len(text) and text[i + 1].isdigit():
            i += 2
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def width_of(text: str) -> int:
    return len(strip(text))


def pad(text: str, width: int) -> str:
    seen = width_of(text)
    return text + " " * max(0, width - seen)


# ---- security ---------------------------------------------------------------

# What an SL buys you. WWIV shipped a table like this and the sysop edited
# it; the shape is what matters - one number, looked up in one place.
@dataclass(frozen=True)
class Level:
    """What one security level is allowed."""

    minutes_per_call: int
    minutes_per_day: int
    can_post: bool = True
    can_download: bool = True
    can_upload: bool = True
    can_chat: bool = True


LEVELS = {
    10: Level(20, 40, can_upload=False),          # a brand new caller
    20: Level(45, 90),                            # validated
    50: Level(90, 180),                           # regular
    100: Level(180, 360),                         # trusted
    255: Level(1440, 1440),                       # the sysop
}

NEW_USER_SL = 10
NEW_USER_DSL = 10
SYSOP_SL = 255


def level(sl: int) -> Level:
    """The rules for a security level, taking the nearest at or below it."""
    for threshold in sorted(LEVELS, reverse=True):
        if sl >= threshold:
            return LEVELS[threshold]
    return LEVELS[10]


# ---- user records -----------------------------------------------------------

@dataclass
class User:
    """One user record. WWIV kept these in USER.LST, numbered from one."""

    number: int
    name: str
    real_name: str = ""
    sl: int = NEW_USER_SL
    dsl: int = NEW_USER_DSL
    ar: str = ""                     # access-restriction flags, A-P
    dar: str = ""                    # the same, for file directories
    calls: int = 0
    posts: int = 0
    uploads: int = 0
    downloads: int = 0
    minutes: float = 0.0             # total, over every call
    today: float = 0.0               # used so far today
    day: str = ""                    # which day `today` refers to
    first_on: str = ""
    last_on: str = ""
    expert: bool = False
    note: str = ""

    @property
    def sysop(self) -> bool:
        return self.sl >= SYSOP_SL

    def rules(self) -> Level:
        return level(self.sl)

    def minutes_left(self) -> float:
        """Of today's allowance, and of this call's, whichever is smaller."""
        rules = self.rules()
        return max(0.0, min(float(rules.minutes_per_call),
                            rules.minutes_per_day - self.today))

    def may(self, ar: str) -> bool:
        """Does this user hold the flag a sub or directory demands?"""
        return not ar or ar.upper() in self.ar.upper()

    def may_download(self, dar: str, dsl: int) -> bool:
        return (self.dsl >= dsl
                and (not dar or dar.upper() in self.dar.upper()))


class Users:
    """USER.LST, as JSON. Numbered from one, and #1 is the sysop."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.records: dict[int, User] = {}
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for entry in raw:
            try:
                self.records[int(entry["number"])] = User(**entry)
            except (TypeError, ValueError, KeyError):
                continue

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(u) for u in self.sorted()], indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    def sorted(self) -> list[User]:
        return [self.records[n] for n in sorted(self.records)]

    def find(self, who: str) -> User | None:
        """By number or by name, which is how WWIV's login prompt worked."""
        who = who.strip()
        if who.startswith("#"):
            who = who[1:]
        if who.isdigit():
            return self.records.get(int(who))
        lowered = who.lower()
        for user in self.records.values():
            if user.name.lower() == lowered:
                return user
        return None

    def create(self, name: str, sysop: bool = False) -> User:
        number = max(self.records, default=0) + 1
        today = time.strftime("%Y-%m-%d")
        user = User(number=number, name=name[:30],
                    sl=SYSOP_SL if sysop else NEW_USER_SL,
                    dsl=SYSOP_SL if sysop else NEW_USER_DSL,
                    ar="ABCDEFGHIJKLMNOP" if sysop else "",
                    dar="ABCDEFGHIJKLMNOP" if sysop else "",
                    first_on=today, last_on=today, day=today)
        self.records[number] = user
        self.save()
        return user

    def begin_call(self, user: User) -> None:
        """Start of a call: roll the daily allowance over if the day changed."""
        today = time.strftime("%Y-%m-%d")
        if user.day != today:
            user.day, user.today = today, 0.0
        user.calls += 1
        user.last_on = today
        self.save()

    def end_call(self, user: User, minutes: float) -> None:
        user.minutes += minutes
        user.today += minutes
        self.save()


# ---- the message subs and file dirs -----------------------------------------

@dataclass
class Sub:
    """One message sub-board. WWIV called them subs and numbered them."""

    key: str
    name: str
    blurb: str = ""
    read_sl: int = 0
    post_sl: int = 10
    ar: str = ""
    anonymous: bool = False


@dataclass
class Dir:
    """One file directory, gated by DSL and DAR rather than SL and AR."""

    key: str
    name: str
    blurb: str = ""
    dsl: int = 0
    dar: str = ""
    upload: bool = False
    roots: list = field(default_factory=list)
    globs: list = field(default_factory=lambda: ["*"])
    flat: bool = False
