"""ANSI screen drawing, the way a 1980s bulletin board did it.

A BBS was graphical. Not windowed - graphical: sixteen colours, the IBM
PC's line-drawing characters, and block glyphs used as pixels. Everything
here builds those bytes.

Two encodings, because two kinds of client call in. A period client
(SyncTERM, NetRunner, mTelnet) speaks code page 437 and wants raw bytes in
the 0x80-0xFF range. A modern terminal speaks UTF-8. The screens are
written once in Unicode and encoded on the way out, so the same drawing
code serves both.

Colour is written as SGR escapes rather than a library, because the escapes
are the interesting part: `ESC[1;33m` is what made a BBS yellow in 1987 and
is what makes one yellow now.
"""

from __future__ import annotations

ESC = "\x1b"
CSI = ESC + "["

RESET = CSI + "0m"
BOLD = CSI + "1m"
BLINK = CSI + "5m"

# The sixteen colours every BBS had, by the names sysops used for them.
COLOURS = {
    "black": 0, "red": 1, "green": 2, "yellow": 3,
    "blue": 4, "magenta": 5, "cyan": 6, "white": 7,
}


def fg(name: str, bright: bool = False) -> str:
    """Foreground colour. `bright` is the high-intensity half of the palette."""
    return f"{CSI}{1 if bright else 0};{30 + COLOURS[name]}m"


def bg(name: str) -> str:
    return f"{CSI}{40 + COLOURS[name]}m"


def colour(text: str, name: str, bright: bool = False) -> str:
    return f"{fg(name, bright)}{text}{RESET}"


# Shorthands, used constantly below.
K, R, G, Y = fg("black"), fg("red"), fg("green"), fg("yellow")
B, M, C, W = fg("blue"), fg("magenta"), fg("cyan"), fg("white")
HR, HG, HY = fg("red", True), fg("green", True), fg("yellow", True)
HB, HM, HC, HW = (fg("blue", True), fg("magenta", True),
                  fg("cyan", True), fg("white", True))
GREY = fg("black", True)

CLS = CSI + "2J" + CSI + "H"
HOME = CSI + "H"
CLEAR_LINE = CSI + "2K"
HIDE_CURSOR = CSI + "?25l"
SHOW_CURSOR = CSI + "?25h"


def at(row: int, col: int) -> str:
    return f"{CSI}{row};{col}H"


# ---- boxes ------------------------------------------------------------------

SINGLE = "┌┐└┘─│├┤"
DOUBLE = "╔╗╚╝═║╠╣"


def box(title: str, lines: list[str], width: int = 70,
        style: str = SINGLE, frame: str = HC, text: str = HW,
        head: str = HY) -> list[str]:
    """A framed panel with its title let into the top edge.

    Returns lines rather than a string so callers can stack panels without
    re-splitting, and so the tests can count columns.
    """
    tl, tr, bl, br, h, v = style[0], style[1], style[2], style[3], style[4], style[5]
    inner = width - 2
    if title:
        label = f" {title} "
        bar = f"{tl}{h}{label}{h * (inner - len(label) - 1)}{tr}"
    else:
        bar = f"{tl}{h * inner}{tr}"

    out = [f"{frame}{bar}{RESET}"]
    for line in lines:
        out.append(f"{frame}{v}{RESET}{text}{pad(line, inner)}{RESET}"
                   f"{frame}{v}{RESET}")
    out.append(f"{frame}{bl}{h * inner}{br}{RESET}")
    if title:
        # Re-colour the title in place; simpler than assembling it coloured.
        out[0] = out[0].replace(f" {title} ", f" {head}{title}{frame} ", 1)
    return out


def visible(text: str) -> str:
    """The text with escape sequences removed - what a reader actually sees."""
    out, i = [], 0
    while i < len(text):
        if text[i] == ESC:
            j = i + 1
            if j < len(text) and text[j] == "[":
                j += 1
                while j < len(text) and not ("@" <= text[j] <= "~"):
                    j += 1
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def width_of(text: str) -> int:
    return len(visible(text))


def pad(text: str, width: int) -> str:
    """Pad to `width` counting only visible characters, and never overrun."""
    seen = width_of(text)
    if seen > width:
        return truncate(text, width)
    return text + " " * (width - seen)


def truncate(text: str, width: int) -> str:
    """Cut to `width` visible characters, keeping escape sequences intact."""
    out, seen, i = [], 0, 0
    while i < len(text) and seen < width:
        if text[i] == ESC:
            j = i + 1
            if j < len(text) and text[j] == "[":
                j += 1
                while j < len(text) and not ("@" <= text[j] <= "~"):
                    j += 1
            out.append(text[i:j + 1])
            i = j + 1
            continue
        out.append(text[i])
        seen += 1
        i += 1
    return "".join(out)


def centre(text: str, width: int = 78) -> str:
    seen = width_of(text)
    if seen >= width:
        return text
    return " " * ((width - seen) // 2) + text


def wrap(text: str, width: int) -> list[str]:
    """Break text to a width, keeping blank lines. Paragraphs, not columns."""
    width = max(20, width)
    out: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            if width_of(line) + width_of(word) + 1 > width and line:
                out.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        out.append(line)
    return out


def rule(width: int = 78, char: str = "═", shade: str = HB) -> str:
    return f"{shade}{char * width}{RESET}"


def gradient(text: str, palette: tuple[str, ...] = (HC, C, HB, B)) -> str:
    """The colour-cycled headline every board had. One shade per character."""
    out = []
    for i, ch in enumerate(text):
        out.append(palette[i % len(palette)] + ch)
    return "".join(out) + RESET


# ---- the logo ---------------------------------------------------------------

# Block-letter art, drawn with the same glyphs a 1987 ANSI artist had. The
# shading (░▒▓█) is what gave those screens depth on a 16-colour display.
_LOGO = r"""
 ███╗   ███╗ ██████╗ ████████╗██╗  ██╗███████╗██████╗
 ████╗ ████║██╔═══██╗╚══██╔══╝██║  ██║██╔════╝██╔══██╗
 ██╔████╔██║██║   ██║   ██║   ███████║█████╗  ██████╔╝
 ██║╚██╔╝██║██║   ██║   ██║   ██╔══██║██╔══╝  ██╔══██╗
 ██║ ╚═╝ ██║╚██████╔╝   ██║   ██║  ██║███████╗██║  ██║
 ╚═╝     ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
        ██████╗ ██████╗  █████╗ ██╗███╗   ██╗
        ██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║
        ██████╔╝██████╔╝███████║██║██╔██╗ ██║
        ██╔══██╗██╔══██╗██╔══██║██║██║╚██╗██║
        ██████╔╝██║  ██║██║  ██║██║██║ ╚████║
        ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
"""


def logo() -> str:
    """The board's opening screen, shaded the way a login screen was.

    Every glyph in it is in code page 437 as well as Unicode - the heavy
    block and the double-line box set are exactly what an IBM PC had - so
    one drawing serves a 1987 client and a 2026 one.
    """
    art = _LOGO
    palette = (HC, HC, HB, HB, B, B)
    out = []
    for i, line in enumerate(art.strip("\n").split("\n")):
        out.append(palette[i % len(palette)] + line + RESET)
    return "\n".join(out)


SHADE = "░▒▓█"


def shaded_bar(width: int = 78) -> str:
    """The fade-in strip that separated a header from a menu."""
    step = max(1, width // 8)
    body = ("░" * step + "▒" * step + "▓" * step + "█" * (width - 3 * step))
    return f"{HB}{body}{RESET}"


# ---- pictures ---------------------------------------------------------------

# Two pixels per character cell: the upper half-block takes the foreground
# colour and the lower half shows through as background. It doubles the
# vertical resolution, which is the difference between a recognisable
# picture and a smudge.
HALF = "▀"


def _xterm256(r: int, g: int, b: int) -> int:
    """Nearest xterm-256 index. The 6x6x6 cube, or the grey ramp when flat."""
    if abs(r - g) < 12 and abs(g - b) < 12 and abs(r - b) < 12:
        grey = round((r + g + b) / 3)
        if grey < 8:
            return 16
        if grey > 248:
            return 231
        return 232 + min(23, round((grey - 8) / 247 * 23))
    q = [round(v / 255 * 5) for v in (r, g, b)]
    return 16 + 36 * q[0] + 6 * q[1] + q[2]


def _nearest16(r: int, g: int, b: int) -> tuple[int, bool]:
    """Nearest of the sixteen colours a 1987 client actually had."""
    best, best_d, best_bright = 0, 1 << 30, False
    for bright in (False, True):
        level = 255 if bright else 170
        for name, index in COLOURS.items():
            cr = level if index & 1 else (85 if bright else 0)
            cg = level if index & 2 else (85 if bright else 0)
            cb = level if index & 4 else (85 if bright else 0)
            d = (cr - r) ** 2 + (cg - g) ** 2 + (cb - b) ** 2
            if d < best_d:
                best, best_d, best_bright = index, d, bright
    return best, best_bright


def _cells(pixel, width: int, height: int, colours: int) -> str:
    """The shared half-block renderer. `pixel(x, y)` returns (r, g, b)."""
    lines = []
    for y in range(0, height, 2):
        row = []
        last = None
        for x in range(width):
            top = pixel(x, y)
            bottom = pixel(x, y + 1) if y + 1 < height else top
            if colours >= 256:
                pair = (_xterm256(*top), _xterm256(*bottom))
                if pair != last:
                    row.append(f"{CSI}38;5;{pair[0]}m{CSI}48;5;{pair[1]}m")
                    last = pair
            else:
                ti, tb = _nearest16(*top)
                bi, _ = _nearest16(*bottom)
                pair = (ti, tb, bi)
                if pair != last:
                    row.append(f"{CSI}{1 if tb else 0};{30 + ti};{40 + bi}m")
                    last = pair
            row.append(HALF)
        lines.append("".join(row) + RESET)
    return "\n".join(lines)


def picture(image, width: int = 60, colours: int = 256) -> str:
    """Render a PIL image as ANSI, two pixels to the character cell.

    `colours` is 256 for a modern client and 16 for a period one. Sixteen
    colours is not a downgrade for authenticity's sake: a real 1980s client
    cannot show more, and a screen full of unsupported escapes is worse than
    a coarse picture.
    """
    image = image.convert("RGB")
    w, h = image.size
    height = max(2, int(width * h / w * 0.5)) * 2      # even: two rows a cell
    image = image.resize((width, height))
    px = image.load()
    return _cells(lambda x, y: px[x, y], width, height, colours)


def picture_tensor(tensor, width: int = 60, colours: int = 256) -> str:
    """The same, from a model-shaped tensor, without going through PIL.

    Drawing is arithmetic on numbers that are already in memory. Requiring
    an imaging library to turn them into coloured blocks would make the one
    part of the gallery that always works depend on the one that might not
    be installed.
    """
    import torch

    if tensor.dim() == 4:
        tensor = tensor[0]
    _, h, w = tensor.shape
    height = max(2, int(width * h / w * 0.5)) * 2
    small = torch.nn.functional.interpolate(
        tensor.unsqueeze(0).float(), size=(height, width),
        mode="bilinear", align_corners=False)[0]
    grid = (small.clamp(0, 1) * 255).round().to(torch.int).tolist()

    def pixel(x: int, y: int):
        return grid[0][y][x], grid[1][y][x], grid[2][y][x]

    return _cells(pixel, width, height, colours)


# ---- the vintage look -------------------------------------------------------
#
# A 1987 menu did not look like a tidy box. It had a shaded header made of
# ░▒▓█, a title in a colour that cycled, entries in two colours so the key
# stood out from the description, a drop shadow under every panel, and a
# double rule across the bottom with the time left on it. That was the
# house style of the whole scene, and it is what makes a screen read as a
# board rather than as a program with borders.

SHADOW = "\x1b[0;30m"          # the shade a panel casts, in dark grey


def fade(width: int, palette: tuple[str, ...] = (HB, HC, C, B)) -> str:
    """The shaded strip that topped a menu: ░ into ▒ into ▓ into █."""
    step = max(1, width // 4)
    parts = []
    for i, glyph in enumerate("░▒▓█"):
        run = step if i < 3 else width - 3 * step
        parts.append(palette[i % len(palette)] + glyph * run)
    return "".join(parts) + RESET


def banner(title: str, width: int = 78, subtitle: str = "") -> list[str]:
    """A menu header the way a board drew one: shading, then the title."""
    out = [fade(width)]
    label = f"  {title}  "
    pad_left = max(0, (width - width_of(label)) // 2)
    out.append(f"{HB}{'▓' * pad_left}{bg('blue')}{HW}{BOLD}{label}"
               f"{RESET}{HB}{'▓' * (width - pad_left - width_of(label))}"
               f"{RESET}")
    if subtitle:
        out.append(centre(f"{C}{subtitle}{RESET}", width))
    out.append(f"{B}{'▀' * width}{RESET}")
    return out


def panel(title: str, lines: list[str], width: int = 74,
          frame: str = HB, head: str = HY, shadow: bool = True) -> list[str]:
    """A framed panel with a drop shadow, as every ANSI menu had.

    The shadow is two characters wide on the right and one row deep,
    drawn in dark grey - which on a black terminal is exactly the effect a
    1990 artist was after and costs nothing but two columns.
    """
    body = box(title, lines, width=width, frame=frame, head=head,
               style=DOUBLE)
    if not shadow:
        return body
    out = [body[0] + f"{SHADOW}▖{RESET}"]
    for line in body[1:]:
        out.append(line + f"{SHADOW}██{RESET}")
    out.append(" " * 2 + f"{SHADOW}{'▀' * (width)}{RESET}")
    return out


def entry(key: str, label: str, note: str = "", available: bool = True,
          key_colour: str = HY, label_colour: str = HW,
          note_colour: str = GREY) -> str:
    """One menu line: `[K] Label   note`, with the key in its own colour.

    Brackets rather than a bare letter, because that is what a board used
    and because it is what makes a menu scannable: the eye finds the
    bracket, not the word.
    """
    if not available:
        return f"{GREY} {key}  {label}{RESET}"
    line = f"{key_colour}[{HW}{key}{key_colour}]{RESET} {label_colour}{label}"
    if note:
        line = f"{line}  {note_colour}{note}"
    return line + RESET


def status(left: str, right: str, width: int = 78,
           ground: str = "blue") -> str:
    """The bar along the bottom: who you are on the left, time on the right."""
    gap = max(1, width - width_of(left) - width_of(right) - 2)
    return (f"{bg(ground)}{HW} {left}{' ' * gap}{HY}{right} {RESET}")


def marquee(text: str, width: int = 78) -> str:
    """A colour-cycled headline, centred, the way a board titled a screen."""
    return centre(gradient(text.upper()), width)
