"""Things that can be computed exactly, computed exactly.

A language model asked for 4271 * 88 produces a number of about the right
length that is wrong, and produces it with the same confidence it produces a
right one. That is the failure mode worth designing around: not that it cannot
do arithmetic, but that nothing in the output distinguishes the times it can.

So anything with a definite answer is computed rather than generated. The model
is never consulted here, and every result is exact or absent - there is no
middle setting where it guesses.

Each solver reports what it did as well as what it found, because a number
with no working is only worth as much as your trust in where it came from.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass

# ---- arithmetic ------------------------------------------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "len": len, "int": int, "float": float, "pow": pow,
}


@dataclass
class Result:
    """An exact answer, and how it was reached."""

    value: str
    working: str = ""
    kind: str = ""

    def render(self) -> str:
        return f"{self.value}\n  {self.working}" if self.working else self.value


def _eval(node):
    """Evaluate an expression tree, allowing only arithmetic.

    Walking the AST rather than calling eval() is the whole point: eval on a
    string somebody typed is arbitrary code execution, and a calculator is not
    worth that.
    """
    import math

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError("only numbers")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if type(node.op) is ast.Pow and (abs(right) > 1000 or abs(left) > 1e6):
            raise ValueError("that power would take longer than it is worth")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name in _FUNCS:
            return _FUNCS[name](*[_eval(a) for a in node.args])
        if hasattr(math, name) and callable(getattr(math, name)):
            return getattr(math, name)(*[_eval(a) for a in node.args])
        raise ValueError(f"unknown function: {name}")
    if isinstance(node, ast.Name):
        import math as _m

        if node.id in ("pi", "e", "tau"):
            return getattr(_m, node.id)
        raise ValueError(f"unknown name: {node.id}")
    raise ValueError("not arithmetic")


def calculate(text: str) -> Result | None:
    """Work out an arithmetic expression, exactly."""
    expression = text.strip().rstrip("=?").strip()
    if not expression or not re.search(
            r"[\d)]\s*[-+*/%^]|\d\s+x\s+\d|\bsqrt|\bsin|\bcos|\blog|\bfactorial",
            expression):
        return None
    # People typing a calculator mean power by ^, never bitwise xor, and a
    # calculator that silently computed xor would be worse than one that
    # refused. `x` only becomes multiplication in an expression made purely of
    # digits and operators, where it cannot be part of a name.
    if re.fullmatch(r"[\d\s.+\-*/%^x()]+", expression):
        expression = expression.replace("x", "*")
    expression = expression.replace("^", "**")
    try:
        value = _eval(ast.parse(expression, mode="eval").body)
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError,
            OverflowError, KeyError, AttributeError):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    shown = f"{value:,}" if isinstance(value, int) else f"{value:,.10g}"
    return Result(shown, f"computed, not generated: {expression}", "arithmetic")


# ---- numbers ---------------------------------------------------------------


def factor(n: int) -> list[int]:
    """Prime factors, smallest first."""
    factors, d = [], 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def about_number(text: str) -> Result | None:
    """Whether a number is prime, and what it is made of."""
    m = re.search(r"\b(?:is\s+)?(\d{1,18})\s+(prime|composite)\b", text.lower()) \
        or re.search(r"\bfactors?\s+of\s+(\d{1,18})\b()", text.lower())
    if not m:
        return None
    n = int(m.group(1))
    if n < 2:
        return Result(f"{n:,} is neither prime nor composite.", kind="number")
    factors = factor(n)
    if len(factors) == 1:
        return Result(f"{n:,} is prime.",
                      "found by trial division, not by guessing", "number")
    product = " x ".join(f"{f:,}" for f in factors)
    return Result(f"{n:,} is not prime: {product}",
                  f"{len(factors)} prime factors", "number")


def statistics(text: str) -> Result | None:
    """Mean, median and spread of a list of numbers."""
    if not re.search(r"\b(mean|average|median|stdev|std|spread|stats)\b",
                     text.lower()):
        return None
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if len(numbers) < 2:
        return None
    import statistics as st

    ordered = sorted(numbers)
    return Result(
        f"n={len(numbers)}  mean={st.mean(numbers):,.6g}  "
        f"median={st.median(numbers):,.6g}  "
        f"sd={st.stdev(numbers):,.6g}  "
        f"min={ordered[0]:,.6g}  max={ordered[-1]:,.6g}",
        "computed from the numbers you gave", "statistics")


# ---- units and bases -------------------------------------------------------

# To metres, kilograms, seconds. Exact where the definition is exact.
_UNITS = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254, "ft": 0.3048,
    "foot": 0.3048, "feet": 0.3048, "yd": 0.9144, "mi": 1609.344,
    "mile": 1609.344, "miles": 1609.344,
    "mg": 1e-6, "g": 0.001, "kg": 1.0, "lb": 0.45359237, "lbs": 0.45359237,
    "pound": 0.45359237, "pounds": 0.45359237, "oz": 0.028349523125,
    "ms": 0.001, "s": 1.0, "sec": 1.0, "min": 60.0, "hour": 3600.0,
    "h": 3600.0, "day": 86400.0, "days": 86400.0, "week": 604800.0,
}
_FAMILY = {u: ("length" if v in (0.001, 0.01, 1.0, 1000.0, 0.0254, 0.3048,
                                 0.9144, 1609.344) else "")
           for u, v in _UNITS.items()}


def convert(text: str) -> Result | None:
    """Convert between units, or between number bases."""
    lowered = text.lower().strip()

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(°?[cf])\b.*\b(?:to|in)\s+(°?[cf])\b",
                  lowered)
    if m:
        value, src, dst = float(m.group(1)), m.group(2)[-1], m.group(3)[-1]
        if src != dst:
            out = value * 9 / 5 + 32 if src == "c" else (value - 32) * 5 / 9
            return Result(f"{out:,.4g}°{dst.upper()}",
                          f"{value:g}°{src.upper()} converted exactly", "units")

    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z]+)\s+(?:to|in)\s+([a-z]+)", lowered)
    if m:
        value, src, dst = float(m.group(1)), m.group(2), m.group(3)
        if src in _UNITS and dst in _UNITS:
            out = value * _UNITS[src] / _UNITS[dst]
            return Result(f"{out:,.10g} {dst}",
                          f"{value:g} {src} converted by definition", "units")

    m = re.search(r"\b(\d+)\s+(?:in|to|as)\s+(binary|hex|hexadecimal|octal)\b",
                  lowered)
    if m:
        n = int(m.group(1))
        base = m.group(2)
        out = {"binary": bin, "hex": hex, "hexadecimal": hex, "octal": oct}[base](n)
        return Result(out, f"{n:,} in {base}", "bases")

    m = re.search(r"\b(?:0x([0-9a-f]+)|0b([01]+))\s+(?:to|in)\s+decimal\b", lowered)
    if m:
        raw = m.group(1) or m.group(2)
        n = int(raw, 16 if m.group(1) else 2)
        return Result(f"{n:,}", f"from base {16 if m.group(1) else 2}", "bases")
    return None


# ---- text ------------------------------------------------------------------


def digest(text: str) -> Result | None:
    """Hash or encode a string, exactly."""
    m = re.match(r"\s*(sha256|sha1|md5|base64|hex)\s+(?:of\s+)?(.+)",
                 text.strip(), re.I | re.S)
    if not m:
        return None
    how, subject = m.group(1).lower(), m.group(2).strip().strip('"\'')
    raw = subject.encode()
    if how == "base64":
        import base64

        return Result(base64.b64encode(raw).decode(),
                      f"base64 of {len(raw)} bytes", "text")
    if how == "hex":
        return Result(raw.hex(), f"hex of {len(raw)} bytes", "text")
    import hashlib

    return Result(getattr(hashlib, how)(raw).hexdigest(),
                  f"{how} of {len(raw)} bytes", "text")


def count_text(text: str) -> Result | None:
    """Count what is countable in a piece of text."""
    m = re.match(r"\s*count\s+(?:the\s+)?(?:words|chars|characters|lines)\s+"
                 r"(?:in\s+)?(.+)", text.strip(), re.I | re.S)
    if not m:
        return None
    subject = m.group(1)
    return Result(
        f"{len(subject.split()):,} words, {len(subject):,} characters, "
        f"{len(subject.splitlines()) or 1:,} line(s)",
        "counted, not estimated", "text")


# ---- the whole set ---------------------------------------------------------

SOLVERS = (calculate, about_number, statistics, convert, digest, count_text)


def solve(text: str) -> Result | None:
    """The first exact answer any solver can give, or None.

    Order matters only where two could match; each is narrow enough that in
    practice at most one does.
    """
    for solver in SOLVERS:
        try:
            found = solver(text)
        except Exception:                                 # noqa: BLE001
            continue
        if found is not None:
            return found
    return None


HELP = """exact answers, computed rather than generated:

  4271 * 88              arithmetic, including sqrt, sin, log, pi
  is 7919 prime          primality and prime factors
  factors of 1234567
  mean of 3 1 4 1 5 9    mean, median, standard deviation, range
  12 km to miles         length, mass and time; also 100 c to f
  255 in hex             binary, hex, octal, and back to decimal
  sha256 of hello        sha256, sha1, md5, base64, hex
  count words in ...     words, characters, lines
"""
