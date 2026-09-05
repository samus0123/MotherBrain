"""Reasoning: propose with the model, check against reality, repair, choose.

A 50M-parameter model cannot think step by step. Asked to, it produces the
*shape* of reasoning - "first... then... therefore..." - with nothing behind
it, which is worse than not trying, because it reads like an argument.

What it can do is generate many candidates and tell you which of them it finds
more likely. That is a real judgement, and it comes from the model rather than
from a rule someone wrote. Pair it with checks that cannot be fooled - does
this parse, does it run, does it produce output - and you get a system that
reasons even though no single forward pass does:

    propose   the model writes several candidates, at different temperatures
    check     each is put to a test reality answers: ast.parse, execution
    repair    a failure is truncated at the error and continued from there
    choose    among survivors, the model's own likelihood breaks the tie
    explain   every step is recorded, including the ones that failed

The trace is the reasoning. It is deliberately not written in the model's
voice: these are things that were done and observed, not claims it is making.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class Step:
    """One thing that was tried, and what came of it."""

    name: str
    detail: str = ""
    ok: bool = True
    note: str = ""


@dataclass
class Trace:
    """The record of how an answer was arrived at."""

    goal: str
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    succeeded: bool = False

    def add(self, name: str, detail: str = "", ok: bool = True,
            note: str = "") -> Step:
        step = Step(name, detail, ok, note)
        self.steps.append(step)
        return step

    def render(self, width: int = 68) -> str:
        rule = "─" * width
        out = [rule, f"  reasoning about: {self.goal}", rule]
        for i, step in enumerate(self.steps, 1):
            mark = "✓" if step.ok else "✗"
            out.append(f"  {i}. {mark} {step.name}")
            if step.detail:
                out.append(f"       {step.detail}")
            if step.note:
                out.append(f"       {step.note}")
        out.append(rule)
        out.append("  " + ("answered" if self.succeeded
                           else "no candidate survived the checks"))
        out.append(rule)
        return "\n".join(out)


# ---- the model's own judgement --------------------------------------------


def likelihood(model, tok, device, text: str) -> float:
    """Mean log-likelihood per token: how ordinary this text looks to it.

    This is the model judging rather than generating, and it is the only
    opinion in here that is actually the model's. Higher is more expected.
    Length is divided out so a short candidate does not win by being short.
    """
    import torch

    ids = tok.encode(text, bos=True)
    if len(ids) < 2:
        return float("-inf")
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    with torch.no_grad():
        logits, _ = model(x, targets=y)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        picked = logprobs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return float(picked.mean())


# ---- checks that cannot be argued with -------------------------------------


def parses(code: str) -> tuple[bool, str, int]:
    """Does this actually compile? Returns (ok, message, failing line)."""
    try:
        ast.parse(code)
        return True, "", 0
    except SyntaxError as exc:
        return False, exc.msg or "invalid syntax", exc.lineno or 0
    except (ValueError, MemoryError, RecursionError) as exc:
        return False, str(exc), 0


def runs(code: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Execute it in a separate process and report what happened.

    A separate process because code written by a language model should not
    share an interpreter with the thing that asked for it.
    """
    try:
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"did not finish within {timeout:g}s"
    except OSError as exc:
        return False, str(exc)
    if proc.returncode != 0:
        last = (proc.stderr or "").strip().splitlines()
        return False, last[-1] if last else f"exit {proc.returncode}"
    return True, (proc.stdout or "").strip()


def truncate_at(code: str, line: int) -> str:
    """Keep the part before a failure, so generation can resume from it.

    Everything up to the bad line was fine; only what followed was not. This
    is what makes a second attempt a repair rather than a fresh guess.
    """
    if line <= 1:
        return ""
    kept = code.splitlines()[:line - 1]
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept) + ("\n" if kept else "")


QUOTES = (chr(34), chr(39))
TRIPLES = (chr(34) * 3, chr(39) * 3)


def patch_up(code: str) -> str:
    """Close what generation left open, and drop what it left half-written.

    Sampling stops at a token limit, not at a sensible place, so the most
    common failure by far is a docstring or bracket opened and never closed -
    the model was not wrong, it was interrupted. Finishing the construct is a
    repair the system can make with certainty, and it is worth trying before
    deciding the candidate is no good.
    """
    text = code.rstrip()
    if not text:
        return code

    # A trailing line that is obviously mid-thought helps nothing.
    trailing = (",", "(", "[", "{", "=", "+", "-", "*", "/", "\\", ".")
    lines = text.splitlines()
    while lines and lines[-1].rstrip().endswith(trailing):
        lines.pop()
    text = "\n".join(lines)

    # Walk it once, tracking what is open. Quotes first: a bracket inside a
    # string is not a bracket.
    stack = []
    closers = {"(": ")", "[": "]", "{": "}"}
    in_string = None
    i = 0
    while i < len(text):
        if in_string:
            if text.startswith(in_string, i):
                i += len(in_string)
                in_string = None
                continue
            if text[i] == "\\":
                i += 2
                continue
            i += 1
            continue
        if text.startswith(TRIPLES, i):
            in_string = text[i:i + 3]
            i += 3
            continue
        ch = text[i]
        if ch in QUOTES:
            in_string = ch
        elif ch in closers:
            stack.append(closers[ch])
        elif ch in ")]}" and stack and stack[-1] == ch:
            stack.pop()
        i += 1

    if in_string:
        text += in_string
    text += "".join(reversed(stack))

    # A block header with nothing under it is a syntax error on its own.
    lines = text.splitlines()
    if lines and lines[-1].rstrip().endswith(":"):
        indent = len(lines[-1]) - len(lines[-1].lstrip())
        lines.append(" " * (indent + 4) + "pass")
        text = "\n".join(lines)
    return text + "\n"


# ---- the loop --------------------------------------------------------------


def reason_code(model, tok, device, want: str, attempts: int = 4,
                max_tokens: int = 140, on_step=None) -> Trace:
    """Write a program that at least compiles, and say how it got there.

    Each attempt is a fresh sample at a different temperature; a candidate
    that fails to parse is cut back to the last good line and continued from
    there, so the next attempt inherits the part that worked.
    """
    from motherbrain.actions import code_seed, stream

    trace = Trace(goal=want)
    opener, head = code_seed(want)
    survivors: list[tuple[float, str]] = []
    prefix = ""

    for attempt in range(attempts):
        # Cooler first, then warmer: if the likely continuation does not work,
        # there is no point asking for it again.
        temperature = 0.4 + 0.25 * attempt
        prompt = head + prefix
        body = "".join(stream(model, tok, device, prompt,
                              max_tokens=max_tokens, temperature=temperature,
                              repetition_penalty=1.2))
        code = f'"""{want}"""\n\n{opener}{prefix}{body}\n'

        ok, message, line = parses(code)
        step = trace.add(
            f"attempt {attempt + 1} at temperature {temperature:.2f}",
            f"{len(body)} characters generated",
            ok=ok,
            note="" if ok else f"does not compile: {message} (line {line})")
        if on_step:
            on_step(step)

        if not ok:
            # Interrupted rather than wrong? Close what is open, then re-check.
            mended = patch_up(code)
            mended_ok, mended_msg, _mended_line = parses(mended)
            trace.add("closed what generation left open",
                      f"{len(mended) - len(code):+d} characters",
                      ok=mended_ok,
                      note="" if mended_ok else f"still {mended_msg}")
            if mended_ok:
                code, ok, message, line = mended, True, "", 0

        if ok:
            score = likelihood(model, tok, device, code)
            trace.add("scored it with the model itself",
                      f"mean log-likelihood {score:.3f} per token")
            survivors.append((score, code))
            if len(survivors) >= 2:
                break
            continue

        # Repair: keep what compiled, resume from there.
        salvaged = truncate_at(f"{opener}{prefix}{body}", line)
        if salvaged and salvaged != prefix:
            prefix = salvaged[len(opener):] if salvaged.startswith(opener) \
                else salvaged
            trace.add("kept the part that compiled",
                      f"{len(prefix)} characters carried into the next attempt")
        else:
            prefix = ""
            trace.add("nothing salvageable; starting over", ok=False)

    if not survivors:
        trace.add("no candidate compiled", ok=False,
                  note="a model this size often cannot; the last attempt is "
                       "returned as-is")
        return trace

    survivors.sort(reverse=True)
    best_score, best = survivors[0]
    trace.add(f"chose the best of {len(survivors)} that compiled",
              f"by the model's own likelihood ({best_score:.3f})")
    trace.answer = best
    trace.succeeded = True
    return trace


def reason_and_run(model, tok, device, want: str, **kw) -> Trace:
    """As reason_code, and then actually run the winner."""
    trace = reason_code(model, tok, device, want, **kw)
    if not trace.succeeded:
        return trace
    ok, output = runs(trace.answer)
    trace.add("ran it", output[:200] if output else "no output", ok=ok,
              note="" if ok else "it compiles but does not run")
    return trace
