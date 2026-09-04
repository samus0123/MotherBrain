"""The work behind the four menu options, with no interface attached.

The terminal console, the browser console and the desktop window all offer the
same four things, and all three used to be free to disagree about what those
things meant. Generation lives here instead, as generators that yield text as
it arrives, so a caller can print it, stream it down a socket, or append it to
a text widget without any of them re-deriving how a prompt is seeded.

Nothing here imports an interface toolkit, and nothing here prints.
"""

from __future__ import annotations

import re
from typing import Iterator

# Words that say what kind of thing is wanted rather than what it does. They
# make poor identifiers, so they are dropped when a request becomes a function
# name.
_FILLER = {"a", "an", "the", "that", "to", "and", "of", "for", "in", "it",
           "with", "program", "script", "make", "write", "create", "build",
           "me", "some", "please"}


def code_seed(want: str) -> tuple[str, str]:
    """Turn a plain-English request into the start of a Python file.

    Returns (opener, head): the text to show as already-written, and the full
    prompt to condition on. A base model cannot be told what to write - it
    continues from context - so the request becomes a docstring and its own
    words become the function name. That is the shape the model saw in
    training, and it is what pulls the body towards the subject.
    """
    words = [w for w in re.findall(r"[A-Za-z]+", want.lower()) if w not in _FILLER]
    slug = "_".join(words[:4]) or "main"
    opener = f"def {slug}("
    return opener, f'"""{want}"""\n\n\n{opener}'


def default_filename(want: str) -> str:
    """A reasonable file name for a program described by `want`."""
    words = [w for w in re.findall(r"[A-Za-z]+", want.lower()) if w not in _FILLER]
    return f"{'_'.join(words[:3]) or 'program'}.py"


def stream(model, tok, device, prompt: str, max_tokens: int = 120,
           temperature: float = 0.8, top_k: int = 40, top_p: float = 0.95,
           repetition_penalty: float = 1.1, image=None) -> Iterator[str]:
    """Yield decoded text as the model produces it.

    Decoding one token at a time can split a multi-byte character across two
    pieces, so bytes are accumulated and only flushed when they decode - the
    alternative is replacement characters appearing mid-word in any language
    that needs them.
    """
    import torch

    from motherbrain.tokenizer import EOS_ID

    ids = torch.tensor([tok.encode(prompt, bos=True)], device=device)
    kwargs = {"max_new_tokens": max_tokens, "temperature": temperature,
              "top_k": top_k, "top_p": top_p,
              "repetition_penalty": repetition_penalty, "eos_id": EOS_ID}
    if image is not None:
        kwargs["images"] = image

    pending: list[int] = []
    for token in model.generate(ids, **kwargs):
        pending.append(token)
        piece = tok.decode(pending)
        if "�" not in piece:
            pending.clear()
            yield piece
    if pending:
        yield tok.decode(pending)


def generate_code(model, tok, device, want: str, max_tokens: int = 120
                  ) -> Iterator[str]:
    """Stream a program written from a description, opener included."""
    opener, head = code_seed(want)
    yield opener
    yield from stream(model, tok, device, head, max_tokens=max_tokens,
                      temperature=0.6, repetition_penalty=1.2)


CODE_CAVEAT = (
    "This is plausible Python, not working Python: a model this size "
    "reproduces the shape of code and cannot be told what to write. "
    "Read it before running it."
)
