"""Growing the model when it learns: every patch adds parameters.

A LoRA patch teaches the model something new without changing its size. This
module does the other thing: it makes the model *bigger* as it learns, by
adding experts. Each patch appends new experts to every mixture-of-experts
layer, trains only those, and leaves the existing weights untouched. The
parameter count therefore rises with every version and never falls.

Two properties make that safe rather than merely impressive:

* **Growth is a no-op at birth.** A new expert's output projection starts at
  zero and its router bias starts at -1e9, so it cannot be selected and
  contributes nothing. The grown model computes exactly what it did before,
  to the bit. Training is what makes it diverge, and only in the direction the
  new information pushes it.
* **Compute does not grow with it.** Only `n_experts_per_token` experts run for
  any token, so a model that has grown through fifty versions costs the same
  per token as it did at version one. That is the whole reason the parameter
  count can be allowed to run away.

A dense model has no experts to append to, so the first growth converts its
feed-forward layers into MoE layers: the existing dense FFN becomes an
always-on shared expert, which preserves its behaviour exactly, and the new
routed experts are added alongside it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from motherbrain.config import ModelConfig
from motherbrain.model import MoE, SwiGLU

# Large enough to zero the softmax weight of an unreleased expert, small enough
# to stay finite in fp32 arithmetic.
HELD_OUT = -1e9


def _new_expert(d_model: int, d_ff: int) -> SwiGLU:
    """An expert that outputs exactly zero until it is trained."""
    expert = SwiGLU(d_model, d_ff)
    nn.init.normal_(expert.gate.weight, std=0.02)
    nn.init.normal_(expert.up.weight, std=0.02)
    nn.init.zeros_(expert.down.weight)  # the output projection is what silences it
    return expert


@torch.no_grad()
def _densify(block, cfg: ModelConfig, n_new: int) -> MoE:
    """Turn a dense feed-forward block into an MoE one, preserving behaviour.

    The existing FFN becomes a shared expert - always on, applied to every
    token - so the layer still computes what it did. The new routed experts are
    additions on top, and start silent.
    """
    grown = ModelConfig.from_dict(cfg.to_dict())
    grown.n_experts = n_new
    grown.n_shared_experts = max(cfg.n_shared_experts, 1)

    moe = MoE(grown)
    moe.shared[0] = block.ffn                     # the original dense FFN, intact
    for i in range(n_new):
        moe.experts[i] = _new_expert(cfg.d_model, cfg.d_ff)
    nn.init.zeros_(moe.router.weight)
    moe.expert_bias.fill_(HELD_OUT)
    return moe


@torch.no_grad()
def _extend(moe: MoE, cfg: ModelConfig, n_new: int) -> None:
    """Append experts to an existing MoE layer, preserving behaviour."""
    old_n = moe.n_experts
    for _ in range(n_new):
        moe.experts.append(_new_expert(cfg.d_model, cfg.d_ff))

    router = nn.Linear(cfg.d_model, old_n + n_new, bias=False)
    nn.init.zeros_(router.weight)
    router.weight[:old_n].copy_(moe.router.weight)
    moe.router = router

    bias = nn.Parameter(torch.full((old_n + n_new,), HELD_OUT))
    bias[:old_n].copy_(moe.expert_bias)
    moe.expert_bias = bias
    moe.n_experts = old_n + n_new


@torch.no_grad()
def grow(model, n_new: int) -> tuple[ModelConfig, list[nn.Parameter]]:
    """Add `n_new` experts to every feed-forward layer of `model`.

    Mutates the model in place and returns its updated config together with the
    parameters the patch trainer should optimise - the new experts, and the
    router that has to learn when to reach for them.
    """
    if n_new < 1:
        raise ValueError("growth must add at least one expert")

    cfg = model.cfg
    trainable: list[nn.Parameter] = []
    grew_dense = False

    for block in model.blocks:
        if block.is_moe:
            first_new = block.ffn.n_experts
            _extend(block.ffn, cfg, n_new)
        else:
            block.ffn = _densify(block, cfg, n_new)
            block.is_moe = True
            first_new = 0
            grew_dense = True
        moe = block.ffn
        for expert in list(moe.experts)[first_new:]:
            trainable.extend(expert.parameters())
        trainable.append(moe.router.weight)
        trainable.append(moe.expert_bias)

    # The config has to describe the model that now exists, or it will not
    # rebuild correctly from a checkpoint.
    new_cfg = ModelConfig.from_dict(cfg.to_dict())
    new_cfg.n_experts = model.blocks[0].ffn.n_experts
    new_cfg.moe_every = 1
    if grew_dense:
        new_cfg.n_shared_experts = max(cfg.n_shared_experts, 1)
    new_cfg.n_experts_per_token = min(
        max(cfg.n_experts_per_token, 1), new_cfg.n_experts)
    model.cfg = new_cfg
    for block in model.blocks:
        block.ffn.cfg = new_cfg
        block.ffn.top_k = new_cfg.n_experts_per_token
    return new_cfg, trainable


@torch.no_grad()
def release(model, n_new: int) -> None:
    """Let the newly added experts be routed to.

    Held-out experts receive no gradient, because top-k never selects them, so
    they would never learn anything. Training clears the bias on the new
    experts only; the existing ones keep whatever bias they had.
    """
    for block in model.blocks:
        if not block.is_moe:
            continue
        bias = block.ffn.expert_bias
        bias[bias.numel() - n_new:] = 0.0
