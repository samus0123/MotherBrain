"""Torch-distributed helpers.

Everything degrades to a sane single-process answer when ``torchrun`` did not
set the usual environment variables, so the same code runs on a laptop and on a
multi-GPU node.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

__all__ = ["DistInfo", "init_distributed", "cleanup_distributed", "all_reduce_mean"]


@dataclass(frozen=True)
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_master(self) -> bool:
        return self.rank == 0


def init_distributed(backend: str | None = None) -> DistInfo:
    """Join the process group if launched under ``torchrun``; otherwise no-op."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistInfo(enabled=False, rank=0, local_rank=0, world_size=1)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size == 1:
        return DistInfo(enabled=False, rank=0, local_rank=local_rank, world_size=1)

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return DistInfo(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: float, info: DistInfo, device: torch.device) -> float:
    """Average a Python scalar across ranks (identity when not distributed)."""
    if not info.enabled:
        return value
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item() / info.world_size)
