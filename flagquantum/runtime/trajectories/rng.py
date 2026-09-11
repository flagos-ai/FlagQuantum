"""World-size-independent random streams for quantum trajectories."""

from __future__ import annotations

import torch

_MASK_64 = (1 << 64) - 1
_TORCH_SEED_MODULUS = (1 << 63) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return value ^ (value >> 31)


def derive_trajectory_seed(base_seed: int, trajectory_id: int) -> int:
    """Derive a stable Torch seed from a base seed and global trajectory ID."""

    if trajectory_id < 0:
        raise ValueError("trajectory_id must be non-negative")
    mixed = _splitmix64(int(base_seed) & _MASK_64)
    mixed = _splitmix64(mixed ^ (int(trajectory_id) & _MASK_64))
    return mixed % _TORCH_SEED_MODULUS


def trajectory_generator(
    base_seed: int,
    trajectory_id: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    """Create the private random stream for one global trajectory."""

    device_type = torch.device(device).type
    generator = torch.Generator(device=device_type)
    generator.manual_seed(derive_trajectory_seed(base_seed, trajectory_id))
    return generator


__all__ = ("derive_trajectory_seed", "trajectory_generator")
