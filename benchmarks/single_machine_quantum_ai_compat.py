"""Small import-safe bridge for single-machine quantum AI benchmark helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import sys

import torch


EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "examples" / "single_machine_quantum_ai"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

_DIMER_EXAMPLE_PATH = EXAMPLE_ROOT / "05_mps_1000q_dimer_training.py"
_SPEC = importlib.util.spec_from_file_location("_fq_dimer_example", _DIMER_EXAMPLE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load {_DIMER_EXAMPLE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def init_dimer_teacher(n_pairs: int, *, device: str) -> torch.Tensor:
    return _MODULE._init_teacher(n_pairs, device=device)


def init_dimer_trainable(teacher: torch.Tensor) -> torch.Tensor:
    return _MODULE._init_trainable(teacher)


def dimer_target(teacher: torch.Tensor) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(_MODULE._dimer_observables_jax(_torch_to_jax(teacher))).to(
        device=teacher.device,
        dtype=teacher.dtype,
    )


def dimer_loss(params: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _MODULE.structured_dimer_mps_loss(params, target)


def _torch_to_jax(tensor: torch.Tensor) -> Any:
    import jax.dlpack

    return jax.dlpack.from_dlpack(tensor.detach().contiguous())
