"""Serializable runtime selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ObservableKind = Literal["z", "z_sum", "hamiltonian"]


@dataclass(frozen=True)
class RuntimePolicy:
    """Serializable execution choice owned by ``Module``."""

    mode: str = "statevector"
    backend: str = "pytorch"
    observable: ObservableKind = "z"
    observable_wires: tuple[int, ...] = (0,)
    allow_backend_fallback: bool = True
    correctness_debug: bool = False
    mps_max_bond: int | None = None
    mps_cutoff: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {
            "statevector",
            "mps",
            "tensor_network",
            "distributed_statevector",
        }:
            raise ValueError(f"unsupported fq.Module mode {self.mode!r}")
        if self.backend not in {"pytorch", "jax"}:
            raise ValueError(f"unsupported fq.Module backend {self.backend!r}")
        if self.observable not in {"z", "z_sum", "hamiltonian"}:
            raise ValueError(f"unsupported fq.Module observable {self.observable!r}")
        if not self.observable_wires or any(wire < 0 for wire in self.observable_wires):
            raise ValueError("observable_wires must contain non-negative wire indices")
        if self.observable == "z" and len(self.observable_wires) != 1:
            raise ValueError("observable='z' requires exactly one observable wire")
        if self.mps_max_bond is not None and int(self.mps_max_bond) <= 0:
            raise ValueError("mps_max_bond must be positive when provided")
        if float(self.mps_cutoff) < 0:
            raise ValueError("mps_cutoff must be non-negative")


__all__ = ("RuntimePolicy",)
