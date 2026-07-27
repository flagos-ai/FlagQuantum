"""Context-local custom operator registration.

Registration uses immutable copy-on-write snapshots.  It never mutates the
built-in matrix table or this module's global namespace.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType, SimpleNamespace
from typing import Mapping

import torch

from . import functional, matrices, operator


@dataclass(frozen=True)
class RegisteredGate:
    """A validated custom matrix gate in the current context."""

    name: str
    matrix: torch.Tensor
    arity: int


_EMPTY: Mapping[str, RegisteredGate] = MappingProxyType({})
_REGISTERED: ContextVar[Mapping[str, RegisteredGate]] = ContextVar(
    "flagquantum_registered_gates", default=_EMPTY
)


def registered_gates() -> Mapping[str, RegisteredGate]:
    """Return the immutable custom-gate snapshot for the current context."""

    return _REGISTERED.get()


def register_gate(name: str, mat: torch.Tensor) -> RegisteredGate:
    """Validate and context-locally register a fixed unitary matrix gate."""

    normalized = str(name).strip().lower()
    if not normalized.isidentifier():
        raise ValueError(f"invalid gate name {name!r}")
    if normalized in matrices.GATE_MAT_DICT:
        raise KeyError(f"gate {normalized!r} is already registered")
    if (
        not isinstance(mat, torch.Tensor)
        or mat.ndim != 2
        or mat.shape[0] != mat.shape[1]
    ):
        raise ValueError("gate matrix must be a square rank-2 torch.Tensor")
    dimension = int(mat.shape[0])
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("gate matrix dimension must be a power of two")
    identity = torch.eye(dimension, dtype=mat.dtype, device=mat.device)
    if not torch.allclose(mat.mH @ mat, identity, atol=1e-6, rtol=1e-5):
        raise ValueError("gate matrix must be unitary")
    existing = _REGISTERED.get().get(normalized)
    if existing is not None:
        if torch.equal(existing.matrix, mat):
            return existing
        raise KeyError(f"gate {normalized!r} is already registered")
    record = RegisteredGate(
        normalized, mat.detach().clone(), dimension.bit_length() - 1
    )
    updated = dict(_REGISTERED.get())
    updated[normalized] = record
    _REGISTERED.set(MappingProxyType(updated))
    return record


def _symbols(record: RegisteredGate) -> SimpleNamespace:
    forward = partial(functional.gate, record.matrix)
    inverse = partial(functional.gate, record.matrix, inverse=True)
    return SimpleNamespace(
        **{
            record.name: forward,
            f"{record.name}_inv": inverse,
        }
    )


def __getattr__(name: str):
    lookup = name.lower()
    inverse = lookup.endswith("_inv")
    gate_name = lookup[:-4] if inverse else lookup
    record = _REGISTERED.get().get(gate_name)
    if record is None:
        raise AttributeError(name)
    symbols = _symbols(record)
    if name.isupper():
        return operator.op_factory(
            gate_name,
            symbols,
            has_params=False,
            trainable=False,
        )
    return getattr(symbols, f"{gate_name}_inv" if inverse else gate_name)


__all__ = ["RegisteredGate", "register_gate", "registered_gates"]
