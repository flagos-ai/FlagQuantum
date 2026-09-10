"""Reusable builder-program state for ``Module`` compilation."""

from __future__ import annotations

from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from ..core.ir import CircuitIR
from ..core.parameters import ParameterExpression


class BuilderBindings:
    """Task-local dynamic tensors consumed by one reusable circuit program."""

    def __init__(self) -> None:
        self._values: ContextVar[tuple[torch.Tensor, ...] | None] = ContextVar(
            "flagquantum_builder_bindings", default=None
        )

    def bind(self, values: tuple[torch.Tensor, ...]) -> None:
        self._values.set(values)

    def clear(self) -> None:
        self._values.set(None)

    def value(self, index: int) -> torch.Tensor:
        values = self._values.get()
        if values is None:
            raise RuntimeError("compiled circuit parameter slots are not bound")
        return values[index]

    def values(self) -> tuple[torch.Tensor, ...]:
        values = self._values.get()
        if values is None:
            raise RuntimeError("compiled circuit parameter slots are not bound")
        return values


class SlotParameterMapping(Mapping[str, Any]):
    def __init__(
        self,
        constants: Mapping[str, Any],
        slots: Mapping[str, int],
        bindings: BuilderBindings,
    ) -> None:
        self._constants = dict(constants)
        self._slots = dict(slots)
        self._bindings = bindings
        self._keys = tuple(dict.fromkeys((*constants, *slots)))

    def __getitem__(self, key: str) -> Any:
        if key in self._slots:
            return self._bindings.value(self._slots[key])
        return self._constants[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)


def _snapshot_value(value: Any) -> Any:
    """Copy IR value containers and detach their PyTorch tensor leaves."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone().requires_grad_(value.requires_grad)
    if isinstance(value, ParameterExpression):
        return replace(value, args=tuple(_snapshot_value(item) for item in value.args))
    if isinstance(value, Mapping):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    return value


def detached_ir_snapshot(ir: CircuitIR) -> CircuitIR:
    """Materialize slots and detach tensors throughout supported IR values."""

    instructions = []
    for instruction in ir.instructions:
        instructions.append(
            replace(
                instruction,
                params=_snapshot_value(instruction.params),
                matrix=_snapshot_value(instruction.matrix),
                metadata=_snapshot_value(instruction.metadata),
            )
        )
    return replace(
        ir,
        instructions=tuple(instructions),
        observables=tuple(
            replace(
                observable,
                coefficient=_snapshot_value(observable.coefficient),
                metadata=_snapshot_value(observable.metadata),
            )
            for observable in ir.observables
        ),
        measurements=tuple(
            replace(measurement, metadata=_snapshot_value(measurement.metadata))
            for measurement in ir.measurements
        ),
        metadata=_snapshot_value(ir.metadata),
    )


@dataclass(frozen=True)
class CompiledInstruction:
    name: str
    wires: tuple[int, ...]
    params: Mapping[str, Any]
    matrix: Any | None
    metadata: Mapping[str, Any]
    parameter_slots: tuple[int, ...]
    parameter_constants: tuple[Any, ...]


__all__ = (
    "BuilderBindings",
    "CompiledInstruction",
    "SlotParameterMapping",
    "detached_ir_snapshot",
)
