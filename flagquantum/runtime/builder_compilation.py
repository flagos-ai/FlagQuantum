"""Reusable builder-program state for ``Module`` compilation."""

from __future__ import annotations

from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from ..core.ir import CircuitIR


class BuilderBindings:
    """Task-local dynamic tensors consumed by one reusable circuit program."""

    def __init__(self) -> None:
        self._values: ContextVar[tuple[torch.Tensor, ...]] = ContextVar(
            "flagquantum_builder_bindings", default=()
        )

    def bind(self, values: tuple[torch.Tensor, ...]) -> None:
        self._values.set(values)

    def clear(self) -> None:
        self._values.set(())

    def value(self, index: int) -> torch.Tensor:
        values = self._values.get()
        if not values:
            raise RuntimeError("compiled circuit parameter slots are not bound")
        return values[index]

    def values(self) -> tuple[torch.Tensor, ...]:
        values = self._values.get()
        if not values:
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


def detached_ir_snapshot(ir: CircuitIR) -> CircuitIR:
    """Materialize dynamic parameter slots without retaining an autograd graph."""

    instructions = []
    for instruction in ir.instructions:
        params = {
            name: (
                value.detach().clone().requires_grad_(value.requires_grad)
                if isinstance(value, torch.Tensor)
                else value
            )
            for name, value in instruction.params.items()
        }
        instructions.append(replace(instruction, params=params))
    return replace(ir, instructions=tuple(instructions))


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
