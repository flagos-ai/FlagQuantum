"""Reusable builder-program state for ``Module`` compilation."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

from ..core.ir import CircuitIR, Instruction
from ..core.parameters import ParameterExpression

if TYPE_CHECKING:
    from ..circuit import Circuit


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

    def _snapshot_parameters(self, values: tuple[torch.Tensor, ...]) -> dict[str, Any]:
        return {
            key: (
                values[self._slots[key]] if key in self._slots else self._constants[key]
            )
            for key in self._keys
        }


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


def can_defer_ir_snapshot(program: Any, template: CircuitIR) -> bool:
    """Replay only pure parameter graphs without lifted mutable state."""
    if not isinstance(program, torch.fx.GraphModule):
        return False
    if template.observables or template.measurements:
        return False
    if any(
        instruction.matrix is not None
        or instruction.metadata
        or (
            isinstance(instruction.params, SlotParameterMapping)
            and any(
                type(value) not in (float, int, str, bool, complex, type(None))
                for value in instruction.params._constants.values()
            )
        )
        for instruction in template.instructions
    ):
        return False
    for node in program.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.op != "call_function" or not isinstance(
            node.target, torch._ops.OpOverload
        ):
            return False
        if node.target._schema.is_mutable or any(
            tag in node.target.tags
            for tag in (
                torch.Tag.nondeterministic_seeded,
                torch.Tag.nondeterministic_bitwise,
            )
        ):
            return False
    return True


def ir_snapshot_input_budget(template: CircuitIR) -> int:
    """Bound retained inputs by the actual scalar or batched consumed slots."""
    slot_bytes = sum(
        value.numel() * value.element_size()
        for instruction in template.instructions
        for value in instruction.params.values()
        if isinstance(value, torch.Tensor)
    )
    return max(512, 4 * slot_bytes)


def deferred_ir_snapshot(
    program: Callable[..., Any],
    template: CircuitIR,
    inputs: tuple[torch.Tensor, ...],
    slot_count: int,
) -> Callable[[], CircuitIR]:
    """Freeze consumed inputs now and materialize diagnostic IR only when read."""
    frozen = tuple(
        value.detach().clone().requires_grad_(value.requires_grad) for value in inputs
    )
    grad_enabled = torch.is_grad_enabled()
    inference_enabled = torch.is_inference_mode_enabled()

    def materialize() -> CircuitIR:
        with (
            torch.inference_mode(inference_enabled),
            torch.set_grad_enabled(grad_enabled),
        ):
            result = program(*frozen)
            values = (result,) if isinstance(result, torch.Tensor) else tuple(result)
            if len(values) != slot_count:
                raise RuntimeError("compiled builder parameter slot count changed")
            instructions = tuple(
                (
                    replace(
                        instruction,
                        params=instruction.params._snapshot_parameters(values),
                    )
                    if isinstance(instruction.params, SlotParameterMapping)
                    else instruction
                )
                for instruction in template.instructions
            )
            return detached_ir_snapshot(replace(template, instructions=instructions))

    return materialize


@dataclass(frozen=True)
class CompiledInstruction(Instruction):
    """Validated template instruction whose parameter slots remain live."""

    params: Mapping[str, Any] = field()
    matrix: Any | None = field()
    metadata: Mapping[str, Any] = field()
    parameter_slots: tuple[int, ...]
    parameter_constants: tuple[Any, ...]

    def __post_init__(self) -> None:
        """Keep validated template IR slots lazy until the builder binds them."""


__all__ = (
    "BuilderBindings",
    "CompiledInstruction",
    "SlotParameterMapping",
    "detached_ir_snapshot",
)


def _builder_tensor_inputs(
    parameters: Any, inputs: torch.Tensor | None
) -> tuple[tuple[torch.Tensor, ...], tuple[str, ...], bool]:
    if isinstance(parameters, (Mapping, torch.nn.ParameterDict)):
        names = tuple(parameters)
        tensors = tuple(parameters[name] for name in names)
    else:
        names = ()
        tensors = (parameters,)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("compiled circuit builder parameters must be tensors")
    return (
        tensors + (() if inputs is None else (inputs,)),
        names,
        inputs is not None,
    )


def _refuse_non_finite_bound_parameters(circuit: Circuit) -> None:
    """Apply the canonical angle domain to the values this call just bound.

    A compiled builder is traced rather than executed, so its gates are
    constructed from tracers and :func:`flagquantum.core.ir._normalize_angle`
    defers the magnitude check: a tracer has no value yet. The values arrive
    here, and without this the program completes with ``tensor([nan])`` while
    the same circuit built directly refuses with ``IRValidationError``.

    The type and dtype rules are static, so they already ran at trace time and
    only the magnitude was deferred. This therefore asks one question -- is
    each bound value finite -- in the cheapest form the value allows: a
    scalar becomes a Python float, a batched angle asks its own tensor.
    Grouping every value into one tensor first was measured slower than
    either, because the grouping costs more than the check.

    Anything that is not a real tensor goes through the canonical rule
    unchanged, and every refusal is raised by that rule, so the domain and
    the error contract stay Core's rather than becoming a second finite check
    per simulator.
    """

    from math import isfinite

    for instruction in circuit._instructions:
        for parameter, value in instruction.params.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                detached = value.detach()
                finite = (
                    isfinite(float(detached))
                    if detached.numel() == 1
                    else bool(detached.isfinite().all())
                )
                if finite:
                    continue
            from ..core.ir import _normalize_angle

            _normalize_angle(value, opcode=instruction.name, parameter=parameter)
