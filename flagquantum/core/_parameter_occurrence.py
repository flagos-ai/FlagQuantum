"""Which declared gate occurrence a flat parameter index controls.

A parameter-shift route reads one rule per user parameter: the displacement the
declaration prescribes and the coefficient that comes with it. The rule applies
to a gate *angle*, so the route first finds the one gate parameter that index
controls and then reads that occurrence's declared rule. Every route built on a
declared rule asks the same question -- ``parameter_shift_gradient`` and
``parameter_shift_hessian`` answer it by probing the circuit builder, and
``batched_parameter_shift_gradient`` by comparing the pair it built -- so the
answer is derived here once rather than restated in each route.

Two facts are refused instead of answered: a parameter that changes the
program's structure, and a parameter that does not enter its gate as the angle
the rule applies to. ``rx(0, theta=2 * scale)`` is the second kind, and a rule
shifted by the declared displacement would differentiate there at the wrong
point.

The profile is not a second source of truth for the declaration: it holds
opcode and parameter names, and every rule is read from
:mod:`flagquantum.core.operator_schema` where it is used.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .ir import ensure_circuit_ir
from .operator_schema import OPERATOR_SCHEMAS

# The displacement one probe moves a parameter by. It is not a shift rule's
# displacement: it only has to move the angle far enough that the probe
# comparison can see which occurrence changed, and that comparison checks it.
_PROBE_DISPLACEMENT = 1.0


def _scalar_parameter(value: Any) -> float:
    tensor = (
        value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    )
    if tensor.numel() != 1 or tensor.is_complex():
        raise ValueError("batched parameter shift requires scalar real gate parameters")
    scalar = float(tensor.cpu())
    if not math.isfinite(scalar):
        raise ValueError("batched parameter shift requires finite gate parameters")
    return scalar


def _circuit_profile_of(program: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Describe a program as its static structure plus its scalar parameters."""

    ir = ensure_circuit_ir(program)
    if any(item.matrix is not None for item in ir.instructions):
        raise ValueError("parameter shift does not support custom matrices")
    structure = (
        ir.n_wires,
        tuple((item.name, item.wires, tuple(item.params)) for item in ir.instructions),
    )
    values = tuple(
        (item.name, name, _scalar_parameter(value))
        for item in ir.instructions
        for name, value in item.params.items()
    )
    return structure, values


def _changed_occurrence(
    base: tuple[tuple[Any, ...], tuple[Any, ...]],
    probes: Sequence[tuple[tuple[Any, ...], tuple[Any, ...]]],
    *,
    parameter_index: int,
) -> tuple[str, str]:
    """Return the gate angle a probe pair moved by the probe displacement.

    The shift rule is applied to the gate's angle, so a user parameter has to be
    that angle. A parameter that scales its gate -- ``rx(0, theta=2 * scale)`` --
    moves the angle by twice the probe displacement, and shifting the *user*
    parameter by the rule's shift would then differentiate at the wrong point.
    That is refused instead of being answered with the angle's derivative.
    """

    base_structure, base_values = base
    if any(structure != base_structure for structure, _ in probes):
        raise ValueError(
            f"parameter {parameter_index} changes circuit structure; parameter "
            "shift requires a static circuit"
        )
    changed = [
        (base_item, probed_items)
        for base_item, *probed_items in zip(
            base_values, *(values for _, values in probes), strict=True
        )
        if any(probed != base_item for probed in probed_items)
    ]
    if len(changed) != 1:
        raise ValueError(
            f"parameter {parameter_index} must control exactly one gate "
            f"parameter; found {len(changed)}"
        )
    (opcode, name, value), probed_values = changed[0]
    displacements = [float(probed[2]) - float(value) for probed in probed_values]
    if not all(
        math.isclose(abs(displacement), _PROBE_DISPLACEMENT, rel_tol=1e-4)
        for displacement in displacements
    ):
        raise ValueError(
            f"parameter {parameter_index} does not enter {opcode}.{name} as its "
            f"angle: a probe of {_PROBE_DISPLACEMENT} moved the angle by "
            f"{displacements}; enter its gate angle directly"
        )
    return opcode, name


def _occurrence_shift_rule(
    profile: tuple[tuple[Any, ...], tuple[Any, ...]],
    circuit_builder: Callable[[torch.Tensor], Any],
    base: torch.Tensor,
    flat: torch.Tensor,
    index: int,
) -> tuple[tuple[float, float], ...]:
    """Find the gate parameter one flat parameter index controls, then its rule."""

    probes = []
    for displacement in (_PROBE_DISPLACEMENT, -_PROBE_DISPLACEMENT):
        probed = flat.clone()
        probed[index] += displacement
        probes.append(_circuit_profile_of(circuit_builder(probed.reshape_as(base))))
    occurrence = _changed_occurrence(profile, probes, parameter_index=index)
    opcode, name = occurrence
    schema = OPERATOR_SCHEMAS.get(opcode)
    if schema is None:
        raise ValueError(
            f"parameter {index} controls unknown opcode {opcode!r}, which "
            "declares no derivative rule"
        )
    try:
        return schema.shift_rule(name)
    except ValueError as error:
        raise ValueError(
            f"parameter {index} controls {opcode}.{name}, which cannot be "
            f"differentiated: {error}"
        ) from error


def _batch_occurrence_coefficient(
    base: tuple[tuple[Any, ...], tuple[Any, ...]],
    plus: tuple[tuple[Any, ...], tuple[Any, ...]],
    minus: tuple[tuple[Any, ...], tuple[Any, ...]],
    *,
    shift: float,
    parameter_index: int,
) -> float:
    """The declared coefficient for the one gate angle a parameter moves."""

    base_structure, base_values = base
    plus_structure, plus_values = plus
    minus_structure, minus_values = minus
    if not base_structure == plus_structure == minus_structure:
        raise ValueError(
            f"parameter {parameter_index} changes circuit structure; "
            "batched parameter shift requires a static circuit"
        )
    changed = []
    for base_item, plus_item, minus_item in zip(
        base_values, plus_values, minus_values, strict=True
    ):
        opcode, name, base_value = base_item
        _, _, plus_value = plus_item
        _, _, minus_value = minus_item
        if not (
            math.isclose(plus_value, base_value, rel_tol=0.0, abs_tol=1e-7)
            and math.isclose(minus_value, base_value, rel_tol=0.0, abs_tol=1e-7)
        ):
            changed.append((opcode, name, base_value, plus_value, minus_value))

    if len(changed) != 1:
        raise ValueError(
            f"parameter {parameter_index} must control exactly one gate occurrence; "
            f"found {len(changed)}"
        )
    opcode, name, base_value, plus_value, minus_value = changed[0]
    schema = OPERATOR_SCHEMAS.get(opcode)
    if schema is None:
        raise ValueError(
            f"parameter {parameter_index} controls unknown opcode {opcode!r}, "
            "which declares no derivative rule"
        )
    try:
        rule = schema.shift_rule(name)
    except ValueError as error:
        raise ValueError(
            f"parameter {parameter_index} controls {opcode}.{name}, which cannot be "
            f"differentiated: {error}"
        ) from error
    if len(rule) != 2:
        declared = schema.parameter_frequencies[schema.parameters.index(name)]
        raise ValueError(
            f"parameter {parameter_index} controls {opcode}.{name}, which declares "
            f"frequencies {declared} and needs {len(rule) // 2} evaluation pairs; "
            "batched parameter shift sends one pair per parameter"
        )
    coefficient, rule_shift = rule[0]
    if not math.isclose(abs(rule_shift), shift, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"parameter {parameter_index} must be shifted by {abs(rule_shift)} to "
            f"differentiate {opcode}.{name}, got {shift}"
        )
    if not (
        math.isclose(plus_value - base_value, shift, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(minus_value - base_value, -shift, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError(
            f"parameter {parameter_index} must enter its gate angle directly"
        )
    return coefficient
