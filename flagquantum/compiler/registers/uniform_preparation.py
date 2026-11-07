# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PrepareUniformSuperposition.decompose_from_registers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/prepare_uniform_superposition.py
# Changes: bound IR, shared clean work, and exact native CH per source Hadamard.

"""Prepare uniform amplitudes over a concrete prefix of a binary register."""

from math import acos

from ...core.ir import Instruction
from .._component_trace.resource_model import ComponentCost, GateCost
from .bitwise import _validate_wires
from .comparison import less_than_constant_ir, less_than_constant_workspace_size
from .control_predicate_resource_model import control_predicate_cost
from .multi_control import multi_and_ir


def _odd_factor(count: int) -> int:
    if type(count) is not int or count < 1:
        raise ValueError("Uniform preparation count must be a positive integer")
    return count // (count & -count)


def uniform_preparation_requirements(
    count: int, controls: int = 0
) -> tuple[int, int, int]:
    """Return the target width, clean work and reusable classical slots."""
    odd = _odd_factor(count)
    if type(controls) is not int or controls < 0:
        raise ValueError("Uniform preparation control count must be nonnegative")
    width = (count - 1).bit_length()
    if width == 0:
        return 0, 0, 0
    if odd == 1:
        work = max(0, controls - 1)
        return width, work, work
    high = (odd - 1).bit_length()
    compare = less_than_constant_workspace_size(high, odd)
    reflection = high + controls - 1
    return width, max(1 + compare, reflection), max(high, reflection)


def _initial_hadamards(
    target: tuple[int, ...],
    controls: tuple[int, ...],
    values: tuple[int, ...],
    workspace: tuple[int, ...],
    slots: tuple[int, ...],
) -> tuple[Instruction, ...]:
    if not controls:
        return tuple(Instruction("h", (qubit,)) for qubit in target)
    control = controls[0]
    prepare: tuple[Instruction, ...] = (
        (Instruction("x", controls),) if values == (0,) else ()
    )
    erase: tuple[Instruction, ...] = prepare
    if len(controls) > 1:
        work = workspace[: len(controls) - 1]
        control = work[-1]
        prepare = multi_and_ir(controls, values, work)
        erase = multi_and_ir(
            controls, values, work, classical_bits=slots[: len(work)], uncompute=True
        )
    return tuple(
        gate
        for qubit in target
        for gate in (*prepare, Instruction("ch", (control, qubit)), *erase)
    )


def uniform_preparation_ir(
    count: int,
    target: tuple[int, ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    controls: tuple[int, ...] = (),
    control_values: tuple[int, ...] = (),
    adjoint: bool = False,
) -> tuple[Instruction, ...]:
    """Apply the pinned preparation or adjoint on caller-bound big-endian qubits.

    With a clean target the active branch prepares the first count amplitudes
    with the source phase. The inactive branch leaves a clean target unchanged.
    For other targets the complete source circuit defines the action. All
    borrowed work returns to zero and measurement slots may be overwritten.
    """
    width, work, slots = uniform_preparation_requirements(count, len(controls))
    if type(adjoint) is not bool:
        raise ValueError("Uniform preparation adjoint must be boolean")
    if len(controls) != len(control_values) or any(
        type(value) is not int or value not in (0, 1) for value in control_values
    ):
        raise ValueError("Uniform preparation requires one zero/one value per control")
    if len(target) != width or len(workspace) != work:
        raise ValueError(
            f"Uniform preparation requires {width} target and {work} clean work qubits"
        )
    if (
        len(classical_bits) != slots
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
        or len(set(classical_bits)) != slots
    ):
        raise ValueError(
            "Uniform preparation requires distinct nonnegative classical slots"
        )
    wires = (*controls, *target, *workspace)
    if wires:
        _validate_wires(wires)
    if not target:
        return ()

    # Hadamards on different targets commute, but retain the adjoint order.
    initial = _initial_hadamards(
        target[::-1] if adjoint else target,
        controls,
        control_values,
        workspace,
        classical_bits,
    )
    odd = _odd_factor(count)
    if odd == 1:
        return initial

    high_width = (odd - 1).bit_length()
    high = target[:high_width]
    theta = acos(1 - (1 << (high_width - 1)) / odd)
    angle = -theta if adjoint else theta
    compare_size = less_than_constant_workspace_size(high_width, odd)
    flag = workspace[0]
    comparison = less_than_constant_ir(
        high, odd, flag, workspace[1 : 1 + compare_size], classical_bits[:high_width]
    )
    threshold_phase = (
        *comparison,
        Instruction("rz", (flag,), params={"theta": angle}),
        *comparison,
    )

    reflect_controls = (*high, *controls)
    reflect_values = (0,) * high_width + control_values
    prefix = workspace[: len(reflect_controls) - 1]
    zero_phase = (
        *multi_and_ir(reflect_controls, reflect_values, prefix),
        Instruction("rz", (prefix[-1],), params={"theta": angle}),
        *multi_and_ir(
            reflect_controls,
            reflect_values,
            prefix,
            classical_bits=classical_bits[: len(prefix)],
            uncompute=True,
        ),
    )
    h_wires = high[::-1] if adjoint else high
    h = tuple(Instruction("h", (wire,)) for wire in h_wires)
    if adjoint:
        return (*h, *zero_phase, *h, *threshold_phase, *initial)
    return (*initial, *threshold_phase, *h, *zero_phase, *h)


def source_uniform_preparation_cost(
    count: int, *, legacy_shims: bool = False
) -> ComponentCost:
    """Estimate the uncontrolled source decomposition without emitting gates."""
    if type(legacy_shims) is not bool:
        raise ValueError("Uniform preparation cost profile must be a boolean")
    width, work, _ = uniform_preparation_requirements(count)
    odd = _odd_factor(count)
    if odd == 1:
        gates = GateCost(clifford=width)
    else:
        high = (odd - 1).bit_length()
        # Retain the source's two LessThanConstant calls and zero predicate.
        # Profile charges belong to the shared And query, not emitted gates.
        erase = control_predicate_cost(
            (1, 1), uncompute=True, legacy_shims=legacy_shims
        )
        comparison_cliffords = (
            2 * high + 3 * odd.bit_count() + 2 + high * erase.gates.clifford
        )
        forward = control_predicate_cost((0,) * high, legacy_shims=legacy_shims)
        reverse = control_predicate_cost(
            (0,) * high, uncompute=True, legacy_shims=legacy_shims
        )
        reflection_cliffords = forward.gates.clifford + reverse.gates.clifford
        ands = 3 * high - 1
        gates = GateCost(
            and_bloq=ands,
            clifford=width + 2 * high + 2 * comparison_cliffords + reflection_cliffords,
            rotation=2,
            measurement=ands,
        )
    return ComponentCost(f"PrepareUniformSuperposition({count})", width + work, gates)
