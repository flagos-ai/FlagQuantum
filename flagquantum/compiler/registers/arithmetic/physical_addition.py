# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_iadd_classical.cc and CircuitBuilder.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/adders/gen_iadd_classical.cc

"""Export mixed addition with its physical call and feedback boundaries."""

from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import PhysicalTrace, lower_physical_trace
from .carry_parity import _Coefficient, carry_parity_ir
from .measured_carry_addition import measured_carry_addition_ir
from .mixed_addition import mixed_addition_ir
from .vented_addition import vented_addition_ir


def _source_tag(name: str, control: int | bool) -> str:
    return f"c_{name}" if type(control) is int else name


def physical_carry_parity(
    source: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    target: tuple[int, ...],
    *,
    carry_in: _Coefficient,
    control: int | bool,
    clean: tuple[int, ...],
    classical_width: int,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export the standalone carry XOR with its source mark and reset."""
    gates = carry_parity_ir(
        source,
        offset,
        target,
        carry_in=carry_in,
        control=control,
        clean=clean,
        measurement_bit=classical_width,
        emit_resets=True,
    )
    mark = ComponentCall(_source_tag("ixor_carries", control), 0, len(gates))
    return lower_physical_trace(gates, classical_width, 0, PhysicalTrace(calls=[mark]))


def physical_vented_addition(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    *,
    clean: tuple[int, ...],
    carry_copy: tuple[int | None, ...],
    vent_bits: tuple[int, ...],
    carry_in: _Coefficient,
    control: int | bool,
    classical_width: int,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export standalone vented carries with caller-bound result inputs."""
    export = PhysicalTrace()
    gates = vented_addition_ir(
        target,
        offset,
        clean=clean,
        carry_copy=carry_copy,
        vent_bits=vent_bits,
        carry_in=carry_in,
        control=control,
        measurement_bit=classical_width,
        emit_resets=True,
        feedback=export.feedback,
        inline_guards=export.inline_guards,
    )
    promoted = len(carry_copy) > len(target)
    padded = len(offset) < len(target) + int(promoted)
    calls: tuple[ComponentCall, ...] = ()
    for _ in range(1 + int(promoted) + int(padded)):
        tag = _source_tag("iadd_classical_vented", control)
        calls = (ComponentCall(tag, 0, len(gates), children=calls),)
    export.calls.extend(calls)
    return lower_physical_trace(gates, classical_width, 0, export)


def physical_mixed_addition(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    *,
    clean: tuple[int, ...],
    dirty: tuple[int, ...],
    control: int | bool,
    classical_width: int,
    carry_in: _Coefficient = False,
    simple: bool = False,
    subtract: bool = False,
    minimize_qubits: bool = False,
    btol: float = float("inf"),
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export mixed addition, subtraction or the simple carry schedule."""
    if simple and subtract:
        raise ValueError("The measured-carry source exports only simple addition")
    export = PhysicalTrace()
    if simple:
        gates = measured_carry_addition_ir(
            target,
            offset,
            clean=clean,
            control=control,
            carry_in=carry_in,
            measurement_bit=classical_width,
            emit_resets=True,
        )
        mark = ComponentCall(
            _source_tag("iadd_classical_simple", control), 0, len(gates)
        )
        export.calls.append(mark)
    else:
        gates = mixed_addition_ir(
            target,
            offset,
            clean=clean,
            dirty=dirty,
            control=control,
            carry_in=carry_in,
            classical_bits=tuple(
                range(classical_width, classical_width + len(target) + 1)
            ),
            minimize_qubits=minimize_qubits,
            btol=btol,
            export=export,
            subtract=subtract,
        )
        if subtract:
            tag = _source_tag("isub_classical", control)
            export.calls = [
                ComponentCall(tag, 0, len(gates), children=tuple(export.calls))
            ]
    return lower_physical_trace(gates, classical_width, 0, export)


def physical_classical_addition(
    target: tuple[int, ...],
    offset: tuple[_Coefficient, ...],
    *,
    clean: tuple[int, ...],
    dirty: tuple[int, ...],
    control: int | bool,
    carry_in: int | None,
    minimize_qubits: bool,
    btol: float,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Bind the generator command's classical offset and optional quantum carry."""
    return physical_mixed_addition(
        target,
        offset,
        clean=clean,
        dirty=dirty,
        control=control,
        classical_width=len(target),
        carry_in=False if carry_in is None else carry_in,
        simple=carry_in is not None,
        minimize_qubits=minimize_qubits,
        btol=btol,
    )
