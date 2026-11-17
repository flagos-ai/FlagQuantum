# SPDX-License-Identifier: Apache-2.0
# Layout adapted from locked Kickmix main_generate_circuit.cc.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/cmd/main_generate_circuit.cc

"""Bind physical register layouts to native arithmetic and table emitters."""

from ...core.ir import CircuitIR
from .._component_trace.trace import ComponentCall, ComponentTrace
from ..artifacts.reversible_phase import ReversiblePhaseProgram
from ..formats.kmb import _MAX_WIRES, KmxRegister
from ..formats.kmx_block_marks import encode_kmx_block_marks
from ..formats.kmx_circuit import KmxCircuit, read_kmx_circuit
from ..lowering.reversible_phase import lower_reversible_phase
from .arithmetic.inplace_addition import inplace_addition_ir
from .comparison.physical_comparison import physical_less_than
from .finite_field.coefficients import ClassicalFieldCoefficient
from .lookup.measured_unlookup import measured_unlookup_operations
from .lookup.recursive_lookup import recursive_lookup_operations


def _quantum_register(wires: tuple[int, ...]) -> KmxRegister:
    return KmxRegister("", tuple(("q", wire) for wire in wires))


def _physical_circuit(
    registers: tuple[KmxRegister, ...],
    program: ReversiblePhaseProgram,
    calls: list[ComponentCall],
) -> KmxCircuit:
    # The raw file carries explicit register declarations instead of a program
    # header. Unused reserved workspace is inferred only if a gate uses it.
    _, _, body = encode_kmx_block_marks(program, calls).partition("\n")
    declarations = (
        line
        for index, register in enumerate(registers)
        for line in register.to_kmx(index)
    )
    return read_kmx_circuit("\n".join((*declarations, body)))


def generate_addition(
    width: int,
    *,
    clean_count: int,
    dirty_count: int,
    controlled: bool,
    minimize_qubits: bool,
) -> KmxCircuit:
    """Generate an in-place adder with source-ordered registers and work."""
    target = tuple(range(width))
    offset = tuple(range(width, 2 * width))
    clean = tuple(range(2 * width, 2 * width + clean_count))
    control_wire = 2 * width + clean_count
    enable = (control_wire,) if controlled else ()
    dirty_start = control_wire + len(enable)
    dirty = tuple(range(dirty_start, dirty_start + dirty_count))
    gates = inplace_addition_ir(
        target[::-1],
        offset[::-1],
        clean=clean,
        dirty=dirty,
        control=control_wire if controlled else True,
        measurement_bit=0,
        minimize_qubits=minimize_qubits,
        emit_resets=True,
    )
    width_with_work = max(
        1,
        dirty_start + dirty_count,
        1 + max((wire for gate in gates for wire in gate.wires), default=-1),
    )
    program = lower_reversible_phase(CircuitIR(width_with_work, gates))
    registers: tuple[KmxRegister, ...] = (
        _quantum_register(target),
        _quantum_register(offset),
    )
    if controlled:
        registers += (_quantum_register(enable),)
    calls = [ComponentCall("iadd", 0, len(program.operations))]
    return _physical_circuit(registers, program, calls)


def generate_comparison(
    width: int,
    *,
    clean_count: int,
    controlled: bool,
    minimize_qubits: bool,
    btol: float,
) -> KmxCircuit:
    """Generate a less-than comparison with the measured and narrow routes."""
    left = tuple(range(width))
    right = tuple(range(width, 2 * width))
    clean = tuple(range(2 * width, 2 * width + clean_count))
    enable = 2 * width + clean_count
    target = enable + int(controlled)
    program, call = physical_less_than(
        left[::-1],
        right[::-1],
        target,
        clean,
        control=enable if controlled else True,
        btol=btol,
        minimize_qubits=minimize_qubits,
    )
    registers: tuple[KmxRegister, ...] = (
        _quantum_register(left),
        _quantum_register(right),
    )
    if controlled:
        registers = (*registers, _quantum_register((enable,)))
    return _physical_circuit(
        (*registers, _quantum_register((target,))), program, [call]
    )


def generate_table(
    address_width: int,
    output_width: int,
    *,
    clean_count: int,
    controlled: bool,
    erase: bool,
) -> KmxCircuit:
    """Generate a full lookup or measured erasure for a declared classical table."""
    if erase and controlled:
        raise ValueError("--controlled not supported with unlookup")
    # Both generators are empty for a zero-width output.
    if output_width and address_width > 30:
        raise ValueError("Portable table generation supports at most 30 address bits")
    rows = (1 << address_width) if output_width else 0
    table_width = rows * output_width
    if table_width + (rows + 1 if erase and rows else 1) > _MAX_WIRES:
        raise ValueError("Table exceeds the physical bit limit")

    address = tuple(range(address_width))
    output = tuple(range(address_width, address_width + output_width))
    clean_start = address_width + output_width
    clean = tuple(range(clean_start, clean_start + clean_count))
    control_wire = clean_start + clean_count
    registers: tuple[KmxRegister, ...] = (
        _quantum_register(address),
        _quantum_register(output),
        KmxRegister("", tuple(("b", bit) for bit in range(table_width))),
    )
    if controlled:
        registers += (_quantum_register((control_wire,)),)
    table = tuple(
        tuple(
            ClassicalFieldCoefficient(bit)
            for bit in reversed(range(start, start + output_width))
        )
        for start in range(0, table_width, max(1, output_width))
    )
    trace = ComponentTrace()
    measurement = table_width + (rows if erase else 0)
    if erase:
        operations = measured_unlookup_operations(
            table,
            address[::-1],
            output[::-1],
            clean,
            tuple(range(table_width, measurement)),
            measurement_bit=measurement,
            trace=trace,
        )
    else:
        operations = recursive_lookup_operations(
            table,
            address[::-1],
            output[::-1],
            clean,
            control=control_wire if controlled else None,
            measurement_bit=measurement,
            trace=trace,
        )
    program = ReversiblePhaseProgram(
        max(1, control_wire + int(controlled)),
        measurement + 1,
        operations,
        tuple(range(table_width)),
    )
    return _physical_circuit(registers, program, trace.calls)
