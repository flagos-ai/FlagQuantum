"""Verify locked reversal order, every signed history and coherent rotation."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.bits.bit_rotation import bit_rotation_ir
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


def _certify(
    program: ReversiblePhaseProgram,
    data: int,
    expected: torch.Tensor,
    *,
    inputs: torch.Tensor | None = None,
) -> None:
    np = pytest.importorskip("numpy")
    events = sum(op.opcode == "hmr" for op in program.operations)
    selected = np.arange(1 << data) if inputs is None else inputs.numpy()
    basis = np.tile(selected, 1 << events)
    histories = np.repeat(np.arange(1 << events), len(selected))
    wires = [((basis >> (data - 1 - i)) & 1).astype(bool) for i in range(data)]
    wires += [np.zeros_like(wires[0]) for _ in range(program.n_qubits - data)]
    bits = [np.zeros_like(wires[0]) for _ in range(program.n_bits)]
    phase = np.zeros_like(wires[0])
    scopes = [np.ones_like(wires[0])]
    event = 0
    for op in program.operations:
        active = scopes[-1]
        if op.condition_bit is not None:
            active = active & bits[op.condition_bit]
        q = op.qubits
        if op.opcode == "push_condition":
            scopes.append(active.copy())
        elif op.opcode == "pop_condition":
            scopes.pop()
        elif op.opcode == "hmr":
            outcome = ((histories >> event) & 1).astype(bool)
            phase ^= active & wires[q[0]] & outcome
            wires[q[0]] &= ~active
            bits[op.result_bit] = np.where(active, outcome, bits[op.result_bit])
            event += 1
        elif op.opcode == "bit_store0":
            bits[op.result_bit].fill(False)
        elif op.opcode == "bit_invert":
            bits[op.result_bit] ^= active
        elif op.opcode == "neg":
            phase ^= active
        elif op.opcode in {"x", "cx", "ccx", "z", "cz", "ccz"}:
            for wire in q[:-1]:
                active = active & wires[wire]
            if op.opcode in {"z", "cz", "ccz"}:
                phase ^= active & wires[q[-1]]
            else:
                wires[q[-1]] ^= active
        elif op.opcode in {"swap", "cswap"}:
            if op.opcode == "cswap":
                active = active & wires[q[0]]
            difference = active & (wires[q[-2]] ^ wires[q[-1]])
            wires[q[-2]] ^= difference
            wires[q[-1]] ^= difference
        else:
            pytest.fail(f"Unexpected rotation leaf {op.opcode}")
    assert not np.any(phase)
    assert not any(np.any(wire) for wire in wires[data:])
    actual = sum(wires[i] * (1 << (data - 1 - i)) for i in range(data))
    np.testing.assert_array_equal(actual, np.tile(expected.numpy(), 1 << events))


def _run_and_certify(
    width: int,
    shift: int | tuple[int | bool | Bit, ...],
    *,
    controlled: bool,
    right: bool,
    quantum_shift: int = 0,
    classical_value: int = 0,
    classical_width: int = 0,
) -> None:
    data = width + quantum_shift + controlled
    control = data - 1 if controlled else None
    # A supplied work wire also tests that unused clean work remains zero.
    gates = bit_rotation_ir(
        tuple(range(width)),
        shift,
        control=control,
        right=right,
        clean=(data,),
        measurement_bit=classical_width,
    )
    prefix = tuple(
        ClassicalXor((), i, constant=bool(classical_value >> i & 1)).instruction()
        for i in range(classical_width)
    )
    program = lower_reversible_phase(CircuitIR(data + 1, (*prefix, *gates)))
    assert decode_kmx(encode_kmx(program)) == program
    basis = torch.arange(1 << data)
    expected_labels = []
    for label in basis.tolist():
        amount = (
            shift
            if type(shift) is int
            else sum(
                (
                    int(coefficient)
                    if isinstance(coefficient, bool)
                    else (
                        (classical_value >> coefficient.slot & 1)
                        if isinstance(coefficient, Bit)
                        else (label >> (data - 1 - coefficient) & 1)
                    )
                )
                << bit
                for bit, coefficient in enumerate(reversed(shift))
            )
        )
        amount = (-amount if right else amount) % width
        word = label >> (data - width)
        rotated = ((word << amount) | (word >> (width - amount))) & ((1 << width) - 1)
        if controlled and not label & 1:
            rotated = word
        expected_labels.append(label ^ ((word ^ rotated) << (data - width)))
    expected = torch.tensor(expected_labels)
    _certify(program, data, expected)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 2).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (data + 1), dtype=torch.complex128)
    initial[::2] = amplitudes
    final = torch.zeros_like(initial)
    final[expected * 2] = amplitudes
    circuit = DynamicCircuit(data + 1, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(data + 1, (*prefix, *gates))
    )
    result = run_dynamic(circuit, shots=32, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), final.expand(32, -1), atol=1e-12, rtol=0
    )
    for bit in range(classical_width):
        assert torch.all(result.classical_bits[:, bit] == (classical_value >> bit & 1))


@pytest.mark.parametrize(
    "width,shift", [(1, -1), (2, 1), (3, -11), (3, 0), (3, 1), (5, 4)]
)
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("right", [False, True])
def test_constant_rotation(
    width: int, shift: int, controlled: bool, right: bool
) -> None:
    _run_and_certify(width, shift, controlled=controlled, right=right)


@pytest.mark.parametrize("mode", ["quantum", "classical", "mixed", "constant_mixed"])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("right", [False, True])
def test_variable_rotation(mode: str, controlled: bool, right: bool) -> None:
    width = 3
    if mode == "quantum":
        shift, quantum, classical = (3, 4), 2, 0
    elif mode == "classical":
        shift, quantum, classical = (Bit(1), Bit(0)), 0, 2
    elif mode == "mixed":
        shift, quantum, classical = (Bit(0), 3), 1, 1
    else:
        shift, quantum, classical = (True, Bit(0), 3), 1, 1
    for value in range(1 << classical):
        _run_and_certify(
            width,
            shift,
            controlled=controlled,
            right=right,
            quantum_shift=quantum,
            classical_value=value,
            classical_width=classical,
        )


def test_source_operand_order_and_merged_control_cost() -> None:
    gates = bit_rotation_ir((3, 2, 1, 0), 1)
    assert [(gate.name, gate.wires) for gate in gates] == [
        ("swap", (3, 0)),
        ("swap", (2, 1)),
        ("swap", (3, 1)),
    ]
    gates = bit_rotation_ir(
        (3, 2, 1, 0), (5, 4), control=6, clean=(7,), measurement_bit=0
    )
    program = lower_reversible_phase(CircuitIR(8, gates))
    assert Counter(op.opcode for op in program.operations) == {
        "ccx": 3,
        "cswap": 4,
        "hmr": 3,
        "cz": 3,
        "cx": 2,
    }


def test_rotation_rejects_invalid_ports_and_workspace() -> None:
    with pytest.raises(ValueError, match="fresh result slot"):
        bit_rotation_ir((0, 1), (2,), control=3)
    with pytest.raises(ValueError, match="fresh result slot"):
        bit_rotation_ir((0, 1), (Bit(0), 2), control=3, clean=(4,), measurement_bit=0)
    with pytest.raises(ValueError, match="60 bits"):
        bit_rotation_ir((0,), (False,) * 61)
    with pytest.raises(ValueError, match="64-bit"):
        bit_rotation_ir((0,), 1 << 63)
    with pytest.raises(ValueError, match="minimum signed"):
        bit_rotation_ir((0, 1, 2), -(1 << 63), right=True)
    assert bit_rotation_ir((), 2) == ()
    assert bit_rotation_ir((0, 1), ()) == ()
