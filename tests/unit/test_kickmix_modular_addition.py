"""Verify locked modular schedules, active histories and cleanup."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.finite_field import ClassicalFieldCoefficient
from flagquantum.compiler.registers.kickmix_modular_addition import (
    kickmix_modular_addition_program,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


def _modulus(value: int, width: int) -> tuple[bool, ...]:
    return tuple(bool(value & (1 << bit)) for bit in reversed(range(width)))


@pytest.mark.parametrize("subtract", [False, True])
@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("width,modulus", [(2, 3), (3, 5)])
def test_bound_modular_operation_matches_complex_amplitudes(
    width: int, modulus: int, classical: bool, subtract: bool
) -> None:
    control = 2 * width
    work = tuple(range(control + 1, control + width + 4))
    program = kickmix_modular_addition_program(
        tuple(range(width)),
        tuple(range(width, control)),
        (
            tuple(ClassicalFieldCoefficient(i) for i in range(width))
            if classical
            else _modulus(modulus, width)
        ),
        clean=work,
        control=control,
        subtract=subtract,
    )
    assert decode_kmx(encode_kmx(program)) == program
    assert program.input_bits == (tuple(range(width)) if classical else ())
    values = tuple(product(range(modulus), range(modulus), range(2)))
    vector = torch.arange(1, len(values) + 1, dtype=torch.float64)
    amplitudes = torch.complex(vector, 3 * vector - 5)
    amplitudes /= amplitudes.norm()

    def address(a: int, b: int, enable: int) -> int:
        return (((a << width) | b) << 1 | enable) << len(work)

    initial = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for amplitude, (a, b, enable) in zip(amplitudes, values, strict=True):
        initial[address(a, b, enable)] = amplitude
        signed = -b if subtract else b
        expected[address((a + signed * enable) % modulus, b, enable)] = amplitude
    inputs = tuple(
        ClassicalXor((), i + 2, constant=bit).instruction()
        for i, bit in enumerate(_modulus(modulus, width))
        if classical
    )
    circuit = DynamicCircuit(
        program.n_qubits, inputs=initial, dtype=torch.complex128
    ).append(
        FieldOperation(
            program.n_qubits,
            (*inputs, *bind_reversible_phase_execution(program, classical_offset=2)),
        )
    )
    result = run_dynamic(circuit, shots=64, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(64, -1), expected.expand(64, -1), atol=1e-12, rtol=0
    )
    if classical:
        assert (
            result.classical_bits[:, 2 : 2 + width].tolist()
            == [list(map(int, _modulus(modulus, width)))] * 64
        )


def _certify_all_histories(
    program: ReversiblePhaseProgram, *, width: int = 2, modulus: int = 3
) -> tuple[int, ...]:
    # Enumerate every outcome, including unused outcomes at disabled effects.
    # Those redundant histories all implement the same signed basis map.
    np = pytest.importorskip("numpy")
    events = sum(op.opcode in {"hmr", "r"} for op in program.operations)
    size = 2 * modulus * modulus
    labels = np.arange((1 << events) * size, dtype=np.int64)
    histories, basis = labels // size, labels % size
    lhs, rhs, control = basis // (2 * modulus), (basis // 2) % modulus, basis % 2
    wires = [((lhs >> i) & 1).astype(bool) for i in range(width)]
    wires += [((rhs >> i) & 1).astype(bool) for i in range(width)]
    wires += [control.astype(bool)]
    wires += [
        np.zeros_like(wires[0]) for _ in range(program.n_qubits - (2 * width + 1))
    ]
    bits = [np.zeros_like(wires[0]) for _ in range(program.n_bits)]
    scopes = [np.ones_like(wires[0])]
    sign = np.zeros_like(wires[0])
    event = 0
    for op in program.operations:
        active = scopes[-1]
        if op.condition_bit is not None:
            active = active & bits[op.condition_bit]
        q = op.qubits
        if op.opcode == "push_condition":
            scopes.append(active)
        elif op.opcode == "pop_condition":
            scopes.pop()
        elif op.opcode in {"hmr", "r"}:
            measured = ((histories >> event) & 1).astype(bool)
            event += 1
            sign ^= active & measured & wires[q[0]]
            wires[q[0]] &= ~active
            if op.result_bit is not None:
                bits[op.result_bit] = np.where(active, measured, bits[op.result_bit])
        elif op.opcode in {"x", "cx", "ccx"}:
            for wire in q[:-1]:
                active = active & wires[wire]
            wires[q[-1]] ^= active
        elif op.opcode in {"neg", "z", "cz", "ccz"}:
            for wire in q:
                active = active & wires[wire]
            sign ^= active
        elif op.opcode == "bit_store0":
            bits[op.result_bit].fill(False)
        elif op.opcode == "bit_invert":
            bits[op.result_bit] ^= active
        else:
            pytest.fail(f"Unexpected native operation {op.opcode}")
    assert not np.any(sign)
    assert not any(np.any(wire) for wire in wires[2 * width + 1 :])
    assert np.all(sum(wires[width + i] * (1 << i) for i in range(width)) == rhs)
    assert np.all(wires[2 * width] == control)
    return tuple(map(int, sum(wires[i] * (1 << i) for i in range(width))))


@pytest.mark.parametrize("subtract", [False, True])
def test_two_bit_modular_signed_basis_map_in_every_history(subtract: bool) -> None:
    np = pytest.importorskip("numpy")
    program = kickmix_modular_addition_program(
        (1, 0),
        (3, 2),
        (True, True),
        clean=(5, 6, 7, 8, 9),
        control=4,
        subtract=subtract,
    )
    actual = np.asarray(_certify_all_histories(program))
    expected = np.array(
        [
            (a + (-b if subtract else b) * c) % 3
            for a, b, c in product(range(3), range(3), range(2))
        ]
    )
    np.testing.assert_array_equal(
        actual.reshape(-1, 18), np.broadcast_to(expected, (len(actual) // 18, 18))
    )
    counts = Counter(op.opcode for op in program.operations)
    assert counts["hmr"] == (9 if subtract else 8)


@pytest.mark.parametrize("subtract", [False, True])
def test_quantum_modulus_is_restored_in_superposition(subtract: bool) -> None:
    program = kickmix_modular_addition_program(
        (0, 1), (2, 3), (4, 5), clean=(7, 8, 9, 10, 11), control=6, subtract=subtract
    )
    state = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for modulus in (2, 3):
        for a, b, enabled in product(range(modulus), range(modulus), range(2)):
            suffix = (b << 8) | (modulus << 6) | (enabled << 5)
            amplitude = complex(a + 1, 3 * b + modulus)
            state[(a << 10) | suffix] = amplitude
            value = (a + (-b if subtract else b) * enabled) % modulus
            expected[(value << 10) | suffix] = amplitude
    scale = state.norm()
    gates = bind_reversible_phase_execution(program)
    circuit = DynamicCircuit(12, inputs=state / scale, dtype=torch.complex128).append(
        FieldOperation(12, gates)
    )
    result = run_dynamic(circuit, shots=48, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(48, -1),
        (expected / scale).expand(48, -1),
        atol=1e-12,
        rtol=0,
    )


def test_quantum_modulus_requires_actual_clean_comparison_workspace() -> None:
    # The native insufficient-work mixed comparison reads past its clean span.
    with pytest.raises(ValueError, match="width minus one clean"):
        kickmix_modular_addition_program(
            (2, 1, 0),
            (5, 4, 3),
            (8, 7, 6),
            clean=(10, 11, 12, 13),
            dirty=(14, 15),
        )
