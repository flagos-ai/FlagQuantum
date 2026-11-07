"""Certify exact modular doubling and inline inverse on the source input domain."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field import ClassicalFieldCoefficient as Bit
from flagquantum.compiler.registers.kickmix_modular_doubling import (
    kickmix_exact_modular_doubling_ir,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_kickmix_bit_rotation import _certify

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width,modulus", [(1, 1), (2, 3), (3, 5), (4, 11), (5, 19)])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("classical", [False, True])
def test_modular_doubling_all_histories_and_complex_state(
    width: int,
    modulus: int,
    controlled: bool,
    inverse: bool,
    classical: bool,
) -> None:
    data = width + controlled + 1
    control = width if controlled else None
    n_work = max(0, width - 2 + controlled) if width > 2 else 0
    clean = tuple(range(data, data + n_work + 1))
    constants = tuple(bool(modulus >> i & 1) for i in reversed(range(width)))
    coefficients = (
        (True, *(Bit(i) for i in range(width - 2)), True)
        if classical and width > 2
        else constants
    )
    slots = width - 2 if classical and width > 2 else 0
    prefix = tuple(
        ClassicalXor((), i, constant=constants[i + 1]).instruction()
        for i in range(slots)
    )
    gates = kickmix_exact_modular_doubling_ir(
        tuple(range(width)),
        coefficients,
        clean=clean,
        control=control,
        inverse=inverse,
        measurement_bit=slots,
    )
    initial_labels, expected_labels = [], []
    for x, enabled, spectator in product(
        range(modulus), range(2 if controlled else 1), range(2)
    ):
        suffix = (enabled << 1 | spectator) if controlled else spectator
        initial_labels.append((x << (data - width)) | suffix)
        value = (
            (x * (pow(2, -1, modulus) if inverse else 2)) % modulus
            if enabled or not controlled
            else x
        )
        expected_labels.append((value << (data - width)) | suffix)
    inputs, expected = torch.tensor(initial_labels), torch.tensor(expected_labels)
    program = lower_reversible_phase(CircuitIR(data + len(clean), (*prefix, *gates)))
    assert decode_kmx(encode_kmx(program)) == program
    _certify(program, data, expected, inputs=inputs)
    amplitudes = torch.complex(
        torch.arange(1, len(inputs) + 1).double(), (3 * inputs - 2).double()
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    final = torch.zeros_like(initial)
    stride = 1 << len(clean)
    initial[inputs * stride] = amplitudes
    final[expected * stride] = amplitudes
    circuit = DynamicCircuit(
        program.n_qubits, inputs=initial, dtype=torch.complex128
    ).append(FieldOperation(program.n_qubits, (*prefix, *gates)))
    result = run_dynamic(circuit, shots=24, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(24, -1), final.expand(24, -1), atol=1e-12, rtol=0
    )
    for bit in range(slots):
        assert torch.all(result.classical_bits[:, bit] == constants[bit + 1])
    reverse = kickmix_exact_modular_doubling_ir(
        tuple(range(width)),
        coefficients,
        clean=clean,
        control=control,
        inverse=not inverse,
        measurement_bit=slots,
    )
    restored = run_dynamic(
        circuit.append(FieldOperation(program.n_qubits, reverse)),
        shots=24,
        seed=41,
        strategy="batched",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(24, -1),
        initial.expand(24, -1),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("width", [3, 4])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("extra", [0, 1])
def test_coherent_quantum_modulus_and_control_are_preserved(
    width: int, inverse: bool, extra: int
) -> None:
    quantum = tuple(range(width, 2 * width - 2))
    control = 2 * width - 2
    data = control + 2
    clean = tuple(range(data, data + width - 1 + extra))
    coefficients = (True, *quantum, True)
    gates = kickmix_exact_modular_doubling_ir(
        tuple(range(width)), coefficients, clean=clean, control=control, inverse=inverse
    )
    inputs, outputs = [], []
    for modulus in range((1 << (width - 1)) | 1, 1 << width, 2):
        for x, enabled, spectator in product(range(modulus), range(2), range(2)):
            middle = (modulus >> 1) & ((1 << (width - 2)) - 1)
            suffix = middle << 2 | enabled << 1 | spectator
            shift = data - width
            inputs.append(x << shift | suffix)
            value = (
                x * (pow(2, -1, modulus) if inverse else 2) % modulus if enabled else x
            )
            outputs.append(value << shift | suffix)
    inputs, outputs = torch.tensor(inputs), torch.tensor(outputs)
    program = lower_reversible_phase(CircuitIR(data + len(clean), gates))
    _certify(program, data, outputs, inputs=inputs)
    amplitudes = torch.complex((inputs + 1).double(), (2 * inputs + 3).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[inputs << len(clean)] = amplitudes
    expected[outputs << len(clean)] = amplitudes
    circuit = DynamicCircuit(
        program.n_qubits, inputs=initial, dtype=torch.complex128
    ).append(FieldOperation(program.n_qubits, gates))
    result = run_dynamic(circuit, shots=32, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), expected.expand(32, -1), atol=1e-12, rtol=0
    )


def test_source_two_bit_reduction_and_workspace_boundary() -> None:
    assert kickmix_exact_modular_doubling_ir((0,), (True,)) == ()
    gates = kickmix_exact_modular_doubling_ir((0, 1), (True, True), control=2)
    assert [(gate.name, gate.wires) for gate in gates] == [("cswap", (2, 1, 0))]
    with pytest.raises(ValueError, match="odd full-capacity"):
        kickmix_exact_modular_doubling_ir((0, 1), (True, False))
    with pytest.raises(ValueError, match="control flag"):
        kickmix_exact_modular_doubling_ir(
            (0, 1, 2), (True, False, True), clean=(4,), control=3
        )
    with pytest.raises(ValueError, match="fresh nonnegative"):
        kickmix_exact_modular_doubling_ir(
            (0, 1, 2), (True, Bit(0), True), clean=(3,), measurement_bit=0
        )


def test_native_counts_distinguish_dirty_and_measured_triple_control() -> None:
    for extra in (0, 1):
        gates = kickmix_exact_modular_doubling_ir(
            (0, 1, 2),
            (True, False, True),
            control=3,
            clean=tuple(range(4, 6 + extra)),
        )
        counts = Counter(
            op.opcode
            for op in lower_reversible_phase(CircuitIR(6 + extra, gates)).operations
        )
        assert counts["ccx"] == (6 if not extra else 4)
        assert counts["hmr"] == (2 if not extra else 3)


@pytest.mark.parametrize("inverse", [False, True])
def test_implicit_dirty_helper_is_restored_with_a_noncontiguous_layout(
    inverse: bool,
) -> None:
    # The native no-clean triple control borrows the lowest free ID (wire 0).
    gates = kickmix_exact_modular_doubling_ir(
        (3, 4, 5), (True, False, True), control=6, clean=(7, 8), inverse=inverse
    )
    program = lower_reversible_phase(CircuitIR(9, gates))
    inputs, outputs = [], []
    for spectator, x, enabled in product(range(8), range(5), range(2)):
        inputs.append(spectator << 4 | x << 1 | enabled)
        value = x * (3 if inverse else 2) % 5 if enabled else x
        outputs.append(spectator << 4 | value << 1 | enabled)
    inputs, outputs = torch.tensor(inputs), torch.tensor(outputs)
    _certify(program, 7, outputs, inputs=inputs)
    amplitudes = torch.complex((inputs + 1).double(), (3 * inputs - 1).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[inputs << 2] = amplitudes
    expected[outputs << 2] = amplitudes
    circuit = DynamicCircuit(9, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(9, gates)
    )
    result = run_dynamic(circuit, shots=24, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(24, -1), expected.expand(24, -1), atol=1e-12, rtol=0
    )
