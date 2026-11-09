"""Check the source fallback independently of exact-body gate selection."""

from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient as Bit,
)
from flagquantum.compiler.registers.modular.modular_doubling_dispatch import (
    modular_doubling_program,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_kickmix_bit_rotation import _certify

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width,modulus", [(3, 5), (4, 11), (5, 19)])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("controlled", [False, True])
def test_full_precision_space_fallback_matches_every_valid_signed_history(
    width: int, modulus: int, inverse: bool, classical: bool, controlled: bool
) -> None:
    control = width if controlled else None
    data = 2 * width - 2 + controlled
    dirty = tuple(range(width + controlled, data))
    clean = tuple(range(data, data + max(5, width + 1)))
    constants = tuple(bool(modulus >> i & 1) for i in reversed(range(width)))
    coefficients = (
        (True, *(Bit(i) for i in range(width - 2)), True) if classical else constants
    )
    program = modular_doubling_program(
        tuple(range(width)),
        coefficients,
        control=control,
        inverse=inverse,
        clean=clean,
        dirty=dirty,
        minimize_qubits=True,
    )
    assert decode_kmx(encode_kmx(program)) == program
    inputs, outputs = [], []
    for x, enabled, borrowed in product(
        range(modulus), range(2 if controlled else 1), range(1 << len(dirty))
    ):
        suffix = (enabled << len(dirty) | borrowed) if controlled else borrowed
        inputs.append(x << (data - width) | suffix)
        value = (
            x * (pow(2, -1, modulus) if inverse else 2) % modulus
            if enabled or not controlled
            else x
        )
        outputs.append(value << (data - width) | suffix)
    inputs, outputs = torch.tensor(inputs), torch.tensor(outputs)
    prefix = tuple(
        ClassicalXor((), i, constant=constants[i + 1]).instruction()
        for i in range(width - 2)
        if classical
    )
    # The signed certificate needs concrete classical inputs; prefix its artifact operations.
    from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
    from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
    from flagquantum.core.ir import CircuitIR

    initialized = lower_reversible_phase(CircuitIR(program.n_qubits, prefix))
    certificate = ReversiblePhaseProgram(
        program.n_qubits, program.n_bits, (*initialized.operations, *program.operations)
    )
    _certify(certificate, data, outputs, inputs=inputs)
    amplitudes = torch.complex((inputs + 1).double(), (3 * inputs - 1).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[inputs << len(clean)] = amplitudes
    expected[outputs << len(clean)] = amplitudes
    circuit = DynamicCircuit(
        program.n_qubits, inputs=initial, dtype=torch.complex128
    ).append(
        FieldOperation(
            program.n_qubits, (*prefix, *bind_reversible_phase_execution(program))
        )
    )
    result = run_dynamic(circuit, shots=8, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=1e-12, rtol=0
    )
    for bit in range(width - 2 if classical else 0):
        assert torch.all(result.classical_bits[:, bit] == constants[bit + 1])


def test_dispatch_and_known_empty_offset_boundary() -> None:
    from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
    from flagquantum.compiler.registers.modular.exact_modular_doubling import (
        exact_modular_doubling_ir,
    )
    from flagquantum.core.ir import CircuitIR

    for inverse in (False, True):
        program = modular_doubling_program(
            (0, 1, 2), (True, False, True), clean=(4, 5, 6), control=3, inverse=inverse
        )
        exact = exact_modular_doubling_ir(
            (0, 1, 2), (True, False, True), clean=(4, 5, 6), control=3, inverse=inverse
        )
        assert program == lower_reversible_phase(CircuitIR(7, exact))
        with pytest.raises(ValueError, match="empty complemented"):
            modular_doubling_program(
                (0, 1, 2),
                (True, True, True),
                clean=(4, 5, 6, 7, 8),
                control=3,
                inverse=inverse,
                minimize_qubits=True,
            )
    with pytest.raises(ValueError, match="ordered tolerance"):
        modular_doubling_program((0,), (True,), btol=float("nan"))


@pytest.mark.ftqc_reference
def test_low_precision_retains_the_locked_native_error_witness() -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    # Unmodified native generator output: n=4, modulus=13, btol=0,
    # minimize_qubits=True, five clean wires, quantum enable.
    native = """R q5
CCX q4 q3 q5
X q0
R q6
CCX q0 q5 q6
CX q6 q1
HMR q6 b0
CZ q0 q5 if b0
CX q5 q1
CX q5 q0
X q0
CX q5 q3
CSWAP q4 q3 q0
CSWAP q4 q2 q1
CSWAP q4 q3 q1
CX q5 q0
HMR q5 b0
CZ q4 q0 if b0"""
    sim = km.Simulator(batch_size=32, seed=41)
    for q in (0, 1, 2, 4):
        sim.write_across_shots(km.q(q), (1 << 32) - 1)
    sim.do(km.Circuit(native))
    assert [sim.read_across_shots(km.q(q), out=int) for q in range(4)] == [
        0,
        (1 << 32) - 1,
        (1 << 32) - 1,
        (1 << 32) - 1,
    ]
    assert all(sim.read_shot_phase(shot) == 0 for shot in range(32))
    assert all(sim.read_across_shots(km.q(q), out=int) == 0 for q in range(5, 10))
    program = modular_doubling_program(
        (3, 2, 1, 0),
        (True, True, False, True),
        control=4,
        clean=(5, 6, 7, 8, 9),
        btol=0,
        minimize_qubits=True,
    )
    actual = encode_kmx(program).splitlines()[1:]
    assert actual == [line for line in native.splitlines() if not line.startswith("R ")]
    approximate_values = (0, 2, 4, 6, 8, 10, 12, 14, 3, 5, 7, 1, 11)

    def label(value: int, enable: int) -> int:
        return int(f"{value:04b}"[::-1], 2) << 1 | enable

    inputs = torch.tensor([label(x, c) for x, c in product(range(13), range(2))])
    outputs = torch.tensor(
        [
            label(approximate_values[x] if c else x, c)
            for x, c in product(range(13), range(2))
        ]
    )
    _certify(program, 5, outputs, inputs=inputs)
    initial = torch.zeros(1024, dtype=torch.complex128)
    initial[label(7, 1) << 5] = 1
    expected = torch.zeros_like(initial)
    expected[label(14, 1) << 5] = 1
    exact = torch.zeros_like(initial)
    exact[label(1, 1) << 5] = 1
    circuit = DynamicCircuit(10, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(10, bind_reversible_phase_execution(program))
    )
    result = run_dynamic(circuit, shots=32, seed=41, strategy="batched")
    rows = result.final_states.reshape(32, -1)
    torch.testing.assert_close(rows, expected.expand(32, -1), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        (rows - exact).norm(dim=1),
        torch.full((32,), 2**0.5, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("inverse", [False, True])
def test_full_precision_fallback_preserves_a_coherent_quantum_modulus(
    inverse: bool,
) -> None:
    program = modular_doubling_program(
        (0, 1, 2),
        (True, 3, True),
        control=4,
        dirty=(5,),
        clean=(6, 7, 8, 9, 10),
        inverse=inverse,
        minimize_qubits=True,
    )
    inputs, outputs = [], []
    for modulus in (5, 7):
        for x, enabled, borrowed in product(range(modulus), range(2), range(2)):
            suffix = ((modulus >> 1 & 1) << 2) | enabled << 1 | borrowed
            inputs.append(x << 3 | suffix)
            value = (
                x * (pow(2, -1, modulus) if inverse else 2) % modulus if enabled else x
            )
            outputs.append(value << 3 | suffix)
    inputs, outputs = torch.tensor(inputs), torch.tensor(outputs)
    _certify(program, 6, outputs, inputs=inputs)
    amplitude = torch.complex((inputs + 1).double(), (3 * inputs - 2).double())
    amplitude /= amplitude.norm()
    initial = torch.zeros(2048, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[inputs << 5] = amplitude
    expected[outputs << 5] = amplitude
    circuit = DynamicCircuit(11, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(11, bind_reversible_phase_execution(program))
    )
    result = run_dynamic(circuit, shots=16, seed=41, strategy="batched")
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), atol=1e-12, rtol=0
    )
