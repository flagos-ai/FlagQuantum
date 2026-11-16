"""Complete period-finding states and independently measured teleportation channels."""

import cmath
import math
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.foundational.quantum_teleportation import (
    main as teleport,
)
from examples.synthesis.algorithms.foundational.quantum_teleportation.comparisons.circuit import (
    measured_branches,
)
from examples.synthesis.algorithms.foundational.quantum_teleportation.comparisons.circuit import (
    reference_circuit as teleport_reference,
)
from examples.synthesis.algorithms.foundational.quantum_teleportation.main import send
from examples.synthesis.algorithms.number_theory_and_cryptography.shor import (
    main as shor,
)
from examples.synthesis.algorithms.number_theory_and_cryptography.shor.comparisons.circuit import (
    reference_circuit as shor_reference,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from flagquantum.synth import CX, H, Output, QBit, R, allocate, drop, phase, qfunc

pytestmark = pytest.mark.integration
ORIGINALS = (
    Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"
)


def check(program: synth.Program, expected: torch.Tensor) -> torch.Tensor:
    raw = fq.run(program).statevector()
    assert isinstance(raw, torch.Tensor)
    state = raw.reshape(expected.numel(), -1)
    torch.testing.assert_close(state[:, 0], expected.flatten(), atol=1e-12, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    return state[:, 0]


@pytest.mark.parametrize("modulus,base", [(21, 11), (15, 2), (21, 8)])
def test_ten_bit_period_finding_full_dirty_residue_and_clean_work(
    modulus: int, base: int
) -> None:
    expected = fq.run(shor_reference(modulus, base)).statevector()
    program = shor.build_program(modulus, base)
    check(program, expected)
    assert len(program.registers["phase_var"]) == 10
    raw = program.probabilities(fq.run(program), registers=("phase_var",))
    numeric = program.numeric_probabilities(fq.run(program), registers=("phase_var",))
    assert dict(numeric) == {
        (Fraction(word[0], 1024),): probability for word, probability in raw.items()
    }
    if (modulus, base) == (21, 11):
        check(
            synth.synthesize(
                ORIGINALS / "algorithms/number_theory_and_cryptography/shor/shor.qmod"
            ),
            expected,
        )


@pytest.mark.parametrize("modulus,base", [(21, 3), (33, 2), (1, 2)])
def test_invalid_period_parameters_refuse(modulus: int, base: int) -> None:
    with pytest.raises(HybridVerificationError, match="modulus|coprime"):
        shor.build_program(modulus, base)


@pytest.mark.parametrize("theta,phi", [(math.pi / 2, 0.0), (0.73, -0.41), (1.12, 0.37)])
def test_private_feedback_full_purification_and_actual_measurement_density(
    theta: float, phi: float
) -> None:
    expected = fq.run(teleport_reference(theta, phi)).statevector()
    full = check(teleport.build_program(theta, phi), expected).reshape(2, 4)
    psi = torch.tensor(
        [[math.cos(theta / 2), cmath.exp(1j * phi) * math.sin(theta / 2)]],
        dtype=torch.complex128,
    )
    branches = measured_branches(psi)
    assert branches.abs().square().sum(dim=1).tolist() == pytest.approx([0.25] * 4)
    measured = branches.T @ branches.conj()
    torch.testing.assert_close(full @ full.conj().T, measured, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        measured, torch.outer(psi.flatten(), psi.flatten().conj()), atol=1e-12, rtol=0
    )
    if phi == 0:
        check(
            synth.synthesize(
                ORIGINALS
                / "algorithms/foundational/quantum_teleportation/quantum_teleportation.qmod"
            ),
            expected,
        )


@qfunc
def entangled_main(
    theta: float, phi: float, bob: Output[QBit], reference: Output[QBit]
) -> None:
    alice: QBit = QBit()
    bell: QBit = QBit()
    allocate(bob)
    allocate(reference)
    allocate(alice)
    allocate(bell)
    R(theta, math.pi / 2, reference)
    CX(reference, alice)
    phase(alice, phi)
    H(bell)
    CX(bell, bob)
    send(alice, bell, bob)
    drop(alice)
    drop(bell)


@pytest.mark.parametrize("theta,phi", [(math.pi / 2, 0.0), (0.73, -0.41), (1.12, 0.37)])
def test_identity_channel_on_entangled_external_reference(
    theta: float, phi: float
) -> None:
    reference = (
        fq.Circuit(4, dtype=torch.complex128)
        .ry(0, theta=theta)
        .cx(0, 1)
        .phase(0, theta=phi)
        .h(2)
        .h(3)
    )
    program = synth.synthesize(
        entangled_main, classical_inputs={"theta": theta, "phi": phi}
    )
    full = check(program, fq.run(reference).statevector()).reshape(4, 4)
    psi = torch.zeros((2, 2), dtype=torch.complex128)
    psi[0, 0] = math.cos(theta / 2)
    psi[1, 1] = torch.tensor(
        cmath.exp(1j * phi) * math.sin(theta / 2), dtype=torch.complex128
    )
    branches = measured_branches(psi)
    assert branches.abs().square().sum(dim=1).tolist() == pytest.approx([0.25] * 4)
    measured = branches.T @ branches.conj()
    torch.testing.assert_close(full @ full.conj().T, measured, atol=1e-12, rtol=0)
    ideal = psi.T.flatten()
    torch.testing.assert_close(
        measured, torch.outer(ideal, ideal.conj()), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize(
    "kind", ["qubits", "workspace", "operations", "depth", "memory"]
)
def test_period_resource_budgets_refuse(kind: str) -> None:
    inputs = {"n": 21, "a": 11}
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if kind == "qubits":
            synth.synthesize(shor.main, classical_inputs=inputs, max_qubits=17)
        elif kind == "workspace":
            synth.synthesize(shor.main, classical_inputs=inputs, max_workspace=2)
        elif kind == "operations":
            synth.synthesize(shor.main, classical_inputs=inputs, max_operations=1)
        elif kind == "depth":
            synth.synthesize(shor.main, classical_inputs=inputs, max_depth=1)
        else:
            synth.synthesize(shor.main, classical_inputs=inputs, memory_limit_bytes=16)
