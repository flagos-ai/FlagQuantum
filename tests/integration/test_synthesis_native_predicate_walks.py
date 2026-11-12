"""Original coined walks retain all amplitudes, phases and clean compiler work."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_quantum_walk.qfold import (
    main as qfold,
)
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_quantum_walk.qfold import (
    physics as qfold_model,
)
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_quantum_walk.qfold.comparisons.circuit import (
    reference_circuit as qfold_reference,
)
from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_quantum_walk.qfold.comparisons.circuit import (
    step_matrix,
)
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.quantum_walk_hypercube import (
    main as hypercube,
)
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.quantum_walk_hypercube import (
    physics as hypercube_model,
)
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.quantum_walk_hypercube.comparisons.circuit import (
    reference_circuit as hypercube_reference,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "tests/fixtures/classiq_qmod/corpus/original"


def check(
    program: synth.Program,
    expected: torch.Tensor,
    registers: tuple[str, ...],
    environment_width: int = 0,
) -> None:
    vector = expected.flatten()
    work = program.report["workspace_qubits"]
    assert isinstance(work, int)
    complete = torch.zeros((vector.numel(), 1 << work), dtype=torch.complex128)
    complete[:, 0] = vector
    result = fq.run(program)
    actual = result.statevector().reshape(vector.numel(), -1)
    torch.testing.assert_close(actual, complete, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == vector.numel().bit_length() - 1 + work
    marginal = vector.abs().square().reshape(-1, 1 << environment_width).sum(dim=1)
    probabilities = program.probabilities(result, registers=registers)
    if registers == ("x", "coin"):
        for word in range(64):
            assert probabilities.get((word // 4, word % 4), 0) == pytest.approx(
                float(marginal[word]), abs=1e-12, rel=0
            )
    else:
        for word in range(4):
            assert probabilities.get((word // 2, word % 2), 0) == pytest.approx(
                float(marginal[word]), abs=1e-12, rel=0
            )


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_hypercube_unchanged_source_native_and_independent_state(steps: int) -> None:
    expected = fq.run(hypercube_reference(steps)).statevector()
    source = (
        ORIGINAL
        / "tutorials/advanced_tutorials/discrete_quantum_walk/quantum_walk_hypercube.qmod"
    )
    for program in (
        synth.synthesize(source, classical_inputs={"t": steps}),
        hypercube.build_program(steps),
    ):
        check(program, expected, ("x", "coin"))


def test_qfold_unchanged_source_native_and_independent_state() -> None:
    expected = fq.run(qfold_reference()).statevector()
    source = (
        ORIGINAL
        / "applications/chemistry/protein_folding/protein_folding_with_quantum_walk/qfold.qmod"
    )
    for program in (synth.synthesize(source), qfold.build_program()):
        check(program, expected, ("phi", "psi"), 2)
    # C/M are retained quantum environment, not claimed zero workspace.
    assert float(expected.flatten().reshape(4, 4)[:, 1:].abs().square().sum()) > 1e-3


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
@pytest.mark.parametrize("variant", [0, 1])
def test_hypercube_nonzero_coherent_position_and_coin_models(
    steps: int, variant: int
) -> None:
    position = tuple((-1.0) ** (i + variant) * (i + 1 + variant) for i in range(16))
    coin = ((0.3, -0.4, 0.5, -0.7), (-0.2, 0.6, 0.4, 0.5))[variant]
    norm = sum(v * v for v in position) ** 0.5
    position = tuple(v / norm for v in position)
    norm = sum(v * v for v in coin) ** 0.5
    coin = tuple(v / norm for v in coin)
    program = hypercube.build_program(steps, position, coin)
    check(
        program,
        fq.run(hypercube_reference(steps, position, coin)).statevector(),
        ("x", "coin"),
    )


@pytest.mark.parametrize(
    "a,b,amplitudes",
    [(0.31, -0.47, (0.3, -0.4, 0.5, -0.7)), (-0.29, 0.83, (-0.2, 0.6, 0.4, 0.5))],
)
def test_qfold_two_nonzero_angle_and_data_models(
    a: float, b: float, amplitudes: tuple[float, ...]
) -> None:
    norm = sum(v * v for v in amplitudes) ** 0.5
    values = tuple(v / norm for v in amplitudes)
    check(
        qfold.build_program(a, b, values),
        fq.run(qfold_reference(a, b, values)).statevector(),
        ("phi", "psi"),
        2,
    )


@pytest.mark.parametrize("basis", range(16))
def test_qfold_every_operator_column_including_nonzero_environment(basis: int) -> None:
    amplitudes = [float(i == basis) for i in range(16)]
    body = (
        "C:QBit=QBit()\nM:QBit=QBit()\nallocate(C)\nallocate(M)\nallocate(phi)\nallocate(psi)\n"
        + f"inplace_prepare_amplitudes({amplitudes!r},0,(M,C,psi,phi))\nwith within():\n    coin(a,b,C,M,psi,phi)\nwith apply():\n    shift(C,M,psi,phi)\noracle(C,M)\ndrop(C)\ndrop(M)"
    )
    extra = "from examples.synthesis.applications.chemistry.protein_folding.protein_folding_with_quantum_walk.qfold.operations import coin,shift,oracle\n"
    program = synth.synthesize(
        definition("a:float,b:float,phi:Output[QBit],psi:Output[QBit]", body, extra),
        classical_inputs={"a": 0.31, "b": -0.47},
    )
    check(program, step_matrix(0.31, -0.47)[:, basis], ("phi", "psi"), 2)


@pytest.mark.parametrize("steps", [-1, 17, True])
def test_invalid_step_count(steps: int) -> None:
    with pytest.raises(ValueError, match="0..16"):
        hypercube.build_program(steps)


@pytest.mark.parametrize("which", ["hypercube", "qfold"])
@pytest.mark.parametrize(
    "budget", ["operations", "width", "workspace", "depth", "memory"]
)
def test_actual_resources_are_enforced(which: str, budget: str) -> None:
    if which == "hypercube":
        entry = hypercube.main
        bindings: dict[str, float | list[float]] = {
            "t": 2,
            "position_amplitudes": list(hypercube_model.POSITION_AMPLITUDES),
            "coin_amplitudes": list(hypercube_model.COIN_AMPLITUDES),
        }
        width = 6
    else:
        entry = qfold.main
        bindings = {
            "a": qfold_model.ANGLES[0],
            "b": qfold_model.ANGLES[1],
            "amplitudes": list(qfold_model.AMPLITUDES),
        }
        width = 4
    with pytest.raises((HybridVerificationError, synth.SynthesisError, PlanningError)):
        if budget == "operations":
            synth.synthesize(entry, classical_inputs=bindings, max_operations=1)
        elif budget == "width":
            synth.synthesize(entry, classical_inputs=bindings, max_qubits=width)
        elif budget == "workspace":
            synth.synthesize(entry, classical_inputs=bindings, max_workspace=0)
        elif budget == "depth":
            synth.synthesize(entry, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(entry, classical_inputs=bindings, memory_limit_bytes=16)
