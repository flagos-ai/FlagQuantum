"""Validate parameterized native Pauli applications and the full FMO workflow."""

from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_state_preparation.adapt_vqe.comparisons.circuit import (
    reference_circuit as adapt_reference,
)
from examples.synthesis.algorithms.quantum_state_preparation.adapt_vqe.main import (
    build_program as adapt_program,
)
from examples.synthesis.algorithms.quantum_state_preparation.adapt_vqe.main import (
    main as adapt_main,
)
from examples.synthesis.applications.chemistry.quantum_walk_fmo.comparisons.circuit import (
    product_formula,
)
from examples.synthesis.applications.chemistry.quantum_walk_fmo.comparisons.circuit import (
    reference_circuit as fmo_reference,
)
from examples.synthesis.applications.chemistry.quantum_walk_fmo.main import (
    build_program as fmo_program,
)
from examples.synthesis.applications.chemistry.quantum_walk_fmo.main import (
    main as fmo_main,
)
from examples.synthesis.applications.chemistry.quantum_walk_fmo.physics import (
    HAMILTONIAN,
)
from examples.synthesis.applications.chemistry.quantum_walk_fmo.workflow import (
    exact_sweep,
    quantum_sweep,
    save_plot,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.errors import PlanningError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"
QUANTUM_TIMES = tuple(index * 0.1 / 19 for index in range(20))
EXACT_TIMES = tuple(index * 0.1 / 99 for index in range(100))


@pytest.mark.parametrize("theta", [0.23, -0.41])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_adapt_vqe_real_array_helpers_match_original_and_independent_circuit(
    theta: float, objective: Literal["width", "depth"]
) -> None:
    source = ROOT / "algorithms/quantum_state_preparation/adapt_vqe/adapt_vqe.qmod"
    original = synth.synthesize(
        source, classical_inputs={"params": [theta]}, objective=objective
    )
    native = synth.synthesize(
        adapt_main, classical_inputs={"params": [theta]}, objective=objective
    )
    for program in (original, native):
        compare_states(program, adapt_reference(theta), ("v",))
        assert program.report["workspace_qubits"] == 0
    assert native.report["resources"] == original.report["resources"]


@pytest.mark.parametrize("time", [0.037, -0.081])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_all_35_fmo_terms_and_identity_phase_match_original_and_physical_matrix(
    time: float, objective: Literal["width", "depth"]
) -> None:
    source = ROOT / "applications/chemistry/quantum_walk_fmo/quantum_walk_fmo.qmod"
    original = synth.synthesize(
        source, classical_inputs={"time_point": time}, objective=objective
    )
    native = synth.synthesize(
        fmo_main, classical_inputs={"time_point": time}, objective=objective
    )
    assert len(HAMILTONIAN.terms) == 35
    for program in (original, native):
        compare_states(program, fmo_reference(time), ("qn",))
        assert program.report["workspace_qubits"] == 0
    assert native.report["resources"] == original.report["resources"]


@pytest.mark.parametrize("basis", range(8))
def test_every_fmo_basis_column_including_padding_preserves_bit_order(
    basis: int,
) -> None:
    preparation = "\n".join(
        f"    X(q[{bit}])" for bit in range(3) if basis & (1 << bit)
    )
    entry = program_definition(
        "from flagquantum.synth import *\n"
        "from examples.synthesis.applications.chemistry.quantum_walk_fmo.physics import HAMILTONIAN\n"
        "@qfunc\ndef main(q:Output[QArray[QBit]]):\n    allocate(3,q)\n"
        + preparation
        + "\n    suzuki_trotter(HAMILTONIAN,0.037,1,10,q)\n"
    )
    reference = fq.Circuit(3, dtype=torch.complex128)
    for bit in range(3):
        if basis & (1 << bit):
            reference.x(2 - bit)
    reference.any(0, 1, 2, matrix=product_formula(0.037))
    compare_states(synth.synthesize(entry), reference, ("q",))


@pytest.fixture(scope="module")
def fmo_sweep() -> tuple[torch.Tensor, tuple[float, ...]]:
    return quantum_sweep(QUANTUM_TIMES)


def test_full_fmo_time_sweep(
    fmo_sweep: tuple[torch.Tensor, tuple[float, ...]],
) -> None:
    populations, durations = fmo_sweep
    exact = exact_sweep(EXACT_TIMES)
    assert populations.shape == (20, 8)
    assert exact.shape == (100, 7)
    assert len(durations) == 20 and all(value >= 0 for value in durations)
    torch.testing.assert_close(
        populations.sum(dim=1), torch.ones(20, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        exact.sum(dim=1), torch.ones(100, dtype=torch.float64), atol=1e-12, rtol=0
    )
    for index, time in enumerate(QUANTUM_TIMES):
        reference = fq.run(fmo_reference(time)).statevector().flatten().abs().square()
        torch.testing.assert_close(populations[index], reference, atol=1e-12, rtol=0)
    # Product-formula padding leakage is physical approximation error, not dirty scratch.
    assert float(populations[7, 7]) > 0


def test_full_fmo_population_plot(
    tmp_path: Path, fmo_sweep: tuple[torch.Tensor, tuple[float, ...]]
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    populations, _ = fmo_sweep
    output = tmp_path / "fmo-populations.png"
    before = plt.get_fignums()
    save_plot(QUANTUM_TIMES, populations, EXACT_TIMES, exact_sweep(EXACT_TIMES), output)
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 10_000
    assert plt.get_fignums() == before


def test_native_application_budgets_are_checked() -> None:
    with pytest.raises((synth.SynthesisError, PlanningError)):
        synth.synthesize(
            fmo_main, classical_inputs={"time_point": 0.037}, max_operations=100
        )
    with pytest.raises((synth.SynthesisError, PlanningError)):
        synth.synthesize(adapt_main, classical_inputs={"params": [0.23]}, max_qubits=3)


def test_public_builders_preserve_the_same_programs() -> None:
    compare_states(adapt_program(), adapt_reference(), ("v",))
    compare_states(fmo_program(), fmo_reference(), ("qn",))
