"""Independent full-state, single-shot and estimator checks for native shadows."""

import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import execute_code_cells

try:
    import numpy as np

    from examples.synthesis.algorithms.metrology.classical_shadow_tomography.overview import (
        workflow as shadow,
    )
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "matplotlib"}:
        raise
    pytest.skip(str(error), allow_module_level=True)

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (
    ROOT
    / "examples/synthesis/algorithms/metrology/classical_shadow_tomography/overview"
)


def _pauli_circuit(thetas: list[float], phis: list[float]) -> fq.Circuit:
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).z(0).cx(0, 1)
    for index in range(2):
        circuit.rz(1 - index, theta=phis[index]).ry(1 - index, theta=thetas[index])
    return circuit


def _clifford_circuit(sequence: tuple[shadow.Gate, ...]) -> fq.Circuit:
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0).z(0).cx(0, 1)
    for gate in sequence:
        if gate.kind == 0:
            circuit.h(1 - gate.first)
        elif gate.kind == 1:
            circuit.s(1 - gate.first)
        else:
            circuit.cx(1 - gate.first, 1 - gate.second)
    return circuit


def _draw(state: torch.Tensor, generator: torch.Generator) -> tuple[int, int]:
    word = int(
        torch.multinomial(state.flatten().abs().square(), 1, generator=generator)
    )
    return word & 1, word >> 1


def _check_state(program: synth.Program, reference: fq.Circuit) -> torch.Tensor:
    actual = fq.run(program).statevector()
    expected = fq.run(reference).statevector()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    torch.testing.assert_close(actual.flatten(), expected.flatten(), atol=1e-12, rtol=0)
    assert program.circuit.n_wires == 2 and program.report["workspace_qubits"] == 0
    assert program.registers == {"q": (0, 1)}
    return expected


def _reference_unitary(sequence: tuple[shadow.Gate, ...]) -> torch.Tensor:
    # Independent basis-vector Circuit propagation, then explicit axis permutation.
    columns = []
    permutation = torch.tensor([0, 2, 1, 3])
    for logical_word in range(4):
        circuit = fq.Circuit(2, dtype=torch.complex128)
        if logical_word & 2:
            circuit.x(1)
        if logical_word & 1:
            circuit.x(0)
        for gate in sequence:
            circuit.gate(
                ("h", "s", "cx")[gate.kind],
                (1 - gate.first, 1 - gate.second) if gate.kind == 2 else 1 - gate.first,
            )
        state = fq.run(circuit).statevector()
        assert isinstance(state, torch.Tensor)
        columns.append(state.flatten()[permutation])
    return torch.stack(columns, dim=1)


def _pauli_reference_estimators(
    bits: torch.Tensor, bases: torch.Tensor
) -> torch.Tensor:
    matrices = (
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.diag([1, -1]).astype(complex),
    )
    rows = []
    for pair, ids in zip(bits.tolist(), bases.tolist(), strict=True):
        factors = []
        for bit, basis in zip(pair, ids, strict=True):
            sign = (1 - 2 * bit) * (-1 if basis == 1 else 1)
            factors.append((np.eye(2) + 3 * sign * matrices[basis]) / 2)
        rows.append(np.kron(*factors))
    return torch.tensor(np.array(rows), dtype=torch.complex128)


@pytest.fixture(scope="module")
def workflow() -> shadow.Workflow:
    return shadow.run_workflow()


def test_all_450_absolute_states_and_independent_same_seed_outcomes(
    workflow: shadow.Workflow,
) -> None:
    rng = np.random.RandomState(552)
    bases = rng.randint(3, size=(400, 2))
    assert workflow.pauli.bases is not None
    np.testing.assert_array_equal(workflow.pauli.bases.numpy(), bases)
    generator = torch.Generator().manual_seed(int(rng.randint(1000000)))
    angles = ((math.pi / 2, math.pi), (math.pi / 2, 3 * math.pi / 2), (0.0, 0.0))
    for index, pair in enumerate(bases):
        a, b = map(int, pair)
        program = shadow.build_pauli_program((a, b))
        expected = _check_state(
            program,
            _pauli_circuit([angles[a][0], angles[b][0]], [angles[a][1], angles[b][1]]),
        )
        assert list(_draw(expected, generator)) == workflow.pauli.bits[index].tolist()
        assert program.report["resources"] == workflow.pauli.resources[index]
    for index in range(50):
        gates = []
        for _ in range(20):
            kind = int(rng.randint(3))
            if kind < 2:
                a, b = int(rng.randint(2)), -1
            else:
                a, b = map(int, rng.choice(2, size=2, replace=False))
            gates.append(shadow.Gate(kind, a, b))
        sequence = tuple(gates)
        assert sequence == workflow.clifford.sequences[index]
        program = shadow.build_clifford_program(sequence)
        expected = _check_state(program, _clifford_circuit(sequence))
        generator = torch.Generator().manual_seed(int(rng.randint(1000000)))
        assert (
            list(_draw(expected, generator)) == workflow.clifford.bits[index].tolist()
        )
        assert program.report["resources"] == workflow.clifford.resources[index]
    assert (
        len(workflow.pauli.resources) == 400 and len(workflow.clifford.resources) == 50
    )
    assert len(workflow.clifford.sequences) == 50
    assert workflow.pauli.sequences == () and workflow.clifford.bases is None


@pytest.mark.parametrize("scale", (1.0, -0.63))
def test_nonzero_arbitrary_basis_bindings_keep_absolute_phase(scale: float) -> None:
    theta = [scale * 0.39, scale * -0.71]
    phi = [scale * 1.23, scale * -0.53]
    program = synth.synthesize(
        shadow.pauli_measurement,
        classical_inputs={"thetas": theta, "phis": phi},
        max_workspace=0,
    )
    _check_state(program, _pauli_circuit(theta, phi))


def test_all_estimator_matrices_and_observable_means(workflow: shadow.Workflow) -> None:
    assert workflow.pauli.bases is not None
    pauli_estimators = _pauli_reference_estimators(
        workflow.pauli.bits, workflow.pauli.bases
    )
    actual = shadow.pauli_estimators(workflow.pauli.bits, workflow.pauli.bases)
    torch.testing.assert_close(actual, pauli_estimators, atol=1e-12, rtol=0)
    clifford_estimators = []
    for bits, sequence in zip(
        workflow.clifford.bits.tolist(), workflow.clifford.sequences, strict=True
    ):
        unitary = _reference_unitary(sequence)
        torch.testing.assert_close(
            shadow.sequence_unitary(sequence), unitary, atol=1e-12, rtol=0
        )
        ket = torch.eye(4, dtype=torch.complex128)[:, 2 * bits[0] + bits[1]]
        inverse = unitary.mH @ ket
        clifford_estimators.append(
            5 * torch.outer(inverse, inverse.conj())
            - torch.eye(4, dtype=torch.complex128)
        )
    matrices = torch.stack(clifford_estimators)
    torch.testing.assert_close(
        shadow.clifford_estimators(workflow.clifford.bits, workflow.clifford.sequences),
        matrices,
        atol=1e-12,
        rtol=0,
    )
    target = torch.tensor(
        [[0.5, 0, 0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [-0.5, 0, 0, 0.5]],
        dtype=torch.complex128,
    )
    zz = np.diag([1, -1, -1, 1])
    for recorded, estimators, divisions in (
        (workflow.pauli, pauli_estimators, 2),
        (workflow.clifford, matrices, 10),
    ):
        density = estimators.mean(0)
        torch.testing.assert_close(recorded.density, density, atol=1e-12, rtol=0)
        assert recorded.frobenius_error == pytest.approx(
            float(torch.linalg.matrix_norm(density - target)), abs=1e-12
        )
        values = np.array([np.trace(zz @ matrix.numpy()).real for matrix in estimators])
        expected = float(
            np.median([chunk.mean() for chunk in np.array_split(values, divisions)])
        )
        assert recorded.zz_estimate == pytest.approx(expected, abs=1e-12)
        assert abs(float(density.trace().real) - 1) < 1e-12
    assert workflow.pauli.zz_estimate == pytest.approx(0.81, abs=1e-12)
    assert workflow.clifford.zz_estimate == pytest.approx(1.0, abs=1e-12)
    # Linear inversion estimates are not physical density matrices in general.
    assert torch.linalg.eigvalsh(workflow.pauli.density).min() < 0
    assert torch.linalg.eigvalsh(workflow.clifford.density).min() < 0


def test_inverse_pauli_channel_handles_y_sign_and_asymmetric_complex_states() -> None:
    identity = torch.eye(2, dtype=torch.complex128)
    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    s = torch.diag(torch.tensor([1, 1j], dtype=torch.complex128))
    unitaries = (h, h @ s, identity)
    state = torch.tensor(
        [0.2 + 0.1j, -0.3 + 0.4j, 0.5 - 0.2j, 0.1 + 0.6j], dtype=torch.complex128
    )
    state /= state.norm()
    expected = torch.outer(state, state.conj())
    reconstructed = torch.zeros((4, 4), dtype=torch.complex128)
    for a, b in itertools.product(range(3), repeat=2):
        u = torch.kron(unitaries[a], unitaries[b])
        probabilities = (u @ state).abs().square()
        # Matrix tensor factors use logical index zero first, unlike packed integers.
        bits = torch.tensor([[word >> 1, word & 1] for word in range(4)])
        bases = torch.tensor([[a, b]] * 4)
        estimates = shadow.pauli_estimators(bits, bases)
        reconstructed += (estimates * probabilities[:, None, None]).sum(0) / 9
    torch.testing.assert_close(reconstructed, expected, atol=1e-12, rtol=0)
    plus_y = torch.tensor([1, 1j], dtype=torch.complex128) / math.sqrt(2)
    torch.testing.assert_close(
        (unitaries[1] @ plus_y).abs().square(),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    y_estimate = shadow.pauli_estimators(
        torch.tensor([[0, 0]]), torch.tensor([[1, 2]])
    )[0]
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    assert float(
        torch.trace(torch.kron(y, identity) @ y_estimate).real
    ) == pytest.approx(-3, abs=1e-12)


def test_even_median_is_not_lower_torch_median() -> None:
    zz = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.complex128))
    matrices = torch.stack([value * zz / 4 for value in (0, 2, 10, 20)])
    assert shadow.median_of_means(matrices, 4) == 6
    assert shadow.median_of_means(matrices, 2) == 8


def test_computed_bounds_are_not_executed_snapshot_budgets(
    workflow: shadow.Workflow,
) -> None:
    assert workflow.pauli_bound == (81065, 10)
    assert workflow.clifford_bound == (108086, 10)
    k = 2 * math.log(200)
    assert shadow.sample_bound(0.1, 0.01, 9) == (math.ceil(34 * 9 / 0.1**2 * k), int(k))
    assert len(workflow.pauli.bits) == 400 < workflow.pauli_bound[0]
    assert len(workflow.clifford.bits) == 50 < workflow.clifford_bound[0]


@pytest.mark.parametrize("kind", ("pauli", "clifford"))
@pytest.mark.parametrize("budget", ("width", "operations", "depth", "memory"))
def test_resource_budget_refusal(kind: str, budget: str) -> None:
    inputs = (
        {"thetas": [0.2, -0.3], "phis": [0.4, -0.5]}
        if kind == "pauli"
        else {
            "kinds": [0, 1, 2, 0] * 5,
            "first": [0, 1, 0, 1] * 5,
            "second": [-1, -1, 1, -1] * 5,
        }
    )
    with pytest.raises(PlanningError if budget == "memory" else synth.SynthesisError):
        synth.synthesize(
            (
                shadow.pauli_measurement
                if kind == "pauli"
                else shadow.clifford_measurement
            ),
            classical_inputs=inputs,
            max_qubits=1 if budget == "width" else None,
            max_operations=0 if budget == "operations" else None,
            max_depth=0 if budget == "depth" else None,
            memory_limit_bytes=1 if budget == "memory" else None,
        )


@pytest.mark.parametrize(
    "bits",
    (
        torch.zeros((0, 2), dtype=torch.int64),
        torch.zeros((1, 3), dtype=torch.int64),
        torch.zeros((1, 2)),
        torch.tensor([[0, 2]]),
    ),
)
def test_bad_snapshot_rows(bits: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="snapshot"):
        shadow.pauli_estimators(bits, torch.zeros((1, 2), dtype=torch.int64))
    with pytest.raises(ValueError, match="snapshot"):
        shadow.clifford_estimators(bits, ())


@pytest.mark.parametrize(
    "bases",
    (
        torch.tensor([[0, 3]]),
        torch.tensor([[0, -1]]),
        torch.zeros((1, 2)),
        torch.zeros((2, 2), dtype=torch.int64),
    ),
)
def test_bad_basis_matrices(bases: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="basis"):
        shadow.pauli_estimators(torch.zeros((1, 2), dtype=torch.int64), bases)


@pytest.mark.parametrize("bases", ((0, 3), (-1, 0), (True, 0)))
def test_bad_bound_basis_choices(bases: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="basis"):
        shadow.build_pauli_program(bases)


@pytest.mark.parametrize(
    "gate",
    (
        shadow.Gate(3, 0, -1),
        shadow.Gate(0, 2, -1),
        shadow.Gate(0, 0, 1),
        shadow.Gate(2, 0, 0),
        shadow.Gate(2, 1, 2),
        shadow.Gate(True, 0, -1),
    ),
)
def test_bad_gate_sequence(gate: shadow.Gate) -> None:
    with pytest.raises(ValueError):
        shadow.sequence_unitary((gate,) * 20)
    with pytest.raises(ValueError):
        shadow.build_clifford_program((gate,) * 20)


def test_bad_sequence_lengths_and_snapshot_pairing() -> None:
    with pytest.raises(ValueError, match="depth-20"):
        shadow.build_clifford_program((shadow.Gate(0, 0, -1),))
    with pytest.raises(ValueError, match="each snapshot"):
        shadow.clifford_estimators(torch.tensor([[0, 0]]), ())


@pytest.mark.parametrize("divisions", (0, -1, 3, 5, True))
def test_bad_median_divisions(divisions: int) -> None:
    with pytest.raises(ValueError, match="divisions"):
        shadow.median_of_means(
            torch.zeros((4, 4, 4), dtype=torch.complex128), divisions
        )


@pytest.mark.parametrize(
    "matrices",
    (
        torch.zeros((4, 4)),
        torch.zeros((1, 4, 4)),
        torch.full((1, 4, 4), complex(float("nan")), dtype=torch.complex128),
        torch.triu(torch.ones((1, 4, 4), dtype=torch.complex128), diagonal=1),
    ),
)
def test_bad_estimator_matrices(matrices: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="estimator|Hermitian"):
        shadow.median_of_means(matrices, 1)


@pytest.mark.parametrize(
    "error,delta,norm",
    (
        (0.0, 0.01, 9.0),
        (0.2, 0.0, 9.0),
        (0.2, 1.0, 9.0),
        (0.2, 0.01, 0.0),
        (float("nan"), 0.01, 9.0),
        (0.2, 0.01, float("inf")),
        (True, 0.01, 9.0),
    ),
)
def test_bad_bound_inputs(error: float, delta: float, norm: float) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        shadow.sample_bound(error, delta, norm)


def test_plots_and_provenance(workflow: shadow.Workflow) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    from matplotlib.patches import Rectangle

    figure = shadow.plot_workflow(workflow)
    for axis, recorded in zip(
        figure.axes[:2], (workflow.pauli, workflow.clifford), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(axis.images[0].get_array()),
            recorded.density.real.numpy(),
            atol=0,
            rtol=0,
        )
    bars = figure.axes[2].patches
    assert [float(p.get_height()) for p in bars if isinstance(p, Rectangle)] == [
        workflow.pauli.zz_estimate,
        workflow.clifford.zz_estimate,
    ]
    provenance = json.loads((EXAMPLE / "provenance.json").read_text())
    assert (
        provenance["original_sha256"]
        == "8e4eea58106c6579813281bc83ed905257e899c4ba4e4785b382b83f4b921f11"
    )
    assert len(provenance["reviewed_code_cells"]) == 20
    assert hashlib.sha256((EXAMPLE / "walkthrough.ipynb").read_bytes()).hexdigest()


def test_clean_notebook_executes_source_size_workflow() -> None:
    path = EXAMPLE / "walkthrough.ipynb"
    notebook = json.loads(path.read_text())
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(path)
    assert count == 5
    report = namespace["workflow"]
    assert isinstance(report, shadow.Workflow)
    assert len(report.pauli.bits) == 400 and len(report.clifford.bits) == 50
