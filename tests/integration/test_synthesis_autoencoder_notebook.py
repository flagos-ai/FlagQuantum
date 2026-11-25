"""Independent Circuit acceptance of both complete native autoencoder routes."""

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.errors import PlanningError
from flagquantum.simulation.density_matrix import density_matrix_from_ir
from tests.integration.synthesis_notebook_support import execute_code_cells

try:
    from examples.synthesis.algorithms.machine_learning.quantum_autoencoder.overview import (
        workflow as ae,
    )
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "matplotlib"}:
        raise
    pytest.skip(str(error), allow_module_level=True)

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (
    ROOT / "examples/synthesis/algorithms/machine_learning/quantum_autoencoder/overview"
)


def _reference(
    data: torch.Tensor, weights: torch.Tensor, mode: Literal["swap", "trash", "decode"]
) -> fq.Circuit:
    width = 7 if mode == "swap" else 4
    circuit = fq.Circuit(width, dtype=torch.complex128)
    q = (3, 2, 1, 0)
    for i in range(4):
        circuit.ry(q[i], theta=math.pi * data[i])
    for layer in range(2):
        for i in range(4):
            circuit.ry(q[i], theta=weights[4 * layer + i])
        for i in range(3):
            circuit.cx(q[i], q[i + 1])
    for i in range(2):
        circuit.ry(q[2 + i], theta=weights[8 + i])
    if mode == "swap":
        circuit.h(4).gate("cswap", (4, 1, 6)).gate("cswap", (4, 0, 5)).h(4)
    if mode == "decode":
        circuit.amplitude_damping(q[2], gamma=1).amplitude_damping(q[3], gamma=1)
        for i in (1, 0):
            circuit.ry(q[2 + i], theta=-weights[8 + i])
        for layer in (1, 0):
            for i in (2, 1, 0):
                circuit.cx(q[i], q[i + 1])
            for i in (3, 2, 1, 0):
                circuit.ry(q[i], theta=-weights[4 * layer + i])
    return circuit


def _reference_readout(
    data: torch.Tensor,
    weights: torch.Tensor,
    *,
    cost: ae.Cost,
    shots: int | None = ae.SHOTS,
    seed: int = 0,
) -> torch.Tensor:
    state = fq.run(_reference(data, weights, cost)).statevector()
    assert isinstance(state, torch.Tensor)
    probabilities = state.flatten().abs().square()
    if cost == "swap":
        p1 = probabilities[[i for i in range(128) if i & 4]].sum()
        probabilities = torch.stack((1 - p1, p1))
    if shots is not None:
        draws = torch.multinomial(
            probabilities,
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        probabilities = (
            torch.bincount(draws, minlength=len(probabilities)).to(torch.float64)
            / shots
        )
    if cost == "swap":
        return probabilities[1]
    counts = torch.tensor(
        [int(bool(i & 4)) + int(bool(i & 8)) for i in range(16)], dtype=torch.float64
    )
    return probabilities @ counts


@pytest.fixture(scope="module")
def workflow() -> ae.Workflow:
    return ae.run_workflow()


@pytest.mark.parametrize("scale", (1.0, -0.63))
@pytest.mark.parametrize("cost", ("swap", "trash"))
def test_full_absolute_states_and_all_ten_gradients(
    scale: float, cost: ae.Cost
) -> None:
    data = scale * torch.tensor([0.19, -0.31, 0.43, -0.59], dtype=torch.float64)
    weights = scale * ae.initial_weights()
    program = ae.build_program(data, weights, mode=cost)
    actual = fq.run(program).statevector()
    expected = fq.run(_reference(data, weights, cost)).statevector()
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
    torch.testing.assert_close(actual.flatten(), expected.flatten(), atol=1e-12, rtol=0)
    w = weights.clone().requires_grad_()
    expected_gradient = torch.autograd.grad(
        _reference_readout(data, w, cost=cost, shots=None), w
    )[0]
    torch.testing.assert_close(
        ae.quantum_gradient(data, weights, cost=cost, shots=None),
        expected_gradient,
        atol=1e-12,
        rtol=0,
    )
    for seed in (3, 17):
        torch.testing.assert_close(
            ae.readout(data, weights, cost=cost, seed=seed),
            _reference_readout(data, weights, cost=cost, seed=seed),
            atol=0,
            rtol=0,
        )
    assert program.report["workspace_qubits"] == 0
    assert program.circuit.n_wires == (7 if cost == "swap" else 4)
    if cost == "swap":
        # Dropped reference is retained entangled environment, not clean scratch.
        assert (
            len({wire for wires in program.registers.values() for wire in wires}) == 5
        )
        state = actual.flatten().reshape(32, 4)
        assert float(state[:, 1:].abs().square().sum()) > 0.01


def test_swap_relation_and_named_array_bit_order() -> None:
    data = torch.tensor(ae.DOMAIN_WALL[0], dtype=torch.float64)
    weights = ae.initial_weights()
    program = ae.build_program(data, weights, mode="trash")
    values = dict(program.probabilities(fq.run(program), registers=("q",)))
    zero_trash = sum(value for (word,), value in values.items() if word & 12 == 0)
    p1 = float(ae.readout(data, weights, cost="swap", shots=None))
    assert p1 == pytest.approx(0.5 * (1 - zero_trash), abs=1e-12)
    assert p1 < 0.003
    assert sum(value for (word,), value in values.items() if word & 3 == 0) < 0.1


@pytest.mark.parametrize("scale", (1.0, -0.63))
def test_reset_is_a_mixed_channel_with_full_density_reference(scale: float) -> None:
    data = scale * torch.tensor([0.19, -0.31, 0.43, -0.59], dtype=torch.float64)
    weights = scale * ae.initial_weights()
    program = ae.build_program(data, weights, mode="decode")
    assert program.execution.state_mode == "density_matrix"
    actual = fq.run(program).state
    assert isinstance(actual, torch.Tensor)
    expected = density_matrix_from_ir(
        _reference(data, weights, "decode").to_ir(), dtype=torch.complex128
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    matrix = actual.reshape(16, 16)
    assert float(torch.trace(matrix @ matrix).real) < 0.99
    assert float(torch.trace(matrix).real) == pytest.approx(1, abs=1e-12)
    assert torch.linalg.eigvalsh(matrix).min() >= -1e-12
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate) and resources.channel_count == 2


def test_source_checkpoint_and_both_sampled_training_updates(
    workflow: ae.Workflow,
) -> None:
    expected_init = torch.tensor(ae.INITIAL_WEIGHTS, dtype=torch.float32).to(
        torch.float64
    )
    assert workflow.quantum_evaluations == 134 and ae.SHOTS == 4096
    for training in (workflow.swap, workflow.trash):
        torch.testing.assert_close(
            training.steps[0].weights_before, expected_init, atol=0, rtol=0
        )
        assert [len(step.data) for step in training.steps] == [2, 1]
        assert sorted(
            tuple(map(int, row))
            for step in training.steps
            for row in step.data.tolist()
        ) == sorted(ae.DOMAIN_WALL)
        weights = expected_init.clone()
        for batch_index, step in enumerate(training.steps):
            base = 7 + 1000 * batch_index
            torch.testing.assert_close(step.weights_before, weights, atol=1e-12, rtol=0)
            predictions = torch.stack(
                [
                    _reference_readout(
                        row, weights, cost=training.cost, seed=base + 21 * i
                    )
                    for i, row in enumerate(step.data)
                ]
            )
            torch.testing.assert_close(predictions, step.predictions, atol=0, rtol=0)
            gradient = torch.zeros(10, dtype=torch.float64)
            for i, row in enumerate(step.data):
                for j in range(10):
                    shift = torch.zeros(10, dtype=torch.float64)
                    shift[j] = math.pi / 2
                    plus = _reference_readout(
                        row,
                        weights + shift,
                        cost=training.cost,
                        seed=base + 21 * i + 1 + 2 * j,
                    )
                    minus = _reference_readout(
                        row,
                        weights - shift,
                        cost=training.cost,
                        seed=base + 21 * i + 2 + 2 * j,
                    )
                    gradient[j] += (
                        predictions[i].sign() * 0.5 * (plus - minus) / len(step.data)
                    )
            torch.testing.assert_close(step.gradient, gradient, atol=1e-12, rtol=0)
            assert step.loss == float(predictions.abs().mean())
            weights -= 0.3 * gradient
            torch.testing.assert_close(step.weights_after, weights, atol=1e-12, rtol=0)
        torch.testing.assert_close(training.weights, weights, atol=1e-12, rtol=0)
        assert float((weights - expected_init).norm()) > 0.001


def test_all_decoder_inputs_and_anomaly_scores(workflow: ae.Workflow) -> None:
    for i, decoded in enumerate(workflow.decoded):
        assert decoded.data == ae.DOMAIN_WALL[i] == decoded.modal_bits
        reference = _reference(
            torch.tensor(decoded.data, dtype=torch.float64),
            workflow.swap.weights,
            "decode",
        )
        expected = density_matrix_from_ir(reference.to_ir(), dtype=torch.complex128)
        torch.testing.assert_close(decoded.density, expected, atol=1e-12, rtol=0)
        probabilities = expected.reshape(16, 16).diagonal().real
        draws = torch.multinomial(
            probabilities,
            4096,
            replacement=True,
            generator=torch.Generator().manual_seed(100 + i),
        )
        torch.testing.assert_close(
            decoded.probabilities,
            torch.bincount(draws, minlength=16).to(torch.float64) / 4096,
            atol=0,
            rtol=0,
        )
    rows = (*ae.DOMAIN_WALL, (1, 0, 1, 0), (1, 1, 1, 1))
    assert {row.data for row in workflow.anomalies} == set(rows)
    for anomaly in workflow.anomalies:
        expected_cost = float(
            _reference_readout(
                torch.tensor(anomaly.data, dtype=torch.float64),
                workflow.swap.weights,
                cost="swap",
                seed=200 + rows.index(anomaly.data),
            )
        )
        assert anomaly.cost == expected_cost
        assert (
            anomaly.detected
            == (abs(expected_cost) > 0.01)
            == (anomaly.data not in ae.DOMAIN_WALL)
        )


@pytest.mark.parametrize("mode", ("swap", "trash", "decode"))
@pytest.mark.parametrize("budget", ("width", "operations", "depth", "memory"))
def test_resource_rejection(
    mode: Literal["swap", "trash", "decode"], budget: str
) -> None:
    with pytest.raises(PlanningError if budget == "memory" else synth.SynthesisError):
        synth.synthesize(
            {"swap": ae.swap_encoder, "trash": ae.encoder, "decode": ae.decoder}[mode],
            classical_inputs={
                "data": [0.0, 0.0, 1.0, 1.0],
                "weights": ae.initial_weights().tolist(),
            },
            max_qubits=3 if budget == "width" else None,
            max_operations=0 if budget == "operations" else None,
            max_depth=0 if budget == "depth" else None,
            memory_limit_bytes=1 if budget == "memory" else None,
        )


@pytest.mark.parametrize("shots", (0, -1, 65537, True))
def test_bad_shot_budgets(shots: int) -> None:
    with pytest.raises(ValueError, match="shots"):
        ae.readout(torch.zeros(4), torch.zeros(10), cost="swap", shots=shots)


@pytest.mark.parametrize("seed", (-1, True))
def test_bad_seeds(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        ae.readout(torch.zeros(4), torch.zeros(10), cost="trash", seed=seed)


@pytest.mark.parametrize(
    "data,weights",
    (
        (torch.zeros(3), torch.zeros(10)),
        (torch.zeros(4), torch.zeros(9)),
        (torch.full((4,), float("nan")), torch.zeros(10)),
        (torch.zeros(4), torch.full((10,), float("inf"))),
    ),
)
def test_bad_quantum_bindings(data: torch.Tensor, weights: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="four data|finite"):
        ae.build_program(data, weights, mode="swap")


def test_figures_and_provenance(workflow: ae.Workflow) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    import numpy as np
    from matplotlib.patches import Rectangle

    figure = ae.plot_workflow(workflow)
    for line, training in zip(
        figure.axes[0].lines, (workflow.swap, workflow.trash), strict=True
    ):
        values = line.get_ydata()
        assert isinstance(values, np.ndarray)
        assert values.tolist() == [step.loss for step in training.steps]
    patches = figure.axes[1].patches
    assert len(patches) == 5
    assert [float(p.get_height()) for p in patches if isinstance(p, Rectangle)] == [
        row.cost for row in workflow.anomalies
    ]
    provenance = json.loads((EXAMPLE / "provenance.json").read_text())
    assert (
        provenance["original_sha256"]
        == "d788d3380b4b275739c0c0e6a90da97de3fa9aec3ce91a21f0f0c52b649b51c8"
    )
    assert len(provenance["reviewed_code_cells"]) == 28
    assert hashlib.sha256((EXAMPLE / "walkthrough.ipynb").read_bytes()).hexdigest()


def test_clean_notebook_executes_both_routes() -> None:
    notebook = json.loads((EXAMPLE / "walkthrough.ipynb").read_text())
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(EXAMPLE / "walkthrough.ipynb")
    assert count == 5
    report = namespace["workflow"]
    assert isinstance(report, ae.Workflow) and report.quantum_evaluations == 134
