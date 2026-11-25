"""Verify the full native checkpoint-backed subset-majority training workflow."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.main import (
    build_program,
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import execute_code_cells

try:
    from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.overview import (
        workflow as hybrid,
    )
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "matplotlib"}:
        raise
    pytest.skip(str(error), allow_module_level=True)

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (
    ROOT
    / "examples/synthesis/algorithms/machine_learning/hybrid_qnn/hybrid_qnn_for_subset_majority/overview"
)


def _reference_probabilities(
    features: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    circuit = fq.Circuit(2, dtype=torch.complex128)
    circuit.rx(1, theta=math.pi * features[0]).rz(1, theta=math.pi * features[1])
    circuit.rx(0, theta=math.pi * features[2]).rz(0, theta=math.pi * features[3])
    circuit.ry(1, theta=weights[0]).ry(0, theta=weights[1]).rzz(1, 0, theta=weights[2])
    state = fq.run(circuit).statevector()
    assert isinstance(state, torch.Tensor)
    return state.flatten().abs().square()


def _reference_readout(
    features: torch.Tensor,
    weights: torch.Tensor,
    *,
    shots: int | None = hybrid.SHOTS,
    seed: int = 0,
) -> torch.Tensor:
    probabilities = _reference_probabilities(features, weights)
    if shots is not None:
        samples = torch.multinomial(
            probabilities,
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        probabilities = torch.bincount(samples, minlength=4).to(torch.float64) / shots
    return torch.stack(
        (probabilities[1] + probabilities[3], probabilities[2] + probabilities[3])
    )


@pytest.fixture(scope="module")
def workflow() -> hybrid.Workflow:
    return hybrid.run_workflow()


@pytest.mark.parametrize("scale", (1.0, -0.63))
def test_full_states_clean_work_and_every_quantum_derivative(scale: float) -> None:
    features = scale * torch.tensor([0.2, -0.3, 0.4, -0.5], dtype=torch.float64)
    weights = scale * torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    compare_states(
        build_program(features.tolist(), weights.tolist()),
        reference_circuit(features.tolist(), weights.tolist()),
        ("result",),
    )
    actual = hybrid.quantum_jacobians(features, weights, shots=None)
    x = features.clone().requires_grad_()
    w = weights.clone().requires_grad_()
    output = _reference_readout(x, w, shots=None)
    rows = [
        torch.autograd.grad(output[index], (x, w), retain_graph=True)
        for index in range(2)
    ]
    expected = (
        torch.stack([row[0] for row in rows]),
        torch.stack([row[1] for row in rows]),
    )
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=0)
    assert float(actual[0].abs().max()) > 0.1
    for seed in (3, 17):
        torch.testing.assert_close(
            hybrid.quantum_readout(features, weights, seed=seed),
            _reference_readout(features, weights, seed=seed),
            atol=0,
            rtol=0,
        )
        sampled = hybrid.quantum_jacobians(features, weights, seed=seed)
        assert (
            max(
                float((left - right).abs().max())
                for left, right in zip(sampled, expected, strict=True)
            )
            < 0.12
        )


def test_original_dataset_checkpoint_and_split() -> None:
    data = hybrid.make_dataset()
    assert data.training.shape == (340, 10) and data.test.shape == (512, 10)
    assert int(data.training_labels.sum()) == 123 and int(data.test_labels.sum()) == 173
    assert (
        hashlib.sha256(bytes(data.training.clone().untyped_storage())).hexdigest()
        == "800453c16f2eead0dab8e74e563fd10fe8efa07ca5eb4d90f6d27a682886b442"
    )
    assert not set(map(tuple, data.training.tolist())) & set(
        map(tuple, data.test.tolist())
    )
    for values, labels in (
        (data.training, data.training_labels),
        (data.test, data.test_labels),
    ):
        torch.testing.assert_close(
            labels[:, 0], (values[:, [1, 3, 4, 6, 7, 9]].sum(1) > 3).to(torch.float64)
        )
    checkpoint = json.loads((EXAMPLE / "checkpoint.json").read_text())
    network = hybrid.load_network()
    for name, parameter in network.named_parameters():
        source = checkpoint["qlayer.weight" if name == "quantum_weight" else name]
        torch.testing.assert_close(
            parameter, torch.tensor(source, dtype=torch.float64), atol=0, rtol=0
        )


def test_entire_exact_classical_gradient_chain_and_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = hybrid.load_network()
    data = hybrid.make_dataset()
    native = hybrid.quantum_readout
    monkeypatch.setattr(
        hybrid, "quantum_readout", lambda x, w, **kwargs: native(x, w, shots=None)
    )
    # Independent fully-autograd Circuit loss differentiates both classical stages.
    reference = hybrid.load_network()
    encoded = reference.encode(data.training[:2])
    output = torch.stack(
        [
            _reference_readout(row, reference.quantum_weight, shots=None)
            for row in encoded
        ]
    )
    loss = torch.nn.functional.mse_loss(
        reference.decode(output), data.training_labels[:2]
    )
    expected = torch.autograd.grad(loss, tuple(reference.parameters()))
    step = hybrid._step(
        network,
        torch.optim.SGD(network.parameters(), lr=0.05),
        data.training[:2],
        data.training_labels[:2],
        3,
    )
    assert step.loss == pytest.approx(float(loss.detach()), abs=1e-12)
    assert float(step.gradients["linear_1.weight"].norm()) > 0.1
    for (name, parameter), gradient in zip(
        reference.named_parameters(), expected, strict=True
    ):
        torch.testing.assert_close(step.gradients[name], gradient, atol=1e-12, rtol=0)
        torch.testing.assert_close(
            step.parameters_after[name],
            parameter.detach() - 0.05 * gradient,
            atol=1e-12,
            rtol=0,
        )


def test_full_sampled_epoch_and_all_updates_against_independent_circuit(
    workflow: hybrid.Workflow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hybrid, "quantum_readout", _reference_readout)
    independent = hybrid.run_workflow()
    assert [step.size for step in workflow.steps] == [64] * 5 + [20]
    assert (
        workflow.quantum_evaluations
        == 340 * 15 + 512
        == independent.quantum_evaluations
    )
    assert workflow.predictions.shape == (512, 1)
    torch.testing.assert_close(
        workflow.predictions, independent.predictions, atol=1e-12, rtol=0
    )
    assert workflow.accuracy == independent.accuracy
    for actual, expected in zip(workflow.steps, independent.steps, strict=True):
        assert actual.loss == pytest.approx(expected.loss, abs=1e-12)
        for field in ("parameters_before", "gradients", "parameters_after"):
            left, right = getattr(actual, field), getattr(expected, field)
            for name in left:
                torch.testing.assert_close(left[name], right[name], atol=1e-12, rtol=0)
        for name in actual.gradients:
            torch.testing.assert_close(
                actual.parameters_after[name],
                actual.parameters_before[name] - 0.05 * actual.gradients[name],
                atol=1e-12,
                rtol=0,
            )
    changed = {
        name: float(
            (
                workflow.steps[-1].parameters_after[name]
                - workflow.steps[0].parameters_before[name]
            ).norm()
        )
        for name in workflow.steps[0].parameters_before
    }
    assert changed["linear_1.weight"] > 0.001 and changed["quantum_weight"] > 0.001


def test_source_figures_and_minimal_budget(workflow: hybrid.Workflow) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    import numpy as np
    from matplotlib.patches import Rectangle

    figure = hybrid.plot_workflow(workflow)
    losses = figure.axes[0].lines[0].get_ydata()
    assert isinstance(losses, np.ndarray)
    assert losses.tolist() == [step.loss for step in workflow.steps]
    patches = figure.axes[1].patches
    assert all(isinstance(p, Rectangle) for p in patches)
    assert (
        sum(float(p.get_height()) for p in patches if isinstance(p, Rectangle)) == 512
    )
    assert workflow.program.circuit.n_wires == 2
    resources = workflow.program.report["resources"]
    assert isinstance(resources, ResourceEstimate) and resources.depth > 0
    minimal = synth.synthesize(
        main,
        classical_inputs={
            "input_vec": [0.2, -0.3, 0.4, -0.5],
            "weight": [0.1, 0.2, 0.3],
        },
        max_qubits=2,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
    assert minimal.circuit.n_wires == 2


@pytest.mark.parametrize("budget", ("width", "operations", "depth", "memory"))
def test_budget_refusal(budget: str) -> None:
    expected = PlanningError if budget == "memory" else synth.SynthesisError
    with pytest.raises(expected):
        synth.synthesize(
            main,
            classical_inputs={
                "input_vec": [0.2, -0.3, 0.4, -0.5],
                "weight": [0.1, 0.2, 0.3],
            },
            max_qubits=1 if budget == "width" else None,
            max_operations=0 if budget == "operations" else None,
            max_depth=0 if budget == "depth" else None,
            memory_limit_bytes=1 if budget == "memory" else None,
        )


@pytest.mark.parametrize("shots", (0, -1, 65537, True))
def test_invalid_shots(shots: int) -> None:
    with pytest.raises(ValueError, match="shots"):
        hybrid.quantum_readout(torch.zeros(4), torch.zeros(3), shots=shots)


@pytest.mark.parametrize("seed", (-1, True, 2**31))
def test_invalid_workflow_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        hybrid.run_workflow(seed=seed)


@pytest.mark.parametrize(
    "features,weights",
    (
        (torch.zeros(3), torch.zeros(3)),
        (torch.zeros(4), torch.zeros(2)),
        (torch.tensor([float("nan"), 0.0, 0.0, 0.0]), torch.zeros(3)),
        (torch.zeros(4), torch.tensor([0.0, float("inf"), 0.0])),
    ),
)
def test_invalid_quantum_bindings(
    features: torch.Tensor, weights: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="four features|finite"):
        hybrid.quantum_readout(features, weights)


@pytest.mark.parametrize("seed", (-1, True))
def test_invalid_quantum_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        hybrid.quantum_readout(torch.zeros(4), torch.zeros(3), seed=seed)


def test_clean_notebook_cells() -> None:
    path = EXAMPLE / "walkthrough.ipynb"
    source = json.loads(path.read_text())
    assert all(
        cell.get("execution_count") is None and cell.get("outputs", []) == []
        for cell in source["cells"]
        if cell["cell_type"] == "code"
    )
    namespace, count = execute_code_cells(path)
    assert count == 5
    actual = namespace["workflow"]
    assert isinstance(actual, hybrid.Workflow) and actual.quantum_evaluations == 5612
