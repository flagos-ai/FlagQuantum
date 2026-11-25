"""Subset-majority data, exact readout, gradients and complete training replay."""

import copy
import itertools
import math
import random

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.main import (
    build_program,
)
from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.workflow import (
    SubsetMajorityNetwork,
    evaluate,
    load_pretrained,
    make_datasets,
    quantum_circuit,
    train,
)
from examples.synthesis.compare_states import compare_states

pytestmark = pytest.mark.integration


def analytic_readout(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # RZZ commutes with Z on both sites; it changes phases but not these marginals.
    polar = math.pi * features[:, (0, 2)]
    azimuth = math.pi * features[:, (1, 3)]
    z = polar.cos() * weights[:2].cos()
    z = z - polar.sin() * azimuth.sin() * weights[:2].sin()
    return (1 - z) / 2


class AnalyticNetwork(SubsetMajorityNetwork):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.activation_1(self.linear_1(self.flatten(features)))
        probabilities = analytic_readout(encoded, self.qlayer.parameters_tensor)
        return self.activation_2(self.linear_2(probabilities))


@pytest.fixture
def network() -> SubsetMajorityNetwork:
    with torch.random.fork_rng():
        torch.manual_seed(1)
        network = SubsetMajorityNetwork()
    load_pretrained(network)
    return network


def test_source_data_split_and_strict_subset_majority() -> None:
    train_data, test_data = make_datasets()
    rows = list(itertools.product((0, 1), repeat=10))
    random.Random(1).shuffle(rows)
    for dataset, expected in ((train_data, rows[:340]), (test_data, rows[340:852])):
        x, y = dataset.tensors
        assert x.tolist() == [list(row) for row in expected]
        assert y.flatten().tolist() == [
            int(sum(row[i] for i in (1, 3, 4, 6, 7, 9)) >= 4) for row in expected
        ]
    assert len(set(rows[:340]) & set(rows[340:852])) == 0


@pytest.mark.parametrize("scale", [1.0, -0.63, 8.7])
def test_trainable_circuit_preserves_complete_synthesized_state(scale: float) -> None:
    features = torch.tensor([[0.2, -0.3, 0.4, -0.5]], dtype=torch.float64) * scale
    weights = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64) * scale
    compare_states(
        build_program(features[0].tolist(), weights.tolist()),
        quantum_circuit(weights, features),
        ("result",),
    )


def test_full_network_forward_and_gradients_match_independent_bloch_reference(
    network: SubsetMajorityNetwork,
) -> None:
    reference = AnalyticNetwork()
    reference.load_state_dict(copy.deepcopy(network.state_dict()))
    dataset, _ = make_datasets()
    # Include an input gradient: classical-encoder training must cross the quantum layer.
    inputs = dataset.tensors[0][:7].clone().requires_grad_()
    expected_inputs = inputs.detach().clone().requires_grad_()
    actual = network(inputs)
    expected = reference(expected_inputs)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    labels = dataset.tensors[1][:7]
    nn.functional.mse_loss(actual, labels).backward()
    nn.functional.mse_loss(expected, labels).backward()
    torch.testing.assert_close(inputs.grad, expected_inputs.grad, atol=1e-12, rtol=0)
    for (name, parameter), (_, expected_parameter) in zip(
        network.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name
        torch.testing.assert_close(
            parameter.grad, expected_parameter.grad, atol=1e-12, rtol=0
        )
    assert network.linear_1.weight.grad.norm() > 0
    assert network.linear_2.weight.grad.norm() > 0
    quantum_gradient = network.qlayer.parameters_tensor.grad
    assert bool((quantum_gradient[:2].abs() > 1e-8).all())
    assert abs(float(quantum_gradient[2])) < 1e-12


def test_quantum_input_and_parameter_gradients_pass_finite_differences(
    network: SubsetMajorityNetwork,
) -> None:
    features = torch.tensor(
        [[0.2, -0.3, 0.4, -0.5], [-0.1, 0.8, -0.2, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = network.qlayer.parameters_tensor.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(
        lambda data, theta: (1 - network.qlayer(data, parameters=theta)) / 2,
        (features, weights),
        eps=1e-6,
        atol=1e-8,
        rtol=1e-6,
    )


def test_complete_checkpoint_training_and_held_out_evaluation(
    network: SubsetMajorityNetwork,
) -> None:
    reference = AnalyticNetwork()
    reference.load_state_dict(copy.deepcopy(network.state_dict()))
    training, testing = make_datasets()
    before = evaluate(network, testing)
    initial_weights = {
        name: p.detach().clone() for name, p in network.named_parameters()
    }
    history = train(network, training)
    reference_history = train(reference, training)
    assert history == pytest.approx(reference_history, abs=1e-12, rel=0)
    for (name, parameter), (_, expected_parameter) in zip(
        network.named_parameters(), reference.named_parameters(), strict=True
    ):
        torch.testing.assert_close(parameter, expected_parameter, atol=1e-12, rtol=0)
        if name != "qlayer.parameters_tensor":
            assert not torch.equal(parameter, initial_weights[name]), name
    delta = (
        network.qlayer.parameters_tensor.detach()
        - initial_weights["qlayer.parameters_tensor"]
    )
    assert bool((delta[:2].abs() > 1e-8).all())
    assert abs(float(delta[2])) < 1e-12
    after = evaluate(network, testing)
    assert after.samples == 512
    assert after.loss < before.loss
    assert after.accuracy >= 0.8
    assert after.loss == pytest.approx(evaluate(reference, testing).loss, abs=1e-12)
    assert after.accuracy == evaluate(reference, testing).accuracy
    # Match the notebook's Python round(), including ties and outputs outside [0, 1].
    predictions = []
    with torch.no_grad():
        for features, _ in DataLoader(testing, batch_size=64):
            predictions.extend(float(value) for value in network(features).flatten())
    labels = testing.tensors[1].flatten().tolist()
    assert (
        after.accuracy
        == sum(round(p) == y for p, y in zip(predictions, labels, strict=True)) / 512
    )


def test_thirty_epochs_from_scratch_update_the_complete_network() -> None:
    with torch.random.fork_rng():
        torch.manual_seed(1)
        network = SubsetMajorityNetwork()
    training, testing = make_datasets()
    before = evaluate(network, testing)
    initial = {
        name: value.detach().clone() for name, value in network.named_parameters()
    }
    history = train(network, training, epochs=30)
    assert len(history) == 30
    assert all(math.isfinite(loss) for loss in history)
    after = evaluate(network, testing)
    assert after.loss < before.loss
    assert after.samples == 512
    for name, parameter in network.named_parameters():
        if name == "qlayer.parameters_tensor":
            assert not torch.equal(initial[name][:2], parameter[:2])
        else:
            assert not torch.equal(initial[name], parameter), name


@pytest.mark.parametrize("epochs", [0, -1, True, 1.5])
def test_invalid_epoch_counts_fail_before_updating(
    network: SubsetMajorityNetwork,
    epochs: int,
) -> None:
    training, _ = make_datasets()
    with pytest.raises(ValueError, match="positive integer"):
        train(network, training, epochs=epochs)
