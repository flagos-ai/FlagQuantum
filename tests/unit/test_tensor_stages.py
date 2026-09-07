import pytest
import torch

from flagquantum.simulation.tensor_contraction import (
    _einsum_pair_by_labels as compatibility_einsum_pair,
)
from flagquantum.simulation.tensor_network.models import (
    PairContractionStep,
    TensorNetworkNode,
)
from flagquantum.simulation.tensor_network.stages import (
    compile_contraction_stages,
    einsum_pair_by_labels,
    einsum_pair_by_labels_with_fallback,
    einsum_pair_pullback,
    einsum_pair_pullback_by_equations,
    execute_contraction_stages,
    execute_pair_steps,
    kahan_add,
)

pytestmark = pytest.mark.unit


def _matrix_nodes() -> tuple[TensorNetworkNode, TensorNetworkNode]:
    return (
        TensorNetworkNode(torch.arange(6.0).reshape(2, 3), (0, 1), name="left"),
        TensorNetworkNode(torch.arange(12.0).reshape(3, 4), (1, 2), name="right"),
    )


def test_legacy_pair_contraction_name_preserves_function_identity():
    assert compatibility_einsum_pair is einsum_pair_by_labels


def test_pair_steps_and_compiled_stages_match_matmul():
    nodes = _matrix_nodes()
    expected = nodes[0].tensor @ nodes[1].tensor
    step = PairContractionStep(
        step=0,
        left="left",
        right="right",
        left_labels=(0, 1),
        right_labels=(1, 2),
        output_labels=(0, 2),
        output_shape=(2, 4),
        estimated_cost=24,
        intermediate_size=8,
    )

    assert torch.equal(execute_pair_steps(nodes, (0, 2), (step,)), expected)

    stage_plan = compile_contraction_stages(nodes, ((0, 1, (0, 2)),))
    stage_result = execute_contraction_stages(nodes, stage_plan)
    assert stage_result.labels == (0, 2)
    assert torch.equal(stage_result.tensor, expected)


def test_pair_fallback_is_owned_by_simulation(monkeypatch):
    def reject_fused_layout(*args, **kwargs):
        raise ValueError("layout-aware fused BMM supports at most 8 axes per group")

    monkeypatch.setattr(
        "flagquantum.simulation.tensor_network.stages.einsum_pair_by_labels",
        reject_fused_layout,
    )
    left, right = _matrix_nodes()

    result, used_fallback = einsum_pair_by_labels_with_fallback(
        left.tensor,
        left.labels,
        right.tensor,
        right.labels,
        (0, 2),
    )

    assert used_fallback is True
    assert torch.equal(result, left.tensor @ right.tensor)


def test_pair_pullback_is_owned_by_simulation():
    torch.manual_seed(17)
    left = torch.randn(2, 3, dtype=torch.complex64, requires_grad=True)
    right = torch.randn(3, 4, dtype=torch.complex64, requires_grad=True)
    output_cotangent = torch.randn(2, 4, dtype=torch.complex64)
    expected_left, expected_right = torch.autograd.grad(
        left @ right,
        (left, right),
        grad_outputs=output_cotangent,
    )

    actual_left, actual_right = einsum_pair_pullback(
        output_cotangent,
        (0, 2),
        left,
        (0, 1),
        right,
        (1, 2),
    )

    torch.testing.assert_close(actual_left, expected_left)
    torch.testing.assert_close(actual_right, expected_right)


def test_batched_compiled_pair_pullback_is_owned_by_simulation():
    torch.manual_seed(23)
    left = torch.randn(2, 3, 4, dtype=torch.complex64, requires_grad=True)
    right = torch.randn(2, 4, 5, dtype=torch.complex64, requires_grad=True)
    output_cotangent = torch.randn(2, 3, 5, dtype=torch.complex64)
    expected_left, expected_right = torch.autograd.grad(
        torch.einsum("bij,bjk->bik", left, right),
        (left, right),
        grad_outputs=output_cotangent,
    )

    actual_left, actual_right = einsum_pair_pullback_by_equations(
        "bik,bjk->bij",
        "bij,bik->bjk",
        output_cotangent,
        left,
        right,
    )

    torch.testing.assert_close(actual_left, expected_left)
    torch.testing.assert_close(actual_right, expected_right)


def test_kahan_accumulation_is_owned_by_simulation():
    values = (torch.tensor(1e8),) + (torch.tensor(3.0),) * 3
    total = compensation = None
    for value in values:
        total, compensation = kahan_add(total, compensation, value)

    naive = sum(values[1:], values[0])
    reference = sum(value.double() for value in values)
    assert total is not None
    assert abs(total.double() - reference) < abs(naive.double() - reference)
