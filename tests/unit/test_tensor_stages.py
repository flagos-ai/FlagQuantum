import pytest
import torch

from flagquantum.simulation.tensor_contraction import (
    _einsum_pair_by_labels as compatibility_einsum_pair,
)
from flagquantum.simulation.tensor_models import PairContractionStep, TensorNetworkNode
from flagquantum.simulation.tensor_stages import (
    compile_contraction_stages,
    einsum_pair_by_labels,
    einsum_pair_by_labels_with_fallback,
    execute_contraction_stages,
    execute_pair_steps,
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
        "flagquantum.simulation.tensor_stages.einsum_pair_by_labels",
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
