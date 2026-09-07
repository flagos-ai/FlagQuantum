import pytest
import torch

from flagquantum.simulation import tensor_contraction, tensor_path_search
from flagquantum.simulation.tensor_network.models import TensorNetworkNode

pytestmark = pytest.mark.unit


def _nodes() -> tuple[TensorNetworkNode, ...]:
    return (
        TensorNetworkNode(torch.arange(6.0).reshape(2, 3), (0, 1), name="left"),
        TensorNetworkNode(torch.arange(12.0).reshape(3, 4), (1, 2), name="right"),
    )


@pytest.mark.parametrize(
    "name",
    (
        "_contract_nodes_greedy",
        "_contract_nodes_quality_multistart",
        "_contract_nodes_quality_reconfigured",
        "_contract_nodes_beam",
        "_contract_nodes_optimal",
        "_tree_from_steps",
        "_linearize_contraction_tree",
    ),
)
def test_compatibility_names_preserve_path_search_function_identity(name):
    assert getattr(tensor_contraction, name) is getattr(tensor_path_search, name)


def test_extracted_path_search_modes_preserve_contraction_result():
    nodes = _nodes()
    expected = nodes[0].tensor @ nodes[1].tensor

    greedy, greedy_steps = tensor_path_search._contract_nodes_greedy(nodes, (0, 2))
    beam, beam_steps = tensor_path_search._contract_nodes_beam(nodes, (0, 2))
    optimal, optimal_steps = tensor_path_search._contract_nodes_optimal(nodes, (0, 2))

    assert torch.equal(greedy, expected)
    assert torch.equal(beam, expected)
    assert torch.equal(optimal, expected)
    assert greedy_steps == beam_steps == optimal_steps
