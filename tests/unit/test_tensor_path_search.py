import pytest
import torch

from flagquantum.simulation.tensor_network import contraction, path_search
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
    assert getattr(contraction, name) is getattr(path_search, name)


def test_extracted_path_search_modes_preserve_contraction_result():
    nodes = _nodes()
    expected = nodes[0].tensor @ nodes[1].tensor

    greedy, greedy_steps = path_search._contract_nodes_greedy(nodes, (0, 2))
    beam, beam_steps = path_search._contract_nodes_beam(nodes, (0, 2))
    optimal, optimal_steps = path_search._contract_nodes_optimal(nodes, (0, 2))

    assert torch.equal(greedy, expected)
    assert torch.equal(beam, expected)
    assert torch.equal(optimal, expected)
    assert greedy_steps == beam_steps == optimal_steps


@pytest.mark.parametrize(
    "dtype", (torch.float32, torch.float64, torch.complex64, torch.complex128)
)
def test_shape_only_tensor_sizes_do_not_allocate_storage(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    expected_element_size = torch.empty((), dtype=dtype).element_size()
    tensor = path_search._DryRunTensor(shape=(2,) * 128, dtype=dtype)

    def reject_allocation(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("shape-only size inspection must not allocate tensors")

    monkeypatch.setattr(torch, "empty", reject_allocation)

    assert tensor.numel() == 1 << 128
    assert tensor.element_size() == expected_element_size


@pytest.mark.parametrize("labels", ((0,), (0, 1, 3)))
@pytest.mark.parametrize("strategy", ("greedy", "beam", "optimal"))
def test_path_search_rejects_label_dimension_mismatch(
    labels: tuple[int, ...], strategy: str
) -> None:
    left, right = _nodes()
    malformed = TensorNetworkNode(left.tensor, labels, name="malformed")
    search = getattr(path_search, f"_contract_nodes_{strategy}")

    with pytest.raises(ValueError, match="labels.*dimensions"):
        search((malformed, right), (0, 2), dry_run=True)


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf")])
@pytest.mark.parametrize("control", ["repeats", "pool size"])
def test_multistart_rejects_noninteger_search_counts(
    value: float | bool, control: str
) -> None:
    with pytest.raises(
        ValueError, match=f"quality multistart {control} must be an integer"
    ):
        if control == "repeats":
            path_search._contract_nodes_quality_multistart(
                _nodes(), (0, 2), repeats=value
            )
        else:
            path_search._contract_nodes_quality_multistart(
                _nodes(), (0, 2), random_pool_size=value
            )


@pytest.mark.parametrize("width", [True, 1.5, float("nan"), float("inf")])
@pytest.mark.parametrize("dry_run", [False, True])
def test_beam_search_rejects_noninteger_width(
    width: float | bool, dry_run: bool
) -> None:
    with pytest.raises(ValueError, match="beam_width must be an integer"):
        path_search._contract_nodes_beam(
            _nodes(), (0, 2), beam_width=width, dry_run=dry_run
        )


@pytest.mark.parametrize("width", [True, 1.5])
@pytest.mark.parametrize("warm_cache", [False, True])
@pytest.mark.parametrize("strategy", ["beam", "beam_sliced"])
def test_profile_cache_does_not_bypass_beam_width_validation(
    monkeypatch: pytest.MonkeyPatch,
    width: float | bool,
    warm_cache: bool,
    strategy: str,
) -> None:
    monkeypatch.setattr(contraction, "_CONTRACTION_PROFILE_CACHE", {})
    nodes = _nodes()
    if warm_cache:
        contraction._contraction_profile(
            nodes, (0, 2), strategy=strategy, beam_width=1, max_intermediate_size=16
        )
    with pytest.raises(ValueError, match="beam_width must be an integer"):
        contraction._contraction_profile(
            nodes, (0, 2), strategy=strategy, beam_width=width, max_intermediate_size=16
        )


@pytest.mark.parametrize("strategy", ["greedy", "beam", "optimal"])
def test_shape_only_output_reordering_never_allocates(
    monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    nodes = _nodes()

    def reject_allocation(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("shape-only output reordering must not allocate")

    monkeypatch.setattr(torch, "empty", reject_allocation)
    result, steps = getattr(path_search, f"_contract_nodes_{strategy}")(
        nodes, (2, 0), dry_run=True
    )
    assert result.shape == (4, 2)
    assert result.numel() == 8
    assert len(steps) == 1


@pytest.mark.parametrize("strategy", ["greedy", "beam", "optimal"])
def test_shape_only_search_exceeds_pytorch_storage_size(strategy: str) -> None:
    nodes = tuple(
        TensorNetworkNode(
            torch.ones(()).expand((2,) * 32),
            tuple(range(index * 32, (index + 1) * 32)),
            name=str(index),
        )
        for index in range(3)
    )
    result, steps = getattr(path_search, f"_contract_nodes_{strategy}")(
        nodes, tuple(reversed(range(96))), dry_run=True
    )
    assert isinstance(result, path_search._DryRunTensor)
    assert result.shape == (2,) * 96
    assert result.numel() == 1 << 96
    assert len(steps) == 2


@pytest.mark.parametrize("strategy", ["greedy", "beam", "optimal"])
def test_real_search_preserves_operand_gradients(strategy: str) -> None:
    left, right = _nodes()
    left.tensor.requires_grad_(True)
    right.tensor.requires_grad_(True)
    result, _ = getattr(path_search, f"_contract_nodes_{strategy}")(
        (left, right), (0, 2)
    )
    assert isinstance(result, torch.Tensor)
    actual = torch.autograd.grad(result.square().sum(), (left.tensor, right.tensor))
    reference = left.tensor @ right.tensor
    expected = torch.autograd.grad(
        reference.square().sum(), (left.tensor, right.tensor)
    )
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)
