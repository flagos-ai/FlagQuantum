import pytest

from flagquantum.compilation.execution_policy import (
    normalize_execution_state_mode,
    recommend_execution_mode,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        ("distributed_statevector", "statevector"),
        ("noisy_mps", "mps"),
        ("distributed_mps", "mps"),
        ("distributed_tensor_network", "tensor_network"),
        ("density_matrix", "density_matrix"),
    ],
)
def test_auto_execution_modes_normalize_to_state_representations(
    selected: str,
    expected: str,
) -> None:
    assert (
        normalize_execution_state_mode(
            "auto",
            auto_selected_mode=selected,
        )
        == expected
    )


def test_explicit_aliases_normalize_without_accepting_runtime_only_labels() -> None:
    assert normalize_execution_state_mode("adaptive_mps") == "mps"
    assert normalize_execution_state_mode("tn") == "tensor_network"
    with pytest.raises(ValueError, match="state_mode"):
        normalize_execution_state_mode("noisy_mps")


def test_blocked_memory_budget_preserves_existing_fallback_policy() -> None:
    assert (
        recommend_execution_mode(
            "tensor_network",
            world_size=1,
            has_noise=False,
            state_bytes=1024,
            memory_limit_bytes=512,
        )
        == "mps"
    )
    assert (
        recommend_execution_mode(
            "mps",
            world_size=4,
            has_noise=True,
            state_bytes=1024,
            memory_limit_bytes=512,
        )
        == "distributed_statevector"
    )
