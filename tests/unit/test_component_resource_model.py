"""Model-only resource accounting, reuse and T conversion."""

import pytest

from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    ModeledResources,
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


def test_model_only_leaf_and_nested_sequential_lifetimes() -> None:
    leaf = ComponentCost("opaque", 8, GateCost(toffoli=2, measurement=1))
    branch = ComponentCost("branch", 9, GateCost(and_bloq=1), (CostCall(leaf, 3, 2),))
    root = ComponentCost(
        "root",
        12,
        GateCost(t=1),
        (CostCall(branch, 7, 1), CostCall(leaf, 2, 6), CostCall(leaf, 0, 100)),
    )
    result = estimate_component_resources(root)
    assert result == ModeledResources(
        GateCost(t=1, toffoli=46, and_bloq=7, measurement=23), 14
    )
    assert result.gates.total_t_count() == 213
    assert result.gates.total_t_count(ts_per_toffoli=7) == 351


@pytest.mark.parametrize("n,b", [(-1, 0), (1, -1), (True, 0)])
def test_invalid_call_metadata(n: int, b: int) -> None:
    component = ComponentCost(
        "invalid", 1, calls=(CostCall(ComponentCost("child", 1), n, b),)
    )
    with pytest.raises(ValueError, match="Call multiplicity"):
        estimate_component_resources(component)
