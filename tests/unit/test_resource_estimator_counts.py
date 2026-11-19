"""Check known-count conversion independently of optional estimator packages."""

import pytest

from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.ecosystem.resource_estimator import microsoft_logical_counts

pytestmark = pytest.mark.unit


def test_declared_counts_fold_the_existing_legacy_t_budget() -> None:
    logical = ModeledResources(
        GateCost(t=2, toffoli=3, cswap=5, and_bloq=7, rotation=11), 17
    )
    assert microsoft_logical_counts(logical) == {
        "numQubits": 17,
        "tCount": 77,
        "rotationCount": 11,
        "rotationDepth": 11,
    }
    assert microsoft_logical_counts(logical, rotation_depth=4)["rotationDepth"] == 4
    assert microsoft_logical_counts(ModeledResources(GateCost(), 0)) == {
        "numQubits": 0,
        "tCount": 0,
        "rotationCount": 0,
        "rotationDepth": 0,
    }


@pytest.mark.parametrize("gates", (GateCost(t=-1), GateCost(rotation=-1)))
def test_negative_counts_fail_before_export(gates: GateCost) -> None:
    with pytest.raises(ValueError, match="concrete nonnegative"):
        microsoft_logical_counts(ModeledResources(gates, 1))


def test_invalid_rotation_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="concrete nonnegative"):
        microsoft_logical_counts(ModeledResources(GateCost(), 1), rotation_depth=-1)
