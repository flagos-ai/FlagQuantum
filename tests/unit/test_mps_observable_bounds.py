"""Observable contraction must reject wires outside the represented state."""

import pytest

from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("wire", [-1, 2])
@pytest.mark.parametrize("observable", ["single_z", "x", "y", "z"])
def test_mps_observable_rejects_out_of_range_wire(wire: int, observable: str) -> None:
    state = MPSState.zero(2)
    with pytest.raises(ValueError, match="wire index out of range"):
        if observable == "single_z":
            state.expectation_z(wire)
        elif observable == "x":
            state.expectation_ps(x=(wire,))
        elif observable == "y":
            state.expectation_ps(y=(wire,))
        else:
            state.expectation_ps(z=(wire,))
