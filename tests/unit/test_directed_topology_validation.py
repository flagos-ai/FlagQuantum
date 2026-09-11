"""Physical topology validation must not silently coerce wire identifiers."""

import pytest

from flagquantum.compiler.directed_topology import DirectedCouplingMap

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("count", [True, 2.5, "3"])
def test_directed_topology_rejects_noninteger_wire_count(count: object) -> None:
    with pytest.raises(ValueError, match="wire count must be an integer"):
        DirectedCouplingMap(count, ())


@pytest.mark.parametrize("wire", [True, 0.5, "1"])
def test_directed_topology_rejects_noninteger_edge_endpoint(wire: object) -> None:
    with pytest.raises(ValueError, match="endpoints must be integers"):
        DirectedCouplingMap(3, ((wire, 2),))
