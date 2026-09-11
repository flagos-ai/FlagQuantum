"""Topology dimensions and cache limits must not be silently rounded."""

import pytest

from flagquantum.compiler.routing import CouplingMap

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", [False, 2.5, "2"])
@pytest.mark.parametrize("field", ["count", "endpoint", "capacity", "rows", "cols"])
def test_coupling_map_rejects_noninteger_configuration(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match="integer"):
        if field == "count":
            CouplingMap(value, ())
        elif field == "endpoint":
            CouplingMap(4, ((value, 3),))
        elif field == "capacity":
            CouplingMap(2, ((0, 1),), path_cache_capacity=value)
        elif field == "rows":
            CouplingMap.grid(value, 2)
        else:
            CouplingMap.grid(2, value)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
@pytest.mark.parametrize(
    "query", ["neighbors", "edge_left", "edge_right", "path_start", "path_goal"]
)
def test_coupling_queries_reject_noninteger_wires_without_touching_cache(
    query: str, value: object
) -> None:
    coupling = CouplingMap.line(3)
    coupling.shortest_path(1, 2)
    before = coupling.path_cache_info()

    with pytest.raises(ValueError, match="integer"):
        if query == "neighbors":
            coupling.neighbors(value)
        elif query == "edge_left":
            coupling.has_edge(value, 2)
        elif query == "edge_right":
            coupling.has_edge(2, value)
        elif query == "path_start":
            coupling.shortest_path(value, 2)
        else:
            coupling.shortest_path(2, value)

    assert coupling.path_cache_info() == before


@pytest.mark.parametrize("wire", [-1, 3])
def test_out_of_range_self_edges_are_rejected(wire: int) -> None:
    with pytest.raises(ValueError, match="outside the device"):
        CouplingMap(3, ((0, 1), (wire, wire)))


def test_valid_self_edges_do_not_change_connectivity() -> None:
    coupling = CouplingMap(3, ((0, 0), (0, 1), (2, 2)))

    assert coupling.edges == ((0, 1),)
    assert coupling.neighbors(0) == (1,)
    assert coupling.neighbors(2) == ()
    assert not coupling.has_edge(0, 0)


@pytest.mark.parametrize("value", [True, 2.5, "3"])
@pytest.mark.parametrize("shape", ["line", "ring"])
def test_topology_factories_validate_count_before_generating_edges(
    shape: str, value: object
) -> None:
    factory = CouplingMap.line if shape == "line" else CouplingMap.ring
    with pytest.raises(ValueError, match="wire count must be an integer"):
        factory(value)
