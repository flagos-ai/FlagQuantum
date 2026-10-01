from concurrent.futures import ThreadPoolExecutor

import pytest

from flagquantum.compiler import CouplingMap
from flagquantum.compiler.routing import UNREACHABLE_DISTANCE

pytestmark = pytest.mark.unit


def test_distance_matches_shortest_path_hop_count() -> None:
    coupling = CouplingMap.grid(3, 4)

    for left in range(coupling.n_wires):
        for right in range(coupling.n_wires):
            assert (
                coupling.distance(left, right)
                == len(coupling.shortest_path(left, right)) - 1
            )


def test_distance_matrix_is_symmetric_with_zero_diagonal() -> None:
    coupling = CouplingMap.grid(3, 3)

    matrix = coupling.distance_matrix()

    assert matrix == (
        (0, 1, 2, 1, 2, 3, 2, 3, 4),
        (1, 0, 1, 2, 1, 2, 3, 2, 3),
        (2, 1, 0, 3, 2, 1, 4, 3, 2),
        (1, 2, 3, 0, 1, 2, 1, 2, 3),
        (2, 1, 2, 1, 0, 1, 2, 1, 2),
        (3, 2, 1, 2, 1, 0, 3, 2, 1),
        (2, 3, 4, 1, 2, 3, 0, 1, 2),
        (3, 2, 3, 2, 1, 2, 1, 0, 1),
        (4, 3, 2, 3, 2, 1, 2, 1, 0),
    )
    for left in range(coupling.n_wires):
        for right in range(coupling.n_wires):
            assert matrix[left][right] == matrix[right][left]
    assert all(matrix[wire][wire] == 0 for wire in range(coupling.n_wires))


def test_ring_geometry_reports_the_shorter_arc() -> None:
    coupling = CouplingMap.ring(8)

    assert coupling.distance(0, 4) == 4
    assert coupling.distance(0, 5) == 3
    assert coupling.distance(6, 1) == 3


def test_disconnected_pairs_are_marked_and_raise_on_query() -> None:
    coupling = CouplingMap(5, ((0, 1), (2, 3)))

    matrix = coupling.distance_matrix()

    assert matrix[0] == (
        0,
        1,
        UNREACHABLE_DISTANCE,
        UNREACHABLE_DISTANCE,
        UNREACHABLE_DISTANCE,
    )
    assert matrix[2] == (
        UNREACHABLE_DISTANCE,
        UNREACHABLE_DISTANCE,
        0,
        1,
        UNREACHABLE_DISTANCE,
    )
    assert coupling.distance(0, 1) == 1
    assert coupling.distance(2, 3) == 1
    with pytest.raises(ValueError, match="No coupling path between wires 0 and 4"):
        coupling.distance(0, 4)


def test_self_distance_is_zero_without_a_path_search() -> None:
    coupling = CouplingMap(4, ((0, 1),))

    assert coupling.distance(3, 3) == 0
    assert coupling.distance_cache_info()["misses"] == 1


def test_distance_index_stores_one_row_per_wire_not_one_entry_per_pair() -> None:
    coupling = CouplingMap.grid(8, 8)

    matrix = coupling.distance_matrix()

    info = coupling.distance_cache_info()
    assert len(matrix) == 64
    assert info == {
        "capacity": 4096,
        "size": 64,
        "hits": 0,
        "misses": 64,
        "evictions": 0,
    }
    # The pair-keyed path cache is a different index and stays untouched.
    assert coupling.path_cache_info() == {
        "capacity": 4096,
        "size": 0,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
    }


def test_repeated_distance_queries_cost_one_search_per_source_wire() -> None:
    coupling = CouplingMap.grid(20, 20)

    for left in range(coupling.n_wires):
        for right in range(coupling.n_wires):
            coupling.distance(left, right)

    info = coupling.distance_cache_info()
    assert info["misses"] == 400
    assert info["hits"] == 400 * 400 - 400
    assert info["evictions"] == 0


def test_distance_row_cache_is_evicted_beyond_capacity() -> None:
    coupling = CouplingMap.line(6, path_cache_capacity=2)

    coupling.distance(0, 5)
    coupling.distance(1, 5)
    coupling.distance(2, 5)

    info = coupling.distance_cache_info()
    assert info["capacity"] == 2
    assert info["size"] == 2
    assert info["misses"] == 3
    assert info["evictions"] == 1


def test_zero_capacity_recomputes_distances_without_storing_rows() -> None:
    coupling = CouplingMap.line(4, path_cache_capacity=0)

    assert coupling.distance(0, 3) == 3
    assert coupling.distance(0, 3) == 3

    assert coupling.distance_cache_info() == {
        "capacity": 0,
        "size": 0,
        "hits": 0,
        "misses": 2,
        "evictions": 0,
    }
    assert coupling.distance_matrix() == (
        (0, 1, 2, 3),
        (1, 0, 1, 2),
        (2, 1, 0, 1),
        (3, 2, 1, 0),
    )


def test_distance_rejects_wires_outside_the_device() -> None:
    coupling = CouplingMap.line(3)

    with pytest.raises(ValueError, match="outside"):
        coupling.distance(-1, 0)
    with pytest.raises(ValueError, match="outside"):
        coupling.distance(0, 3)
    with pytest.raises(ValueError, match="must be an integer"):
        coupling.distance(0.0, 1)  # type: ignore[arg-type]


def test_concurrent_distance_queries_are_safe_and_consistent() -> None:
    coupling = CouplingMap.grid(10, 10, path_cache_capacity=16)
    reference = CouplingMap.grid(10, 10).distance_matrix()
    pairs = tuple((index, 99 - index) for index in range(20))
    workload = tuple(pair for pair in pairs for _ in range(4))

    with ThreadPoolExecutor(max_workers=8) as executor:
        distances = tuple(executor.map(lambda pair: coupling.distance(*pair), workload))

    assert distances == tuple(reference[left][right] for left, right in workload)
    info = coupling.distance_cache_info()
    assert info["size"] <= info["capacity"] == 16


def test_distance_index_does_not_change_routing_or_its_evidence() -> None:
    from flagquantum.compiler import route_to_topology
    from flagquantum.core.ir import CircuitIR, Instruction

    ir = CircuitIR(
        n_wires=8,
        instructions=tuple(
            Instruction("cx", (index, (index * 5 + 3) % 8)) for index in range(6)
        ),
    )
    reference = CouplingMap.grid(2, 4)
    probed = CouplingMap.grid(2, 4)

    routed_reference = route_to_topology(ir, reference)
    # Warm the distance index first: routing evidence must not depend on it.
    probed.distance_matrix()
    routed_probed = route_to_topology(ir, probed)

    assert routed_probed == routed_reference
    assert probed.path_cache_info() == reference.path_cache_info()
