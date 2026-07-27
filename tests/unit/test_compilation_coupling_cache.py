from concurrent.futures import ThreadPoolExecutor

import pytest

from flagquantum.compilation.routing import CouplingMap

pytestmark = pytest.mark.unit


def test_adjacency_is_deduplicated_sorted_and_stable() -> None:
    coupling = CouplingMap(
        5,
        ((2, 0), (0, 2), (2, 4), (2, 1), (3, 2)),
    )

    assert coupling.edges == ((0, 2), (2, 4), (1, 2), (2, 3))
    assert coupling.neighbors(2) == (0, 1, 3, 4)
    assert coupling.neighbors(0) == (2,)
    assert coupling.has_edge(4, 2)


def test_shortest_path_cache_is_bidirectional() -> None:
    coupling = CouplingMap.line(8)

    forward = coupling.shortest_path(0, 7)
    reverse = coupling.shortest_path(7, 0)

    assert forward == tuple(range(8))
    assert reverse == tuple(reversed(forward))
    assert coupling._path_cache == {
        (0, 7): forward,
        (7, 0): reverse,
    }


def test_repeated_large_grid_routes_reuse_cached_tuple() -> None:
    coupling = CouplingMap.grid(20, 20)

    first = coupling.shortest_path(0, 399)
    second = coupling.shortest_path(0, 399)

    assert second is first
    assert len(first) == 39
    assert len(coupling._path_cache) == 2


def test_path_cache_is_bounded_and_reports_evictions() -> None:
    coupling = CouplingMap.line(10, path_cache_capacity=4)

    coupling.shortest_path(0, 9)
    coupling.shortest_path(1, 8)
    coupling.shortest_path(2, 7)

    info = coupling.path_cache_info()
    assert info == {
        "capacity": 4,
        "size": 4,
        "hits": 0,
        "misses": 3,
        "evictions": 2,
    }


def test_zero_capacity_disables_storage_but_retains_diagnostics() -> None:
    coupling = CouplingMap.line(4, path_cache_capacity=0)

    coupling.shortest_path(0, 3)
    coupling.shortest_path(0, 3)

    assert coupling.path_cache_info() == {
        "capacity": 0,
        "size": 0,
        "hits": 0,
        "misses": 2,
        "evictions": 0,
    }


def test_concurrent_path_queries_are_safe_and_bounded() -> None:
    coupling = CouplingMap.grid(10, 10, path_cache_capacity=16)
    pairs = tuple((index, 99 - index) for index in range(20))
    workload = tuple(pair for pair in pairs for _ in range(4))

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = tuple(
            executor.map(
                lambda pair: coupling.shortest_path(*pair),
                workload,
            )
        )

    assert all(
        path[0] == start and path[-1] == goal
        for path, (start, goal) in zip(paths, workload)
    )
    info = coupling.path_cache_info()
    assert info["size"] <= info["capacity"] == 16
    assert info["hits"] > 0
    assert info["evictions"] > 0


def test_invalid_cache_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero or at least two"):
        CouplingMap.line(4, path_cache_capacity=1)
