"""Check locked pair, QROM prefix and multi-electron model parity."""

import pytest

from flagquantum.algorithms.primitives.grid_potential_resource_model import (
    source_grid_pair_potential_cost,
    source_grid_potential_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


def assert_locked_cost(native, source):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    actual = estimate_component_resources(native)
    expected = get_cost_value(source, QECGatesCost())
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(actual.gates, category) == getattr(expected, category)
    assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "coordinate,poly,target", [(1, 8, 12), (2, 8, 12), (3, 10, 24), (6, 15, 24)]
)
def test_source_pair_potential_graph(coordinate, poly, target):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt import (
        build_qrom_data_for_poly_fit,
        get_inverse_square_root_poly_coeffs,
    )
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PairPotential

    data = build_qrom_data_for_poly_fit(
        2 * coordinate + 2, poly, get_inverse_square_root_poly_coeffs()
    )
    source = PairPotential(
        coordinate, tuple(tuple(map(int, row)) for row in data), poly, target
    )
    native = source_grid_pair_potential_cost(coordinate, poly, target)
    leaves = [
        instance.bloq for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
    ]
    assert [type(leaf).__name__ for leaf in leaves] == [
        call.component.name for call in native.calls for _ in range(call.repetitions)
    ]
    for call, leaf in zip(
        (edge for edge in native.calls for _ in range(edge.repetitions)),
        leaves,
        strict=True,
    ):
        assert_locked_cost(call.component, leaf)
    assert_locked_cost(native, source)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "electrons,grid,poly", [(1, 1, 8), (2, 2, 8), (3, 2, 8), (4, 5, 12)]
)
def test_source_potential_graph_and_pair_count(electrons, grid, poly):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PotentialEnergy

    source = PotentialEnergy(electrons, grid, poly)
    native = source_grid_potential_cost(electrons, grid, poly)
    assert (
        len(native.calls)
        == len(source.decompose_bloq().bloq_instances)
        == electrons * (electrons - 1) // 2
    )
    assert_locked_cost(native, source)
