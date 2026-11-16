"""Check locked pair, QROM prefix and multi-electron model parity."""

from itertools import combinations

import pytest

from flagquantum.algorithms.primitives.grid_potential_resource_model import (
    source_grid_pair_potential_cost,
    source_grid_pair_potential_signature,
    source_grid_potential_cost,
    source_grid_potential_signature,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


def assert_locked_cost(native, source, legacy=False):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    actual = estimate_component_resources(native)
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
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


def assert_source_signature(native, original):
    from qualtran import QAny, Side

    assert native.thru_registers_only
    for actual, expected in zip(native, original, strict=True):
        assert actual.dtype.encoding == "opaque"
        assert isinstance(expected.dtype, QAny)
        assert expected.side == Side.THRU
        assert (actual.name, actual.bitsize, actual.shape) == (
            expected.name,
            expected.bitsize,
            expected.shape,
        )
        assert tuple(actual.all_idxs()) == tuple(expected.all_idxs())


def assert_pair_operands(composite):
    from qualtran import LeftDangle, Soquet

    steps = {}
    for instance, inputs, outputs in composite.iter_bloqsoqs():
        steps.setdefault(type(instance.bloq).__name__, []).append((inputs, outputs))

    forward = steps["OutOfPlaceAdder"][:3]
    inverse = steps["OutOfPlaceAdder"][3:]
    square_inputs, (axes, distance) = steps["SumOfSquares"][0]
    assert tuple(square_inputs["input"]) == tuple(out[2] for _, out in forward)
    final = composite.final_soqs()
    for axis, ((inputs, (left, right, _)), (undo_inputs, outputs)) in enumerate(
        zip(forward, inverse, strict=True)
    ):
        assert inputs == {
            "a": Soquet(LeftDangle, composite.signature.get_left("system_i"), (axis,)),
            "b": Soquet(LeftDangle, composite.signature.get_left("system_j"), (axis,)),
        }
        assert undo_inputs == {"a": left, "b": right, "c": axes[axis]}
        assert outputs == (final["system_i"][axis], final["system_j"][axis])

    (cast_inputs, (selection,)), (uncast_inputs, (distance_again,)) = steps["Cast"]
    lookup_inputs, (selected, *coefficients) = steps["QROM"][0]
    polynomial_inputs, (poly_distance, used_coefficients, polynomial) = steps[
        "PolynmomialEvaluationInverseSquareRoot"
    ][0]
    newton_inputs, (newton_distance, newton_poly, estimate) = steps[
        "NewtonRaphsonApproxInverseSquareRoot"
    ][0]
    rotation_inputs, (used_estimate,) = steps["QuantumVariableRotation"][0]
    assert cast_inputs == {"reg": distance}
    assert lookup_inputs["selection"] == selection
    assert uncast_inputs == {"reg": selected}
    assert polynomial_inputs["x_sq"] == distance_again
    assert tuple(polynomial_inputs["in_coeff"]) == tuple(coefficients)
    assert newton_inputs["x_sq"] == poly_distance
    assert newton_inputs["poly"] == polynomial
    assert rotation_inputs == {"phi": estimate}
    assert {inputs["reg"] for inputs, _ in steps["Free"]} == {
        newton_distance,
        newton_poly,
        used_estimate,
        *used_coefficients,
    }
    assert {out[0] for _, out in steps["Allocate"]} == {
        *(lookup_inputs[f"target{bank}_"] for bank in range(4)),
        polynomial_inputs["out"],
        newton_inputs["target"],
    }


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "coordinate,poly,target", [(1, 8, 12), (2, 8, 12), (3, 10, 24), (6, 15, 24)]
)
@pytest.mark.parametrize("legacy", (False, True))
def test_source_pair_potential_graph(coordinate, poly, target, legacy):
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
    native = source_grid_pair_potential_cost(
        coordinate, poly, target, legacy_shims=legacy
    )
    assert_source_signature(
        source_grid_pair_potential_signature(coordinate), source.signature
    )
    composite = source.decompose_bloq()
    leaves = [instance.bloq for instance, _, _ in composite.iter_bloqsoqs()]
    assert_pair_operands(composite)
    # The frozen pair passes the second coordinate directly to an unsigned adder.
    assert leaves[0].on_classical_vals(a=1, b=1)["c"] == 2
    assert [type(leaf).__name__ for leaf in leaves] == [
        call.component.name for call in native.calls for _ in range(call.repetitions)
    ]
    for call, leaf in zip(
        (edge for edge in native.calls for _ in range(edge.repetitions)),
        leaves,
        strict=True,
    ):
        assert_locked_cost(call.component, leaf, legacy)
    assert_locked_cost(native, source, legacy)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "electrons,grid,poly", [(1, 1, 8), (2, 2, 8), (3, 2, 8), (4, 5, 12)]
)
def test_source_potential_graph_and_pair_count(electrons, grid, poly):
    pytest.importorskip("qualtran")
    from qualtran import LeftDangle, Soquet
    from qualtran.bloqs.chemistry.trotter.grid_ham.potential import PotentialEnergy

    source = PotentialEnergy(electrons, grid, poly)
    native = source_grid_potential_cost(electrons, grid, poly)
    signature = source_grid_potential_signature(electrons, grid)
    assert_source_signature(signature, source.signature)
    composite = source.decompose_bloq()
    steps = sorted(composite.iter_bloqsoqs(), key=lambda step: step[0].i)
    pairs = tuple(combinations(range(electrons), 2))
    assert len(native.calls) == len(steps) == len(pairs)
    system = source.signature.get_left("system")
    endpoints = [
        tuple(Soquet(LeftDangle, system, (electron, axis)) for axis in range(3))
        for electron in range(electrons)
    ]
    for (left, right), (_, inputs, outputs) in zip(pairs, steps, strict=True):
        assert tuple(inputs["system_i"]) == endpoints[left]
        assert tuple(inputs["system_j"]) == endpoints[right]
        endpoints[left], endpoints[right] = map(tuple, outputs)
    assert [tuple(row) for row in composite.final_soqs()["system"]] == endpoints
    assert_locked_cost(native, source)


@pytest.mark.parametrize("legacy", (False, True))
def test_pair_potential_consumes_replaceable_shared_addition_models(
    monkeypatch, legacy
):
    import flagquantum.algorithms.primitives.grid_potential_resource_model as module
    from flagquantum.compiler._component_trace.resource_model import (
        ComponentCost,
        GateCost,
    )

    forward = ComponentCost("OutOfPlaceAdder", 17, GateCost(t=13))
    inverse = ComponentCost("OutOfPlaceAdder", 19, GateCost(measurement=7))
    seen = []

    def model(width, *, adjoint=False, legacy_shims=False):
        seen.append((width, adjoint, legacy_shims))
        return inverse if adjoint else forward

    monkeypatch.setattr(module, "source_out_of_place_add_cost", model)
    root = module.source_grid_pair_potential_cost(2, 8, 12, legacy_shims=legacy)
    assert seen == [(2, False, legacy), (2, True, legacy)]
    assert root.calls[0].component is forward and root.calls[0].repetitions == 3
    assert root.calls[2].component is inverse and root.calls[2].repetitions == 3
