"""Compare locked Hubbard leaf call graphs, multiplicities and logical costs."""

import pytest

from flagquantum.algorithms.primitives.hubbard_resource_model import (
    source_hubbard_hopping_plaquette_cost,
    source_hubbard_hopping_tile_cost,
    source_hubbard_interaction_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


def assert_source_hubbard_cost(component, source):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    actual = estimate_component_resources(component)
    expected = get_cost_value(source, QECGatesCost())
    for field in (
        "t",
        "clifford",
        "rotation",
        "measurement",
        "and_bloq",
        "toffoli",
        "cswap",
    ):
        assert getattr(actual.gates, field) == getattr(expected, field)
    assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())
    return actual


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "length,angle,coupling", [(2, 0, 4), (4, 0.125, 2), (8, -0.12, 3)]
)
@pytest.mark.parametrize("hamming_weight", [False, True])
def test_hubbard_source_leaf_graphs(length, angle, coupling, hamming_weight):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.trotter.hubbard.hopping import (
        HoppingTile,
        HoppingTileHWP,
    )
    from qualtran.bloqs.chemistry.trotter.hubbard.interaction import (
        Interaction,
        InteractionHWP,
    )
    from qualtran.resource_counting import SympySymbolAllocator

    interaction = InteractionHWP if hamming_weight else Interaction
    hopping = HoppingTileHWP if hamming_weight else HoppingTile
    cases = [
        (
            source_hubbard_interaction_cost(
                length, angle, coupling, hamming_weight=hamming_weight
            ),
            interaction(length, angle, coupling),
        ),
        (
            source_hubbard_hopping_tile_cost(
                length, angle, coupling, pink=False, hamming_weight=hamming_weight
            ),
            hopping(length, angle, coupling, pink=False),
        ),
    ]
    for model, source in cases:
        assert model.qubits == source.signature.n_qubits()
        calls = source.build_call_graph(SympySymbolAllocator())
        assert [(child.component.name, child.repetitions) for child in model.calls] == [
            (type(child).__name__, count) for child, count in calls.items()
        ]
        for child_model, child_source in zip(model.calls, calls, strict=True):
            assert_source_hubbard_cost(child_model.component, child_source)
        assert_source_hubbard_cost(model, source)


@pytest.mark.ftqc_reference
def test_hopping_plaquette_phase_can_be_modeled_as_t():
    pytest.importorskip("qualtran")
    from math import pi

    from qualtran.bloqs.chemistry.trotter.hubbard.hopping import HoppingPlaquette

    angle = pi / 4
    cost = assert_source_hubbard_cost(
        source_hubbard_hopping_plaquette_cost(angle), HoppingPlaquette(angle)
    )
    assert cost.gates.t == 10
    assert cost.gates.rotation == 0


def test_model_validation():
    with pytest.raises(ValueError, match="positive even"):
        source_hubbard_hopping_tile_cost(3, 0.2)
    with pytest.raises(ValueError, match="positive integer"):
        source_hubbard_interaction_cost(0, 0.2, 4)
    with pytest.raises(ValueError, match="finite"):
        source_hubbard_hopping_plaquette_cost(float("nan"))
