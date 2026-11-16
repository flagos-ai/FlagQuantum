"""Compare locked Hubbard leaf call graphs, multiplicities and logical costs."""

import pytest

from flagquantum.algorithms.primitives.hubbard_resource_model import (
    source_hubbard_hopping_plaquette_cost,
    source_hubbard_hopping_plaquette_signature,
    source_hubbard_hopping_tile_cost,
    source_hubbard_hopping_tile_signature,
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


@pytest.mark.ftqc_reference
def test_hubbard_hopping_declared_registers_and_child_parameters():
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, QAny, QBit, Side
    from qualtran.bloqs.chemistry.trotter.hubbard.hopping import (
        HoppingPlaquette,
        HoppingTile,
        HoppingTileHWP,
    )
    from qualtran.resource_counting import SympySymbolAllocator
    from sympy import Symbol

    kappa, eps = -0.375, 1e-6
    plaquette = HoppingPlaquette(kappa, eps=eps)
    (port,) = source_hubbard_hopping_plaquette_signature()
    (original,) = plaquette.signature
    assert port.dtype.encoding == "bit" and isinstance(original.dtype, QBit)
    assert (port.name, port.bitsize, port.shape) == (
        original.name,
        original.bitsize,
        original.shape,
    )
    assert original.side == Side.THRU
    assert tuple(port.all_idxs()) == tuple(original.all_idxs())
    (fourier, count), (rz, rotations) = plaquette.build_call_graph(
        SympySymbolAllocator()
    ).items()
    assert (fourier.k, fourier.n, fourier.eps, count) == (0, 1, eps, 4)
    assert (rz.angle, rz.eps, rotations) == (kappa, eps, 2)
    with pytest.raises(DecomposeNotImplementedError):
        plaquette.decompose_bloq()

    for tile in (HoppingTile, HoppingTileHWP):
        symbolic = Symbol("L", integer=True, positive=True)
        for length in (2, 8, symbolic):
            signature = source_hubbard_hopping_tile_signature(length)
            source = tile(length, kappa / 2, tau=2, eps=eps)
            (original,) = source.signature
            (port,) = signature
            assert signature.get_left("system") == signature.get_right("system")
            assert original.side == Side.THRU and isinstance(original.dtype, QAny)
            assert port.dtype.encoding == "opaque"
            assert (port.name, port.bitsize, port.shape) == (
                original.name,
                original.bitsize,
                original.shape,
            )
            assert signature.n_qubits() == source.signature.n_qubits()

        calls = tile(8, kappa / 2, tau=2, eps=eps).build_call_graph(
            SympySymbolAllocator()
        )
        if tile is HoppingTile:
            assert calls == {plaquette: 32}
        else:
            (fourier, count), (phasing, batches) = calls.items()
            assert (fourier.k, fourier.n, fourier.eps, count) == (0, 1, eps, 128)
            assert (phasing.bitsize, phasing.exponent, phasing.eps, batches) == (
                32,
                kappa,
                eps,
                2,
            )
