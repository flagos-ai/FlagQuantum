"""Check both locked lattice schedules and their resource-only boundary."""

import pytest

from flagquantum.algorithms.primitives.hubbard_resource_model import (
    source_hubbard_interaction_signature,
)
from flagquantum.algorithms.primitives.hubbard_trotter import (
    _suzuki_schedule,
    source_hubbard_suzuki_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_hubbard_resource_model import assert_source_hubbard_cost

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("length", [2, 4, 8])
@pytest.mark.parametrize("hamming_weight", [False, True])
@pytest.mark.parametrize("strip_layer", [False, True])
def test_hubbard_suzuki_matches_ordered_source_calls(
    length, hamming_weight, strip_layer
):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, LeftDangle, QAny, Side, Soquet
    from qualtran.bloqs.chemistry.trotter.hubbard.trotter_step import (
        build_plaq_hwp_unitary_second_order_suzuki,
        build_plaq_unitary_second_order_suzuki,
    )

    factory = (
        build_plaq_hwp_unitary_second_order_suzuki
        if hamming_weight
        else build_plaq_unitary_second_order_suzuki
    )
    coupling, timestep, hopping, eps = 4.0, 0.1234, 0.8, 1e-6
    source = factory(length, coupling, timestep, hopping, eps, strip_layer=strip_layer)
    signature = source_hubbard_interaction_signature(length)
    (port,) = signature
    (source_port,) = source.signature
    assert isinstance(source_port.dtype, QAny) and source_port.side == Side.THRU
    assert port.dtype.encoding == "opaque"
    assert signature.get_left("system") == signature.get_right("system") == port
    assert (port.name, port.bitsize, port.shape) == (
        source_port.name,
        source_port.bitsize,
        source_port.shape,
    )
    assert tuple(source_port.all_idxs()) == tuple(port.all_idxs())
    terms = ("interaction", "pink", "gold")
    assert tuple(
        zip((terms[i] for i in source.indices), source.coeffs, strict=True)
    ) == (_suzuki_schedule(strip_layer))
    interaction, pink, gold = source.bloqs
    assert (
        interaction.length,
        interaction.angle,
        interaction.hubb_u,
        interaction.eps,
    ) == (length, 0, coupling, eps)
    assert [
        (node.length, node.angle, node.tau, node.eps, node.pink)
        for node in (pink, gold)
    ] == [(length, 0, hopping, eps, True), (length, 0, hopping, eps, False)]
    native = source_hubbard_suzuki_cost(
        length, 4.0, 0.1234, 0.8, hamming_weight=hamming_weight, strip_layer=strip_layer
    )
    composite = source.decompose_bloq()
    steps = tuple(composite.iter_bloqsoqs())
    assert len(native.calls) == len(steps)
    endpoints = tuple(
        Soquet(LeftDangle, source_port, idx) for idx in source_port.all_idxs()
    )
    for call, coefficient, (instance, inputs, outputs) in zip(
        native.calls, source.coeffs, steps, strict=True
    ):
        original = instance.bloq
        assert (original.length, original.eps, original.angle) == (
            length,
            eps,
            2 * coefficient * timestep,
        )
        assert tuple(inputs["system"]) == endpoints
        endpoints = tuple(outputs[0])
        assert call.repetitions == 1
        name = type(original).__name__
        if hasattr(original, "pink"):
            color = "pink" if original.pink else "gold"
            name = f"{name}[{color}]"
        assert call.component.name == name
        assert_source_hubbard_cost(call.component, original)
        with pytest.raises(DecomposeNotImplementedError):
            original.decompose_bloq()
    assert tuple(composite.final_soqs()["system"]) == endpoints
    assert_source_hubbard_cost(native, source)
    report = estimate_ftqc_resources(native)
    assert report.emitted is None
    assert report.logical.gates.t == 12 * length**2


def test_example_hubbard_phasing_budget():
    cost = source_hubbard_suzuki_cost(4, 4.0, 0.1234, hamming_weight=True)
    report = estimate_ftqc_resources(cost)
    assert report.logical.gates.total_t_count() == 912
    assert report.logical.gates.measurement == 70
    assert report.logical.peak_logical_qubits == 19
