"""Check both locked lattice schedules and their resource-only boundary."""

import pytest

from flagquantum.algorithms.primitives.hubbard_trotter import source_hubbard_suzuki_cost
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
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.chemistry.trotter.hubbard.trotter_step import (
        build_plaq_hwp_unitary_second_order_suzuki,
        build_plaq_unitary_second_order_suzuki,
    )

    factory = (
        build_plaq_hwp_unitary_second_order_suzuki
        if hamming_weight
        else build_plaq_unitary_second_order_suzuki
    )
    source = factory(length, 4.0, 0.1234, 0.8, strip_layer=strip_layer)
    native = source_hubbard_suzuki_cost(
        length, 4.0, 0.1234, 0.8, hamming_weight=hamming_weight, strip_layer=strip_layer
    )
    calls = [
        instance.bloq for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
    ]
    assert len(native.calls) == len(calls)
    for call, original in zip(native.calls, calls, strict=True):
        assert call.repetitions == 1
        name = type(original).__name__
        if hasattr(original, "pink"):
            color = "pink" if original.pink else "gold"
            name = f"{name}[{color}]"
        assert call.component.name == name
        assert_source_hubbard_cost(call.component, original)
        with pytest.raises(DecomposeNotImplementedError):
            original.decompose_bloq()
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
