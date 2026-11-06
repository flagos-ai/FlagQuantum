"""Compare single-factorization PREPARE and SELECT with unmodified source."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.primitives.single_factorization_resource_model import (
    source_single_factorization_inner_prepare_cost,
    source_single_factorization_outer_prepare_cost,
    source_single_factorization_select_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


def _assert_model(model, source):
    from qualtran import DecomposeNotImplementedError
    from qualtran.resource_counting import (
        QECGatesCost,
        QubitCount,
        SympySymbolAllocator,
        get_cost_value,
    )

    assert model.qubits == source.signature.n_qubits()
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    gates = get_cost_value(source, QECGatesCost())
    modeled = estimate_component_resources(model)
    assert {
        key: value for key, value in asdict(modeled.gates).items() if value
    } == gates.asdict()
    assert modeled.peak_logical_qubits == get_cost_value(source, QubitCount())
    expected = source.build_call_graph(SympySymbolAllocator())
    assert [edge.repetitions for edge in model.calls] == list(expected.values())
    for edge, child in zip(model.calls, expected, strict=True):
        assert edge.component.qubits == get_cost_value(child, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "params", [(50, 10, 12, 7, 4, 8), (13, 8, 4, 8, 1, 2), (21, 6, 3, 4, 3, 2)]
)
def test_inner_source_model(params):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sf.prepare import InnerPrepareSingleFactorization

    _assert_model(
        source_single_factorization_inner_prepare_cost(*params),
        InnerPrepareSingleFactorization(*params),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("auxiliaries,alias", [(50, 12), (31, 6), (24, 4)])
def test_outer_source_model(auxiliaries, alias):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sf.prepare import OuterPrepareSingleFactorization

    _assert_model(
        source_single_factorization_outer_prepare_cost(auxiliaries, alias),
        OuterPrepareSingleFactorization(auxiliaries, alias),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("spin_orbitals", [2, 6, 10, 16])
@pytest.mark.parametrize("control", [False, True])
def test_select_source_model(spin_orbitals, control):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sf.select_bloq import SelectSingleFactorization

    _assert_model(
        source_single_factorization_select_cost(
            spin_orbitals, additional_control=control
        ),
        SelectSingleFactorization(spin_orbitals, additional_control=control),
    )


@pytest.mark.ftqc_reference
def test_outer_source_small_table_rejects_automatic_block():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sf.prepare import OuterPrepareSingleFactorization
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    with pytest.raises(AssertionError):
        get_cost_value(OuterPrepareSingleFactorization(1, 8), QECGatesCost())
    with pytest.raises(ValueError, match="rows >= target width"):
        source_single_factorization_outer_prepare_cost(1, 8)
