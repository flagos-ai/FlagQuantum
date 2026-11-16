"""Compare single-factorization PREPARE and SELECT with unmodified source."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.primitives.single_factorization_resource_model import (
    source_single_factorization_inner_prepare_cost,
    source_single_factorization_inner_prepare_signature,
    source_single_factorization_outer_prepare_cost,
    source_single_factorization_outer_prepare_signature,
    source_single_factorization_select_cost,
    source_single_factorization_select_signature,
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
        leaf = estimate_component_resources(edge.component).gates
        assert {key: value for key, value in asdict(leaf).items() if value} == (
            get_cost_value(child, QECGatesCost()).asdict()
        )


def assert_source_signature(native, source):
    from qualtran import QAny, QBit

    assert [
        (
            reg.name,
            reg.side.value,
            reg.shape,
            QBit() if reg.dtype.encoding == "bit" else QAny(reg.bitsize),
        )
        for reg in native
    ] == [(reg.name, reg.side.value, reg.shape, reg.dtype) for reg in source]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "params", [(50, 10, 12, 7, 4, 8), (13, 8, 4, 8, 1, 2), (21, 6, 3, 4, 3, 2)]
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_inner_source_model(params, adjoint):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sf.prepare import InnerPrepareSingleFactorization

    source = InnerPrepareSingleFactorization(*params)
    source = source.adjoint() if adjoint else source
    signature = source_single_factorization_inner_prepare_signature(*params[:2])
    assert_source_signature(
        signature.adjoint() if adjoint else signature, source.signature
    )
    _assert_model(
        source_single_factorization_inner_prepare_cost(*params, adjoint=adjoint), source
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("auxiliaries,alias", [(50, 12), (31, 6), (24, 4)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_outer_source_model(auxiliaries, alias, adjoint):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sf.prepare import OuterPrepareSingleFactorization

    source = OuterPrepareSingleFactorization(auxiliaries, alias)
    source = source.adjoint() if adjoint else source
    signature = source_single_factorization_outer_prepare_signature(auxiliaries)
    assert_source_signature(
        signature.adjoint() if adjoint else signature, source.signature
    )
    _assert_model(
        source_single_factorization_outer_prepare_cost(
            auxiliaries, alias, adjoint=adjoint
        ),
        source,
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("spin_orbitals", [2, 6, 10, 16])
@pytest.mark.parametrize("control", [False, True])
def test_select_source_model(spin_orbitals, control):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sf.select_bloq import SelectSingleFactorization

    source = SelectSingleFactorization(spin_orbitals, additional_control=control)
    signature = source_single_factorization_select_signature(
        spin_orbitals, additional_control=control
    )
    assert_source_signature(signature, source.signature)
    _assert_model(
        source_single_factorization_select_cost(
            spin_orbitals, additional_control=control
        ),
        source,
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


@pytest.mark.ftqc_reference
def test_prepare_signature_omits_source_zero_width_ports():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sf.prepare import (
        InnerPrepareSingleFactorization,
        OuterPrepareSingleFactorization,
    )

    assert_source_signature(
        source_single_factorization_inner_prepare_signature(0, 2),
        InnerPrepareSingleFactorization(0, 2, 4).signature,
    )
    assert_source_signature(
        source_single_factorization_outer_prepare_signature(0),
        OuterPrepareSingleFactorization(0, 4).signature,
    )
