"""Reconcile the native chemistry resource tutorial against locked models."""

import pytest

from examples.algorithms.chemistry_resource_tutorial import (
    ALIAS_BITS,
    ANGLE_BITS,
    ELECTRON_COUNTS,
    SPIN_ORBITALS,
    THC_RANK,
    approximate_alpha_first_quantization,
    df_model,
    first_quantization_models,
    first_quantization_select_breakdown,
    first_quantization_select_score,
    modeled_ccz,
    thc_fixture,
    thc_models,
    thc_prepare_t_breakdown,
    thc_select_score,
    volume_from_rs_eta,
)
from examples.algorithms.double_factorization_resources import (
    double_factorization_block_score,
    double_factorization_block_signature,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def ccz(source):
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.resource_counting.generalizers import generalize_cswap_approx

    counts = get_cost_value(source, QECGatesCost(), generalizer=generalize_cswap_approx)
    return counts.total_t_and_ccz_count(ts_per_rotation=0)["n_ccz"]


def test_thc_reference_fixture_and_both_counting_conventions():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.block_encoding import LCUBlockEncoding
    from qualtran.bloqs.chemistry.thc import PrepareTHC, SelectTHC
    from qualtran.bloqs.chemistry.thc.select_bloq import THCRotations
    from qualtran.resource_counting.classify_bloqs import classify_t_count_by_bloq_type

    values, eta, zeta = thc_fixture()
    native = thc_models(values, eta, zeta)
    prepare = PrepareTHC.from_hamiltonian_coeffs(
        values, eta, zeta, num_bits_state_prep=ALIAS_BITS
    )
    select = SelectTHC(
        THC_RANK,
        SPIN_ORBITALS,
        num_bits_theta=ANGLE_BITS,
        keep_bitsize=prepare.keep_bitsize,
        kr1=16,
        kr2=16,
    )
    rotations = THCRotations(THC_RANK, SPIN_ORBITALS, ANGLE_BITS, kr1=16, kr2=16)
    assert native.normalization == pytest.approx(prepare.l1_norm_of_coeffs)
    assert thc_prepare_t_breakdown(native.prepare) == classify_t_count_by_bloq_type(
        prepare
    )
    models = native.select, native.prepare, native.unprepare, native.rotations
    source = select, prepare, prepare.adjoint(), rotations
    costs = tuple(modeled_ccz(m) for m in models)
    assert costs == tuple(ccz(s) for s in source) == (7105, 3195, 839, 1914)
    assert (
        sum(costs[:3]) == ccz(LCUBlockEncoding(select=select, prepare=prepare)) == 11139
    )
    assert estimate_component_resources(native.prepare).gates.t == 7560
    assert len(thc_select_score().columns) == 15


def test_double_factorization_reference():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.double_factorization import (
        DoubleFactorizationBlockEncoding,
    )

    source = DoubleFactorizationBlockEncoding(
        SPIN_ORBITALS, 360, 13031, ALIAS_BITS, num_bits_rot=ANGLE_BITS
    )
    model = df_model()
    assert modeled_ccz(model) == ccz(source) == 21728
    signature = double_factorization_block_signature(
        360, SPIN_ORBITALS, 13031, ANGLE_BITS, 8
    )
    assert [(r.name, r.shape, r.bitsize) for r in signature] == [
        (r.name, r.shape, r.bitsize) for r in source.signature
    ]
    assert len(double_factorization_block_score(model, signature).columns) == 7


@pytest.mark.parametrize("eta", ELECTRON_COUNTS)
def test_first_quantization_reference_and_component_breakdown(eta):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        PrepareFirstQuantization,
        SelectFirstQuantization,
    )
    from qualtran.resource_counting.generalizers import (
        ignore_alloc_free,
        ignore_split_join,
    )

    alpha = approximate_alpha_first_quantization(volume_from_rs_eta(3.0, eta), eta, 6)
    prepare = PrepareFirstQuantization(6, eta, eta, eta, sum_of_l1_coeffs=alpha)
    select = SelectFirstQuantization(6, eta, eta, eta)
    models = first_quantization_models(eta)
    assert tuple(modeled_ccz(m) for m in models) == tuple(
        ccz(s) for s in (select, prepare, prepare.adjoint())
    )
    source_calls = select.bloq_counts(
        generalizer=[ignore_split_join, ignore_alloc_free]
    )
    assert first_quantization_select_breakdown(models[0]) == {
        str(child): repetitions * ccz(child)
        for child, repetitions in source_calls.items()
    }
    assert len(first_quantization_select_score().columns) == 12
