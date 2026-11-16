"""Compare sparse chemistry tutorial diagrams and budgets with the frozen source."""

from dataclasses import asdict

import numpy as np
import pytest

from examples.algorithms.chemistry_resource_tutorial import modeled_ccz
from examples.algorithms.sparse_chemistry_resources import (
    sparse_integral_fixture,
    sparse_prepare_graph,
    sparse_select_graph,
)
from flagquantum.algorithms.primitives.multiplexers.select_sparse import SelectSparse
from flagquantum.algorithms.primitives.sparse_chemistry_walk import SparseChemistryWalk
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_tutorial_source_fixture_and_direct_children():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sparse import (
        PrepareSparse,
    )
    from qualtran.bloqs.chemistry.sparse import (
        SelectSparse as SourceSelect,
    )
    from qualtran.bloqs.chemistry.sparse.prepare_test import build_random_test_integrals
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    fixture = sparse_integral_fixture(3)
    for actual, expected in zip(fixture, build_random_test_integrals(3), strict=True):
        np.testing.assert_array_equal(actual, expected)
    walk = SparseChemistryWalk.from_integrals(*fixture, probability_bits=4)
    native_prep = sparse_prepare_graph(walk.prepare)
    reference_prep = PrepareSparse.from_hamiltonian_coeffs(
        6, *fixture, num_bits_state_prep=4
    )
    native_sel = sparse_select_graph(4)
    reference_sel = SourceSelect(4)
    for model, source in ((native_prep, reference_prep), (native_sel, reference_sel)):
        assert_source_model(model, source)
        # PREPARE's direct call order is declared. SELECT's can differ from its circuit order.
        native_children = sorted(
            (
                edge.repetitions,
                estimate_component_resources(edge.component).peak_logical_qubits,
                sorted(
                    (k, v)
                    for k, v in asdict(
                        estimate_component_resources(edge.component).gates
                    ).items()
                    if v
                ),
            )
            for edge in model.calls
        )
        reference_children = sorted(
            (
                repetitions,
                get_cost_value(child, QubitCount()),
                sorted(get_cost_value(child, QECGatesCost()).asdict().items()),
            )
            for child, repetitions in source.bloq_counts().items()
        )
        assert native_children == reference_children


@pytest.mark.parametrize(("spin", "rotation"), [(8, 3), (12, 4), (16, 3)])
def test_openfermion_fixture_generalized_budgets_match_source(spin, rotation):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sparse import (
        PrepareSparse,
    )
    from qualtran.bloqs.chemistry.sparse import (
        SelectSparse as SourceSelect,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.resource_counting.generalizers import generalize_cswap_approx

    fixture = sparse_integral_fixture(spin // 2)
    walk = SparseChemistryWalk.from_integrals(
        *fixture, probability_bits=12, rotation_bits=rotation, log_block_size=5
    )
    source_prep = PrepareSparse.from_hamiltonian_coeffs(
        spin,
        *fixture,
        num_bits_state_prep=12,
        num_bits_rot_aa=rotation,
        log_block_size=5,
    )
    models = (
        SelectSparse.source_resource_model(spin),
        walk.prepare.resource_model(),
        walk.prepare.resource_model(adjoint=True),
    )
    source = (SourceSelect(spin), source_prep, source_prep.adjoint())
    for model, operation in zip(models, source, strict=True):
        gates = get_cost_value(
            operation, QECGatesCost(), generalizer=generalize_cswap_approx
        )
        assert (
            modeled_ccz(model)
            == gates.total_t_and_ccz_count(ts_per_rotation=0)["n_ccz"]
        )
