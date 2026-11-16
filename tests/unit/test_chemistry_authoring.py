"""Check the authoring tutorial's estimates, ordered data and physical example."""

import numpy as np
import pytest

from examples.algorithms.chemistry_authoring import (
    DensePreparationSketch,
    authoring_integrals,
    dense_alias_rows,
    tutorial_ccz_equivalents,
)
from flagquantum.algorithms.primitives.sparse_chemistry_data import (
    sparse_chemistry_data,
)
from flagquantum.algorithms.primitives.sparse_chemistry_prepare import (
    PrepareSparseChemistry,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


def test_authoring_cost_boundaries_and_wiring():
    one, two = authoring_integrals(5)
    initial = DensePreparationSketch(10, 16)
    assert tutorial_ccz_equivalents(initial.cost()) == 328
    for precision, expected, peak in ((10, 562, 152), (4, 502, 116)):
        sketch = DensePreparationSketch(10, precision)
        rows = dense_alias_rows(one, two, precision)
        assert len(rows) == 5 and all(len(row) == 650 for row in rows)
        model = sketch.detailed_cost(rows)
        assert tutorial_ccz_equivalents(model) == expected
        assert estimate_component_resources(model).peak_logical_qubits == peak
        assert [call.component.name for call in model.calls] == [
            "PrepareUniformSuperposition(5)",
            "ContiguousIndexFourD",
            "SelectSwapQROM",
            "Hadamard",
            "LessThanEqual",
            "CSwap",
            "LessThanEqual",
        ]
        assert model.calls[1].component.gates.toffoli == 26
        score = sketch.register_score()
        assert "alt_pqrs[3]" in score.output_order
        assert sum(column.step.name == "CSwap" for column in score.columns) == 4
    sparse = PrepareSparseChemistry(sparse_chemistry_data(one, two, probability_bits=4))
    assert len(sparse.data.indices) == 135
    assert tutorial_ccz_equivalents(sparse.resource_model()) == 153


@pytest.mark.ftqc_reference
def test_frozen_alias_fixture_and_lookup_cost(monkeypatch):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.chem_tutorials import gen_random_chem_ham
    from qualtran.bloqs.data_loading.select_swap_qrom import (
        SelectSwapQROM,
        find_optimal_log_block_size,
    )
    from qualtran.linalg.lcu_util import (
        preprocess_probabilities_for_reversible_sampling,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.resource_counting.generalizers import generalize_cswap_approx

    source_rng = np.random.RandomState(7)
    monkeypatch.setattr(np.random, "random", source_rng.random)
    monkeypatch.setattr(np.random, "normal", source_rng.normal)
    source_one, source_two = gen_random_chem_ham(10)
    one, two = authoring_integrals(5)
    np.testing.assert_array_equal(one, source_one)
    np.testing.assert_array_equal(two, source_two)
    for precision in (4, 10):
        rows = dense_alias_rows(one, two, precision)
        sketch = DensePreparationSketch(10, precision)
        source_alt, source_keep, _ = preprocess_probabilities_for_reversible_sampling(
            np.abs(np.r_[source_one.ravel(), source_two.ravel()]),
            epsilon=2**-precision / sketch.rows,
        )
        assert rows[-1] == tuple(source_keep)
        for idx, alternate in enumerate(source_alt):
            expected = (
                (*np.unravel_index(alternate, (5, 5)), 0, 0)
                if alternate < 25
                else np.unravel_index(alternate - 25, (5, 5, 5, 5))
            )
            assert tuple(column[idx] for column in rows[:-1]) == expected
        widths = (3,) * 4 + (precision,)
        log = find_optimal_log_block_size(sketch.rows, sum(widths))
        source = SelectSwapQROM.build_from_data(
            *(np.asarray(row) for row in rows),
            target_bitsizes=widths,
            log_block_sizes=[log],
            use_dirty_ancilla=False,
        )
        cost = get_cost_value(
            source, QECGatesCost(), generalizer=generalize_cswap_approx
        ).total_t_and_ccz_count(ts_per_rotation=0)
        lookup = sketch.detailed_cost(rows).calls[2].component
        assert cost["n_t"] == 0
        assert tutorial_ccz_equivalents(lookup) == cost["n_ccz"]
