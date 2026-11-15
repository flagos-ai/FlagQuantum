"""Check both frozen projectile notebook examples."""

from collections import Counter

import pytest

from examples.algorithms.first_quantization_projectile_catalog import notebook_cases
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_first_quantization_projectile_resource_model import (
    assert_projectile_signature,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_projectile_notebook_graphs_match_original() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile import (
        PrepareFirstQuantizationWithProj,
        SelectFirstQuantizationWithProj,
    )
    from qualtran.resource_counting import QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    originals = (
        PrepareFirstQuantizationWithProj(6, 8, 10, 10, 10),
        SelectFirstQuantizationWithProj(6, 8, 10, 10, 10),
    )
    for (signature, model), source in zip(
        notebook_cases().values(), originals, strict=True
    ):
        assert_projectile_signature(signature, source.signature)
        actual = modeled_call_graph(model, max_depth=1)
        _, expected = source.call_graph(max_depth=1, generalizer=ignore_split_join)
        assert Counter(
            (
                child.name,
                estimate_ftqc_resources(child).logical.peak_logical_qubits,
                count,
            )
            for child, count in actual.leaf_counts.items()
        ) == Counter(
            (type(child).__name__, get_cost_value(child, QubitCount()), count)
            for child, count in expected.items()
        )
        assert_source_model(model, source)
