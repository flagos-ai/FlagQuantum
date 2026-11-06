"""Check the concrete Hubbard notebook graph and gate queries."""

from collections import Counter
from dataclasses import asdict

import pytest

from examples.algorithms.hubbard_qubitization_resources import notebook_models
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_hubbard_notebook_models_match_locked_child_costs_and_multiplicities():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        PrepareHubbard,
        SelectHubbard,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    source_models = (
        (4, SelectHubbard(4, 4)),
        (4, PrepareHubbard(4, 4, t=1.0, u=4.0)),
        (11, SelectHubbard(11, 11, control_val=1)),
    )
    for (side, model), (source_side, original) in zip(
        notebook_models(), source_models, strict=True
    ):
        assert side == source_side
        graph = modeled_call_graph(model, max_depth=1)
        _, expected = original.call_graph(max_depth=1, generalizer=ignore_split_join)
        assert Counter(
            (
                node.name.partition("(")[0],
                tuple(
                    sorted(
                        (key, value)
                        for key, value in asdict(
                            estimate_component_resources(node).gates
                        ).items()
                        if value
                    )
                ),
                count,
            )
            for node, count in graph.leaf_counts.items()
        ) == Counter(
            (
                type(node).__name__,
                tuple(sorted(get_cost_value(node, QECGatesCost()).asdict().items())),
                count,
            )
            for node, count in expected.items()
        )
        actual = estimate_component_resources(model)
        assert {key: value for key, value in asdict(actual.gates).items() if value} == (
            get_cost_value(original, QECGatesCost()).asdict()
        )
        assert actual.peak_logical_qubits == get_cost_value(original, QubitCount())
