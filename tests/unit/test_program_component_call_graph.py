"""Source adjoint counts and stored Program calls share existing graph accounting."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_expansion import _hierarchy
from tests.unit.test_component_hierarchy_adjoint import (
    paired_components,
    source_adjoint,
    wrapped_add,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_counts(
    record: ProgramRecord, source: Any, terminals: set[str], depth: int | None
) -> None:
    expected = source.decompose_bloq()
    pairs = paired_components(record, expected, terminals)
    pairs["main"] = source
    names = {bloq: name for name, bloq in pairs.items()}
    stop = {pairs[name] for name in terminals if name in pairs}
    graph, totals = source.call_graph(keep=lambda bloq: bloq in stop, max_depth=depth)
    actual = program_component_call_graph(
        record, keep=terminals.__contains__, max_depth=depth
    )
    assert actual.direct_calls == {
        (names[parent], names[child]): data["n"]
        for parent, child, data in graph.edges.data()
    }
    assert actual.leaf_counts == {names[bloq]: count for bloq, count in totals.items()}
    if depth != 0:
        assert {
            child: count
            for (parent, child), count in actual.direct_calls.items()
            if parent == "main"
        } == {names[bloq]: count for bloq, count in source.bloq_counts().items()}


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("repeats", [1, 2])
@pytest.mark.parametrize("max_depth", [None, 0, 1, 2])
def test_hierarchical_adjoint_direct_and_weighted_terminal_counts_match_source(
    depth: int, repeats: int, max_depth: int | None
) -> None:
    record, source = _hierarchy(depth, repeats)
    inverse = adjoint_component_hierarchy(record, {"h": "h"})
    source_counts(inverse, source_adjoint(source), {"h"}, max_depth)
    if max_depth is None:
        calls = program_component_call_graph(inverse)
        assert calls.leaf_counts == {"h": repeats**depth}
        assert calls.estimate_leaf_cost(lambda name: int(name == "h")) == repeats**depth


@pytest.mark.parametrize("depth", [2, 3])
@pytest.mark.parametrize("max_depth", [None, 0, 1, 2])
def test_specialized_add_resource_calls_close_against_independent_model(
    depth: int, max_depth: int | None
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    record, source = wrapped_add(depth)
    inverse = adjoint_component_hierarchy(record, {"add": "add_adjoint"})
    source_counts(inverse, source_adjoint(source), {"add_adjoint"}, max_depth)
    if max_depth is None:
        graph = program_component_call_graph(
            inverse, keep=lambda name: name == "add_adjoint"
        )
        assert graph.leaf_counts == {"add_adjoint": 2}
        assert (
            graph.estimate_leaf_cost(lambda name: 8 if name == "add_adjoint" else 0)
            == get_cost_value(source_adjoint(source), QECGatesCost()).total_t_count()
            == 16
        )
        assert (
            graph.estimate_leaf_cost(lambda name: 2 if name == "add_adjoint" else 0)
            == 4
        )


def test_many_to_one_specialized_adjoints_merge_call_multiplicities() -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import Hadamard

    record, _ = _hierarchy(1)
    h = record.functions[-1]
    one, two = replace(h, name="first"), replace(h, name="second")
    builder = ComponentProgramBuilder(Signature(h.registers or ()), (h, one, two))
    outputs = builder.call_dict("first", builder.inputs)
    outputs = builder.call_dict("second", outputs)
    record = builder.finalize(outputs)
    inverse = adjoint_component_hierarchy(record, {"first": "h", "second": "h"})
    bb, outputs = BloqBuilder.from_signature(Hadamard().signature)
    for _ in range(2):
        outputs = bb.add_d(Hadamard(), **outputs)
    source_counts(inverse, source_adjoint(bb.finalize(**outputs)), {"h"}, None)
    graph = program_component_call_graph(inverse)
    assert graph.direct_calls == {("main", "h"): 2}
    assert graph.leaf_counts == {"h": 2}


def test_keep_policy_can_stop_at_root_and_callback_errors_propagate() -> None:
    record, _ = _hierarchy(3)
    graph = program_component_call_graph(record, keep=lambda name: True)
    assert graph.direct_calls == {} and graph.leaf_counts == {"main": 1}

    def fail(name: str) -> bool:
        raise RuntimeError(name)

    with pytest.raises(RuntimeError, match="main"):
        program_component_call_graph(record, keep=fail)


@pytest.mark.parametrize("depth", [-1, True])
def test_invalid_graph_depth_rejected_by_existing_policy(depth: int) -> None:
    record, _ = _hierarchy(2)
    with pytest.raises(ValueError, match="nonnegative integer"):
        program_component_call_graph(record, max_depth=depth)


def test_invalid_unused_body_rejected_before_resource_callback() -> None:
    record, _ = _hierarchy(1)
    bad = replace(record.functions[-1], name="unused", blocks=())
    record = replace(record, functions=(*record.functions, bad))
    seen: list[str] = []

    def observe(name: str) -> bool:
        seen.append(name)
        return False

    with pytest.raises(ValueError, match="invalid component graph"):
        program_component_call_graph(record, keep=observe)
    assert seen == []
