"""Compare native cost-table presentation with unmodified source formatters."""

import csv
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, get_args
from xml.etree import ElementTree

import pytest
from sympy import Symbol

from flagquantum.compiler._component_trace.cost_query import (
    GATE_COST,
    PEAK_LOGICAL_QUBITS,
    query_component_costs,
)
from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_modeled_call_graph_dot,
    format_call_graph_cost_data,
    render_call_graph_svg,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    with (REFERENCE.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for path in ("drawing/bloq_counts_graph.py", "resource_counting/_bloq_counts.py"):
        raw = (REFERENCE / "qualtran" / path).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            == blobs["qualtran/" + path]
        )


@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
@pytest.mark.parametrize(
    "gates",
    [
        GateCost(),
        GateCost(
            t=2, toffoli=3, cswap=4, and_bloq=5, clifford=6, rotation=7, measurement=8
        ),
        QECGateCost(
            t=Symbol("t"),
            toffoli=Symbol("c"),
            cswap=Symbol("s"),
            and_bloq=Symbol("a"),
            clifford=Symbol("k"),
            rotation=Symbol("r"),
            measurement=Symbol("m"),
        ),
        QECGateCost(t=0, rotation=Symbol("r", positive=True)),
    ],
)
@pytest.mark.parametrize("width_first", [False, True])
def test_complete_rows_match_source(
    gates: GateCost | QECGateCost, aggregation: GateAggregation, width_first: bool
) -> None:
    from qualtran.drawing import GraphvizCallGraph
    from qualtran.resource_counting import GateCounts, QECGatesCost, QubitCount

    width = Symbol("n", integer=True, positive=True)
    entries = [(PEAK_LOGICAL_QUBITS, width), (GATE_COST, gates)]
    source_entries = [
        (QubitCount(), width),
        (QECGatesCost(), GateCounts(**asdict(gates))),
    ]
    if not width_first:
        entries.reverse()
        source_entries.reverse()
    data = {"root": dict(entries), "empty": {}}
    before = {node: dict(row) for node, row in data.items()}
    expected = GraphvizCallGraph.format_cost_data(
        {"root": dict(source_entries), "empty": {}}, agg_gate_counts=aggregation
    )
    actual = format_call_graph_cost_data(data, aggregation=aggregation)
    assert list(actual) == list(expected)
    assert actual == expected
    assert list(actual["root"].items()) == list(expected["root"].items())
    assert data == before


def test_unknown_metrics_warn_and_preserve_colliding_key_order() -> None:
    from qualtran.drawing import GraphvizCallGraph
    from qualtran.resource_counting import QubitCount

    class Label:
        def __str__(self) -> str:
            return "Qubits"

    metric = Label()
    with pytest.warns(UserWarning, match="Unknown cost metric") as native_warnings:
        actual = format_call_graph_cost_data(
            {
                "first": {PEAK_LOGICAL_QUBITS: 3, metric: "custom"},
                "second": {metric: "before", PEAK_LOGICAL_QUBITS: 4},
            }
        )
    with pytest.warns(UserWarning, match="Unknown cost key") as source_warnings:
        expected = GraphvizCallGraph.format_cost_data(
            {
                "first": {QubitCount(): 3, metric: "custom"},
                "second": {metric: "before", QubitCount(): 4},
            }
        )
    assert len(native_warnings) == len(source_warnings) == 2
    assert (
        actual == expected == {"first": {"Qubits": "custom"}, "second": {"Qubits": "4"}}
    )
    assert Path(native_warnings[0].filename) == Path(__file__)


@pytest.mark.parametrize("value", [None, 3, {}, "invalid"])
def test_invalid_gate_values_fail_without_mutating_input(value: Any) -> None:
    data = {"root": {GATE_COST: value}}
    with pytest.raises(TypeError, match="Gate cost annotations"):
        format_call_graph_cost_data(data)
    assert data["root"][GATE_COST] is value
    assert format_call_graph_cost_data({"root": {GATE_COST: GateCost(t=3)}}) == {
        "root": {"Ts": "3"}
    }


def test_invalid_aggregation_and_empty_rows_follow_source_dispatch() -> None:
    from qualtran.drawing import GraphvizCallGraph
    from qualtran.resource_counting import GateCounts, QECGatesCost

    invalid: Any = "unsupported"
    with pytest.raises(ValueError, match="Unknown gate aggregation"):
        format_call_graph_cost_data(
            {"root": {GATE_COST: GateCost()}}, aggregation=invalid
        )
    with pytest.raises(ValueError, match="Unknown aggregation mode"):
        GraphvizCallGraph.format_cost_data(
            {"root": {QECGatesCost(): GateCounts()}}, agg_gate_counts=invalid
        )
    assert format_call_graph_cost_data({}, aggregation=invalid) == {}
    assert format_call_graph_cost_data({"root": {}}, aggregation=invalid) == {
        "root": {}
    }


@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
@pytest.mark.parametrize("depth", [0, 1, None])
def test_existing_modeled_entry_preserves_full_costs(
    depth: int | None, aggregation: GateAggregation
) -> None:
    leaf = ComponentCost("Leaf", 5, GateCost(t=2, and_bloq=3, rotation=4))
    root = ComponentCost("Root", 1, calls=(CostCall(leaf, 7, bystander_qubits=3),))
    table = query_component_costs(root, (PEAK_LOGICAL_QUBITS, GATE_COST))
    annotations = format_call_graph_cost_data(table, aggregation=aggregation)
    dot = export_modeled_call_graph_dot(root, max_depth=depth, aggregation=aggregation)
    assert annotations[root]["Qubits"] == "8"
    assert dot.count("shape=plaintext") == (1 if depth == 0 else 2)
    for key, value in annotations[root].items():
        assert f"<TD>{key}</TD><TD>{value}</TD>" in dot
    if depth == 0:
        assert "Leaf" not in dot
    ElementTree.fromstring(render_call_graph_svg(dot))


def test_formatting_callback_boundary_accepts_escape_sensitive_values() -> None:
    from flagquantum.drawer.call_graph import export_call_graph_dot

    data = {"root": {"A&B": "<tag>"}}
    with pytest.warns(UserWarning):
        annotations = format_call_graph_cost_data(data)
    dot = export_call_graph_dot("root", {}, annotations=annotations)
    assert "A&amp;B" in dot and "&lt;tag&gt;" in dot
    ElementTree.fromstring(render_call_graph_svg(dot))
