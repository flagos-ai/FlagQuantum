"""Check ordered resource rewrites against the frozen source files."""

from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.call_graph import compose_cost_generalizer

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_source_composition_preserves_order_and_stops_at_omitted_input() -> None:
    upstream = pytest.importorskip("qualtran.resource_counting._generalization")
    from qualtran.bloqs.for_testing import TestAtom

    reference = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/resource_counting"
    )
    for name, expected in (
        ("_generalization.py", "67b8fee061e8d5a8c0a58cbaada5b671c18437d5"),
        ("_generalization_test.py", "79515dc591132a448375c436be3a73b8939e806e"),
    ):
        path = reference / name
        if not path.is_file():
            pytest.skip("Frozen source files are not installed")
        data = path.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected

    tagged, canonical = TestAtom(tag="test"), TestAtom()
    owners = {tagged: "tagged", canonical: "canonical", None: None}
    source_events: list[tuple[str, str | None]] = []
    native_events: list[tuple[str, str | None]] = []

    def source_rewrite(node: object) -> object:
        source_events.append(("rewrite", owners[node]))
        return canonical

    def source_drop(node: object) -> None:
        source_events.append(("drop", owners[node]))

    def rewrite(node: str) -> str:
        native_events.append(("rewrite", node))
        return "canonical"

    def drop(node: str) -> None:
        native_events.append(("drop", node))

    for source_rules, native_rules in (
        ((), ()),
        ((source_rewrite,), (rewrite,)),
        ((source_drop,), (drop,)),
        ((source_rewrite, source_drop), (rewrite, drop)),
        ((source_drop, source_rewrite), (drop, rewrite)),
    ):
        source_rule = upstream._make_composite_generalizer(*source_rules)
        native_rule = compose_cost_generalizer(native_rules)
        for node in (None, tagged):
            source_events.clear()
            native_events.clear()
            assert native_rule(owners[node]) == owners[source_rule(node)]
            assert native_events == source_events
