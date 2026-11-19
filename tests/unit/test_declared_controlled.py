"""Compare generic controlled declarations with frozen source call graphs."""

from collections.abc import Hashable
from typing import Any

import pytest

from flagquantum.compiler._component_trace.call_graph import (
    RawCallees,
    SymbolAllocator,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.controlled import ControlledCall
from flagquantum.compiler._component_trace.resource_generalization import ControlPattern
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.errors import CapabilityError

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")


@pytest.mark.parametrize("as_set", [False, True])
@pytest.mark.parametrize("specialized", [False, True])
@pytest.mark.parametrize("controls", [(1,), (0,), (0, 1)])
def test_original_raw_counts_identity_specialization_and_allocator(
    as_set: bool, specialized: bool, controls: tuple[int, ...]
) -> None:
    from qualtran import Bloq, Controlled, CtrlSpec
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.for_testing import TestAtom
    from qualtran.resource_counting import SympySymbolAllocator

    symbols = pytest.importorskip("sympy")
    n = symbols.Symbol("n")
    spec = CtrlSpec(cvs=controls[0] if len(controls) == 1 else list(controls))
    children = (TestAtom("a"), TestAtom("b"))
    if specialized:

        def specialized_control(self: object, ctrl_spec: Any) -> Any:
            return XGate().controlled(ctrl_spec)

        children = tuple(
            type(
                name,
                (Bloq,),
                {"signature": SourceSignature(()), "controlled": specialized_control},
            )()
            for name in ("a", "b")
        )
    entries = ((children[0], n), (children[1], 2))
    raw = set(entries) if as_set else dict(entries)

    def source_calls(self: object, ssa: object) -> Any:
        return raw

    source = type(
        "Parent",
        (Bloq,),
        {"signature": SourceSignature(()), "build_call_graph": source_calls},
    )()
    original = Controlled(source, spec)
    expected = original.build_call_graph(SympySymbolAllocator())
    node = ControlledCall("Parent", controls)
    assert str(node) == str(original)
    assert node == ControlledCall("Parent", controls) and hash(node) == hash(
        ControlledCall("Parent", controls)
    )
    seen = []

    def declare(child: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        assert child == "Parent"
        seen.append(allocator)
        items: list[tuple[Hashable, SymbolicInt]] = [
            (str(child), count) for child, count in entries
        ]
        return set(items) if as_set else dict(items)

    def specialize(child: Hashable, pattern: ControlPattern) -> Hashable:
        assert pattern == controls
        return (
            str(XGate().controlled(spec))
            if specialized
            else ControlledCall(child, pattern)
        )

    allocator = SymbolAllocator()
    actual = node.callee_counts(declare, control=specialize, allocator=allocator)
    assert seen == [allocator]
    assert {str(k): v for k, v in actual.items()} == {
        str(k): v for k, v in expected.items()
    }
    if not specialized:
        assert actual == node.callee_counts(declare)


def test_full_declared_graph_symbolic_counts_and_zero_edges() -> None:
    symbols = pytest.importorskip("sympy")
    n = symbols.Symbol("n")
    controls = (0, 1)
    forward: dict[str, dict[Hashable, SymbolicInt]] = {
        "root": {"shared": n, "leaf": 0},
        "shared": {"leaf": 2},
        "leaf": {},
    }

    def declare(child: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        assert isinstance(child, str)
        return forward[child]

    def calls(child: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        assert isinstance(child, ControlledCall)
        return child.callee_counts(declare, allocator=allocator)

    root: Hashable = ControlledCall("root", controls)
    actual = symbolic_call_graph(root, calls)
    leaf = ControlledCall("leaf", controls)
    assert actual.leaf_counts == {leaf: 2 * n}
    assert actual.direct_calls[(root, leaf)] == 0
    assert str(ControlledCall("root", n)) == f"C[{n}][root]"


def test_missing_declaration_and_callback_failure_propagate() -> None:
    node: Hashable = ControlledCall("atomic")
    assert isinstance(node, ControlledCall)

    def missing(child: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        raise CapabilityError("unavailable-child")

    with pytest.raises(CapabilityError, match="unavailable-child"):
        node.callee_counts(missing)
    root: Hashable = node
    graph = symbolic_call_graph(root, missing, keep=lambda _: True)
    assert graph.leaf_counts == {node: 1}

    def declaration(
        child: Hashable, allocator: SymbolAllocator
    ) -> RawCallees[Hashable]:
        result: dict[Hashable, SymbolicInt] = {"leaf": 1}
        return result

    def fail(child: Hashable, controls: ControlPattern) -> Hashable:
        raise ValueError("control-specialization")

    with pytest.raises(ValueError, match="control-specialization"):
        node.callee_counts(declaration, control=fail)


def test_all_recursive_edges_match_original_declared_control_graph() -> None:
    from qualtran import Bloq, Controlled, CtrlSpec
    from qualtran import Signature as SourceSignature

    symbols = pytest.importorskip("sympy")
    n = symbols.Symbol("n", positive=True)
    declarations = {"root": {"shared": n, "leaf": 1}, "shared": {"leaf": 2}, "leaf": {}}
    nodes: dict[str, Any] = {}
    spec = CtrlSpec()

    def source_control(self: Any, ctrl_spec: Any) -> Any:
        return Controlled(self, ctrl_spec)

    for name in reversed(tuple(declarations)):

        def source_declaration(self: Any, ssa: Any) -> Any:
            return {
                nodes[child]: count
                for child, count in declarations[type(self).__name__].items()
            }

        nodes[name] = type(
            name,
            (Bloq,),
            {
                "signature": SourceSignature(()),
                "build_call_graph": source_declaration,
                "controlled": source_control,
            },
        )()
    source_graph, source_totals = Controlled(nodes["root"], spec).call_graph()

    def forward(child: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        assert isinstance(child, str)
        return [(name, count) for name, count in declarations[child].items()]

    def controlled(child: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        assert isinstance(child, ControlledCall)
        return child.callee_counts(forward, allocator=allocator)

    root: Hashable = ControlledCall("root")
    graph = symbolic_call_graph(root, controlled)
    assert {
        (str(a), str(b)): count for (a, b), count in graph.direct_calls.items()
    } == {(str(a), str(b)): data["n"] for a, b, data in source_graph.edges.data()}
    assert {str(node): count for node, count in graph.leaf_counts.items()} == {
        str(node): count for node, count in source_totals.items()
    }
