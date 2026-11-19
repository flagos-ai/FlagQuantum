"""Default inverse resource nodes retain source identities and declarations."""

from collections.abc import Hashable
from typing import Any

import pytest

from flagquantum.compiler._component_trace.adjoint import AdjointCall, adjoint_call
from flagquantum.compiler._component_trace.call_graph import (
    RawCallees,
    SymbolAllocator,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.errors import CapabilityError

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")
pytest.importorskip("numpy")


def test_names_identity_hashing_and_double_adjoint() -> None:
    from qualtran import Adjoint
    from qualtran.bloqs.for_testing.atom import TestAtom

    for tag in (None, "", "a", "a†"):
        source = TestAtom(tag)
        name = str(source)
        node = AdjointCall(name)
        assert str(node) == str(Adjoint(source))
        assert node.adjoint() is name
        assert adjoint_call(node) is name
        assert adjoint_call(name) == node
        assert len({node, AdjointCall(name)}) == 1
        assert AdjointCall(node).adjoint() is node


@pytest.mark.parametrize("as_set", [False, True])
@pytest.mark.parametrize("specialized", [False, True])
def test_raw_symbolic_counts_match_unmodified_source(
    as_set: bool, specialized: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import Adjoint, Bloq
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import Hadamard
    from qualtran.bloqs.for_testing.atom import TestAtom
    from qualtran.resource_counting import SympySymbolAllocator

    symbols = pytest.importorskip("sympy")
    n = symbols.Symbol("n")
    signature = SourceSignature(())

    def source_inverse(self: object) -> Any:
        return Hadamard()

    if specialized:
        children = tuple(
            type(name, (Bloq,), {"signature": signature, "adjoint": source_inverse})()
            for name in ("First", "Second")
        )
    else:
        children = TestAtom("First"), TestAtom("Second")
    entries = ((children[0], n), (children[1], 2))
    raw = set(entries) if as_set else dict(entries)

    def source_calls(self: object, ssa: object) -> Any:
        return raw

    source = type(
        "Parent", (Bloq,), {"signature": signature, "build_call_graph": source_calls}
    )()
    expected = Adjoint(source).build_call_graph(SympySymbolAllocator())
    original_nodes = {str(child): child for child in children}
    seen: list[SymbolAllocator] = []

    def declare(node: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        assert node == "Parent"
        seen.append(allocator)
        values: list[tuple[Hashable, SymbolicInt]] = [
            (str(child), count) for child, count in entries
        ]
        return set(values) if as_set else dict(values)

    def inverse(node: Hashable) -> Hashable:
        assert isinstance(node, str)
        child = original_nodes[node]
        return "H" if specialized else adjoint_call(str(child))

    allocator = SymbolAllocator()
    actual = AdjointCall("Parent").callee_counts(
        declare, inverse=inverse, allocator=allocator
    )
    assert seen == [allocator]
    assert {str(key): count for key, count in actual.items()} == {
        str(key): count for key, count in expected.items()
    }


def test_complete_default_adjoint_graph_matches_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran import Adjoint
    from qualtran.bloqs.for_testing.with_call_graph import TestBloqWithCallGraph

    source = Adjoint(TestBloqWithCallGraph())
    expected_graph, expected_totals = source.call_graph()
    # Resolve forward source declarations only in this differential boundary.
    forward: dict[str, Any] = {}
    pending = [source.subbloq]
    while pending:
        child = pending.pop()
        if str(child) in forward:
            continue
        forward[str(child)] = child
        pending.extend(
            callee.subbloq if isinstance(callee, Adjoint) else callee
            for callee in Adjoint(child).bloq_counts()
        )

    def declare(node: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        assert isinstance(node, str)
        raw = forward[node].bloq_counts()
        values: dict[Hashable, SymbolicInt] = {
            str(child): count for child, count in raw.items()
        }
        return values

    def calls(node: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        if isinstance(node, AdjointCall):
            return node.callee_counts(declare, inverse=inverse, allocator=allocator)
        return declare(node, allocator)

    def inverse(node: Hashable) -> Hashable:
        # The native bookkeeping declarations have specialized inverses.
        if node == "Split":
            return "Join"
        if node == "Join":
            return "Split"
        return adjoint_call(node)

    root: Hashable = AdjointCall(str(source.subbloq))
    actual = symbolic_call_graph(root, calls)
    assert {(str(a), str(b)): n for (a, b), n in actual.direct_calls.items()} == {
        (str(a), str(b)): data["n"] for a, b, data in expected_graph.edges.data()
    }
    assert {str(a): n for a, n in actual.leaf_counts.items()} == {
        str(a): n for a, n in expected_totals.items()
    }


def test_missing_declaration_propagates_and_graph_can_keep_it_terminal() -> None:
    def missing(node: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        raise CapabilityError("missing-decomposition")

    node = AdjointCall("atomic")
    with pytest.raises(CapabilityError, match="missing-decomposition"):
        node.callee_counts(missing)
    root: Hashable = node
    graph = symbolic_call_graph(root, lambda n, s: node.callee_counts(missing))
    assert graph.leaf_counts == {node: 1}
    assert graph.direct_calls == {}


def test_native_declaration_allocator_and_unwrap_in_graph() -> None:
    symbols = SymbolAllocator()
    original = "leaf"
    inverse_leaf = AdjointCall(original)

    def declare(node: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        assert node == "root"
        return ((inverse_leaf, allocator.new_symbol("n")),)

    result = AdjointCall("root").callee_counts(declare, allocator=symbols)
    assert list(result) == [original]
    assert str(result[original]) == "_n0"
    assert str(symbols.new_symbol("n")) == "_n1"


def test_default_numerics_use_stored_decomposition_and_native_leaf_actions() -> None:
    import numpy as np
    import torch
    from qualtran import Adjoint, Bloq, BloqBuilder
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import XGate

    from examples.algorithms.adjoint_programs import serial_atoms, tutorial_adjoint
    from flagquantum.compiler.registers.component_graph import iter_component_calls
    from flagquantum.core.ir.program import FunctionRecord
    from flagquantum.simulation.component_classical_state import ComponentClassicalState
    from tests.unit.test_register_adjoint_tutorial import _dense

    signature = SourceSignature.build(reg=1)
    builder, inputs = BloqBuilder.from_signature(signature)
    q = inputs["reg"]
    for _ in range(3):
        q = builder.add(XGate(), q=q)
    composite = builder.finalize(reg=q)

    def decompose(self: object) -> Any:
        return composite

    def forbidden(self: object, *args: object, **kwargs: object) -> Any:
        raise AssertionError("Default adjoint must use the stored decomposition")

    source = type(
        "StoredX",
        (Bloq,),
        {
            "signature": signature,
            "decompose_bloq": decompose,
            "on_classical_vals": forbidden,
            "my_tensors": forbidden,
        },
    )()
    inverse = Adjoint(source)
    native = tutorial_adjoint(serial_atoms())
    order = tuple(i for i, _, _ in iter_component_calls(native))

    def values(function: FunctionRecord, inputs: dict[str, object]) -> object:
        assert function.name.startswith("atom")
        bit = inputs["q"]
        assert isinstance(bit, (int, np.integer))
        return {"q": 1 - bit}

    for bit in (0, 1):
        state = ComponentClassicalState(native, order, {"reg": bit}, values)
        assert tuple(state.simulate().values()) == inverse.call_classically(reg=bit)
    torch.testing.assert_close(
        _dense(native),
        torch.as_tensor(np.asarray(inverse.tensor_contract())),
        atol=0,
        rtol=0,
    )


def test_equal_display_names_do_not_merge_distinct_native_nodes() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Token:
        index: int

        def __str__(self) -> str:
            return "same-label"

    first, second = Token(0), Token(1)

    def declare(node: Hashable, allocator: SymbolAllocator) -> RawCallees[Hashable]:
        return ((first, 2), (second, 3))

    result = AdjointCall("root").callee_counts(declare)
    assert result == {AdjointCall(first): 2, AdjointCall(second): 3}
    assert str(AdjointCall(first)) == str(AdjointCall(second))
    assert adjoint_call(AdjointCall(first)) is first
