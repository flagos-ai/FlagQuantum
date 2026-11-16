"""Compare complete call-graph protocol behaviors with frozen source witnesses."""

import json
from collections.abc import Iterable
from hashlib import sha1
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.call_graph_protocol_catalog import (
    BINARY_AND,
    conjunction_counts_agree,
    conjunction_graph,
    declared_function_graph,
    resource_declaration,
    swap_decomposition_program,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    RawCallees,
    SymbolAllocator,
    component_callee_counts,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ConjunctionCall,
    PhaseCall,
    ResourceCall,
    generalize_rotation,
)
from flagquantum.compiler._component_trace.symbolic_math import asymptotic_order
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.value_math import SymbolicInt
from flagquantum.drawer.call_graph import format_call_graph_debug_text
from flagquantum.errors import CapabilityError

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "call_graph.ipynb": "8c6d5e894f2b0ba1729a74bde77dfbb73aa61e7f",
    "_call_graph.py": "8c1a6debd8c8a2e06d3a082dc6b6cd6656e4f522",
    "_call_graph_test.py": "af540e50c004fa3d7a2eaa22a221060dea42e737",
}


def _locked_asset(name: str) -> bytes:
    path = REFERENCE / "resource_counting" / name
    if not path.is_file():
        pytest.skip(f"Frozen call-graph asset {name} is not installed")
    data = path.read_bytes()
    assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == BLOBS[name]
    return data


def test_locked_files() -> None:
    pytest.importorskip("qualtran")
    for name in BLOBS:
        _locked_asset(name)


@pytest.mark.parametrize("expression", [0, 1.5, sympy.Symbol("n") ** 2 + 1])
def test_asymptotic_order_uses_infinity(expression: Any) -> None:
    source = pytest.importorskip("qualtran.resource_counting._call_graph")
    assert asymptotic_order(expression) == source.big_O(expression)


def test_allocator_order_is_shared_by_callbacks_and_rewrites() -> None:
    source = pytest.importorskip("qualtran.resource_counting._call_graph")
    locked = source.SympySymbolAllocator()
    native = SymbolAllocator()
    assert [native.new_symbol(name) for name in ("cv", "n", "cv", "")] == [
        locked.new_symbol(name) for name in ("cv", "n", "cv", "")
    ]
    n = native.new_symbol("n")
    assert n == locked.new_symbol("n")
    seen = []

    def declare(node: str, allocator: SymbolAllocator) -> RawCallees[str]:
        assert allocator is native
        seen.append(node)
        if node == "parent":
            return (("first", n), ("second", 3), ("first", 2), ("omitted", 15))
        return {}

    def rewrite(node: str) -> str | None:
        return {"first": "child", "second": "child", "omitted": None}.get(node, node)

    graph = symbolic_call_graph(
        "parent", declare, generalizer=rewrite, allocator=native
    )
    assert list(graph.direct_calls.items()) == [(("parent", "child"), n + 5)]
    assert graph.leaf_counts == {"child": n + 5}
    assert seen == ["parent", "child"]


def test_unavailable_callback_and_strict_errors_keep_the_count_boundary() -> None:
    def unavailable(_: str, __: SymbolAllocator) -> Iterable[tuple[str, SymbolicInt]]:
        yield "partial", 1
        raise CapabilityError("No decomposition available")

    assert component_callee_counts("root", unavailable) == {}
    assert symbolic_call_graph("root", unavailable).leaf_counts == {"root": 1}
    with pytest.raises(CapabilityError, match="No decomposition"):
        component_callee_counts("root", unavailable, ignore_unavailable=False)


@pytest.mark.parametrize("fixture", ["diamond", "funnel"])
@pytest.mark.parametrize("generalized", [False, True])
def test_frozen_shared_graphs_and_debug_output(fixture: str, generalized: bool) -> None:
    source = pytest.importorskip("qualtran.resource_counting._call_graph")
    fixtures = import_module("qualtran.resource_counting._call_graph_test")
    root, generalizer = getattr(fixtures, f"make_{fixture}_graph")()
    reference, counts = source.get_bloq_call_graph(
        root, generalizer=generalizer if generalized else None
    )
    declarations: dict[str, dict[str, SymbolicInt]]
    if fixture == "diamond":
        declarations = {
            "a": {"b1": 1, "b2": 1},
            "b1": {"c": 1},
            "b2": {"c": 1},
            "b": {"c": 1},
        }

        def canonical(node: str) -> str:
            return "b" if node in {"b1", "b2"} else node

    else:
        declarations = {
            "x": {"a1": 1, "a2": 1},
            "a1": {"b": 1},
            "a2": {"b": 1},
            "a": {"b": 1},
            "b": {"c": 1},
        }

        def canonical(node: str) -> str:
            return "a" if node in {"a1", "a2"} else node

    graph = symbolic_call_graph(
        root.name, declarations, generalizer=canonical if generalized else None
    )
    assert graph.direct_calls == {
        (a.name, b.name): data["n"] for a, b, data in reference.edges.data()
    }
    assert graph.leaf_counts == {key.name: n for key, n in counts.items()}
    assert format_call_graph_debug_text(graph.root, graph.direct_calls) == (
        source.format_call_graph_debug_text(reference)
    )


def test_notebook_conjunctions_against_frozen_declarations_and_check() -> None:
    gates = pytest.importorskip("qualtran.bloqs.mcmt.and_bloq")
    basic = import_module("qualtran.bloqs.basic_gates")
    resources = import_module("qualtran.resource_counting")
    testing = import_module("qualtran.testing")
    node: ResourceCall = BINARY_AND
    assert component_callee_counts(node, resource_declaration) == {}
    assert gates.And().bloq_counts() == {}

    def from_native(call: ResourceCall) -> Any:
        if isinstance(call, ConjunctionCall):
            assert isinstance(call.controls, tuple) and len(call.controls) == 2
            return gates.And(*call.controls)
        assert isinstance(call, CliffordCall) and call.opcode == "x"
        return basic.XGate()

    for values in ((1,) * 6, (1, 0) * 3):
        src = gates.MultiAnd(cvs=values)
        native = conjunction_graph(values)
        assert {
            from_native(call): count for call, count in native.leaf_counts.items()
        } == (src.call_graph()[1])
        generalize = conjunction_graph(values, generalize_activation=True)
        upstream_symbols = resources.SympySymbolAllocator()
        a, b = upstream_symbols.new_symbol("cv1"), upstream_symbols.new_symbol("cv2")

        def normalize(call: Any, a: Any = a, b: Any = b) -> Any:
            return gates.And(a, b) if isinstance(call, gates.And) else call

        sigma = resources.get_bloq_call_graph(
            src, generalizer=normalize, ssa=upstream_symbols
        )[1]
        assert {
            from_native(call): count for call, count in generalize.leaf_counts.items()
        } == sigma
        if 0 in values:
            assert sigma[basic.XGate()] == 6
        assert conjunction_counts_agree(values)
    testing.assert_equivalent_bloq_example_counts(gates._multi_and)


def test_frozen_logarithmic_and_decomposition_counts() -> None:
    fixtures = pytest.importorskip("qualtran.resource_counting._call_graph_test")
    source = import_module("qualtran.resource_counting")
    subcalls = tuple(f"sub_{index}" for index in range(10))
    declarations: dict[str, dict[str, SymbolicInt]] = {
        "log": {"sub_0": sympy.log(100)},
        "decomp": dict.fromkeys(("split", *subcalls, "join"), 1),
        **{name: {"T": 3} for name in subcalls},
    }

    def declare(node: str, _: SymbolAllocator) -> RawCallees[str]:
        return declarations.get(node, {})

    log_direct = component_callee_counts("log", declare)
    decomp_direct = component_callee_counts("decomp", declare)
    assert list(log_direct.values()) == [
        count for _, count in source.get_bloq_callee_counts(fixtures.BigBloq(100))
    ]
    assert (
        len(decomp_direct)
        == len(source.get_bloq_callee_counts(fixtures.DecompBloq(10)))
        == 12
    )
    assert {
        str(value) for value in symbolic_call_graph("log", declare).leaf_counts.values()
    } == {
        str(value)
        for value in fixtures.get_big_bloq_counts_graph_1(fixtures.BigBloq(100))[
            1
        ].values()
    }
    actual = symbolic_call_graph("decomp", declare)
    assert actual.leaf_counts == {"split": 1, "join": 1, "T": 30}
    assert sorted(actual.leaf_counts.values()) == sorted(
        fixtures.DecompBloq(10).call_graph()[1].values()
    )
    filtered = symbolic_call_graph(
        "decomp",
        declare,
        generalizer=lambda node: None if node in {"split", "join"} else node,
    )
    assert filtered.leaf_counts == {"T": 30}


def test_duplicate_and_adjoint_phase_callees_follow_source_generalization() -> None:
    fixtures = pytest.importorskip("qualtran.resource_counting._call_graph_test")
    source = import_module("qualtran.resource_counting")
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.resource_counting.generalizers import generalize_rotation_angle

    phase: ResourceCall = PhaseCall("t")
    inverse = PhaseCall("t", adjoint=True)
    original = TGate()
    for native_pair, source_pair in ((phase, original), (inverse, original.adjoint())):

        def raw(
            _: ResourceCall, __: SymbolAllocator, other: ResourceCall = native_pair
        ) -> RawCallees[ResourceCall]:
            return ((phase, 1), (other, 2))

        root = fixtures.OnlyCallGraphBloqShim(
            "phase", callees=[(original, 1), (source_pair, 2)]
        )
        direct = component_callee_counts(phase, raw)
        assert sorted(direct.values()) == sorted(
            count for _, count in source.get_bloq_callee_counts(root)
        )
        merged = component_callee_counts(phase, raw, generalizer=generalize_rotation)
        assert (
            list(merged.values())
            == [
                count
                for _, count in source.get_bloq_callee_counts(
                    root, generalizer=generalize_rotation_angle
                )
            ]
            == [3]
        )


def test_symbolic_declaration_and_stored_swap_reach_append() -> None:
    source = pytest.importorskip("qualtran")
    gates = import_module("qualtran.bloqs.basic_gates")
    n = sympy.Symbol("n")
    graph = declared_function_graph(n)
    path = REFERENCE / "resource_counting/call_graph.ipynb"
    cell = json.loads(_locked_asset(path.name))["cells"][15]
    namespace: dict[str, Any] = {
        "__name__": __name__,
        "And": import_module("qualtran.bloqs.mcmt").And,
    }
    exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
    upstream_graph = namespace["MyFunc"](n=n).call_graph()[1]
    assert (
        sorted(map(str, graph.leaf_counts.values()))
        == sorted(map(str, upstream_graph.values()))
        == ["2*n", "5"]
    )

    builder = source.BloqBuilder()
    a, b = builder.add_register("q0", 1), builder.add_register("q1", 1)
    a, b = builder.add(gates.CNOT(), ctrl=a, target=b)
    b, a = builder.add(gates.CNOT(), ctrl=b, target=a)
    a, b = builder.add(gates.CNOT(), ctrl=a, target=b)
    original = builder.finalize(q0=a, q1=b)
    assert original.bloq_counts() == {gates.CNOT(): 3}

    program = swap_decomposition_program()
    assert program_component_call_graph(program).leaf_counts == {"CNOT": 3}
    bound = bind_register_program(program, ((0,), (1,)))
    cirq = import_module("cirq")
    wires = tuple(cirq.LineQubit.range(2))
    upstream, registers = original.to_cirq_circuit_and_quregs(
        q0=[wires[0]], q1=[wires[1]]
    )
    expected = tuple(
        ("cx", tuple(wires.index(qubit) for qubit in gate.qubits))
        for gate in upstream.all_operations()
        if gate.gate == cirq.CNOT
    )
    assert len(expected) == len(tuple(upstream.all_operations()))
    assert (
        tuple((g.name, g.wires) for g in bound.circuit.instructions)
        == expected
        == (
            ("cx", (0, 1)),
            ("cx", (1, 0)),
            ("cx", (0, 1)),
        )
    )
    assert (
        bound.outputs
        == tuple(
            tuple(wires.index(qubit) for qubit in registers[name])
            for name in ("q0", "q1")
        )
        == ((0,), (1,))
    )
    assert not bound.events
    state = torch.tensor([1j, 2 + 3j, 4 - 5j, -6], dtype=torch.complex128)
    state /= state.norm()
    appended = Circuit(2, inputs=state, dtype=torch.complex128).append(
        FieldOperation(2, bound.circuit.instructions)
    )
    torch.testing.assert_close(appended.state().flatten(), state[[0, 2, 1, 3]])
