"""Compare the declaration-only fixture and source package entry mapping."""

import pytest

from examples.algorithms.resource_call_graphs import declared_example
from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import ResourceCall
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from tests.unit.atomic_register_fixtures import AtomicRegisterFixture
from tests.unit.composition_fixtures import composition_fixture
from tests.unit.multi_register_fixture import multi_register_fixture
from tests.unit.test_cast_add_register_fixture import cast_add_fixture

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_name(node: ResourceCall) -> str:
    assert isinstance(node, ComponentCost)
    return {
        "Declared program": "TestBloqWithCallGraph",
        "Serial": "TestSerialCombo",
        "Parallel": "TestParallelCombo",
        "Atom": "TestAtom",
    }.get(node.name, node.name)


def test_tag_generalization_and_declared_only_boundary() -> None:
    pytest.importorskip("qualtran")
    from attrs import evolve
    from qualtran.bloqs.for_testing import TestAtom, TestBloqWithCallGraph
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    root, declarations = declared_example()
    root_key: ResourceCall = root
    atom = next(node for node in declarations[root] if source_name(node) == "TestAtom")

    def untag(node: ResourceCall) -> ResourceCall:
        assert isinstance(node, ComponentCost)
        return atom if node.name.startswith("Atom ") else node

    graph = symbolic_call_graph(root_key, declarations, generalizer=untag)
    source = TestBloqWithCallGraph()
    original, leaves = source.call_graph(
        generalizer=lambda node: (
            evolve(node, tag=None) if isinstance(node, TestAtom) else node
        )
    )
    assert len(source.signature) == root.qubits == 0
    assert len(graph.direct_calls) == original.number_of_edges() == 7
    assert len(graph.leaf_counts) == len(leaves) == 3
    assert {
        (source_name(a), source_name(b)): n for (a, b), n in graph.direct_calls.items()
    } == {(str(a), str(b)): data["n"] for a, b, data in original.edges(data=True)}
    assert {source_name(node): count for node, count in graph.leaf_counts.items()} == {
        str(node): count for node, count in leaves.items()
    }
    native = query_qec_gate_cost(root, declared_calls=declarations, generalizer=untag)
    assert native.t == get_cost_value(source, QECGatesCost()).t

    with pytest.raises(
        NotImplementedError, match="does not support classical simulation"
    ):
        source.call_classically()
    with pytest.raises(ValueError, match="unknown classical component entry"):
        ComponentClassicalState(
            composition_fixture("serial"), (), {}, lambda f, v: None, entry=root.name
        )


def test_source_fixture_package_exports_have_native_endpoints() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs import for_testing

    def signature_names(record: ProgramRecord) -> tuple[str, ...]:
        main = next(
            function for function in record.functions if function.name == "main"
        )
        assert main.registers is not None
        return tuple(register.name for register in main.registers)

    assert declared_example()[0].qubits == 0
    endpoints = {
        "TestAtom": tuple(r.name for r in AtomicRegisterFixture("x").signature),
        "TestTwoBitOp": tuple(r.name for r in AtomicRegisterFixture("cx").signature),
        "TestCastToFrom": signature_names(cast_add_fixture(4)),
        "TestMultiRegister": signature_names(multi_register_fixture()),
        "TestBloqWithCallGraph": (),
        "TestParallelCombo": signature_names(composition_fixture("parallel")),
        "TestSerialCombo": signature_names(composition_fixture("serial")),
    }
    original = {name for name in vars(for_testing) if name.startswith("Test")}
    assert endpoints.keys() == original
    for name, ports in endpoints.items():
        assert (
            tuple(register.name for register in getattr(for_testing, name)().signature)
            == ports
        )
