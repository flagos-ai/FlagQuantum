"""Generic Always declarations delegate source graphs and existing typed bodies."""

from collections.abc import Hashable, Mapping
from dataclasses import replace
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler._component_trace.always import AlwaysCall
from flagquantum.compiler._component_trace.call_graph import (
    SymbolAllocator,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import CliffordCall
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.primitives.always_component import (
    always_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("controlled", (False, True))
@pytest.mark.parametrize("width", (1, 3, "n"))
def test_generic_wrapper_delegates_original_direct_graph_without_child_edge(
    controlled: bool, width: int | str
) -> None:
    sympy = pytest.importorskip("sympy")
    from qualtran.bloqs.basic_gates import CSwap, Swap, TwoBitCSwap, TwoBitSwap
    from qualtran.bloqs.bookkeeping import Always
    from qualtran.resource_counting import SympySymbolAllocator

    n = sympy.Symbol("n") if width == "n" else width
    original_child = (CSwap if controlled else Swap)(n)
    from flagquantum.compiler.registers.primitives.swap_component import SwapDeclaration

    child = SwapDeclaration(n, controlled)
    leaf = next(iter(child.direct_calls()))
    wrapper = AlwaysCall(child)
    allocator = SymbolAllocator()
    visited = []

    def declaration(node: Hashable, symbols: SymbolAllocator) -> dict[Hashable, Any]:
        visited.append(node)
        assert symbols is allocator
        if isinstance(node, AlwaysCall):
            return node.callee_counts(declaration, allocator=symbols)
        return child.direct_calls() if node == child else {}

    expected = Always(original_child).build_call_graph(SympySymbolAllocator())
    assert expected == {(TwoBitCSwap if controlled else TwoBitSwap)(): n}
    assert wrapper.callee_counts(declaration, allocator=allocator) == {leaf: n}
    root: Hashable = wrapper
    graph = symbolic_call_graph(root, declaration, allocator=allocator)
    assert graph.direct_calls == {(wrapper, leaf): n}
    assert graph.leaf_counts == {leaf: n}
    assert child in visited and (wrapper, child) not in graph.direct_calls
    assert str(wrapper) == str(child)
    assert wrapper.adjoint().adjoint() == wrapper

    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.compiler._component_trace.cost_query import (
        CostMetric,
        query_component_cost,
    )
    from flagquantum.compiler._component_trace.qec_gate_cost import (
        QECGateCost,
        query_qec_gate_cost,
    )
    from flagquantum.compiler._component_trace.resource_generalization import (
        ControlledSwapCall,
    )

    # The independent standard leaf model consumes the wrapper's delegated edges.
    assert isinstance(leaf, (CliffordCall, ControlledSwapCall))
    leaf_cost = query_qec_gate_cost(leaf)
    metric = CostMetric[Hashable, QECGateCost](
        lambda node, query: sum(
            (
                query(callee) * count
                for callee, count in declaration(node, allocator).items()
            ),
            QECGateCost(),
        ),
        QECGateCost(),
    )
    static_costs: dict[Hashable, QECGateCost] = {leaf: leaf_cost}
    modeled = query_component_cost(root, metric, static_costs=static_costs)
    assert (
        modeled.asdict()
        == get_cost_value(Always(original_child), QECGatesCost()).asdict()
    )


def test_same_allocator_duplicate_children_and_original_declaration_failures() -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError, QUInt
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.bookkeeping import Allocate, Always
    from qualtran.bloqs.mcmt import And
    from qualtran.resource_counting import SympySymbolAllocator

    symbols = SymbolAllocator()
    wrapper = AlwaysCall("child")
    calls = []

    def declared(
        node: Hashable, allocator: SymbolAllocator
    ) -> list[tuple[Hashable, Any]]:
        assert node == "child" and allocator is symbols
        calls.append(node)
        n = allocator.new_symbol("n")
        return [("t", n), ("x", 1), ("t", 2)]

    first = wrapper.callee_counts(declared, allocator=symbols)
    second = wrapper.callee_counts(declared, allocator=symbols)
    assert list(first) == ["t", "x"] and calls == ["child", "child"]
    assert str(first["t"]) == "_n0 + 2" and str(second["t"]) == "_n1 + 2"

    def unavailable(node: Hashable, allocator: SymbolAllocator) -> dict[Hashable, int]:
        raise diagnostic

    for source, error in (
        (XGate(), DecomposeNotImplementedError),
        (And(), DecomposeTypeError),
        (Allocate(QUInt(2)), DecomposeTypeError),
    ):
        with pytest.raises(error) as expected:
            Always(source).build_call_graph(SympySymbolAllocator())
        diagnostic = expected.value

        with pytest.raises(error) as actual:
            wrapper.callee_counts(unavailable)
        assert actual.value is diagnostic


@pytest.mark.parametrize("gate", ("x", "h"))
@pytest.mark.parametrize("depth", (0, 1, 2))
def test_actual_foreign_wrapper_references_resolve_to_existing_child_body(
    gate: str, depth: int
) -> None:
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.bookkeeping import Always
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    source = Always(XGate() if gate == "x" else Hadamard())
    proto = bloqs_to_proto(source, max_depth=depth)
    wire = proto.SerializeToString(deterministic=True)
    library = deserialize_component_library(wire)
    assert serialize_component_library(library) == wire
    assert bloqs_from_proto(proto)[0] == source
    assert len(library.entries) == 2

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None
        if "subbloq" in arguments:
            assert set(arguments) == {"subbloq"}
            child = arguments["subbloq"]
            assert isinstance(child, FunctionRecord)
            return always_component_program(
                ProgramRecord((child,)), child.name, name
            ).functions[0]
        assert arguments == {}
        return bit_gate_function(
            name, registers, CircuitIR(1, (Instruction(gate, (0,)),))
        )

    program = resolve_component_program(
        library, 0, {entry.declaration.name: factory for entry in library.entries}
    )
    assert program.functions[0].registers == tuple(native_signature(source.signature))
    bound = bind_register_program(program, ((1,),))
    assert bound.outputs == ((1,),) and bound.circuit.instructions == (
        Instruction(gate, (1,)),
    )
    generator = torch.Generator().manual_seed(71)
    initial = torch.randn(4, dtype=torch.complex128, generator=generator)
    initial /= initial.norm()
    expected = (
        initial.reshape(2, 2) @ torch.as_tensor(source.subbloq.tensor_contract()).T
    )
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(2, bound.circuit.instructions))
    torch.testing.assert_close(
        circuit.state().flatten(), expected.flatten(), rtol=0, atol=2e-12
    )
    circuit.append(FieldOperation(2, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)


def test_undeclared_typed_child_stays_logical_until_body_replacement() -> None:
    from flagquantum.compiler._component_trace.program_cost_query import (
        query_program_qec_gate_cost,
    )
    from flagquantum.core.ir.program.signatures import Signature

    child = FunctionRecord("child", (), registers=tuple(Signature.build(q=1)))
    program = always_component_program(ProgramRecord((child,)), "child")
    assert query_program_qec_gate_cost(
        program, {"child": CliffordCall("x", 1)}, "always"
    ).asdict() == {"clifford": 1}
    with pytest.raises(ValueError, match="Missing QEC terminal key"):
        query_program_qec_gate_cost(program, {}, "always")
    with pytest.raises(ValueError, match="physical lowering requires component bodies"):
        bind_register_program(program, ((0,),), entry="always")
    assert child.registers is not None
    replacement = bit_gate_function(
        "child", child.registers, CircuitIR(1, (Instruction("x", (0,)),))
    )
    restored = replace(
        program,
        functions=tuple(
            replacement if f.name == "child" else f for f in program.functions
        ),
    )
    assert bind_register_program(
        restored, ((0,),), entry="always"
    ).circuit.instructions == (
        Instruction("x", (0,)),
    )


def test_original_generic_graph_reuses_allocator_through_nested_wrappers() -> None:
    from qualtran.bloqs.bookkeeping import Always
    from qualtran.bloqs.for_testing.with_call_graph import TestBloqWithCallGraph
    from qualtran.resource_counting import SympySymbolAllocator

    source = Always(Always(TestBloqWithCallGraph()))
    original_symbols = SympySymbolAllocator()
    native_symbols = SymbolAllocator()
    wrapper = AlwaysCall(AlwaysCall("fixture"))

    def declaration(node: Hashable, symbols: SymbolAllocator) -> dict[Hashable, Any]:
        assert symbols is native_symbols
        if isinstance(node, AlwaysCall):
            return node.callee_counts(declaration, allocator=symbols)
        assert node == "fixture"
        return {
            "TestParallelCombo": 1,
            "TestSerialCombo": 1,
            "TestAtom": symbols.new_symbol("n"),
        }

    for _ in range(3):
        expected = source.build_call_graph(original_symbols)
        assert wrapper.callee_counts(declaration, allocator=native_symbols) == {
            type(child).__name__: count for child, count in expected.items()
        }
