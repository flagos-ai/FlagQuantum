"""Compare native QREF exports against the locked source and QREF schema."""

from hashlib import sha1
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import torch
from sympy import Integer, Symbol

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.resource_interchange_catalog import (
    alias_interchange_example,
    alias_preparation_qref,
    uniform_preparation_qref,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.compiler.formats.legacy_component_export import (
    component_entry_from_program,
)
from flagquantum.compiler.formats.legacy_component_library import (
    ComponentDeclaration,
    ComponentLibrary,
    ComponentLibraryEntry,
    deserialize_component_library,
)
from flagquantum.compiler.formats.qref_resources import (
    QrefComponent,
    qref_call_graph,
    qref_component,
    qref_component_graph,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "__init__.py": "741991b9f3dce077089729e3cd75efdf904c56ee",
    "_bloq_to_qref.py": "aab3422ba170c0c909ee152d46537f78335829b8",
    "_bloq_to_qref_test.py": "e7092a50ce00daf63488a419b0bd8609e937fc38",
    "bartiq_demo.ipynb": "60e8055c7669d4dd69c8d42c4b882c9a7bfc7fc0",
}
BIT = quantum_bit_type()
CN = QrefComponent("CNOT", "CNOT", legacy_gate_budget(GateCost(clifford=1)))
PAIR = QrefComponent(
    "CompositeBloq", "CompositeBloq", legacy_gate_budget(GateCost(clifford=2))
)


def qref_reference() -> Any:
    pytest.importorskip("qref")
    source = pytest.importorskip("qualtran.qref_interop")
    assert version("qualtran") == "0.7.0"
    assert version("qref") == "0.11.0"
    return source


def test_reference_qref_family_blobs_and_package_callable() -> None:
    module = qref_reference()
    for name, expected in BLOBS.items():
        path = REFERENCE / "qref_interop" / name
        if not path.is_file():
            pytest.skip("Frozen QREF source assets are not installed")
        data = path.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected
    assert (
        module.bloq_to_qref
        is import_module("qualtran.qref_interop._bloq_to_qref").bloq_to_qref
    )


def _two_cnot_program(cross_connected: bool) -> ProgramRecord:
    controls = (Register("ctrl", BIT), Register("target", BIT))
    cnot = bit_gate_function("cnot", controls, Circuit(2).cx(0, 1).to_ir())
    builder = ComponentProgramBuilder(
        Signature((Register("q0", BIT), Register("q1", BIT))), (cnot,)
    )
    q0, q1 = builder.call_tuple(
        "cnot", {"ctrl": builder.inputs["q0"], "target": builder.inputs["q1"]}
    )
    if cross_connected:
        q0, q1 = builder.call_tuple("cnot", {"ctrl": q1, "target": q0})
    else:
        q0, q1 = builder.call_tuple("cnot", {"ctrl": q0, "target": q1})
    return builder.finalize({"q0": q0, "q1": q1})


@pytest.mark.parametrize("cross_connected", [False, True])
def test_program_ssa_exports_source_topology_and_preserves_execution(
    cross_connected: bool,
) -> None:
    qref = qref_reference()
    fixtures = import_module("qualtran.qref_interop._bloq_to_qref_test")
    original = (
        fixtures._two_cross_connected_cnots()[0]
        if cross_connected
        else fixtures._two_connected_cnots()[0]
    )
    program = _two_cnot_program(cross_connected)
    child = program.functions[1]
    library = ComponentLibrary(
        entries=(
            component_entry_from_program(
                program, "main", {"main": 0, "cnot": 1}, ComponentDeclaration("pair")
            ),
            ComponentLibraryEntry(
                1, ComponentDeclaration("cnot", registers=child.registers)
            ),
        )
    )
    exported = qref_component_graph(library, 0, {0: PAIR, 1: CN})
    assert exported.model_dump() == qref.bloq_to_qref(original).model_dump()
    assert import_module("qref.verification").verify_topology(exported)
    counted = qref_call_graph(
        symbolic_call_graph("root", {"root": {"cnot": 2}}),
        {"root": PAIR, "cnot": CN},
        component_routines={"root": exported.program.model_dump()},
    )
    with pytest.raises(ValueError, match="ports appear in a connection"):
        qref.bloq_to_qref(original, from_callgraph=True)
    assert not counted.program.ports and not counted.program.connections
    assert [child.name for child in counted.program.children] == [
        "CNOT_0",
        "CNOT_1",
        "CNOT_repeated",
    ]
    assert counted.program.children[-1].repetition.count == "2"

    bound = bind_register_program(program, ((0,), (1,)))
    expected = (0, 1), ((1, 0) if cross_connected else (0, 1))
    assert tuple(gate.wires for gate in bound.circuit.instructions) == expected
    assert all(gate.name == "cx" for gate in bound.circuit.instructions)
    assert bound.outputs == (((1,), (0,)) if cross_connected else ((0,), (1,)))
    assert not bound.events
    state = torch.tensor([1 + 2j, 3 - 2j, -4 + 3j, 5 - 4j], dtype=torch.complex128)
    state /= state.norm()
    appended = Circuit(2, inputs=state, dtype=torch.complex128).append(
        FieldOperation(2, bound.circuit.instructions)
    )
    indices = (0, 2, 3, 1) if cross_connected else (0, 1, 2, 3)
    torch.testing.assert_close(appended.state().flatten(), state[list(indices)])


def test_leaf_direction_array_and_symbolic_width() -> None:
    source = qref_reference()
    gates = import_module("qualtran.bloqs.basic_gates")
    exports = qref_component(CN, (Register("ctrl", BIT), Register("target", BIT)))
    assert exports == source.bloq_to_qref(gates.CNOT())

    n = Symbol("n")
    example = QrefComponent("Part", "Part", {"t": n, "clifford": 0, "rotations": 0})
    routine = qref_component(
        example,
        (
            Register("words", unsigned_register_type(2 * n), (2,)),
            Register("destination", BIT, (n,), RegisterSide.RIGHT),
        ),
    ).program
    assert [port.name for port in routine.ports] == [
        "in_words_0",
        "in_words_1",
        "out_destination",
        "out_words_0",
        "out_words_1",
    ]
    assert routine.local_variables == {
        "in_words_0_size": "2*n",
        "in_words_1_size": "2*n",
        "out_words_0_size": "2*n",
        "out_words_1_size": "2*n",
    }
    assert routine.input_params == ["n"]
    integer_width = Integer(3)
    amount = {"clifford": integer_width, "t": Integer(0), "rotations": Integer(0)}
    integral = qref_component(
        QrefComponent("OnEach", "OnEach", amount),
        (Register("q", opaque_register_type(integer_width)),),
    )
    assert integral == source.bloq_to_qref(gates.OnEach(integer_width, gates.XGate()))
    with pytest.raises(NotImplementedError, match="more than one"):
        qref_component(example, (Register("matrix", BIT, (2, 2)),))


@pytest.mark.parametrize("selector", [False, True, "match", "miss"])
def test_named_decomposition_selection_matches_source(selector: bool | str) -> None:
    source = qref_reference()
    wire = import_module("qualtran.serialization.bloq")
    fixtures = import_module("qualtran.serialization.bloq_test")
    original = fixtures.TestTwoCNOT()
    library = deserialize_component_library(
        wire.bloqs_to_proto(original).SerializeToString()
    )
    metadata = {
        0: QrefComponent("TestTwoCNOT", "TestTwoCNOT", PAIR.resources),
        1: CN,
    }
    native_rules = (
        selector
        if isinstance(selector, bool)
        else ("TestTwoCNOT" if selector == "match" else "CNOT",)
    )
    upstream_rules = (
        selector
        if isinstance(selector, bool)
        else (
            (
                type(original)
                if selector == "match"
                else import_module("qualtran.bloqs.basic_gates").CNOT
            ),
        )
    )
    assert qref_component_graph(
        library, 0, metadata, decomposition_rules=native_rules
    ) == source.bloq_to_qref(original, decomposition_rules=upstream_rules)


@pytest.mark.parametrize("count", [1, 3, Symbol("n")])
def test_symbolic_call_graph_preserves_repetition_and_source_model(
    count: SymbolicInt,
) -> None:
    source = qref_reference()
    gates = import_module("qualtran.bloqs.basic_gates")
    original = gates.OnEach(count, gates.XGate())
    graph = symbolic_call_graph("root", {"root": {"X": count}})
    models = {
        "root": QrefComponent(
            "OnEach",
            "OnEach",
            {"t": 0 * count, "clifford": count, "rotations": 0 * count},
        ),
        "X": QrefComponent("XGate", "XGate", legacy_gate_budget(GateCost(clifford=1))),
    }
    exported = qref_call_graph(graph, models)
    assert exported == source.bloq_to_qref(original, from_callgraph=True)


def test_alias_lesson_exports_pinned_complete_topology_and_resource_models() -> None:
    reference = qref_reference()
    preparation = import_module("qualtran.bloqs.state_preparation")
    example = alias_interchange_example()
    original = preparation.StatePreparationAliasSampling.from_probabilities(
        list(range(24))
    )
    assert example.table.mu == original.mu == 21
    emitted = alias_preparation_qref(example)
    frozen = reference.bloq_to_qref(original.decompose_bloq())
    assert emitted.model_dump() == frozen.model_dump()
    bartiq = pytest.importorskip("bartiq")
    assert version("bartiq") == "0.12.1"
    emitted.program.resources = []
    compiled = bartiq.compile_routine(emitted).routine
    assert {key: value.value for key, value in compiled.resources.items()} == {
        "t": 307,
        "clifford": 1770,
        "rotations": 2,
    }

    n = Symbol("N")
    native_symbolic = uniform_preparation_qref(n)
    source_symbolic = reference.bloq_to_qref(preparation.PrepareUniformSuperposition(n))
    assert native_symbolic.model_dump() == source_symbolic.model_dump()
