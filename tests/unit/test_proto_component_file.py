"""Frozen component wire messages and typed native library records."""

import ast
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha1
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sympy import Symbol

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.formats._legacy_component_schema import (
    _component_library_message,
)
from flagquantum.compiler.formats.legacy_component_arguments import (
    ComponentArgument,
    ComponentReference,
    ControlSpecification,
    ExternalGatePayload,
    deserialize_component_argument,
    serialize_component_argument,
)
from flagquantum.compiler.formats.legacy_component_discovery import (
    discover_component_library,
)
from flagquantum.compiler.formats.legacy_component_export import (
    component_entry_from_program,
)
from flagquantum.compiler.formats.legacy_component_library import (
    ComponentOccurrence,
    LibraryBoundary,
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_control_specs import (
    control_registers_from_spec,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "protos/bloq.proto": "40e467b563bf100259312c6020ea116484e91baa",
    "protos/bloq_pb2.py": "383b347f00bf99dde57290882b9ce0a4c3ef1a62",
    "protos/bloq_pb2.pyi": "b3481a7aabd70722b68a91541b53a7473f53d2d4",
    "serialization/bloq.py": "132930b2804536155f5b242dbede88518fcca459",
    "serialization/bloq_test.py": "d423a5bca747ed4828ea7758ee8df670723d8e5f",
    "serialization/resolver_dict.py": "b10adb30b493afda81fcff2e330618415df655a2",
}


def test_frozen_library_wire_and_dependencies() -> None:
    for name, expected in BLOBS.items():
        data = (REFERENCE / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected
    source = import_module("qualtran.protos.bloq_pb2")
    stub = ast.parse((REFERENCE / "protos/bloq_pb2.pyi").read_text())
    seen: set[tuple[str, str]] = set()

    def compare(left: Any, right: Any) -> None:
        key = (left.full_name, right.full_name)
        if key in seen:
            return
        seen.add(key)
        assert left.GetOptions().map_entry == right.GetOptions().map_entry
        assert len(left.fields) == len(right.fields)
        if left.file is source.DESCRIPTOR:
            scope = stub.body
            for name in left.full_name.split(".")[1:]:
                cls = next(
                    node
                    for node in scope
                    if isinstance(node, ast.ClassDef) and node.name == name
                )
                scope = cls.body
            constructor = next(
                node
                for node in scope
                if isinstance(node, ast.FunctionDef) and node.name == "__init__"
            )
            assert [arg.arg for arg in constructor.args.kwonlyargs] == [
                field.name for field in right.fields
            ]
        for original, native in zip(left.fields, right.fields, strict=True):
            assert (original.number, original.type, original.label) == (
                native.number,
                native.type,
                native.label,
            )
            assert original.has_presence == native.has_presence
            assert getattr(original.containing_oneof, "name", None) == getattr(
                native.containing_oneof, "name", None
            )
            if original.enum_type is not None:
                assert [value.number for value in original.enum_type.values] == [
                    value.number for value in native.enum_type.values
                ]
            if original.message_type is not None:
                compare(original.message_type, native.message_type)

    compare(source.BloqLibrary.DESCRIPTOR, _component_library_message().DESCRIPTOR)
    assert len(seen) == 29


def test_named_arguments_cover_every_frozen_arm() -> None:
    source = import_module("qualtran")
    adapter = import_module("qualtran.serialization.bloq")
    types = import_module("qualtran.protos.bloq_pb2")
    cirq = import_module("cirq")
    point = import_module("qualtran.bloqs.cryptography.ecc").ECPoint(15, 13, 17, 0)
    register = source.Register("control", source.QBit(), shape=(2,))
    values = (
        0,
        0.0,
        "",
        Symbol("a") * Symbol("b") + Symbol("c") / 10,
        np.arange(100, dtype=np.complex128).reshape(10, 10),
        cirq.CNOT,
        source.QInt(3),
        register,
        (register,),
        source.CtrlSpec(cvs=[0, 1]),
        1.5 + 2.25j,
        point,
    )
    fixtures = [adapter.arg_to_proto(name="parameter", val=value) for value in values]
    fixtures.extend(
        (
            adapter.arg_to_proto(name="nil", val=()),
            types.BloqArg(name="child", subbloq=0),
        )
    )
    assert len({p.WhichOneof("val") for p in fixtures}) == 13
    for original in fixtures:
        data = original.SerializeToString(deterministic=True)
        native = deserialize_component_argument(data)
        assert native.name == original.name
        assert serialize_component_argument(native) == data
        if original.HasField("cirq_json_gzip"):
            assert native.value == ExternalGatePayload(original.cirq_json_gzip)
        if original.HasField("subbloq"):
            assert native.value == ComponentReference(0)
    with pytest.raises(SerializationError, match="no selected value"):
        deserialize_component_argument(types.BloqArg(name="unused").SerializeToString())
    for value in ([1, 2], ((1, 2), (3, 4))):
        original = adapter.arg_to_proto(name="array", val=value)
        assert serialize_component_argument(ComponentArgument("array", value)) == (
            original.SerializeToString(deterministic=True)
        )


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_frozen_fixtures_preserve_table_edges_shallow_rows_and_counts(
    depth: int,
) -> None:
    adapter = import_module("qualtran.serialization.bloq")
    fixtures = import_module("qualtran.serialization.bloq_test")
    pair = fixtures.TestTwoCSwap(3)
    for component in (
        fixtures.TestCNOTCirq(),
        fixtures.TestTwoCNOT().decompose_bloq(),
        pair,
        fixtures.TestMetaBloq(pair, pair.decompose_bloq()),
    ):
        original = adapter.bloqs_to_proto(component, name="library", max_depth=depth)
        data = original.SerializeToString(deterministic=True)
        native = deserialize_component_library(data)
        assert serialize_component_library(native) == data
        for entry, row in zip(native.entries, original.table, strict=True):
            assert entry.identifier == row.bloq_id
            assert entry.declaration.name == row.bloq.name
            assert [child for child, _ in entry.counts] == sorted(row.bloq_counts)
            assert len(entry.connections) == len(row.decomposition)
            for edge, original_edge in zip(
                entry.connections, row.decomposition, strict=True
            ):
                for port, old in (
                    (edge.source, original_edge.left),
                    (edge.destination, original_edge.right),
                ):
                    assert port.register.name == old.register.name
                    assert port.index == tuple(old.index)
                    if old.HasField("dangling_t"):
                        assert port.owner == LibraryBoundary(old.dangling_t)
                    else:
                        assert port.owner == ComponentOccurrence(
                            old.bloq_instance.instance_id, old.bloq_instance.bloq_id
                        )


def test_modular_exponentiation_fixture_preserves_full_wire_library() -> None:
    adapter = import_module("qualtran.serialization.bloq")
    fixtures = import_module("qualtran.serialization.bloq_test")
    composite = fixtures.ModExp.make_for_shor(17 * 19, g=8).decompose_bloq()
    original = adapter.bloqs_to_proto(composite, max_depth=1)
    data = original.SerializeToString(deterministic=True)
    native = deserialize_component_library(data)
    assert len(native.entries) == 9
    assert serialize_component_library(native) == data
    assert composite in adapter.bloqs_from_proto(original)


@pytest.mark.parametrize("max_depth", [0, 1, 2, 3])
@pytest.mark.parametrize("cutoff", [None, 0])
def test_native_discovery_matches_frozen_depth_and_predicate(
    max_depth: int, cutoff: int | None
) -> None:
    adapter = import_module("qualtran.serialization.bloq")
    fixtures = import_module("qualtran.serialization.bloq_test")
    pair = fixtures.TestTwoCSwap(3)
    roots = (fixtures.TestMetaBloq(pair, pair.decompose_bloq()), pair)
    complete = adapter.bloqs_to_proto(*roots, name="discovery", max_depth=4)
    catalog = deserialize_component_library(complete.SerializeToString())
    pair_id = next(
        row.identifier
        for row in catalog.entries
        if row.declaration.name.endswith(".TestTwoCSwap")
    )
    discovered = discover_component_library(
        catalog,
        (0, pair_id),
        max_depth=max_depth,
        predicate=lambda occurrence: occurrence.instance_id != cutoff,
    )
    source = adapter.bloqs_to_proto(
        *roots,
        name="discovery",
        max_depth=max_depth,
        pred=lambda instance: instance.i != cutoff,
    )
    assert serialize_component_library(discovered) == source.SerializeToString(
        deterministic=True
    )


def _native_cnot(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    assert registers is not None and not arguments
    return bit_gate_function(name, registers, Circuit(2).cx(0, 1).to_ir())


def test_decoded_source_connections_reach_append_with_exact_gate_order() -> None:
    adapter = import_module("qualtran.serialization.bloq")
    fixtures = import_module("qualtran.serialization.bloq_test")
    cirq = import_module("cirq")
    composite = fixtures.TestTwoCNOT().decompose_bloq()
    data = adapter.bloqs_to_proto(composite).SerializeToString(deterministic=True)
    library = deserialize_component_library(data)
    leaf = library.entries[1].declaration.name
    factories = {leaf: _native_cnot}
    program = resolve_component_program(library, 0, factories)
    root, child = library.entries
    exported = component_entry_from_program(
        program,
        "main",
        {"main": 0, "legacy_component_1": 1},
        root.declaration,
        counts=root.counts,
    )
    assert exported == root
    assert (
        serialize_component_library(replace(library, entries=(exported, child))) == data
    )
    bound = bind_register_program(program, ((0,), (1,)))
    wire_map = tuple(cirq.LineQubit.range(2))
    source, returned = composite.to_cirq_circuit_and_quregs(
        q1=[wire_map[0]], q2=[wire_map[1]]
    )
    expected = tuple(
        ("cx", tuple(wire_map.index(q) for q in op.qubits))
        for op in source.all_operations()
    )
    assert expected == (("cx", (0, 1)), ("cx", (1, 0)))
    assert tuple((g.name, g.wires) for g in bound.circuit.instructions) == expected
    assert (
        bound.outputs
        == tuple(
            tuple(wire_map.index(q) for q in returned[name]) for name in ("q1", "q2")
        )
        == ((1,), (0,))
    )
    assert not bound.events

    initial = torch.tensor([1 + 2j, -3 + 4j, 5 - 6j, 7], dtype=torch.complex128)
    initial /= initial.norm()
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(2, bound.circuit.instructions))
    upstream = cirq.Simulator(dtype=np.complex128).simulate(
        source, initial_state=initial.numpy(), qubit_order=wire_map
    )
    torch.testing.assert_close(
        circuit.state().flatten(),
        torch.from_numpy(upstream.final_state_vector),
        rtol=0,
        atol=2e-12,
    )


def test_composite_resolution_refuses_unregistered_or_inconsistent_inputs() -> None:
    adapter = import_module("qualtran.serialization.bloq")
    fixtures = import_module("qualtran.serialization.bloq_test")
    source = adapter.bloqs_to_proto(fixtures.TestTwoCNOT().decompose_bloq())
    library = deserialize_component_library(source.SerializeToString())
    root, child = library.entries
    with pytest.raises(SerializationError, match="No native factory"):
        resolve_component_program(library, 0, {})
    with pytest.raises(SerializationError, match="Unknown component definition"):
        resolve_component_program(replace(library, entries=(root,)), 0, {})
    malformed = replace(root, connections=(*root.connections, root.connections[0]))
    duplicate = replace(library, entries=(malformed, child))
    with pytest.raises(SerializationError, match="Duplicate connection"):
        resolve_component_program(duplicate, 0, {child.declaration.name: _native_cnot})


def test_resolved_subcomponent_and_adjoint_keep_every_measurement_branch() -> None:
    source = import_module("qualtran")
    adapter = import_module("qualtran.serialization.bloq")
    predicate = import_module("qualtran.bloqs.mcmt.ctrl_spec_and")
    resource = import_module("qualtran.resource_counting")
    forward = predicate.CtrlSpecAnd(source.CtrlSpec(cvs=[0, 1, 1, 0]))
    erase = forward.adjoint()
    root = source.Signature(forward.control_registers)
    builder, inputs = source.BloqBuilder.from_signature(root)
    composite = builder.finalize(
        **builder.add_d(erase, **builder.add_d(forward, **inputs))
    )
    library = deserialize_component_library(
        adapter.bloqs_to_proto(composite).SerializeToString()
    )
    clauses: dict[str, ControlSpecification] = {}

    def native_forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        assert registers is not None
        spec = arguments["ctrl_spec"]
        assert isinstance(spec, ControlSpecification)
        clauses[name] = spec
        return control_predicate_program(
            registers[:1], control_registers_from_spec(spec.clauses), name=name
        )

    def native_erase(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        assert registers is not None
        child = arguments["subbloq"]
        assert isinstance(child, FunctionRecord)
        return control_predicate_program(
            registers[:1],
            control_registers_from_spec(clauses[child.name].clauses),
            name=name,
            uncompute=True,
        )

    names = {
        type(op).__module__ + "." + type(op).__qualname__: factory
        for op, factory in ((forward, native_forward), (erase, native_erase))
    }
    program = resolve_component_program(library, 0, names)
    row = library.entries[0]
    exported = component_entry_from_program(
        program,
        "main",
        {
            "main": 0,
            **{
                f"legacy_component_{entry.identifier}": entry.identifier
                for entry in library.entries[1:]
            },
        },
        row.declaration,
        counts=row.counts,
    )
    assert exported == row
    bound = bind_register_program(program, ((0,), (1,), (2,), (3,)), clean=(4, 5, 6))
    appended = Circuit(7).append(FieldOperation(7, bound.circuit.instructions)).to_ir()
    expected = reference_arithmetic_gates(
        composite, {"ctrl_0": (0, 1, 2, 3)}, (4, 5, 6), {6: 0, 5: 1, 4: 2}
    )
    assert_source(appended.instructions, expected)
    counts = Counter(gate.name for gate in appended.instructions)
    model = resource.get_cost_value(composite, resource.QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 12
    assert counts["measure"] == model.measurement == 3
    assert bound.outputs == ((0,), (1,), (2,), (3,))

    values = torch.arange(16, dtype=torch.float64)
    amplitudes = torch.complex(values + 1, 4 - values)
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::8] = amplitudes / amplitudes.norm()
    for history in product((0, 1), repeat=3):
        torch.testing.assert_close(
            forced_branch(initial, appended.instructions, history),
            initial / 2**1.5,
            rtol=0,
            atol=2e-12,
        )
