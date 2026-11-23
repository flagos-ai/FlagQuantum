"""Resolve frozen fixed-gate libraries through the ordinary append consumer."""

import ast
import csv
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha1
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.compound_control_resource import (
    CONTROLLED_HADAMARD_COST,
)
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    FixedTCall,
    PhaseCall,
    ResourceCall,
)
from flagquantum.compiler.formats.legacy_component_arguments import ComponentArgument
from flagquantum.compiler.formats.legacy_component_library import (
    ComponentLibrary,
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.errors import SerializationError
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
CASES = (
    ("XGate", False, "x"),
    ("ZGate", False, "z"),
    ("CZ", False, "cz"),
    ("CNOT", False, "cx"),
    ("Hadamard", False, "h"),
    ("CHadamard", False, "ch"),
    ("SGate", False, "s"),
    ("SGate", True, "sdg"),
    ("TGate", False, "t"),
    ("TGate", True, "tdg"),
    ("Toffoli", False, "ccx"),
    ("YGate", False, "y"),
    ("CYGate", False, "cy"),
)


def _source(name: str, adjoint: bool) -> Any:
    gates = pytest.importorskip("qualtran.bloqs.basic_gates")
    constructor = getattr(gates, name)
    return (
        constructor(is_adjoint=adjoint) if name in ("SGate", "TGate") else constructor()
    )


def _library(source: Any) -> ComponentLibrary:
    adapter = import_module("qualtran.serialization.bloq")
    return deserialize_component_library(
        adapter.bloqs_to_proto(source, max_depth=0).SerializeToString(
            deterministic=True
        )
    )


def test_frozen_resolver_inventory_and_complete_fixed_gate_sources() -> None:
    blobs = {
        "serialization/resolver_dict.py": "b10adb30b493afda81fcff2e330618415df655a2",
    }
    # The source inventory is checked in with this repository, independently
    # of the local acceptance receipt.
    inventory = (
        Path(__file__).resolve().parents[2]
        / "docs/development/ftqc_source_inventory.csv"
    )
    with inventory.open() as stream:
        rows = tuple(csv.DictReader(stream))
    for row in rows:
        path = row["path"].removeprefix("qualtran/")
        if path in {
            f"bloqs/basic_gates/{name}{suffix}.py"
            for name in ("cnot", "hadamard", "s_gate", "t_gate", "toffoli", "y_gate")
            for suffix in ("", "_test")
        }:
            blobs[path] = row["git_blob"]
    assert len(blobs) == 13
    for path, expected in blobs.items():
        data = (REFERENCE / path).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected
    tree = ast.parse((REFERENCE / "serialization/resolver_dict.py").read_text())
    resolver = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "RESOLVER_DICT"
    )
    assert isinstance(resolver, ast.Dict)
    names = {ast.literal_eval(key) for key in resolver.keys if key is not None}
    assert len(names) == 303
    assert set(fixed_gate_factories()) <= names
    assert len(fixed_gate_factories()) == 11
    assert callable(import_module("kickmix").Simulator)
    assert import_module("kickmix").__version__ == "0.1"


@pytest.mark.parametrize("name,adjoint,primitive", CASES)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_library_gate_order_complex_amplitudes_and_clean_work(
    name: str, adjoint: bool, primitive: str, dtype: torch.dtype
) -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    source = _source(name, adjoint)
    library = _library(source)
    data = serialize_component_library(library)
    assert serialize_component_library(deserialize_component_library(data)) == data
    program = resolve_component_program(library, 0, fixed_gate_factories())
    width = source.signature.n_qubits()
    # Reverse physical operands, retaining one entangled spectator and a clean
    # work slot. Logical source register/element order remains MSB first.
    wires = tuple(reversed(range(1, width + 1)))
    bound = bind_register_program(program, tuple((wire,) for wire in wires))
    assert tuple((gate.name, gate.wires) for gate in bound.circuit.instructions) == (
        (primitive, wires),
    )
    assert bound.outputs == tuple((wire,) for wire in wires)
    assert not bound.events
    size = 1 << (width + 2)
    values = torch.arange(size // 2, dtype=torch.float64)
    amplitudes = torch.complex(values + 1, 3 - 2 * values).to(dtype)
    initial = torch.zeros(size, dtype=dtype)
    initial[::2] = amplitudes / amplitudes.norm()
    circuit = Circuit(width + 2, inputs=initial, dtype=dtype)
    circuit.append(FieldOperation(width + 2, bound.circuit.instructions))
    qubits = tuple(cirq.LineQubit.range(width + 2))
    offset = 0
    quregs = {}
    for register in source.signature:
        count = register.total_bits()
        quregs[register.name] = np.array(
            [qubits[wire] for wire in wires[offset : offset + count]]
        ).reshape((*register.shape, register.bitsize))
        offset += count
    operation, _ = source.as_cirq_op(cirq.SimpleQubitManager(), **quregs)
    result = cirq.Simulator(
        dtype=np.complex64 if dtype == torch.complex64 else np.complex128
    ).simulate(
        cirq.Circuit(operation), initial_state=initial.numpy(), qubit_order=qubits
    )
    tolerance = 3e-7 if dtype == torch.complex64 else 2e-12
    actual = circuit.state().flatten()
    torch.testing.assert_close(
        actual, torch.from_numpy(result.final_state_vector), rtol=0, atol=tolerance
    )
    assert torch.count_nonzero(actual[1::2]) == 0
    counts = Counter(gate.name for gate in bound.circuit.instructions)
    assert counts == Counter({primitive: 1})
    assert counts["measure"] == counts["reset"] == 0


def test_composite_uses_shared_mapping_and_preserves_order() -> None:
    source = pytest.importorskip("qualtran")
    gates = import_module("qualtran.bloqs.basic_gates")
    builder, inputs = source.BloqBuilder.from_signature(
        source.Signature.build(a=1, b=1)
    )
    a, b = inputs["a"], inputs["b"]
    a = builder.add(gates.Hadamard(), q=a)
    a, b = builder.add(gates.CNOT(), ctrl=a, target=b)
    b = builder.add(gates.TGate(is_adjoint=True), q=b)
    a, b = builder.add(gates.CYGate(), ctrl=a, target=b)
    composite = builder.finalize(a=a, b=b)
    adapter = import_module("qualtran.serialization.bloq")
    library = deserialize_component_library(
        adapter.bloqs_to_proto(composite).SerializeToString()
    )
    bound = bind_register_program(
        resolve_component_program(library, 0, fixed_gate_factories()), ((2,), (0,))
    )
    assert [(gate.name, gate.wires) for gate in bound.circuit.instructions] == [
        ("h", (2,)),
        ("cx", (2, 0)),
        ("tdg", (0,)),
        ("cy", (2, 0)),
    ]
    assert not bound.events


@pytest.mark.parametrize("name,adjoint,primitive", CASES)
def test_complete_unitary_round_trip_and_independent_resource_model(
    name: str, adjoint: bool, primitive: str
) -> None:
    source = _source(name, adjoint)
    width = source.signature.n_qubits()
    bound = bind_register_program(
        resolve_component_program(_library(source), 0, fixed_gate_factories()),
        tuple((wire,) for wire in range(width)),
    )
    circuit = Circuit(width, dtype=torch.complex128).append(
        FieldOperation(width, bound.circuit.instructions)
    )
    expected = torch.as_tensor(source.tensor_contract())
    torch.testing.assert_close(get_unitary(circuit), expected, rtol=0, atol=2e-12)
    inverse = source.adjoint()
    restored = bind_register_program(
        resolve_component_program(_library(inverse), 0, fixed_gate_factories()),
        tuple((wire,) for wire in range(width)),
    )
    circuit.append(FieldOperation(width, restored.circuit.instructions))
    torch.testing.assert_close(
        get_unitary(circuit),
        torch.eye(1 << width, dtype=torch.complex128),
        rtol=0,
        atol=2e-12,
    )
    keys: dict[str, ResourceCall] = {
        "XGate": CliffordCall("x", 1),
        "ZGate": CliffordCall("z", 1),
        "CZ": CliffordCall("cz", 2),
        "CNOT": CliffordCall("cx", 2),
        "Hadamard": CliffordCall("h", 1),
        "CHadamard": CONTROLLED_HADAMARD_COST,
        "SGate": PhaseCall("s", adjoint),
        "TGate": PhaseCall("t", adjoint),
        "Toffoli": FixedTCall("toffoli"),
        "YGate": CliffordCall("y", 1),
        "CYGate": CliffordCall("cy", 2),
    }
    resource = import_module("qualtran.resource_counting")
    original = resource.get_cost_value(source, resource.QECGatesCost())
    modeled = query_qec_gate_cost(keys[name])
    for field in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(modeled, field) == getattr(original, field)
    # The atomic stream and the modeled synthesis categories are distinct.
    assert len(bound.circuit.instructions) == 1
    assert bound.circuit.instructions[0].name == primitive


def test_factory_replacement_uses_identical_resolver_and_append_consumer() -> None:
    library = _library(_source("Hadamard", False))
    mapping = fixed_gate_factories()
    key = library.entries[0].declaration.name
    original = mapping[key]

    def hierarchical(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        leaf = original("replacement_leaf", registers, arguments)
        assert isinstance(leaf, FunctionRecord) and leaf.registers is not None
        builder = ComponentProgramBuilder(Signature(leaf.registers), (leaf,), name=name)
        return builder.finalize(builder.call_dict(leaf.name, builder.inputs))

    reference = None
    for factory in (original, hierarchical):
        mapping[key] = factory
        bound = bind_register_program(
            resolve_component_program(library, 0, mapping), ((1,),)
        )
        circuit = Circuit(2, dtype=torch.complex128).append(
            FieldOperation(2, bound.circuit.instructions)
        )
        actual = get_unitary(circuit)
        if reference is not None:
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        reference = actual


def test_mapping_is_explicit_replaceable_and_validates_constructor_boundary() -> None:
    library = _library(_source("SGate", False))
    row = library.entries[0]
    factory = fixed_gate_factories()[row.declaration.name]
    standalone = factory("standalone", None, {})
    assert isinstance(standalone, FunctionRecord)
    assert standalone.registers == row.declaration.registers
    with pytest.raises(SerializationError, match="Invalid register signature"):
        factory("bad", (Register("other", quantum_bit_type()),), {})
    with pytest.raises(SerializationError, match="Unexpected constructor"):
        factory("bad", None, {"extra": 1})
    for invalid in (2, -1, 1.0, "yes"):
        with pytest.raises(SerializationError, match="is_adjoint"):
            factory("bad", None, {"is_adjoint": invalid})
    changed = replace(row.declaration, arguments=(ComponentArgument("is_adjoint", 1),))
    program = resolve_component_program(
        replace(library, entries=(replace(row, declaration=changed),)),
        0,
        fixed_gate_factories(),
    )
    bound = bind_register_program(program, ((0,),))
    assert bound.circuit.instructions[0].name == "sdg"
    mapping = fixed_gate_factories()
    mapping.clear()
    assert len(fixed_gate_factories()) == 11
    with pytest.raises(SerializationError, match="No native factory"):
        resolve_component_program(library, 0, mapping)
