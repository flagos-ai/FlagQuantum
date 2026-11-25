"""Close original OnEach metadata, actual library and typed SDK consumers."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
import sympy
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    ResolvedArgument,
    resolve_component_program,
)
from flagquantum.compiler.registers.bits.on_each_declaration import OnEachDeclaration
from flagquantum.compiler.registers.component_graph import _namespace_helpers
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_register_on_each import bit_child
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
cirq = pytest.importorskip("cirq")


def native_declaration(source: Any, child: FunctionRecord) -> OnEachDeclaration:
    from qualtran import QAny
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    dtype = native_signature(
        SourceSignature((SourceRegister("q", source.target_dtype or QAny(source.n)),))
    )[0].dtype
    return OnEachDeclaration(
        source.n, child, None if source.target_dtype is None else dtype
    )


@pytest.mark.parametrize(
    "width,kind",
    [
        (width, kind)
        for width in (0, 1, 3, "n")
        for kind in (
            "default",
            "unsigned",
            "signed",
            "ones",
            "fixed",
            "signed_fixed",
            "signed_fraction",
        )
        if not (kind == "ones" and width in (0, 1))
        and not (kind == "signed" and width == 0)
        and not (kind in {"signed_fixed", "signed_fraction"} and width in (0, 1))
    ],
)
def test_generic_declaration_and_actual_foreign_library_resolution(
    width: int | str, kind: str
) -> None:
    from qualtran import QFxp, QInt, QIntOnesComp, QUInt
    from qualtran.bloqs.basic_gates import OnEach, XGate
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    n = sympy.Symbol("n") if width == "n" else width
    assert isinstance(n, (int, sympy.Expr))
    targets = {
        "default": None,
        "unsigned": QUInt(n),
        "fixed": QFxp(n, -2),
    }
    if kind in {"signed", "ones"}:
        targets[kind] = QInt(n) if kind == "signed" else QIntOnesComp(n)
    if kind in {"signed_fixed", "signed_fraction"}:
        targets[kind] = QFxp(n, -2 if kind == "signed_fixed" else 1, signed=True)
    source = OnEach(n, XGate(), targets[kind])
    native = native_declaration(source, bit_child("x"))
    assert native.signature == native_signature(source.signature)
    assert native.direct_calls() == {"x": source.bloq_counts()[XGate()]}
    for depth in (0, 1, 2):
        proto = bloqs_to_proto(source, max_depth=depth)
        wire = proto.SerializeToString(deterministic=True)
        assert serialize_component_library(deserialize_component_library(wire)) == wire
        assert bloqs_from_proto(proto)[0] == source
    library = deserialize_component_library(
        bloqs_to_proto(source, max_depth=0).SerializeToString()
    )
    names = {entry.declaration.name for entry in library.entries}

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        assert registers is not None
        if "gate" not in arguments:
            assert arguments == {}
            return replace(bit_child("x"), name=name)
        child = arguments["gate"]
        assert isinstance(child, FunctionRecord)
        target = arguments.get("target_dtype")
        assert target is None or isinstance(target, RegisterType)
        declaration = OnEachDeclaration(n, child, target)
        assert tuple(declaration.signature) == registers
        if not isinstance(n, int):
            return FunctionRecord(name, (), registers=registers)
        program = declaration.program()
        program = replace(
            program,
            functions=(
                replace(program.functions[0], name=name),
                *program.functions[1:],
            ),
        )
        return _namespace_helpers(program, name)

    resolved = resolve_component_program(library, 0, dict.fromkeys(names, factory))
    assert resolved.functions[0].registers == tuple(native.signature)
    if isinstance(n, int):
        direct = bind_register_program(
            native.program(), (tuple(range(n)),), borrowed=(n,)
        )
        actual = bind_register_program(resolved, (tuple(range(n)),), borrowed=(n,))
        assert actual.circuit == direct.circuit and actual.outputs == direct.outputs
        assert not actual.events and verify_program_record(resolved) == ()
    else:
        with pytest.raises(ValueError, match="concrete width"):
            native.program()


@pytest.mark.parametrize(
    "kind",
    ("opaque", "uint", "signed", "ones", "fixed", "signed_fixed", "signed_fraction"),
)
@pytest.mark.parametrize("opcode", ("x", "h", "s", "sdg", "t", "tdg"))
def test_actual_typed_sdk_body_and_spectator_complex_roundtrip(
    kind: str, opcode: str
) -> None:
    from qualtran import QAny, QFxp, QInt, QIntOnesComp, QUInt
    from qualtran.bloqs.basic_gates import Hadamard, OnEach, SGate, TGate, XGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    targets = {
        "opaque": QAny(3),
        "uint": QUInt(3),
        "signed": QInt(3),
        "ones": QIntOnesComp(3),
        "fixed": QFxp(3, -2),
        "signed_fixed": QFxp(3, -2, signed=True),
        "signed_fraction": QFxp(3, 1, signed=True),
    }
    leaves = {
        "x": XGate(),
        "h": Hadamard(),
        "s": SGate(),
        "sdg": SGate().adjoint(),
        "t": TGate(),
        "tdg": TGate().adjoint(),
    }
    source = OnEach(3, leaves[opcode], targets[kind])
    declaration = native_declaration(source, bit_child(opcode))
    wires = (3, 0, 2)
    bound = bind_register_program(declaration.program(), (wires,), borrowed=(1,))
    operation = source.on(*cirq.LineQubit.range(3))
    imported = import_cirq_register_gate(operation, signature=declaration.signature)
    assert imported is not None
    sdk = bind_register_program(imported, (wires,), borrowed=(1,))
    assert sdk.circuit.instructions == bound.circuit.instructions
    assert bound.outputs == (wires,) and not bound.events
    assert Counter(g.name for g in bound.circuit.instructions) == {opcode: 3}
    circuit = fq.Circuit(4, dtype=torch.complex128)
    circuit.append(FieldOperation(4, bound.circuit.instructions))
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128)
    expected = torch.zeros((16, 16), dtype=torch.complex128)
    order = (*wires, 1)
    inverse = tuple(order.index(wire) for wire in range(4))
    for column in range(16):
        state = torch.eye(16, dtype=torch.complex128)[:, column]
        expected[:, column] = (
            (matrix @ state.reshape(2, 2, 2, 2).permute(order).reshape(8, 2))
            .reshape(2, 2, 2, 2)
            .permute(inverse)
            .reshape(-1)
        )
    torch.testing.assert_close(get_unitary(circuit), expected, rtol=0, atol=2e-12)
    from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter

    undo = import_register_emitter(source).snapshots[1].instructions
    mapped = tuple(replace(g, wires=tuple(wires[w] for w in g.wires)) for g in undo)
    circuit.append(FieldOperation(4, mapped))
    torch.testing.assert_close(
        get_unitary(circuit), torch.eye(16, dtype=torch.complex128), rtol=0, atol=2e-12
    )
    costs = get_cost_value(source, QECGatesCost())
    assert costs.t == (3 if opcode in {"t", "tdg"} else 0)
    assert costs.clifford == (0 if opcode in {"t", "tdg"} else 3)
    assert get_cost_value(source, QubitCount()) == 3


def test_source_constructor_metadata_and_executable_failures() -> None:
    from qualtran import Bloq, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import OnEach, XGate, ZeroEffect, ZeroState

    class ShapedMetadata:
        @property
        def signature(self) -> Any:
            return SourceSignature((SourceRegister("q", QBit(), shape=(1,)),))

    shaped_child = type("ShapedChild", (Bloq,), {"signature": ShapedMetadata.signature})
    for leaf in (ZeroState(), ZeroEffect(), shaped_child(), XGate()):
        source = OnEach(3, leaf, QBit() if isinstance(leaf, XGate) else None)
        child = FunctionRecord(
            "child", (), registers=tuple(native_signature(leaf.signature))
        )
        native = native_declaration(source, child)
        assert native.signature == native_signature(source.signature)
        assert native.direct_calls() == {"child": 3}
        with pytest.raises((ValueError, IndexError, AttributeError)):
            native.program()
    for registers in (
        (),
        (Register("r", opaque_register_type(1)),),
        (Register("q", opaque_register_type(2)),),
    ):
        with pytest.raises(ValueError):
            OnEachDeclaration(3, FunctionRecord("bad", (), registers=registers))
    with pytest.raises(ValueError, match="target dtype"):
        OnEachDeclaration(3, bit_child("x"), opaque_register_type(2))
