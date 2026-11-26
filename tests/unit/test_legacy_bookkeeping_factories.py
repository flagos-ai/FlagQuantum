"""Resolve complete lifetime and storage-view sources using existing owners."""

from dataclasses import replace
from importlib import import_module
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.formats.legacy_bookkeeping_factories import (
    bookkeeping_factories,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
    fixed_gate_factories,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.errors import SerializationError
from tests.unit.test_legacy_fixed_gate_factories import _library
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
KINDS = ("Allocate", "Free", "Split", "Join", "Cast")
DTYPES = (
    "QBit",
    "QAny",
    "QUInt",
    "QInt",
    "QFxp",
    "QIntOnesComp",
    "BQUInt",
    "QMontgomeryUInt",
)


def _dtype(kind: str) -> Any:
    source = pytest.importorskip("qualtran")
    constructor = getattr(source, kind)
    if kind == "QBit":
        return constructor()
    if kind == "QFxp":
        return constructor(3, -2)
    if kind == "BQUInt":
        return constructor(3, 5)
    return constructor(3)


@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("kind", KINDS)
def test_original_constructor_signatures_and_exact_storage_views(
    dtype_name: str, kind: str
) -> None:
    source = pytest.importorskip("qualtran")
    bookkeeping = import_module("qualtran.bloqs.bookkeeping")
    dtype = _dtype(dtype_name)
    constructor = getattr(bookkeeping, kind)
    original = (
        constructor(source.QAny(dtype.num_qubits), dtype)
        if kind == "Cast"
        else constructor(dtype)
    )
    library = _library(original)
    record = resolve_component_program(library, 0, bookkeeping_factories())
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == tuple(native_signature(original.signature))
    width = dtype.num_qubits
    wires = tuple(reversed(range(width)))
    inputs = (
        ()
        if kind == "Allocate"
        else (tuple((wire,) for wire in wires) if kind == "Join" else (wires,))
    )
    bound = bind_register_program(
        record,
        inputs,
        clean=wires if kind == "Allocate" else (),
        allow_workspace_outputs=kind == "Allocate",
        input_ownership=(
            tuple((wire, False) for wire in wires) if kind == "Free" else ()
        ),
    )
    assert not bound.circuit.instructions
    if kind == "Allocate":
        assert bound.outputs == (wires,)
    elif kind == "Free":
        assert bound.outputs == ()
    elif kind == "Split":
        assert bound.outputs == tuple((wire,) for wire in wires)
    else:
        assert bound.outputs == (wires,)
    resource = import_module("qualtran.resource_counting")
    assert resource.get_cost_value(original, resource.QECGatesCost()).asdict() == {}


@pytest.mark.parametrize("kind", ("Allocate", "Free"))
@pytest.mark.parametrize("dirty", (False, True))
def test_symbolic_lifetimes_remain_logical_and_refuse_physical_binding(
    kind: str, dirty: bool
) -> None:
    source = pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    constructor = getattr(import_module("qualtran.bloqs.bookkeeping"), kind)
    original = constructor(source.QUInt(sympy.Symbol("n")), dirty)
    record = resolve_component_program(_library(original), 0, bookkeeping_factories())
    assert verify_program_record(record) == ()
    operation = record.functions[0].blocks[0].operations[0]
    assert operation.workspace_dirty is dirty
    with pytest.raises(ValueError, match="unsupported executable register descriptor"):
        bind_register_program(record, ())


@pytest.mark.parametrize("shape", ((), (2,), (1, 2), (0, 2)))
def test_serialized_cast_shape_and_absent_signature(shape: tuple[int, ...]) -> None:
    source = pytest.importorskip("qualtran")
    constructor = import_module("qualtran.bloqs.bookkeeping").Cast
    original = constructor(source.QAny(3), source.QUInt(3), shape)
    library = _library(original)
    row = library.entries[0]
    absent = replace(row, declaration=replace(row.declaration, registers=None))
    record = resolve_component_program(
        replace(library, entries=(absent,)), 0, bookkeeping_factories()
    )
    assert record.functions[0].registers == tuple(native_signature(original.signature))
    count = 1
    for extent in shape:
        count *= extent
    words = tuple(tuple(range(3 * i, 3 * i + 3)) for i in range(count))
    bound = bind_register_program(record, words, clean=(0,) if not words else ())
    assert (
        bound.outputs == words and not bound.events and not bound.circuit.instructions
    )


@pytest.mark.parametrize("dirty", (False, True))
def test_complete_serialized_lifetime_view_gate_cycle_restores_complex_workspace(
    dirty: bool,
) -> None:
    source = pytest.importorskip("qualtran")
    gates = import_module("qualtran.bloqs.basic_gates")
    bookkeeping = import_module("qualtran.bloqs.bookkeeping")
    builder, inputs = source.BloqBuilder.from_signature(
        source.Signature.build(spectator=1)
    )
    word = builder.add(
        bookkeeping.Allocate(source.QAny(3), dirty),
    )
    bits = builder.add(bookkeeping.Split(source.QAny(3)), reg=word)
    for index in range(3):
        bits[index] = builder.add(gates.Hadamard(), q=bits[index])
        bits[index] = builder.add(gates.Hadamard(), q=bits[index])
    word = builder.add(bookkeeping.Join(source.QAny(3)), reg=bits)
    word = builder.add(bookkeeping.Cast(source.QAny(3), source.QUInt(3)), reg=word)
    builder.add(bookkeeping.Free(source.QUInt(3), dirty), reg=word)
    composite = builder.finalize(**inputs)
    adapter = import_module("qualtran.serialization.bloq")
    library = deserialize_component_library(
        adapter.bloqs_to_proto(composite).SerializeToString()
    )
    record = resolve_component_program(
        library, 0, fixed_gate_factories() | bookkeeping_factories()
    )
    work = (2, 0, 3)
    bound = bind_register_program(
        record, ((1,),), clean=() if dirty else work, borrowed=work if dirty else ()
    )
    assert bound.outputs == ((1,),)
    assert [(event.kind, event.wires, event.dirty) for event in bound.events] == [
        ("allocate", work, dirty),
        ("free", work, dirty),
    ]
    assert [gate.name for gate in bound.circuit.instructions] == ["h"] * 6
    expected = reference_arithmetic_gates(
        composite,
        {"spectator": (1,)},
        work,
        {},
        borrowed_allocations=work if dirty else None,
    )
    assert_source(bound.circuit.instructions, expected)
    for wire in work:
        assert [
            gate.wires for gate in bound.circuit.instructions if wire in gate.wires
        ] == [(wire,), (wire,)]
    initial = torch.complex(
        torch.arange(1, 17, dtype=torch.float64),
        torch.arange(17, 33, dtype=torch.float64),
    )
    if not dirty:
        initial[:] = 0
        initial[[0, 4]] = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128)
    initial /= initial.norm()
    circuit = Circuit(4, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(4, bound.circuit.instructions)
    )
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    if not dirty:
        actual = circuit.state().flatten().clone()
        actual[0] = actual[4] = 0
        assert actual.abs().max() < 2e-12
    resource = import_module("qualtran.resource_counting")
    model = resource.get_cost_value(composite, resource.QECGatesCost())
    assert model.clifford == 6 and model.measurement == model.total_t_count() == 0


def test_invalid_constructor_arguments_and_existing_cast_storage_boundary() -> None:
    mapping = bookkeeping_factories()
    bit = quantum_bit_type()
    for kind in KINDS:
        factory = mapping[f"qualtran.bloqs.bookkeeping.{kind.lower()}.{kind}"]
        with pytest.raises(SerializationError, match="Unexpected constructor"):
            factory("bad", None, {"unknown": 1})
        with pytest.raises(SerializationError, match="requires a dtype"):
            factory("bad", None, {})
    allocate = mapping["qualtran.bloqs.bookkeeping.allocate.Allocate"]
    for dirty in (2, "yes", 1.0):
        with pytest.raises(SerializationError, match="dirty flag"):
            allocate("bad", None, {"dtype": bit, "dirty": dirty})
    with pytest.raises(SerializationError, match="Invalid register signature"):
        allocate("bad", (), {"dtype": bit})
    cast = mapping["qualtran.bloqs.bookkeeping.cast.Cast"]
    for shape in (-1, (-1,), (1.0,), "2"):
        with pytest.raises(SerializationError, match="Cast shape"):
            cast("bad", None, {"inp_dtype": bit, "out_dtype": bit, "shape": shape})
    with pytest.raises(SerializationError, match="permission flag"):
        cast("bad", None, {"allow_quantum_to_classical": 2})
    with pytest.raises(ValueError, match="explicit opt-in"):
        cast(
            "bad",
            None,
            {
                "inp_dtype": bit,
                "out_dtype": classical_bit_type(),
                "allow_quantum_to_classical": 1,
            },
        )
    assert set(bookkeeping_factories()) == {
        *(f"qualtran.bloqs.bookkeeping.{kind.lower()}.{kind}" for kind in KINDS),
        "qualtran.bloqs.bookkeeping.always.Always",
        "qualtran.bloqs.bookkeeping.arbitrary_clifford.ArbitraryClifford",
    }


@pytest.mark.parametrize("shape", ((), (2,)))
def test_symbolic_cast_uses_logical_signature_and_preserves_physical_refusal(
    shape: tuple[int, ...],
) -> None:
    source = pytest.importorskip("qualtran")
    sympy = pytest.importorskip("sympy")
    constructor = import_module("qualtran.bloqs.bookkeeping").Cast
    original = constructor(
        source.QInt(sympy.Symbol("n")), source.QUInt(sympy.Symbol("n")), shape
    )
    record = resolve_component_program(_library(original), 0, bookkeeping_factories())
    assert verify_program_record(record) == ()
    assert record.functions[0].registers == tuple(native_signature(original.signature))
    assert not record.functions[0].blocks
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, ())
