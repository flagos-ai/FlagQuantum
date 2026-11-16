"""Compare the complete legacy register-type wire family against frozen artifacts."""

import ast
import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from sympy import Symbol

from flagquantum.compiler.formats.legacy_register_types import (
    _register_type_message,
    deserialize_register_type,
    serialize_register_type,
)
from flagquantum.compiler.registers.lookup import qrom_table
from flagquantum.core.ir.program import register_types as types
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.errors import SerializationError

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "protos/data_types.proto": "2848ffb0a07e0e2b161c332c884202c54a4ae846",
    "protos/data_types_pb2.py": "e7e669456dc1abe8727d84553a59bfc04b22f946",
    "protos/data_types_pb2.pyi": "ec9125233e1f257e1efa307b03e055aae3ad9e42",
    "serialization/data_types.py": "9f7aebe00f0d69d3b447e08309627f8978344b17",
    "serialization/data_types_test.py": "ff88e4e11b781bbcedf773535c9c66f750c843d3",
}


def _fields(descriptor: Any) -> list[tuple[str, int, int, str | None]]:
    return [
        (
            field.name,
            field.number,
            field.type,
            field.containing_oneof.name if field.containing_oneof is not None else None,
        )
        for field in descriptor.fields
    ]


def test_all_frozen_descriptors_and_stub_constructors() -> None:
    for name, digest in BLOBS.items():
        data = (REFERENCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    source = import_module("qualtran.protos.data_types_pb2").QDataType.DESCRIPTOR
    native = _register_type_message().DESCRIPTOR
    schema = (REFERENCE / "protos/data_types.proto").read_text()
    assert 'import "qualtran/protos/args.proto";' in schema
    assert _fields(source) == _fields(native)
    assert all(field.containing_oneof.name == "val" for field in source.fields)
    assert len(source.fields) == 8
    stubs = {
        node.name: node
        for node in ast.parse(
            (REFERENCE / "protos/data_types_pb2.pyi").read_text()
        ).body
        if isinstance(node, ast.ClassDef)
    }
    for expected, actual in zip(source.fields, native.fields, strict=True):
        assert (
            f"{expected.message_type.name} {expected.name} = {expected.number};"
            in schema
        )
        assert _fields(expected.message_type) == _fields(actual.message_type)
        for field in actual.message_type.fields:
            assert field.has_presence
            assert field.containing_oneof.name == f"_{field.name}"
            if field.type == field.TYPE_MESSAGE:
                assert [
                    f.name for f in field.message_type.oneofs_by_name["val"].fields
                ] == ["int_val", "sympy_expr"]
    for descriptor in [source, *(field.message_type for field in source.fields)]:
        constructor = next(
            node
            for node in stubs[descriptor.name].body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        assert [arg.arg for arg in constructor.args.kwonlyargs] == [
            field.name for field in descriptor.fields
        ]


a, b = Symbol("a"), Symbol("b")
CASES = [
    ("QBit", (), types.quantum_bit_type()),
    *(
        (source_name, (width,), factory(width))
        for source_name, factory in (
            ("QAny", types.opaque_register_type),
            ("QInt", types.signed_register_type),
            ("QIntOnesComp", types.ones_complement_register_type),
            ("QUInt", types.unsigned_register_type),
            ("QMontgomeryUInt", types.montgomery_register_type),
        )
        for width in (10, 1000, a * b)
    ),
    *(
        ("BQUInt", (width, bound), types.bounded_unsigned_register_type(width, bound))
        for width, bound in ((10, 1), (5, 10), (1000, 1000), (a * b, 10), (a, b))
    ),
    *(
        (
            "QFxp",
            (width, frac, signed),
            types.fixed_point_register_type(width, frac, signed),
        )
        for width, frac, signed in (
            (10, 5, True),
            (10, 5, False),
            (5, 5, False),
            (a, a, False),
        )
    ),
]


@pytest.mark.parametrize("name,args,dtype", CASES)
def test_all_original_values_and_native_exact_bytes(
    name: str, args: tuple[object, ...], dtype: RegisterType
) -> None:
    native_source = import_module("qualtran.dtype")
    adapter = import_module("qualtran.serialization.data_types")
    expected = getattr(native_source, name)(*args)
    original = adapter.data_type_to_proto(expected)
    data = serialize_register_type(dtype)
    assert data == original.SerializeToString(deterministic=True)
    assert deserialize_register_type(data) == dtype
    assert adapter.data_type_from_proto(type(original).FromString(data)) == expected


def test_montgomery_modulus_loss_and_unsupported_types_are_explicit() -> None:
    source_types = import_module("qualtran.dtype")
    adapter = import_module("qualtran.serialization.data_types")
    loses_modulus = adapter.data_type_to_proto(source_types.QMontgomeryUInt(3, 7))
    assert adapter.data_type_from_proto(loses_modulus).modulus is None
    assert deserialize_register_type(
        loses_modulus.SerializeToString()
    ) == types.montgomery_register_type(3)
    with pytest.raises(SerializationError, match="Montgomery modulus"):
        serialize_register_type(types.montgomery_register_type(3, 7))
    with pytest.raises(SerializationError, match="no legacy quantum"):
        serialize_register_type(types.classical_unsigned_register_type(0))
    with pytest.raises(SerializationError, match="width one"):
        serialize_register_type(RegisterType("bit", 3))


def test_absent_and_explicit_false_preserve_default_or_presence() -> None:
    message = _register_type_message()()
    message.qfxp.bitsize.int_val = 4
    message.qfxp.num_frac.int_val = 2
    assert not message.qfxp.HasField("signed")
    dtype = deserialize_register_type(message.SerializeToString())
    assert dtype == types.fixed_point_register_type(4, 2, False)
    encoded = type(message).FromString(serialize_register_type(dtype))
    assert encoded.qfxp.HasField("signed") and encoded.qfxp.signed is False
    message.qfxp.ClearField("num_frac")
    with pytest.raises(SerializationError, match="qfxp"):
        deserialize_register_type(message.SerializeToString())
    with pytest.raises(SerializationError, match="no selected"):
        deserialize_register_type(b"")


def test_decoded_types_drive_same_qrom_signature_and_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = qrom_table.QromTable.from_data([0, 1, 2, 3], num_controls=1)
    expected = baseline.bind_packed(
        tuple(range(5)),
        tuple(range(5, 5 + baseline.workspace_size)),
        tuple(range(baseline.workspace_size)),
    )

    def bounded(width: int, bound: int) -> RegisterType:
        return deserialize_register_type(
            serialize_register_type(types.bounded_unsigned_register_type(width, bound))
        )

    def opaque(width: int) -> RegisterType:
        return deserialize_register_type(
            serialize_register_type(types.opaque_register_type(width))
        )

    monkeypatch.setattr(qrom_table, "opaque_register_type", opaque)
    monkeypatch.setattr(qrom_table, "bounded_unsigned_register_type", bounded)
    table = qrom_table.QromTable.from_data([0, 1, 2, 3], num_controls=1)
    assert table.signature == baseline.signature
    assert (
        table.bind_packed(
            tuple(range(5)),
            tuple(range(5, 5 + table.workspace_size)),
            tuple(range(table.workspace_size)),
        )
        == expected
    )
