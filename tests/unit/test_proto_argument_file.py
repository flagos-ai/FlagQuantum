"""Verify frozen argument descriptors, serializers, and wire bytes."""

import ast
import hashlib
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sympy
from numpy.typing import NDArray

from flagquantum.compiler.formats import legacy_arguments as native
from flagquantum.core.value_math import SymbolicInt
from flagquantum.errors import SerializationError

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "protos/args.proto": "f967f4a41c47ba005fe82e8018f08a67e43d1303",
    "protos/args_pb2.py": "65f741b5a144f58b8914479cd609be3be1cc60b2",
    "protos/args_pb2.pyi": "5ea037cbb4d22a74df33f76f14ac1708726e44dc",
    "serialization/args.py": "68c2b3136d36287d07c2814ed520eaecd98d2425",
    "serialization/args_test.py": "d099ec8fde0a01f3443218c362556bd4f8a85b15",
}


def test_frozen_schema_generated_descriptors_and_stub() -> None:
    for name, digest in BLOBS.items():
        data = (REFERENCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    schema = (REFERENCE / "protos/args.proto").read_text()
    assert 'syntax = "proto3";' in schema
    assert "oneof val {" in schema
    source = import_module("qualtran.protos.args_pb2")
    stubs = {
        node.name: node
        for node in ast.parse((REFERENCE / "protos/args_pb2.pyi").read_text()).body
        if isinstance(node, ast.ClassDef)
    }
    assert tuple(stubs) == ("IntOrSympy", "NDArray", "Complex")

    for name, factory, names, types in (
        (
            "IntOrSympy",
            native._integer_message,
            ("int_val", "sympy_expr"),
            ("INT64", "STRING"),
        ),
        ("NDArray", native._array_message, ("ndarray",), ("BYTES",)),
        ("Complex", native._complex_message, ("real", "imag"), ("FLOAT", "FLOAT")),
    ):
        expected = [
            (field, tag, tp)
            for tag, (field, tp) in enumerate(zip(names, types, strict=True), 1)
        ]
        for descriptor in (getattr(source, name).DESCRIPTOR, factory().DESCRIPTOR):
            assert [
                (field.name, field.number, types[index])
                for index, field in enumerate(descriptor.fields)
                if field.type == getattr(field, f"TYPE_{types[index]}")
            ] == expected
            assert [oneof.name for oneof in descriptor.oneofs] == (
                ["val"] if name == "IntOrSympy" else []
            )
        constructor = next(
            item
            for item in stubs[name].body
            if isinstance(item, ast.FunctionDef) and item.name == "__init__"
        )
        assert tuple(arg.arg for arg in constructor.args.kwonlyargs) == names
        for field, tag, tp in expected:
            assert f"{tp.lower()} {field} = {tag};" in schema
    source_adapter = ast.parse((REFERENCE / "serialization/args.py").read_text())
    source_complex = next(
        n
        for n in source_adapter.body
        if isinstance(n, ast.FunctionDef) and n.name == "complex_from_proto"
    )
    annotation = source_complex.args.args[0].annotation
    assert isinstance(annotation, ast.Attribute) and annotation.attr == "IntOrSympy"


x, y, z = sympy.symbols("x y z")
INTEGERS = (
    0,
    1,
    -(1 << 63),
    (1 << 63) - 1,
    sympy.Integer(0),
    x * y + z / 10,
    sympy.floor(x / 2) + sympy.sqrt(y),
    sympy.log(x) + sympy.sin(y),
    sympy.Float("0.123456789012345678901", 25) * x,
)


@pytest.mark.parametrize("value", INTEGERS)
def test_integer_oneof_and_safe_expression_bytes(value: SymbolicInt) -> None:
    source = import_module("qualtran.serialization.args")
    proto = source.int_or_sympy_to_proto(value)
    data = native.serialize_integer_argument(value)
    assert data == proto.SerializeToString(deterministic=True)
    original = source.int_or_sympy_from_proto(proto)
    decoded = native.deserialize_integer_argument(data)
    assert decoded == original
    assert type(decoded) is type(original)


def test_unsupported_symbolic_code_missing_value_and_overflow() -> None:
    source = import_module("qualtran.protos.args_pb2")
    with pytest.raises(SerializationError, match="selected value"):
        native.deserialize_integer_argument(b"")
    with pytest.raises(SerializationError, match="signed int64"):
        native.serialize_integer_argument(1 << 63)
    for text in ("__import__('os')", "x.__class__", "x[0]", "unknown(x)"):
        with pytest.raises(SerializationError, match="Unsupported"):
            native.deserialize_integer_argument(
                source.IntOrSympy(sympy_expr=text).SerializeToString()
            )
    with pytest.raises(SerializationError, match="exponent"):
        native.deserialize_integer_argument(
            source.IntOrSympy(sympy_expr="2**99999999").SerializeToString()
        )


@pytest.mark.parametrize(
    "array",
    [
        np.arange(100, dtype=np.float64),
        np.array(5, dtype=">i4"),
        np.empty((0, 3), dtype=np.complex64),
        np.asfortranarray(np.arange(12).reshape(3, 4)),
        np.array([(3, 2j)], dtype=[("label", ">i4"), ("value", "<c8")]),
    ],
)
def test_arrays_match_exact_source_npy_bytes(array: NDArray[Any]) -> None:
    source = import_module("qualtran.serialization.args")
    proto = source.ndarray_to_proto(array)
    data = native.serialize_array_argument(array)
    assert data == proto.SerializeToString(deterministic=True)
    decoded = native.deserialize_array_argument(data)
    np.testing.assert_array_equal(decoded, source.ndarray_from_proto(proto))
    assert decoded.dtype == array.dtype
    assert decoded.flags.f_contiguous == array.flags.f_contiguous


def test_object_arrays_and_malformed_containers_are_refused() -> None:
    source = import_module("qualtran.protos.args_pb2")
    array = np.array([object()], dtype=object)
    with pytest.raises(SerializationError, match="Invalid array argument"):
        native.serialize_array_argument(array)
    with BytesIO() as stream:
        np.save(stream, array, allow_pickle=True)
        object_payload = stream.getvalue()
    with BytesIO() as stream:
        np.savez(stream, value=[1, 2])
        zip_payload = stream.getvalue()
    for payload in (object_payload, zip_payload, b"not an array"):
        with pytest.raises(SerializationError, match="NumPy array"):
            native.deserialize_array_argument(
                source.NDArray(ndarray=payload).SerializeToString()
            )


@pytest.mark.parametrize(
    "value", [1j, complex(np.complex64(1j)), complex(-0.0, 0.0), 1 / 3 + 1j / 7, 3e38j]
)
def test_complex_has_source_float32_precision_and_bytes(value: complex) -> None:
    source = import_module("qualtran.serialization.args")
    proto = source.complex_to_proto(value)
    data = native.serialize_complex_argument(value)
    assert data == proto.SerializeToString(deterministic=True)
    assert native.deserialize_complex_argument(data) == source.complex_from_proto(proto)
