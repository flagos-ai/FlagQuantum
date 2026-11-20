"""Verify frozen argument descriptors, serializers, and wire bytes."""

import ast
import hashlib
from importlib import import_module
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import sympy
import torch

try:
    import numpy as np
    from numpy.typing import NDArray
except ModuleNotFoundError:
    pytest.skip(
        "NumPy is required for argument reference checks", allow_module_level=True
    )

from examples.algorithms.control_predicate_catalog import PredicateExample
from flagquantum.compiler.formats import legacy_arguments as native
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_registers import (
    deserialize_register,
    serialize_register,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.value_math import SymbolicInt
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_control_predicate_file import _physical_pair

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "protos/args.proto": "f967f4a41c47ba005fe82e8018f08a67e43d1303",
    "protos/args_pb2.py": "65f741b5a144f58b8914479cd609be3be1cc60b2",
    "protos/args_pb2.pyi": "5ea037cbb4d22a74df33f76f14ac1708726e44dc",
    "serialization/__init__.py": "a79cdd57a577d220fc8fa47a94186302dfda6aa0",
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


def test_serialization_namespace_retains_docstring_only_initialization() -> None:
    root = Path(__file__).resolve().parents[2]
    for path in (
        REFERENCE / "serialization/__init__.py",
        root / "flagquantum/compiler/formats/__init__.py",
    ):
        initializers = ast.parse(path.read_text())
        assert ast.get_docstring(initializers)
        assert len(initializers.body) == 1


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
    sympy.Piecewise((1, x < y), (2, True)),
    sympy.Piecewise((1, sympy.Eq(x, y)), (2, True)),
    sympy.Piecewise((1, sympy.Ne(x, y)), (2, True)),
    sympy.Piecewise((1, (x <= 4) & (y > 1)), (2, True)),
    sympy.Piecewise((1, sympy.Eq(x, 0) | (x < y)), (2, True)),
    sympy.Piecewise((1, (x > 0) ^ (y > 0)), (2, True)),
    sympy.Piecewise((sympy.Piecewise((1, x > y), (2, True)), y > z), (3, True)),
    sympy.Mod(x, y),
    sympy.binomial(x, 2),
    sympy.KroneckerDelta(x, y),
    sympy.factorial(x),
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


def test_piecewise_dimensions_and_counts_reach_existing_consumers() -> None:
    source = import_module("qualtran")
    register_codec = import_module("qualtran.serialization.registers")
    arg_codec = import_module("qualtran.serialization.args")
    library_proto = import_module("qualtran.protos.bloq_pb2")
    predicate = import_module("qualtran.bloqs.mcmt.ctrl_spec_and")
    resource = import_module("qualtran.resource_counting")
    dimension = sympy.Piecewise((3, x < 0), (4, True))
    source_register = source.Register("ctrl_0", source.QBit(), shape=(dimension,))
    data = register_codec.register_to_proto(source_register).SerializeToString()
    register = deserialize_register(data)
    assert register.shape_symbolic == (dimension,)
    decoded_dimension = register.shape_symbolic[0]
    assert isinstance(decoded_dimension, sympy.Expr)
    assert serialize_register(register) == data

    original = library_proto.BloqLibrary(name="piecewise count")
    row = original.table.add(bloq_id=0)
    row.bloq.name = "fixture"
    row.bloq_counts[0].CopyFrom(arg_codec.int_or_sympy_to_proto(dimension))
    counts_wire = original.SerializeToString(deterministic=True)
    library = deserialize_component_library(counts_wire)
    assert library.entries[0].counts == ((0, dimension),)
    assert serialize_component_library(library) == counts_wire

    for value, expected_width in ((-1, 3), (0, 4)):
        width = int(decoded_dimension.subs(x, value))
        assert width == expected_width
        activation = tuple(i % 2 for i in range(width))
        specialized = Register(register.name, register.dtype, shape=(width,))
        native_predicate = PredicateExample(
            (specialized,), (ControlRegister(1, activation),)
        )
        source_predicate = predicate.CtrlSpecAnd(source.CtrlSpec(cvs=list(activation)))
        _, forward, erase = _physical_pair(native_predicate, source_predicate)
        work = width - 1
        forward_model = resource.get_cost_value(
            source_predicate, resource.QECGatesCost()
        )
        erase_model = resource.get_cost_value(
            source_predicate.adjoint(), resource.QECGatesCost()
        )
        t_count = sum(g.name in {"t", "tdg"} for g in forward)
        assert t_count == forward_model.total_t_count() == 4 * work
        assert (
            sum(g.name == "measure" for g in erase) == erase_model.measurement == work
        )
        assert resource.get_cost_value(source_predicate, resource.QubitCount()) == (
            width + work
        )
        indices = torch.arange(1 << width, dtype=torch.float64)
        amplitudes = torch.complex(indices + 1, 5 - indices)
        inputs = torch.zeros(1 << (width + work), dtype=torch.complex128)
        inputs[:: 1 << work] = amplitudes / amplitudes.norm()
        computed = forced_branch(inputs, forward, ())
        for history in product((0, 1), repeat=work):
            torch.testing.assert_close(
                forced_branch(computed, erase, history),
                inputs / 2 ** (work / 2),
                atol=2e-12,
                rtol=0,
            )


def test_unsupported_symbolic_code_missing_value_and_overflow() -> None:
    source = import_module("qualtran.protos.args_pb2")
    with pytest.raises(SerializationError, match="selected value"):
        native.deserialize_integer_argument(b"")
    with pytest.raises(SerializationError, match="signed int64"):
        native.serialize_integer_argument(1 << 63)
    for text in (
        "__import__('os')",
        "x.__class__",
        "x[0]",
        "unknown(x)",
        "Eq(x, y)",
        "x < y",
        "Piecewise((1, __import__('os')), (2, True))",
        "Piecewise((x.__class__, x > y), (2, True))",
        "Piecewise((1, x[0]), (2, True))",
        "Piecewise((1, x, y))",
    ):
        with pytest.raises(SerializationError, match="Unsupported"):
            native.deserialize_integer_argument(
                source.IntOrSympy(sympy_expr=text).SerializeToString()
            )
    with pytest.raises(SerializationError, match="exponent"):
        native.deserialize_integer_argument(
            source.IntOrSympy(sympy_expr="2**99999999").SerializeToString()
        )
    for text, match in (
        ("Piecewise((2**99999999, x < 0), (1, True))", "exponent"),
        ("factorial(10**100)", "factorial"),
        ("binomial(2**1000, 10000)", "binomial"),
    ):
        with pytest.raises(SerializationError, match=match):
            native.deserialize_integer_argument(
                source.IntOrSympy(sympy_expr=text).SerializeToString()
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


@pytest.mark.parametrize(
    "text", ("log2(n)", "ceiling(log2(n + 1))", "ceiling(log2(floor(n) + 1))")
)
def test_source_base_two_width_expression_preserves_wire_and_meaning(text: str) -> None:
    from qualtran.protos.args_pb2 import IntOrSympy
    from sympy.codegen.cfunctions import log2

    wire = IntOrSympy(sympy_expr=text).SerializeToString(deterministic=True)
    value = native.deserialize_integer_argument(wire)
    assert native.serialize_integer_argument(value) == wire
    assert isinstance(value, sympy.Expr)
    assert value.has(log2)
    source_decoder = import_module("qualtran.serialization.args")
    original_decoded = source_decoder.int_or_sympy_from_proto(
        IntOrSympy(sympy_expr=text)
    )
    assert str(original_decoded) == str(value)
    assert not original_decoded.has(log2)
    for n in (1, 2, 3, 7, 8, 15, 16, 31):
        numerical = value.subs(sympy.Symbol("n"), n).evalf()
        original = (
            sympy.sympify(text, locals={"log2": log2})
            .subs(sympy.Symbol("n"), n)
            .evalf()
        )
        assert numerical == original


@pytest.mark.parametrize(
    "text",
    (
        "log2()",
        "log2(n, 2)",
        "log2(n, base=2)",
        "log2.__call__(n)",
        "log2(__import__('os'))",
    ),
)
def test_base_two_parser_retains_call_and_code_execution_refusals(text: str) -> None:
    from qualtran.protos.args_pb2 import IntOrSympy

    with pytest.raises(SerializationError):
        native.deserialize_integer_argument(
            IntOrSympy(sympy_expr=text).SerializeToString()
        )
