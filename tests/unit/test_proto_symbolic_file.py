"""Verify the frozen recursive symbolic wire format and native constructors."""

import ast
import hashlib
from importlib import import_module
from pathlib import Path

import pytest
import sympy

from flagquantum.compiler.formats.legacy_symbolic_terms import (
    _term_message,
    deserialize_symbolic_term,
    serialize_symbolic_term,
)
from flagquantum.core.value_math import SymbolicReal
from flagquantum.errors import SerializationError

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "protos/sympy.proto": "0afe51aab0f228faea86be740396e69f20b9c3c1",
    "protos/sympy_pb2.py": "c29c7cbcad61b6d1fe92074e3d0fe17864e4ec97",
    "protos/sympy_pb2.pyi": "868aae3c3f43224a4439b7b486bec8d6227e1528",
    "serialization/sympy_to_proto.py": "eefe901f0428c8c34e0f856143585cf2af0b1710",
    "serialization/sympy_to_proto_test.py": "d1853abe918ea0cd8070a467337c0ed284deffef",
}


def test_generated_descriptor_matches_every_recursive_field_and_enum() -> None:
    for name, digest in BLOBS.items():
        data = (REFERENCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    source = import_module("qualtran.protos.sympy_pb2")
    descriptor_types = import_module("google.protobuf.descriptor_pb2")
    expected, actual = (descriptor_types.FileDescriptorProto() for _ in range(2))
    source.DESCRIPTOR.CopyToProto(expected)
    _term_message().DESCRIPTOR.file.CopyToProto(actual)
    for proto in (expected, actual):
        proto.name = ""
        proto.package = ""
        for message in proto.message_type:
            for field in message.field:
                if field.type_name:
                    field.type_name = field.type_name.rsplit(".", 1)[-1]
    assert actual == expected
    stubs = ast.parse((REFERENCE / "protos/sympy_pb2.pyi").read_text())
    for message in expected.message_type:
        cls = next(
            node
            for node in stubs.body
            if isinstance(node, ast.ClassDef) and node.name == message.name
        )
        constructor = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        assert {arg.arg for arg in constructor.args.kwonlyargs} == {
            field.name for field in message.field
        }


def test_all_original_fixtures_and_original_operand_types() -> None:
    source = import_module("qualtran.serialization.sympy_to_proto")
    fixtures = import_module("qualtran.serialization.sympy_to_proto_test")
    values = (
        *fixtures.sympy_parameters_to_test,
        *fixtures.sympy_exprs_to_test,
        sympy.Float("0.7"),
    )
    assert len(values) == 21
    for value in values:
        message = source.sympy_expr_to_proto(value)
        expected = source.sympy_expr_from_proto(message)
        wire = serialize_symbolic_term(value)
        assert wire == message.SerializeToString(deterministic=True)
        actual = deserialize_symbolic_term(wire)
        assert actual == expected and type(actual) is type(expected)


a, b = sympy.symbols("a b")


@pytest.mark.parametrize(
    "value",
    [
        -(1 << 31),
        (1 << 31) - 1,
        1 / 3,
        sympy.Mod(a, b),
        sympy.log(a),
        sympy.floor(a),
        sympy.Min(a, b),
    ],
)
def test_additional_supported_values_match_frozen_bytes(value: SymbolicReal) -> None:
    source = import_module("qualtran.serialization.sympy_to_proto")
    message = source.sympy_expr_to_proto(value)
    data = serialize_symbolic_term(value)
    assert data == message.SerializeToString(deterministic=True)
    assert deserialize_symbolic_term(data) == source.sympy_expr_from_proto(message)


def test_literal_symbol_preserves_identity_and_assumption_limit() -> None:
    source = import_module("qualtran.serialization.sympy_to_proto")
    spaced = sympy.Symbol("x y")
    assert source.sympy_expr_from_proto(source.sympy_expr_to_proto(spaced)) == (
        sympy.Symbol("x"),
        sympy.Symbol("y"),
    )
    assert deserialize_symbolic_term(serialize_symbolic_term(spaced)) == spaced
    constrained = sympy.Symbol("x", positive=True)
    decoded = deserialize_symbolic_term(serialize_symbolic_term(constrained))
    assert decoded == sympy.Symbol("x")
    assert decoded != constrained


def test_malformed_terms_and_unsupported_values() -> None:
    source = import_module("qualtran.protos.sympy_pb2")
    for message in (
        source.Term(function=99),
        source.Term(),
        source.Term(operands=[source.Operand(parameter=source.Parameter())]),
        source.Term(
            operands=[source.Operand(parameter=source.Parameter(const_symbol=99))]
        ),
    ):
        with pytest.raises(SerializationError, match="(?i)legacy symbolic"):
            deserialize_symbolic_term(message.SerializeToString())
    data = source.Term(
        function=source.Pow,
        operands=[
            source.Operand(parameter=source.Parameter(const_int=2)),
            source.Operand(parameter=source.Parameter(const_int=99999999)),
        ],
    ).SerializeToString()
    with pytest.raises(SerializationError, match="exponent"):
        deserialize_symbolic_term(data)
    invalid_arity = source.Term.FromString(data)
    invalid_arity.operands.add().parameter.const_int = 1
    with pytest.raises(SerializationError, match="arity"):
        deserialize_symbolic_term(invalid_arity.SerializeToString())
    with pytest.raises(SerializationError, match="Invalid legacy symbolic value"):
        serialize_symbolic_term(1 << 31)
    with pytest.raises(SerializationError, match="Unsupported"):
        serialize_symbolic_term(sympy.Piecewise((1, a > 0), (0, True)))
