"""Verify frozen nested point descriptors and the native arithmetic consumer."""

import ast
import hashlib
from importlib import import_module
from pathlib import Path

import pytest
from sympy import Symbol

from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint
from flagquantum.algorithms.primitives.elliptic_curve_window_addition import (
    EllipticCurveWindowAddition,
)
from flagquantum.compiler.formats.legacy_curve_points import (
    _curve_message,
    deserialize_curve_point,
    serialize_curve_point,
)
from flagquantum.errors import SerializationError
from tests.unit.test_elliptic_curve_window_addition import layout

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "protos/ec_point.proto": "c57fa1b53dec69f9617f13e0670103ba76eb14a9",
    "protos/ec_point_pb2.py": "f28f2c1105c8d310fc49e3252794dfcc29187b4a",
    "protos/ec_point_pb2.pyi": "ac82158cd6d0d49842a7f49655fb88d98cdb5eac",
    "serialization/ec_point.py": "fb690758868bed239c614610e2d3c73bb582983a",
    "serialization/ec_point_test.py": "14dc066a1789f9881ffef652c2f6b266dd232f31",
}
FIELDS = ("x", "y", "mod", "curve_a")


def test_frozen_nested_point_schema_generated_descriptor_and_stub() -> None:
    for name, digest in BLOBS.items():
        data = (REFERENCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    source_module = import_module("qualtran.protos.ec_point_pb2")
    schema = (REFERENCE / "protos/ec_point.proto").read_text()
    assert 'import "qualtran/protos/args.proto";' in schema
    for number, field in enumerate(FIELDS, 1):
        assert (
            f"{'optional ' if field == 'curve_a' else ''}IntOrSympy {field} = {number};"
            in schema
        )
    for descriptor in (source_module.ECPoint.DESCRIPTOR, _curve_message().DESCRIPTOR):
        assert [field.name for field in descriptor.fields] == list(FIELDS)
        assert [field.number for field in descriptor.fields] == [1, 2, 3, 4]
        assert [
            field.name for field in descriptor.oneofs_by_name["_curve_a"].fields
        ] == ["curve_a"]
        for field in descriptor.fields:
            assert field.type == field.TYPE_MESSAGE
            assert field.has_presence
            assert [
                nested.name
                for nested in field.message_type.oneofs_by_name["val"].fields
            ] == [
                "int_val",
                "sympy_expr",
            ]
    stub = ast.parse((REFERENCE / "protos/ec_point_pb2.pyi").read_text())
    point = next(node for node in stub.body if isinstance(node, ast.ClassDef))
    constructor = next(
        node
        for node in point.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert tuple(param.arg for param in constructor.args.kwonlyargs) == FIELDS
    for item in point.body:
        if isinstance(item, ast.FunctionDef) and item.name in FIELDS:
            assert isinstance(item.returns, ast.Attribute)
            assert item.returns.attr == "IntOrSympy"


u = Symbol("u")


@pytest.mark.parametrize(
    "point",
    [
        ECPoint(15, 13, 17, 0),
        ECPoint(0, 2, 7, 3),
        ECPoint(0, 2, 7),
        ECPoint.inf(17, 0),
        ECPoint(u + 1, 2 * u, 17, 3),
        ECPoint(u, u + 1, 17, 0) * 3,
    ],
)
def test_exact_nested_wire_and_source_roundtrip(point: ECPoint) -> None:
    source = import_module("qualtran.serialization.ec_point")
    source_type = import_module("qualtran.bloqs.cryptography.ecc").ECPoint
    expected = source_type(point.x, point.y, point.mod, point.curve_a)
    fixture = source.ec_point_to_proto(expected)
    data = serialize_curve_point(point)
    assert data == fixture.SerializeToString(deterministic=True)
    assert deserialize_curve_point(data, ECPoint) == point
    assert source.ec_point_from_proto(type(fixture).FromString(data)) == expected


def test_missing_coefficient_uses_native_default_and_empty_present_field_fails() -> (
    None
):
    source = import_module("qualtran.serialization.ec_point")
    proto = _curve_message()()
    proto.x.int_val = 0
    proto.y.int_val = 0
    proto.mod.int_val = 17
    assert not proto.HasField("curve_a")
    missing = proto.SerializeToString()
    reference_message = import_module("qualtran.protos.ec_point_pb2").ECPoint
    with pytest.raises(SyntaxError):
        source.ec_point_from_proto(reference_message.FromString(missing))
    assert deserialize_curve_point(missing, ECPoint) == ECPoint.inf(17, 0)

    proto.curve_a.SetInParent()
    with pytest.raises(SerializationError, match="curve_a"):
        deserialize_curve_point(proto.SerializeToString(), ECPoint)
    proto.ClearField("x")
    with pytest.raises(SerializationError, match="field x"):
        deserialize_curve_point(proto.SerializeToString(), ECPoint)


def test_decode_point_feeds_the_same_windowed_append_operands() -> None:
    point = ECPoint(0, 1, 3, 1)
    decoded = deserialize_curve_point(serialize_curve_point(point), ECPoint)
    plain = EllipticCurveWindowAddition(3, point, 1)
    rebound = EllipticCurveWindowAddition(3, decoded, 1)
    assert rebound.table == plain.table
    assert rebound.workspace_requirements() == plain.workspace_requirements()
    assert layout(rebound) == layout(plain)
