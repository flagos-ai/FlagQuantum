"""Check all frozen annotation source artifacts against native byte interchange."""

import ast
import hashlib
import re
from importlib import import_module
from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.compiler.formats import resource_annotations
from flagquantum.errors import SerializationError

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "protos/__init__.py": "ce76a8d10a5909daeb7b1113c8cc7f26cc31b410",
    "protos/annotations.proto": "3fd709194fe0340b819eb0705883e30236643952",
    "protos/annotations_pb2.py": "ac809603130c9ee4ca3a99a2cb034caff8bf0c16",
    "protos/annotations_pb2.pyi": "a8dbed196240727f6f59db30df282fd9f7e1389f",
    "serialization/annotations.py": "a4ebff629a3e10e60a2025a3b91e1bd18c09bf47",
    "serialization/annotations_test.py": "bca834902649f4fe1bb871affb422a6eb92f0b12",
}


def test_schema_generated_module_and_typed_stub_cover_identical_fields() -> None:
    for name, digest in BLOBS.items():
        data = (REFERENCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    assert not ast.parse((REFERENCE / "protos/__init__.py").read_text()).body
    schema = (REFERENCE / "protos/annotations.proto").read_text()
    assert 'syntax = "proto3";' in schema
    names = re.findall(r"\bint64\s+(\w+)\s*=\s*(\d+)\s*;", schema)
    generated = import_module("qualtran.protos.annotations_pb2").TComplexity
    native = resource_annotations._annotation_factory()
    for cls in (generated, native):
        assert [
            (field.name, str(field.number)) for field in cls.DESCRIPTOR.fields
        ] == names
        assert all(
            field.type == field.TYPE_INT64 and not field.has_presence
            for field in cls.DESCRIPTOR.fields
        )
    stub = ast.parse((REFERENCE / "protos/annotations_pb2.pyi").read_text())
    message = next(node for node in stub.body if isinstance(node, ast.ClassDef))
    ctor = next(
        node
        for node in message.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert [parameter.arg for parameter in ctor.args.kwonlyargs] == [
        name for name, _ in names
    ]
    assert {
        attribute.target.id
        for attribute in message.body
        if isinstance(attribute, ast.AnnAssign)
        and isinstance(attribute.target, ast.Name)
        and isinstance(attribute.annotation, ast.Attribute)
        and attribute.annotation.attr == "int"
        and attribute.target.id.islower()
    } == {name for name, _ in names}


@pytest.mark.parametrize(
    "clifford,rotations,t",
    [
        (0, 0, 0),
        (100, 1000, 10),
        (7, 2, 9),
        (-(1 << 63), -1, (1 << 63) - 1),
        (0, 0, 1 << 40),
    ],
)
def test_native_and_original_annotations_roundtrip_exact_bytes(
    clifford: int, rotations: int, t: int
) -> None:
    protocol = import_module("qualtran.cirq_interop.t_complexity_protocol")
    adapter = import_module("qualtran.serialization.annotations")
    original = protocol.TComplexity(clifford=clifford, rotations=rotations, t=t)
    source = adapter.t_complexity_to_proto(original)
    native = GateCost(clifford=clifford, rotation=rotations, t=t)
    data = resource_annotations.serialize_resource_annotation(native)
    assert data == source.SerializeToString(deterministic=True)
    assert resource_annotations.deserialize_resource_annotation(data) == native
    assert adapter.t_complexity_from_proto(type(source).FromString(data)) == original


def test_legacy_projection_uses_existing_resource_model() -> None:
    source = import_module("qualtran.resource_counting")
    annotations = import_module("qualtran.serialization.annotations")
    counts = GateCost(
        t=1, toffoli=2, cswap=3, and_bloq=4, clifford=5, rotation=6, measurement=7
    )
    budget = legacy_gate_budget(counts)
    projected = GateCost(
        t=int(budget["t"]),
        clifford=int(budget["clifford"]),
        rotation=int(budget["rotations"]),
    )
    expected = source.GateCounts(
        t=1, toffoli=2, cswap=3, and_bloq=4, clifford=5, rotation=6, measurement=7
    )
    assert resource_annotations.serialize_resource_annotation(
        projected
    ) == annotations.t_complexity_to_proto(
        expected.to_legacy_t_complexity()
    ).SerializeToString(
        deterministic=True
    )
    with pytest.raises(SerializationError, match="reduced"):
        resource_annotations.serialize_resource_annotation(counts)


def test_wire_parser_and_export_failures() -> None:
    # Unknown optional fields are skipped on conversion to native values.
    assert resource_annotations.deserialize_resource_annotation(
        bytes.fromhex("08072001")
    ) == GateCost(clifford=7)
    with pytest.raises(SerializationError, match="payload"):
        resource_annotations.deserialize_resource_annotation(b"\x08\x80")
    with pytest.raises(SerializationError, match="signed int64"):
        resource_annotations.serialize_resource_annotation(GateCost(t=1 << 63))
