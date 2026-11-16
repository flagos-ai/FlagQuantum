"""Compare frozen register declarations and their native append consumer."""

import ast
import hashlib
from collections import Counter
from importlib import import_module
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from sympy import Symbol

from examples.algorithms.control_predicate_catalog import PredicateExample
from flagquantum.compiler.formats.legacy_registers import (
    _register_message,
    _registers_message,
    deserialize_register,
    deserialize_registers,
    serialize_register,
    serialize_registers,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_control_predicate_file import _physical_pair

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "protos/registers.proto": "2881112dcbf85f0e30e2a1980f64a9eb45fd7101",
    "protos/registers_pb2.py": "b37ff02eee02e07ef42274ead0e3a5b0a42ebd42",
    "protos/registers_pb2.pyi": "e51a9bf0969b693e6c9495dde8c7a4c7df10e03d",
    "serialization/registers.py": "8f449a4877476b781d5bc081bf79fab5eb3a9ced",
    "serialization/registers_test.py": "9517380dea8da66086aa757d2f695c10f1038fd4",
}


def test_frozen_fields_nested_side_enum_and_constructor_stubs() -> None:
    for name, digest in BLOBS.items():
        data = (REFERENCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    source = import_module("qualtran.protos.registers_pb2")
    schema = (REFERENCE / "protos/registers.proto").read_text()
    assert "repeated IntOrSympy shape = 3;" in schema
    assert "repeated Register registers = 1;" in schema
    stub = ast.parse((REFERENCE / "protos/registers_pb2.pyi").read_text())
    for original, native in (
        (source.Register.DESCRIPTOR, _register_message().DESCRIPTOR),
        (source.Registers.DESCRIPTOR, _registers_message().DESCRIPTOR),
    ):
        assert [(f.name, f.number, f.label, f.type) for f in original.fields] == [
            (f.name, f.number, f.label, f.type) for f in native.fields
        ]
        cls = next(
            node
            for node in stub.body
            if isinstance(node, ast.ClassDef) and node.name == original.name
        )
        ctor = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        assert {arg.arg for arg in ctor.args.kwonlyargs} == {
            field.name for field in native.fields
        }
    for cls in (source.Register, _register_message()):
        assert [
            (v.name, v.number) for v in cls.DESCRIPTOR.enum_types_by_name["Side"].values
        ] == [
            ("UNKNOWN", 0),
            ("LEFT", 1),
            ("RIGHT", 2),
            ("THRU", 3),
        ]


a, b, c, d = (Symbol(name) for name in "abcd")
CASES = (
    (1, (2, 3), RegisterSide.RIGHT),
    (10, (), RegisterSide.LEFT),
    (1000, (10, 1, 20), RegisterSide.THRU),
    (a * b, (c + d,), RegisterSide.THRU),
)


@pytest.mark.parametrize("width,shape,side", CASES)
def test_single_and_ordered_collection_match_frozen_bytes(
    width: SymbolicInt,
    shape: tuple[SymbolicInt, ...],
    side: RegisterSide,
) -> None:
    source = import_module("qualtran")
    adapter = import_module("qualtran.serialization.registers")
    names = ("my_reg", "my_reg2", "my_reg3")
    native = tuple(
        Register(name, opaque_register_type(width), shape, side) for name in names
    )
    expected = tuple(
        source.Register(
            name, source.QAny(width), shape=shape, side=source.Side(side.value)
        )
        for name in names
    )
    for register, original in zip(native, expected, strict=True):
        data = serialize_register(register)
        proto = adapter.register_to_proto(original)
        assert data == proto.SerializeToString(deterministic=True)
        assert deserialize_register(data) == register
        assert adapter.register_from_proto(type(proto).FromString(data)) == original
    data = serialize_registers(iter(native))
    assert data == adapter.registers_to_proto(expected).SerializeToString(
        deterministic=True
    )
    assert deserialize_registers(data) == native


def test_collection_preserves_declarations_and_signature_owns_duplicate_policy() -> (
    None
):
    bit = quantum_bit_type()
    left = Register("same", bit, (2,), RegisterSide.LEFT)
    right = Register("same", bit, (3,), RegisterSide.RIGHT)
    assert deserialize_registers(b"") == ()
    assert serialize_registers(()) == b""
    result = deserialize_registers(serialize_registers((right, left)))
    assert tuple(Signature(result)) == (right, left)
    duplicated = deserialize_registers(serialize_registers((left, left)))
    assert duplicated == (left, left)
    with pytest.raises(ValueError, match="more than once"):
        Signature(duplicated)


def test_invalid_side_is_explicit_and_unknown_enum_is_rejected() -> None:
    source = import_module("qualtran")
    adapter = import_module("qualtran.serialization.registers")
    malformed = source.Register("r", source.QBit(), side=source.Side(0))
    message = adapter.register_to_proto(malformed)
    assert message.side == 0
    with pytest.raises(ValueError, match="Unknown Side"):
        adapter.register_from_proto(message)
    with pytest.raises(SerializationError, match="Unknown register side"):
        serialize_register(Register("r", quantum_bit_type(), side=RegisterSide(0)))
    for value in (0, 99):
        message.side = value
        with pytest.raises(SerializationError, match=f"Unknown register side {value}"):
            deserialize_register(message.SerializeToString())


def test_decoded_registers_feed_same_predicate_append_and_cleanup() -> None:
    source = import_module("qualtran")
    adapter = import_module("qualtran.serialization.registers")
    mcmt = import_module("qualtran.bloqs.mcmt.ctrl_spec_and")
    original = mcmt.CtrlSpecAnd(source.CtrlSpec(cvs=np.array([[0, 1], [1, 0]])))
    data = adapter.registers_to_proto(original.control_registers).SerializeToString()
    registers = deserialize_registers(data)
    assert tuple(register.shape for register in registers) == ((2, 2),)
    native = PredicateExample(registers, (ControlRegister(1, (0, 1, 1, 0)),))
    n, forward, erase = _physical_pair(native, original)
    assert n == 4
    assert Counter(gate.name for gate in forward) == {
        "cx": 18,
        "h": 6,
        "t": 6,
        "tdg": 6,
        "x": 4,
        "s": 3,
    }
    assert Counter(gate.name for gate in erase) == {
        "x": 4,
        "h": 3,
        "measure": 3,
        "cz": 3,
        "reset": 3,
    }
    resource = import_module("qualtran.resource_counting")
    forward_cost = resource.get_cost_value(original, resource.QECGatesCost())
    erase_cost = resource.get_cost_value(original.adjoint(), resource.QECGatesCost())
    assert forward_cost.total_t_count() == 12
    assert forward_cost.and_bloq == 3 and forward_cost.clifford == 4
    assert erase_cost.measurement == 3 and erase_cost.clifford == 7
    assert resource.get_cost_value(original, resource.QubitCount()) == 7

    indices = torch.arange(16, dtype=torch.float64)
    amplitudes = torch.complex(indices + 1, 5 - indices)
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::8] = amplitudes / amplitudes.norm()
    computed = forced_branch(initial, forward, ())
    expected = torch.zeros_like(initial)
    for address in range(16):
        # The successive prefixes 01, 011, and 0110 are stored on the work bits.
        prefix = 4 * (address >> 2 == 1) + 2 * (address >> 1 == 3) + (address == 6)
        expected[(address << 3) | prefix] = initial[address << 3]
    torch.testing.assert_close(computed, expected, rtol=0, atol=2e-12)
    for branch in product((0, 1), repeat=3):
        torch.testing.assert_close(
            forced_branch(computed, erase, branch), initial / 2**1.5, rtol=0, atol=2e-12
        )
