"""Verify the frozen control wire family and decoded predicate execution."""

import ast
import hashlib
from collections import Counter
from importlib import import_module
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from sympy import Symbol

from examples.algorithms.control_predicate_catalog import PredicateExample
from flagquantum.compiler.formats.legacy_control_specs import (
    ControlClauses,
    _control_spec_message,
    control_registers_from_spec,
    deserialize_control_spec,
    serialize_control_spec,
)
from flagquantum.core.ir.program import register_types as types
from flagquantum.core.ir.program.registers import Register
from flagquantum.errors import SerializationError
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_control_predicate_file import _physical_pair

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran"
BLOBS = {
    "protos/ctrl_spec.proto": "38415ce602abd14ee5032f1c3047101f13116690",
    "protos/ctrl_spec_pb2.py": "6e55523682744103e3710f850fb30fac727ae59c",
    "protos/ctrl_spec_pb2.pyi": "778e150a702ff44936a6e7cb897963b2ef2eaa89",
    "serialization/ctrl_spec.py": "30895b51e932e2efa2571dac19373a210fb3a5a3",
    "serialization/ctrl_spec_test.py": "c7415d0a2b7a4648af1c282439858b13c630caf5",
}


def test_frozen_descriptors_and_stub_fields() -> None:
    for name, digest in BLOBS.items():
        data = (REFERENCE / name).read_bytes()
        assert hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest
    schema = (REFERENCE / "protos/ctrl_spec.proto").read_text()
    assert "repeated QDataType qdtypes = 1;" in schema
    assert "repeated NDArray cvs = 2;" in schema
    source = import_module("qualtran.protos.ctrl_spec_pb2").CtrlSpec.DESCRIPTOR
    expected = (("qdtypes", 1), ("cvs", 2))
    for descriptor in (source, _control_spec_message().DESCRIPTOR):
        assert [(field.name, field.number) for field in descriptor.fields] == list(
            expected
        )
        for field in descriptor.fields:
            assert (
                field.type == field.TYPE_MESSAGE and field.label == field.LABEL_REPEATED
            )
        assert len(descriptor.fields[0].message_type.oneofs_by_name["val"].fields) == 8
        assert [field.name for field in descriptor.fields[1].message_type.fields] == [
            "ndarray"
        ]
    stub = ast.parse((REFERENCE / "protos/ctrl_spec_pb2.pyi").read_text())
    cls = next(node for node in stub.body if isinstance(node, ast.ClassDef))
    ctor = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert tuple(arg.arg for arg in ctor.args.kwonlyargs) == ("qdtypes", "cvs")


def _original_cases() -> tuple[tuple[ControlClauses, Any], ...]:
    source = import_module("qualtran")
    bit = types.quantum_bit_type()
    return (
        (((bit, np.array(1)),), source.CtrlSpec()),
        (((bit, np.array(0)),), source.CtrlSpec(cvs=0)),
        (
            ((types.unsigned_register_type(4), np.array(6)),),
            source.CtrlSpec(qdtypes=source.QUInt(4), cvs=6),
        ),
        (((bit, np.array([0, 1, 1, 0])),), source.CtrlSpec(cvs=[0, 1, 1, 0])),
        (
            ((bit, np.array([0, 1])), (bit, np.array([1, 0]))),
            source.CtrlSpec(
                qdtypes=[source.QBit(), source.QBit()],
                cvs=[[0, 1], [1, 0]],
            ),
        ),
    )


def test_all_original_values_and_native_bytes() -> None:
    adapter = import_module("qualtran.serialization.ctrl_spec")
    for clauses, spec in _original_cases():
        original = adapter.ctrl_spec_to_proto(spec)
        data = serialize_control_spec(clauses)
        assert data == original.SerializeToString(deterministic=True)
        assert adapter.ctrl_spec_from_proto(type(original).FromString(data)) == spec
        decoded = deserialize_control_spec(data)
        assert len(decoded) == len(clauses)
        for (actual_dtype, actual), (expected_dtype, expected) in zip(
            decoded, clauses, strict=True
        ):
            assert actual_dtype == expected_dtype and actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            np.testing.assert_array_equal(actual, expected)


def test_shaped_refusal_and_malformed_parallel_lists() -> None:
    source = import_module("qualtran")
    shaped = import_module("qualtran.symbolics").Shaped((Symbol("n"),))
    adapter = import_module("qualtran.serialization.ctrl_spec")
    with pytest.raises(ValueError, match="cannot serialize Shaped"):
        adapter.ctrl_spec_to_proto(source.CtrlSpec(cvs=shaped))
    with pytest.raises(SerializationError, match="concrete NumPy arrays"):
        serialize_control_spec(((types.quantum_bit_type(), shaped),))
    message = _control_spec_message()()
    message.qdtypes.add().qbit.SetInParent()
    with pytest.raises(SerializationError, match="counts differ"):
        deserialize_control_spec(message.SerializeToString())
    assert deserialize_control_spec(serialize_control_spec(())) == ()


@pytest.mark.parametrize("fixture", [2, 3, 4])
def test_decoded_spec_drives_source_exact_append_operands_and_all_branches(
    fixture: int,
) -> None:
    clause, spec = _original_cases()[fixture]
    original = import_module("qualtran.bloqs.mcmt.ctrl_spec_and").CtrlSpecAnd(spec)
    resource = import_module("qualtran.resource_counting")
    decoded = deserialize_control_spec(serialize_control_spec(clause))
    groups = tuple(
        Register(source.name, dtype, values.shape)
        for source, (dtype, values) in zip(
            original.control_registers, decoded, strict=True
        )
    )
    native = PredicateExample(groups, control_registers_from_spec(decoded))
    assert native.activation == tuple(original._flat_cvs)
    n, forward, erase = _physical_pair(native, original)
    assert n == 4
    counts = Counter(gate.name for gate in forward)
    forward_cost = resource.get_cost_value(original, resource.QECGatesCost())
    erase_cost = resource.get_cost_value(original.adjoint(), resource.QECGatesCost())
    assert counts["t"] + counts["tdg"] == forward_cost.total_t_count() == 12
    assert sum(gate.name == "measure" for gate in erase) == erase_cost.measurement == 3

    basis = torch.arange(1 << n, dtype=torch.float64)
    initial = torch.zeros(1 << (2 * n - 1), dtype=torch.complex128)
    amplitudes = torch.complex(basis + 1, basis - 5)
    initial[:: 1 << (n - 1)] = amplitudes / amplitudes.norm()
    computed = forced_branch(initial, forward, ())
    expected = torch.zeros_like(initial)
    for address, amplitude in enumerate(amplitudes / amplitudes.norm()):
        prefix = 0
        matches = True
        for i, bit in enumerate(original._flat_cvs):
            matches = matches and (address >> (n - 1 - i)) & 1 == bit
            if i:
                prefix = (prefix << 1) | matches
        expected[(address << (n - 1)) | prefix] = amplitude
    torch.testing.assert_close(computed, expected, atol=2e-12, rtol=0)
    for history in product((0, 1), repeat=n - 1):
        torch.testing.assert_close(
            forced_branch(computed, erase, history),
            initial / 2 ** ((n - 1) / 2),
            atol=2e-12,
            rtol=0,
        )


def test_signed_and_fixed_point_keep_raw_integer_activation() -> None:
    clauses = (
        (types.ones_complement_register_type(3), np.array([-1, 2])),
        (types.fixed_point_register_type(3, 1, signed=True), np.array(-2)),
        (types.bounded_unsigned_register_type(3, 5), np.array(4)),
    )
    controls = control_registers_from_spec(
        deserialize_control_spec(serialize_control_spec(clauses))
    )
    assert tuple(register.bits() for register in controls) == (
        (1, 1, 0, 0, 1, 0),
        (1, 1, 0),
        (1, 0, 0),
    )
