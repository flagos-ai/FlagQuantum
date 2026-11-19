# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source behavior audited against the frozen Qualtran v0.7.0 quantum_graph.py.
"""Audit the complete frozen graph metadata through native SSA consumers."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest
import torch

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")

from examples.algorithms.component_construction_tutorial import BoundComponentExample
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.width_schedule import WidthConnection
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ValueRef, verify_program_record
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
from flagquantum.drawer.component_debug_text import _port_labels
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_graph import register_connections
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_component_debug_text import normalized_source_text
from tests.unit.test_register_bindings import SourceContract

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")
SOURCE_REGISTER: Callable[[Register], Any] = SourceContract.register


@pytest.fixture(scope="module")
def source() -> Any:
    from qualtran._infra import quantum_graph

    return quantum_graph


def test_complete_source_graph_and_original_tests_are_frozen() -> None:
    root = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran/_infra"
    for name, blob in (
        ("quantum_graph.py", "923ddb3b6c79606ae5bd5c9ed23d22675b2028d9"),
        ("quantum_graph_test.py", "c4d12d29788d07975a139454622917c061bfd0f9"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_original_scalar_and_indexed_endpoints_use_existing_bindings(
    source: Any,
) -> None:
    from qualtran.bloqs.for_testing import TestTwoBitOp

    for name, shape, index in (("x", (), ()), ("y", (10, 2), (5, 0))):
        register = Register(name, RegisterType("opaque", 10), shape)
        original = source.Soquet(
            source.BloqInstance(TestTwoBitOp(), i=0),
            SOURCE_REGISTER(register),
            index,
        )
        native = RegisterBinding(ValueRef("entry", 0, "register"), register, index)
        assert native.index == original.idx
        assert native.register.side is RegisterSide.THRU
        labels = _port_labels((register,))
        assert original.pretty() in labels
        assert str(original).endswith("." + original.pretty())
        assert native == replace(native)
        assert hash(native) == hash(replace(native))
        with pytest.raises(FrozenInstanceError):
            setattr(native, "index", ())

    register = Register("y", RegisterType("opaque", 10), (10, 2))
    for invalid_index in ((), (5,)):
        with pytest.raises(ValueError, match="Bad index"):
            source.Soquet(source.LeftDangle, SOURCE_REGISTER(register), invalid_index)
        with pytest.raises(ValueError, match="Bad index"):
            RegisterBinding(ValueRef("entry", 0, "register"), register, invalid_index)


def test_original_instance_equality_hash_and_type_selection(source: Any) -> None:
    from qualtran.bloqs.for_testing import TestAtom, TestTwoBitOp

    first = source.BloqInstance(TestAtom(), i=1)
    equal = source.BloqInstance(TestAtom(), i=1)
    other = source.BloqInstance(TestTwoBitOp(), i=1)
    assert first == equal and first != other
    assert hash(first) == hash(equal) == hash(other) == hash(1)
    assert str(first) == "TestAtom<1>"
    assert first.bloq_is(TestAtom) and not other.bloq_is(TestAtom)
    assert not source.LeftDangle.bloq_is(TestAtom)
    assert not source.RightDangle.bloq_is(TestAtom)

    # Native occurrence identity combines the selected function with its ordinal.
    signature = Signature.build(q=1)
    atom = bit_gate_function("atom", tuple(signature), CircuitIR(1, ()))
    different = replace(atom, name="different")
    instances = {(atom, 1): "atom", (different, 1): "different"}
    assert instances[(replace(atom), 1)] == "atom"
    assert instances[(different, 1)] == "different"
    assert len(instances) == 2


@pytest.mark.parametrize("width", [0, 1, 10])
def test_equal_width_connection_storage_and_mismatch_boundary(
    source: Any, width: int
) -> None:
    source_left = SOURCE_REGISTER(Register("a", RegisterType("opaque", width)))
    source_right = SOURCE_REGISTER(Register("b", RegisterType("uint", width)))
    original = source.Connection(
        source.Soquet(source.LeftDangle, source_left),
        source.Soquet(source.RightDangle, source_right),
    )
    native = WidthConnection(0, 1, width)
    assert native.qubits == original.shape == width
    assert original == source.Connection(original.left, original.right)
    assert hash(original) == hash(source.Connection(original.left, original.right))
    assert str(original) == "LeftDangle.a -> RightDangle.b"

    mismatched = SOURCE_REGISTER(Register("b", RegisterType("uint", width + 1)))
    invalid = source.Connection(
        original.left, source.Soquet(source.RightDangle, mismatched)
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        _ = invalid.shape

    # The native verifier checks the same edge at the Program boundary.
    value = ValueRef("entry", 0, "register")
    from flagquantum.core.ir.program import BlockRecord, FunctionRecord, ProgramRecord

    record = ProgramRecord(
        (
            FunctionRecord(
                "main",
                (BlockRecord("entry", (value,), return_values=(value,)),),
                registers=(
                    Register(
                        "a", RegisterType("opaque", width), side=RegisterSide.LEFT
                    ),
                    Register("b", RegisterType("uint", width), side=RegisterSide.RIGHT),
                ),
            ),
        )
    )
    assert verify_program_record(record) == ()
    function = record.functions[0]
    assert function.registers is not None
    bad = replace(
        function,
        registers=(
            function.registers[0],
            Register("b", RegisterType("uint", width + 1), side=RegisterSide.RIGHT),
        ),
    )
    assert {d.code for d in verify_program_record(ProgramRecord((bad,)))} == {
        "program.register_type"
    }


def test_symbolic_connection_width_stays_metadata(source: Any) -> None:
    import sympy

    width: SymbolicInt = sympy.Symbol("n", integer=True, positive=True)
    register = Register("word", RegisterType("opaque", width))
    native = RegisterBinding(ValueRef("entry", 0, "register"), register)
    original = source.Connection(
        source.Soquet(source.LeftDangle, SOURCE_REGISTER(register)),
        source.Soquet(source.RightDangle, SOURCE_REGISTER(register)),
    )
    edge = WidthConnection(0, 1, native.register.bitsize)
    assert edge.qubits == original.shape == width
    assert native.index == original.left.idx == original.right.idx == ()


@pytest.mark.parametrize("width", [0, 2])
def test_classical_word_connections_retain_logical_width_and_identity(
    source: Any, width: int
) -> None:
    from qualtran import CUInt
    from qualtran import Register as SourceRegister

    from flagquantum.core.ir.program.register_types import (
        classical_unsigned_register_type,
    )

    register = Register("word", classical_unsigned_register_type(width))
    original_register = SourceRegister("word", CUInt(width))
    original = source.Connection(
        source.Soquet(source.LeftDangle, original_register),
        source.Soquet(source.RightDangle, original_register),
    )
    before = RegisterBinding(ValueRef("entry", 0, "register"), register)
    after = RegisterBinding(ValueRef("entry", 1, "register"), register)
    assert before != after and len({before, after}) == 2
    assert before.register.bitsize == after.register.bitsize == original.shape == width
    assert register.total_qubits() == original_register.total_qubits() == 0
    assert register.total_cbits() == original_register.total_cbits() == width


def test_real_repeated_calls_boundary_identity_connectivity_and_append(
    source: Any,
) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from flagquantum.drawer.component_debug_text import format_component_debug_text

    signature = Signature.build(q=1)
    flip = bit_gate_function(
        "flip", tuple(signature), CircuitIR(1, (Instruction("x", (0,)),))
    )
    builder = ComponentProgramBuilder(signature, (flip,))
    binding = builder.inputs["q"]
    outputs = builder.call_dict("flip", {"q": binding})
    outputs = builder.call_dict("flip", outputs)
    record = builder.finalize(outputs)
    assert verify_program_record(record) == ()
    source_builder, inputs = BloqBuilder.from_signature(XGate().signature)
    q = source_builder.add(XGate(), q=inputs["q"])
    q = source_builder.add(XGate(), q=q)
    original = source_builder.finalize(q=q)
    assert format_component_debug_text(record) == normalized_source_text(
        record, original
    )
    assert [i for i, _, _ in iter_component_calls(record)] == [0, 1]
    assert (
        len({v for op in record.functions[0].blocks[0].operations for v in op.results})
        == 2
    )
    edges = register_connections(component_register_score(record))
    assert [(edge.producer, edge.consumer, edge.width) for edge in edges] == [
        (None, 0, 1),
        (0, 1, 1),
        (1, None, 1),
    ]
    assert edges[0] != edges[-1]
    assert {edges[0]: "input", edges[-1]: "output"}[edges[0]] == "input"
    assert source.LeftDangle is not source.RightDangle
    assert source.LeftDangle != source.RightDangle
    assert {source.LeftDangle: "input", source.RightDangle: "output"}[
        source.RightDangle
    ] == "output"
    with pytest.raises(ValueError, match="Do not instantiate"):
        source.DanglingT("extra")

    bound = bind_register_program(record, ((0,),), clean=(2,))
    assert bound.events == () and bound.outputs == ((0,),)
    assert bound.circuit.instructions == (Instruction("x", (0,)),) * 2
    state = torch.tensor([1 + 2j, 3j, -1 + 4j, 2 - 3j], dtype=torch.complex128)
    state /= state.norm()
    joint = torch.kron(state, torch.tensor([1, 0], dtype=torch.complex128))
    circuit = Circuit(3, inputs=joint, dtype=torch.complex128).append(
        BoundComponentExample(record, ((0,),))
    )
    np.testing.assert_allclose(original.tensor_contract(), np.eye(2), atol=1e-15)
    np.testing.assert_allclose(
        get_unitary(bound.circuit, dtype=torch.complex128).numpy(),
        np.kron(original.tensor_contract(), np.eye(4)),
        atol=1e-15,
    )
    actual = circuit.state().reshape(4, 2)
    torch.testing.assert_close(actual[:, 0], state, atol=1e-15, rtol=0)
    assert torch.count_nonzero(actual[:, 1]) == 0
    bound_circuit = Circuit(3, inputs=joint, dtype=torch.complex128).compose(
        bound.circuit
    )
    torch.testing.assert_close(
        bound_circuit.state(), circuit.state(), atol=1e-15, rtol=0
    )
    model = get_cost_value(original, QECGatesCost())
    assert model.clifford == 2 and model.measurement == 0 and model.total_t_count() == 0
