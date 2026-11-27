"""Preserve empty source declarations and executable Montgomery half comparisons."""

from importlib import import_module
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import cirq
    import torch

    _source = import_module("qualtran.bloqs.arithmetic.comparison")
    _types = import_module("qualtran")
    _costs = import_module("qualtran.resource_counting")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_comparison_factories import (
    comparison_factories,
)
from flagquantum.compiler.formats.legacy_component_library import (
    deserialize_component_library,
    serialize_component_library,
)
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.comparison.comparison_resource_model import (
    source_less_than_constant_qec_cost,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.ecosystem._default_linear_combination_oracles import _signature
from flagquantum.ecosystem.cirq._register_import import import_cirq_register_gate
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_comparison_component import allocation_order
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_qubitization_walk_component import (
    reference_runtime as reference_runtime,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_sparse_chemistry_prepare import _branch
from tests.unit.test_sparse_chemistry_prepare_component import bind_used_workspace

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _sdk_inputs(record: ProgramRecord) -> tuple[tuple[int, ...], ...]:
    ports = []
    offset = 0
    for register in Signature(record.functions[0].registers or ()).lefts():
        for _ in register.all_idxs():
            width = register.bitsize
            assert isinstance(width, int)
            ports.append(tuple(range(offset, offset + width)))
            offset += width
    return tuple(ports)


@pytest.mark.parametrize("width,threshold", ((0, 0), (0, 1), (0, 2), (0, -1), (2, -1)))
def test_declaration_only_constant_categories_do_not_invent_a_peak(
    width: int, threshold: int
) -> None:
    source = _source.LessThanConstant(width, threshold)
    expected = _costs.get_cost_value(source, _costs.QECGatesCost())
    assert (
        source_less_than_constant_qec_cost(width, threshold).asdict()
        == expected.asdict()
    )


@pytest.mark.parametrize(
    "kind,value",
    (
        ("LessThanConstant", -1),
        ("EqualsAConstant", -1),
        ("EqualsAConstant", 4),
        ("EqualsAConstant", 5),
    ),
)
def test_serializable_constant_with_failed_source_body_keeps_logical_declaration(
    kind: str, value: int
) -> None:
    source = getattr(_source, kind)(2, value)
    record = resolve_component_program(_library(source, 0), 0, comparison_factories())
    assert record.functions[0].registers == _signature(source.signature)
    assert not record.functions[0].blocks
    with pytest.raises(ValueError, match="classical value"):
        source.tensor_contract()
    if kind == "LessThanConstant":
        # Its SDK ArithmeticGate unitary remains defined despite the failed
        # Bloq decomposition. Exercise that distinct, existing native route.
        sdk_record = import_cirq_register_gate(source)
        assert sdk_record is not None
        bound = bind_register_program(sdk_record, _sdk_inputs(sdk_record))
        torch.testing.assert_close(
            get_unitary(bound.circuit, dtype=torch.complex128),
            torch.as_tensor(cirq.unitary(source)),
            rtol=0,
            atol=2e-12,
        )


def _declaration(kind: str) -> Any:
    if kind.startswith("lt_"):
        return _source.LessThanConstant(0, int(kind[-1]))
    if kind.startswith("eq_"):
        return _source.EqualsAConstant(0, int(kind[-1]))
    if kind == "equals":
        return _source.Equals(_types.QUInt(0))
    if kind == "controlled":
        return _source.CLinearDepthGreaterThan(_types.QUInt(0))
    if kind.startswith("linear"):
        return _source.LinearDepthGreaterThan(0, signed=kind.endswith("signed"))
    if kind == "signed_half":
        return _source.LinearDepthHalfLessThan(_types.QInt(0))
    left, right = {"tree00": (0, 0), "tree01": (0, 1), "tree10": (1, 0)}[kind]
    return _source.LessThanEqual(left, right)


@pytest.mark.parametrize(
    "kind",
    (
        "lt_0",
        "lt_1",
        "lt_2",
        "eq_0",
        "eq_1",
        "equals",
        "controlled",
        "linear",
        "linear_signed",
        "signed_half",
        "tree00",
        "tree01",
        "tree10",
    ),
)
def test_source_valid_empty_transport_retains_declaration_and_physical_refusal(
    kind: str,
) -> None:
    source = _declaration(kind)
    library = _library(source, 0)
    wire = serialize_component_library(library)
    assert serialize_component_library(deserialize_component_library(wire)) == wire
    record = resolve_component_program(library, 0, comparison_factories())
    assert record.functions[0].registers == _signature(source.signature)
    assert not record.functions[0].blocks
    assert not verify_program_record(record)
    error = IndexError if kind == "signed_half" else (KeyError, TypeError)
    with pytest.raises(error):
        source.tensor_contract()
    ports = []
    offset = 0
    for register in source.signature.lefts():
        width = int(register.total_bits())
        ports.append(tuple(range(offset, offset + width)))
        offset += width
    with pytest.raises(ValueError, match="requires component bodies"):
        bind_register_program(
            record,
            tuple(ports),
            clean=tuple(range(10, 20)),
            allow_workspace_outputs=True,
        )
    if isinstance(source, cirq.ArithmeticGate):
        sdk_record = import_cirq_register_gate(source)
        assert sdk_record is not None
        bound = bind_register_program(sdk_record, _sdk_inputs(sdk_record))
        torch.testing.assert_close(
            get_unitary(bound.circuit, dtype=torch.complex128),
            torch.as_tensor(cirq.unitary(source)),
            rtol=0,
            atol=2e-12,
        )


@pytest.mark.parametrize(
    "name",
    (
        "_HalfLinearDepthGreaterThan",
        "LinearDepthHalfGreaterThan",
        "LinearDepthHalfGreaterThanEqual",
        "LinearDepthHalfLessThan",
        "LinearDepthHalfLessThanEqual",
    ),
)
def test_empty_montgomery_half_source_stream_complex_state_inverse_and_work(
    name: str,
) -> None:
    dtype = _types.QMontgomeryUInt(0)
    source = getattr(_source, name)(dtype)
    forward = resolve_component_program(_library(source, 0), 0, comparison_factories())
    inverse = resolve_component_program(
        _library(source.adjoint(), 0), 0, comparison_factories()
    )
    assert not verify_program_record(forward) and not verify_program_record(inverse)
    bound = bind_register_program(
        forward, ((), ()), clean=tuple(range(20)), allow_workspace_outputs=True
    )
    outputs = dict(
        zip((r.name for r in source.signature.rights()), bound.outputs, strict=True)
    )
    reference = reference_arithmetic_gates(
        source,
        {"a": (), "b": ()},
        allocation_order(bound),
        MeasurementSlots(bound.circuit.instructions),
        expected_outputs=outputs,
        reuse_allocations=False,
    )
    assert_source(bound.circuit.instructions, reference)
    count = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert count == 0
    actual = _branch(bound.circuit.instructions, bound.circuit.n_wires, (), {0: 1 + 0j})
    kept = tuple(w for word in bound.outputs for w in word)
    private = set(range(bound.circuit.n_wires)) - set(kept)
    mask = sum(1 << (bound.circuit.n_wires - 1 - w) for w in private)
    assert all(not key & mask for key in actual)
    decoded = {
        sum(
            1 << (len(kept) - 1 - i)
            for i, w in enumerate(kept)
            if key & (1 << (bound.circuit.n_wires - 1 - w))
        ): value
        for key, value in actual.items()
    }
    expected = {
        i: complex(v)
        for i, v in enumerate(source.tensor_contract().reshape(-1))
        if abs(v) > 1e-13
    }
    assert decoded == pytest.approx(expected, abs=2e-12)
    backward = bind_register_program(
        inverse,
        bound.outputs,
        clean=tuple(wire for wire in range(20) if wire not in kept),
        input_ownership=tuple((wire, False) for wire in kept),
    )
    reverse_reference = reference_arithmetic_gates(
        source.adjoint(),
        outputs,
        allocation_order(backward),
        MeasurementSlots(backward.circuit.instructions),
        expected_outputs={"a": (), "b": ()},
        reuse_allocations=False,
    )
    assert_source(backward.circuit.instructions, reverse_reference)
    signature = Signature(
        (
            *Signature(forward.functions[0].registers or ()).lefts(),
            Register("spectator", quantum_bit_type()),
        )
    )
    builder = ComponentProgramBuilder(signature)
    restored = builder.call_program(
        inverse,
        builder.call_program(
            forward, {"a": builder.inputs["a"], "b": builder.inputs["b"]}
        ),
    )
    joint = builder.finalize({**restored, "spectator": builder.inputs["spectator"]})
    bound_joint, logical = bind_used_workspace(joint)
    assert logical == 1
    initial = {0: 0.6 + 0j, 1 << (bound_joint.circuit.n_wires - 1): 0.8j}
    actual = _branch(
        bound_joint.circuit.instructions, bound_joint.circuit.n_wires, (), initial
    )
    assert actual == pytest.approx(initial, abs=2e-12)
    state = torch.zeros(1 << bound_joint.circuit.n_wires, dtype=torch.complex128)
    for key, value in initial.items():
        state[key] = torch.tensor(value, dtype=torch.complex128)
    circuit = Circuit(bound_joint.circuit.n_wires, inputs=state, dtype=torch.complex128)
    circuit.append(
        FieldOperation(bound_joint.circuit.n_wires, bound_joint.circuit.instructions)
    )
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)
    assert (
        bound_joint.circuit.from_json(bound_joint.circuit.to_json())
        == bound_joint.circuit
    )
