"""Serialized local gates retain exact phase, decomposition and native ownership."""

import sys
from cmath import exp
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_component_arguments import ExternalGatePayload
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem.cirq._component_library import cirq_gate_library_factories
from flagquantum.errors import SerializationError
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_legacy_arithmetic_factories import _library
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_stored_rotation_array_resolution import _factories

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ftqc_reference,
    pytest.mark.usefixtures("locked_source"),
]
cirq = pytest.importorskip("cirq")


@pytest.mark.parametrize("kind", ("zpow", "cxpow", "matrix1", "matrix2", "scalar"))
def test_complete_local_gate_payloads_depths_phase_and_workspace(
    kind: str, locked_source: Any
) -> None:
    gates = {
        "zpow": cirq.ZPowGate(exponent=0.375),
        "cxpow": cirq.CXPowGate(exponent=-0.375),
        "matrix1": cirq.MatrixGate(
            exp(0.19j) * cirq.testing.random_unitary(2, random_state=17)
        ),
        "matrix2": cirq.MatrixGate(
            exp(-0.27j) * cirq.testing.random_unitary(4, random_state=23)
        ),
        "scalar": cirq.GlobalPhaseGate(exp(0.31j)),
    }
    gate = gates[kind]
    source = locked_source.CirqGateAsBloq(gate)
    record = resolve_component_program(
        _library(source, 0), 0, cirq_gate_library_factories()
    )
    assert verify_program_record(record) == ()
    assert (
        resolve_component_program(_library(source, 1), 0, cirq_gate_library_factories())
        == record
    )
    width = cirq.num_qubits(gate)
    ports = tuple((i,) for i in range(width))
    bound = bind_register_program(record, ports, clean=tuple(range(width, width + 3)))
    assert bound.outputs == ports
    torch.testing.assert_close(
        logical_operator(bound.circuit, width),
        torch.tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
            1 << width, 1 << width
        ),
        atol=1e-11,
        rtol=0,
    )
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit


def test_symbolic_gate_payload_and_execution_without_source_import(
    locked_source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sympy

    theta = sympy.Symbol("theta")
    gate = cirq.XPowGate(exponent=theta)
    library = _library(locked_source.CirqGateAsBloq(gate), 0)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "qualtran", None)
        record = resolve_component_program(library, 0, cirq_gate_library_factories())
        bound = bind_register_program(record, ((0,),))
        circuit = Circuit(1, dtype=torch.complex128).append(
            FieldOperation(1, bound.circuit.instructions)
        )
        actual = get_unitary(circuit.bind_parameters({"theta": 0.37}))
    expected = cirq.unitary(cirq.resolve_parameters(gate, {"theta": 0.37}))
    torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-11, rtol=0)
    nonlinear = cirq.XPowGate(exponent=sympy.sin(theta) + theta**2 / 3)
    with pytest.raises(TypeError, match="JSON serializable"):
        _library(locked_source.CirqGateAsBloq(nonlinear), 0)


@pytest.mark.parametrize("width", (1, 2))
def test_stored_arrays_with_arbitrary_local_payloads(width: int) -> None:
    from qualtran.bloqs.rotations.programmable_rotation_gate_array import (
        ProgrammableRotationGateArray,
    )

    rotation = cirq.MatrixGate(
        exp(0.19j) * cirq.testing.random_unitary(1 << width, random_state=17)
    )
    interleaver = cirq.MatrixGate(
        exp(-0.27j) * cirq.testing.random_unitary(1 << width, random_state=23)
    )
    source = ProgrammableRotationGateArray(
        (1, 3),
        (2, 1),
        kappa=2,
        rotation_gate=rotation,
        interleaved_unitaries=(interleaver,),
    )
    record = resolve_component_program(
        _library(source, 1), 0, _factories() | cirq_gate_library_factories()
    )
    logical = width + 3
    bound = bind_register_program(
        record,
        ((0,), (1, 2), tuple(range(3, logical))),
        clean=tuple(range(logical, logical + 3)),
    )
    from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter

    captured = import_register_emitter(source)
    expected = logical_operator(captured.snapshots[0], logical)
    if width == 2:
        with pytest.raises(ValueError, match="non-thru registers in C\\[And\\]"):
            source.tensor_contract()
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical),
        expected,
        atol=2e-10,
        rtol=0,
    )
    assert bound.outputs == ((0,), (1, 2), tuple(range(3, logical)))


@pytest.mark.parametrize("values", (0, 1, (0, 1)))
def test_external_control_polarities_and_grouped_layout(
    values: Any, locked_source: Any
) -> None:
    from qualtran import CtrlSpec

    gate = cirq.MatrixGate(exp(0.19j) * cirq.testing.random_unitary(2, random_state=17))
    source = locked_source.CirqGateAsBloq(gate).controlled(CtrlSpec(cvs=values))
    record = resolve_component_program(
        _library(source, 0), 0, cirq_gate_library_factories()
    )
    inputs = []
    offset = 0
    for register in record.functions[0].registers or ():
        for _ in register.all_idxs():
            inputs.append(tuple(range(offset, offset + int(register.bitsize))))
            offset += int(register.bitsize)
    bound = bind_register_program(
        record, tuple(inputs), clean=tuple(range(offset, offset + 3))
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, offset),
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )


def test_invalid_payload_and_missing_unitary_fail_explicitly(
    locked_source: Any,
) -> None:
    factory = cirq_gate_library_factories()[
        "qualtran.cirq_interop._cirq_to_bloq.CirqGateAsBloq"
    ]
    for arguments in ({}, {"gate": 3}, {"gate": ExternalGatePayload(b"invalid")}):
        with pytest.raises(SerializationError, match="(payload|constructor)"):
            factory("main", None, arguments)
    source = locked_source.CirqGateAsBloq(cirq.ResetChannel())
    record = resolve_component_program(
        _library(source, 0), 0, cirq_gate_library_factories()
    )
    assert not record.functions[0].blocks
    with pytest.raises(ValueError, match="component bodies"):
        bind_register_program(record, ((0,),))


def test_nested_external_controls_retain_new_control_first(locked_source: Any) -> None:
    from qualtran import Controlled, CtrlSpec

    gate = cirq.MatrixGate(exp(0.19j) * cirq.testing.random_unitary(2, random_state=17))
    source = Controlled(
        Controlled(locked_source.CirqGateAsBloq(gate), CtrlSpec(cvs=0)), CtrlSpec(cvs=1)
    )
    record = resolve_component_program(
        _library(source, 0), 0, cirq_gate_library_factories()
    )
    bound = bind_register_program(record, ((0,), (1,), (2,)), clean=(3, 4, 5))
    expected = cirq.unitary(
        gate.controlled(control_values=(0,)).controlled(control_values=(1,))
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, 3), torch.tensor(expected), atol=1e-11, rtol=0
    )


def test_other_controlled_children_delegate_to_their_existing_owner() -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import Hadamard

    from flagquantum.compiler.formats.legacy_controlled_factories import (
        controlled_via_and_factories,
    )
    from flagquantum.compiler.formats.legacy_fixed_gate_factories import (
        fixed_gate_factories,
    )

    source = Controlled(Hadamard(), CtrlSpec())
    fallback = controlled_via_and_factories()[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]
    factories = fixed_gate_factories() | cirq_gate_library_factories(
        controlled_factory=fallback
    )
    record = resolve_component_program(_library(source, 0), 0, factories)
    bound = bind_register_program(record, ((0,), (1,)), clean=(2,))
    torch.testing.assert_close(
        logical_operator(bound.circuit, 2),
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
