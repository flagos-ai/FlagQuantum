from __future__ import annotations

from math import pi

import pytest
import torch

import flagquantum as fq
from flagquantum.ecosystem import get_adapter
from flagquantum.ecosystem.qiskit import (
    QiskitConversionError,
    export_qiskit,
    from_qiskit,
    import_qiskit,
    to_qiskit,
)

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit, QuantumRegister  # noqa: E402
from qiskit.circuit import Parameter as QiskitParameter  # noqa: E402
from qiskit.circuit.library import UnitaryGate  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.qiskit]


def test_registered_adapter_delegates_through_common_contract() -> None:
    adapter = get_adapter("qiskit")
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    imported = adapter.import_program(circuit)
    exported = adapter.export_program(imported.ir)

    assert imported.report.adapter == "qiskit"
    assert tuple(item.name for item in imported.ir.instructions) == ("h", "cx")
    assert tuple(item.operation.name for item in exported.artifact.data) == ("h", "cx")


def test_qiskit_import_builds_versioned_ir_with_parameters_and_measurements() -> None:
    theta = QiskitParameter("theta")
    circuit = QuantumCircuit(3, 2, name="source")
    circuit.global_phase = 0.125
    circuit.h(0)
    circuit.rx(theta, 1)
    circuit.cx(1, 2)
    circuit.measure(2, 1)

    result = import_qiskit(circuit)

    assert result.report.lossless
    assert result.ir.version == fq.IR_VERSION
    assert result.ir.n_wires == 3
    assert tuple(item.name for item in result.ir.instructions) == (
        "h",
        "rx",
        "cx",
        "measure",
    )
    assert result.ir.instructions[1].params["theta"] == fq.Parameter("theta")
    assert result.ir.instructions[-1].metadata["classical_bit"] == 1
    assert result.ir.metadata["interop"]["num_clbits"] == 2
    assert result.ir.metadata["interop"]["global_phase"] == pytest.approx(0.125)


def test_flagquantum_ir_exports_to_qiskit_and_round_trips_gate_semantics() -> None:
    theta = fq.Parameter("theta")
    circuit = fq.Circuit(3).h(0).ry(1, theta=theta).cx(1, 2)

    exported = export_qiskit(circuit)
    names = tuple(item.operation.name for item in exported.circuit.data)

    assert exported.report.lossless
    assert names == ("h", "ry", "cx")
    assert tuple(item.name for item in from_qiskit(exported.circuit).instructions) == (
        "h",
        "ry",
        "cx",
    )
    assert {parameter.name for parameter in exported.circuit.parameters} == {"theta"}
    assert (
        exported.circuit.metadata["flagquantum_ir_hash"] == circuit.to_ir().content_hash
    )


def test_barrier_requires_explicit_lossy_conversion_and_reports_skip() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.barrier(0)
    circuit.x(0)

    with pytest.raises(QiskitConversionError) as captured:
        from_qiskit(circuit)

    assert captured.value.report.issues[0].code == "barrier_dropped"
    result = import_qiskit(circuit, allow_lossy=True)
    assert not result.report.lossless
    assert tuple(item.name for item in result.ir.instructions) == ("h", "x")


def test_qiskit_control_flow_fails_closed() -> None:
    circuit = QuantumCircuit(1, 1)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(0)

    with pytest.raises(QiskitConversionError) as captured:
        from_qiskit(circuit)

    assert captured.value.report.blockers[0].code == "unsupported_control_flow"


def test_nonstandard_register_layout_requires_explicit_flattening() -> None:
    register = QuantumRegister(2, "logical")
    circuit = QuantumCircuit(register)
    circuit.h(register[0])

    with pytest.raises(QiskitConversionError) as captured:
        from_qiskit(circuit)

    assert captured.value.report.issues[0].code == "register_layout_flattened"
    result = import_qiskit(circuit, allow_lossy=True)
    assert result.ir.instructions[0].wires == (0,)
    assert result.ir.metadata["interop"]["qregs"] == (("logical", 2),)


def test_static_custom_unitary_round_trips_without_entering_runtime_kernels() -> None:
    matrix = [[0, 1], [1, 0]]
    circuit = QuantumCircuit(1)
    circuit.append(UnitaryGate(matrix, label="custom-x"), [0])

    ir = from_qiskit(circuit)
    assert ir.instructions[0].name == "unitary"
    assert torch.equal(
        ir.instructions[0].matrix,
        torch.tensor(matrix, dtype=ir.instructions[0].matrix.dtype),
    )

    round_trip = to_qiskit(ir)
    assert round_trip.data[0].operation.name == "unitary"
    assert round_trip.data[0].operation.label == "custom-x"


def test_multi_qubit_custom_unitary_fails_closed_until_basis_order_is_defined() -> None:
    circuit = QuantumCircuit(2)
    circuit.append(UnitaryGate(torch.eye(4).numpy()), [0, 1])

    with pytest.raises(QiskitConversionError) as captured:
        from_qiskit(circuit)

    assert (
        captured.value.report.blockers[0].code
        == "multi_qubit_unitary_wire_order_unverified"
    )


def test_execution_measurement_requests_are_not_silently_discarded() -> None:
    ir = fq.CircuitIR(
        1,
        (fq.Instruction("h", (0,)),),
        measurements=(fq.MeasurementNode("sample", (0,), shots=10),),
    )

    with pytest.raises(QiskitConversionError) as captured:
        to_qiskit(ir)

    assert captured.value.report.blockers[0].code == (
        "measurement_requests_not_embedded"
    )


def test_flagquantum_parameter_expression_exports_without_eager_qiskit_state() -> None:
    theta = fq.Parameter("theta")
    ir = fq.CircuitIR(
        1,
        (
            fq.Instruction(
                "rz",
                (0,),
                params={"theta": -(theta * 2.0) + pi},
            ),
        ),
    )

    circuit = to_qiskit(ir)
    assert {parameter.name for parameter in circuit.parameters} == {"theta"}
    bound = circuit.assign_parameters({"theta": 0.25})
    assert float(bound.data[0].operation.params[0]) == pytest.approx(pi - 0.5)
