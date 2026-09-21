from __future__ import annotations

from math import pi

import pytest
import torch

import flagquantum as fq
from flagquantum.core.parameters import parameter_names_in_value
from flagquantum.ecosystem import get_adapter
from flagquantum.ecosystem.qiskit import (
    QiskitConversionError,
    export_qiskit,
    from_qiskit,
    import_qiskit,
    qiskit_statevector_to_flagquantum,
    to_qiskit,
)

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit, QuantumRegister  # noqa: E402
from qiskit.circuit import Parameter as QiskitParameter  # noqa: E402
from qiskit.circuit.library import UnitaryGate  # noqa: E402
from qiskit.quantum_info import Statevector, random_unitary  # noqa: E402

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


@pytest.mark.parametrize("width", (2, 3))
def test_multi_qubit_custom_unitary_round_trips_local_basis_order(width: int) -> None:
    matrix = random_unitary(2**width, seed=100 + width).data
    qargs = list(reversed(range(width)))
    circuit = QuantumCircuit(width)
    circuit.append(UnitaryGate(matrix, label=f"custom-{width}"), qargs)

    ir = from_qiskit(circuit)
    round_trip = to_qiskit(ir)

    assert ir.instructions[0].wires == tuple(qargs)
    assert round_trip.data[0].operation.label == f"custom-{width}"
    torch.testing.assert_close(
        torch.as_tensor(round_trip.data[0].operation.to_matrix()),
        torch.as_tensor(matrix),
        rtol=0.0,
        atol=1e-12,
    )


def test_multi_qubit_custom_unitary_matches_qiskit_on_reordered_sparse_wires() -> None:
    matrix = random_unitary(4, seed=731).data
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.ry(0.37, 1)
    circuit.x(3)
    circuit.append(UnitaryGate(matrix, label="asymmetric"), [3, 1])

    ir = from_qiskit(circuit)
    flagquantum_state = fq.Circuit.from_ir(ir, dtype=torch.complex128).state()[0]
    qiskit_state = qiskit_statevector_to_flagquantum(
        Statevector.from_instruction(circuit).data,
        4,
    ).to(torch.complex128)

    torch.testing.assert_close(
        flagquantum_state,
        qiskit_state,
        rtol=0.0,
        atol=1e-10,
    )


def test_custom_unitary_export_fails_closed_for_invalid_matrix() -> None:
    ir = fq.CircuitIR(
        2,
        (
            fq.Instruction(
                "unitary",
                (0, 1),
                matrix=torch.ones((4, 4), dtype=torch.complex128),
            ),
        ),
        shape=(1, 4),
    )

    with pytest.raises(QiskitConversionError) as captured:
        to_qiskit(ir)

    assert captured.value.report.blockers[0].code == "non_unitary_custom_matrix"


@pytest.mark.parametrize(
    ("matrix", "issue_code"),
    (
        (torch.eye(3, dtype=torch.complex128), "invalid_custom_unitary_shape"),
        (
            torch.diag(
                torch.tensor(
                    [1.0, 1.0, 1.0, complex(float("nan"), 0.0)],
                    dtype=torch.complex128,
                )
            ),
            "invalid_custom_unitary_values",
        ),
    ),
)
def test_custom_unitary_export_reports_invalid_matrix_contract(
    matrix: torch.Tensor,
    issue_code: str,
) -> None:
    ir = fq.CircuitIR(
        2,
        (fq.Instruction("unitary", (0, 1), matrix=matrix),),
        shape=(1, 4),
    )

    with pytest.raises(QiskitConversionError) as captured:
        to_qiskit(ir)

    assert captured.value.report.blockers[0].code == issue_code


def test_custom_unitary_export_rejects_width_above_certified_limit() -> None:
    ir = fq.CircuitIR(
        4,
        (
            fq.Instruction(
                "unitary",
                (0, 1, 2, 3),
                matrix=torch.eye(16, dtype=torch.complex128),
            ),
        ),
        shape=(1, 16),
    )

    with pytest.raises(QiskitConversionError) as captured:
        to_qiskit(ir)

    assert (
        captured.value.report.blockers[0].code == "custom_unitary_width_exceeds_limit"
    )


def test_execution_measurement_requests_are_not_silently_discarded() -> None:
    from flagquantum.core.ir import MeasurementNode

    ir = fq.CircuitIR(
        1,
        (fq.Instruction("h", (0,)),),
        measurements=(MeasurementNode("sample", (0,), shots=10),),
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


def test_qiskit_arithmetic_parameter_expression_imports_and_round_trips() -> None:
    theta = QiskitParameter("theta")
    offset = QiskitParameter("offset")
    circuit = QuantumCircuit(1)
    circuit.rz(-(theta * 2.0) + offset, 0)

    imported = import_qiskit(circuit)

    expression = imported.ir.instructions[0].params["theta"]
    assert parameter_names_in_value(expression) == ("offset", "theta")
    bound_ir = (
        fq.Circuit.from_ir(imported.ir)
        .bind_parameters({"theta": 0.25, "offset": pi})
        .to_ir()
    )
    assert bound_ir.instructions[0].params["theta"] == pytest.approx(pi - 0.5)

    round_trip = to_qiskit(imported.ir)
    qiskit_bound = round_trip.assign_parameters({"theta": 0.25, "offset": pi})
    assert float(qiskit_bound.data[0].operation.params[0]) == pytest.approx(pi - 0.5)


def test_qiskit_parameter_expression_outside_arithmetic_subset_fails_closed() -> None:
    theta = QiskitParameter("theta")
    circuit = QuantumCircuit(1)
    circuit.rz(theta.sin(), 0)

    with pytest.raises(QiskitConversionError) as captured:
        import_qiskit(circuit)

    blocker = captured.value.report.blockers[0]
    assert blocker.code == "unsupported_parameter_expression"
    assert "sin" in blocker.message
