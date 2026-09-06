from __future__ import annotations

from pathlib import Path

import pytest
import torch

import flagquantum as fq
import flagquantum.operators as fqo
from flagquantum.ecosystem.qiskit import (
    from_qiskit,
    qiskit_statevector_to_flagquantum,
    run_qiskit_conformance,
    semantic_fingerprint,
    to_qiskit,
)

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.qiskit]
ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib
    return tomllib.loads(
        (ROOT / "contracts" / "qiskit-interop-contract.toml").read_text(
            encoding="utf-8"
        )
    )


def test_every_contract_operation_converts_in_both_directions() -> None:
    values = (0.173, -0.291, 0.419)
    for operation in _contract()["operations"]:
        schema = fqo.gate_info(operation["flagquantum"])
        n_wires = max(3, schema.n_wires)
        wires = tuple(range(schema.n_wires))
        parameters = values[: schema.n_parameters]

        qiskit_circuit = QuantumCircuit(n_wires)
        qiskit_circuit.h(0)
        qiskit_circuit.ry(0.231, 1)
        qiskit_circuit.x(2)
        getattr(qiskit_circuit, operation["qiskit"])(*parameters, *wires)
        imported = from_qiskit(qiskit_circuit)
        assert imported.instructions[-1].name == operation["flagquantum"]
        assert tuple(imported.instructions[-1].params) == tuple(
            operation["flagquantum_parameters"]
        )

        flagquantum_ir = fq.CircuitIR(
            n_wires,
            (
                fq.Instruction("h", (0,)),
                fq.Instruction("ry", (1,), params={"theta": 0.231}),
                fq.Instruction("x", (2,)),
                fq.Instruction(
                    operation["flagquantum"],
                    wires,
                    params=dict(zip(operation["flagquantum_parameters"], parameters)),
                ),
            ),
            shape=(1, 2**n_wires),
        )
        exported = to_qiskit(flagquantum_ir)
        assert exported.data[-1].operation.name == operation["qiskit"]
        flagquantum_state = fq.Circuit.from_ir(
            flagquantum_ir, dtype=torch.complex128
        ).state()[0]
        qiskit_state = qiskit_statevector_to_flagquantum(
            Statevector.from_instruction(exported).data,
            n_wires,
        ).to(dtype=torch.complex128)
        torch.testing.assert_close(
            flagquantum_state,
            qiskit_state,
            rtol=0.0,
            atol=1e-10,
            msg=lambda message: (
                f"statevector mismatch for {operation['flagquantum']}: {message}"
            ),
        )


def test_statevector_and_round_trip_golden_cases_pass() -> None:
    result = run_qiskit_conformance()

    assert result.schema == "flagquantum_qiskit_conformance_v1"
    assert result.qiskit_version.startswith(("2.0.", "2.5."))
    assert result.passed
    assert result.adapter_contract is not None
    assert result.adapter_contract.passed
    assert [case.kind for case in result.adapter_contract.cases] == [
        "round_trip",
        "round_trip",
        "round_trip",
        "rejection",
    ]
    assert len(result.cases) == 3
    assert all(case.maximum_absolute_error <= 1e-10 for case in result.cases)
    assert all(
        case.source_fingerprint == case.round_trip_fingerprint for case in result.cases
    )
    payload = result.to_dict()
    assert payload["schema"] == result.schema
    assert payload["passed"] is True
    assert payload["adapter_contract"]["schema"] == (
        "flagquantum_interop_conformance_v1"
    )
    assert [case["name"] for case in payload["cases"]] == [
        case.name for case in result.cases
    ]


def test_qiskit_little_endian_statevector_is_normalized_to_wire_major() -> None:
    qiskit_circuit = QuantumCircuit(3)
    qiskit_circuit.x(0)
    state = qiskit.quantum_info.Statevector.from_instruction(qiskit_circuit).data

    converted = qiskit_statevector_to_flagquantum(state, 3)

    expected = torch.zeros(8, dtype=converted.dtype)
    expected[4] = 1
    assert torch.equal(converted, expected)


def test_measurement_classical_bit_indices_survive_export_and_aer() -> None:
    ir = fq.CircuitIR(
        3,
        (
            fq.Instruction("x", (0,)),
            fq.Instruction(
                "measure",
                (0,),
                metadata={"is_dynamic": True, "classical_bit": 2},
            ),
            fq.Instruction(
                "measure",
                (2,),
                metadata={"is_dynamic": True, "classical_bit": 0},
            ),
        ),
    )
    circuit = to_qiskit(ir)
    memory = AerSimulator().run(circuit, shots=4, memory=True).result().get_memory()

    assert memory == ["100"] * 4
    round_trip = from_qiskit(circuit)
    assert tuple(
        item.metadata["classical_bit"]
        for item in round_trip.instructions
        if item.name == "measure"
    ) == (2, 0)


def test_semantic_fingerprint_ignores_transport_provenance_only() -> None:
    source = fq.Circuit(2).h(0).cx(0, 1).to_ir()
    round_trip = from_qiskit(to_qiskit(source))

    assert semantic_fingerprint(source) == semantic_fingerprint(round_trip)
