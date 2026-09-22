from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.parameters import Parameter
from flagquantum.ecosystem import get_adapter
from flagquantum.ecosystem.braket import (
    BraketConversionError,
    from_braket,
    import_braket,
    to_braket,
)

braket = pytest.importorskip("braket.circuits")
np = import_module("numpy")
pytestmark = [pytest.mark.integration, pytest.mark.braket]
ROOT = Path(__file__).resolve().parents[1]

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

CONTRACT = tomllib.loads(
    (ROOT / "contracts" / "braket-circuit-interop-contract.toml").read_text(
        encoding="utf-8"
    )
)


def _braket_state(circuit: braket.Circuit) -> torch.Tensor:
    unitary = np.asarray(circuit.to_unitary(), dtype=np.complex128)
    return torch.as_tensor(unitary[:, 0].copy())


def test_braket_round_trip_preserves_complex128_semantics_and_order() -> None:
    source = (
        Circuit(3, dtype=torch.complex128)
        .x(2)
        .h(0)
        .ry(1, theta=0.231)
        .cx(0, 2)
        .cz(2, 1)
    )
    external = to_braket(source.to_ir())
    round_trip = from_braket(external)
    assert round_trip.dtype == "complex128"
    assert [item.name for item in round_trip.instructions] == [
        "x",
        "h",
        "ry",
        "cx",
        "cz",
    ]
    expected = Circuit.from_ir(round_trip, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(
        _braket_state(external), expected, atol=1e-10, rtol=1e-10
    )
    assert get_adapter("braket").name == "braket"


@pytest.mark.parametrize("mapping", CONTRACT["operations"])
def test_every_declared_gate_matches_at_complex128(mapping: dict[str, str]) -> None:
    opcode = mapping["flagquantum"]
    schema = OPERATOR_SCHEMAS[opcode]
    parameters = {
        name: 0.173 * (index + 1) for index, name in enumerate(schema.parameters)
    }
    preparations = tuple(
        Instruction("ry", (wire,), {"theta": 0.119 * (wire + 1)})
        for wire in range(schema.arity)
    )
    ir = CircuitIR(
        schema.arity,
        preparations + (Instruction(opcode, tuple(range(schema.arity)), parameters),),
        dtype="complex128",
    )
    external = to_braket(ir)
    restored = from_braket(external)
    expected = Circuit.from_ir(ir, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(
        _braket_state(external), expected, atol=1e-10, rtol=1e-10
    )
    actual = Circuit.from_ir(restored, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_idle_wire_extent_is_preserved_with_explicit_identity() -> None:
    ir = CircuitIR(3, (Instruction("x", (2,)),), dtype="complex128")
    external = to_braket(ir)
    assert tuple(int(qubit) for qubit in external.qubits) == (0, 1, 2)
    restored = from_braket(external)
    assert restored.n_wires == 3
    expected = Circuit.from_ir(ir, dtype=torch.complex128).state()[0]
    actual = Circuit.from_ir(restored, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_unsupported_braket_semantics_fail_closed_with_diagnostics() -> None:
    theta = braket.FreeParameter("theta")
    cases = (
        (braket.Circuit().measure(0), "measurement_not_represented"),
        (braket.Circuit().state_vector(), "result_type_not_represented"),
        (
            braket.Circuit().rx(0, theta),
            "symbolic_parameter_not_supported",
        ),
        (
            braket.Circuit().unitary([0], np.eye(2)),
            "unsupported_instruction",
        ),
    )
    for circuit, issue_code in cases:
        with pytest.raises(BraketConversionError) as caught:
            from_braket(circuit)
        assert issue_code in {issue.code for issue in caught.value.report.issues}

    converted = import_braket(braket.Circuit().measure(0), allow_lossy=True)
    assert converted.report.issues[0].code == "measurement_not_represented"


def test_symbolic_parameters_and_execution_requests_fail_closed() -> None:
    symbolic = CircuitIR(
        1, (Instruction("rx", (0,), {"theta": Parameter("theta")}),), dtype="complex128"
    )
    with pytest.raises(BraketConversionError, match="symbolic_parameter_not_supported"):
        to_braket(symbolic)
    measured = Circuit(1).h(0).to_ir()
    measured = CircuitIR(
        measured.n_wires,
        measured.instructions,
        measurements=(MeasurementNode("sample", (0,), shots=10),),
        dtype="complex128",
    )
    with pytest.raises(BraketConversionError, match="measurement_not_represented"):
        to_braket(measured)


def test_empty_braket_circuit_has_no_invented_wire_extent() -> None:
    with pytest.raises(BraketConversionError, match="at least one referenced qubit"):
        from_braket(braket.Circuit(), allow_lossy=True)
