from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.parameters import Parameter
from flagquantum.interop import get_adapter
from flagquantum.interop.pennylane import (
    PennyLaneConversionError,
    from_pennylane,
    import_pennylane,
    to_pennylane,
)

qml = pytest.importorskip("pennylane")
pytestmark = [pytest.mark.integration, pytest.mark.pennylane]
ROOT = Path(__file__).resolve().parents[1]

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

CONTRACT = tomllib.loads(
    (ROOT / "contracts" / "pennylane-interop-contract.toml").read_text(encoding="utf-8")
)


def _state(script):
    matrix = torch.as_tensor(qml.matrix(script, wire_order=list(script.wires)))
    initial = torch.zeros(matrix.shape[0], dtype=torch.complex128)
    initial[0] = 1
    return matrix.to(torch.complex128) @ initial


def test_quantum_script_round_trip_preserves_complex128_semantics() -> None:
    source = (
        Circuit(3, dtype=torch.complex128)
        .h(0)
        .ry(1, theta=0.231)
        .cx(0, 2)
        .rzz(1, 2, theta=-0.317)
    )
    script = to_pennylane(source.to_ir())
    round_trip = from_pennylane(script)
    assert round_trip.dtype == "complex128"
    assert [item.name for item in round_trip.instructions] == ["h", "ry", "cx", "rzz"]
    expected = Circuit.from_ir(round_trip, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(_state(script), expected, atol=1e-10, rtol=1e-10)
    assert get_adapter("pennylane").name == "pennylane"


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
    script = to_pennylane(ir)
    expected = Circuit.from_ir(ir, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(_state(script), expected, atol=1e-10, rtol=1e-10)


def test_noncontiguous_wires_fail_closed_and_lossy_is_explicit() -> None:
    script = qml.tape.QuantumScript([qml.Hadamard("left"), qml.CNOT(["left", "right"])])
    with pytest.raises(PennyLaneConversionError, match="wire_labels_flattened"):
        from_pennylane(script)
    result = import_pennylane(script, allow_lossy=True)
    assert result.ir.n_wires == 2
    assert result.report.issues[0].code == "wire_labels_flattened"


def test_measurements_shots_and_unsupported_ops_fail_closed() -> None:
    script = qml.tape.QuantumScript(
        [qml.Rot(0.1, 0.2, 0.3, 0)], [qml.state()], shots=10
    )
    with pytest.raises(PennyLaneConversionError) as caught:
        from_pennylane(script)
    codes = {issue.code for issue in caught.value.report.issues}
    assert {
        "finite_shots_not_represented",
        "measurements_not_represented",
        "unsupported_operation",
    } <= codes


def test_symbolic_parameters_and_idle_extent_fail_closed() -> None:
    symbolic = CircuitIR(
        1, (Instruction("rx", (0,), {"theta": Parameter("theta")}),), dtype="complex128"
    )
    with pytest.raises(
        PennyLaneConversionError, match="symbolic_parameter_not_supported"
    ):
        to_pennylane(symbolic)
    idle = CircuitIR(2, (Instruction("x", (0,)),), dtype="complex128")
    with pytest.raises(
        PennyLaneConversionError, match="idle_wire_extent_not_represented"
    ):
        to_pennylane(idle)
