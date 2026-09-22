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
from flagquantum.ecosystem.cirq import (
    CirqConversionError,
    from_cirq,
    import_cirq,
    to_cirq,
)

cirq = pytest.importorskip("cirq")
np = import_module("numpy")
pytestmark = [pytest.mark.integration, pytest.mark.cirq]
ROOT = Path(__file__).resolve().parents[1]

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

CONTRACT = tomllib.loads(
    (ROOT / "contracts" / "cirq-interop-contract.toml").read_text(encoding="utf-8")
)


def _cirq_state(circuit: cirq.Circuit, n_wires: int) -> torch.Tensor:
    result = cirq.Simulator(dtype=np.complex128).simulate(
        circuit, qubit_order=cirq.LineQubit.range(n_wires)
    )
    return torch.as_tensor(result.final_state_vector.copy())


def test_cirq_round_trip_preserves_complex128_semantics_and_order() -> None:
    source = (
        Circuit(3, dtype=torch.complex128)
        .x(2)
        .h(0)
        .ry(1, theta=0.231)
        .cx(0, 2)
        .cz(2, 1)
    )
    external = to_cirq(source.to_ir())
    round_trip = from_cirq(external)
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
        _cirq_state(external, 3), expected, atol=1e-10, rtol=1e-10
    )
    assert get_adapter("cirq").name == "cirq"


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
    external = to_cirq(ir)
    restored = from_cirq(external)
    expected = Circuit.from_ir(ir, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(
        _cirq_state(external, schema.arity), expected, atol=1e-10, rtol=1e-10
    )
    actual = Circuit.from_ir(restored, dtype=torch.complex128).state()[0]
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_parallel_moment_and_noncanonical_qubits_require_explicit_loss() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    parallel = cirq.Circuit(cirq.Moment([cirq.X(q0), cirq.Y(q1)]))
    with pytest.raises(CirqConversionError, match="moment_structure_flattened"):
        from_cirq(parallel)
    converted = import_cirq(parallel, allow_lossy=True)
    assert converted.report.issues[0].code == "moment_structure_flattened"

    noncontiguous = cirq.Circuit(cirq.X(cirq.LineQubit(3)))
    with pytest.raises(CirqConversionError, match="non_contiguous_line_qubits"):
        from_cirq(noncontiguous)
    named = cirq.Circuit(cirq.X(cirq.NamedQubit("q")))
    with pytest.raises(CirqConversionError, match="non_line_qubit"):
        from_cirq(named)


def test_unsupported_cirq_semantics_fail_closed_with_diagnostics() -> None:
    import sympy

    q0, q1 = cirq.LineQubit.range(2)
    cases = (
        (cirq.Circuit(cirq.measure(q0)), "measurement_not_represented"),
        (
            cirq.Circuit(cirq.X(q0), cirq.global_phase_operation(1j)),
            "global_phase_not_represented",
        ),
        (cirq.Circuit(cirq.ISWAP(q0, q1)), "unsupported_operation"),
        (
            cirq.Circuit(cirq.rx(sympy.Symbol("theta"))(q0)),
            "symbolic_parameter_not_supported",
        ),
        (cirq.Circuit(cirq.X(q0).with_tags("tag")), "unsupported_operation"),
        (
            cirq.Circuit(cirq.X(q1).with_classical_controls("m")),
            "unsupported_operation",
        ),
    )
    for circuit, issue_code in cases:
        with pytest.raises(CirqConversionError) as caught:
            from_cirq(circuit)
        assert issue_code in {issue.code for issue in caught.value.report.issues}


def test_symbolic_parameters_idle_extent_and_execution_requests_fail_closed() -> None:
    symbolic = CircuitIR(
        1, (Instruction("rx", (0,), {"theta": Parameter("theta")}),), dtype="complex128"
    )
    with pytest.raises(CirqConversionError, match="symbolic_parameter_not_supported"):
        to_cirq(symbolic)
    idle = CircuitIR(2, (Instruction("x", (0,)),), dtype="complex128")
    with pytest.raises(CirqConversionError, match="idle_wire_extent_not_represented"):
        to_cirq(idle)
    measured = Circuit(1).h(0).to_ir()
    measured = CircuitIR(
        measured.n_wires,
        measured.instructions,
        measurements=(MeasurementNode("sample", (0,), shots=10),),
        dtype="complex128",
    )
    with pytest.raises(CirqConversionError, match="measurement_not_represented"):
        to_cirq(measured)
