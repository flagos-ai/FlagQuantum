from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
import torch

from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.core.parameters import Parameter, ParameterExpression
from flagquantum.ecosystem import get_adapter
from flagquantum.ecosystem.cirq import (
    CirqConversionError,
    from_cirq,
    import_cirq,
    to_cirq,
)

cirq = pytest.importorskip("cirq")
np = import_module("numpy")
sympy = import_module("sympy")
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
    q0, q1 = cirq.LineQubit.range(2)
    cases = (
        (
            cirq.Circuit(cirq.X(q0), cirq.global_phase_operation(1j)),
            "global_phase_not_represented",
        ),
        (cirq.Circuit(cirq.ISWAP(q0, q1)), "unsupported_operation"),
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


def test_terminal_measurements_round_trip_keys_order_and_classical_bits() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    source = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q0)]),
            cirq.Moment([cirq.CNOT(q0, q1)]),
            cirq.Moment(
                [
                    cirq.measure(q2, q0, key="readout"),
                    cirq.measure(q1, key="aux"),
                ]
            ),
        ]
    )

    imported = from_cirq(source)
    measurements = [
        instruction
        for instruction in imported.instructions
        if instruction.name == "measure"
    ]
    assert [instruction.wires for instruction in measurements] == [(2,), (0,), (1,)]
    assert [
        instruction.metadata["measurement_key"] for instruction in measurements
    ] == [
        "readout",
        "readout",
        "aux",
    ]
    assert [instruction.metadata["classical_bit"] for instruction in measurements] == [
        0,
        1,
        2,
    ]
    assert all(
        instruction.metadata["is_dynamic"] is True for instruction in measurements
    )
    assert not any(
        type(value).__module__.startswith(("cirq", "sympy"))
        for instruction in measurements
        for value in instruction.metadata.values()
    )

    exported = to_cirq(imported)
    exported_measurements = [
        operation
        for operation in exported.all_operations()
        if isinstance(operation.gate, cirq.MeasurementGate)
    ]
    assert [operation.gate.key for operation in exported_measurements] == [
        "readout",
        "aux",
    ]
    assert [operation.qubits for operation in exported_measurements] == [
        (q2, q0),
        (q1,),
    ]
    restored = from_cirq(exported)
    assert restored.instructions == imported.instructions


@pytest.mark.parametrize(
    ("circuit_factory", "issue_code"),
    [
        (
            lambda q0, q1: cirq.Circuit(cirq.measure(q0, key="early"), cirq.X(q1)),
            "measurement_not_terminal",
        ),
        (
            lambda q0, _q1: cirq.Circuit(
                cirq.measure(q0, key="m", invert_mask=(True,))
            ),
            "measurement_invert_mask_not_representable",
        ),
        (
            lambda q0, _q1: cirq.Circuit(
                cirq.MeasurementGate(
                    1,
                    key="m",
                    confusion_map={(0,): np.eye(2)},
                ).on(q0)
            ),
            "measurement_confusion_map_not_representable",
        ),
        (
            lambda q0, q1: cirq.Circuit(
                [
                    cirq.Moment([cirq.measure(q0, key="same")]),
                    cirq.Moment([cirq.measure(q1, key="same")]),
                ]
            ),
            "duplicate_measurement_key",
        ),
        (
            lambda q0, _q1: cirq.Circuit(cirq.MeasurementGate(1, key="").on(q0)),
            "measurement_key_empty",
        ),
    ],
)
def test_unsupported_measurement_forms_fail_closed(
    circuit_factory, issue_code: str
) -> None:
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(CirqConversionError) as caught:
        from_cirq(circuit_factory(q0, q1))
    assert issue_code in {issue.code for issue in caught.value.report.issues}


def test_non_qubit_measurement_fails_closed() -> None:
    qid = cirq.LineQid(0, dimension=3)
    with pytest.raises(CirqConversionError) as caught:
        from_cirq(cirq.Circuit(cirq.measure(qid, key="trit")))
    assert "measurement_qid_dimension_not_representable" in {
        issue.code for issue in caught.value.report.issues
    }


def test_owned_measurement_metadata_fails_closed_when_not_representable() -> None:
    malformed = CircuitIR(
        2,
        (
            Instruction("x", (0,)),
            Instruction(
                "measure",
                (0, 1),
                metadata={
                    "is_dynamic": True,
                    "classical_bit": 1,
                    "measurement_key": "m",
                },
            ),
        ),
        dtype="complex128",
    )
    with pytest.raises(CirqConversionError) as caught:
        to_cirq(malformed)
    assert "measurement_not_representable" in {
        issue.code for issue in caught.value.report.issues
    }


@pytest.mark.parametrize(
    ("instructions", "issue_code"),
    [
        (
            (
                Instruction("x", (0,)),
                Instruction(
                    "measure",
                    (0,),
                    metadata={
                        "is_dynamic": True,
                        "classical_bit": 0,
                        "measurement_key": "m",
                    },
                ),
                Instruction("x", (1,)),
            ),
            "measurement_not_terminal",
        ),
        (
            tuple(
                Instruction(
                    "measure",
                    (wire,),
                    metadata={
                        "is_dynamic": True,
                        "classical_bit": classical_bit,
                        "measurement_key": key,
                    },
                )
                for classical_bit, (wire, key) in enumerate(
                    ((0, "first"), (1, "second"), (0, "first"))
                )
            ),
            "duplicate_measurement_key",
        ),
    ],
)
def test_owned_terminal_and_unique_key_requirements_fail_closed(
    instructions: tuple[Instruction, ...], issue_code: str
) -> None:
    with pytest.raises(CirqConversionError) as caught:
        to_cirq(CircuitIR(2, instructions, dtype="complex128"))
    assert issue_code in {issue.code for issue in caught.value.report.issues}


def test_idle_extent_and_execution_requests_fail_closed() -> None:
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


def test_symbolic_rotation_parameters_round_trip_and_bind_equivalently() -> None:
    theta, phi = sympy.symbols("theta phi")
    q0 = cirq.LineQubit(0)
    source = cirq.Circuit(
        cirq.rx(theta + 2 * phi)(q0),
        cirq.ry(2 * theta - phi)(q0),
        cirq.rz(-theta)(q0),
    )

    imported = from_cirq(source)
    owned = Circuit.from_ir(imported, dtype=torch.complex128)
    assert owned.parameter_names == ("phi", "theta")

    def contains_sympy(value) -> bool:
        if isinstance(value, sympy.Basic):
            return True
        if isinstance(value, ParameterExpression):
            return any(contains_sympy(arg) for arg in value.args)
        return False

    assert not any(
        contains_sympy(value)
        for instruction in imported.instructions
        for value in instruction.params.values()
    )

    exported = to_cirq(imported)
    assert {str(symbol) for symbol in cirq.parameter_symbols(exported)} == {
        "phi",
        "theta",
    }
    for assignment in (
        {"theta": 0.23, "phi": -0.41},
        {"theta": -1.17, "phi": 0.09},
    ):
        native = cirq.resolve_parameters(source, assignment)
        restored = cirq.resolve_parameters(exported, assignment)
        expected = owned.bind_parameters(assignment).state()[0]
        torch.testing.assert_close(
            _cirq_state(native, 1), expected, atol=1e-10, rtol=1e-10
        )
        torch.testing.assert_close(
            _cirq_state(restored, 1), expected, atol=1e-10, rtol=1e-10
        )

    owned_subtraction = CircuitIR(
        1,
        (
            Instruction(
                "rx",
                (0,),
                {"theta": Parameter("theta") - Parameter("phi")},
            ),
        ),
        dtype="complex128",
    )
    exported_subtraction = to_cirq(owned_subtraction)
    resolved_subtraction = cirq.resolve_parameters(
        exported_subtraction, {"theta": 0.7, "phi": 0.2}
    )
    torch.testing.assert_close(
        _cirq_state(resolved_subtraction, 1),
        Circuit.from_ir(owned_subtraction, dtype=torch.complex128)
        .bind_parameters({"theta": 0.7, "phi": 0.2})
        .state()[0],
        atol=1e-10,
        rtol=1e-10,
    )


def test_symbolic_rzz_parameter_round_trips_and_binds_equivalently() -> None:
    theta = sympy.Symbol("theta")
    qubits = cirq.LineQubit.range(2)
    source = cirq.Circuit(
        cirq.ZZPowGate(exponent=theta / sympy.pi, global_shift=-0.5).on(*qubits)
    )

    imported = from_cirq(source)
    exported = to_cirq(imported)
    assignment = {"theta": -0.317}
    expected = Circuit.from_ir(imported, dtype=torch.complex128).bind_parameters(
        assignment
    )

    torch.testing.assert_close(
        _cirq_state(cirq.resolve_parameters(exported, assignment), 2),
        expected.state()[0],
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize(
    "expression",
    [
        lambda symbol: sympy.sin(symbol),
        lambda symbol: symbol**2,
        lambda symbol: 1 / symbol,
        lambda symbol: symbol + sympy.I,
    ],
)
def test_unsupported_cirq_parameter_expressions_fail_closed(expression) -> None:
    theta = sympy.Symbol("theta")
    circuit = cirq.Circuit(cirq.rx(expression(theta))(cirq.LineQubit(0)))
    with pytest.raises(CirqConversionError) as caught:
        from_cirq(circuit)
    assert {issue.code for issue in caught.value.report.issues} == {
        "unsupported_parameter_expression"
    }


def test_symbolic_nonrotation_and_malformed_owned_expression_fail_closed() -> None:
    theta = sympy.Symbol("theta")
    with pytest.raises(CirqConversionError, match="unsupported_parameter_expression"):
        from_cirq(cirq.Circuit((cirq.X**theta)(cirq.LineQubit(0))))

    malformed = CircuitIR(
        1,
        (
            Instruction(
                "rx",
                (0,),
                {"theta": ParameterExpression("power", (Parameter("theta"), 2.0))},
            ),
        ),
        dtype="complex128",
    )
    with pytest.raises(CirqConversionError, match="unsupported_parameter_expression"):
        to_cirq(malformed)


def test_distinct_cirq_symbols_cannot_collapse_to_one_owned_name() -> None:
    plain = sympy.Symbol("theta")
    positive = sympy.Symbol("theta", positive=True)
    circuit = cirq.Circuit(cirq.rx(plain + positive)(cirq.LineQubit(0)))
    with pytest.raises(CirqConversionError) as caught:
        from_cirq(circuit)
    assert caught.value.report.issues[0].code == "unsupported_parameter_expression"
    assert "symbol" in caught.value.report.issues[0].message
