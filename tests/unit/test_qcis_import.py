"""QCIS syntax, diagnostics, and loss policy without external toolchains."""

from __future__ import annotations

import keyword
import math
import sys

import pytest

from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.contracts import InteropConversionError
from flagquantum.ecosystem.qcis import import_qcis

pytestmark = pytest.mark.unit


def test_sparse_labels_and_measurement_order_survive_serialization() -> None:
    result = import_qcis(
        "# source\r\n X Q100 // flip\r\nI Q2 4\nM Q100 Q2", allow_lossy=True
    )
    ir = result.ir
    assert ir.n_wires == 2
    assert ir.instructions[0].wires == (1,)
    assert ir.measurements[0].wires == (1, 0)
    assert ir.metadata["interop"]["qubits"] == ("Q2", "Q100")
    assert CircuitIR.from_json(ir.to_json()).content_hash == ir.content_hash


@pytest.mark.parametrize(
    "name,opcode,angle",
    [
        ("X2P", "rx", math.pi / 2),
        ("X2M", "rx", -math.pi / 2),
        ("Y2P", "ry", math.pi / 2),
        ("Y2M", "ry", -math.pi / 2),
    ],
)
def test_half_rotations_preserve_the_defined_phase(
    name: str, opcode: str, angle: float
) -> None:
    instruction = import_qcis(f"{name} Q1").ir.instructions[0]
    assert (instruction.name, instruction.params["theta"]) == (opcode, angle)


@pytest.mark.parametrize(
    "text", ["+2.3e-1", "-.23", ".23", "2.", "-7.5", "0", "01", "-01"]
)
def test_bound_numeric_angles(text: str) -> None:
    assert import_qcis(f"RZ Q0 {text}").ir.instructions[0].params["theta"] == float(
        text
    )


@pytest.mark.parametrize(
    "source,code",
    [
        ("# empty", "empty_source"),
        ("X Q01", "invalid_qubit"),
        ("X Q-1", "invalid_qubit"),
        ("X q0", "invalid_qubit"),
        ("CZ Q2 Q2", "duplicate_qubit"),
        ("X Q0 Q1", "invalid_arity"),
        ("RZ Q0", "invalid_arity"),
        ("M", "invalid_qubit"),
        ("M Q0 Q0", "duplicate_qubit"),
        ("M Q0\nM Q0", "duplicate_measurement"),
        ("M Q0\nX Q1", "measurement_not_terminal"),
        ("RZ Q0 theta/phi", "invalid_parameter"),
        ("RZ Q0 theta**2", "invalid_parameter"),
        ("RZ Q0 __import__('os')", "invalid_parameter"),
        ("RZ Q0 NaN", "invalid_parameter"),
        ("RZ Q0 1e999", "invalid_parameter"),
        ("I Q0 -3", "invalid_duration"),
        ("I Q0 1.5", "invalid_duration"),
        ("RESET Q0", "unsupported_operation"),
        ("PULSE Q0 1", "unsupported_operation"),
        ("RXY Q0 1", "invalid_arity"),
        ("X Q0;", "invalid_qubit"),
    ],
)
@pytest.mark.parametrize("allow_lossy", [False, True])
def test_invalid_programs_fail_even_with_loss_permission(
    source: str, code: str, allow_lossy: bool
) -> None:
    with pytest.raises(InteropConversionError) as raised:
        import_qcis(source, allow_lossy=allow_lossy)
    assert raised.value.report.blockers[0].code == code


def test_diagnostic_uses_physical_source_line_and_operation() -> None:
    with pytest.raises(InteropConversionError, match=r"line 4 \(RZ\)"):
        import_qcis("\n# comment\n\n RZ Q5 sin(theta)")


def test_timing_is_never_silently_discarded() -> None:
    source = "I Q17 60\nB Q17 Q2\nM Q17\nB Q17 Q2"
    with pytest.raises(InteropConversionError, match="allow_lossy") as raised:
        import_qcis(source)
    assert not raised.value.report.lossless
    result = import_qcis(source, allow_lossy=True)
    assert not result.report.lossless and not result.report.blockers
    assert len(result.report.issues) == 3
    assert result.ir.instructions[0].name == "i"
    assert (
        result.ir.metadata["interop"]["timing_annotations"][0]["duration_ticks"] == 60
    )


def test_terminal_measurements_can_be_split_over_lines() -> None:
    ir = import_qcis("X Q9\nM Q9\nM Q1 Q3").ir
    assert ir.measurements[0].wires == (2, 0, 1)


def test_argument_types_are_not_coerced() -> None:
    with pytest.raises(TypeError, match="source"):
        import_qcis(1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="allow_lossy"):
        import_qcis("X Q0", allow_lossy=1)  # type: ignore[arg-type]


def test_oversized_integer_operands_keep_domain_diagnostics() -> None:
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(640)
        huge = "9" * 641
        for text, code, operation in [
            (f"X Q{huge}", "invalid_qubit", "X"),
            (f"I Q7 {huge}", "invalid_duration", "I"),
        ]:
            with pytest.raises(
                InteropConversionError, match=r"line 2 .*integer operand"
            ) as raised:
                import_qcis("# comment\n" + text, allow_lossy=True)
            issue = raised.value.report.blockers[0]
            assert issue.code == code
            assert issue.operation_name == operation
            assert "999999" in issue.message
    finally:
        sys.set_int_max_str_digits(previous)


@pytest.mark.parametrize(
    "angle",
    [
        "1/0",
        "theta/(2-2)",
        "1e308*2",
        "theta/1e-320",
        "sin(theta)",
        "theta.real",
        "theta[0]",
        "[theta]",
        "True",
        "False",
        "None",
        "_qcis_parameter_0",
        "(lambda+_qcis_parameter_0)",
        "lambda.__class__",
        "lambda(0)",
        "(lambda:0)",
        "(theta if lambda else phi)",
        "0x10",
        "1_000",
        "𝒙",
        "(",
        "(theta",
        "theta)",
        "(theta if x else phi)",
        "theta@phi",
        "(" * 40 + "theta" + ")" * 40 + "+" + "theta+" * 130 + "theta",
        "-" * 34 + "theta",
        "a" * 4097,
    ],
)
def test_unsafe_or_unrepresentable_angles_keep_domain_context(angle: str) -> None:
    with pytest.raises(InteropConversionError, match=r"line 2 \(RXY\)") as raised:
        import_qcis(f"# input\nRXY Q7 0 {angle}", allow_lossy=True)
    assert raised.value.report.blockers[0].code == "invalid_parameter"
    assert raised.value.report.blockers[0].operation_name == "RXY"


def test_parameter_expression_serialization_and_binding() -> None:
    from flagquantum.core.parameters import (
        bind_parameter_value,
        parameter_names_in_value,
    )

    converted = import_qcis("RXY Q7 (phi + pi/2) (2*theta-theta*phi/2)\nRZ Q7 +theta")
    assert converted.report.lossless
    ir = CircuitIR.from_json(converted.ir.to_json())
    assert ir.content_hash == converted.ir.content_hash
    assert parameter_names_in_value(tuple(i.params for i in ir.instructions)) == (
        "phi",
        "theta",
    )
    angles = [
        bind_parameter_value(i.params["theta"], {"phi": 0.3, "theta": 0.7})
        for i in ir.instructions
    ]
    assert angles == pytest.approx(
        [-(0.3 + math.pi / 2), 1.4 - 0.7 * 0.3 / 2, 0.3 + math.pi / 2, 0.7]
    )
    with pytest.raises(KeyError, match="phi"):
        bind_parameter_value(ir.instructions[0].params["theta"], {"theta": 0.7})


def test_constant_expressions_are_folded_to_numeric_ir() -> None:
    ir = import_qcis("RXY Q7 (-pi / 3) (+2.3e-1 + 0.4/2)").ir
    assert [i.params["theta"] for i in ir.instructions] == pytest.approx(
        [math.pi / 3, 0.43, -math.pi / 3]
    )


@pytest.mark.parametrize(
    "name", [name for name in keyword.kwlist if name not in {"True", "False", "None"}]
)
def test_python_keywords_are_qcis_parameter_names(name: str) -> None:
    from flagquantum.core.parameters import (
        bind_parameter_value,
        parameter_names_in_value,
    )

    converted = import_qcis(f"RZ Q7 {name}\nRXY Q7 ({name}/2) (-{name}+pi)")
    ir = CircuitIR.from_json(converted.ir.to_json())
    assert converted.report.lossless
    assert parameter_names_in_value(tuple(i.params for i in ir.instructions)) == (name,)
    angles = [
        bind_parameter_value(i.params["theta"], {name: 0.6}) for i in ir.instructions
    ]
    assert angles == pytest.approx([0.6, -0.3, math.pi - 0.6, 0.3])


def test_keyword_and_ordinary_names_keep_distinct_bindings() -> None:
    from flagquantum.core.parameters import (
        bind_parameter_value,
        parameter_names_in_value,
    )

    ir = import_qcis("RZ Q0 (lambda+class+theta+lambda_1+1e-3)").ir
    angle = ir.instructions[0].params["theta"]
    assert parameter_names_in_value(angle) == ("class", "lambda", "lambda_1", "theta")
    assert bind_parameter_value(
        angle, {"lambda": 0.1, "class": 0.2, "theta": 0.3, "lambda_1": 0.4}
    ) == pytest.approx(1.001)
