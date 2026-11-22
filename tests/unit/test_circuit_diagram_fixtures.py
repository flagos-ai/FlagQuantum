"""Verify complete unchanged diagram-test support and native owner replacement."""

import json
from collections import Counter
from pathlib import Path

import pytest

from flagquantum.compiler.formats.kmb import decode_kmb, encode_kmb
from flagquantum.compiler.formats.kmb_operations import OPERATION_CODES
from flagquantum.drawer.reversible_phase_diagram import render_circuit_text
from tests.unit.circuit_diagram_fixtures import (
    circuit_with_every_operation,
    diagram_failure,
    expect_circuit_has_text_diagram,
    split_lines,
    undent,
)

pytestmark = pytest.mark.unit


def test_complete_unchanged_helpers() -> None:
    path = Path(__file__).with_name("data") / "circuit_testing.json"
    data: object = json.loads(path.read_text())
    assert isinstance(data, dict)
    for text, expected in data["undent"]:
        assert undent(text) == expected, repr(text)
    for text, expected in data["lines"]:
        assert split_lines(text) == expected, repr(text)
    circuit = circuit_with_every_operation()
    actual = render_circuit_text(circuit)
    assert actual == data["circuit"][1]
    assert encode_kmb(circuit.to_kmx()).hex() == data["circuit"][2]
    for text, expected in data["expect"]:
        failure = diagram_failure("q0: -X-", text)
        if expected == "ok":
            assert failure is None, repr(text)
        else:
            # GoogleTest prepends the original failed expression and values.
            original = bytes.fromhex(expected)
            assert (
                failure is not None
                and original
                == b"Value of: false\n  Actual: false\nExpected: true\n"
                + failure
                + b"\n"
            ), repr(text)
    expect_circuit_has_text_diagram(circuit, "\n" + actual)
    assert sum(map(len, (data["undent"], data["lines"], data["expect"]))) + 1 == 1080


def test_every_operation_factory_keeps_exact_records_and_binary_storage() -> None:
    circuit = circuit_with_every_operation()
    histogram = Counter(record.opcode for record in circuit)
    effective = {
        record.opcode
        + (
            "_IF"
            if record.condition_bit is not None and record.opcode != "PUSH_CONDITION"
            else ""
        )
        for record in circuit
    }
    assert effective == set(OPERATION_CODES)
    assert (
        sum(histogram[name] for name in histogram if name.startswith("DEBUG_PRINT"))
        == 6
    )
    assert histogram["CSWAP"] == 2
    assert histogram["PUSH_CONDITION"] == histogram["POP_CONDITION"] == 2
    assert decode_kmb(encode_kmb(circuit.to_kmx())) == circuit.to_kmx()
    another = circuit_with_every_operation()
    assert another == circuit and another is not circuit


def test_assertion_reuses_renderer_and_preserves_failure_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit = circuit_with_every_operation()
    calls: list[object] = []

    def replacement(value: object) -> str:
        calls.append(value)
        return "q0: -X-"

    monkeypatch.setattr(
        "tests.unit.circuit_diagram_fixtures.render_circuit_text", replacement
    )
    expect_circuit_has_text_diagram(circuit, "\n  q0: -X-\n")
    with pytest.raises(AssertionError) as error:
        expect_circuit_has_text_diagram(circuit, "q0: -Y-")
    assert error.value.args == (diagram_failure("q0: -X-", "q0: -Y-"),)
    assert calls == [circuit, circuit]


@pytest.mark.parametrize("data", [[], {}, {"circuit": []}, {"circuit": [3]}])
def test_invalid_factory_data_refuses(
    data: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "read_text", lambda self: json.dumps(data))
    with pytest.raises(ValueError, match="Invalid complete circuit"):
        circuit_with_every_operation()


def test_original_classical_only_normalization_defect() -> None:
    path = Path(__file__).with_name("data") / "circuit_testing.json"
    data: object = json.loads(path.read_text())
    assert isinstance(data, dict)
    text, expected = data["expect_classical"][0]
    failure = diagram_failure(text, text)
    assert failure is not None
    assert (
        bytes.fromhex(expected)
        == b"Value of: false\n  Actual: false\nExpected: true\n" + failure + b"\n"
    )
    # Source removes a required leading space from an already exact diagram.
    assert undent(text) != text
