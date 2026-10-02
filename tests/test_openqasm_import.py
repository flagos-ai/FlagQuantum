"""OpenQASM import: the round trip, the measurement mapping, and the refusals.

The importer exists so that a program exported by ``Circuit.to_openqasm`` -- or
written by hand in the same subset -- can be read back as a FlagQuantum program
without a second, silently different interpretation of the text. These tests
therefore check the export and import directions against each other rather than
against a hand-written expectation wherever that is possible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler.openqasm import emit_openqasm
from flagquantum.compiler.openqasm_import import (
    OpenQASMImport,
    OpenQASMImportError,
    import_openqasm,
    import_openqasm_to_ir,
)
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "openqasm-import-v1-candidate.json"


def _header(version: float, n_qubits: int, n_bits: int) -> list[str]:
    """Build the header of a source that declares ``version`` verbatim.

    The declared version is written as given rather than normalized, so an
    unsupported version reaches the importer as the source really states it.
    """

    if version >= 3.0:
        return [
            f"OPENQASM {version};",
            'include "stdgates.inc";',
            f"qubit[{n_qubits}] q;",
            f"bit[{n_bits}] c;",
        ]
    return [
        f"OPENQASM {version};",
        'include "qelib1.inc";',
        f"qreg q[{n_qubits}];",
        f"creg c[{n_bits}];",
    ]


def _measure_all(version: float, n_bits: int) -> list[str]:
    if version >= 3.0:
        return [f"c[{bit}] = measure q[{bit}];" for bit in range(n_bits)]
    return [f"measure q[{bit}] -> c[{bit}];" for bit in range(n_bits)]


def _source(body: list[str], *, version: float, n_qubits: int, n_bits: int) -> str:
    return "\n".join(
        [
            *_header(version, n_qubits, n_bits),
            *body,
            *_measure_all(version, n_bits),
        ]
    )


def test_exported_program_imports_back_to_an_equivalent_circuit() -> None:
    """The user journey: export a circuit, import the text, run the result."""

    circuit = fq.Circuit(2).h(0).cx(0, 1).ry(1, theta=0.37)
    for version in (2.0, 3.0):
        text = emit_openqasm(circuit, version=version)
        program = fq.from_openqasm(text)
        rebuilt = program.to_circuit()
        assert program.version == version
        assert program.n_qubits == 2
        assert [instruction.name for instruction in program.instructions] == [
            instruction.name for instruction in circuit.to_ir().instructions
        ]
        assert torch.allclose(
            fq.run(circuit).state, fq.run(rebuilt).state, atol=1e-6
        ), version


def test_imported_program_runs_with_the_public_sampling_api() -> None:
    """An imported program is an ordinary program, so shots work on it."""

    text = emit_openqasm(fq.Circuit(2).h(0).cx(0, 1))
    circuit = fq.from_openqasm(text).to_circuit()
    result = fq.run(circuit, outputs=fq.counts(), shots=256)
    assert set(result.counts[0]) <= {"00", "11"}
    assert sum(result.counts[0].values()) == 256


def test_import_to_ir_returns_canonical_ir_with_the_measurement_node() -> None:
    text = emit_openqasm(fq.Circuit(3).x(2))
    ir = import_openqasm_to_ir(text)
    assert ir.n_wires == 3
    assert [instruction.name for instruction in ir.instructions] == ["x"]
    assert len(ir.measurements) == 1
    assert ir.measurements[0].kind == "counts"
    assert ir.measurements[0].wires == (0, 1, 2)


@pytest.mark.parametrize("opcode", sorted(OPERATOR_SCHEMAS))
@pytest.mark.parametrize("version", [2.0, 3.0])
def test_every_emittable_opcode_survives_the_round_trip(
    opcode: str, version: float
) -> None:
    """Each opcode the compiler can emit must import back to the same program."""

    schema = OPERATOR_SCHEMAS[opcode]
    if schema.channel:
        pytest.skip("channels are not OpenQASM gates")
    n_qubits = max(schema.arity, 2)
    circuit = fq.Circuit(n_qubits)
    wires = tuple(range(schema.arity))
    parameters = {
        name: 0.1 * (index + 1) for index, name in enumerate(schema.parameters)
    }
    getattr(circuit, opcode)(*wires, **parameters)
    text = emit_openqasm(circuit, version=version)

    program = import_openqasm(text)

    assert program.to_openqasm() == text
    assert torch.allclose(
        fq.run(circuit).state, fq.run(program.to_circuit()).state, atol=1e-6
    )


def test_comments_blank_lines_and_spacing_do_not_change_the_program() -> None:
    """Comments and spacing carry no meaning, so they are not a refusal."""

    plain = _source(["h q[0];", "cx q[0], q[1];"], version=2.0, n_qubits=2, n_bits=2)
    noisy = "\n".join(
        [
            "OPENQASM 2.0;",
            "",
            'include "qelib1.inc"; // the standard library',
            "qreg q[2];",
            "",
            "creg c[2];",
            "h   q[0];",
            "cx q[0],q[1]; // entangle",
            "measure q[0]->c[0];",
            "measure q[1] -> c[1];",
            "",
        ]
    )
    assert import_openqasm(noisy).instructions == import_openqasm(plain).instructions


def test_terminal_measurement_records_which_qubit_each_classical_bit_reads() -> None:
    """A permuted measurement block is read in classical-bit order."""

    arrow = _source(["x q[2];"], version=2.0, n_qubits=3, n_bits=2).replace(
        "measure q[0] -> c[0];\nmeasure q[1] -> c[1];",
        "measure q[2] -> c[0];\nmeasure q[0] -> c[1];",
    )
    program = import_openqasm(arrow)
    assert program.measurement_qubits == (2, 0)

    assigned = "\n".join(
        [
            *_header(3.0, 3, 2),
            "x q[2];",
            "c[0] = measure q[2];",
            "c[1] = measure q[0];",
        ]
    )
    assert import_openqasm(assigned).measurement_qubits == (2, 0)


def test_openqasm3_whole_register_measurement_is_read_as_the_identity_mapping() -> None:
    text = "\n".join([*_header(3.0, 3, 3), "x q[1];", "c = measure q;"])
    program = import_openqasm(text)
    assert program.measurement_qubits == (0, 1, 2)
    assert program.whole_register_measurement is True
    assert program.to_openqasm() == text
    assert fq.run(program.to_circuit(), outputs=fq.counts(), shots=32).counts[0] == {
        "010": 32
    }


def test_import_result_reports_the_declared_version_and_width() -> None:
    for version in (2.0, 3.0):
        source = _source(["x q[0];"], version=version, n_qubits=2, n_bits=2)
        program = import_openqasm(source)
        assert isinstance(program, OpenQASMImport)
        assert program.version == version
        assert program.n_qubits == 2
        assert program.source == source
        assert program.to_openqasm() == source


def test_import_result_keeps_the_deprecated_n_wires_alias_warning() -> None:
    program = import_openqasm(_source([], version=3.0, n_qubits=1, n_bits=1))
    with pytest.warns(DeprecationWarning):
        assert program.n_wires == 1


def test_non_text_source_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="must be str"):
        fq.from_openqasm(b"OPENQASM 2.0;")


def test_openqasm3_three_angle_u_gate_is_the_canonical_u3_spelling() -> None:
    text = "\n".join(
        [*_header(3.0, 1, 1), "U(0.4, 0.5, 0.6) q[0];", "c[0] = measure q[0];"]
    )
    program = import_openqasm(text)
    assert [instruction.name for instruction in program.instructions] == ["u3"]
    assert program.whole_register_measurement is False
    assert program.to_openqasm() == text


REFUSALS: tuple[tuple[str, str, str], ...] = (
    (
        "empty_source",
        "",
        "no statement",
    ),
    (
        "unsupported_version",
        "\n".join([*_header(3.1, 1, 1), "x q[0];", "c[0] = measure q[0];"]),
        "not supported",
    ),
    (
        "malformed_statement",
        "\n".join(
            [
                'include "qelib1.inc";',
                "qreg q[1];",
                "creg c[1];",
                "x q[0];",
                "measure q[0] -> c[0];",
            ]
        ),
        "must declare OPENQASM",
    ),
    (
        "unknown_gate",
        _source(["toffoli q[0], q[1];"], version=2.0, n_qubits=2, n_bits=2),
        "not in the canonical OpenQASM subset",
    ),
    (
        "unknown_register",
        _source(["x r[0];"], version=2.0, n_qubits=1, n_bits=1),
        "does not name the declared quantum register",
    ),
    (
        "out_of_range_qubit",
        _source(["x q[4];"], version=2.0, n_qubits=1, n_bits=1),
        "outside q[1]",
    ),
    (
        "invalid_parameters",
        _source(["rx(0.1, 0.2) q[0];"], version=2.0, n_qubits=1, n_bits=1),
        "takes 1 parameter(s), got 2",
    ),
    (
        "unbound_parameter",
        _source(["rx(theta) q[0];"], version=2.0, n_qubits=1, n_bits=1),
        "imports bound programs only",
    ),
    (
        "duplicate_measurement_target",
        "\n".join(
            [
                *_header(2.0, 2, 1),
                "measure q[0] -> c[0];",
                "measure q[1] -> c[0];",
            ]
        ),
        "written by two measurements",
    ),
    (
        "incomplete_measurement",
        _source(["x q[0];"], version=2.0, n_qubits=2, n_bits=2).replace(
            "measure q[1] -> c[1];", ""
        ),
        "partly defined",
    ),
    (
        "not_canonical_text",
        _source(["u1(0.1) q[0];"], version=3.0, n_qubits=1, n_bits=1),
        "which FlagQuantum writes as",
    ),
)


@pytest.mark.parametrize(
    ("issue_code", "source", "message"), REFUSALS, ids=[item[0] for item in REFUSALS]
)
def test_refusals_are_fail_closed_and_name_an_issue_code(
    issue_code: str, source: str, message: str
) -> None:
    """Unsupported text is refused, never imported approximately."""

    with pytest.raises(OpenQASMImportError, match=re.escape(message)) as caught:
        fq.from_openqasm(source)
    assert caught.value.issue_code == issue_code
    assert issue_code in str(caught.value)


def test_declared_refusal_codes_are_closed_and_every_one_is_reachable() -> None:
    """The refusal vocabulary is a closed set, and the contract lists it."""

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    declared = tuple(contract["refusal_issue_codes"])
    assert sorted(declared) == sorted(set(declared))
    assert set(declared) == {issue_code for issue_code, _, _ in REFUSALS}


def test_unsupported_statements_after_the_measurement_block_are_refused() -> None:
    text = _source(["x q[0];"], version=2.0, n_qubits=1, n_bits=1) + "\nx q[0];"
    with pytest.raises(OpenQASMImportError, match="follows the measurement block"):
        fq.from_openqasm(text)


def test_barrier_and_reset_are_refused_because_they_are_not_opcodes() -> None:
    for statement, name in (("barrier q[0];", "barrier"), ("reset q[0];", "reset")):
        with pytest.raises(OpenQASMImportError) as caught:
            fq.from_openqasm(_source([statement], version=2.0, n_qubits=1, n_bits=1))
        assert caught.value.issue_code == "unknown_gate"
        assert name in str(caught.value)


def test_a_second_quantum_register_declaration_is_refused() -> None:
    """A redefinition is refused as a redefinition, not as a bad operand.

    Both dialects are checked, and the second register is never referenced.
    Without the explicit declaration check the refusal that arrives names an
    operand or an index, which describes what the redefinition broke rather
    than the declaration that caused it.
    """

    for version, statement in ((2.0, "qreg r[2];"), (3.0, "qubit[2] r;")):
        text = _source([statement, "x q[0];"], version=version, n_qubits=1, n_bits=1)
        with pytest.raises(OpenQASMImportError) as caught:
            fq.from_openqasm(text)
        assert caught.value.issue_code == "malformed_statement"
        assert "declares a second register" in str(caught.value)
        assert statement in str(caught.value)


def test_a_second_classical_register_declaration_is_refused() -> None:
    for version, statement in ((2.0, "creg d[2];"), (3.0, "bit[2] d;")):
        text = _source([statement, "x q[0];"], version=version, n_qubits=1, n_bits=1)
        with pytest.raises(OpenQASMImportError) as caught:
            fq.from_openqasm(text)
        assert caught.value.issue_code == "malformed_statement"
        assert "declares a second register" in str(caught.value)


def test_redeclaring_the_first_register_is_refused_as_a_redefinition() -> None:
    """The one-register rule is a rule, not a side effect of the size check."""

    text = _source(["qreg q[4];", "x q[3];"], version=2.0, n_qubits=1, n_bits=1)
    with pytest.raises(OpenQASMImportError) as caught:
        fq.from_openqasm(text)
    assert caught.value.issue_code == "malformed_statement"
    assert "declares a second register" in str(caught.value)


def test_arbitrary_power_of_a_gate_is_refused() -> None:
    text = "\n".join([*_header(3.0, 1, 1), "pow(-1) @ x q[0];", "c[0] = measure q[0];"])
    with pytest.raises(OpenQASMImportError) as caught:
        fq.from_openqasm(text)
    assert caught.value.issue_code == "unknown_gate"
    assert "pow(-1) @ sx" in str(caught.value)


def test_sqrt_x_inverse_uses_the_openqasm3_power_spelling() -> None:
    text = "\n".join(
        [*_header(3.0, 1, 1), "pow(-1) @ sx q[0];", "c[0] = measure q[0];"]
    )
    program = import_openqasm(text)
    assert [instruction.name for instruction in program.instructions] == ["sxdg"]
    assert program.to_openqasm() == text
