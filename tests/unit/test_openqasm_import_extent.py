"""What the OpenQASM importer reaches, and where it stops.

The importer is the inverse of the emitter over one canonical subset, and that
subset is a property of the pair rather than of either half alone: the names the
importer accepts in a version lane are exactly the names the emitter writes in
that same lane. This module measures that identity, counts the vocabulary it
runs over, and pins the refusals that hold the boundary up, so the pair's reach
is a number and a partition rather than a sentence.

Three of the numbers are not obvious from either table:

* the shared spelling tables declare twenty-eight spellings that reach
  twenty-seven native opcodes, because ``cp`` and ``cu1`` are two spellings of
  ``cphase`` and nothing else collides;
* the two lanes write different names for the same program -- OpenQASM 2 writes
  ``cu1``, ``u1`` and ``u2`` where OpenQASM 3 writes ``cp``, ``p`` and
  ``U(...)``, and OpenQASM 2 has no ``sx`` at all, so it lowers ``sx`` to the
  identity ``h; s; h`` -- so each lane's accepted set is a different set;
* one native opcode is reachable without being a spelling in either table:
  ``sxdg`` enters as OpenQASM 3's ``pow(-1) @ sx``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flagquantum.compiler.openqasm import emit_openqasm
from flagquantum.compiler.openqasm_gates import FIXED_GATES, PARAMETERIZED_GATES
from flagquantum.core.ir import get_operator_schema

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]

LANES = (2.0, 3.0)

# The two tables are the emitter's and the importer's shared copy of each gate
# spelling. Reading them rather than restating them is what makes the counts
# below a measurement of the shipped pair.
SPELLINGS: dict[str, str | tuple[str, tuple[str, ...]]] = {
    **FIXED_GATES,
    **PARAMETERIZED_GATES,
}

PARAMETER_VALUES = {"theta": 0.7, "phi": 0.3, "lbd": 1.1}

# OpenQASM 3 writes the inverse of a gate as ``pow(-1) @ gate``. The emitter
# uses the form for exactly one opcode, ``sxdg``, and the importer accepts the
# same one form.
INVERSE_FORM = "pow(-1) @ sx"
INVERSE_OPCODE = "sxdg"

# A complete program in each lane: the two register declarations the emitter
# writes, and the terminal measurement block that covers the whole register.
HEADER = {
    2.0: 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\ncreg c[3];\n',
    3.0: 'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\nbit[3] c;\n',
}
TERMINAL = {
    2.0: "measure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\n",
    3.0: "c = measure q;\n",
}

# Names that exist in OpenQASM and in CUDA-Q's front end but that no lane of
# this pair reads. They are outside the shared tables rather than misspelled.
FOREIGN_NAMES = ("ch", "rzz(0.4)", "cu3(0.1, 0.2, 0.3)", "u(0.1, 0.2, 0.3)")

# Statements the emitter never writes, so a lane refuses them by name.
UNWRITTEN_STATEMENTS = (
    "reset q[0];",
    "barrier q[0], q[1];",
    "delay[10ns] q[0];",
    "box { h q[0]; }",
    "gphase(0.3);",
    "input float t;",
    "opaque my a;",
    "gate my a { h a; }",
    "for i in [0:2] { h q[i]; }",
)


def _declaration(spelling: str) -> tuple[str, dict[str, float] | None]:
    """Return the native opcode and parameter values for one spelling."""

    spec = SPELLINGS[spelling]
    if isinstance(spec, tuple):
        opcode, names = spec
        return opcode, {name: PARAMETER_VALUES[name] for name in names}
    return spec, None


def _circuit(opcode: str, parameters: dict[str, float] | None = None):
    """A three-wire circuit holding one instance of one native opcode."""

    import flagquantum as fq

    arity = get_operator_schema(opcode).arity
    return fq.Circuit(3).gate(opcode, tuple(range(arity)), params=parameters)


def _statement(spelling: str, version: float) -> str:
    """The spelling written by hand, in the text form that lane writes it."""

    opcode, parameters = _declaration(spelling)
    arity = get_operator_schema(opcode).arity
    arguments = ""
    if parameters is not None:
        arguments = "(" + ", ".join(str(v) for v in parameters.values()) + ")"
    operands = ", ".join(f"q[{index}]" for index in range(arity))
    return f"{spelling}{arguments} {operands};"


def _emitted(spelling: str, version: float) -> str:
    """The text one declared spelling produces in one lane."""

    return emit_openqasm(_circuit(*_declaration(spelling)).to_ir(), version=version)


def _heads(text: str) -> set[str]:
    """The names a piece of emitted text writes as a statement head."""

    reserved = {"OPENQASM", "include", "qubit", "bit", "qreg", "creg"}
    heads = set()
    for line in text.splitlines():
        if not line.endswith(";") or line.split()[0] in reserved:
            continue
        heads.add(line.split("(")[0].split()[0].rstrip(";"))
    return heads


def _written(version: float) -> set[str]:
    """The spellings this lane's emitter writes by name."""

    return {
        spelling
        for spelling in SPELLINGS
        if spelling in _heads(_emitted(spelling, version))
    }


def _issue_code(source: str) -> str | None:
    """The refusal code a source raises, or ``None`` when it is accepted.

    Every refusal carries an ``issue_code``. A failure that reached this helper
    without one would be a defect in the importer rather than a boundary fact,
    so the exception's own class name is returned and the assertion that reads
    it fails loudly instead of passing by accident.
    """

    import flagquantum as fq

    try:
        fq.from_openqasm(source)
    except Exception as error:
        return getattr(error, "issue_code", type(error).__name__)
    return None


def _accepted(version: float) -> set[str]:
    """The spellings this lane's importer accepts, written by hand."""

    accepted = set()
    for spelling in SPELLINGS:
        source = (
            HEADER[version] + _statement(spelling, version) + "\n" + TERMINAL[version]
        )
        if _issue_code(source) is None:
            accepted.add(spelling)
    return accepted


def _program(*statements: str, version: float = 3.0) -> str:
    return HEADER[version] + "\n".join(statements) + "\n" + TERMINAL[version]


def test_the_importer_is_a_stable_export_with_an_authorized_contract() -> None:
    """The row this module reconciles claimed no OpenQASM text importer exists."""

    snapshot = json.loads(
        (ROOT / "docs" / "public_api_v1.json").read_text(encoding="utf-8")
    )
    assert "from_openqasm" in snapshot["stable_exports"]
    assert "from_openqasm" in getattr(
        __import__("flagquantum"), "__all__"
    ), "a stable export must be exported"

    contract = json.loads(
        (ROOT / "contracts" / "openqasm-import-v1-candidate.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["implementation_authorized"] is True
    assert tuple(contract["supported_versions"]) == LANES
    assert (ROOT / contract["approval"]["approval_record"]).is_file()


def test_the_spelling_table_is_the_importer_s_whole_vocabulary() -> None:
    """A spelling the table does not declare has no reader to reach."""

    assert len(SPELLINGS) == 28
    opcodes = {_declaration(spelling)[0] for spelling in SPELLINGS}
    assert len(opcodes) == 27, "one native opcode carries two spellings"
    sharing = {
        opcode: {
            spelling for spelling in SPELLINGS if _declaration(spelling)[0] == opcode
        }
        for opcode in opcodes
    }
    assert {opcode: names for opcode, names in sharing.items() if len(names) > 1} == {
        "cphase": {"cp", "cu1"}
    }


def test_each_lane_accepts_exactly_the_names_that_lane_writes() -> None:
    """The subset is a property of the pair, so both directions must agree.

    Counting either side alone would miss the failure that matters: a spelling
    the emitter writes and the importer refuses makes the round trip lossy, and
    a spelling the importer accepts and the emitter never writes makes the
    importer wider than the pair's own text.
    """

    measured = {}
    for version in LANES:
        written, accepted = _written(version), _accepted(version)
        assert (
            not written - accepted
        ), f"lane {version} writes {sorted(written - accepted)} and refuses them"
        assert (
            not accepted - written
        ), f"lane {version} accepts {sorted(accepted - written)} and never writes them"
        measured[version] = len(accepted)

    assert measured == {2.0: 25, 3.0: 24}


def test_the_two_lanes_read_different_names_for_the_same_opcodes() -> None:
    """Each lane is canonical against itself, not against the other lane."""

    only_two = _accepted(2.0) - _accepted(3.0)
    only_three = _accepted(3.0) - _accepted(2.0)
    assert only_two == {"cu1", "u1", "u2", "u3"}
    assert only_three == {"cp", "p", "sx"}


@pytest.mark.parametrize("version", LANES)
@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_every_declared_spelling_round_trips_byte_exactly(
    spelling: str, version: float
) -> None:
    """Reading the emitter's own text back must reproduce that text."""

    import flagquantum as fq

    text = _emitted(spelling, version)
    reopened = fq.from_openqasm(text)
    assert emit_openqasm(reopened.to_circuit().to_ir(), version=version) == text


@pytest.mark.parametrize(
    ("native", "lowering"),
    [("sx", ("h", "s", "h")), ("sxdg", ("h", "sdg", "h"))],
)
def test_the_openqasm_2_lane_lowers_the_square_root_of_x_exactly(
    native: str, lowering: tuple[str, ...]
) -> None:
    """qelib1 has no ``sx``, so ``sx`` may not be emitted under another name.

    ``H S H`` is the square root of ``X`` and ``H Sdg H`` is its inverse, so the
    OpenQASM 2 body is the three gates that define the operation rather than a
    stand-in for it. The substitution is checked rather than assumed: a lane
    that wrote a single ``h`` would still round-trip against itself, because the
    canonical-text guard compares the source to the program it denotes.
    """

    import torch

    import flagquantum as fq
    from flagquantum.simulation.unitary import get_unitary

    body = emit_openqasm(_circuit(native).to_ir(), version=2.0).splitlines()[
        4 : 4 + len(lowering)
    ]
    assert tuple(line.split()[0] for line in body) == lowering

    def unitary(sequence: tuple[str, ...]):
        circuit = fq.Circuit(1)
        for gate in sequence:
            circuit = circuit.gate(gate, (0,))
        return get_unitary(circuit.to_ir())

    difference = unitary((native,)) - unitary(lowering)
    assert float(difference.abs().max()) < 1e-6, "complex64 storage precision"
    assert torch.equal(unitary(("sx",)) @ unitary(("sx",)), unitary(("x",)))


def test_the_openqasm_3_lane_writes_the_inverse_of_the_square_root_as_a_power() -> None:
    text = emit_openqasm(_circuit(INVERSE_OPCODE).to_ir(), version=3.0)
    assert text.splitlines()[4] == f"{INVERSE_FORM} q[0];"


@pytest.mark.parametrize("version", LANES)
def test_the_only_inverse_form_is_the_one_the_emitter_writes(version: float) -> None:
    """``pow(-1) @ sx`` is the one power form, and it is not a table spelling."""

    import flagquantum as fq

    assert INVERSE_FORM.split()[2] in SPELLINGS
    assert INVERSE_OPCODE not in {_declaration(s)[0] for s in SPELLINGS}

    text = _program(INVERSE_FORM + " q[0];", version=version)
    if version == 2.0:
        assert _issue_code(text) == "not_canonical_text"
        return
    imported = fq.from_openqasm(text)
    assert [instruction.name for instruction in imported.instructions] == [
        INVERSE_OPCODE
    ]


@pytest.mark.parametrize(
    "statement",
    ["pow(2) @ sx q[0];", "pow(-1) @ x q[0];"],
)
def test_a_power_form_the_emitter_does_not_write_is_refused(statement: str) -> None:
    """A reader that evaluated any power would import a different program."""

    assert _issue_code(_program(statement)) in {"malformed_statement", "unknown_gate"}


@pytest.mark.parametrize("name", FOREIGN_NAMES)
def test_a_name_outside_the_table_is_refused_as_an_unknown_gate(name: str) -> None:
    assert _issue_code(_program(f"{name} q[0];")) == "unknown_gate"


@pytest.mark.parametrize(
    "statement",
    ["if (c[0] == 1) x q[0];", "if (c == 1) x q[0];"],
)
def test_a_conditional_is_refused_rather_than_skipped(statement: str) -> None:
    """The condition is read as a parameter list, which is the honest failure.

    A reader that dropped the branch would import a different program, and one
    that bound the classical bits as parameters would import a different one
    again. It does neither: the branch is refused, and the code says which
    grammar rule refused it.
    """

    assert _issue_code(_program(statement)) == "unbound_parameter"


def test_the_openqasm_2_whole_register_measure_is_malformed() -> None:
    """OpenQASM 2 has two measurement forms and this pair reads one of them.

    The form it reads is the one the emitter writes: one statement per bit.
    """
    assert _issue_code(_program("measure q -> c;", version=2.0)) == (
        "malformed_statement"
    )


def test_a_repeated_measurement_target_is_refused_rather_than_overwritten() -> None:
    """Measuring one bit twice is a different program, not a redundant one."""

    repeated = _program("measure q[0] -> c[0];", version=2.0)
    assert _issue_code(repeated) == "duplicate_measurement_target"


@pytest.mark.parametrize("statement", UNWRITTEN_STATEMENTS)
def test_a_statement_the_emitter_never_writes_is_refused_by_name(
    statement: str,
) -> None:
    assert _issue_code(_program(statement)) in {
        "malformed_statement",
        "unknown_gate",
    }


def test_reset_and_barrier_are_refused_as_gates_rather_than_ignored() -> None:
    """Silently dropping either would change the program's meaning."""

    for statement in ("reset q[0];", "barrier q[0], q[1];"):
        assert _issue_code(_program(statement)) == "unknown_gate"


@pytest.mark.parametrize("version", LANES)
def test_a_program_needs_a_terminal_block_covering_the_whole_register(
    version: float,
) -> None:
    """The importer reads a program that ends in a result, not a prefix of one."""

    assert _issue_code(HEADER[version] + "h q[0];\n") == "incomplete_measurement"

    partial = {2.0: "measure q[0] -> c[0];\n", 3.0: "c[0] = measure q[0];\n"}[version]
    assert (
        _issue_code(HEADER[version] + "h q[0];\n" + partial) == "incomplete_measurement"
    )


@pytest.mark.parametrize("version", LANES)
def test_a_complete_program_imports_the_instructions_it_declares(
    version: float,
) -> None:
    """The negative tests above are only meaningful beside a positive one."""

    import flagquantum as fq

    imported = fq.from_openqasm(_program("h q[0];", "cx q[0], q[1];", version=version))
    assert [(i.name, i.wires) for i in imported.instructions] == [
        ("h", (0,)),
        ("cx", (0, 1)),
    ]
