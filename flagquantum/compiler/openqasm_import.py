"""Import OpenQASM 2 or OpenQASM 3 text as a FlagQuantum program.

This module is the inverse of ``flagquantum.compiler.openqasm.emit_openqasm`` and
reads the canonical subset that emitter writes. Parsing and the final consistency
check use different machinery on purpose: the parser decides what each statement
means, and the program it built is then exported again and required to reproduce
the source text exactly. A construct outside the subset therefore fails closed
instead of being imported with a shifted meaning, which matters because the two
directions disagree about several gates -- OpenQASM 3 writes ``u2(phi, lbd)`` as
``U(pi/2, phi, lbd)``, and a reader that accepted any three-angle ``U`` as ``u3``
would import a different program than it read.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, NoReturn

from ..core.ir import CircuitIR, Instruction, MeasurementNode
from ..errors import CompilationError
from .openqasm import emit_openqasm
from .openqasm_gates import FIXED_GATES, PARAMETERIZED_GATES

SUPPORTED_VERSIONS = (2.0, 3.0)

_ISSUE_CODES = (
    "empty_source",
    "unsupported_version",
    "malformed_statement",
    "unknown_gate",
    "unknown_register",
    "out_of_range_qubit",
    "invalid_parameters",
    "unbound_parameter",
    "duplicate_measurement_target",
    "incomplete_measurement",
    "not_canonical_text",
)

_VERSION = re.compile(r"OPENQASM\s+(?P<version>[0-9]+(?:\.[0-9]+)?)\s*;")
_INCLUDES = {2.0: 'include "qelib1.inc";', 3.0: 'include "stdgates.inc";'}

_NAME_FIRST_REGISTER = re.compile(
    r"(?P<kind>qreg|creg)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"\[\s*(?P<size>[1-9][0-9]*)\s*\]\s*;"
)
_SIZE_FIRST_REGISTER = re.compile(
    r"(?P<kind>qubit|bit)\s*\[\s*(?P<size>[1-9][0-9]*)\s*\]\s*"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;"
)
_REFERENCE = re.compile(
    r"(?P<register>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(?P<index>[0-9]+)\s*\]"
)
_GATE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*"
    r"(?:\((?P<parameters>[^()]*)\))?\s*"
    r"(?P<operands>[^;]+?)\s*;"
)
_MEASURE_THROUGH_ARROW = re.compile(
    r"measure\s+(?P<qubit>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(?P<qubit_index>[0-9]+)\s*\]"
    r"\s*->\s*"
    r"(?P<clbit>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(?P<clbit_index>[0-9]+)\s*\]\s*;"
)
_MEASURE_INTO_BIT = re.compile(
    r"(?P<clbit>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(?P<clbit_index>[0-9]+)\s*\]"
    r"\s*=\s*measure\s+"
    r"(?P<qubit>[A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(?P<qubit_index>[0-9]+)\s*\]\s*;"
)
_MEASURE_INTO_REGISTER = re.compile(
    r"(?P<clbit>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*measure\s+"
    r"(?P<qubit>[A-Za-z_][A-Za-z0-9_]*)\s*;"
)
# OpenQASM 3 writes the inverse of a gate as ``pow(-1) @ gate ...``. The emitter
# uses that form for exactly one opcode, ``sxdg``, so import accepts the same one
# form and refuses every other power rather than evaluating it.
_INVERSE_PREFIX = re.compile(r"pow\s*\(\s*-1\s*\)\s*@\s*(?P<rest>.+?)\s*;", re.DOTALL)
_SQRT_X_OPERANDS = re.compile(r"sx\s+(?P<operands>.+?)\s*$")

# OpenQASM 3 spells the three-angle gate ``u3`` as ``U``. It has no separate
# two-angle gate, so ``u2(phi, lbd)`` is written as the angle-pi/2 member of
# ``u3``; the two denote the same unitary and, because the emitter writes the
# angle as the Python representation of pi/2, the same text.
_U_IS_U3 = "u3"

_MEASUREMENT_PATTERNS = (
    _MEASURE_THROUGH_ARROW,
    _MEASURE_INTO_BIT,
    _MEASURE_INTO_REGISTER,
)


class OpenQASMImportError(CompilationError):
    """OpenQASM text is outside the canonical subset FlagQuantum imports.

    Attributes:
        issue_code: Stable identifier for the reason the text was refused.
        line: One-based source line the refusal applies to, or ``None``.
    """

    def __init__(
        self, issue_code: str, message: str, *, line: int | None = None
    ) -> None:
        if issue_code not in _ISSUE_CODES:
            raise AssertionError(f"undeclared import issue code {issue_code!r}")
        location = f" (line {line})" if line is not None else ""
        super().__init__(f"[{issue_code}] OpenQASM import refused{location}: {message}")
        self.issue_code = issue_code
        self.line = line


@dataclass(frozen=True, slots=True)
class _Register:
    """One declared quantum or classical register."""

    name: str
    size: int


def _refuse(issue_code: str, message: str, *, line: int | None = None) -> NoReturn:
    raise OpenQASMImportError(issue_code, message, line=line)


def _statements(source: str) -> tuple[str, list[tuple[int, str]]]:
    """Return the text without comment or blank lines, and one pair per statement."""

    normalized = source.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    return normalized, [
        (number, text)
        for number, raw in enumerate(normalized.split("\n"), start=1)
        for text in (raw.split("//", 1)[0].strip(),)
        if text
    ]


def _register(statement: str) -> _Register | None:
    """Read one register declaration in either dialect's spelling."""

    matched = _NAME_FIRST_REGISTER.fullmatch(
        statement
    ) or _SIZE_FIRST_REGISTER.fullmatch(statement)
    if matched is None:
        return None
    return _Register(matched.group("name"), int(matched.group("size")))


def _parameters(text: str | None, *, line: int) -> tuple[float, ...]:
    if text is None:
        return ()
    values: list[float] = []
    for item in text.split(","):
        expression = item.strip()
        if not expression:
            _refuse("invalid_parameters", "a parameter is empty", line=line)
        try:
            value = float(expression)
        except ValueError:
            _refuse(
                "unbound_parameter",
                f"parameter {expression!r} is not a number; FlagQuantum imports "
                "bound programs only, so substitute a value before import",
                line=line,
            )
        if not math.isfinite(value):
            _refuse(
                "invalid_parameters",
                f"parameter {expression!r} is not finite",
                line=line,
            )
        values.append(value)
    return tuple(values)


def _operands(text: str, *, register: str, line: int) -> tuple[int, ...]:
    """Return the qubit indices an operand list names, requiring one register."""

    matches = list(_REFERENCE.finditer(text))
    if not matches or _REFERENCE.sub("", text).replace(",", "").strip():
        _refuse(
            "malformed_statement",
            f"operand list {text.strip()!r} is not a list of register references",
            line=line,
        )
    for match in matches:
        if match.group("register") != register:
            _refuse(
                "unknown_register",
                f"operand {match.group(0)!r} does not name the declared quantum "
                f"register {register!r}",
                line=line,
            )
    return tuple(int(match.group("index")) for match in matches)


def _check_qubits(qubits: tuple[int, ...], *, register: _Register, line: int) -> None:
    if len(set(qubits)) != len(qubits):
        _refuse(
            "malformed_statement",
            f"the statement names the same qubit more than once: {qubits}",
            line=line,
        )
    for qubit in qubits:
        if qubit >= register.size:
            _refuse(
                "out_of_range_qubit",
                f"qubit {qubit} is outside {register.name}[{register.size}]",
                line=line,
            )


def _instruction(
    name: str,
    parameters: tuple[float, ...],
    qubits: tuple[int, ...],
    *,
    line: int,
) -> Instruction:
    if name == "U":
        name = _U_IS_U3
    parameter_names: tuple[str, ...]
    if name in FIXED_GATES:
        opcode, parameter_names = FIXED_GATES[name], ()
    elif name in PARAMETERIZED_GATES:
        opcode, parameter_names = PARAMETERIZED_GATES[name]
    else:
        _refuse(
            "unknown_gate",
            f"gate {name!r} is not in the canonical OpenQASM subset FlagQuantum writes",
            line=line,
        )
    if len(parameters) != len(parameter_names):
        _refuse(
            "invalid_parameters",
            f"gate {name!r} takes {len(parameter_names)} parameter(s), got "
            f"{len(parameters)}",
            line=line,
        )
    try:
        return Instruction(
            opcode, qubits, params=dict(zip(parameter_names, parameters, strict=True))
        )
    except (TypeError, ValueError) as error:
        _refuse(
            "malformed_statement",
            f"gate {name!r} cannot act on {len(qubits)} qubit(s): {error}",
            line=line,
        )


def _single_measurement(
    matched: re.Match[str], *, qreg: _Register, creg: _Register, line: int
) -> tuple[int, int]:
    """Return the ``(classical bit, qubit)`` pair one measurement statement reads."""

    qubit = matched.group("qubit")
    qubit_index = int(matched.group("qubit_index"))
    clbit = matched.group("clbit")
    clbit_index = int(matched.group("clbit_index"))
    if qubit != qreg.name or clbit != creg.name:
        _refuse(
            "unknown_register",
            f"measurement uses {qubit!r} and {clbit!r}, not the declared "
            f"{qreg.name!r} and {creg.name!r}",
            line=line,
        )
    if qubit_index >= qreg.size:
        _refuse(
            "out_of_range_qubit",
            f"measurement reads {qubit}[{qubit_index}], outside "
            f"{qreg.name}[{qreg.size}]",
            line=line,
        )
    if clbit_index >= creg.size:
        _refuse(
            "out_of_range_qubit",
            f"measurement writes {clbit}[{clbit_index}], outside "
            f"{creg.name}[{creg.size}]",
            line=line,
        )
    return clbit_index, qubit_index


def _read_measurements(
    statements: list[tuple[int, str]],
    *,
    index: int,
    qreg: _Register,
    creg: _Register,
) -> tuple[tuple[int, ...], bool, int]:
    """Read the terminal measurement block.

    Returns the qubit behind each classical bit, whether the source measured the
    whole register at once, and the index of the first statement after the block.
    """

    line, text = statements[index]
    whole = _MEASURE_INTO_REGISTER.fullmatch(text)
    if whole is not None:
        source, target = whole.group("qubit"), whole.group("clbit")
        if source != qreg.name or target != creg.name:
            _refuse(
                "unknown_register",
                f"measurement uses {source!r} and {target!r}, not the declared "
                f"{qreg.name!r} and {creg.name!r}",
                line=line,
            )
        if creg.size != qreg.size:
            _refuse(
                "incomplete_measurement",
                f"the whole-register measurement needs {creg.name} to be as wide "
                f"as {qreg.name}, but they are {creg.size} and {qreg.size}",
                line=line,
            )
        return tuple(range(qreg.size)), True, index + 1

    if creg.size > qreg.size:
        _refuse(
            "incomplete_measurement",
            f"{creg.name} is wider than {qreg.name}, so no measurement block can "
            "define every classical bit from a distinct qubit",
            line=line,
        )
    qubits: list[int | None] = [None] * creg.size
    while index < len(statements):
        line, text = statements[index]
        matched = _MEASURE_THROUGH_ARROW.fullmatch(text) or _MEASURE_INTO_BIT.fullmatch(
            text
        )
        if matched is None:
            break
        clbit_index, qubit_index = _single_measurement(
            matched, qreg=qreg, creg=creg, line=line
        )
        if qubits[clbit_index] is not None:
            _refuse(
                "duplicate_measurement_target",
                f"{creg.name}[{clbit_index}] is written by two measurements",
                line=line,
            )
        qubits[clbit_index] = qubit_index
        index += 1
    missing = [slot for slot, qubit in enumerate(qubits) if qubit is None]
    if missing:
        _refuse(
            "incomplete_measurement",
            f"no measurement writes {creg.name}{missing}, so the classical "
            "register is only partly defined",
        )
    resolved = tuple(qubit for qubit in qubits if qubit is not None)
    if len(set(resolved)) != len(resolved):
        _refuse(
            "duplicate_measurement_target",
            f"the measurements read the same qubit more than once: {resolved}",
        )
    return resolved, False, index


def _parse(
    source: str,
) -> tuple[int, float, tuple[Instruction, ...], tuple[int, ...], bool]:
    normalized, statements = _statements(source)
    if not statements:
        _refuse("empty_source", "the source text has no statement")

    first_line, first = statements[0]
    version_match = _VERSION.fullmatch(first)
    if version_match is None:
        _refuse(
            "malformed_statement",
            f"the first statement must declare OPENQASM 2.0 or 3.0, found {first!r}",
            line=first_line,
        )
    version = float(version_match.group("version"))
    if version not in SUPPORTED_VERSIONS:
        _refuse(
            "unsupported_version",
            f"OpenQASM version {version} is not supported; supported versions are "
            "2.0 and 3.0",
            line=first_line,
        )

    expected_include = _INCLUDES[version]
    if len(statements) < 2 or statements[1][1] != expected_include:
        _refuse(
            "malformed_statement",
            f"the second statement must be {expected_include!r}, which is what "
            f"OPENQASM {version} requires",
            line=statements[1][0] if len(statements) > 1 else None,
        )

    if len(statements) < 4:
        _refuse(
            "malformed_statement",
            "the version, include, quantum register and classical register must "
            "come first",
        )
    qreg = _register(statements[2][1])
    creg = _register(statements[3][1])
    if qreg is None or creg is None:
        _refuse(
            "malformed_statement",
            "the third and fourth statements must declare the quantum and "
            "classical registers",
            line=statements[2][0] if qreg is None else statements[3][0],
        )
    if qreg.name == creg.name:
        _refuse(
            "malformed_statement",
            f"the quantum and classical registers are both named {qreg.name!r}",
            line=statements[3][0],
        )

    instructions: list[Instruction] = []
    index = 4
    while index < len(statements):
        line, text = statements[index]
        if any(pattern.fullmatch(text) for pattern in _MEASUREMENT_PATTERNS):
            break
        inverse = _INVERSE_PREFIX.fullmatch(text)
        if inverse is not None:
            square_root = _SQRT_X_OPERANDS.fullmatch(inverse.group("rest"))
            if square_root is None:
                _refuse(
                    "unknown_gate",
                    f"statement {text!r} is an inverse the canonical subset does "
                    "not write; the only inverse FlagQuantum emits is "
                    "'pow(-1) @ sx q[i];'",
                    line=line,
                )
            qubits = _operands(
                square_root.group("operands"), register=qreg.name, line=line
            )
            _check_qubits(qubits, register=qreg, line=line)
            instructions.append(Instruction("sxdg", qubits))
            index += 1
            continue
        matched = _GATE.fullmatch(text)
        if matched is None or matched.group("name") in (qreg.name, creg.name):
            _refuse(
                "malformed_statement",
                f"statement {text!r} is not a gate application",
                line=line,
            )
        parameters = _parameters(matched.group("parameters"), line=line)
        qubits = _operands(matched.group("operands"), register=qreg.name, line=line)
        _check_qubits(qubits, register=qreg, line=line)
        instructions.append(
            _instruction(matched.group("name"), parameters, qubits, line=line)
        )
        index += 1

    if index >= len(statements):
        _refuse(
            "incomplete_measurement",
            "the program has no terminal measurement block",
        )
    qubits, whole, index = _read_measurements(
        statements, index=index, qreg=qreg, creg=creg
    )
    if index != len(statements):
        _refuse(
            "malformed_statement",
            f"statement {statements[index][1]!r} follows the measurement block; "
            "FlagQuantum imports programs whose measurements are terminal",
            line=statements[index][0],
        )
    return qreg.size, version, tuple(instructions), qubits, whole


def _canonical_statement(statement: str) -> str:
    """Collapse the spacing OpenQASM ignores, leaving the meaning in place.

    Comments and blank lines are already gone by this point. What is left is
    spacing around punctuation, which no OpenQASM reader treats as meaningful,
    so two statements that differ only there are the same statement.
    """

    spaced = re.sub(r"([(),;=\[\]])", r" \1 ", statement.replace("->", " -> "))
    return " ".join(spaced.split())


def _first_difference(
    statements: list[tuple[int, str]], written: list[tuple[int, str]]
) -> str:
    """Name the first statement the source does not share with its program."""

    for (line, source_text), (_, written_text) in zip(
        statements, written, strict=False
    ):
        if _canonical_statement(source_text) != _canonical_statement(written_text):
            return f"line {line} reads {source_text!r}, which FlagQuantum writes as {written_text!r}"
    shorter, longer = (
        ("source", "program")
        if len(statements) < len(written)
        else ("program", "source")
    )
    return (
        f"the {shorter} has {min(len(statements), len(written))} statements and the "
        f"{longer} has {max(len(statements), len(written))}"
    )


@dataclass(frozen=True, slots=True)
class OpenQASMImport:
    """An imported OpenQASM program together with what was read to produce it.

    Attributes:
        n_qubits: Width of the quantum register the source declared.
        version: The OpenQASM version the source declared, 2.0 or 3.0.
        instructions: The imported program, in source order.
        measurement_qubits: The qubit behind each classical bit, in bit order.
        source: The exact text the program was imported from.
        whole_register_measurement: Whether the source measured the whole
            quantum register in one statement, which OpenQASM 3 alone allows;
            it changes only the measurement statement, not the mapping.
    """

    n_qubits: int
    version: float
    instructions: tuple[Instruction, ...]
    measurement_qubits: tuple[int, ...]
    source: str
    whole_register_measurement: bool = field(default=False, repr=False, compare=False)

    @property
    def n_wires(self) -> int:
        """Deprecated alias for :attr:`n_qubits`."""

        from ..core._qubit_aliases import warn_qubit_alias

        warn_qubit_alias("n_wires", "n_qubits", stacklevel=3)
        return self.n_qubits

    def to_ir(self) -> CircuitIR:
        """Return the canonical FlagQuantum IR of the imported program."""

        return CircuitIR(
            self.n_qubits,
            self.instructions,
            measurements=(
                MeasurementNode("counts", self.measurement_qubits, shots=None),
            ),
        )

    def to_circuit(self) -> Any:
        """Return the imported program as a :class:`flagquantum.Circuit`."""

        from ..circuit import Circuit

        return Circuit.from_ir(self.to_ir())

    def to_openqasm(self) -> str:
        """Return the canonical text of the imported program.

        The text is written in the version the source declared and with the same
        measurement statement the source used, so for text that is already
        canonical this reproduces it exactly.

        Examples:
            >>> import flagquantum as fq
            >>> text = (
            ...     'OPENQASM 3.0;\\ninclude "stdgates.inc";\\n'
            ...     'qubit[2] q;\\nbit[2] c;\\nh q[0];\\ncx q[0], q[1];\\n'
            ...     'c = measure q;'
            ... )
            >>> fq.from_openqasm(text).to_openqasm() == text
            True
        """

        return emit_openqasm(
            self.to_ir(),
            version=self.version,
            result_wires=(
                None if self.whole_register_measurement else self.measurement_qubits
            ),
        )


def import_openqasm(source: str) -> OpenQASMImport:
    """Import OpenQASM 2 or OpenQASM 3 text as a FlagQuantum program.

    The accepted text is the canonical subset FlagQuantum writes, so a program
    can be exported and imported back unchanged. Text outside that subset is
    refused with an :class:`OpenQASMImportError` that names the reason in
    ``issue_code``; nothing is imported approximately.

    Args:
        source: The complete text of one OpenQASM program.

    Returns:
        The imported program, the version it declared, and the classical-bit to
        qubit mapping its terminal measurement block records.

    Raises:
        OpenQASMImportError: The text is outside the imported subset.
        TypeError: ``source`` is not text.

    Examples:
        >>> import flagquantum as fq
        >>> text = '''OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[2];
        ... creg c[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... measure q[0] -> c[0];
        ... measure q[1] -> c[1];'''
        >>> program = fq.from_openqasm(text)
        >>> [instruction.name for instruction in program.instructions]
        ['h', 'cx']
        >>> program.measurement_qubits
        (0, 1)
    """

    if not isinstance(source, str):
        raise TypeError(f"OpenQASM source must be str, got {type(source).__name__}")

    _, statements = _statements(source)
    normalized = "\n".join(text for _, text in statements)
    n_qubits, version, instructions, qubits, whole = _parse(normalized)

    ir = CircuitIR(
        n_qubits,
        instructions,
        measurements=(MeasurementNode("counts", qubits, shots=None),),
    )
    written = _statements(
        emit_openqasm(ir, version=version, result_wires=None if whole else qubits)
    )[1]
    if [_canonical_statement(text) for _, text in written] != [
        _canonical_statement(text) for _, text in statements
    ]:
        _refuse(
            "not_canonical_text",
            "the source is not the text FlagQuantum writes for the program it "
            "denotes, so importing it would change the program "
            f"({_first_difference(statements, written)})",
        )
    return OpenQASMImport(
        n_qubits=n_qubits,
        version=version,
        instructions=instructions,
        measurement_qubits=qubits,
        source=source,
        whole_register_measurement=whole,
    )


def import_openqasm_to_ir(source: str) -> CircuitIR:
    """Import OpenQASM 2 or OpenQASM 3 text as canonical FlagQuantum IR.

    Examples:
        >>> import flagquantum as fq
        >>> program = fq.from_openqasm("OPENQASM 2.0;\\ninclude \\"qelib1.inc\\";\\n"
        ...     "qreg q[1];\\ncreg c[1];\\nx q[0];\\nmeasure q[0] -> c[0];")
        >>> [instruction.name for instruction in program.to_ir().instructions]
        ['x']
    """

    return import_openqasm(source).to_ir()


__all__ = (
    "OpenQASMImport",
    "OpenQASMImportError",
    "SUPPORTED_VERSIONS",
    "import_openqasm",
    "import_openqasm_to_ir",
)
