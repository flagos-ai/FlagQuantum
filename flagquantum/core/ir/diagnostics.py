"""Typed diagnostics for the internal IR levels.

Crossing an internal level never changes a public schema, so a refusal is
reported as data rather than raised as an exception: one `LevelDiagnostic` per
refusal, naming a code from the registry below and the level that raised it.

This module owns that registry. The boundary facade, the verifier, and the
program level all refuse through `_diagnostic`, so a code is declared in exactly
one place and a level is attributed to it in exactly one place. `IR-006`'s
taxonomy of support states lives with the level that produces them -- a state
classifies a conversion, and the program level is where a conversion happens --
while what lives here is which refusals exist and where each one happens.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The level names a diagnostic is attributed to. They are spelled here as
#: literals rather than imported from the boundary so that this module stays the
#: leaf of the package's import graph: the boundary, the verifier, and the
#: program level all import this module, and none of them is imported by it.
PUBLIC_LEVEL = "circuit"
LEVEL_PROGRAM = "program"
LEVEL_QUANTUM = "quantum"
LEVEL_TARGET = "target"

#: Every refusal the internal levels can report, and the level that raises it.
#: A code is added here before it is raised, and `tools/check_multi_level_ir_contract.py`
#: reads both directions across every refusal site so the vocabulary cannot drift
#: from the lowering and the verifier.
DIAGNOSTIC_LEVELS: dict[str, str] = {
    "level.not_circuit_ir": PUBLIC_LEVEL,
    "level.dynamic_instruction": PUBLIC_LEVEL,
    "level.conditional_instruction": PUBLIC_LEVEL,
    "level.round_trip_mismatch": LEVEL_PROGRAM,
    "level.not_program_record": LEVEL_PROGRAM,
    "level.no_canonical_payload": LEVEL_PROGRAM,
    "level.unsupported_exit": LEVEL_PROGRAM,
    "level.payload_invalid": LEVEL_PROGRAM,
    "program.block_duplicate": LEVEL_PROGRAM,
    "program.block_missing": LEVEL_PROGRAM,
    "program.edge_arity": LEVEL_PROGRAM,
    "program.edge_kind": LEVEL_PROGRAM,
    "program.edge_missing": LEVEL_PROGRAM,
    "program.edge_unexpected": LEVEL_PROGRAM,
    "program.edge_unknown": LEVEL_PROGRAM,
    "program.entry_missing": LEVEL_PROGRAM,
    "program.function_arity": LEVEL_PROGRAM,
    "program.opcode_unknown": LEVEL_PROGRAM,
    "program.operand_arity": LEVEL_PROGRAM,
    "program.operand_kind": LEVEL_PROGRAM,
    "program.parameter_missing": LEVEL_PROGRAM,
    "program.result_kind": LEVEL_PROGRAM,
    "program.terminator_unknown": LEVEL_PROGRAM,
    "program.value_redefined": LEVEL_PROGRAM,
    "program.value_undefined": LEVEL_PROGRAM,
    "value.identity": LEVEL_PROGRAM,
    "value.linear_duplicate": LEVEL_PROGRAM,
    "value.linear_unconsumed": LEVEL_PROGRAM,
}


class LevelContractError(ValueError):
    """Raised when a caller asks the boundary for something it does not define."""


@dataclass(frozen=True)
class LevelDiagnostic:
    """One typed refusal, classified by the level that raised it.

    Attributes:
        code: Dotted name from `DIAGNOSTIC_LEVELS`.
        message: Human-readable statement of what was refused.
        level: The level at which the refusal happened.
    """

    code: str
    message: str
    level: str


def _diagnostic(code: str, message: str) -> LevelDiagnostic:
    """Build one diagnostic, taking its level from the declared registry."""
    try:
        level = DIAGNOSTIC_LEVELS[code]
    except KeyError:  # pragma: no cover - a defect, not a caller error
        raise LevelContractError(f"undeclared diagnostic code {code!r}") from None
    return LevelDiagnostic(code=code, message=message, level=level)
