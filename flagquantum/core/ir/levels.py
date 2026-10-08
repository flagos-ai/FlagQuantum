"""Internal IR level boundary: its vocabulary, and the two directions across it.

The public IR is `CircuitIR` schema 1.0. Phase 4 adds program, quantum, and target
levels behind it. Nothing in this module is public: it is not re-exported from
`flagquantum.core.ir`, is absent from `docs/public_api_v1.json`, and adds no root
export.

`contracts/multi-level-ir-internal-v1-candidate.json` owns the boundary's seven
claims -- level entry and exit, value identity, linearity, joins, failure
categories, round trip, and rejection -- and
`tools/check_multi_level_ir_contract.py` is its reader. This module supplies the
two things that contract names:

* the vocabulary the claims are spelled in -- `INTERNAL_LEVELS`,
  `SUPPORT_STATES`, `VALUE_KINDS`, `TERMINATORS`, and `DIAGNOSTIC_LEVELS`;
* the entry and exit, `circuit_to_program` and `program_to_circuit`, plus the
  `lower_to_level` dispatcher that refuses every exit the contract declares but
  this implementation does not own yet.

The program level itself lives beside this module, in
`flagquantum/core/ir/program/`, and this module re-exports the names the contract
and the conformance suite read through the boundary. The module is not a shim
around nothing: it states the vocabulary in one place, and it is the only place
that decides which declared exits are real, so the conformance suite
`tests/unit/test_multi_level_ir_contract.py` reads a level through one boundary
whether the level behind it is a contract fake or an implementation.

The lowering refuses every conversion it cannot state exactly -- dynamic
structure, conditional regions, an opcode outside the operator vocabulary, and any
record whose operation graph does not re-derive the payload it carries -- rather
than approximating one.
"""

from __future__ import annotations

from typing import Any

from . import CircuitIR as CircuitIR
from .diagnostics import DIAGNOSTIC_LEVELS as DIAGNOSTIC_LEVELS
from .diagnostics import LEVEL_PROGRAM as LEVEL_PROGRAM
from .diagnostics import LEVEL_QUANTUM as LEVEL_QUANTUM
from .diagnostics import LEVEL_TARGET as LEVEL_TARGET
from .diagnostics import PUBLIC_LEVEL as PUBLIC_LEVEL
from .diagnostics import LevelContractError as LevelContractError
from .diagnostics import LevelDiagnostic as LevelDiagnostic
from .diagnostics import _diagnostic
from .program import INVALID_INPUT as INVALID_INPUT
from .program import LINEAR_VALUE_KINDS as LINEAR_VALUE_KINDS
from .program import SUPPORT_STATES as SUPPORT_STATES
from .program import SUPPORTED_EXACT as SUPPORTED_EXACT
from .program import TERMINATORS as TERMINATORS
from .program import UNSUPPORTED_WITH_DIAGNOSTICS as UNSUPPORTED_WITH_DIAGNOSTICS
from .program import VALUE_KINDS as VALUE_KINDS
from .program import BlockRecord as BlockRecord
from .program import Edge as Edge
from .program import FunctionRecord as FunctionRecord
from .program import LevelConversion as LevelConversion
from .program import OpcodeSignature as OpcodeSignature
from .program import Operation as Operation
from .program import ProgramRecord as ProgramRecord
from .program import ValueRef as ValueRef
from .program import circuit_to_program as circuit_to_program
from .program import classify_public_circuit as classify_public_circuit
from .program import opcode_signature as opcode_signature
from .program import program_to_circuit as program_to_circuit
from .program import verify_program_record as verify_program_record

# Every name above is imported as itself on purpose. This module is the boundary
# the contract and the conformance suite read through, and the gate requires it to
# declare no `__all__`, so a plain `from .program import ValueRef` would be read by
# the linter as an unused import and deleted -- taking the boundary's vocabulary
# with it. Re-spelling each name is what keeps the delegation visible.

#: The levels Phase 4 adds behind the public IR. Every one of them is internal.
INTERNAL_LEVELS = (LEVEL_PROGRAM, LEVEL_QUANTUM, LEVEL_TARGET)

#: The exits this implementation realizes. `lower_to_level` refuses every other
#: declared level by name rather than returning an approximation of it.
REALIZED_EXITS = (LEVEL_PROGRAM,)


def lower_to_level(program: Any, target_level: str) -> LevelConversion:
    """Cross the boundary towards `target_level`, refusing unbuilt exits.

    Only `program` is realized here. The quantum and target levels are declared
    by the contract and are refused by name until the implementation that owns
    them lands, so a caller never receives an approximation of a level that does
    not exist.
    """
    if target_level in (LEVEL_QUANTUM, LEVEL_TARGET):
        return LevelConversion(
            state=UNSUPPORTED_WITH_DIAGNOSTICS,
            level=LEVEL_PROGRAM,
            diagnostics=(
                _diagnostic(
                    "level.unsupported_exit",
                    f"the {target_level!r} level is declared but not implemented",
                ),
            ),
        )
    if target_level in REALIZED_EXITS:
        return circuit_to_program(program)
    if target_level == PUBLIC_LEVEL:
        return classify_public_circuit(program)
    raise LevelContractError(f"unknown target level {target_level!r}")


def round_trip(program: Any) -> LevelConversion:
    """Carry a public circuit across the program level and back again.

    Returns the restore result, with kind `circuit` and the canonical payload the
    input started from, when both directions are exact. Exactness is enforced
    where the payload is restored, so a payload the boundary cannot re-canonicalize
    is refused rather than reported as an exact round trip.
    """
    forward = circuit_to_program(program)
    if forward.state != SUPPORTED_EXACT or forward.record is None:
        return forward
    return program_to_circuit(forward.record)
