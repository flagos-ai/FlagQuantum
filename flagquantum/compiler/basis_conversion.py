"""Convert a program from one named gate set into another.

`native_gate_legalization` answers the deployment question: is this program
expressible by the gates a target has *evidence* for?  Every decision there goes
through a `TargetCapabilitySnapshot`, and a gate is admitted by a declared fact
with a support status, an exposure and an evidence reference.  That is the right
shape for a device, and it is the wrong shape for the other question a compiler
is asked, which has no device in it at all: rewrite this circuit from the basis
it is written in into a basis the caller names.

This module is that second question, and it is the same engine rather than a
second one.  `convert_instructions` owns the per-instruction loop, the
deterministic rewrite choice, the added-operation bound and the records;
`native_gate_legalization` calls it too and the two differ in the predicate that
decides whether a leaf can run and in the sentence a refusal prints.  The three
hidden decisions the loop needs -- which z-rotation, which pi/2 pulse and which
entangler the Euler and KAK syntheses are driven with -- are derived from the
*published opcode set* here as well, so a snapshot-backed target and a named
basis answer them by the same ordered tables instead of by two copies.

A named basis is a set of gate names, and it is validated as one before any
instruction is touched: every name has to be a declared opcode of the operator
schema, it has to be unitary, and it has to be at least one wire wide.  A name
outside the schema is what a caller gets from a typo or from a gate another
toolchain spells differently, and the refusal names every offending entry at
once rather than the first, so one call is enough to fix the list.  Channels are
refused for the same reason `translate` has no rule for them: a named basis is a
gate set, and a channel is not a gate.

Nothing here runs a basis conversion nobody asked for.  There is no "native"
default, because a program converted without a named basis is a program rewritten
into whatever the compiler felt like.  And nothing here selects a *device* basis:
the module carries no vendor gate-set table, because a vendor's basis is a
published fact about hardware that belongs to a capability snapshot carrying an
evidence reference, not to a constant in the compiler.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field, replace

from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.operator_schema import canonical_opcode, get_operator_schema
from ..errors import CompilationError
from .basis_translation import EQUIVALENCE_RULES, EquivalenceRule, translate
from .one_qubit_synthesis import (
    HALF_PI_PULSE_OPCODES,
    Z_ROTATION_OPCODES,
    synthesize_one_qubit_matrix,
)
from .two_qubit_synthesis import SUPERCONTROLLED_ENTANGLERS, synthesize_two_qubit


class BasisConversionError(CompilationError):
    """A program cannot be expressed in the named basis, or the basis is invalid."""


@dataclass(frozen=True)
class GateDecompositionRecord:
    """One deterministic source-to-basis instruction replacement."""

    instruction_index: int
    source_opcode: str
    replacement_opcodes: tuple[str, ...]


@dataclass(frozen=True)
class BasisConversionResult:
    """A program in the named basis, plus the record of how it got there."""

    source_program: CircuitIR = field(repr=False)
    program: CircuitIR
    source_content_hash: str
    result_content_hash: str
    source_basis: tuple[str, ...]
    target_basis: tuple[str, ...]
    decompositions: tuple[GateDecompositionRecord, ...]
    conversion_identity: str

    def __post_init__(self) -> None:
        if self.source_program.content_hash != self.source_content_hash:
            raise ValueError("basis source program does not match source_content_hash")
        if self.program.content_hash != self.result_content_hash:
            raise ValueError("basis result program does not match result_content_hash")

    @property
    def changed(self) -> bool:
        return bool(self.decompositions)


def require_basis(gates: Iterable[str]) -> tuple[str, ...]:
    """Return `gates` as a validated, canonicalised, sorted named basis.

    A bare string is refused rather than iterated: `require_basis("hcx")` would
    otherwise silently become a three-gate basis built from the letters, which
    is a typo the caller cannot see. Every entry has to name a declared unitary
    opcode of at least one wire, and every offending entry is named in one
    message, sorted and deduplicated, because a caller fixing a basis wants the
    whole list rather than the first mistake in it.
    """

    if isinstance(gates, (str, bytes)):
        raise BasisConversionError(
            "a named basis is a sequence of gate names, not a string; "
            "pass ('h', 'cx') rather than 'hcx'"
        )
    unknown: set[str] = set()
    refused: set[str] = set()
    canonical: set[str] = set()
    for entry in gates:
        if not isinstance(entry, str) or not entry.strip():
            raise BasisConversionError(
                "every named-basis entry must be a non-empty gate name"
            )
        name = canonical_opcode(entry)
        schema = get_operator_schema(name)
        if schema is None:
            unknown.add(entry.strip())
        elif schema.channel or not schema.unitary or schema.arity < 1:
            refused.add(name)
        else:
            canonical.add(name)
    if unknown:
        raise BasisConversionError(
            "unknown opcode in the named basis: "
            + ", ".join(repr(name) for name in sorted(unknown))
        )
    if refused:
        raise BasisConversionError(
            "a named basis holds gates, and these are not gates: "
            + ", ".join(repr(name) for name in sorted(refused))
        )
    if not canonical:
        raise BasisConversionError("a named basis needs at least one gate")
    return tuple(sorted(canonical))


def z_rotation_opcode(published: Collection[str]) -> str | None:
    """Return the z-rotation among `published`, or None when there is none.

    The vocabulary is iterated rather than the published set, because a basis
    may publish more than one of `rz`, `phase`, and `u1` -- they are the same
    gate up to a global phase -- and the choice has to be a fixed function of
    the published set rather than of a collection's iteration order.
    """

    for opcode in Z_ROTATION_OPCODES:
        if opcode in published:
            return opcode
    return None


def half_pi_pulse_opcode(published: Collection[str]) -> str | None:
    """Return the pi/2 x-rotation among `published`, or None."""

    for opcode in HALF_PI_PULSE_OPCODES:
        if opcode in published:
            return opcode
    return None


def entangler_opcode(published: Collection[str]) -> str | None:
    """Return the supercontrolled entangler among `published`, or None.

    `SUPERCONTROLLED_ENTANGLERS` is ordered so a parameter-free spelling is
    preferred over one that has to be applied at an angle, so the answer for a
    given published set never depends on a mapping's insertion order.
    """

    for opcode in SUPERCONTROLLED_ENTANGLERS:
        if opcode in published:
            return opcode
    return None


def matrix_replacement(
    instruction: Instruction,
    *,
    z_rotation: str | None,
    pulse_opcode: str | None,
    entangler: str | None,
) -> tuple[Instruction, ...] | None:
    """Rewrite a matrix-carrying instruction, which no schema table describes.

    One wire goes through Euler synthesis, two through KAK synthesis, and
    anything wider is refused because the entangler basis reaches exactly two.
    """

    if z_rotation is None or pulse_opcode is None:
        return None
    if len(instruction.wires) == 1:
        return synthesize_one_qubit_matrix(
            instruction.matrix,
            wire=instruction.wires[0],
            z_rotation=z_rotation,
            pulse_opcode=pulse_opcode,
            metadata=instruction.metadata,
        )
    if len(instruction.wires) == 2 and entangler is not None:
        return synthesize_two_qubit(
            instruction.matrix,
            wires=instruction.wires,
            entangler=entangler,
            z_rotation=z_rotation,
            pulse_opcode=pulse_opcode,
            metadata=instruction.metadata,
        )
    return None


# The four sentences the shared loop can be stopped by, as templates the caller
# renders. A missing rewrite is "not native" against a device snapshot and
# "outside the named basis" in a basis conversion, and moving that vocabulary
# into the caller is what lets the loop stay single. `{name}` is the opcode of
# the instruction being rewritten and `{gate}` the opcode the rewrite needed.
REFUSAL_KINDS = ("matrix", "no_rule", "escapes", "budget")


def convert_instructions(
    program: CircuitIR,
    *,
    published: Collection[str],
    can_run: Callable[[Instruction], bool],
    refusals: Mapping[str, str],
    error: type[CompilationError],
    max_added_operations: int = 256,
    rules: Mapping[str, tuple[EquivalenceRule, ...]] = EQUIVALENCE_RULES,
) -> tuple[CircuitIR, tuple[GateDecompositionRecord, ...]]:
    """Rewrite every instruction so that each leaf satisfies `can_run`.

    `published` is the set of opcode names the caller's target offers, which is
    what the three synthesis bases are derived from; `can_run` is the finer
    predicate the loop actually asks, so a caller that knows which *parameters*
    a target declares keeps that precision. They are separate arguments because
    a target genuinely can answer them differently: a snapshot may declare `rz`
    while admitting only a subset of its parameters.

    `refusals` and `error` are the caller's vocabulary for the four ways this
    can stop, so the same loop serves a legalization against a device snapshot
    and a conversion into a named basis without either caller's wording leaking
    into the other's messages.

    `rules` is the equivalence table the search composes. It defaults to the
    built-in one, and a caller that extended it with
    `basis_translation.with_equivalence_rule` passes the result, which is how a
    rule outside the built-in table reaches a program without a second registry.
    A registered rule is not permission to leave the basis: every leaf it builds
    is re-checked against `can_run` like any other.
    """

    if not isinstance(max_added_operations, int) or isinstance(
        max_added_operations, bool
    ):
        raise TypeError("max_added_operations must be an integer")
    if max_added_operations < 0:
        raise ValueError("max_added_operations must be non-negative")
    if set(refusals) != set(REFUSAL_KINDS):
        raise ValueError(f"refusals must have exactly the keys {REFUSAL_KINDS}")
    z_rotation = z_rotation_opcode(published)
    pulse_opcode = half_pi_pulse_opcode(published)

    instructions: list[Instruction] = []
    records: list[GateDecompositionRecord] = []
    for index, instruction in enumerate(program.instructions):
        if can_run(instruction):
            instructions.append(instruction)
            continue
        if instruction.matrix is not None:
            replacement = matrix_replacement(
                instruction,
                z_rotation=z_rotation,
                pulse_opcode=pulse_opcode,
                entangler=entangler_opcode(published),
            )
            if replacement is None:
                raise error(refusals["matrix"].format(name=instruction.name))
        else:
            replacement = translate(
                instruction,
                can_run=can_run,
                z_rotation=z_rotation,
                pulse_opcode=pulse_opcode,
                rules=rules,
            )
            if replacement is None:
                raise error(refusals["no_rule"].format(name=instruction.name))
        for item in replacement:
            if not can_run(item):
                raise error(
                    refusals["escapes"].format(name=instruction.name, gate=item.name)
                )
        added = len(instructions) + len(replacement) - index - 1
        if added > max_added_operations:
            raise error(refusals["budget"].format(name=instruction.name))
        instructions.extend(replacement)
        records.append(
            GateDecompositionRecord(
                instruction_index=index,
                source_opcode=instruction.name,
                replacement_opcodes=tuple(item.name for item in replacement),
            )
        )
    result = replace(program, instructions=tuple(instructions)) if records else program
    return result, tuple(records)


def _names(instructions: Iterable[Instruction]) -> tuple[str, ...]:
    """The distinct opcodes of a program, sorted, for a report and not a claim."""

    return tuple(sorted({instruction.name for instruction in instructions}))


def _identity(
    source: CircuitIR,
    result: CircuitIR,
    target_basis: tuple[str, ...],
    decompositions: tuple[GateDecompositionRecord, ...],
) -> str:
    payload = {
        "source_content_hash": source.content_hash,
        "result_content_hash": result.content_hash,
        "target_basis": list(target_basis),
        "decompositions": [
            {
                "instruction_index": item.instruction_index,
                "source_opcode": item.source_opcode,
                "replacement_opcodes": list(item.replacement_opcodes),
            }
            for item in decompositions
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


_REFUSALS = {
    "matrix": "custom matrix instruction {name!r} has no named-basis contract",
    "no_rule": (
        "instruction {name!r} is outside the named basis "
        "and has no verified decomposition"
    ),
    "escapes": (
        "decomposition of {name!r} requires a gate outside the named basis: {gate!r}"
    ),
    "budget": "named-basis decomposition exceeds max_added_operations",
}


def convert_basis(
    program: object,
    *,
    gates: Iterable[str],
    max_added_operations: int = 256,
    rules: Mapping[str, tuple[EquivalenceRule, ...]] = EQUIVALENCE_RULES,
) -> BasisConversionResult:
    """Return `program` rewritten into the named basis, or refuse it by name.

    Every leaf of the result is one of the named gates, and every rewrite is
    exact. A refusal names the instruction it is about: an instruction outside
    the basis with no verified decomposition, a decomposition that itself needs
    a gate outside the basis, or a rewrite whose added operations exceed the
    bound. Nothing is approximated, and no measurement, reset, or channel is
    converted, because `translate` has no rule for one and a conversion that
    silently dropped one would report a program that is not the program it was
    given.

    The report carries the source basis as the distinct opcodes the program was
    written in, so the conversion is readable without re-deriving either basis
    from the two programs.

    `rules` is the equivalence table the search may use, defaulting to the
    built-in one. A caller who has a verified identity the table does not carry
    registers it with `basis_translation.with_equivalence_rule` and passes the
    result here; the named basis is still the only thing the result is allowed
    to contain, so extending the search cannot widen what "in the basis" means.
    """

    target = require_basis(gates)
    source = ensure_circuit_ir(program)
    published = set(target)
    result, decompositions = convert_instructions(
        source,
        published=published,
        can_run=lambda item: item.name in published,
        refusals=_REFUSALS,
        error=BasisConversionError,
        max_added_operations=max_added_operations,
        rules=rules,
    )
    return BasisConversionResult(
        source_program=source,
        program=result,
        source_content_hash=source.content_hash,
        result_content_hash=result.content_hash,
        source_basis=_names(source.instructions),
        target_basis=target,
        decompositions=decompositions,
        conversion_identity=_identity(source, result, target, decompositions),
    )


__all__ = [
    "REFUSAL_KINDS",
    "BasisConversionError",
    "BasisConversionResult",
    "GateDecompositionRecord",
    "convert_basis",
    "convert_instructions",
    "entangler_opcode",
    "half_pi_pulse_opcode",
    "matrix_replacement",
    "require_basis",
    "z_rotation_opcode",
]
