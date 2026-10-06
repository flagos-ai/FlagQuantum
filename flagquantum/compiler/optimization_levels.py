"""The declared optimization levels of the target-independent pass ladder.

`flagquantum/compiler/pipeline.py` runs its passes to a fixed point, and the
order inside one round is what a pass reads. Before this module a caller had no
way to ask for less than the whole round: `optimize(program)` either ran
everything or, through `compile(..., optimize=False)`, nothing at all. This
module is the declaration a caller reads instead -- which passes each level
runs, in which order, and which level this release refuses because the pass it
needs is not implemented yet.

The ladder is stated as **names rather than callables**, on purpose. `pipeline`
imports its passes inside the fixed-point function to keep the import graph
one-directional, so a table of callables here would be a second import of the
same implementations and the two could drift. Names keep one implementation and
let this table be read without importing the compiler at all.

Where the levels come from
--------------------------

The four levels are Qiskit's `optimization_level` on `transpile()`. The
mapping below was read from the shipped 1.2.4 preset plugins rather than from
the prose documentation, because the two disagree about one pass -- see
"Qiskit's own divergence" below. The evidence is
`qiskit/transpiler/preset_passmanagers/builtin_plugins.py`, whose
`OptimizationPassManager.pass_manager` and `DefaultInitPassManager.pass_manager`
name the passes per level:

===========  =========================================================
 level        Qiskit 1.2.4 names in the optimization and init stages
===========  =========================================================
 ``0``        the optimization stage is `None`; the init stage runs only
              `generate_unroll_3q` when a layout or coupling map is set
 ``1``        `Optimize1qGatesDecomposition`, `InverseCancellation`
 ``2``        `Collect2qBlocks`, `ConsolidateBlocks`, `UnitarySynthesis`,
              then a loop of `Optimize1qGatesDecomposition` and
              `CommutativeCancellation`; the init stage adds
              `RemoveDiagonalGatesBeforeMeasure`
 ``3``        the same list with `UnitarySynthesis` inside the loop
              instead of in the prologue, over a `MinimumPoint` loop
===========  =========================================================

Anchors, all in Qiskit 1.2.4: `builtin_plugins.py:529` opens the optimization
stage plugin and returns `None` for level 0 at :624; the per-level `_opt` lists
are at :561, :578 and :584; the init stage's level 2/3 block is at :144.

This ladder is the FlagQuantum reading of that table, not a transcription of
it. Two differences are deliberate and both are recorded in
`flagquantum/compiler/README.md`:

1. **Level 3 is reserved, not implemented.** Qiskit's level 3 differs from its
   level 2 only by moving `UnitarySynthesis` into the loop, and this package's
   loop has no unitary-synthesis stage to move. Refusing level 3 is the honest
   encoding of that: a level that silently ran level 2's passes while reporting
   3 would be a compatibility promise this release cannot keep, and narrowing
   the accepted set later is a breaking change while widening it is not.
2. **The unroll-to-basis step the levels gate in Qiskit is not here.** In
   Qiskit every level's loop carries an `_unroll_if_out_of_basis` conditional
   controller, because the pass manager rewrites into a target basis as it
   optimizes. In this package basis translation is a separate entry point with
   its own refusal behaviour, so a level changes the target-independent pass
   set and nothing else.

Qiskit's own divergence between its documentation and its code, measured while
writing this table and worth recording because it cost a wrong first mapping:
the docstring of `level_1_pass_manager` advertises "adjacent gate collapse and
redundant reset removal", but `generate_pre_op_passmanager` is called with
`remove_reset_in_zero=False` in `level1.py:85`, `level2.py:84` and
`level3.py:101`, so in 1.2.4 no preset level actually runs
`RemoveResetInZeroState`. This ladder keeps `remove_zero_state_resets` in level
1 because the pass belongs to level 1's subject matter -- work that provably
cannot be observed, as opposed to work that can be merged -- and notes the
divergence rather than following a flag that contradicts its own docstring.

The mapping from Qiskit's named passes onto the passes this package has:

=================================  ==============================================
 Qiskit 1.2.4                       FlagQuantum
=================================  ==============================================
 `InverseCancellation`              `merge_self_inverse`,
                                    `merge_inverse_pairs`,
                                    `remove_identity_gates`
 `Optimize1qGatesDecomposition`      `merge_adjacent_rotations`,
                                    `collapse_one_qubit_runs`
 `CommutativeCancellation`           `merge_commuting_rotations`,
                                    `cancel_commuting_self_inverse`
 `RemoveDiagonalGatesBeforeMeasure`  `remove_diagonal_gates_before_measure`
 `Collect2qBlocks` +                 `collapse_two_qubit_blocks`
 `ConsolidateBlocks`
 `RemoveResetInZeroState`            `remove_zero_state_resets`
=================================  ==============================================

Level 1 runs the first two rows and `RemoveResetInZeroState`'s; level 2 adds the
rest. That is why level 1 excludes
`remove_diagonal_gates_before_measure` -- Qiskit adds
`RemoveDiagonalGatesBeforeMeasure` in the init stage only for levels 2 and 3 --
and excludes both the commutation pair and the two-qubit block pass, which
Qiskit also places at 2 and above. Level 1's subject matter is therefore exactly
"merge what is adjacent, and drop what cannot be observed", with no gate ever
moved past another.
"""

from __future__ import annotations

from ..errors import CompilationError

#: The level `optimize` runs when a caller does not name one. It is the level
#: that runs every pass of a fixed-point round, so the default is the behaviour
#: this entry point already had before the parameter existed.
DEFAULT_OPTIMIZATION_LEVEL = 2

#: The levels this release accepts. `optimize(program, optimization_level=n)`
#: is defined for exactly these values.
IMPLEMENTED_OPTIMIZATION_LEVELS: tuple[int, ...] = (0, 1, 2)

#: The levels the ladder declares but this release refuses. A reserved level is
#: not an alias for a smaller one: it is a level whose pass set is not available,
#: and it fails closed with the reason.
RESERVED_OPTIMIZATION_LEVELS: tuple[int, ...] = (3,)

#: Every level the ladder names, accepted or reserved. This is the domain a
#: caller reads to learn that level 3 exists and is refused, which the accepted
#: set alone cannot express.
DECLARED_OPTIMIZATION_LEVELS: tuple[int, ...] = (
    IMPLEMENTED_OPTIMIZATION_LEVELS + RESERVED_OPTIMIZATION_LEVELS
)

#: Why each reserved level is refused, keyed by level. The text is the error a
#: caller sees, so it names the stage that is missing rather than only saying no.
_RESERVED_REASONS: dict[int, str] = {
    3: (
        "level 3 needs the unitary-synthesis stage Qiskit's level 3 moves into "
        "the optimization loop, and this package has no unitary-synthesis pass "
        "to move; level 2 runs every pass this release has"
    ),
}

#: The passes each accepted level runs, once per fixed-point round, in the order
#: it runs them. Level 2 is the complete round body and the default; level 1 is
#: the subsequence that collapses adjacent gates and removes work that cannot be
#: observed, without ever commuting a gate past another.
#:
#: The order carries reasons, and they are the reasons this table -- not the
#: loop -- is where a reader finds them:
#:
#: - `remove_zero_state_resets` runs first because a reset it can remove is
#:   removable whatever the passes below do, and removing it hands them a shorter
#:   program. Its own reach is what needs the loop: `x(0) x(0) reset(0)` only
#:   exposes the reset once the pair cancels.
#: - `remove_diagonal_gates_before_measure` runs second for the same reason: a
#:   diagonal gate read out only by measurements is unobservable whatever the
#:   passes below do, and dropping it hands them a shorter program.
#: - `merge_self_inverse` then `merge_inverse_pairs` then
#:   `merge_adjacent_rotations`, which is the order in which each one's result can
#:   expose the next one's pattern.
#: - `merge_commuting_rotations` sits straight after the adjacent merge because it
#:   is the same arithmetic on the same parameter and differs only in how far
#:   apart the halves may sit.
#: - `collapse_one_qubit_runs` runs after the commutation passes, and
#:   `collapse_two_qubit_blocks` runs last, so that a block the two-qubit pass
#:   composes is one the single-qubit passes have already had: the two-qubit pass
#:   folds the single-qubit gates it draws in as members, and keeps those members
#:   the smallest spelling of themselves. Its own reach needs the loop, because
#:   the `cz` that replaces `swap cz swap` is a gate the next round can then
#:   cancel against a neighbour.
#: - `remove_identity_gates` appears twice on purpose: once after the inversion
#:   passes and once after the commutation passes, because a commute can be what
#:   puts two cancelling halves next to each other.
OPTIMIZATION_LEVEL_STAGES: dict[int, tuple[str, ...]] = {
    0: (),
    1: (
        "remove_zero_state_resets",
        "remove_identity_gates",
        "merge_self_inverse",
        "merge_inverse_pairs",
        "merge_adjacent_rotations",
        "collapse_one_qubit_runs",
    ),
    2: (
        "remove_zero_state_resets",
        "remove_diagonal_gates_before_measure",
        "remove_identity_gates",
        "merge_self_inverse",
        "merge_inverse_pairs",
        "merge_adjacent_rotations",
        "merge_commuting_rotations",
        "cancel_commuting_self_inverse",
        "remove_identity_gates",
        "collapse_one_qubit_runs",
        "collapse_two_qubit_blocks",
    ),
}


def _is_level_number(level: object) -> bool:
    """Whether `level` is an integer level rather than something that coerces.

    `bool` is deliberately excluded even though it is an `int` subclass:
    `optimization_level=True` would otherwise be accepted as level 1, which
    silently turns the two-valued spelling of a stronger request into a weaker
    level. A float is excluded for the same reason, since `2.0 == 2` would make
    an unvalidated value look validated.
    """

    return isinstance(level, int) and not isinstance(level, bool)


def optimization_level_stages(level: int) -> tuple[str, ...]:
    """Return the pass names one fixed-point round runs at `level`.

    Raises `CompilationError` for a level this release reserves and for anything
    that is not a level number, so an unsupported request fails at the entry
    point rather than producing a program that quietly carries the wrong amount
    of optimization.
    """

    if not _is_level_number(level):
        raise CompilationError(
            f"optimization level must be an integer, got {level!r}; "
            f"this release implements {IMPLEMENTED_OPTIMIZATION_LEVELS} "
            f"and reserves {RESERVED_OPTIMIZATION_LEVELS}"
        )
    if level in OPTIMIZATION_LEVEL_STAGES:
        return OPTIMIZATION_LEVEL_STAGES[level]
    if level in _RESERVED_REASONS:
        raise CompilationError(
            f"optimization level {level} is reserved, not implemented: "
            f"{_RESERVED_REASONS[level]}"
        )
    raise CompilationError(
        f"unknown optimization level {level}; this release implements "
        f"{IMPLEMENTED_OPTIMIZATION_LEVELS} and reserves "
        f"{RESERVED_OPTIMIZATION_LEVELS}"
    )


__all__ = (
    "DECLARED_OPTIMIZATION_LEVELS",
    "DEFAULT_OPTIMIZATION_LEVEL",
    "IMPLEMENTED_OPTIMIZATION_LEVELS",
    "OPTIMIZATION_LEVEL_STAGES",
    "RESERVED_OPTIMIZATION_LEVELS",
    "optimization_level_stages",
)
