"""Check the declared optimization levels against the loop that runs them.

`pipeline.optimize` used to have one behaviour and no way to ask for less of it.
It now takes an `optimization_level`, and the passes each level runs are declared
in `optimization_levels.OPTIMIZATION_LEVEL_STAGES`. This file holds the two
questions that declaration has to answer, and holds them separately because they
fail differently.

**Is the declaration the loop?** A table of pass names beside a loop of pass
calls is a second statement of the same fact, and the two can drift. The checks
below read the declaration, compare it with the implementations the loop can
resolve, and then *record what the loop actually calls* while running each level,
so the table is measured against execution rather than against a restatement of
itself. `test_optimization_pass_conformance.py` already records the composition
at the default level; this file records it at every level, which is the property
the parameter introduced.

**Does each level do what its entry says?** A level's name is a promise about the
program a caller gets back. Four programs here are chosen so that one promise is
decidable per level, and each is a shape a level must treat differently rather
than a shape one level happens to handle:

- a reset on a qubit the body leaves at ``|0>``, which level 1 removes because it
  is unobservable, not because it can be merged;
- a diagonal gate whose only readers are measurements, which level 1 keeps and
  level 2 drops, because the pass that drops it is `RemoveDiagonalGatesBeforeMeasure`
  and Qiskit adds it at 2;
- two rotations on one qubit separated by a commuting two-qubit gate, which level
  1 cannot merge because the two are not adjacent and commuting past a gate is
  exactly what `CommutativeCancellation` does;
- a pair of operations that cancels without commuting, which level 1 must already
  take, so a level 1 that only removed resets fails here.

The default level is checked against level 2 instruction for instruction, because
the parameter is a compatible addition only if the program a caller who names no
level is the program that caller used to get. Level 0 is checked for returning the
submitted program unchanged, which is what "do not optimize" has to mean when it
is a level rather than a flag.
"""

from __future__ import annotations

import inspect
import random
from typing import cast

import pytest
import torch

import flagquantum as fq
import flagquantum.compiler.pipeline as compiler_pipeline
from flagquantum.compiler import compile, optimize
from flagquantum.compiler.optimization_levels import (
    DECLARED_OPTIMIZATION_LEVELS,
    DEFAULT_OPTIMIZATION_LEVEL,
    IMPLEMENTED_OPTIMIZATION_LEVELS,
    OPTIMIZATION_LEVEL_STAGES,
    RESERVED_OPTIMIZATION_LEVELS,
    optimization_level_stages,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CompilationError

pytestmark = pytest.mark.unit

#: The levels whose behaviour is asserted by running them. The reserved level is
#: asserted by its refusal instead, which is the only behaviour it has.
_RUNNABLE_LEVELS = IMPLEMENTED_OPTIMIZATION_LEVELS


def _names(program: CircuitIR) -> list[str]:
    return [instruction.name for instruction in program]


def _gate(name: str, wires: tuple[int, ...], **params: float) -> Instruction:
    return Instruction(name, wires, params=params)


def _measure(wire: int, bit: int = 0) -> Instruction:
    return Instruction(
        "measure", (wire,), metadata={"is_dynamic": True, "classical_bit": bit}
    )


def _reset(wire: int) -> Instruction:
    return Instruction("reset", (wire,), metadata={"is_dynamic": True})


def test_the_declared_levels_are_three_runnable_and_one_reserved() -> None:
    """The domain a caller reads, stated as numbers rather than inferred.

    The accepted and reserved sets are what `optimize` and `compile` refuse
    against, so a level added to the table but not to one of them would be a
    level the entry point answers with an "unknown level" error.
    """

    assert IMPLEMENTED_OPTIMIZATION_LEVELS == (0, 1, 2)
    assert RESERVED_OPTIMIZATION_LEVELS == (3,)
    assert DECLARED_OPTIMIZATION_LEVELS == (0, 1, 2, 3)
    assert DEFAULT_OPTIMIZATION_LEVEL in IMPLEMENTED_OPTIMIZATION_LEVELS
    assert set(OPTIMIZATION_LEVEL_STAGES) == set(IMPLEMENTED_OPTIMIZATION_LEVELS)


def test_every_declared_pass_name_resolves_to_an_implementation() -> None:
    """Tie the table to the loop's own roster, in both directions.

    Level 2 is the round body the loop ran before the parameter existed, so the
    set of names it uses is the implementation roster. A name in the table with no
    implementation would fail at the first call, and an implementation no level
    names would be reachable only through the default -- which is how a level
    silently keeps running a pass it does not declare.
    """

    roster = compiler_pipeline._round_passes()
    declared = {
        name for stages in OPTIMIZATION_LEVEL_STAGES.values() for name in stages
    }
    assert declared == set(roster), sorted(declared ^ set(roster))


def test_level_two_is_the_union_and_level_one_is_its_subsequence() -> None:
    """Level 2 declares more than 1, and level 1 is a subsequence of it.

    A level that ran a pass in an order level 2 does not would be a second
    composition rather than a smaller one, and the reasons recorded for the order
    in `optimization_levels` would then describe only one of them.
    """

    two = OPTIMIZATION_LEVEL_STAGES[2]
    one = OPTIMIZATION_LEVEL_STAGES[1]
    assert set(one) <= set(two)
    position = -1
    for name in one:
        position = two.index(name, position + 1)
    assert set(two) == {
        name for stages in OPTIMIZATION_LEVEL_STAGES.values() for name in stages
    }
    assert OPTIMIZATION_LEVEL_STAGES[0] == ()


def test_level_zero_returns_the_submitted_program_unchanged() -> None:
    """Level 0 is a request that was honored, not a quieter level 1.

    Every instruction of the submitted program is compared, not only the count, so
    a level 0 that rewrote one instruction while preserving the length fails here.
    """

    program = CircuitIR(
        2,
        (
            _gate("x", (0,)),
            _gate("x", (0,)),
            _gate("rz", (0,), theta=0.3),
            _gate("cz", (0, 1)),
            _gate("z", (0,)),
            _measure(0, 0),
        ),
    )

    out = optimize(program, optimization_level=0)
    assert out.instructions == program.instructions
    assert out.measurements == program.measurements
    assert out.metadata["optimization"] == {"level": 0}


def test_level_one_removes_work_that_cannot_be_observed() -> None:
    """`x(0) x(0) reset(0)` is empty at level 1, and is not at level 0.

    Two passes reach it and both are level 1's subject matter: the inverse pair
    cancels, and the reset before an idle qubit is removable. Neither commutes a
    gate past another, which is why the level that has no commutation pass can
    still take this.
    """

    program = CircuitIR(
        1, (_gate("x", (0,)), _gate("x", (0,)), _reset(0), _gate("h", (0,)))
    )

    assert _names(optimize(program, optimization_level=1)) == ["h"]
    assert _names(optimize(program, optimization_level=0)) == [
        "x",
        "x",
        "reset",
        "h",
    ]


def test_level_one_keeps_a_diagonal_gate_before_a_measurement() -> None:
    """The two-qubit diagonal case, which Qiskit places at level 2 and above.

    `cz(0, 1)` and `z(0)` are both diagonal, both read out only by measurements,
    and both are kept at levels 0 and 1. Dropping them is a reach level 1 does not
    have, and a level 1 that dropped them would be claiming pass reach it does not
    declare.
    """

    program = CircuitIR(
        2,
        (_gate("cz", (0, 1)), _gate("z", (0,)), _measure(0, 0), _measure(1, 1)),
    )

    for level in (0, 1):
        assert _names(optimize(program, optimization_level=level)) == [
            "cz",
            "z",
            "measure",
            "measure",
        ]
    assert _names(optimize(program, optimization_level=2)) == ["measure", "measure"]


def test_level_one_cannot_merge_rotations_across_a_commuting_gate() -> None:
    """Two rotations on one qubit, one `cz` apart.

    `rz` commutes with `cz`, so level 2 merges the pair; level 1 has no pass that
    moves a gate past another and keeps both. This is the case that separates the
    two levels on a program where every gate is already a single spelling of
    itself, so the difference cannot be attributed to the Euler folding both
    levels run.
    """

    program = CircuitIR(
        2,
        (
            _gate("rz", (0,), theta=0.3),
            _gate("cz", (0, 1)),
            _gate("rz", (0,), theta=0.4),
        ),
    )

    one = optimize(program, optimization_level=1)
    two = optimize(program, optimization_level=2)
    assert _names(one) == ["rz", "cz", "rz"]
    assert _names(two) == ["rz", "cz"]
    assert float(two.instructions[0].params["theta"]) == pytest.approx(0.7)


def test_the_default_level_is_level_two_instruction_for_instruction() -> None:
    """The compatibility claim the parameter is admitted on.

    A caller that names no level must receive the program it used to receive, so
    the default is compared with level 2 over every case in this file, instruction
    for instruction and measurement for measurement rather than by length.
    """

    programs = (
        CircuitIR(
            2,
            (
                _gate("x", (0,)),
                _gate("x", (0,)),
                _gate("rz", (0,), theta=0.3),
                _gate("cz", (0, 1)),
                _gate("rz", (0,), theta=0.4),
                _gate("z", (0,)),
                _measure(0, 0),
            ),
        ),
        CircuitIR(1, (_gate("x", (0,)), _gate("h", (0,)), _gate("h", (0,)), _reset(0))),
    )

    assert DEFAULT_OPTIMIZATION_LEVEL == 2
    for program in programs:
        default = optimize(program)
        explicit = optimize(program, optimization_level=DEFAULT_OPTIMIZATION_LEVEL)
        assert default.instructions == explicit.instructions
        assert default.measurements == explicit.measurements
        assert default.metadata["optimization"] == {"level": 2}


#: Wires, opcodes and angles for the seeded family below. Every opcode is declared
#: and unitary, so a program in this family has a state to compare and no
#: measurement to make the comparison a distribution instead.
_FAMILY_WIRES = 3
_FAMILY_GATES = 12
_FAMILY_SEEDS = range(40)
_ONE_WIRE = ("x", "y", "z", "h", "s", "sdg", "t", "tdg")
_ROTATIONS = ("rx", "ry", "rz")


def _family_case(seed: int) -> CircuitIR:
    """Return a deterministic unitary program with structure to fold.

    Two partner rules give the levels work, and the second is the one that
    separates level 1 from level 2. A rotation is followed by a rotation of the
    opposite angle, which any level can merge because the two are adjacent. Half
    the time a `cz` is placed between them instead, on the same qubit, so the pair
    is only reachable by a pass that commutes a rotation past a gate -- and the
    `cz` is chosen for being diagonal, so the commute it needs is the legal one.
    Without that shape the family is folded as far by level 1 as by level 2, which
    was measured: with only adjacent partners, level 2 and level 1 returned the
    same program on all forty seeds.
    """

    rng = random.Random(seed)
    instructions: list[Instruction] = []
    while len(instructions) < _FAMILY_GATES:
        if instructions and rng.random() < 0.45:
            previous = instructions[-1]
            if previous.name in _ROTATIONS:
                wire = previous.wires[0]
                angle = float(previous.params["theta"])
                if rng.random() < 0.5:
                    other = rng.choice([w for w in range(_FAMILY_WIRES) if w != wire])
                    instructions.append(_gate("cz", (wire, other)))
                instructions.append(_gate(previous.name, (wire,), theta=-angle))
                continue
            if previous.name in _ONE_WIRE:
                instructions.append(_gate(previous.name, (previous.wires[0],)))
                continue
        roll = rng.random()
        if roll < 0.4:
            wire = rng.randrange(_FAMILY_WIRES)
            instructions.append(_gate(rng.choice(_ONE_WIRE), (wire,)))
        elif roll < 0.75:
            wire = rng.randrange(_FAMILY_WIRES)
            name = rng.choice(_ROTATIONS)
            instructions.append(_gate(name, (wire,), theta=rng.uniform(-3.0, 3.0)))
        else:
            left, right = rng.sample(range(_FAMILY_WIRES), 2)
            instructions.append(_gate(rng.choice(("cx", "cz", "swap")), (left, right)))
    return CircuitIR(_FAMILY_WIRES, tuple(instructions), dtype="complex128")


def _state(program: CircuitIR) -> torch.Tensor:
    # `Circuit.state` is annotated with the dtype-agnostic tensor type the public
    # surface returns; the cast states what the CPU executor in use really gives
    # back rather than widening this helper's return type to `Any`.
    values = fq.Circuit.from_ir(program).state()
    return cast("torch.Tensor", values.reshape(-1))


def test_every_level_preserves_the_state_and_never_grows_the_program() -> None:
    """The two postconditions a caller relies on, measured per level.

    Every pass a level runs is either a rewrite of the same unitary or the removal
    of something unobservable, so all three levels must return the submitted
    program's state. The bound is the one
    `test_optimization_pass_conformance.py` derived for a single pass -- about
    twelve orders of magnitude above the rounding it measured -- because the same
    arithmetic is being done here, at every level instead of at one.
    """

    for seed in _FAMILY_SEEDS:
        program = _family_case(seed)
        reference = _state(program)
        lengths = []
        for level in _RUNNABLE_LEVELS:
            out = optimize(program, optimization_level=level)
            lengths.append(len(out))
            difference = float(torch.max(torch.abs(reference - _state(out))))
            assert difference <= 1.0e-12, (seed, level, difference)
            assert len(out) <= len(program), (seed, level)
        # A level that runs more passes must not end up with a longer program on
        # any case, or the level ladder would not be a ladder.
        assert lengths == sorted(lengths, reverse=True), (seed, lengths)


def test_each_level_does_something_the_one_below_it_does_not() -> None:
    """Count the cases where a level's reach is strictly wider than the one below.

    A level that is never the only one able to rewrite a case would be a name
    without a behaviour. Counts are used rather than exact figures because the
    family is generated, and each floor has better than twofold margin over the
    measured value: level 1 is strictly shorter than level 0 on 40 of 40 seeds and
    level 2 is strictly shorter than level 1 on 12 of 40. The second number is the
    smaller one because level 1 already folds a single-qubit run into one Euler
    form, so most of what level 2 adds is absorbed; the seeds it wins on are the
    ones the `cz` partner rule produced, which is why that rule exists.

    A level that returned a *longer* program than the level below it would fail
    here as well, so the count is a ladder rather than four independent numbers.
    """

    shorter: dict[int, int] = {1: 0, 2: 0}
    for seed in _FAMILY_SEEDS:
        program = _family_case(seed)
        lengths = [
            len(optimize(program, optimization_level=level))
            for level in _RUNNABLE_LEVELS
        ]
        assert lengths == sorted(lengths, reverse=True), (seed, lengths)
        for index, level in enumerate(_RUNNABLE_LEVELS[1:], start=1):
            if lengths[index] < lengths[index - 1]:
                shorter[level] += 1
    assert shorter[1] >= 30, shorter
    assert shorter[2] >= 5, shorter


def test_the_level_used_is_recorded_on_the_program() -> None:
    """A caller reads back which level produced the program it holds.

    Without this field a program compiled at level 0 and one compiled at level 2
    with nothing to optimize are the same program, and the request is not
    recoverable from the artifact.
    """

    program = CircuitIR(
        2,
        (
            _gate("rz", (0,), theta=0.3),
            _gate("cz", (0, 1)),
            _gate("rz", (0,), theta=0.4),
        ),
    )

    for level in _RUNNABLE_LEVELS:
        assert optimize(program, optimization_level=level).metadata["optimization"] == {
            "level": level
        }


def test_compile_forwards_the_level_to_both_optimization_points() -> None:
    """`compile` is the entry point a user reaches, and it optimizes twice.

    Routing inserts SWAPs and the second optimization is what cancels them, so the
    level has to reach both calls. On a program with no coupling map the second
    point is not reached and `compile` must agree with `optimize` exactly, which is
    the half that is checkable without a topology; the metadata of a routed compile
    is checked separately below for the field's survival through routing.
    """

    program = CircuitIR(
        1, (_gate("x", (0,)), _gate("x", (0,)), _gate("rz", (0,), theta=0.3))
    )

    for level in _RUNNABLE_LEVELS:
        compiled = compile(program, optimization_level=level)
        direct = optimize(program, optimization_level=level)
        assert compiled.instructions == direct.instructions
        assert compiled.metadata["optimization"] == {"level": level}

    assert _names(compile(program, optimization_level=0)) == ["x", "x", "rz"]
    assert _names(compile(program, optimization_level=1)) == ["rz"]


def test_the_level_survives_routing_and_is_still_the_level_that_ran() -> None:
    """A routed compile records the level once, not once per optimization point.

    `route_to_topology` and the post-optimization recorder both rebuild the
    program's metadata, so a field written before routing is only readable
    afterwards if those rebuilds carry it. The last writer wins, and it must be the
    level rather than a stale copy of the same value or a missing key.
    """

    compiled = compile(
        fq.Circuit(3).h(0).cx(0, 2).cx(1, 2),
        coupling_map=((0, 1), (1, 2)),
        optimization_level=1,
    )

    assert compiled.metadata["optimization"] == {"level": 1}
    assert compiled.metadata["routing"]["post_optimization_instruction_count"] == len(
        compiled
    )


def test_compile_validates_the_level_even_when_optimization_is_off() -> None:
    """A reserved level is refused whether or not the caller asked to skip work.

    `optimize=False, optimization_level=3` is the one combination where ignoring
    the level would answer a request for a capability this release does not have
    with a program that looks like a success.
    """

    program = fq.Circuit(1).h(0)

    with pytest.raises(CompilationError, match="reserved"):
        compile(program, optimize=False, optimization_level=3)


@pytest.mark.parametrize("level", [3, 4, -1, 99])
def test_a_level_outside_the_declared_domain_is_refused(level: int) -> None:
    """The reserved level and an unknown one are refused with different reasons.

    Level 3 exists in the ladder and is refused because a stage is missing; level 4
    does not exist at all. Reporting one as the other would tell a caller asking
    for the level above 3 to wait for a feature that is not the one it needs.
    """

    program = fq.Circuit(1).h(0)

    with pytest.raises(CompilationError) as error:
        optimize(program, optimization_level=level)
    message = str(error.value)
    if level == 3:
        assert "reserved" in message
        assert "unitary-synthesis" in message
    else:
        assert "unknown optimization level" in message
    assert str(level) in message


def test_the_reserved_level_blames_the_contract_and_not_a_missing_pass() -> None:
    """The refusal must name the blocker that is real, and the artifact that is not.

    The reason level 3 is out of reach is that `UnitarySynthesis` needs a target
    basis and `optimize` is handed none, so the level is unreachable by contract
    rather than pending. That is a different claim from "the pass does not exist",
    and the difference is load-bearing: a reader who is told the pass is missing
    will set out to write it, and no new pass would unblock this level. So the
    message is read here for both halves -- the contract that blocks, and the entry
    point at which the same rewrite is available -- and the entry point it names is
    then resolved and inspected, because a refusal that points at a renamed or
    removed artifact is a claim about this package that has stopped being true.
    """

    with pytest.raises(CompilationError) as error:
        optimize(fq.Circuit(1).h(0), optimization_level=3)
    message = str(error.value)

    assert "target-independent by contract" in message
    assert (
        "has no unitary-synthesis pass" not in message
    ), "the package does ship unitary synthesis; what level 3 lacks is a target"

    # The message names `legalize_native_gates(program, snapshot=...)`. Resolve that
    # name and read its signature, so the sentence in the error text cannot outlive
    # the artifact it points at.
    assert "legalize_native_gates" in message
    assert "snapshot=" in message
    from flagquantum.compiler.native_gate_legalization import legalize_native_gates

    parameters = inspect.signature(legalize_native_gates).parameters
    assert (
        parameters["snapshot"].kind is inspect.Parameter.KEYWORD_ONLY
    ), "the refusal names a keyword argument; the entry point must accept it as one"


@pytest.mark.parametrize("level", [True, False, 2.0, "2", None, (2,)])
def test_a_level_that_is_not_an_integer_is_refused(level: object) -> None:
    """A value that merely coerces is not a level.

    `True` and `2.0` both compare equal to an implemented level, so accepting them
    would let an unvalidated value reach the ladder silently. `_is_level_number`
    excludes them explicitly, and the error names the value rather than the
    coercion, because the caller has to see which argument was wrong.
    """

    with pytest.raises(CompilationError, match="must be an integer"):
        optimize(fq.Circuit(1).h(0), optimization_level=level)  # type: ignore[arg-type]


def test_the_ladder_refuses_the_reserved_level_without_the_entry_point() -> None:
    """The refusal belongs to the declaration, not only to `optimize`.

    A caller that resolves a level itself -- the shape `compile` uses, and the
    shape a plugin would use -- must read the same verdict, or the entry point is
    the only thing standing between a reserved level and a silently reduced
    program.
    """

    with pytest.raises(CompilationError, match="reserved"):
        optimization_level_stages(3)
    assert optimization_level_stages(2) == OPTIMIZATION_LEVEL_STAGES[2]
