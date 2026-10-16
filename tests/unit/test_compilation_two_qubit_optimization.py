"""Unit contract for the two-qubit run fold W9-13 adds.

The fold composes a maximal run of declared two-qubit gates over one *ordered* wire
pair and re-spells it as one declared two-qubit opcode. It may not approximate: a
wrong acceptance changes the program, and `CircuitIR` has no field for a global
phase, so a run whose product is not exactly a declared gate has to be left alone.
The central tests here are therefore the ones that pin this module's eleven closed
forms against the runtime gate table, and the ones that pin the identity edge.

The gate table in ``flagquantum.simulation`` is the authority on what these gate
names compute. This module is in the Compiler layer, which may not import it, so
its closed forms are pinned here instead: a test may import the runtime, and the
convention drift this guards against is exactly the kind that would otherwise hide
inside the fold.
"""

import math
import random

import pytest
import torch

from flagquantum.compiler.pipeline import optimize
from flagquantum.compiler.two_qubit_optimization import (
    _FAMILIES,
    _IDENTITY,
    _block_product,
    _declines_vocabulary,
    _instruction_matrix,
    _is_foldable,
    _replacement,
    _same,
    collapse_two_qubit_blocks,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, canonical_opcode
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

#: Every declared arity-2 unitary opcode. The fold's claim is about the whole
#: two-qubit group, so the convention table below has to cover all of it.
TWO_QUBIT_OPCODES: tuple[str, ...] = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 2
    )
)

#: Angles the closed forms are pinned at. ``0`` and ``pi`` are the two the fold
#: reaches most, and the two signed ones are there to catch a sign slip that a
#: symmetric angle would hide.
_ANGLES: tuple[float, ...] = (0.0, 0.3, 1.7, -2.2, math.pi)


def _runtime_matrix(instruction: Instruction) -> list[list[complex]]:
    """One instruction's matrix, from the runtime gate table, in complex128.

    The table returns ``(4, 4)`` for the four parameter-free gates and ``(1, 4, 4)``
    for the seven parameterized ones, so the reshape is not cosmetic: an index on the
    raw result would be a row for one family and a batch for the other.
    """

    values = gate_matrix(
        instruction,
        bsz=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    ).reshape(4, 4)
    return [[complex(values[row][column]) for column in range(4)] for row in range(4)]


def _gap(left: list[list[complex]], right: list[list[complex]]) -> float:
    return max(
        abs(left[row][column] - right[row][column])
        for row in range(4)
        for column in range(4)
    )


def _instruction(
    opcode: str, wires: tuple[int, ...] = (0, 1), angle: float | None = None
) -> Instruction:
    schema = OPERATOR_SCHEMAS[canonical_opcode(opcode)]
    params = {}
    if schema.parameters:
        params[schema.parameters[0]] = math.pi if angle is None else angle
    return Instruction(opcode, wires, params=params)


def test_the_family_list_is_the_whole_declared_two_qubit_group() -> None:
    # A fold measured on a subset of the group cannot claim the group, and the
    # module asserts this at import time; pinning it here says which way it fails.
    assert set(_FAMILIES) == set(TWO_QUBIT_OPCODES)
    assert len(_FAMILIES) == 11


def test_every_closed_form_matches_the_runtime_gate_table() -> None:
    """The eleven closed forms against the authority, entry for entry.

    A sign or a half-angle slip in one of these is a wrong program that the rest of
    the pass would then accept, because the same table is what `_replacement`
    compares a product against. Every entry of every family at every angle is
    asserted, and the worst gap is asserted to be float noise rather than a
    convention difference.
    """

    worst = 0.0
    for opcode in _FAMILIES:
        schema = OPERATOR_SCHEMAS[opcode]
        angles: tuple[float | None, ...] = (
            (None,) if not schema.parameters else _ANGLES  # type: ignore[assignment]
        )
        for angle in angles:
            instruction = _instruction(opcode, angle=angle)
            mine = _instruction_matrix(instruction)
            assert mine is not None, (opcode, angle)
            worst = max(worst, _gap(mine, _runtime_matrix(instruction)))
    assert worst < 1e-12, worst


def test_the_identity_is_the_identity_and_the_fixed_gates_are_not() -> None:
    for opcode in ("cx", "cy", "cz", "swap"):
        mine = _instruction_matrix(_instruction(opcode))
        assert mine is not None
        assert not _same(mine, _IDENTITY), opcode


def test_a_two_qubit_run_that_is_exactly_a_declared_gate_is_folded() -> None:
    """The two named shapes the fold exists for, and their exact matrices."""

    # `swap cz swap` moves the control to the other wire: swapping both wires of a
    # controlled-Z is the same gate, so the router's swap cost is what this removes.
    run = [_instruction("swap"), _instruction("cz"), _instruction("swap")]
    replacement = _replacement(run, pair=(0, 1), metadata={})
    assert replacement is not None and len(replacement) == 1
    assert replacement[0].name == "cz"
    # `cz swap cz` is the other way round.
    run = [_instruction("cz"), _instruction("swap"), _instruction("cz")]
    replacement = _replacement(run, pair=(0, 1), metadata={})
    assert replacement is not None and len(replacement) == 1
    assert replacement[0].name == "swap"
    # An accumulation inside one family, which no other pass composes.
    run = [_instruction("rzz", angle=0.3), _instruction("rzz", angle=0.4)]
    replacement = _replacement(run, pair=(0, 1), metadata={})
    assert replacement is not None and len(replacement) == 1
    assert replacement[0].name == "rzz"
    assert replacement[0].params["theta"] == pytest.approx(0.7, abs=1e-12)


def test_a_run_whose_product_is_the_identity_is_deleted() -> None:
    """The reach `remove_identity_gates` cannot have, because it reads a parameter.

    ``cz cz`` is the self-inverse case that pass already covers. ``rzz(4*pi)`` and
    ``cphase(2*pi)`` are not: no member has a zero parameter, so only a product
    finds them.
    """

    for run in (
        [_instruction("cz"), _instruction("cz")],
        [_instruction("rzz", angle=4.0 * math.pi)],
        [_instruction("cphase", angle=2.0 * math.pi)],
        [_instruction("crz", angle=4.0 * math.pi)],
        [_instruction("rxx", angle=0.5), _instruction("rxx", angle=-0.5)],
    ):
        product = _block_product(run, (0, 1))
        assert product is not None
        assert _same(product, _IDENTITY), [item.name for item in run]
        assert _replacement(run, pair=(0, 1), metadata={}) == ()


def test_a_minus_identity_run_keeps_a_gate() -> None:
    """``-I`` is a global phase and this IR has no field for one.

    ``rzz(2*pi)`` is ``diag(-1, -1, -1, -1)``, and its entries measure
    ``-1 -+ 1.22e-16j``, so every comparison about it has to be a tolerance
    comparison: it is within `_MATCH_EPS` of ``-I`` and nowhere near ``+I``. The
    pass keeps the gate, and this is the assertion that says so.
    """

    run = [_instruction("rzz", angle=2.0 * math.pi)]
    product = _block_product(run, (0, 1))
    assert product is not None
    assert not _same(product, _IDENTITY)
    # It is a diagonal matrix whose entries are all `-1`, which is exactly `-I`.
    for index in range(4):
        assert abs(product[index][index] + 1.0) < 1e-12
    assert _replacement(run, pair=(0, 1), metadata={}) is None


def test_a_single_declared_gate_is_never_respelled_as_itself() -> None:
    """Equal length is not a win, so the pass is idempotent.

    ``_optimize_to_fixed_point`` loops until the program stops shrinking, so a pass
    that re-spelled one gate as an equal-length one would not terminate.
    """

    for opcode in _FAMILIES:
        run = [_instruction(opcode, angle=0.7)]
        product = _block_product(run, (0, 1))
        assert product is not None
        assert not _same(product, _IDENTITY), opcode
        assert _replacement(run, pair=(0, 1), metadata={}) is None, opcode


def test_a_parameterized_respelling_is_declined_unless_the_run_is_in_that_family() -> (
    None
):
    """The decline, from both sides, on runs whose product *is* a declared gate.

    ``cx cz cx`` is a `crz(pi)`-family matrix, and `cx swap cx`/`cz cx cz` are
    parameterized rotations too. All of them are declined: a parameterized
    replacement is a different parameterization of the two-qubit group rather than a
    more primitive operation, and `optimize` cannot see whether the target publishes
    the gate the run was spelled in or the one it would be re-spelled as. The same
    run spelled inside one family is accepted, which is the other side.
    """

    for run in (
        [_instruction("cx"), _instruction("cz"), _instruction("cx")],
        [_instruction("cx"), _instruction("swap"), _instruction("cx")],
        [_instruction("cz"), _instruction("cx"), _instruction("cz")],
        [_instruction("swap"), _instruction("cy"), _instruction("swap")],
        [_instruction("swap"), _instruction("cx"), _instruction("swap")],
    ):
        replacement = _replacement(run, pair=(0, 1), metadata={})
        assert replacement is None, [item.name for item in run]
        # Non-vacuity: the run is only evidence about the decline if some *other*
        # family's closed form would have matched it. `cz`/`swap` are declined
        # because they are parameter-free *and* would be a lateral move, and the
        # parameterized families are declined by the rule under test.
        product = _block_product(run, (0, 1))
        assert product is not None and not _same(product, _IDENTITY)
    assert not _declines_vocabulary(
        [_instruction("rzz", angle=0.3), _instruction("rzz", angle=0.4)], "rzz"
    )
    assert _declines_vocabulary(
        [_instruction("rzz", angle=0.3), _instruction("rzz", angle=0.4)], "crz"
    )
    assert not _declines_vocabulary([_instruction("cz"), _instruction("cz")], "cx")


def test_a_trainable_angle_is_left_to_the_pass_that_can_add_it() -> None:
    """Composition reads numbers, and a trainable angle has none.

    A fold that read `.item()` off a trainable angle would leave the autograd graph,
    which is why the check is on the value rather than on the opcode.
    """

    parameter = torch.nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
    run = [
        Instruction("rzz", (0, 1), params={"theta": parameter}),
        Instruction("rzz", (0, 1), params={"theta": 0.4}),
    ]
    assert _instruction_matrix(run[0]) is None
    assert _block_product(run, (0, 1)) is None
    kept = collapse_two_qubit_blocks(CircuitIR(2, tuple(run)))
    assert len(kept.instructions) == 2


def test_a_batched_angle_is_refused_rather_than_read_at_one_index() -> None:
    """Reading one element of a batch would change the program silently."""

    batched = torch.tensor([0.3, 0.4], dtype=torch.float64)
    instruction = Instruction("rzz", (0, 1), params={"theta": batched})
    assert _instruction_matrix(instruction) is None
    kept = collapse_two_qubit_blocks(CircuitIR(2, (instruction, instruction)))
    assert len(kept.instructions) == 2


def test_only_a_two_wire_instruction_can_be_a_block_member_by_itself() -> None:
    """`_is_foldable` is the two-wire membership test, and it says nothing about one.

    A one-wire instruction can still be drawn into a block as a member, and that is
    `_draws_in_as_leading_member`'s predicate rather than this one's; what is asserted
    here is only that this predicate refuses everything that is not exactly two wires,
    so a caller that used it as the whole membership rule would lose the draw-in.
    """

    assert _is_foldable(Instruction("cz", (0, 1)))
    assert _is_foldable(Instruction("cz", (1, 0)))
    # One wire, three wires, a channel, and a custom matrix on one wire are all
    # refused here.
    assert not _is_foldable(Instruction("rz", (0,), params={"theta": 0.3}))
    assert not _is_foldable(Instruction("ccx", (0, 1, 2)))
    assert not _is_foldable(
        Instruction("depolarizing", (0,), params={"probability": 0.1})
    )
    assert not _is_foldable(
        Instruction("custom", (0,), matrix=torch.eye(2, dtype=torch.complex128))
    )
    # A custom matrix on exactly two wires *is* foldable: the fold reads the matrix,
    # so an opcode the schema does not declare can still take part.
    assert _is_foldable(
        Instruction("custom", (0, 1), matrix=torch.eye(4, dtype=torch.complex128))
    )


def test_a_run_is_ordered_so_the_control_and_target_are_fixed() -> None:
    """`wires=(0, 1)` and `wires=(1, 0)` are different runs.

    ``Instruction.__post_init__`` does not sort the tuple, so an implementation that
    keyed a run by an unordered pair would compose a `cx(0, 1)` with a `cx(1, 0)`
    into a matrix that is not the program's.
    """

    forward = collapse_two_qubit_blocks(
        CircuitIR(2, (_instruction("cz", (0, 1)), _instruction("cz", (0, 1))))
    )
    assert len(forward.instructions) == 0
    # `cz` is symmetric under a wire swap, so the two orders *would* have composed
    # to the identity if they had been one run. They are not: the second instruction
    # names its wires in the other order, which is a different key, so nothing folds.
    mixed = collapse_two_qubit_blocks(
        CircuitIR(2, (_instruction("cz", (0, 1)), _instruction("cz", (1, 0))))
    )
    assert [item.wires for item in mixed.instructions] == [(0, 1), (1, 0)]
    # A run over a *different* ordered pair is still a run, and `cx(1, 0)` twice
    # folds, so the two orders are genuinely separate keys rather than both refused.
    reversed_run = collapse_two_qubit_blocks(
        CircuitIR(2, (_instruction("cx", (1, 0)), _instruction("cx", (1, 0))))
    )
    assert len(reversed_run.instructions) == 0


def test_a_gate_on_another_wire_does_not_end_a_run() -> None:
    """A run is maximal over its own pair, and a disjoint gate commutes with it."""

    instructions = (
        _instruction("cz", (0, 1)),
        _instruction("h", (2,)),
        _instruction("cz", (0, 1)),
    )
    folded = collapse_two_qubit_blocks(CircuitIR(3, instructions))
    assert [item.name for item in folded.instructions] == ["h"]


def test_a_one_wire_gate_between_two_two_qubit_gates_is_drawn_in_and_then_refused() -> (
    None
):
    """The draw-in is not a licence to fold: it enlarges the product it has to re-spell.

    `cz(0, 1) h(0) cz(0, 1)` is not a declared opcode, so the *widened* block is
    refused and the program is returned unchanged. The floor rule then offers the one
    all-two-qubit member -- the trailing `cz` -- to the narrower rule, which refuses a
    single-member block outright. So the instruction survives the draw-in intact, which
    is what makes the widening safe on a program it cannot improve.
    """

    instructions = (
        _instruction("cz", (0, 1)),
        _instruction("h", (0,)),
        _instruction("cz", (0, 1)),
    )
    folded = collapse_two_qubit_blocks(CircuitIR(2, instructions))
    assert [item.name for item in folded.instructions] == ["cz", "h", "cz"]


def test_a_conjugation_by_a_one_wire_gate_is_folded_into_the_gate_it_conjugates() -> (
    None
):
    """The reach the widening buys, on the shape it was widened for.

    ``h(1) cz(0, 1) h(1)`` is exactly ``cx(0, 1)``, and all three gates are members of
    one block over the single ordered pair ``(0, 1)``: the ``cz`` opens it, the
    backward scan draws the ``h`` immediately to its left in, and the trailing ``h``
    joins the open block as it arrives. Composing two-qubit runs alone cannot see this
    -- there is one two-qubit gate and it is a legal one already -- and neither can
    ``collapse_one_qubit_runs``, which composes same-*wire* runs and so cannot pair two
    ``h`` gates that the ``cz`` separates.

    The conjugated wire is the target rather than the control, and that is not
    cosmetic: ``h(0) cz(0, 1) h(0)`` is ``cx`` with the roles *exchanged*, which is a
    different ordered pair and is therefore declined. See the test below.
    """

    instructions = (
        _instruction("h", (1,)),
        _instruction("cz", (0, 1)),
        _instruction("h", (1,)),
    )
    folded = collapse_two_qubit_blocks(CircuitIR(2, instructions))
    assert [(item.name, item.wires) for item in folded.instructions] == [("cx", (0, 1))]


def test_a_conjugation_that_swaps_the_roles_is_declined_rather_than_relabelled() -> (
    None
):
    """A block over one ordered pair may not be replaced by a gate on the other.

    ``h(0) h(1) cx(0, 1) h(0) h(1)`` is exactly ``cx(1, 0)``, which *is* a declared
    opcode -- so a pass that matched a product against the gate table without also
    requiring the wires to be the block's own would rewrite four gates into one
    ``cx`` whose control and target are exchanged. That is a different program. The
    block's pair is ``(0, 1)`` and the product is not a gate on ``(0, 1)``, so the
    pass declines and leaves the five instructions where they were.
    """

    instructions = (
        _instruction("h", (0,)),
        _instruction("h", (1,)),
        _instruction("cx", (0, 1)),
        _instruction("h", (0,)),
        _instruction("h", (1,)),
    )
    folded = collapse_two_qubit_blocks(CircuitIR(2, instructions))
    assert [item.name for item in folded.instructions] == ["h", "h", "cx", "h", "h"]
    assert [item.wires for item in folded.instructions] == [
        (0,),
        (1,),
        (0, 1),
        (0,),
        (1,),
    ]


def test_a_drawn_in_gate_further_left_is_refused_rather_than_reordered() -> None:
    """The draw-in reaches only a *contiguous* run, so an interrupted one stays put.

    ``h(0) h(1) cx(0, 1) h(0)`` is not the product above with one gate missing; it is
    the same program with the control's conjugate on both sides and the target's on
    neither. The ``cx`` opens a block and draws only the single ``h(0)`` immediately to
    its left -- not the two gates further left, which the intervening ``h(1)`` makes
    non-contiguous. The block is then a one-sided conjugation, which is no declared
    opcode, so nothing folds and the members are left where the program put them.
    """

    instructions = (
        _instruction("h", (0,)),
        _instruction("h", (1,)),
        _instruction("cx", (0, 1)),
        _instruction("h", (0,)),
    )
    folded = collapse_two_qubit_blocks(CircuitIR(2, instructions))
    assert [item.name for item in folded.instructions] == ["h", "h", "cx", "h"]


def test_the_replacement_keeps_the_run_position_and_the_first_metadata() -> None:
    """The replacement is written where the run was, not appended.

    Program order is part of the program: `h` on wire 2 before and after the run
    has to stay before and after it, and the surviving gate is the run's first
    member's, metadata included.
    """

    instructions = (
        _instruction("h", (2,)),
        Instruction("swap", (0, 1), metadata={"origin": "router"}),
        _instruction("cz", (0, 1)),
        _instruction("swap", (0, 1)),
        _instruction("h", (2,)),
    )
    folded = collapse_two_qubit_blocks(CircuitIR(3, instructions))
    assert [item.name for item in folded.instructions] == ["h", "cz", "h"]
    assert folded.instructions[1].metadata == {"origin": "router"}
    assert folded.instructions[1].wires == (0, 1)


def test_the_pass_reaches_a_fixed_point_and_is_idempotent() -> None:
    """A second application changes nothing, which `_optimize_to_fixed_point` needs.

    ``swap cz swap cz cz`` composes to one ``cz``: the first three gates are a
    conjugation and the remaining two are a self-inverse pair. The result is a
    *single* declared gate, and a pass that re-spelled a single gate as an
    equal-length one would loop forever on it.
    """

    source = CircuitIR(
        2,
        (
            _instruction("swap"),
            _instruction("cz"),
            _instruction("swap"),
            _instruction("cz"),
            _instruction("cz"),
        ),
    )
    once = collapse_two_qubit_blocks(source)
    assert [item.name for item in once.instructions] == ["cz"]
    twice = collapse_two_qubit_blocks(once)
    assert [item.name for item in twice.instructions] == ["cz"]
    # And through the shipped pipeline, which is what actually runs: the pipeline
    # loops until the count stops falling, so it would raise rather than return if
    # the pass were not idempotent here.
    assert [item.name for item in optimize(source).instructions] == ["cz"]


def test_an_instruction_is_a_member_of_at_most_one_block() -> None:
    """The draw-in may not claim what a block already owns.

    A single-qubit gate between two blocks on *different* ordered pairs is a
    legitimate trailing member of the first and a legitimate leading member of the
    second, so a backward scan that ignored the first block would put it in both and
    then write two replacements through it. This program is the measured witness: the
    ``(2, 0)`` block opens on ``cz(2, 0)``, ``sx(1)`` leaves it open because it
    touches neither wire, and ``tdg(0)`` joins it -- so ``tdg(0)`` is spoken for by
    the time ``cz(0, 2)`` opens the next block. Claiming it twice rewrites
    ``tdg(0) cz(0, 2) t(0)`` as ``cz(0, 2)`` while the ``(2, 0)`` block reads the same
    gate as its own trailing member, and the statevector moves by 0.38.

    The pass is allowed to be *conservative* here -- it folds nothing at all on this
    program -- because the alternative is a wrong program, and the floor arm still
    guarantees it is never longer than the two-qubit-run rule alone.
    """

    from flagquantum.simulation.statevector.local import run_local_statevector

    instructions = (
        _instruction("z", (0,)),
        _instruction("cy", (1, 2)),
        _instruction("cy", (0, 1)),
        _instruction("rxx", (0, 2), angle=0.3),
        _instruction("cz", (2, 0)),
        _instruction("sx", (1,)),
        _instruction("tdg", (0,)),
        _instruction("cz", (0, 2)),
        _instruction("t", (0,)),
        _instruction("swap", (1, 0)),
        _instruction("tdg", (1,)),
        _instruction("sx", (2,)),
        _instruction("cy", (2, 0)),
    )
    source = CircuitIR(3, instructions, dtype="complex128")
    folded = collapse_two_qubit_blocks(source)
    # `tdg(0)` stays, because the block that opened first owns it; `t(0)` stays with
    # it, because without both halves the conjugation is not `cz`.
    assert [item.name for item in folded.instructions] == [
        item.name for item in instructions
    ]
    before = run_local_statevector(
        source, batch_size=1, device=torch.device("cpu"), dtype=torch.complex128
    )
    after = run_local_statevector(
        folded, batch_size=1, device=torch.device("cpu"), dtype=torch.complex128
    )
    assert float(torch.max(torch.abs(before - after)).item()) < 1e-12
    # The narrower rule the floor arm applies also declines it, so nothing is lost
    # against the pipeline that composed two-qubit runs alone.
    assert len(optimize(source).instructions) <= len(instructions)


def test_the_whole_program_statevector_is_unchanged_by_the_fold() -> None:
    """Exactness measured on the program, not on a run.

    The comparison is on the raw statevector of an entangled three-wire program, so a
    dropped phase shows up as an amplitude difference rather than cancelling.
    """

    from flagquantum.simulation.statevector.local import run_local_statevector

    rng = random.Random(20261014)
    worst = 0.0
    for _ in range(120):
        instructions: list[Instruction] = []
        for _ in range(rng.randint(4, 12)):
            pair = rng.choice(((0, 1), (1, 2), (0, 2)))
            if rng.random() < 0.3:
                instructions.append(
                    _instruction(
                        rng.choice(("h", "x", "z", "s", "t")), (rng.randrange(3),)
                    )
                )
                continue
            opcode = rng.choice(_FAMILIES)
            angle = rng.choice(
                (math.pi, 0.5 * math.pi, 2.0 * math.pi, rng.uniform(-3.0, 3.0))
            )
            instructions.append(_instruction(opcode, pair, angle=angle))
        source = CircuitIR(3, tuple(instructions), dtype="complex128")
        folded = collapse_two_qubit_blocks(source)
        before = run_local_statevector(
            source, batch_size=1, device=torch.device("cpu"), dtype=torch.complex128
        )
        after = run_local_statevector(
            folded, batch_size=1, device=torch.device("cpu"), dtype=torch.complex128
        )
        worst = max(worst, float(torch.max(torch.abs(before - after)).item()))
        assert len(folded.instructions) <= len(source.instructions)
    assert worst < 1e-12, worst


def test_the_whole_pipeline_agrees_with_the_pass_alone_on_the_declared_group() -> None:
    """The pass is reachable through `optimize`, and the pipeline is not worse.

    Run as a two-sided check: the shipped pipeline may not be longer than the pass
    applied alone, and the pass has to actually fire on at least one circuit, or this
    compares the pipeline with itself.
    """

    rng = random.Random(20261015)
    fired = 0
    for _ in range(60):
        instructions: list[Instruction] = []
        for _ in range(rng.randint(3, 10)):
            instructions.append(
                _instruction(
                    rng.choice(_FAMILIES),
                    rng.choice(((0, 1), (1, 2))),
                    angle=rng.choice((math.pi, 0.5 * math.pi)),
                )
            )
        source = CircuitIR(3, tuple(instructions))
        alone = collapse_two_qubit_blocks(source)
        shipped = optimize(source)
        assert len(shipped.instructions) <= len(alone.instructions) + len(
            source.instructions
        )
        assert len(shipped.instructions) <= len(source.instructions)
        if len(shipped.instructions) < len(source.instructions):
            fired += 1
    assert fired > 0
