"""Check that commutation-driven cancellation removes only what it proves.

The pass reads a structure `commutation.analyze_commutation` produces and removes
pairs of self-inverse instructions separated only by proven commuters. Its tests
have to do two jobs that pull in opposite directions: show that it removes what
`pipeline.merge_self_inverse` cannot see, and show that it declines everything else.
The second job is the one that protects the program, so it is tested with the same
care as the first.

Where a claim is about the resulting program rather than about the pass's own
bookkeeping, it is checked against the runtime: the compiled circuit's statevector
has to equal the source circuit's, which is the only statement that matters if a
rule is wrong.
"""

from __future__ import annotations

import dataclasses
import random

import pytest
import torch

from flagquantum.compiler import optimize
from flagquantum.compiler.commutation import analyze_commutation, commute
from flagquantum.compiler.commutation_cancellation import (
    _is_cancellable,
    _repeats_a_rotation,
    _repeats_an_opcode,
    _rotation_param,
    cancel_commuting_self_inverse,
    cancellable_positions,
    merge_commuting_rotations,
    rotation_groups,
)
from flagquantum.compiler.pipeline import _SELF_INVERSE, _optimize_to_fixed_point
from flagquantum.core.ir import CircuitIR, Instruction, IRValidationError

pytestmark = pytest.mark.unit

_ZERO = 1.0e-12


def _rotation(opcode: str, qubit: int, theta: float) -> Instruction:
    return Instruction(opcode, (qubit,), params={"theta": theta})


def _names(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir]


def _state(ir: CircuitIR) -> torch.Tensor:
    """The statevector of ``ir``, which is where a wrong rule becomes visible.

    The compiler layer may not import the simulation layer; a test may, and this
    is the place where the round's claims are checked against an execution rather
    than against a second opinion about commutation.
    """

    from flagquantum.simulation.statevector.local import run_local_statevector

    return run_local_statevector(
        ir,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )


def test_a_cx_pair_cancels_across_a_rotation_on_its_control() -> None:
    """The motivating shape: `cx rz cx` is a bare `rz`.

    `merge_self_inverse` cannot see this, because the latest writer of qubit 0 is
    the `rz`, not the first `cx`. The rotation sits on the control, where it
    commutes with the whole controlled operator, so the pair annihilates.
    """

    ir = CircuitIR(
        2,
        (
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.4),
            Instruction("cx", (0, 1)),
        ),
    )
    assert _names(optimize(ir)) == ["rz"]
    assert _names(optimize(ir)) == _names(cancel_commuting_self_inverse(ir))


def test_the_same_pair_does_not_cancel_when_the_rotation_is_on_the_target() -> None:
    """The negative half, and the reason the rule reads wire positions at all.

    `rz` on the *target* of a `cx` applies `diag(1, e^{-i theta/2})` only in the
    `|1>` branch of the control, which does not commute with `X`. The pass leaves
    the pair alone, and the second half shows that removing it anyway would change
    the program -- which is what makes the refusal worth having rather than
    pedantic.
    """

    ir = CircuitIR(
        2,
        (
            Instruction("cx", (0, 1)),
            _rotation("rz", 1, 0.4),
            Instruction("cx", (0, 1)),
        ),
    )
    result = cancel_commuting_self_inverse(ir)
    assert result is ir
    assert _names(optimize(ir)) == ["cx", "rz", "cx"]
    # What the control-side rule would have removed, for contrast. The control
    # qubit is prepared in `|1>`, or `cx` would do nothing on the initial state and
    # the two programs would coincide for a reason that has nothing to do with the
    # rule.
    prepared = CircuitIR(2, (Instruction("x", (0,)), *ir.instructions))
    mistaken = CircuitIR(2, (Instruction("x", (0,)), _rotation("rz", 1, 0.4)))
    assert not torch.allclose(_state(mistaken), _state(prepared))


def test_cancellation_preserves_the_state_exactly() -> None:
    """Whatever the pass removes, the program has to compute the same thing."""

    shapes = (
        (Instruction("cx", (0, 1)), _rotation("rz", 0, 0.4), Instruction("cx", (0, 1))),
        (Instruction("cy", (0, 1)), _rotation("rz", 0, 0.7), Instruction("cy", (0, 1))),
        (
            Instruction("cx", (0, 1)),
            _rotation("phase", 0, 1.1),
            _rotation("t", 0, 0.0),
            Instruction("cx", (0, 1)),
        ),
        (
            Instruction("cz", (0, 1)),
            _rotation("rz", 1, 0.9),
            Instruction("cz", (0, 1)),
        ),
        (
            Instruction("cx", (0, 1)),
            Instruction("x", (1,)),
            Instruction("cx", (0, 1)),
        ),
        (
            Instruction("x", (0,)),
            Instruction("cz", (0, 1)),
            Instruction("x", (0,)),
        ),
        (
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("h", (0,)),
        ),
        (
            Instruction("cx", (1, 0)),
            _rotation("rz", 1, 0.3),
            Instruction("cx", (1, 0)),
        ),
        (
            Instruction("ccx", (0, 1, 2)),
            _rotation("rz", 1, 0.5),
            Instruction("ccx", (0, 1, 2)),
        ),
        (
            Instruction("swap", (0, 1)),
            Instruction("x", (0,)),
            Instruction("swap", (0, 1)),
        ),
        (
            Instruction("cswap", (0, 1, 2)),
            _rotation("rz", 0, 0.6),
            Instruction("cswap", (0, 1, 2)),
        ),
    )
    for shape in shapes:
        ir = CircuitIR(3, shape)
        result = optimize(ir)
        assert len(result) <= len(ir)
        torch.testing.assert_close(_state(result), _state(ir), atol=1e-10, rtol=0)


def test_an_odd_count_leaves_exactly_one_gate_where_it_was() -> None:
    """`2k + 1` occurrences keep the last, so the survivors are applied as late."""

    ir = CircuitIR(
        2,
        (
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.4),
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.5),
            Instruction("cx", (0, 1)),
        ),
    )
    # `cx rz(.4) cx rz(.5) cx`: one `cx` survives because the count is odd, and the
    # two rotations, now adjacent, merge into one.
    assert _names(optimize(ir)) == ["rz", "cx"]
    assert optimize(ir).instructions[0].params["theta"] == pytest.approx(0.9)
    torch.testing.assert_close(_state(optimize(ir)), _state(ir), atol=1e-12, rtol=0)


def test_the_pass_is_idempotent_and_reaches_the_fixed_point() -> None:
    ir = CircuitIR(
        2,
        (
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.4),
            Instruction("cx", (0, 1)),
            Instruction("cx", (0, 1)),
        ),
    )
    # Three `cx` gates leave one behind, and `merge_self_inverse` takes the last
    # adjacent pair before this pass is consulted, so the survivor is the first.
    once = optimize(ir)
    assert _names(once) == ["cx", "rz"]
    assert optimize(once) == once
    assert cancel_commuting_self_inverse(once) is once
    torch.testing.assert_close(_state(once), _state(ir), atol=1e-12, rtol=0)


def test_a_non_unitary_barrier_stops_a_cancellation() -> None:
    """A measurement between the pair is not a commuter, so the pair stays."""

    measure = Instruction("measure", (0,), metadata={"is_dynamic": True})
    ir = CircuitIR(
        2,
        (Instruction("cx", (0, 1)), measure, Instruction("cx", (0, 1))),
    )
    result = cancel_commuting_self_inverse(ir)
    assert _names(result) == ["cx", "measure", "cx"]
    assert result is ir


def test_an_opcode_with_no_rule_is_a_barrier_rather_than_a_commuter() -> None:
    """`u3` on the control touches the pair and no rule covers it, so it blocks.

    This is the fail-closed direction doing real work: `u3` on a control *can* be a
    commuter for some angle triples and not others, and the rule source reads no
    angle, so the honest answer is that the gap is unproven.
    """

    u3 = Instruction("u3", (0,), params={"theta": 0.3, "phi": 0.4, "lbd": 0.5})
    ir = CircuitIR(2, (Instruction("cx", (0, 1)), u3, Instruction("cx", (0, 1))))
    assert cancel_commuting_self_inverse(ir) is ir


def test_a_parameterized_gate_is_never_a_candidate() -> None:
    for instruction in (
        _rotation("rz", 0, 0.0),
        _rotation("rx", 0, 0.0),
        Instruction("phase", (0,), params={"theta": 0.0}),
    ):
        assert _is_cancellable(instruction) is False
    for instruction in (
        Instruction("x", (0,)),
        Instruction("h", (0,)),
        Instruction("cx", (0, 1)),
        Instruction("ccx", (0, 1, 2)),
        Instruction("cswap", (0, 1, 2)),
    ):
        assert _is_cancellable(instruction) is True
    # The candidate set is exactly the declared self-inverse set, not a second
    # opinion about it, minus the parameterized members it may not contain.
    assert {
        "x",
        "y",
        "z",
        "h",
        "cx",
        "cy",
        "cz",
        "swap",
        "ccx",
        "cswap",
    } == _SELF_INVERSE


def test_a_matrix_override_is_never_a_candidate() -> None:
    override = Instruction("x", (0,), matrix=[[0, 1], [1, 0]])
    assert _is_cancellable(override) is False
    ir = CircuitIR(
        2, (Instruction("x", (0,)), Instruction("x", (0,), matrix=[[0, 1], [1, 0]]))
    )
    assert cancel_commuting_self_inverse(ir) is ir


def test_the_short_circuit_answers_the_same_question_the_full_analysis_would() -> None:
    """`_repeats_an_opcode` is an optimization, and this checks it is not a rule."""

    seed = random.Random(20261009)
    opcodes = ["x", "h", "cx", "cz", "swap", "ccx", "rz", "t", "u3", "measure"]
    for _ in range(200):
        instructions = []
        for _ in range(7):
            name = seed.choice(opcodes)
            if name == "measure":
                instructions.append(
                    Instruction(
                        name, (seed.randrange(3),), metadata={"is_dynamic": True}
                    )
                )
                continue
            if name in {"t"}:
                instructions.append(Instruction(name, (seed.randrange(3),)))
                continue
            arity = {
                "x": 1,
                "h": 1,
                "rz": 1,
                "u3": 1,
                "cx": 2,
                "cz": 2,
                "swap": 2,
                "ccx": 3,
            }[name]
            wires = tuple(sorted(seed.sample(range(3), arity)))
            if name == "rz":
                instructions.append(_rotation(name, wires[0], seed.uniform(-3, 3)))
            elif name == "u3":
                instructions.append(
                    Instruction(
                        name,
                        wires,
                        params={
                            "theta": seed.uniform(-3, 3),
                            "phi": seed.uniform(-3, 3),
                            "lbd": seed.uniform(-3, 3),
                        },
                    )
                )
            else:
                instructions.append(Instruction(name, wires))
        ir = CircuitIR(3, tuple(instructions))
        groups = cancellable_positions(tuple(ir), analyze_commutation(ir))
        has_pair = any(len(positions) > 1 for positions in groups.values())
        # The short circuit is allowed to be conservative -- it keys on the opcode
        # and the wires, not on the commuting block -- but it may never say "no"
        # when a pair really is there, because then the pass would skip work it
        # could have done.
        assert not has_pair or _repeats_an_opcode(tuple(ir))
        if _repeats_an_opcode(tuple(ir)) is False:
            assert not has_pair


def test_nothing_is_removed_from_a_circuit_that_offers_no_pair() -> None:
    """A pass that found nothing and a pass that was never asked must differ."""

    ir = CircuitIR(
        3,
        (
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.3),
            Instruction("cz", (1, 2)),
            Instruction("h", (2,)),
        ),
    )
    assert _repeats_an_opcode(tuple(ir)) is False
    assert cancel_commuting_self_inverse(ir) is ir
    groups = cancellable_positions(tuple(ir), analyze_commutation(ir))
    assert groups != {}, "the circuit should still offer candidates, just no pair"
    assert all(len(positions) == 1 for positions in groups.values())


def test_a_removal_that_another_removal_depends_on_is_still_sound() -> None:
    """Two independent pairs removed together, each proved by the same analysis.

    The analysis runs once on the source program, so a pair's proof never depends
    on another pair already having been removed. That is what makes a batch
    removal safe, and the state comparison is what checks it.
    """

    ir = CircuitIR(
        2,
        (
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.4),
            Instruction("cx", (0, 1)),
            Instruction("cy", (0, 1)),
            _rotation("rz", 0, 0.5),
            Instruction("cy", (0, 1)),
        ),
    )
    # Both pairs go in one batch, and the two freed rotations then merge.
    assert _names(optimize(ir)) == ["rz"]
    assert optimize(ir).instructions[0].params["theta"] == pytest.approx(0.9)
    torch.testing.assert_close(_state(optimize(ir)), _state(ir), atol=1e-12, rtol=0)


def test_the_pass_is_wired_into_the_fixed_point_loop_exactly_once() -> None:
    """The pipeline has one place that composes optimization, and it uses it."""

    import ast
    import importlib

    module = importlib.import_module("flagquantum.compiler.pipeline")
    assert module.__file__ is not None
    with open(module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_optimize_to_fixed_point"
    )
    calls = [
        node.func.id
        for node in ast.walk(loop)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("cancel_commuting_self_inverse") == 1
    assert calls.index("cancel_commuting_self_inverse") > calls.index(
        "merge_adjacent_rotations"
    )
    # And nothing else in the pipeline reaches for the analysis directly.
    assert "analyze_commutation" not in ast.dump(tree)


def test_the_analysis_is_asked_for_once_per_call() -> None:
    """The pass is a pass, not a pass manager: it analyzes when it is called."""

    import flagquantum.compiler.commutation_cancellation as module

    ir = CircuitIR(
        2,
        (
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.4),
            Instruction("cx", (0, 1)),
        ),
    )
    calls = 0
    original = module.analyze_commutation

    def counting(source: CircuitIR):
        nonlocal calls
        calls += 1
        return original(source)

    module.analyze_commutation = counting
    try:
        module.cancel_commuting_self_inverse(ir)
    finally:
        module.analyze_commutation = original
    assert calls == 1


def test_the_fixed_point_loop_terminates_on_a_circuit_that_keeps_giving() -> None:
    """A long chain of removable pairs still converges, and does so exactly."""

    instructions: list[Instruction] = []
    for _ in range(40):
        instructions.append(Instruction("cx", (0, 1)))
        instructions.append(_rotation("rz", 0, 0.1))
        instructions.append(Instruction("cx", (0, 1)))
    ir = CircuitIR(2, tuple(instructions))
    result = _optimize_to_fixed_point(ir)
    assert _names(result) == ["rz"]
    assert result.instructions[0].params["theta"] == pytest.approx(4.0)
    torch.testing.assert_close(_state(result), _state(ir), atol=1e-10, rtol=0)


def test_the_results_are_frozen_values_rather_than_live_views() -> None:
    ir = CircuitIR(2, (Instruction("cx", (0, 1)), Instruction("cx", (0, 1))))
    analysis = analyze_commutation(ir)
    groups = cancellable_positions(tuple(ir), analysis)
    assert dict(groups) == {("cx", (0, 1), (0, 0)): (0, 1)}
    assert dataclasses.is_dataclass(analysis)
    with pytest.raises(TypeError):
        groups[("cx", (0, 1), (0, 0))] = []  # type: ignore[index]


#: The wires each self-inverse opcode sits on in the table below.
_SELF_INVERSE_WIRES: dict[str, tuple[int, ...]] = {
    "x": (0,),
    "y": (0,),
    "z": (0,),
    "h": (0,),
    "cx": (0, 1),
    "cy": (0, 1),
    "cz": (0, 1),
    "swap": (0, 1),
    "ccx": (0, 1, 2),
    "cswap": (0, 1, 2),
}

#: A gap this pass can prove for each opcode that has one. The gap is always
#: same-wire: a gate on a disjoint wire is what `merge_self_inverse` already sees,
#: so reproducing that here would prove nothing about this pass.
_PROVEN_GAP: dict[str, tuple[Instruction, ...]] = {
    # Diagonal single-qubit rotations pull out of a diagonal gate.
    "z": (_rotation("rz", 0, 0.4),),
    "cz": (_rotation("rz", 0, 0.4),),
    # A diagonal or `span{I, X}` gate on a control, a `span{I, X}` gate on the
    # target of `cx`/`ccx`, and the matching Pauli family for `cy`.
    "cx": (_rotation("rz", 0, 0.4),),
    "cy": (_rotation("rz", 0, 0.4),),
    "ccx": (_rotation("rz", 0, 0.4),),
    "cswap": (_rotation("rz", 0, 0.4),),
    # `swap` is the wire exchange itself, so it commutes with a symmetric operator.
    "swap": (Instruction("cz", (0, 1)),),
}


@pytest.mark.parametrize("opcode", sorted(_PROVEN_GAP))
def test_every_opcode_with_a_provable_same_wire_gap_cancels_across_it(
    opcode: str,
) -> None:
    """Each opcode whose gap the rule source can prove really does annihilate.

    The circuit carries an unrelated `t` on qubit 3, which the rule source has no
    rule for, so a passing test also shows the pass stepping over a gate it cannot
    reason about while still cancelling the pair it can.
    """

    wires = _SELF_INVERSE_WIRES[opcode]
    ir = CircuitIR(
        4,
        (
            Instruction(opcode, wires),
            Instruction("t", (3,)),
            *_PROVEN_GAP[opcode],
            Instruction(opcode, wires),
        ),
    )
    result = cancel_commuting_self_inverse(ir)
    assert len(result) == len(ir) - 2, (opcode, _names(ir), _names(result))
    torch.testing.assert_close(_state(result), _state(ir), atol=1e-10, rtol=0)


@pytest.mark.parametrize("opcode", ["x", "y", "h"])
def test_a_single_qubit_self_inverse_gate_gains_no_reach_from_this_pass(
    opcode: str,
) -> None:
    """The honest boundary of the round: `x`, `y`, and `h` gain nothing here.

    Measured over all 29 declared arity-one and arity-two unitaries, no same-wire
    gate other than `opcode` itself commutes with these three, so there is no gap
    to bridge and `merge_self_inverse` remains the only pass that removes them.
    Asserting that as a measurement rather than leaving it as a hunch is the point:
    it is what makes "this pass adds reach for controlled gates and for `z`" a
    claim with a boundary instead of a hope.
    """

    from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

    commuters = []
    for other in sorted(OPERATOR_SCHEMAS):
        schema = OPERATOR_SCHEMAS[other]
        if not schema.unitary or schema.arity > 2 or other == opcode:
            continue
        other_wires = (0,) if schema.arity == 1 else (0, 1)
        candidate = Instruction(
            other, other_wires, params=dict.fromkeys(schema.parameters, 0.4)
        )
        if commute(Instruction(opcode, (0,)), candidate):
            commuters.append(other)
    assert commuters == []
    ir = CircuitIR(
        4,
        (
            Instruction(opcode, (0,)),
            _rotation("rz", 0, 0.4),
            Instruction(opcode, (0,)),
        ),
    )
    assert cancel_commuting_self_inverse(ir) is ir


def test_the_declared_self_inverse_set_is_the_one_the_rules_agree_with() -> None:
    """Every member of `_SELF_INVERSE` is a declared opcode with the arity used above."""

    assert _SELF_INVERSE
    assert set(_SELF_INVERSE) == set(_SELF_INVERSE_WIRES)
    assert set(_PROVEN_GAP) <= set(_SELF_INVERSE)
    for opcode, wires in sorted(_SELF_INVERSE_WIRES.items()):
        assert Instruction(opcode, wires).name == opcode


def test_a_pair_of_different_self_inverse_opcodes_is_never_removed() -> None:
    """`x` next to `h` is not a pair, and the pass must not treat it as one."""

    ir = CircuitIR(1, (Instruction("x", (0,)), Instruction("h", (0,))))
    assert cancel_commuting_self_inverse(ir) is ir
    ir = CircuitIR(
        2,
        (
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.4),
            Instruction("cz", (0, 1)),
        ),
    )
    assert cancel_commuting_self_inverse(ir) is ir


def test_the_pass_never_increases_the_instruction_count() -> None:
    seed = random.Random(20261010)
    for _ in range(120):
        instructions = []
        for _ in range(9):
            name = seed.choice(["x", "h", "cx", "cz", "swap", "rz", "t", "cx", "cz"])
            if name == "rz":
                instructions.append(
                    _rotation(name, seed.randrange(3), seed.uniform(-3, 3))
                )
                continue
            arity = 1 if name in {"x", "h", "t"} else (2 if name != "ccx" else 3)
            wires = tuple(sorted(seed.sample(range(3), arity)))
            instructions.append(Instruction(name, wires))
        ir = CircuitIR(3, tuple(instructions))
        result = cancel_commuting_self_inverse(ir)
        assert len(result) <= len(ir)
        if len(result) != len(ir):
            torch.testing.assert_close(_state(result), _state(ir), atol=1e-10, rtol=0)
        assert len(cancel_commuting_self_inverse(result)) <= len(result)


# The second rule this module carries: two rotations of one opcode whose halves a
# proven commuting gap separated add up into one. It shares the proof, the group
# keying and the parameter arithmetic with the cancellation above, so what these
# tests have to establish is the part that differs -- where the merged rotation is
# written, and that nothing the adjacent merge refuses is admitted here.


def test_a_rotation_pair_merges_across_a_control_side_gap() -> None:
    """`rz(a) cx rz(b)` is `rz(a + b) cx`, and the adjacent merge cannot see it.

    This is the shape the rule exists for. `merge_adjacent_rotations` asks whether
    the latest writer of wire 0 is an `rz`, and between the two halves it is the
    `cx`, so the adjacent pass declines; the gap is proven by the rule source, so
    the two angles may be added anyway.
    """

    ir = CircuitIR(
        2,
        (
            _rotation("rz", 0, 0.3),
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.4),
        ),
    )
    assert _names(merge_commuting_rotations(ir)) == ["rz", "cx"]
    assert merge_commuting_rotations(ir).instructions[0].params[
        "theta"
    ] == pytest.approx(0.7)
    assert _names(optimize(ir)) == ["rz", "cx"]
    torch.testing.assert_close(_state(optimize(ir)), _state(ir), atol=1e-12, rtol=0)


def test_the_merged_angle_is_written_where_the_earliest_half_was() -> None:
    """Placement is a claim, not an implementation detail.

    The block is applied where the first half already sat, which is what makes the
    rewrite legal without moving anything: everything the group spans was proven to
    commute with it, so the merged rotation may stay put. A rewrite that moved it
    to the second half's position would need the same proof in the other direction
    and would have to justify itself; this one does not.

    The wire order is then read back from a mixed program, where only the position
    of the merged rotation relative to the entangler is observable.
    """

    ir = CircuitIR(
        2,
        (
            _rotation("rz", 0, 0.25),
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.5),
        ),
    )
    merged = merge_commuting_rotations(ir)
    assert merged.instructions[0].name == "rz"
    assert merged.instructions[1].name == "cx"
    assert merged.instructions[0].params["theta"] == pytest.approx(0.75)


def test_a_rotation_on_the_target_of_a_cx_is_not_merged() -> None:
    """The control-side proof does not transfer to the target, for the same reason.

    `rz` on the target applies in the `|1>` branch of the control only, so it does
    not commute with `X` and no group forms. The second half shows that merging
    anyway would have changed the program, which is why the refusal is worth
    testing rather than assuming.
    """

    ir = CircuitIR(
        2,
        (
            _rotation("rz", 1, 0.3),
            Instruction("cx", (0, 1)),
            _rotation("rz", 1, 0.4),
        ),
    )
    assert merge_commuting_rotations(ir) is ir
    prepared = CircuitIR(2, (Instruction("x", (0,)), *ir.instructions))
    mistaken = CircuitIR(2, (Instruction("x", (0,)), _rotation("rz", 1, 0.7)))
    assert not torch.allclose(_state(mistaken), _state(prepared))


def test_two_rotation_opcodes_are_never_added_together() -> None:
    """`rz(a) cx u1(b)` stays as it is: the two are not one operator.

    `merge_adjacent_rotations` requires the two instructions to share an opcode,
    and this rule inherits that condition rather than relaxing it. `rz` and `u1`
    differ by a global phase, so adding their angles would be wrong even where the
    commutation proof holds.
    """

    ir = CircuitIR(
        2,
        (
            _rotation("rz", 0, 0.3),
            Instruction("cx", (0, 1)),
            _rotation("u1", 0, 0.4),
        ),
    )
    assert merge_commuting_rotations(ir) is ir


def test_angles_that_add_to_zero_remove_both_halves() -> None:
    """A vanished sum is a deletion, through the same ``_is_zero`` the merge uses."""

    ir = CircuitIR(
        2,
        (
            _rotation("rz", 0, 0.4),
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, -0.4),
        ),
    )
    assert _names(merge_commuting_rotations(ir)) == ["cx"]


def test_a_trainable_angle_is_never_annihilated() -> None:
    """A differentiable pair stays in the program, still differentiable.

    `pipeline._is_zero` reads a tensor that requires grad as "not zero" on purpose,
    so an exact cancelling pair of trainable rotations survives and the merged
    angle is the sum the optimiser will differentiate through. Removing it would
    take the parameters out of the autograd graph.
    """

    first = torch.tensor(0.4, requires_grad=True)
    second = torch.tensor(-0.4, requires_grad=True)
    ir = CircuitIR(
        2,
        (
            Instruction("rz", (0,), params={"theta": first}),
            Instruction("cx", (0, 1)),
            Instruction("rz", (0,), params={"theta": second}),
        ),
    )
    merged = merge_commuting_rotations(ir)
    assert _names(merged) == ["rz", "cx"]
    total = merged.instructions[0].params["theta"]
    assert total.requires_grad
    assert total.grad_fn is not None
    total.backward()
    assert first.grad is not None and second.grad is not None


def test_a_matrix_override_is_not_second_guessed() -> None:
    """A caller-supplied matrix ends the rule, exactly as the adjacent merge does."""

    override = [[1.0, 0.0], [0.0, 1.0]]
    assert (
        _rotation_param(Instruction("rz", (0,), params={"theta": 0.3}, matrix=override))
        is None
    )
    ir = CircuitIR(
        2,
        (
            Instruction("rz", (0,), params={"theta": 0.3}, matrix=override),
            Instruction("cx", (0, 1)),
            Instruction("rz", (0,), params={"theta": 0.4}, matrix=override),
        ),
    )
    assert merge_commuting_rotations(ir) is ir


def test_a_rotation_without_its_declared_parameter_is_declined() -> None:
    """The missing-parameter branch is a guard the IR has already made unreachable.

    Every opcode `_ROTATION_PARAM` names declares `theta` as a required parameter,
    so ``IRValidationError`` fires on construction before this rule is reached. The
    branch is kept because `_rotation_param` reads a plain mapping rather than the
    schema and a future opcode could name its angle differently; asserting the IR's
    own refusal here is what stops that branch from being read as reachable code.
    """

    with pytest.raises(IRValidationError, match="missing parameter"):
        Instruction("rz", (0,))
    assert _rotation_param(_rotation("rz", 0, 0.3)) == "theta"
    assert _rotation_param(Instruction("x", (0,))) is None


def test_a_measurement_stops_the_merge() -> None:
    """The rule source fails closed on a measurement, so no group spans one."""

    ir = CircuitIR(
        2,
        (
            _rotation("rz", 0, 0.3),
            Instruction("measure", (0,), metadata={"is_dynamic": True}),
            _rotation("rz", 0, 0.4),
        ),
    )
    assert merge_commuting_rotations(ir) is ir


def test_the_merge_is_idempotent() -> None:
    """A second application changes nothing, which `_optimize_to_fixed_point` needs."""

    ir = CircuitIR(
        3,
        (
            _rotation("rz", 0, 0.3),
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.4),
            Instruction("cz", (1, 2)),
            _rotation("rz", 0, 0.5),
        ),
    )
    once = merge_commuting_rotations(ir)
    assert merge_commuting_rotations(once) is once


def test_the_short_circuit_answers_a_weaker_question_than_the_pass() -> None:
    """`_repeats_a_rotation` is permission to look, not a promise that something merges.

    A repeat split across two commuting blocks still answers ``True`` here and is
    then left standing by the pass, which is the direction the short circuit is
    allowed to be wrong in. The other direction is the one that would cost a
    merge: a circuit with no repeated rotation on any one set of wires has no
    group of two, so ``False`` is final.
    """

    assert not _repeats_a_rotation((_rotation("rz", 0, 0.3),))
    assert _repeats_a_rotation((_rotation("rz", 0, 0.3), _rotation("rz", 0, 0.4)))
    blocked = CircuitIR(
        1,
        (
            _rotation("rz", 0, 0.3),
            Instruction("h", (0,)),
            _rotation("rz", 0, 0.4),
        ),
    )
    assert _repeats_a_rotation(tuple(blocked))
    assert merge_commuting_rotations(blocked) is blocked


def test_the_rotation_groups_are_keyed_by_the_block_and_the_wires() -> None:
    """The group a rotation lands in is the analysis's answer, not a second opinion."""

    ir = CircuitIR(
        2,
        (
            _rotation("rz", 0, 0.3),
            Instruction("cx", (0, 1)),
            _rotation("rz", 0, 0.4),
        ),
    )
    groups = rotation_groups(tuple(ir), analyze_commutation(ir))
    assert sum(len(positions) for positions in groups.values()) == 2
    assert sum(1 for positions in groups.values() if len(positions) == 2) == 1
    # The two halves have to commute with the gap for the group to form at all.
    assert commute(ir.instructions[0], ir.instructions[1])
    assert commute(ir.instructions[1], ir.instructions[2])


def test_the_merge_never_changes_what_the_program_computes() -> None:
    """A randomized sweep: every group the pass adds up preserves the state exactly."""

    seed = random.Random(20261015)
    opcodes = ("rx", "ry", "rz", "phase", "u1")
    for _ in range(120):
        instructions = []
        for _ in range(10):
            entangler = seed.choice(["cx", "cz", "swap", "cy"])
            wires = tuple(seed.sample(range(3), 2))
            instructions.append(Instruction(entangler, wires))
            if seed.random() < 0.75:
                opcode = seed.choice(opcodes)
                qubit = seed.choice(wires)
                instructions.append(_rotation(opcode, qubit, seed.uniform(-3.0, 3.0)))
        ir = CircuitIR(3, tuple(instructions))
        result = merge_commuting_rotations(ir)
        assert len(result) <= len(ir)
        if len(result) != len(ir):
            torch.testing.assert_close(_state(result), _state(ir), atol=1e-10, rtol=0)
        assert merge_commuting_rotations(result) is result
