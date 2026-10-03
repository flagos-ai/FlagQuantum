"""Check the declared-inverse pass against the declaration and the runtime.

`inverse_cancellation.merge_inverse_pairs` removes two different opcodes that the
schema declares to be inverses of each other. The rule it needs already exists as
`OperatorSchema.adjoint`, so the first job here is to show that the pass reads that
declaration rather than restating it, and the second is to show that what the
declaration means is what the pass assumes: that the two operators really do
multiply to the identity.

Where a claim is about the resulting program rather than about the pass's own
bookkeeping, it is checked against the runtime: the optimized circuit's statevector
has to equal the source circuit's. The compiler layer may not import the simulation
layer, and a test may, which is why this file exists.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest
import torch

import flagquantum.compiler.inverse_cancellation as inverse_cancellation_module
import flagquantum.compiler.pipeline as pipeline_module
from flagquantum.compiler import optimize
from flagquantum.compiler.commutation_cancellation import cancel_commuting_self_inverse
from flagquantum.compiler.inverse_cancellation import (
    _declared_partner,
    inverse_pairs,
    merge_inverse_pairs,
)
from flagquantum.compiler.pipeline import _SELF_INVERSE
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, inverse_operator
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

_MODULE_PATH = Path(inverse_cancellation_module.__file__)

#: The declaration is exact, so a comparison of two products uses one absolute
#: tolerance and no relative slack.
_ATOL = 1.0e-12

#: Every opcode and partner the pass can act on, read from the pass itself.
_PAIRS = inverse_pairs()

#: The pairs the schema declares today. Pinned so that a change to the declaration
#: shows up as a changed reach figure rather than as a quietly different pass.
_DECLARED = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t", "sx": "sxdg", "sxdg": "sx"}

#: Opcodes whose `adjoint` is a rule over an angle rather than a partner opcode. The
#: pass declines all of them, because their product is the identity only at one
#: specific angle and `merge_adjacent_rotations` is the pass that reads an angle.
_PARAMETERIZED = (
    "rx",
    "ry",
    "rz",
    "phase",
    "u1",
    "u2",
    "u3",
    "crx",
    "cry",
    "crz",
    "rxx",
    "ryy",
    "rzz",
    "cphase",
)


def _source() -> str:
    return _MODULE_PATH.read_text(encoding="utf-8")


def _names(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir]


def _state(ir: CircuitIR) -> torch.Tensor:
    from flagquantum.simulation.statevector.local import run_local_statevector

    return run_local_statevector(
        ir,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )


def _matrix(instruction: Instruction) -> torch.Tensor:
    local = gate_matrix(
        instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
    )
    width = len(instruction.wires)
    return local.reshape(-1, 2**width, 2**width)[0]


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """The `Constant` nodes that are docstrings rather than code string literals."""

    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            if not node.body:
                continue
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


def _imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return sorted(names)


@pytest.mark.parametrize(
    ("opcode", "partner"), sorted(_DECLARED.items()), ids=sorted(_DECLARED)
)
def test_every_declared_pair_cancels_however_it_is_ordered(
    opcode: str, partner: str
) -> None:
    """The motivating shape, in both orders, on the declaration's own pairs.

    The second half checks the program rather than the pass: a pair the schema calls
    inverse has to leave the statevector alone, or the declaration is wrong rather
    than the pass.
    """

    for order in ((opcode, partner), (partner, opcode)):
        ir = CircuitIR(1, tuple(Instruction(name, (0,)) for name in order))
        assert _names(merge_inverse_pairs(ir)) == []
        assert _names(optimize(ir)) == []
        assert torch.allclose(
            _state(optimize(ir)), _state(CircuitIR(1, ())), atol=_ATOL, rtol=0
        )


def test_the_candidate_set_is_exactly_the_declared_partner_pairs() -> None:
    """Re-derive the reach from the schema along a path the pass does not share."""

    derived = {}
    for schema in OPERATOR_SCHEMAS.values():
        if not schema.unitary or schema.parameters:
            continue
        inverse = inverse_operator(schema, {})
        if inverse is None:
            continue
        candidate, params = inverse
        if candidate != schema.opcode:
            assert params == {}, schema.opcode
            derived[schema.opcode] = candidate
    assert dict(_PAIRS) == derived
    assert dict(_PAIRS) == _DECLARED
    assert len(_PAIRS) == 6


def test_the_candidate_set_is_disjoint_from_the_self_inverse_pass() -> None:
    """Both are adjacency passes, and neither may restate the other's rule.

    `merge_self_inverse` owns the opcodes whose declared inverse is themselves. The
    two candidate sets have to be disjoint, or the two passes would be making the
    same decision twice from two places.
    """

    assert set(_PAIRS) & _SELF_INVERSE == set()
    for opcode in _SELF_INVERSE:
        assert _declared_partner(opcode) is None
    for opcode, partner in _PAIRS.items():
        assert _declared_partner(opcode) == partner
        assert _declared_partner(partner) == opcode


def test_each_pair_really_does_multiply_to_the_identity() -> None:
    """The declaration's meaning, read off the matrices execution uses.

    This is the claim the pass is built on. It is checked here and not in the pass,
    because `flagquantum.compiler` may not import `flagquantum.simulation`. The
    control at the end is what makes the check non-vacuous: an arbitrary choice of
    two opcodes of the same arity is not the identity, so agreement here is evidence
    about this pairing rather than about the tolerance.
    """

    identity = torch.eye(2, dtype=torch.complex128)
    for opcode, partner in _PAIRS.items():
        first = _matrix(Instruction(opcode, (0,)))
        second = _matrix(Instruction(partner, (0,)))
        assert torch.allclose(second @ first, identity, atol=_ATOL, rtol=0), opcode
        assert torch.allclose(first @ second, identity, atol=_ATOL, rtol=0), opcode

    mismatched = _matrix(Instruction("t", (0,))) @ _matrix(Instruction("s", (0,)))
    assert not torch.allclose(mismatched, identity, atol=_ATOL, rtol=0)


def test_the_module_holds_no_opcode_table_of_its_own() -> None:
    """The rule has to be read from the declaration, never restated here.

    A copied table is a second source of truth for one fact, and the two copies are
    free to drift. The first half looks for the opcodes as code string literals --
    docstrings are excluded, so the module is still allowed to name them in prose --
    and the second half looks for a module-level collection literal that could hold
    them.
    """

    tree = ast.parse(_source())
    docstrings = _docstring_nodes(tree)
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }
    assert literals & (set(_DECLARED) | set(_DECLARED.values())) == set()

    tables: list[str] = []
    for statement in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            targets, value = list(statement.targets), statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets, value = [statement.target], statement.value
        if value is None or not isinstance(
            value, (ast.Dict, ast.Set, ast.List, ast.Tuple)
        ):
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name)]
        if names != ["__all__"]:
            tables.extend(names)
    assert tables == []


def test_the_module_takes_its_rule_from_the_operator_schema() -> None:
    """The dependency the previous test depends on, stated directly."""

    imports = _imports(ast.parse(_source()))
    assert "core.operator_schema.OPERATOR_SCHEMAS" in imports
    assert "core.operator_schema.get_operator_schema" in imports
    assert "core.operator_schema.inverse_operator" in imports
    # The wire index that `merge_self_inverse` also uses, rather than a second one.
    assert "pipeline._WireLocalProgram" in imports


@pytest.mark.parametrize(
    "opcode",
    [
        "measure",
        "reset",
        "barrier",
        "bit_flip",
        "phase_flip",
        "depolarizing",
        "amplitude_damping",
        "not_a_gate",
    ],
)
def test_a_gate_the_schema_does_not_declare_has_no_partner(opcode: str) -> None:
    """Fail closed: an undeclared or non-unitary opcode is outside the rule."""

    assert _declared_partner(opcode) is None


@pytest.mark.parametrize("opcode", _PARAMETERIZED)
def test_a_parameterized_gate_is_declined_even_with_a_declared_inverse(
    opcode: str,
) -> None:
    """The angle rules are `merge_adjacent_rotations`' shape, not this pass's."""

    assert OPERATOR_SCHEMAS[opcode].adjoint in {
        "negate_parameters",
        "adjoint_u2_angles",
        "adjoint_u3_angles",
    }
    assert _declared_partner(opcode) is None
    assert opcode not in _PAIRS


def test_a_zero_angle_rotation_still_cancels_through_the_adjacent_pass() -> None:
    """Declining an angle rule is a division of labour, not a hole.

    Two rotations of one opcode whose angles sum to zero are removed by
    `merge_adjacent_rotations`, which is allowed to read the angles. The check is
    here so that the decline above cannot be read as a missing optimization.
    """

    ir = CircuitIR(
        1,
        (
            Instruction("rx", (0,), params={"theta": 0.3}),
            Instruction("rx", (0,), params={"theta": -0.3}),
        ),
    )
    assert _declared_partner("rx") is None
    assert _names(optimize(ir)) == []


def test_a_pair_cancels_across_an_instruction_on_another_wire() -> None:
    """Adjacency is measured on the wire the pair acts on, not on program order.

    A single-qubit gate on a different wire commutes with both members, so finding
    it in the middle is not an obstacle.
    """

    ir = CircuitIR(
        2,
        (
            Instruction("s", (0,)),
            Instruction("x", (1,)),
            Instruction("sdg", (0,)),
        ),
    )
    assert _names(merge_inverse_pairs(ir)) == ["x"]
    assert _names(optimize(ir)) == ["x"]
    assert torch.allclose(_state(optimize(ir)), _state(ir), atol=_ATOL, rtol=0)


def test_a_pair_does_not_cancel_across_a_gate_on_its_own_wire() -> None:
    """The measured decline, and the whole of the gap this pass leaves open.

    `s(1) cx(0, 1) sdg(1)` is not an identity. Proving that it is one would mean
    proving that the `cx` commutes with the pair and then removing the pair across
    it, which is a strictly harder question than this pass answers -- it reads only
    the inverse declaration and never a commutation. So the pair is declined, and
    the refusal is asserted together with the reason it is safe: the two members
    are not redundant here, and removing them anyway would change the program. That
    is what stops the decline being read as conservatism about a shape that was
    free to remove.
    """

    ir = CircuitIR(
        2, (Instruction("s", (1,)), Instruction("cx", (0, 1)), Instruction("sdg", (1,)))
    )
    assert merge_inverse_pairs(ir) is ir
    assert _names(optimize(ir)) == ["s", "cx", "sdg"]

    prepared = CircuitIR(
        2,
        (
            Instruction("x", (0,)),
            Instruction("x", (1,)),
            *ir.instructions,
        ),
    )
    mistaken = CircuitIR(
        2,
        (Instruction("x", (0,)), Instruction("x", (1,)), Instruction("cx", (0, 1))),
    )
    assert not torch.allclose(_state(prepared), _state(mistaken), atol=_ATOL, rtol=0)


def test_a_declined_cancellation_is_real_and_the_decline_is_the_passs_scope() -> None:
    """What the same shape costs, measured, so the decline cannot be read as safety.

    `s(0) rz(0) sdg(0)` is a bare `rz`, because conjugating `Z` by `S` leaves it
    alone. The two members therefore really are removable here, and this pass still
    declines them: it proves no commutation and reads no angle, so it has nothing
    to stand on. Asserting the equality to a bare rotation is what turns "declined"
    into a statement about this pass's scope rather than about the shape being
    unprovable, and the pass's own benchmark counts how many such rows there are.

    The contrast case is one wire wider. `rz` is diagonal and `cx` has a diagonal
    action on its control, so `cx(0, 1) rz(0) cx(0, 1)` really is a bare `rz` too.
    This pass declines it for exactly the same reason -- there is no rule in it that
    could have reached the rotation, and the pair is not even the pair being asked
    about. The `cx` pair is self-inverse and the rotation sits on a wire that gate
    acts on diagonally, though, so `cancel_commuting_self_inverse` reaches it once
    the two passes run in one pipeline. The two claims are asserted separately: the
    refusal belongs to this pass and the removal belongs to the commuting pass, so
    neither statement covers for the other.
    """

    declared_pair = CircuitIR(
        1,
        (
            Instruction("s", (0,)),
            Instruction("rz", (0,), params={"theta": 0.4}),
            Instruction("sdg", (0,)),
        ),
    )
    assert merge_inverse_pairs(declared_pair) is declared_pair
    # The pipeline does still reach the bare rotation, but through its one-qubit
    # fold and not through this pass: the fold is the pass that reads an angle, and
    # this one holds no rule that could have reached across the rotation.
    assert _names(optimize(declared_pair)) == ["rz"]
    bare_rotation = CircuitIR(1, (Instruction("rz", (0,), params={"theta": 0.4}),))
    assert torch.allclose(
        _state(declared_pair), _state(bare_rotation), atol=_ATOL, rtol=0
    )

    across_a_control = CircuitIR(
        2,
        (
            Instruction("cx", (0, 1)),
            Instruction("rz", (0,), params={"theta": 0.4}),
            Instruction("cx", (0, 1)),
        ),
    )
    bare_rotation_on_two_wires = CircuitIR(
        2, (Instruction("rz", (0,), params={"theta": 0.4}),)
    )
    assert merge_inverse_pairs(across_a_control) is across_a_control
    assert _names(cancel_commuting_self_inverse(across_a_control)) == ["rz"]
    assert _names(optimize(across_a_control)) == ["rz"]
    assert torch.allclose(
        _state(across_a_control), _state(bare_rotation_on_two_wires), atol=_ATOL, rtol=0
    )


def test_an_odd_run_keeps_exactly_one_member_where_it_was() -> None:
    """A run of `2k + 1` loses `2k`, so what survives is applied as late as before."""

    ir = CircuitIR(
        1,
        (Instruction("s", (0,)), Instruction("s", (0,)), Instruction("sdg", (0,))),
    )
    assert _names(merge_inverse_pairs(ir)) == ["s"]
    assert _names(optimize(ir)) == ["s"]
    assert torch.allclose(_state(optimize(ir)), _state(ir), atol=_ATOL, rtol=0)


def test_a_different_opcode_between_a_pair_blocks_it() -> None:
    """`s t sdg` is carried by `t`, so the pair across it is not adjacent.

    The shape is a real refusal and not a lost opportunity: `s t sdg` equals `t`, so
    the two members were removable in fact, and this pass still declines them because
    it proves nothing about the gate between them. The pipeline's one-qubit fold
    reaches the same answer by reading the angle instead, and the two assertions are
    kept apart so that the refusal stays this pass's and the reach stays the fold's.
    """

    ir = CircuitIR(
        1,
        (Instruction("s", (0,)), Instruction("t", (0,)), Instruction("sdg", (0,))),
    )
    assert merge_inverse_pairs(ir) is ir
    assert _names(merge_inverse_pairs(ir)) == ["s", "t", "sdg"]
    # The pipeline reaches one instruction, and that is the one-qubit fold re-spelling
    # the whole run as the rotation it equals -- `s t sdg` really is `t`. Which pass
    # reaches it matters: this pass must leave all three standing, so the assertion
    # above is the one that would move if the pair were cancelled across `t`.
    optimized = optimize(ir)
    assert _names(optimized) == ["phase"]
    assert len(optimized) < len(ir)
    assert torch.allclose(_state(optimized), _state(ir), atol=_ATOL, rtol=0)


def test_two_interleaved_pairs_on_different_wires_both_cancel() -> None:
    """A pair is per wire, so two pairs can straddle each other in program order."""

    ir = CircuitIR(
        2,
        (
            Instruction("s", (0,)),
            Instruction("s", (1,)),
            Instruction("sdg", (0,)),
            Instruction("sdg", (1,)),
        ),
    )
    assert _names(merge_inverse_pairs(ir)) == []
    assert _names(optimize(ir)) == []


def test_the_alias_of_a_member_is_the_member() -> None:
    """`sd` and `td` are declared aliases, so they name the same pair member."""

    ir = CircuitIR(1, (Instruction("s", (0,)), Instruction("sd", (0,))))
    assert _names(ir) == ["s", "sdg"]
    assert _names(merge_inverse_pairs(ir)) == []


def test_a_caller_supplied_matrix_is_not_second_guessed() -> None:
    """A `matrix` override is the caller telling the compiler what the gate is.

    The pass has no way to check that an override still means `s`, so it declines
    rather than trusting the opcode name. The control shows that the same two opcodes
    without the override are removed, so the refusal is about the override.
    """

    override = _matrix(Instruction("s", (0,)))
    overridden = CircuitIR(
        1,
        (
            Instruction("s", (0,), matrix=override),
            Instruction("sdg", (0,), matrix=override),
        ),
    )
    assert merge_inverse_pairs(overridden) is overridden

    plain = CircuitIR(1, (Instruction("s", (0,)), Instruction("sdg", (0,))))
    assert _names(merge_inverse_pairs(plain)) == []


def test_a_lone_member_gains_no_reach_from_this_pass() -> None:
    """One member of a pair is not a pair."""

    for opcode in _PAIRS:
        ir = CircuitIR(1, (Instruction(opcode, (0,)),))
        assert merge_inverse_pairs(ir) is ir
        assert _names(optimize(ir)) == [opcode]


def test_the_pass_returns_the_source_object_when_it_changes_nothing() -> None:
    """Cheap and observable: a caller can tell the two outcomes apart."""

    untouched = CircuitIR(1, (Instruction("s", (0,)),))
    assert merge_inverse_pairs(untouched) is untouched
    changed = CircuitIR(1, (Instruction("s", (0,)), Instruction("sdg", (0,))))
    assert merge_inverse_pairs(changed) is not changed


def test_the_pass_is_idempotent() -> None:
    """A second application finds nothing: the survivors are not adjacent partners."""

    ir = CircuitIR(
        2,
        (
            Instruction("s", (0,)),
            Instruction("s", (0,)),
            Instruction("sdg", (0,)),
            Instruction("t", (1,)),
            Instruction("tdg", (1,)),
            Instruction("sx", (1,)),
        ),
    )
    once = merge_inverse_pairs(ir)
    assert _names(once) == ["s", "sx"]
    assert merge_inverse_pairs(once) is once
    assert optimize(once) == optimize(ir)


def test_the_pass_is_wired_into_the_fixed_point_loop_exactly_once() -> None:
    """One call site, inside the loop, so the loop can use what it frees."""

    tree = ast.parse(Path(pipeline_module.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "merge_inverse_pairs"
    ]
    assert len(calls) == 1

    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_optimize_to_fixed_point"
    )
    assert any(call in ast.walk(loop) for call in calls)


def test_random_runs_of_the_declared_pairs_preserve_the_state() -> None:
    """The property, over shapes no hand-written case enumerates.

    The seed is fixed so a failure can be reproduced, and the shrinking count is
    asserted so that the sweep cannot pass by never exercising the pass at all.
    """

    seed = random.Random(20261011)
    opcodes = sorted(_PAIRS)
    shrank = 0
    for _ in range(120):
        ir = CircuitIR(
            2,
            tuple(
                Instruction(seed.choice(opcodes), (seed.randrange(2),))
                for _ in range(seed.randrange(1, 10))
            ),
        )
        optimized = optimize(ir)
        assert len(optimized) <= len(ir)
        assert torch.allclose(_state(optimized), _state(ir), atol=_ATOL, rtol=0)
        if len(optimized) < len(ir):
            shrank += 1
    assert shrank > 20
