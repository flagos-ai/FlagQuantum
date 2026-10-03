import random
import time
from dataclasses import replace

import pytest
import torch

from flagquantum.compiler import optimize
from flagquantum.compiler.one_qubit_optimization import collapse_one_qubit_runs
from flagquantum.compiler.pipeline import (
    _ROTATION_PARAM,
    _SELF_INVERSE,
    _add_values,
    _is_zero,
    _replace_param,
)
from flagquantum.core.ir import CircuitIR, Instruction, ensure_circuit_ir

pytestmark = pytest.mark.unit


def test_self_inverse_gates_cancel_across_disjoint_wire_gate() -> None:
    middle = Instruction("h", (1,))
    ir = CircuitIR(
        2,
        (Instruction("x", (0,)), middle, Instruction("x", (0,))),
    )

    assert optimize(ir).instructions == (middle,)


def test_rotation_gates_merge_across_disjoint_wire_gate() -> None:
    middle = Instruction("h", (1,))
    ir = CircuitIR(
        2,
        (
            Instruction("rx", (0,), params={"theta": 0.1}),
            middle,
            Instruction("rx", (0,), params={"theta": 0.2}),
        ),
    )

    compiled = optimize(ir)

    assert tuple(item.name for item in compiled) == ("rx", "h")
    assert compiled.instructions[0].params["theta"] == pytest.approx(0.3)


def test_touching_gate_blocks_self_inverse_and_rotation_rewrites() -> None:
    self_inverse = CircuitIR(
        2,
        (
            Instruction("x", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("x", (0,)),
        ),
    )
    rotations = CircuitIR(
        2,
        (
            Instruction("rx", (0,), params={"theta": 0.1}),
            Instruction("cx", (0, 1)),
            Instruction("rx", (0,), params={"theta": 0.2}),
        ),
    )

    assert optimize(self_inverse).instructions == self_inverse.instructions
    assert optimize(rotations).instructions == rotations.instructions


def test_wire_local_trainable_rotation_merge_preserves_both_gradients() -> None:
    theta = torch.tensor(0.2, requires_grad=True)
    phi = torch.tensor(-0.1, requires_grad=True)
    ir = CircuitIR(
        2,
        (
            Instruction("ry", (0,), params={"theta": theta}),
            Instruction("h", (1,)),
            Instruction("ry", (0,), params={"theta": phi}),
        ),
    )

    merged = optimize(ir).instructions[0].params["theta"]
    merged.backward()

    assert theta.grad == pytest.approx(1.0)
    assert phi.grad == pytest.approx(1.0)


def _reference_last_touching(
    instructions: list[Instruction],
    wires: tuple[int, ...],
) -> int | None:
    """The reverse list scan the wire index replaced."""

    target = set(wires)
    for index in range(len(instructions) - 1, -1, -1):
        if not target.isdisjoint(instructions[index].wires):
            return index
    return None


def _reference_remove_identity_gates(ir: CircuitIR) -> CircuitIR:
    out = []
    for instruction in ir:
        if instruction.name in {"i", "id"}:
            continue
        param_name = _ROTATION_PARAM.get(instruction.name)
        if param_name is not None and _is_zero(instruction.params.get(param_name)):
            continue
        out.append(instruction)
    return replace(ir, instructions=tuple(out))


def _reference_merge_self_inverse(ir: CircuitIR) -> CircuitIR:
    out: list[Instruction] = []
    for instruction in ir:
        previous_index = _reference_last_touching(out, instruction.wires)
        previous = None if previous_index is None else out[previous_index]
        if (
            instruction.name in _SELF_INVERSE
            and previous is not None
            and previous.name == instruction.name
            and previous.wires == instruction.wires
            and not instruction.params
            and not previous.params
        ):
            assert previous_index is not None
            out.pop(previous_index)
        else:
            out.append(instruction)
    return replace(ir, instructions=tuple(out))


def _reference_merge_adjacent_rotations(ir: CircuitIR) -> CircuitIR:
    out: list[Instruction] = []
    for instruction in ir:
        param_name = _ROTATION_PARAM.get(instruction.name)
        previous_index = _reference_last_touching(out, instruction.wires)
        previous = None if previous_index is None else out[previous_index]
        if (
            param_name is not None
            and previous is not None
            and previous.name == instruction.name
            and previous.wires == instruction.wires
            and previous.matrix is None
            and instruction.matrix is None
            and param_name in previous.params
            and param_name in instruction.params
        ):
            assert previous_index is not None
            merged = _add_values(
                previous.params[param_name], instruction.params[param_name]
            )
            if _is_zero(merged):
                out.pop(previous_index)
            else:
                out[previous_index] = _replace_param(previous, param_name, merged)
        else:
            out.append(instruction)
    return replace(ir, instructions=tuple(out))


def _reference_optimize(circuit_or_ir: object) -> CircuitIR:
    """The whole optimization pipeline with the wire-local traversal re-derived.

    Only the traversal of the three wire-local passes is re-derived here. The
    opcode tables and the parameter arithmetic are imported from the
    implementation so that this oracle differs from the code under test in
    exactly one respect: the reverse list scan that the wire index replaced. If
    the two ever disagree, the index is wrong.

    The three passes outside that traversal are called from both sides in the same
    position rather than re-derived, because each is a different pass with its own
    rule and not the subject of this oracle. ``remove_zero_state_resets`` reads the
    register's initial state instead of an opcode table;
    ``cancel_commuting_self_inverse`` reads the commutation rule source, which reads
    the runtime's gate matrices; ``collapse_one_qubit_runs`` has a traversal of its
    own. All three are in the loop because the loop has to be the one the
    implementation runs for the round-by-round comparison to mean anything.
    """

    from flagquantum.compiler.commutation_cancellation import (
        cancel_commuting_self_inverse,
    )
    from flagquantum.compiler.zero_state_reset import remove_zero_state_resets

    ir = ensure_circuit_ir(circuit_or_ir)
    for _ in range(len(ir) + 1):
        previous_count = len(ir)
        ir = remove_zero_state_resets(ir)
        ir = _reference_remove_identity_gates(ir)
        ir = _reference_merge_self_inverse(ir)
        ir = _reference_merge_adjacent_rotations(ir)
        ir = cancel_commuting_self_inverse(ir)
        ir = _reference_remove_identity_gates(ir)
        ir = collapse_one_qubit_runs(ir)
        if len(ir) == previous_count:
            return ir
    raise AssertionError("the reference optimizer did not reach a fixed point")


_SINGLE_WIRE = ("h", "x", "y", "z", "i", "id", "rx", "ry", "rz", "phase", "u1")
_TWO_WIRE = ("cx", "cz", "swap")
_THREE_WIRE = ("ccx", "cswap")
# A reset is not a gate: it carries no parameter, no matrix, and the dynamic flag the
# IR requires of an opcode the operator schema does not declare. It is in the alphabet
# so that the differential test below actually drives the pass that removes one.
_DYNAMIC = ("reset",)
# Zero sums are reachable: 0.25 + -0.25 and 1e-13 + 0.0 both collapse, which is
# what exercises the branch that removes a merged rotation instead of rewriting it.
_ANGLES = (0.0, 0.25, -0.25, 0.5, -0.5, 1.0, 1e-13)


def _random_circuit(rng: random.Random) -> CircuitIR:
    wire_count = rng.randint(2, 7)
    instructions = []
    for _ in range(rng.randint(1, 40)):
        name = rng.choice(_SINGLE_WIRE + _TWO_WIRE + _THREE_WIRE + _DYNAMIC)
        width = 3 if name in _THREE_WIRE else (2 if name in _TWO_WIRE else 1)
        if width > wire_count:
            continue
        params = {}
        metadata = {}
        if name in _ROTATION_PARAM:
            params = {_ROTATION_PARAM[name]: rng.choice(_ANGLES)}
        if name in _DYNAMIC:
            metadata = {"is_dynamic": True}
        instructions.append(
            Instruction(
                name,
                tuple(rng.sample(range(wire_count), width)),
                params=params,
                metadata=metadata,
            )
        )
    return CircuitIR(wire_count, tuple(instructions))


def test_wire_index_agrees_with_the_reverse_scan_on_random_circuits() -> None:
    rewritten = 0
    for seed in range(80):
        ir = _random_circuit(random.Random(seed))
        reference = _reference_optimize(ir)

        assert optimize(ir).instructions == reference.instructions, f"seed {seed}"

        rewritten += len(ir) - len(reference)

    # A differential test whose circuits never change proves nothing. 80 seeds
    # remove this many instructions; the seed set is fixed, so this is stable.
    assert rewritten > 150


def test_wire_index_reaches_back_across_many_untouched_wires() -> None:
    # The scan this replaced walked the whole untouched prefix, so the answer had
    # to be correct when the matching gate is 2000 instructions old.
    padding = tuple(Instruction("h", (wire,)) for wire in range(1, 2001))
    ir = CircuitIR(2001, (Instruction("x", (0,)), *padding, Instruction("x", (0,))))

    assert optimize(ir).instructions == padding


def test_wire_index_keeps_the_latest_writer_not_the_first() -> None:
    # Two gates on wire 0 separated by a gate on wire 1: the second x pairs with
    # the first, and the third x then has no partner left to cancel against.
    ir = CircuitIR(
        2,
        (
            Instruction("x", (0,)),
            Instruction("h", (1,)),
            Instruction("x", (0,)),
            Instruction("x", (0,)),
        ),
    )

    assert optimize(ir).instructions == (Instruction("h", (1,)), Instruction("x", (0,)))


def test_wide_disjoint_circuit_optimizes_in_linear_not_quadratic_time() -> None:
    # Every gate sits on its own wire, so the reverse scan never found a
    # neighbour and walked the whole output prefix every time: O(gates**2). The
    # wire index makes each pass O(gates). Eight times the gates cost 66x the
    # time before the change and 8.4x after it, measured on the development
    # machine with the protocol below. The bound is a ratio against the same
    # machine's own smaller run, so a slower or loaded machine does not fail it.
    small = CircuitIR(4000, tuple(Instruction("h", (wire,)) for wire in range(4000)))
    large = CircuitIR(32000, tuple(Instruction("h", (wire,)) for wire in range(32000)))

    def best_of_three(ir: CircuitIR) -> float:
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            optimize(ir)
            durations.append(time.perf_counter() - start)
        return min(durations)

    small_seconds = best_of_three(small)
    large_seconds = best_of_three(large)

    # 24x is three times the linear expectation and well under the 66x a reverse
    # scan produces, so the assertion has margin on both sides.
    assert large_seconds < 24 * small_seconds + 0.05
