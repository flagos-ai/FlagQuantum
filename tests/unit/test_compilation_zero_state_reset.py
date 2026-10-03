"""Check the zero-state reset pass against the IR contract and the runtime.

`zero_state_reset.remove_zero_state_resets` removes a ``reset`` from a wire that no
surviving instruction has touched. The rule it needs is not a gate identity and not a
schema declaration: it is a fact about the register, that a ``CircuitIR`` starts in
``|0...0>``. So the first job here is to show that the fact is a property of the IR
rather than an assumption, and the second is to show that a program really does
execute the same after such a reset is removed.

Two instruments are used, and the choice between them is the honest part. On a
measure-free program the statevector is compared exactly. On a program that measures,
the comparison cannot be exact: a reset reads a random draw, so removing one shifts
the generator stream and the same seed no longer produces the same per-shot
trajectory even though the distribution is unchanged. There the test compares outcome
shares over a fixed seed, and a control proves the instrument can see the difference
it is looking for -- removing a reset that was *not* on a zero wire moves the share by
half.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest
import torch

import flagquantum.compiler.pipeline as pipeline_module
import flagquantum.compiler.zero_state_reset as zero_state_reset_module
from flagquantum.compiler import optimize
from flagquantum.compiler.zero_state_reset import remove_zero_state_resets
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.runtime.dynamic import DynamicCircuit, run_dynamic

pytestmark = pytest.mark.unit

_MODULE_PATH = Path(zero_state_reset_module.__file__)

#: A statevector comparison of two programs that differ by a no-op reset is exact.
_ATOL = 1.0e-12

#: Outcome shares are compared over a fixed seed, so this is not a flaky bound: it is
#: the largest share difference the seeded trajectories of these circuits produce, with
#: room for a generator that draws in a different order. The control below moves a
#: share by 0.5, two orders of magnitude more.
_SHARE_TOLERANCE = 0.05

_SHOTS = 8000
_SEED = 20261109


def _source() -> str:
    return _MODULE_PATH.read_text(encoding="utf-8")


def _names(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir]


def _state(ir: CircuitIR) -> torch.Tensor:
    """The register state a measure-free dynamic program ends in.

    The statevector simulator has no ``reset``, so the runtime that defines one is the
    only exact instrument here. Its per-trajectory final state is what the program
    computes, and a reset leaves every trajectory in the same state even though the
    outcome it read on the way was random.
    """

    return run_dynamic(_dynamic(ir), shots=2, seed=_SEED).final_states


def _dynamic(ir: CircuitIR) -> DynamicCircuit:
    return DynamicCircuit.from_ir(ir)


def _shares(circuit: DynamicCircuit) -> list[float]:
    """The share of shots landing on each outcome, ordered by outcome index."""

    result = run_dynamic(circuit, shots=_SHOTS, seed=_SEED)
    bits = result.classical_bits
    width = int(bits.shape[1])
    counts: dict[int, int] = {}
    for row in bits.tolist():
        written = [value for value in row if value >= 0]
        outcome = 0
        for shift, value in enumerate(written):
            outcome |= int(value) << shift
        counts[outcome] = counts.get(outcome, 0) + 1
    return [counts.get(index, 0) / _SHOTS for index in range(1 << width)]


def _worst_share_difference(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """The ``id()`` of every constant that is a module, class, or function docstring."""

    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            found.add(id(body[0].value))
    return found


def _reset(wire: int = 0, **metadata: object) -> Instruction:
    return Instruction("reset", (wire,), metadata={"is_dynamic": True, **metadata})


def test_the_register_has_no_initial_state_but_the_zero_state() -> None:
    """The rule rests on this: no field of the IR can carry another initial state."""

    import dataclasses

    fields = {field.name for field in dataclasses.fields(CircuitIR)}
    assert fields == {
        "n_wires",
        "instructions",
        "version",
        "dtype",
        "shape",
        "observables",
        "measurements",
        "metadata",
    }
    assert not hasattr(CircuitIR(2, ()), "initial_state")


def test_the_module_names_exactly_one_gate_and_it_is_reset() -> None:
    """The one gate name in the pass is the rule, so a second one would be a bug."""

    tree = ast.parse(_source())
    docstrings = _docstring_nodes(tree)
    exported = {
        id(element)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
        for element in ast.walk(node.value)
    }
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and id(node) not in exported
    }
    assert literals == {"reset"}


def test_the_pass_imports_nothing_that_could_read_a_matrix() -> None:
    """A rule of proof reads the program, not the gate tables."""

    tree = ast.parse(_source())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert sorted(imported) == ["__future__", "core.ir", "dataclasses"]


@pytest.mark.parametrize("wire", [0, 1, 2])
def test_a_leading_reset_is_removed_even_with_later_gates_on_its_wire(
    wire: int,
) -> None:
    ir = CircuitIR(
        3,
        (
            _reset(wire),
            Instruction("h", (wire,)),
            Instruction("measure", (wire,), metadata={"is_dynamic": True}),
        ),
    )

    assert _names(remove_zero_state_resets(ir)) == ["h", "measure"]


def test_a_reset_before_another_wire_is_independent_of_that_wire() -> None:
    ir = CircuitIR(2, (_reset(0), Instruction("x", (1,)), _reset(1)))

    assert _names(remove_zero_state_resets(ir)) == ["x", "reset"]


def test_a_leading_run_collapses_completely() -> None:
    """A removed reset touches nothing, so the next one is leading too."""

    ir = CircuitIR(2, (_reset(0), _reset(0), _reset(0), Instruction("h", (0,))))

    assert _names(remove_zero_state_resets(ir)) == ["h"]


@pytest.mark.parametrize(
    "blocker",
    [
        Instruction("h", (0,)),
        Instruction("x", (0,)),
        Instruction("z", (0,)),
        Instruction("i", (0,)),
        Instruction("cx", (0, 1)),
        Instruction("measure", (0,), metadata={"is_dynamic": True}),
        Instruction("rz", (0,), params={"theta": 0.0}),
    ],
)
def test_a_reset_after_anything_on_its_wire_is_kept(blocker: Instruction) -> None:
    ir = CircuitIR(2, (blocker, _reset(0)))

    assert _names(remove_zero_state_resets(ir)) == [blocker.name, "reset"]


def test_two_resets_after_a_measure_are_both_kept() -> None:
    """The first blocked reset survives, and it is what blocks the second."""

    ir = CircuitIR(
        2,
        (
            Instruction("measure", (0,), metadata={"is_dynamic": True}),
            _reset(0),
            _reset(0),
        ),
    )

    assert _names(remove_zero_state_resets(ir)) == ["measure", "reset", "reset"]


def test_a_reset_after_a_pair_the_other_passes_cancel_needs_the_fixed_point() -> None:
    """The gain is the loop's, and it is the reason the pass runs inside it."""

    ir = CircuitIR(
        2,
        (
            Instruction("x", (0,)),
            Instruction("x", (0,)),
            _reset(0),
            Instruction("h", (0,)),
        ),
    )

    assert _names(remove_zero_state_resets(ir)) == ["x", "x", "reset", "h"]
    assert _names(optimize(ir)) == ["h"]


def test_a_reset_the_pass_reads_only_when_it_is_one_wire_and_bare() -> None:
    two_wire = Instruction("reset", (0, 1), metadata={"is_dynamic": True})
    parameterized = Instruction(
        "reset", (0,), params={"theta": 0.1}, metadata={"is_dynamic": True}
    )
    with_matrix = Instruction("reset", (0,), matrix=torch.eye(2, dtype=torch.complex64))

    ir = CircuitIR(2, (two_wire, parameterized, with_matrix))

    assert _names(remove_zero_state_resets(ir)) == ["reset", "reset", "reset"]


def test_a_conditional_reset_on_a_zero_wire_is_still_removable() -> None:
    """A condition selects whether a no-op happens; it does not touch the wire."""

    ir = CircuitIR(
        2,
        (
            _reset(0, conditions=((0, 1),)),
            Instruction("measure", (1,), metadata={"is_dynamic": True}),
        ),
    )

    assert _names(remove_zero_state_resets(ir)) == ["measure"]


def test_a_conditional_reset_after_a_measure_is_kept() -> None:
    ir = CircuitIR(
        2,
        (
            Instruction("measure", (0,), metadata={"is_dynamic": True}),
            _reset(0, conditions=((0, 1),)),
        ),
    )

    assert _names(remove_zero_state_resets(ir)) == ["measure", "reset"]


def test_a_program_without_a_reset_is_returned_unchanged() -> None:
    ir = CircuitIR(2, (Instruction("h", (0,)), Instruction("cx", (0, 1))))

    assert remove_zero_state_resets(ir) is ir


def test_a_program_whose_only_reset_is_blocked_is_returned_unchanged() -> None:
    ir = CircuitIR(2, (Instruction("h", (0,)), _reset(0)))

    assert remove_zero_state_resets(ir) is ir


def test_the_pass_is_idempotent() -> None:
    ir = CircuitIR(
        2,
        (
            _reset(0),
            _reset(0),
            Instruction("h", (0,)),
            _reset(1),
            Instruction("x", (1,)),
            _reset(1),
        ),
    )

    once = remove_zero_state_resets(ir)

    assert remove_zero_state_resets(once) is once


def test_the_pass_is_wired_into_the_fixed_point_loop_exactly_once() -> None:
    tree = ast.parse(Path(pipeline_module.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "remove_zero_state_resets"
    ]

    assert len(calls) == 1


def test_a_removed_reset_leaves_the_statevector_identical() -> None:
    ir = CircuitIR(
        3,
        (
            _reset(0),
            _reset(2),
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("rz", (1,), params={"theta": 0.7}),
            Instruction("cx", (1, 2)),
        ),
    )
    optimized = remove_zero_state_resets(ir)
    assert len(optimized) == 4

    assert torch.allclose(_state(ir), _state(optimized), atol=_ATOL)


def test_removing_a_reset_that_was_not_on_a_zero_wire_would_not_be_identical() -> None:
    """The control: the statevector comparison can see a wrong removal."""

    correct = CircuitIR(2, (Instruction("x", (0,)), _reset(0)))
    wrong = CircuitIR(2, (Instruction("x", (0,)),))

    assert not torch.allclose(_state(correct), _state(wrong), atol=_ATOL)


def test_the_outcome_share_instrument_can_detect_a_wrong_removal() -> None:
    """The control for the distribution comparison below."""

    correct = DynamicCircuit(2).h(0).reset(0).measure(0)
    wrong = DynamicCircuit(2).h(0).measure(0)

    difference = _worst_share_difference(_shares(correct), _shares(wrong))

    assert difference > 0.4


def test_a_removed_reset_leaves_the_outcome_distribution_unchanged() -> None:
    source = (
        DynamicCircuit(2).reset(0).reset(0).h(0).measure(0).reset(1).x(1).measure(1)
    )
    optimized = _dynamic(optimize(source))
    assert _names(optimize(source)) == ["h", "measure", "x", "measure"]

    difference = _worst_share_difference(_shares(source), _shares(optimized))

    assert difference < _SHARE_TOLERANCE


def _random_dynamic_circuit(rng: random.Random, width: int) -> CircuitIR:
    instructions: list[Instruction] = []
    for _ in range(rng.randint(2, 24)):
        choice = rng.random()
        qubit = rng.randrange(width)
        if choice < 0.3:
            instructions.append(_reset(qubit))
        elif choice < 0.45:
            instructions.append(
                Instruction(
                    "measure",
                    (qubit,),
                    metadata={"is_dynamic": True, "classical_bit": qubit},
                )
            )
        elif choice < 0.6:
            instructions.append(
                Instruction(
                    "rz", (qubit,), params={"theta": rng.choice((0.4, -0.9, 1.3))}
                )
            )
        elif choice < 0.8:
            instructions.append(Instruction(rng.choice(("h", "x", "z")), (qubit,)))
        else:
            other = (qubit + 1) % width
            instructions.append(Instruction("cx", (qubit, other)))
    return CircuitIR(width, tuple(instructions))


def test_random_dynamic_programs_agree_with_the_runtime_outcome_by_outcome() -> None:
    """The end-to-end check: optimization moves no outcome distribution."""

    rng = random.Random(20261110)
    programs = [_random_dynamic_circuit(rng, 3) for _ in range(24)]

    removed = 0
    executed = 0
    worst = 0.0
    for ir in programs:
        optimized = optimize(ir)
        removed += len(ir) - len(optimized)
        difference = _worst_share_difference(
            _shares(_dynamic(ir)), _shares(_dynamic(optimized))
        )
        executed += 1
        worst = max(worst, difference)

    assert executed == 24
    assert removed > 20
    assert worst < _SHARE_TOLERANCE


def test_optimization_never_leaves_a_removable_reset_behind() -> None:
    """Every reset in an optimized program is one the rule refuses, and refuses for
    a reason: a surviving instruction on its wire."""

    rng = random.Random(20261111)
    programs = [_random_dynamic_circuit(rng, 3) for _ in range(24)]

    resets = 0
    for ir in programs:
        optimized = optimize(ir)
        touched: set[int] = set()
        for instruction in optimized:
            if instruction.name == "reset":
                resets += 1
                assert not touched.isdisjoint(instruction.wires)
            touched.update(instruction.wires)

    assert resets > 20


def test_the_pass_only_deletes_and_never_invents_an_instruction() -> None:
    """The rule removes; it never synthesizes a program of its own.

    The property is stated of this pass rather than of ``optimize``, because the
    pipeline as a whole does not hold it and says so: ``collapse_one_qubit_runs``
    re-spells a same-wire run as one gate, so the pipeline can emit an instruction
    the caller never wrote. A synthesized instruction here would mean this pass had
    a rule other than "remove a reset on a clean wire", so it is still checked.
    """

    rng = random.Random(20261112)
    programs = [_random_dynamic_circuit(rng, 3) for _ in range(24)]

    for ir in programs:
        optimized = remove_zero_state_resets(ir)
        remaining = list(
            (instruction.name, instruction.wires) for instruction in optimized
        )
        cursor = 0
        for instruction in ir:
            key = (instruction.name, instruction.wires)
            if cursor < len(remaining) and remaining[cursor] == key:
                cursor += 1
        assert cursor == len(remaining)
