"""Unit contract for the same-wire single-qubit run fold W9-05 adds.

The fold composes a maximal run of single-qubit gates on one wire into one exact
sequence over ``u3``, ``phase``, and ``rz``. It may not drop a global phase,
because ``CircuitIR`` has no field to record one in, so the central test here is
the one that pins the fold's matrix against the runtime gate table rather than
against this module's own arithmetic.

The gate table in ``flagquantum.simulation`` is the authority. This module is in
the Compiler layer, which may not import it, so the closed forms it uses are
pinned here instead: a test may import the runtime, and the convention drift this
guards against is exactly the kind that would otherwise hide inside the fold.
"""

import cmath
import random
from dataclasses import replace

import pytest
import torch

import flagquantum.compiler as compiler
from flagquantum.compiler.one_qubit_optimization import (
    _IDENTITY,
    _declines_vocabulary,
    _emit,
    _fold_run,
    _instruction_matrix,
    _matmul,
    _run_product,
    _single_basis_pair,
    collapse_one_qubit_runs,
)
from flagquantum.compiler.pipeline import optimize
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, get_operator_schema
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

#: Every declared arity-1 unitary opcode. The fold's claim is about the whole
#: single-qubit group, so the convention table below has to cover all of it.
SINGLE_QUBIT_OPCODES: tuple[str, ...] = tuple(
    sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 1
    )
)

#: The three opcodes whose runtime matrix carries a global phase that the Euler
#: triple `one_qubit_synthesis` tabulates does not, and the phase each one
#: carries. `rz` is a half-angle rotation written on one basis state where `u1`
#: is a full-angle one; `sx` and `sxdg` are the square roots of X, whose matrix
#: is a quarter turn from the identity in phase.
_CONVENTION_DEGREES = {"rz": -0.5, "sx": 0.25, "sxdg": -0.25}


def _runtime_matrix(instruction: Instruction, n_wires: int = 1) -> list[list[complex]]:
    """One instruction's matrix, from the runtime gate table, in complex128."""

    values = gate_matrix(
        instruction,
        bsz=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    ).reshape(2**n_wires, 2**n_wires)
    return [
        [complex(values[row][column]) for column in range(2**n_wires)]
        for row in range(2**n_wires)
    ]


def _gap(left: list[list[complex]], right: list[list[complex]]) -> float:
    size = len(left)
    return max(
        abs(left[row][column] - right[row][column])
        for row in range(size)
        for column in range(size)
    )


def _sample(opcode: str, rng: random.Random, wire: int = 0) -> Instruction:
    schema = get_operator_schema(opcode)
    assert schema is not None, opcode
    return Instruction(
        opcode,
        (wire,),
        params={name: rng.uniform(-6.0, 6.0) for name in schema.parameters},
    )


def _run_product_from_table(instructions: list[Instruction]) -> list[list[complex]]:
    """The composed matrix of a run, taken from the runtime gate table."""

    product = [row[:] for row in _IDENTITY]
    for instruction in instructions:
        product = _matmul(_runtime_matrix(instruction), product)
    return product


def test_every_declared_single_qubit_opcode_reads_back_exactly() -> None:
    """The fold's closed forms against the authority, over the whole group."""

    rng = random.Random(20261006)
    assert len(SINGLE_QUBIT_OPCODES) == 18
    worst = 0.0
    for opcode in SINGLE_QUBIT_OPCODES:
        for _ in range(40):
            instruction = _sample(opcode, rng)
            matrix = _instruction_matrix(instruction)
            assert matrix is not None, opcode
            worst = max(worst, _gap(matrix, _runtime_matrix(instruction)))
    # Every opcode is compared, so no gate can pass by not being read.
    assert worst < 1e-14


def test_the_three_convention_phases_are_the_only_deviations() -> None:
    """Only `sx`, `sxdg`, and `rz` carry a phase the Euler table does not.

    `_instruction_matrix` reproduces the runtime matrix for the fifteen opcodes
    whose Euler triple is already the runtime's matrix, and rescales the three
    that are not. This test measures both halves: that the fifteen are exact
    *without* a rescale, and that the three are exact only *with* one, so a
    change to `_convention_phase` cannot look like a change to the tables.
    """

    rng = random.Random(7)
    deviating: set[str] = set()
    for opcode in SINGLE_QUBIT_OPCODES:
        instruction = _sample(opcode, rng)
        matrix = _instruction_matrix(instruction)
        assert matrix is not None
        runtime = _runtime_matrix(instruction)
        if _gap(matrix, runtime) > 1e-14:
            deviating.add(opcode)
    assert deviating == set()
    assert set(_CONVENTION_DEGREES) == {"rz", "sx", "sxdg"}

    # `rz` at a non-zero angle is `phase` of the same angle times a half-angle
    # phase that depends on the angle, which is why the rescale has to read the
    # parameter rather than being a constant. Both matrices here come from the
    # runtime table and the factor is pure Python, so this pins the closed form
    # against the authority instead of against itself.
    for theta in (0.0, 0.91, -2.4):
        rotated = _runtime_matrix(Instruction("rz", (0,), params={"theta": theta}))
        scaled = _runtime_matrix(Instruction("phase", (0,), params={"theta": theta}))
        factor = cmath.exp(1j * -0.5 * theta)
        expected = [[factor * entry for entry in row] for row in scaled]
        assert _gap(rotated, expected) < 1e-15, theta


@pytest.mark.parametrize("length", (2, 3, 4, 5, 6, 7))
def test_a_folded_run_keeps_the_runs_matrix(length: int) -> None:
    """The fold is exact, against the runtime table, on every run length."""

    rng = random.Random(1000 + length)
    worst = 0.0
    folded_count = 0
    for _ in range(250):
        run = tuple(
            _sample(rng.choice(SINGLE_QUBIT_OPCODES), rng) for _ in range(length)
        )
        replacement = _fold_run(list(run))
        if replacement is None:
            continue
        folded_count += 1
        source = _run_product_from_table(list(run))
        result = _run_product_from_table(list(replacement))
        worst = max(worst, _gap(source, result))
        assert len(replacement) < length
    # A run of this length that never folds would make the assertion vacuous. A
    # two-gate run over the whole group folds about half the time: the product has
    # to be reachable as one `u3`, one `phase`, or one `rz`.
    assert folded_count > 100
    assert worst < 1e-12


def test_a_run_that_cannot_be_shorter_is_declined() -> None:
    # A run whose replacement would be no shorter is returned unchanged rather
    # than rewritten into the same number of gates, which is what makes the pass
    # idempotent and the fixed-point loop in the pipeline terminate. `sx sx`
    # reaches one `u3`, so its refusal is not about length -- the vocabulary
    # decline is what declines it, and the test below is where that is pinned.
    # What is pinned here is that a refused run is left exactly as it was.
    run = [Instruction("sx", (0,)), Instruction("sx", (0,))]
    assert _single_basis_pair(run) == ("rz", "sx")
    assert _fold_run(run) is None
    assert collapse_one_qubit_runs(CircuitIR(1, tuple(run))).instructions == tuple(run)


def test_a_run_one_target_vocabulary_spells_is_declined() -> None:
    """The decline that stops the fold from being undone by lowering.

    A run of `rz` and `sx` is what a target that publishes those two gates emits
    and consumes, so composing it here would only be re-spelled downstream. The
    same holds for a run of pulses with no z-rotation: the target lowering of a
    `u3` is five leaves, which is more than the short pulse run it replaced.
    """

    run = [
        Instruction("rz", (0,), params={"theta": 0.4}),
        Instruction("sx", (0,)),
        Instruction("rz", (0,), params={"theta": 0.7}),
    ]
    assert _single_basis_pair(run) == ("rz", "sx")
    assert collapse_one_qubit_runs(CircuitIR(1, tuple(run))).instructions == tuple(run)

    pulses_only = [Instruction("sx", (0,)), Instruction("sx", (0,))]
    assert _single_basis_pair(pulses_only) == ("rz", "sx")
    assert collapse_one_qubit_runs(CircuitIR(1, tuple(pulses_only))).instructions == (
        tuple(pulses_only)
    )

    # The decline is about one vocabulary, not about z-rotations: a run that
    # mixes two z-rotations is folded, because no single target alphabet spells
    # it and the fold is what shortens it.
    mixed = [
        Instruction("phase", (0,), params={"theta": 0.4}),
        Instruction("rz", (0,), params={"theta": 0.7}),
        Instruction("sx", (0,)),
        Instruction("sx", (0,)),
    ]
    assert _single_basis_pair(mixed) is None
    assert len(collapse_one_qubit_runs(CircuitIR(1, tuple(mixed))).instructions) < len(
        mixed
    )


def test_a_vocabulary_run_whose_product_is_the_identity_is_deleted() -> None:
    """The decline is about writing a run back, and there is nothing to write.

    A run spelled in one z-rotation/pulse vocabulary is what a target that
    publishes those two gates emits and consumes, so re-spelling it here would
    only be undone downstream. That reason needs a replacement: when the run's
    product is the identity the replacement is empty, and no lowering can put
    back what was never written. Both shapes below are what makes this a real gain
    rather than a restatement of `remove_identity`: neither is reachable by
    deleting a single gate, because no gate in either run is itself the identity.
    A run of two full `rz` turns also composes to the identity, but it is not
    spelled in the vocabulary -- it has no pulse -- so it was never declined and
    is not this rule's business.
    """

    shapes = (
        [Instruction("sx", (0,))] * 4,
        [
            Instruction("rx", (0,), params={"theta": 0.7}),
            Instruction("rx", (0,), params={"theta": -0.7}),
        ],
    )
    for run in shapes:
        assert _single_basis_pair(run) is not None
        for instruction in run:
            assert _instruction_matrix(instruction) != _IDENTITY
        assert _emit(_run_product(run), qubit=0, metadata={}) == ()
        assert _fold_run(list(run)) == ()
        assert collapse_one_qubit_runs(CircuitIR(1, tuple(run))).instructions == ()


def test_the_vocabulary_decline_still_holds_when_the_replacement_has_gates() -> None:
    """The exemption is the empty replacement, not a relaxation of the decline.

    `sx sx` reaches one `u3`, and a `u3` is exactly what the target's lowering
    re-spells into five gates; that is the case the decline was measured on and
    it is deliberately unchanged.
    """

    run = [Instruction("sx", (0,)), Instruction("sx", (0,))]
    replacement = _emit(_run_product(run), qubit=0, metadata={})
    assert [item.name for item in replacement] == ["u3"]
    assert _declines_vocabulary(run, replacement) is True

    # The rule is asked about the replacement as well as the run, so the same run
    # is not declined once its replacement is empty. Asserting both halves of the
    # rule on one run is what stops the exemption from being read as a rewrite of
    # `_single_basis_pair`.
    assert _declines_vocabulary(run, ()) is False
    assert _declines_vocabulary([Instruction("x", (0,))] * 2, replacement) is False


def test_a_minus_identity_vocabulary_run_is_not_deleted() -> None:
    """The edge the exemption must not cross: `-I` is a global phase.

    `CircuitIR` has no field for a global phase, so `rx(pi) rx(pi)` has to be
    written back as a rotation. Deleting it would be a smaller program than the
    one asked for, which is the wrong direction for this pass to fail in.
    """

    run = [
        Instruction("rx", (0,), params={"theta": cmath.pi}),
        Instruction("rx", (0,), params={"theta": cmath.pi}),
    ]
    product = _run_product(run)
    assert _gap(product, [[-1 + 0j, 0j], [0j, -1 + 0j]]) < 1e-12
    assert [item.name for item in _emit(product, qubit=0, metadata={})] == ["rz"]
    assert _fold_run(list(run)) is None
    assert collapse_one_qubit_runs(CircuitIR(1, tuple(run))).instructions == tuple(run)


def test_no_run_is_deleted_unless_its_product_is_the_identity() -> None:
    """A dense sweep, because a deletion is the direction that needs evidence.

    The fold reads a product of floats, so "the replacement is empty" is a
    tolerance test rather than an equality test. This pins what that tolerance
    actually reaches: every emptied run in the sweep is within `1e-12` of the
    identity, and the sweep is asserted to have emptied something, because a
    sweep that deletes nothing would pass this test while proving nothing.
    """

    rng = random.Random(20261119)
    angles = (
        0.0,
        1e-16,
        1e-12,
        1e-8,
        1e-4,
        1e-3,
        0.3,
        cmath.pi / 2,
        cmath.pi,
        2 * cmath.pi,
        2 * cmath.pi - 1e-9,
        2 * cmath.pi + 1e-9,
        4 * cmath.pi,
        -2 * cmath.pi,
        6 * cmath.pi,
    )
    emptied = 0
    wrong = 0
    minus_identity = 0
    minus_identity_emptied = 0
    worst = 0.0
    for _ in range(4000):
        run = [
            _random_angle_instruction(rng.choice(SINGLE_QUBIT_OPCODES), angles, rng)
            for _ in range(rng.randint(2, 5))
        ]
        product = _run_product(run)
        if product is None:
            continue
        if _emit(product, qubit=0, metadata={}):
            if _gap(product, [[-1 + 0j, 0j], [0j, -1 + 0j]]) < 1e-12:
                minus_identity += 1
            continue
        emptied += 1
        distance = _gap(product, _IDENTITY)
        worst = max(worst, distance)
        if distance > 1e-12:
            wrong += 1
        if _gap(product, [[-1 + 0j, 0j], [0j, -1 + 0j]]) < 1e-12:
            minus_identity_emptied += 1
    assert emptied > 0
    assert wrong == 0
    # Both halves of the boundary are populated: the sweep saw `-I` products, and
    # it deleted none of them. Without the first assertion the second would hold
    # on a population that never reached the edge.
    assert minus_identity > 0
    assert minus_identity_emptied == 0
    assert worst < 1e-12


def _random_angle_instruction(
    opcode: str, angles: tuple[float, ...], rng: random.Random
) -> Instruction:
    schema = get_operator_schema(opcode)
    assert schema is not None, opcode
    return Instruction(
        opcode,
        (0,),
        params={name: rng.choice(angles) for name in schema.parameters},
    )


def test_a_run_carrying_a_trainable_angle_is_left_alone() -> None:
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(-0.27, dtype=torch.float64, requires_grad=True)
    run = (
        Instruction("ry", (0,), params={"theta": theta}),
        Instruction("h", (1,)),
        Instruction("ry", (0,), params={"theta": phi}),
    )
    ir = CircuitIR(2, run)

    # An angle with a gradient has no number to compose with, so the run is not
    # read at all -- on either side of the `h`, which is on another wire.
    for instruction in run:
        if instruction.params:
            assert _instruction_matrix(instruction) is None
    assert collapse_one_qubit_runs(ir).instructions == run

    # The existing merge pass still folds the two trainable rotations into one,
    # and both gradients survive, which is the behaviour the fold must not take
    # over: a composed matrix would have no gradient to give back.
    merged = optimize(ir).instructions[0].params["theta"]
    merged.backward()
    assert theta.grad == pytest.approx(1.0)
    assert phi.grad == pytest.approx(1.0)


def test_a_batched_angle_is_left_to_the_runtimes_own_validation() -> None:
    """A fold may not turn a refused program into a running one.

    A matrix of angles is refused by the runtime, but its single element is
    readable, so composing it here would produce a scalar gate that the runtime
    then broadcasts to every batch entry -- a different program that raises no
    error. The fold therefore refuses any angle that carries a shape, and the
    refusal has to survive the whole pipeline, not just this pass.
    """

    for shape in ((1, 1), (1,), (3,)):
        theta = torch.ones(*shape)
        run = (
            Instruction("h", (0,)),
            Instruction("ry", (0,), params={"theta": theta}),
        )
        ir = CircuitIR(2, run, shape=shape if len(shape) > 1 else ())
        assert _instruction_matrix(run[1]) is None, shape
        assert collapse_one_qubit_runs(ir).instructions == run, shape
        assert optimize(ir).instructions == run, shape

    # A genuine zero-dimensional tensor is one angle and still folds, so the
    # guard is about the shape and not about the type.
    scalar = torch.tensor(0.31)
    assert scalar.ndim == 0
    assert (
        _instruction_matrix(Instruction("ry", (0,), params={"theta": scalar}))
        is not None
    )
    folded = collapse_one_qubit_runs(
        CircuitIR(
            2,
            (
                Instruction("h", (0,)),
                Instruction("ry", (0,), params={"theta": scalar}),
            ),
        )
    )
    assert [item.name for item in folded.instructions] == ["u3"]


def test_the_fold_is_wire_local_and_ends_at_a_wider_gate() -> None:
    """A wider gate ends the run instead of joining it, and does not move.

    `cx` is not foldable, so each side of it folds separately and the entangler
    keeps its position in the program. If the fold reached across it the two
    sides would compose and the program would be wrong.
    """

    ir = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("t", (0,)),
            Instruction("t", (0,)),
            Instruction("h", (1,)),
        ),
    )
    folded = collapse_one_qubit_runs(ir)

    # `h h` is the identity and is removed; `t t` is `s`, which is one `phase`.
    assert [item.name for item in folded.instructions] == ["cx", "phase", "h"]
    assert folded.instructions[0] == ir.instructions[2]
    assert folded.instructions[2] == ir.instructions[5]


def test_a_non_unitary_instruction_is_not_swallowed_by_a_fold() -> None:
    """A measurement or a channel ends the run and stays in place."""

    measure = Instruction("measure", (0,), metadata={"is_dynamic": True})
    ir = CircuitIR(
        1,
        (
            Instruction("h", (0,)),
            Instruction("h", (0,)),
            measure,
            Instruction("t", (0,)),
            Instruction("t", (0,)),
        ),
    )
    folded = collapse_one_qubit_runs(ir)

    assert folded.instructions[0] == measure
    assert [item.name for item in folded.instructions] == ["measure", "phase"]


def test_the_fold_is_idempotent() -> None:
    rng = random.Random(4242)
    checked = 0
    for _ in range(400):
        length = rng.randint(2, 8)
        run = tuple(
            _sample(rng.choice(SINGLE_QUBIT_OPCODES), rng) for _ in range(length)
        )
        once = collapse_one_qubit_runs(CircuitIR(1, run))
        twice = collapse_one_qubit_runs(once)
        assert twice.instructions == once.instructions
        checked += 1
    assert checked == 400


def test_the_fold_reproduces_a_whole_program_statevector() -> None:
    """Exactness on an entangled program, where a dropped phase would show.

    A one-wire program cannot fail this test, because a global phase cancels in
    every amplitude there. The entangler makes the folded rotations land on a
    relative phase, which is what the runtime comparison can see.
    """

    from flagquantum.simulation.statevector.local import run_local_statevector

    rng = random.Random(31337)
    worst = 0.0
    for _ in range(60):
        instructions: list[Instruction] = []
        for _ in range(rng.randint(4, 14)):
            if rng.random() < 0.25:
                instructions.append(Instruction("cx", (0, 1)))
            else:
                single = _sample(
                    rng.choice(SINGLE_QUBIT_OPCODES), rng, wire=rng.randrange(2)
                )
                instructions.append(single)
        source = CircuitIR(2, tuple(instructions), dtype="complex128")
        folded = collapse_one_qubit_runs(source)
        before = run_local_statevector(
            source, batch_size=1, device=torch.device("cpu"), dtype=torch.complex128
        )
        after = run_local_statevector(
            folded, batch_size=1, device=torch.device("cpu"), dtype=torch.complex128
        )
        worst = max(worst, float(torch.max(torch.abs(before - after)).item()))
    assert worst < 1e-12


def test_the_fold_preserves_program_metadata_and_wire_width() -> None:
    ir = CircuitIR(
        3,
        (
            Instruction("h", (2,)),
            Instruction("h", (2,)),
            Instruction("cx", (0, 1)),
        ),
        dtype="complex128",
    )
    folded = replace(ir, instructions=collapse_one_qubit_runs(ir).instructions)

    assert folded.n_wires == 3
    assert folded.dtype == "complex128"
    assert folded.instructions == ir.instructions[2:]


def test_the_fold_is_not_a_public_compiler_export() -> None:
    """The pass is internal, like the three passes it runs beside."""

    assert not hasattr(compiler, "collapse_one_qubit_runs")
    assert not hasattr(compiler, "remove_identity_gates")
    assert not hasattr(compiler, "merge_self_inverse")
    assert not hasattr(compiler, "merge_adjacent_rotations")


def test_the_pipeline_still_reaches_a_fixed_point_with_the_fold_in_it() -> None:
    rng = random.Random(99)
    for _ in range(200):
        length = rng.randint(1, 30)
        instructions = [
            _sample(rng.choice(SINGLE_QUBIT_OPCODES), rng, wire=rng.randrange(3))
            for _ in range(length)
        ]
        ir = CircuitIR(3, tuple(instructions))
        once = optimize(ir)
        assert optimize(once).instructions == once.instructions
