"""Conformance of the state-preparation boundary against one shared contract.

A classical amplitude vector reaches the compiler as a request to prepare a state,
and there are two answers to it: `synthesize_state_preparation`, which spells the
preparation in a target's published vocabulary, and
`flagquantum.algorithms.primitives.state_preparation`, the fixed-vocabulary routine
that `algorithms.svd` and `algorithms.pca` already call. The second is the frozen
reference: the change that adds the first does not edit it, and it stays named by
its two consumers. That is the replacement ARCH-012 clause 2 asks for -- a new
horizontal abstraction is admitted only while at least one implementation can be
swapped in behind the same boundary without modifying its consumers -- so this
module states the shared postconditions once and drives both implementations
through all of them, rather than asserting a different subset per implementation.

Both implementations meet a single consumer here: each produces a circuit, each
circuit is lowered to `CircuitIR`, and `run_local_statevector` runs both. That
consumer is the one the repository already has, and it is not told which
implementation produced its input.

The contract has five parts and it holds for every implementation that answers:

* the vocabulary is published: neither implementation invents an opcode, and the
  compiler-side one emits no entangler outside `SUPERCONTROLLED_ENTANGLERS`;
* the reach is stated rather than assumed: every case in the family is served by
  every published entangler, and a basis outside the published rosters is refused
  as `None` rather than answered with a replacement that was never verified;
* the observable is preserved up to exactly one global phase, so the instrument is
  the overlap and not the entrywise residual -- measured, and controlled below;
* the cost is a property of the basis and of the register width rather than of the
  state, and the entangler the caller names buys only the spelling of the control
  flip, never a different state; and
* the implementations are deterministic, do not mutate their input, keep every
  annotated leaf annotated, and answer with IR a consumer can validate and
  serialize. The refusals are typed: `None` for a basis this compiler cannot hold,
  `ValueError` for an input that is not an amplitude vector.

**Four properties of this boundary were measured rather than assumed, and each is
asserted as a difference so that a check never seen to fail is not mistaken for
evidence.** They are the reason the compiler-side implementation does not reuse the
one-qubit builder:

* a ladder's branches must drop the *same* global phase, and
  `one_qubit_synthesis`'s short Euler forms do not: they move it by a full `pi` when
  the polar angle reaches `pi`, which is a global phase for a single leaf and a
  relative phase once a ladder is involved. The general form
  `RZ(lam) PULSE RZ(theta - pi) PULSE RZ(phi - pi)` holds one constant instead.
  `test_the_short_euler_forms_cannot_carry_a_ladder_phase` measures both.
* only a true `RZ` can carry that form's z-rotation. `phase` and `u1` differ from
  `RZ(theta)` by exactly `exp(1j * theta / 2)`, which is harmless for one leaf and
  branch dependent over a ladder, because the angle is the branch index.
  `test_the_phase_two_rotations_differ_from_rz_by_half_their_angle` measures the
  mechanism and `test_only_a_true_rz_holds_one_dropped_phase_over_a_ladder`
  measures the consequence.
* a zero branch of a magnitude ladder may not be dropped the way a whole zero
  ladder may: a whole walk of flips multiplies to the identity, one branch's do not.
  `test_a_whole_zero_ladder_is_dropped_and_a_lone_zero_branch_is_not` measures the
  difference between doing it and not doing it.
* the basis buys the spelling of one `CX` and nothing else, so the leaf counts of
  the uniform family decompose into the flip cost the roster itself measures plus a
  per-level constant. `test_the_uniform_family_count_is_the_ladder_arithmetic` and
  `test_the_control_flip_cost_is_the_rosters_own_measurement` assert that
  arithmetic, so the counts and the cost cannot drift apart silently.

The case family is generated rather than enumerated, and the two numbers pinned as
counts are pinned on families where every drop is an exact comparison against an
exactly zero quantity: the `|0...0>` vector, the computational-basis vectors, and
the uniform superposition. Their ladder angles are exactly `0.0` or exactly `pi`,
so no count below depends on how a platform rounds a nearly cancelling sum. The
random points of the family are asserted only against their state and against the
frozen reference's state. This mirrors
`tests/team/compiler/test_two_qubit_decomposition_conformance.py` and
`test_optimization_pass_conformance.py`; what
`tests/unit/test_algorithms_state_preparation.py` cannot give it is that the file
states each implementation's own rule and never what the two share.
"""

from __future__ import annotations

import cmath
import math
import random
from collections.abc import Sequence

import pytest
import torch

from flagquantum.algorithms.primitives import state_preparation as reference
from flagquantum.compiler.one_qubit_synthesis import (
    Z_ROTATION_OPCODES,
    synthesize_one_qubit_matrix,
)
from flagquantum.compiler.state_preparation_synthesis import (
    _CONTROL_FLIP,
    _LADDER_Z_ROTATION,
    _control_flips,
    _control_walk,
    _magnitude_angles,
    _magnitude_ladder,
    _normalised,
    _phase_angles,
    _phase_ladder,
    _ry_euler_leaves,
    _walsh,
    synthesize_state_preparation,
)
from flagquantum.compiler.two_qubit_synthesis import (
    SUPERCONTROLLED_ENTANGLERS,
    synthesize_two_qubit,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.gate_matrix import gate_matrix
from flagquantum.simulation.statevector.local import run_local_statevector

pytestmark = pytest.mark.unit

#: The seed the pinned numbers in the comments below were measured at on the
#: authoring host. Nothing here is asserted to that host's last bit; see the margin
#: each number carries.
_SEED = 20261026

#: The two spellings every measurement in this file is driven with, the z-rotation
#: read from the module that owns the choice so this file cannot drift from it.
_Z_ROTATION = _LADDER_Z_ROTATION
_PULSE = "sx"

#: The six entries of the roster, read from the mapping rather than restated. The
#: measured order is `('cx', 'cz', 'cy', 'rzz', 'rxx', 'ryy')`.
_ENTANGLERS = tuple(SUPERCONTROLLED_ENTANGLERS)

#: The one-qubit vocabulary the compiler-side implementation may emit. `z_rotation`
#: is a ladder's only z-rotation, so naming the pair here is a vocabulary statement
#: and not a preference.
_ONE_QUBIT = frozenset({_Z_ROTATION, _PULSE})

#: What `synthesize_two_qubit` spends to spell one `CX(control, target)` in each
#: basis, measured at `_Z_ROTATION`/`_PULSE`. `cx` is the entangler itself, so its
#: flip is one leaf; every other route conjugates it into the named entangler. These
#: are asserted against that function's own answer below, not kept as a table.
_FLIP_COST = {
    "cx": 1,
    "cz": 9,
    "cy": 6,
    "rzz": 7,
    "rxx": 13,
    "ryy": 13,
}

#: The uniform superposition's leaf count for each entangler, `n = 1` through `5`,
#: measured at `_Z_ROTATION`/`_PULSE`.
_UNIFORM_COUNTS = {
    "cx": (4, 10, 18, 30, 50),
    "cz": (4, 26, 66, 142, 290),
    "cy": (4, 20, 48, 100, 200),
    "rzz": (4, 22, 54, 114, 230),
    "rxx": (4, 34, 90, 198, 410),
    "ryy": (4, 34, 90, 198, 410),
}

#: The leaf count of every computational-basis vector at `cx`, `n = 1` through `3`.
#: These are exact rather than rounded: a basis vector's ladder angles are exactly
#: `0.0` or exactly `pi`, and a `pi` group is three leaves rather than four because
#: its middle `rz` collapses to the identity. They are pinned because they are the
#: only family here whose cost is small enough to state as a row a reader can check
#: by hand.
_BASIS_COUNTS = (
    (0, 3),
    (0, 10, 3, 13),
    (0, 20, 10, 30, 3, 23, 13, 33),
)

#: The frozen reference's own leaf count on the uniform family, `n = 1` through `4`.
#: It is the counter-arm of `_UNIFORM_COUNTS`: the two implementations are free to
#: differ here, and this records by how much, so a claim that the replacement is
#: cheaper is made against a measurement rather than an expectation.
_REFERENCE_UNIFORM_COUNTS = (2, 10, 26, 58)

#: The tolerance every state comparison in this file uses when the quantity being
#: compared is a `complex128` state vector produced by an exactly specified circuit.
#: Measured on the authoring host the worst residual of either implementation is
#: `3.9e-16`, so this is two and a half orders of magnitude above the arithmetic
#: noise of the host it was measured on, and it is a bound rather than an equality
#: because the rounding of a sum is a property of the host.
_STATE_ATOL = 1.0e-12

#: The tolerance the comparison against the frozen reference uses. The reference
#: labels its amplitudes `complex64` at its own validation step, so it carries a
#: single-precision relative error of `1.19e-07`; measured entrywise the two answers
#: agree to `2.6e-08` through `5.7e-08`. This is one order of magnitude above
#: `complex64` epsilon and a bound rather than an equality because the reference's
#: own `2**n` by `2**n` solve has the conditioning of its input.
_REFERENCE_ATOL = 1.0e-06


def _uniform(n_qubits: int) -> list[complex]:
    """The uniform superposition's amplitudes, as Python complex numbers."""
    return [complex(1.0 / math.sqrt(2**n_qubits))] * (2**n_qubits)


def _basis(n_qubits: int, index: int) -> list[complex]:
    """The `index`-th computational-basis vector on `n_qubits` wires."""
    vector = [complex(0.0)] * (2**n_qubits)
    vector[index] = complex(1.0)
    return vector


def _direction(values: Sequence[complex]) -> torch.Tensor:
    """The `complex128` unit vector along `values`."""
    norm = math.sqrt(sum(abs(value) ** 2 for value in values))
    return torch.tensor([value / norm for value in values], dtype=torch.complex128)


def _state(leaves: Sequence[Instruction], n_wires: int) -> torch.Tensor:
    """Lower `leaves` to IR and run them, returning the flat state vector.

    This is the consumer both implementations are driven through, and it is
    deliberately the shipped simulation entry rather than a simulator local to this
    file: the point of the conformance module is that the two implementations meet
    where the repository already runs a circuit, not only where a test wants them to.
    """
    program = CircuitIR(n_wires=n_wires, instructions=tuple(leaves))
    program.validate()
    return run_local_statevector(
        program,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    ).reshape(-1)


def _prepared(
    values: Sequence[complex],
    *,
    entangler: str = "cx",
    qubits: Sequence[int] | None = None,
    **options: object,
) -> tuple[Instruction, ...]:
    """The compiler-side preparation of `values`, asserting it was served."""
    leaves = synthesize_state_preparation(
        values, qubits=qubits, entangler=entangler, **options
    )
    assert leaves is not None, f"{entangler!r} refused a preparation it should serve"
    return leaves


def _overlap(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The inner product of `left` with `right`, both flattened.

    `torch.vdot` conjugates its first argument, so the phase this returns is the
    phase of `right` relative to `left`, which is the direction the global-phase
    correction below divides out.
    """
    return torch.vdot(left.reshape(-1), right.reshape(-1))


def _gap(left: torch.Tensor, right: torch.Tensor, atol: float = _STATE_ATOL) -> float:
    """The entrywise residual once the one global phase is divided out."""
    overlap = _overlap(left, right)
    assert float(abs(overlap)) > 0.5, "the two states are not the same state"
    offset = overlap / abs(overlap)
    residual = float((right.reshape(-1) / offset - left.reshape(-1)).abs().max())
    assert residual < atol, f"entrywise residual {residual} exceeds {atol}"
    return residual


def _matrix_phase(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """The global phase of `right` relative to `left`, for two `d x d` matrices.

    A unitary of dimension `d` has `trace(U† U) == d`, so the phase is the trace
    overlap normalised by `d` rather than the untraced inner product the vector case
    uses.
    """
    overlap = torch.trace(left.conj().T @ right) / left.shape[0]
    return overlap / abs(overlap)


def _matrix_residual(
    left: torch.Tensor, right: torch.Tensor, atol: float = _STATE_ATOL
) -> float:
    """The entrywise residual of two matrices once one global phase is divided out."""
    return float((right / _matrix_phase(left, right) - left).abs().max())


def _matrix_gap(left: torch.Tensor, right: torch.Tensor, atol: float = _STATE_ATOL):
    """`_gap` for two same-shaped unitary matrices."""
    overlap = torch.trace(left.conj().T @ right) / left.shape[0]
    assert abs(float(abs(overlap)) - 1.0) < atol, "the two matrices are not one matrix"
    residual = _matrix_residual(left, right, atol)
    assert residual < atol, f"entrywise residual {residual} exceeds {atol}"
    return residual


def _one_wire_matrix(leaf: Instruction) -> torch.Tensor:
    """The `2x2` matrix of a single-wire instruction."""
    block = gate_matrix(leaf, bsz=1, device=torch.device("cpu"), dtype=torch.complex128)
    return block.reshape(-1, 2, 2)[0]


def _two_wire_matrix(leaf: Instruction) -> torch.Tensor:
    """The `4x4` matrix of a one- or two-wire instruction on a two-wire register.

    A single-wire leaf is lifted with the identity on the other wire, wire `0`
    reading as the most significant index bit, which is the convention both
    `synthesize_state_preparation` and the simulator use.
    """
    block = gate_matrix(leaf, bsz=1, device=torch.device("cpu"), dtype=torch.complex128)
    identity = torch.eye(2, dtype=torch.complex128)
    if len(leaf.wires) == 2:
        return block.reshape(-1, 4, 4)[0]
    single = block.reshape(-1, 2, 2)[0]
    if leaf.wires[0] == 0:
        return torch.kron(single, identity)
    return torch.kron(identity, single)


def _two_wire_unitary(leaves: Sequence[Instruction]) -> torch.Tensor:
    """The `4x4` unitary of a two-wire circuit, in emission order."""
    total = torch.eye(4, dtype=torch.complex128)
    for leaf in leaves:
        total = _two_wire_matrix(leaf) @ total
    return total


def _ry_matrix(theta: float) -> list[list[complex]]:
    """The `2x2` matrix of `RY(theta)`."""
    cosine = math.cos(0.5 * theta)
    sine = math.sin(0.5 * theta)
    return [[cosine, -sine], [sine, cosine]]


def _dropped_phase(leaves: Sequence[Instruction], theta: float) -> float:
    """The phase `s` with `product == s * RY(theta)`, for a single-wire circuit."""
    product = torch.eye(2, dtype=torch.complex128)
    for leaf in leaves:
        product = _one_wire_matrix(leaf) @ product
    target = torch.tensor(_ry_matrix(theta), dtype=torch.complex128)
    # `product == s * RY(theta)` gives `s == trace(RY(theta)† @ product) / 2`.
    return cmath.phase(complex(torch.trace(target.conj().T @ product) / 2.0))


def _flips(
    wires: Sequence[int], *, entangler: str = "cx"
) -> dict[tuple[int, int], tuple[Instruction, ...]]:
    """The control-flip table the ladders of this file are driven with."""
    table = _control_flips(
        wires,
        entangler=entangler,
        z_rotation=_Z_ROTATION,
        pulse_opcode=_PULSE,
        metadata={},
    )
    assert table is not None, f"{entangler!r} cannot spell a control flip"
    return table


def _random_points(n_qubits: int, count: int, seed: int) -> list[list[complex]]:
    """`count` deterministic random directions on `n_qubits`, as Python complex."""
    generator = random.Random(seed)
    points = []
    for _ in range(count):
        raw = [
            complex(generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0))
            for _ in range(2**n_qubits)
        ]
        norm = math.sqrt(sum(abs(value) ** 2 for value in raw))
        points.append([value / norm for value in raw])
    return points


def _ramp(n_qubits: int) -> list[complex]:
    """A uniform-magnitude state with a linear phase ramp over the basis index."""
    size = 2**n_qubits
    return [
        cmath.exp(1j * 2.0 * math.pi * index / size) / math.sqrt(size)
        for index in range(size)
    ]


def _family() -> list[tuple[int, list[complex]]]:
    """The generated case family as `(n_qubits, amplitudes)`.

    It mixes structured and random points on purpose. The uniform superposition and
    the computational-basis vectors have exactly zero or exactly `pi` ladder angles,
    which is what makes the pinned counts portable; the random points exercise the
    same code with nothing exact about them and are asserted only against a state.
    The phase-ramp points have uniform magnitudes, so the magnitude pass is empty and
    the phase pass is not.
    """
    points: list[tuple[int, list[complex]]] = []
    for n_qubits in (1, 2, 3):
        points.append((n_qubits, _uniform(n_qubits)))
        for index in range(2**n_qubits):
            points.append((n_qubits, _basis(n_qubits, index)))
        points.append((n_qubits, _ramp(n_qubits)))
        for values in _random_points(n_qubits, 3, _SEED + n_qubits * 10):
            points.append((n_qubits, values))
    return points


#: The family, generated once. `_state` is the expensive part of this file and
#: several checks walk the whole family, so it is built at import.
_FAMILY = _family()


def test_the_published_vocabulary_is_the_basis_and_nothing_else() -> None:
    """Neither implementation invents an opcode, and no entangler escapes the roster."""
    allowed = {entangler: _ONE_QUBIT | {entangler} for entangler in _ENTANGLERS}
    for entangler in _ENTANGLERS:
        for n_qubits, values in _FAMILY:
            names = {leaf.name for leaf in _prepared(values, entangler=entangler)}
            assert names <= allowed[entangler], (
                f"{entangler!r} on {n_qubits} wires emitted"
                f" {sorted(names - allowed[entangler])}"
            )
    # The reference arm, for the same reason: it may not invent a name either, and
    # the vocabulary it was frozen with is the fixed one it must keep.
    for n_qubits, values in _FAMILY:
        program = reference.arbitrary_state(
            torch.tensor(values, dtype=torch.complex64)
        ).to_ir()
        names = {instruction.name for instruction in program.instructions}
        assert names <= {"ry", "rz", "cx"}, f"reference emitted {sorted(names)}"


def test_every_published_entangler_serves_the_whole_family() -> None:
    """The reach is stated rather than assumed: the roster serves, and it agrees."""
    shared: tuple[Instruction, ...] | None = None
    for n_qubits, values in _FAMILY:
        want = _direction(values)
        for entangler in _ENTANGLERS:
            leaves = synthesize_state_preparation(values, entangler=entangler)
            assert leaves is not None, f"{entangler!r} refuses {n_qubits}-wire states"
            state = _state(leaves, n_qubits)
            assert abs(float(abs(_overlap(want, state))) - 1.0) < _STATE_ATOL
            if entangler == "cx":
                shared = leaves
            else:
                # Every basis is the same state, so the answers must agree with each
                # other and not only with the target.
                assert _gap(_state(shared, n_qubits), state) < _STATE_ATOL


def test_the_observable_is_preserved_up_to_one_global_phase() -> None:
    """The prepared state is the target times one phase, for every basis."""
    for n_qubits, values in _FAMILY:
        want = _direction(values)
        for entangler in _ENTANGLERS:
            leaves = _prepared(values, entangler=entangler)
            _gap(want, _state(leaves, n_qubits))


def test_the_entrywise_residual_cannot_stand_in_for_the_overlap() -> None:
    """Control: the instrument is the overlap because the residual is large.

    Without dividing the global phase out, the same two states differ by `O(1)`. A
    check written on the residual alone would fail on every correct answer, and a
    check written on the overlap alone would pass on one that only shares a phase,
    so the pair is what pins the equivalence at exactly one global phase.
    """
    worst = 0.0
    for n_qubits, values in _FAMILY:
        leaves = _prepared(values)
        state = _state(leaves, n_qubits)
        want = _direction(values)
        raw = float((state - want).abs().max())
        worst = max(worst, raw)
        _gap(want, state)
    assert worst > 0.1


def test_the_replacement_agrees_with_the_frozen_reference() -> None:
    """The two implementations prepare the same state, through the same consumer."""
    for n_qubits, values in _FAMILY:
        program = reference.arbitrary_state(
            torch.tensor(values, dtype=torch.complex64)
        ).to_ir()
        left = _state(program.instructions, program.n_wires)
        right = _state(_prepared(values), n_qubits)
        assert abs(float(abs(_overlap(left, right))) - 1.0) < _STATE_ATOL
        _gap(left, right, atol=_REFERENCE_ATOL)


def test_the_two_implementations_differ_by_a_quarter_turn_lattice() -> None:
    """The dropped phase is per implementation, and it is measured, not assumed.

    The two answers are the same state times two different constants, so the overlap
    between them is a unit complex number. Measured on the authoring host its phase is
    a multiple of `pi/2` at every point of the family and it is data dependent within
    that lattice: `0` at `n = 1` for the uniform, ramp and random points, `pi` for the
    `n = 2` uniform, `+pi/2` for the `n = 3` uniform. Two claims are asserted, and the
    second is the one that matters:

    * the phase lands on the quarter-turn lattice, which is what two answers built
      from the same ladder angles and the same fixed `pi/2` pulses must do; and
    * `|0...0>` is the exception and is exact rather than merely on the lattice. There
      the compiler side emits no leaf at all and the reference emits its ladder, so a
      shared global phase would be a statement about the reference's own arithmetic
      rather than about this boundary. Measured, the phase there is `0` to
      `_REFERENCE_ATOL`, so the two answers are the same vector and not merely the
      same ray.

    The sign is the reference's global-phase column moving, so it is deliberately not
    pinned; the lattice, the exact zero, and the state agreement are this boundary's.
    """
    turns = set()
    for n_qubits, values in _FAMILY:
        program = reference.arbitrary_state(
            torch.tensor(values, dtype=torch.complex64)
        ).to_ir()
        left = _state(program.instructions, program.n_wires)
        right = _state(_prepared(values), n_qubits)
        overlap = _overlap(left, right)
        assert abs(float(abs(overlap)) - 1.0) < _STATE_ATOL
        angle = cmath.phase(complex(overlap))
        count = angle / (0.5 * math.pi)
        assert abs(count - round(count)) < _REFERENCE_ATOL, f"{angle} is off lattice"
        turns.add(round(count) % 4)
        if values == _basis(n_qubits, 0):
            assert abs(angle) < _REFERENCE_ATOL, f"|0...0> is {angle} apart"
    # The lattice is not vacuous: the family reaches more than one of its four points,
    # so the assertion above is not a statement that every phase is zero.
    assert len(turns) > 1


def test_the_replacement_needs_no_consumer_edit() -> None:
    """The swap is behind an existing boundary: the consumer is not told which.

    `run_local_statevector` receives a `CircuitIR` and nothing about its origin. The
    two implementations are swapped in front of it and it answers both. The
    reference's own public shape is asserted here too, because that shape is what
    `algorithms.svd` and `algorithms.pca` call and what the swap must leave alone.
    """
    values = _random_points(2, 1, _SEED + 1)[0]
    tensor = torch.tensor(values, dtype=torch.complex64)
    want = _direction(values)
    ours = CircuitIR(n_wires=2, instructions=_prepared(values))
    theirs = reference.arbitrary_state(tensor).to_ir()
    assert isinstance(ours, CircuitIR) and isinstance(theirs, CircuitIR)
    for program in (ours, theirs):
        _gap(want, _state(program.instructions, program.n_wires), _REFERENCE_ATOL)
        assert program.to_json() == CircuitIR.from_dict(program.to_dict()).to_json()
    # The consumers' two call shapes, unchanged: a builder and an in-place append.
    built = reference.arbitrary_state(tensor).to_ir()
    assert len(built.instructions) == len(theirs.instructions)
    circuit = reference.Circuit(2)
    assert reference.append_arbitrary_state(circuit, tensor, (0, 1)) is None
    assert len(circuit.to_ir().instructions) == len(theirs.instructions)


def test_the_uniform_family_count_is_the_ladder_arithmetic() -> None:
    """The cost of the uniform superposition decomposes into its two spendings.

    A uniform state has exactly one non-zero branch per magnitude level, so its `ry`
    groups contribute `4 * n` leaves and the whole leaf count is
    `4 * n + flip_cost * (2**n - 2)`: `n` four-leaf groups, and `2**n - 2` control
    flips because level `j` spends `2**j` and level `0` spends none. The phase pass is
    empty here, checked separately below, so nothing else is added. The measured counts
    are asserted to equal that arithmetic rather than the other way round, so a change
    in the flip cost moves both sides together.

    The pulse count is only `2 * n` for the entangler that *is* a pulse-free flip.
    Every other basis spells the flip out of pulses of its own, so its count is higher
    by a measured amount and the assertion is a floor rather than an equality; pinning
    the equality for `cx` alone keeps the number exact where it is exact.
    """
    for n_qubits in (1, 2, 3, 4, 5):
        size = 2**n_qubits
        for entangler in _ENTANGLERS:
            leaves = _prepared(_uniform(n_qubits), entangler=entangler)
            names = [leaf.name for leaf in leaves]
            assert len(leaves) == _UNIFORM_COUNTS[entangler][n_qubits - 1]
            assert len(leaves) == 4 * n_qubits + _FLIP_COST[entangler] * (size - 2)
            assert names.count(_PULSE) >= 2 * n_qubits
            if entangler == "cx":
                # One `cx` per flip, one group of two pulses per level, and nothing
                # else: the flip is free of pulses only in this basis.
                assert names.count(_PULSE) == 2 * n_qubits
                assert names.count("cx") == size - 2
    # The reference arm of the same table, so the two implementations' costs are read
    # off one family rather than off two separate measurements.
    for n_qubits, expected in enumerate(_REFERENCE_UNIFORM_COUNTS, start=1):
        program = reference.arbitrary_state(
            torch.tensor(_uniform(n_qubits), dtype=torch.complex64)
        ).to_ir()
        assert len(program.instructions) == expected
    # The reference arm of the same table, so the two implementations' costs are read
    # off one family rather than off two separate measurements.
    for n_qubits, expected in enumerate(_REFERENCE_UNIFORM_COUNTS, start=1):
        program = reference.arbitrary_state(
            torch.tensor(_uniform(n_qubits), dtype=torch.complex64)
        ).to_ir()
        assert len(program.instructions) == expected


def test_the_control_flip_cost_is_the_rosters_own_measurement() -> None:
    """The basis buys the spelling of one `CX` and nothing else.

    `_FLIP_COST` is asserted against `synthesize_two_qubit`'s own answer rather than
    against a table this file keeps, so the leaf counts above and this cost cannot
    drift apart silently. Every route reproduces `CX` itself up to one global phase.
    """
    target = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.complex128,
    )
    for entangler in _ENTANGLERS:
        leaves = synthesize_two_qubit(
            _CONTROL_FLIP,
            wires=(0, 1),
            entangler=entangler,
            z_rotation=_Z_ROTATION,
            pulse_opcode=_PULSE,
        )
        assert leaves is not None, f"{entangler!r} refused the flip"
        assert len(leaves) == _FLIP_COST[entangler]
        product = _two_wire_unitary(leaves)
        _matrix_gap(target, product)
        assert {leaf.name for leaf in leaves} <= _ONE_QUBIT | {entangler}


def test_the_shortest_spelling_is_the_callers_basis() -> None:
    """The cheapest answer is the one whose entangler the caller already has."""
    for n_qubits in (2, 3, 4):
        counts = {
            entangler: len(_prepared(_uniform(n_qubits), entangler=entangler))
            for entangler in _ENTANGLERS
        }
        assert set(counts) == set(_ENTANGLERS)
        assert counts["cx"] == min(counts.values())
        assert sum(1 for count in counts.values() if count == counts["cx"]) == 1
        # A more expensive flip costs more in proportion, rather than in a way that
        # depends on the state: the spread is exactly the flip cost's spread.
        for entangler, count in counts.items():
            assert count - counts["cx"] == (_FLIP_COST[entangler] - 1) * (
                2**n_qubits - 2
            )


def test_a_one_wire_register_never_spends_an_entangler() -> None:
    """With no ladder there is no control flip, so the basis cannot cost anything."""
    for entangler in _ENTANGLERS:
        leaves = _prepared(_uniform(1), entangler=entangler)
        assert len(leaves) == 4
        assert {leaf.name for leaf in leaves} == _ONE_QUBIT
    # A basis the roster refuses is still refused, because the refusal is decided
    # before a register width can excuse it.
    assert synthesize_state_preparation(_uniform(1), entangler="iswap") is None


def test_the_basis_vector_counts_are_the_exact_zero_drops() -> None:
    """A power-of-two family whose every drop is an exact comparison.

    These are the counts a reader can check by hand, and they are pinned for that
    reason. `|0...0>` costs nothing at every width, because every ladder angle is
    exactly `0.0` and every level is dropped whole. Any other basis vector costs a
    four-leaf group wherever its bit pattern is set, three leaves where the angle is
    exactly `pi` because that group's middle `rz` is the identity, and one control
    flip per step of each ladder it reaches.
    """
    for n_qubits, expected in enumerate(_BASIS_COUNTS, start=1):
        counts = [
            len(_prepared(_basis(n_qubits, index))) for index in range(2**n_qubits)
        ]
        assert counts == list(expected)
        assert counts[0] == 0
    for n_qubits in (4, 5):
        assert len(_prepared(_basis(n_qubits, 0))) == 0


def test_the_zero_state_is_the_identity_and_not_a_global_phase() -> None:
    """The zero drops in the magnitude pass are exact, in the way that matters.

    `|0...0>` produces no leaves at all, which is only correct if every dropped
    ladder is the identity rather than the identity times one phase per level. The
    check is on the state: an empty circuit and the prepared `|0...0>` are the same
    vector, entrywise and without dividing anything out.
    """
    for n_qubits in (1, 2, 3, 4):
        size = 2**n_qubits
        leaves = _prepared(_basis(n_qubits, 0))
        assert leaves == ()
        state = _state(leaves, n_qubits)
        want = torch.zeros(size, dtype=torch.complex128)
        want[0] = complex(1.0)
        assert float((state - want).abs().max()) == 0.0


def test_the_phase_pass_is_empty_on_the_uniform_family_and_not_otherwise() -> None:
    """The phase pass is separable from the magnitude pass, and measured so.

    Uniform magnitudes make every phase angle exactly zero, so the phase pass
    contributes nothing and the counts above are the magnitude pass alone. A state
    with a phase ramp makes the magnitude pass uniform and the phase pass non-empty,
    which is the counter-arm: the property asserted is that the two passes are told
    apart by the input, not that one of them is always idle.
    """
    for n_qubits in (1, 2, 3, 4, 5):
        angles = _phase_angles(_uniform(n_qubits), n_qubits)
        assert len(angles) == n_qubits
        assert all(not any(level) for level in angles)
    for n_qubits in (1, 2, 3):
        size = 2**n_qubits
        ramp = _ramp(n_qubits)
        assert _magnitude_angles(ramp, n_qubits, 0) == pytest.approx(
            [0.5 * math.pi] * 1, abs=_STATE_ATOL
        )
        assert any(any(level) for level in _phase_angles(ramp, n_qubits))
        assert len(_prepared(ramp)) > 4 * n_qubits + (size - 2)


def test_the_phase_pass_runs_after_every_magnitude_ladder() -> None:
    """The shipped answer is exactly its two documented passes in order.

    All magnitude ladders first, then all phase ladders. This is stated against the
    module's own passes rather than by inspecting the emitted names, because an `rz`
    leaf and a control flip appear in both passes and names cannot separate them;
    the sequence can. The `_normalised` call is what the shipped entry point does to
    its input first, so the two sides are fed the same angles rather than the same
    vector: comparing raw amplitudes would compare against a differently rounded
    normalisation.
    """
    for n_qubits, values in _FAMILY:
        if n_qubits == 1:
            continue
        data = _normalised(values)
        flips = _flips(tuple(range(n_qubits)))
        rebuilt: list[Instruction] = []
        for level in range(n_qubits):
            rebuilt.extend(
                _magnitude_ladder(
                    _magnitude_angles(data, n_qubits, level),
                    target=level,
                    controls=tuple(range(level)),
                    flips=flips,
                    z_rotation=_Z_ROTATION,
                    pulse_opcode=_PULSE,
                    metadata={},
                )
            )
        for level, angles in enumerate(_phase_angles(data, n_qubits)):
            rebuilt.extend(
                _phase_ladder(
                    angles,
                    target=level,
                    controls=tuple(range(level)),
                    flips=flips,
                    z_rotation=_Z_ROTATION,
                    metadata={},
                )
            )
        assert tuple(rebuilt) == _prepared(values)


def test_the_control_walk_spends_two_to_the_k_flips_and_returns_home() -> None:
    """A `k`-control ladder visits every branch once and spends exactly `2**k` flips.

    The flips must multiply to the identity, which is what makes the ladder a
    controlled rotation rather than a change of basis: the walk starts at branch `0`,
    ends at `2**(k-1)`, and the closing flip is always the first control. The count
    is the property that bounds every ladder's cost in the pinned table above.
    """
    for k in range(1, 6):
        controls = tuple(range(k))
        order, flips = _control_walk(controls)
        assert len(order) == len(flips) == 2**k
        assert sorted(order) == list(range(2**k))
        assert order[0] == 0
        assert order[-1] == 2 ** (k - 1)
        assert flips[-1] == controls[0]
        assert set(flips) <= set(controls)
        # Replaying the walk from branch `0` visits `order` and returns to `0`.
        # Flipping the walk's `i`-th control toggles bit `k - 1 - i` of the branch
        # index, so `controls[0]` is the most significant branch bit and the last
        # control is the least significant one. The two assertions below are what
        # pins that reading, rather than the comment stating it.
        bit = {control: k - 1 - index for index, control in enumerate(controls)}
        branch = 0
        for step in range(1, 2**k):
            branch ^= 1 << bit[flips[step - 1]]
            assert branch == order[step]
        branch ^= 1 << bit[flips[-1]]
        assert branch == 0


def test_only_a_true_rz_can_carry_a_ladder() -> None:
    """Control: the refusal is decided by the basis and not by the input.

    The same amplitude vector is handed to `synthesize_state_preparation` with each
    member of `Z_ROTATION_OPCODES`. Exactly one of them is served, and the others are
    refused rather than answered, which is the fail-closed half of the module's
    contract. `phase` and `u1` are legal z-rotations for a single leaf: the schema
    declares them, and `one_qubit_synthesis` emits them. What they cannot do is hold
    one dropped phase across the branches of a ladder, which is why the accepted set
    is a strict subset of the declared one.
    """
    values = _uniform(3)
    assert set(Z_ROTATION_OPCODES) > {_Z_ROTATION}
    for z_rotation in Z_ROTATION_OPCODES:
        answer = synthesize_state_preparation(values, z_rotation=z_rotation)
        if z_rotation == _Z_ROTATION:
            assert answer is not None
        else:
            assert answer is None, f"{z_rotation!r} should not carry a ladder"


def test_the_short_euler_forms_cannot_carry_a_ladder_phase() -> None:
    """Control: the reason the general Euler form is emitted unconditionally.

    `synthesize_one_qubit_matrix` takes the shortest available spelling of `RY(theta)`
    and those short forms do not hold one dropped phase across the polar-angle
    branches: at `theta == pi` the phase moves by a full `pi`, and at `theta == 0`
    there is nothing to hold it with at all. A ladder multiplies branch-dependent
    phases into relative phases, so it may not use them. The shipped general form
    holds one constant over every angle measured here. Both arms are asserted as a
    *spread*, because the constant itself is a global phase and therefore
    unobservable; what a ladder needs is that it does not move.
    """
    angles = (0.0, 1.0e-9, 0.3, 0.5 * math.pi, 2.0, math.pi - 1.0e-9, math.pi, -1.7)
    short: list[float] = []
    general: list[float] = []
    for theta in angles:
        leaves = synthesize_one_qubit_matrix(
            _ry_matrix(theta), wire=0, z_rotation=_Z_ROTATION
        )
        assert leaves is not None
        short.append(_dropped_phase(leaves, theta))
        general.append(
            _dropped_phase(
                _ry_euler_leaves(
                    theta,
                    target=0,
                    z_rotation=_Z_ROTATION,
                    pulse_opcode=_PULSE,
                    metadata={},
                ),
                theta,
            )
        )
    assert max(general) - min(general) < _STATE_ATOL
    assert max(abs(value + 0.5 * math.pi) for value in general) < _STATE_ATOL
    assert max(short) - min(short) > math.pi - 1.0e-06


def test_the_phase_two_rotations_differ_from_rz_by_half_their_angle() -> None:
    """The mechanism behind the refusal: the offset is half the leaf's own angle.

    `phase(theta)` and `u1(theta)` are `exp(1j * theta / 2) * RZ(theta)`. For one leaf
    that is a global phase; over a ladder the angle is the branch index, so the offset
    becomes branch dependent. The scalar is measured here rather than inferred from
    the schema, and it is what the refusal of those two opcodes is protecting.
    """
    for theta in (0.2, 1.0, math.pi, -1.7, 3.0):
        rotation = _one_wire_matrix(
            Instruction(_Z_ROTATION, (0,), params={"theta": theta})
        )
        for opcode in ("phase", "u1"):
            got = _one_wire_matrix(Instruction(opcode, (0,), params={"theta": theta}))
            offsets = [
                complex(got[row][column] / rotation[row][column])
                for row in range(2)
                for column in range(2)
                if abs(rotation[row][column]) > _STATE_ATOL
            ]
            assert offsets
            offset = offsets[0]
            assert all(abs(value - offset) < _STATE_ATOL for value in offsets)
            assert abs(offset) == pytest.approx(1.0, abs=_STATE_ATOL)
            assert cmath.phase(offset) == pytest.approx(0.5 * theta, abs=_STATE_ATOL)
            assert float((got - offset * rotation).abs().max()) < _STATE_ATOL


def test_only_a_true_rz_holds_one_dropped_phase_over_a_ladder() -> None:
    """Control: the consequence of the half-angle offset, on a real ladder.

    The same phase ladder is emitted twice, once with `rz` and once with `phase`. The
    two unitaries differ by `O(1)` entrywise, which would be unobservable on its own
    if the difference were a constant; it is not, because the same comparison on a
    second branch gives a different relative phase. So no single factor turns the
    refused basis into an accepted one, which is exactly what the refusal asserts.
    """
    flips = _flips((0, 1))
    offsets: list[float] = []
    for angles in ([1.0, 0.5], [3.0, -3.0]):
        with_rz = _two_wire_unitary(
            _phase_ladder(
                angles,
                target=1,
                controls=(0,),
                flips=flips,
                z_rotation=_Z_ROTATION,
                metadata={},
            )
        )
        with_phase = _two_wire_unitary(
            _phase_ladder(
                angles,
                target=1,
                controls=(0,),
                flips=flips,
                z_rotation="phase",
                metadata={},
            )
        )
        assert float((with_phase - with_rz).abs().max()) > 1.0e-02
        offsets.append(cmath.phase(complex(_matrix_phase(with_rz, with_phase))))
    assert offsets[0] != pytest.approx(offsets[1], abs=1.0e-03)
    assert abs(offsets[0] - offsets[1]) > 1.0e-03


def test_a_whole_zero_ladder_is_dropped_and_a_lone_zero_branch_is_not() -> None:
    """The one simplification the magnitude pass is allowed, and its limit.

    A ladder whose every branch angle is zero is dropped whole: the flips multiply to
    the identity and every branch would apply `RY(0)`. A single zero branch may drop
    only its rotation group, keeping the flips the walk needs to reach the branches
    around it. The two arms are separated by an explicit emitter that keeps every
    group, and the control below shows the flips are load-bearing rather than
    decorative, so "keep the flips" is a claim with content.
    """
    flips = _flips((0, 1))
    assert (
        _magnitude_ladder(
            [0.0, 0.0],
            target=1,
            controls=(0,),
            flips=flips,
            z_rotation=_Z_ROTATION,
            pulse_opcode=_PULSE,
            metadata={},
        )
        == []
    )
    # The flips alone are the identity, which is why the whole-ladder drop is exact.
    assert (
        float((_two_wire_unitary(list(flips[(0, 1)]) * 2) - torch.eye(4)).abs().max())
        == 0.0
    )
    # `1.2 + (-1.2)` is exactly `0.0`, so the first Walsh component is exactly zero
    # while the second is not. Branch `0` is the first step of the walk, so the
    # shipped ladder reaches it with no flip and reaches the non-zero branch with one.
    angles = [1.2, -1.2]
    alpha = _walsh(angles, 1)
    assert alpha[0] == 0.0 and alpha[1] != 0.0
    shipped = _magnitude_ladder(
        angles,
        target=1,
        controls=(0,),
        flips=flips,
        z_rotation=_Z_ROTATION,
        pulse_opcode=_PULSE,
        metadata={},
    )
    order, walk = _control_walk((0,))
    explicit: list[Instruction] = []
    for step, branch in enumerate(order):
        if step:
            explicit.extend(flips[(walk[step - 1], 1)])
        explicit.extend(
            _ry_euler_leaves(
                alpha[branch],
                target=1,
                z_rotation=_Z_ROTATION,
                pulse_opcode=_PULSE,
                metadata={},
            )
        )
    explicit.extend(flips[(walk[-1], 1)])
    # The shipped ladder is the one that keeps every flip and drops the zero group.
    zero = _ry_euler_leaves(
        0.0,
        target=1,
        z_rotation=_Z_ROTATION,
        pulse_opcode=_PULSE,
        metadata={},
    )
    assert zero, "the general Euler form renders RY(0) with a leaf, not as nothing"
    assert len(shipped) == len(explicit) - len(zero)
    assert _matrix_gap(_two_wire_unitary(explicit), _two_wire_unitary(shipped))
    # The control: the flips are what make the ladder controlled. With the same
    # rotation groups and no flips at all the non-zero branch's group applies to both
    # branches, which is a different unitary once one global phase is divided out --
    # so "keep the flips" is not a statement about bookkeeping.
    without = _magnitude_ladder(
        angles,
        target=1,
        controls=(0,),
        flips=dict.fromkeys(flips, ()),
        z_rotation=_Z_ROTATION,
        pulse_opcode=_PULSE,
        metadata={},
    )
    # Every flip of the walk is a control flip the ladder would otherwise rely on.
    removed = sum(
        len(flips[(walk[step - 1], 1)]) for step in range(1, len(order))
    ) + len(flips[(walk[-1], 1)])
    assert removed == 2 * len(flips[(walk[0], 1)])
    assert len(without) == len(shipped) - removed
    assert _matrix_residual(_two_wire_unitary(shipped), _two_wire_unitary(without)) > (
        1.0e-02
    )


def test_the_amplitude_validation_errors_are_typed() -> None:
    """`ValueError` for an input that is not an amplitude vector, by name."""
    cases = (
        (7, "amplitudes must be a sequence of numbers, got int"),
        ("0101", "amplitudes must be a sequence of numbers, got str"),
        ([0.5], "amplitudes needs at least two entries, got 1"),
        (
            [0.5] * 6,
            "amplitudes length must be a power of two, got 6; a register of n"
            " qubits carries 2**n amplitudes",
        ),
        (
            [0.0] * 4,
            "amplitudes must not be the zero vector; nothing can be normalised",
        ),
        ([[0.5, 0.5]], "amplitudes must be one-dimensional, entry 0 is a sequence"),
        ([0.5, "a"], "amplitudes entry 1 is not a number: 'a'"),
    )
    for value, message in cases:
        with pytest.raises(ValueError) as error:
            synthesize_state_preparation(value)
        assert str(error.value) == message
    # The accepted arm of the same call, so the check is not a blanket refusal.
    assert len(_prepared([0.5, 0.5])) == 4


def test_the_wire_validation_errors_are_typed() -> None:
    """`ValueError` for a wire map that does not fit the vector, by name."""
    cases = (
        ((0,), "qubits must name 2 wires for this amplitude vector, got 1"),
        ((0, 0), "qubits must name distinct wires"),
        ((-1, 0), "qubits must be non-negative integers"),
        ((True, 0), "qubits must be non-negative integers"),
        ((0, 1.0), "qubits must be non-negative integers"),
    )
    for qubits, message in cases:
        with pytest.raises(ValueError) as error:
            synthesize_state_preparation([0.5] * 4, qubits=qubits)
        assert str(error.value) == message
    assert len(_prepared([0.5] * 4, qubits=(3, 0))) == 10


def test_the_amplitude_input_is_any_sequence_or_tolist_object() -> None:
    """The input is accepted as numbers, which is why no library is imported.

    An object exposing `tolist` is converted first, so a tensor or an array of any
    framework reaches this boundary without the compiler package importing the library
    that produced it. A gradient-carrying tensor is accepted too, because the angles
    are a classical precomputation rather than a differentiable path; the assertion is
    that its answer is the same as for the plain values, so no gradient is silently
    threaded into a discrete decomposition.
    """
    values = [0.5, 0.5j, -0.5, 0.5]
    tensor = torch.tensor(values, dtype=torch.complex64)
    baseline = tuple(leaf.name for leaf in _prepared(values))
    assert baseline
    for arm in (list(values), tuple(values), tensor, tensor.tolist()):
        assert tuple(leaf.name for leaf in _prepared(arm)) == baseline
    assert tuple(leaf.name for leaf in _prepared([1, 0])) == ()
    carrying = tensor.clone().requires_grad_(True)
    assert tuple(leaf.name for leaf in _prepared(carrying)) == baseline
    assert all(
        isinstance(leaf.params.get("theta"), float)
        for leaf in _prepared(carrying)
        if "theta" in leaf.params
    )


def test_the_preparation_is_deterministic_and_does_not_mutate_its_input() -> None:
    """Two calls agree leaf for leaf, and the vector handed in is not written to."""
    for n_qubits, values in _FAMILY:
        before = list(values)
        first = synthesize_state_preparation(values)
        second = synthesize_state_preparation(values)
        assert first is not None and second is not None
        assert isinstance(first, tuple)
        assert first == second
        assert values == before
        for entangler in _ENTANGLERS:
            assert _prepared(values, entangler=entangler) == _prepared(
                values, entangler=entangler
            )


def test_the_metadata_reaches_every_leaf() -> None:
    """An annotation the caller supplies is on every leaf, or on none."""
    annotation = {"origin": "state-preparation-conformance"}
    for n_qubits, values in _FAMILY:
        plain = _prepared(values)
        marked = synthesize_state_preparation(values, metadata=annotation)
        assert marked is not None
        assert len(marked) == len(plain)
        assert all(leaf.metadata == annotation for leaf in marked)
        assert all(leaf.metadata is not annotation for leaf in marked)
        assert all(leaf.metadata == {} for leaf in plain)
        assert synthesize_state_preparation(values, metadata={}) == plain
    # `|0...0>` has no leaf to annotate, and that is still a served answer.
    assert synthesize_state_preparation(_basis(2, 0), metadata=annotation) == ()


def test_the_wire_map_is_a_permutation_rather_than_a_relabelling() -> None:
    """`qubits` names where the register lives, most significant first.

    A permutation of the wires permutes the amplitudes and nothing else, so the same
    preparation on a permuted map is the same state with its index bits moved. The
    assertion is stated on the moved support, so an implementation that read `qubits`
    as a contiguous offset would fail rather than pass by symmetry.
    """
    values = _uniform(3)
    want = _direction(values)
    for qubits in ((0, 1, 2), (2, 1, 0), (0, 2, 1), (1, 0, 2), (1, 3, 5)):
        n_wires = max(qubits) + 1
        state = _state(_prepared(values, qubits=qubits), n_wires)
        placed = torch.zeros(2**n_wires, dtype=torch.complex128)
        for index in range(8):
            target = 0
            for position, wire in enumerate(qubits):
                target |= ((index >> (2 - position)) & 1) << (n_wires - 1 - wire)
            placed[target] = want[index]
        # The preparation is only defined up to one global phase, so the comparison
        # divides it out rather than requiring the two conventions to pick the same
        # one; the support below is asserted separately and is phase independent.
        _gap(placed, state)
        assert float(abs(_overlap(placed, state))) > 0.99
        support = {int(index) for index in state.abs().gt(1.0e-9).nonzero().flatten()}
        assert support == {
            sum(
                ((index >> (2 - position)) & 1) << (n_wires - 1 - wire)
                for position, wire in enumerate(qubits)
            )
            for index in range(8)
        }


def test_the_result_is_a_valid_circuit() -> None:
    """The answer is IR a consumer can validate, serialize, and re-read."""
    for n_qubits, values in _FAMILY:
        leaves = _prepared(values)
        program = CircuitIR(n_wires=n_qubits, instructions=leaves)
        program.validate()
        assert program.to_dict()["kind"] == "flagquantum.circuit_ir"
        assert program.n_wires == n_qubits
        assert tuple(program.instructions) == leaves
        assert program.to_json() == CircuitIR.from_dict(program.to_dict()).to_json()
        assert (
            program.content_hash == CircuitIR.from_dict(program.to_dict()).content_hash
        )
