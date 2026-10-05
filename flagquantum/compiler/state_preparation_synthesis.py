"""State preparation in a target's own gate vocabulary.

A classical amplitude vector is prepared by a ladder of uniformly controlled
rotations, following Möttönen, Vartiainen, Bergholm, and Salomaa, "Transformation
of quantum states using uniformly controlled rotations", *Quantum Information and
Computation* **5**, 467 (2005), arXiv:quant-ph/0407010. The register is walked one
qubit at a time: level `j` acts on qubit `j` and is controlled by qubits `0` to
`j-1`, so a level with `k` controls carries `2**k` branch angles and is emitted as
a Gray-code walk of `2**k` control flips with one leaf between consecutive
branches. Two passes are needed, all magnitude ladders first and then all phase
ladders, because a ladder's magnitude angles depend on the amplitudes' moduli and
its phase angles on their arguments.

What this module adds over the circuit-construction routine in
`flagquantum.algorithms.primitives.state_preparation` is that the answer is
spelled in a target's published vocabulary rather than in a fixed one. The caller
names the z-rotation the target publishes, the opcode it publishes for a pi/2
pulse about x, and one supercontrolled entangler from
`SUPERCONTROLLED_ENTANGLERS`; the control flip of every ladder step is handed to
`synthesize_two_qubit`, so this module owns the ladder and that one owns the
spelling of `CX`. Raising the entangler cost from `cx` to, say, `cz` therefore
changes the leaf count without changing the prepared state.

Three properties are load-bearing and each was measured rather than assumed.

*The dropped global phase has to be the same in every branch of a ladder.* Each
leaf is emitted up to a global phase that FlagQuantum IR has no field to record,
which is harmless for one leaf and fatal for a ladder: a branch dependent offset
is a relative phase, and the prepared state would be wrong. So the magnitude
ladder never delegates to `synthesize_one_qubit_matrix` and never takes the short
Euler forms in `one_qubit_synthesis._leaves`. Measured, those forms move the
dropped phase by `pi` when the polar angle reaches `pi`, while the general form
`RZ(lam) PULSE RZ(theta - pi) PULSE RZ(phi - pi)` holds one constant offset for
every angle. `_ry_euler_leaves` is that general form unconditionally.

*Only a true `RZ` can carry the z-rotation of that form.* `Z_ROTATION_OPCODES`
lists `phase` and `u1` beside `rz` because they differ from it by a global phase,
which is a fine trade for a single leaf. It is not a fine trade here: the offset
is `exp(1j * angle / 2)`, and the angle is the branch index, so a `phase` or `u1`
leaf makes the offset branch dependent. Measured, the general `ry` form holds
`-pi/2` for every angle with `rz` and walks from `+pi/2` to `+2.5707963268` as the
angle goes from `0` to `2`. `synthesize_state_preparation` therefore returns None
for them rather than a wrong state.

*The shortest spelling is chosen by the caller's basis, not here.* A rejected
basis is reported as None, the same refusal `synthesize_two_qubit` gives, so a
fail-closed caller can name the gate its target is missing instead of receiving a
replacement that was never verified.

The classical cost is the premise this whole family rests on, and it is inherited
rather than introduced: reading the amplitude vector is `O(2**n)`, so the input is
already exponential in the register width. Preparing a state is cheap only given
its classical description. This module prepares the state exactly, up to the one
unrecorded global phase and the working precision of the amplitudes it is handed;
it makes no performance, capacity, or hardware claim, and it selects no runtime.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.ir import Instruction
from ..core.operator_schema import get_operator_schema
from .one_qubit_synthesis import HALF_PI_PULSE_OPCODES, Z_ROTATION_OPCODES
from .two_qubit_synthesis import (
    SUPERCONTROLLED_ENTANGLERS,
    _named_entangler_matrix,
    _swap_register,
    synthesize_two_qubit,
)

__all__ = ("synthesize_state_preparation",)

#: The only member of `Z_ROTATION_OPCODES` whose dropped phase does not depend on
#: its own angle, which is what a ladder needs; see the module docstring.
_LADDER_Z_ROTATION = "rz"

#: `CX` read with the first qubit on the most significant index bit, which is the
#: order `synthesize_two_qubit` reads a matrix. `two_qubit_synthesis` owns the
#: `cx` reading, `_named_entangler_matrix` hands it back in the low-control
#: convention that module's algebra is derived in, and `_swap_register` is that
#: exchange and its own inverse, so this is its matrix rather than a second copy.
_CONTROL_FLIP = _swap_register(_named_entangler_matrix("cx"))

_HALF_PI = 0.5 * math.pi


def _as_numbers(source: Any) -> list[Any]:
    """Return `source` as a flat list, or raise ValueError.

    Anything exposing `tolist` is converted first, so an array or a tensor can be
    handed in without this package importing the library that produced it. A
    string is iterable but is not a sequence of amplitudes, so it is left to the
    number check below to refuse it by name.
    """
    if hasattr(source, "tolist"):
        source = source.tolist()
    if isinstance(source, (str, bytes)):
        raise ValueError(
            f"amplitudes must be a sequence of numbers, got {type(source).__name__}"
        )
    try:
        entries = list(source)
    except TypeError as error:
        raise ValueError(
            f"amplitudes must be a sequence of numbers, got {type(source).__name__}"
        ) from error
    for index, entry in enumerate(entries):
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            raise ValueError(
                f"amplitudes must be one-dimensional, entry {index} is a sequence"
            )
    return entries


def _normalised(source: Any) -> list[complex]:
    """Return `source` as `2**n` amplitudes of unit euclidean norm.

    Raises ValueError for anything that is not a one-dimensional vector of at
    least two entries, whose length is not a power of two, or whose norm is zero.
    """
    entries = _as_numbers(source)
    length = len(entries)
    if length < 2:
        raise ValueError(f"amplitudes needs at least two entries, got {length}")
    if length & (length - 1):
        raise ValueError(
            f"amplitudes length must be a power of two, got {length}; a register of"
            " n qubits carries 2**n amplitudes"
        )
    values: list[complex] = []
    for index, entry in enumerate(entries):
        try:
            values.append(complex(entry))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"amplitudes entry {index} is not a number: {entry!r}"
            ) from error
    norm = _norm(values)
    if norm == 0.0:
        raise ValueError(
            "amplitudes must not be the zero vector; nothing can be normalised"
        )
    return [value / norm for value in values]


def _norm(values: Sequence[complex]) -> float:
    """Return the euclidean norm of a vector of complex numbers."""
    total = 0.0
    for value in values:
        total = math.hypot(total, abs(value))
    return total


def _register(qubits: Any, n_qubits: int) -> tuple[int, ...]:
    """Return the qubits the register occupies, most significant bit first.

    Raises ValueError when `qubits` does not name `n_qubits` distinct
    non-negative integers. `bool` is rejected even though it is an `int`, because
    a qubit index is not a truth value.
    """
    if qubits is None:
        return tuple(range(n_qubits))
    try:
        named = list(qubits)
    except TypeError as error:
        raise ValueError("qubits must be a sequence of qubit indices") from error
    for qubit in named:
        if isinstance(qubit, bool) or not isinstance(qubit, int) or qubit < 0:
            raise ValueError("qubits must be non-negative integers")
    if len(named) != n_qubits:
        raise ValueError(
            f"qubits must name {n_qubits} qubits for this amplitude vector, got"
            f" {len(named)}"
        )
    if len(set(named)) != len(named):
        raise ValueError("qubits must name distinct qubits")
    return tuple(named)


def _magnitude_angles(
    data: Sequence[complex], n_qubits: int, level: int
) -> list[float]:
    """Return the `ry` angle of every branch of one level of the magnitude pass.

    Level `j` splits the register into `2**j` blocks of `2**(n-j)` amplitudes.
    Inside a block the qubit `j` bit picks the half, and `2 * atan2` of the two
    half-norms is the rotation that gives the halves their relative weight.
    """
    block = 2 ** (n_qubits - level)
    half = block // 2
    return [
        2.0
        * math.atan2(
            _norm(data[start + half : start + block]),
            _norm(data[start : start + half]),
        )
        for start in (branch * block for branch in range(2**level))
    ]


def _phase_angles(data: Sequence[complex], n_qubits: int) -> list[list[float]]:
    """Return the `rz` angle of every branch of every level of the phase pass.

    Each uniformly controlled `rz` at level `j` shifts the phase of a basis state
    by `0.5 * eps_j * gamma`, where `eps_j` is `+1` when the qubit `j` bit is set
    and `-1` otherwise, and `gamma` is that branch's angle. The state whose bits
    are all zero is not exempt -- every ladder contributes its `-gamma/2` term
    there -- so the phases are reachable only up to one additive constant per
    level. Taking the angle of a branch as the difference between the mean phase
    of its two halves makes that constant cancel inside the difference, and one
    additive constant per level is one global phase for the whole preparation.

    The means form a pyramid: `pyramid[m]` holds the mean phase over each block of
    `2**m` amplitudes, so the halves of a level `j` branch are two neighbours in
    `pyramid[n - j - 1]`.
    """
    means = [cmath.phase(value) for value in data]
    pyramid = [means]
    for _ in range(n_qubits):
        means = [
            0.5 * (means[2 * index] + means[2 * index + 1])
            for index in range(len(means) // 2)
        ]
        pyramid.append(means)
    return [
        [coarser[2 * branch + 1] - coarser[2 * branch] for branch in range(2**level)]
        for level, coarser in (
            (level, pyramid[n_qubits - level - 1]) for level in range(n_qubits)
        )
    ]


def _walsh(angles: Sequence[float], n_controls: int) -> list[float]:
    """Return the Gray-code-ordered angles of a uniformly controlled ladder.

    A uniformly controlled rotation applies `R(gamma[b])` on branch `b`, but the
    walk that visits every branch once reversing one control at a time sees the
    branches in Gray order, and a control that is still set when the next branch
    is reached carries the rotation with it. The angle to apply on the `p`-th step
    is therefore the Walsh transform `2**-k * sum_q (-1)**popcount(p & q) *
    gamma[q]` of the branch angles.
    """
    scale = float(2**n_controls)
    return [
        sum(
            (-1.0 if bin(step & branch).count("1") % 2 else 1.0) * angles[branch]
            for branch in range(2**n_controls)
        )
        / scale
        for step in range(2**n_controls)
    ]


def _control_walk(controls: Sequence[int]) -> tuple[list[int], list[int]]:
    """Return the Gray-code branch order and the control flipped at each step.

    `flips[step - 1]` is the control changed to reach the `step`-th branch, and
    `flips[2**k - 1]` is the closing flip that returns the control register to
    where it started, so a ladder of `k` controls spends exactly `2**k` flips and
    the flips multiply to the identity. The Gray code ends at `2**(k-1)`, which
    differs from the start only in the most significant control, so the closing
    flip is always `controls[0]`.
    """
    order = [0]
    flips: list[int] = []
    previous = 0
    for step in range(1, 2 ** len(controls)):
        current = step ^ (step >> 1)
        flips.append(controls[len(controls) - (current ^ previous).bit_length()])
        order.append(current)
        previous = current
    flips.append(controls[0])
    return order, flips


def _pulse_leaf(
    opcode: str, *, target: int, metadata: Mapping[str, Any]
) -> Instruction:
    """The pi/2 pulse as `opcode`, parameterised only when the schema asks."""
    schema = get_operator_schema(opcode)
    parameters = schema.parameters if schema is not None else ()
    if not parameters:
        return Instruction(opcode, (target,), metadata=metadata)
    return Instruction(
        opcode, (target,), params=dict.fromkeys(parameters, _HALF_PI), metadata=metadata
    )


def _ry_euler_leaves(
    theta: float,
    *,
    target: int,
    z_rotation: str,
    pulse_opcode: str,
    metadata: Mapping[str, Any],
) -> list[Instruction]:
    """Return `RY(theta)` on one qubit as a z-rotation and pi/2 pulses.

    The general form of `one_qubit_synthesis` is emitted unconditionally, with
    `(theta, phi, lam) = (theta, 0, 0)`, because its short forms depend on the
    polar angle branch and so change the global phase this module drops. A ladder
    needs one and the same dropped phase in every branch; `rz` delivers it and
    `phase` and `u1` do not, which is why `synthesize_state_preparation` refuses
    them. A zero z-angle is the identity and is dropped.
    """
    leaves: list[Instruction] = []
    for opcode, angle in (
        (z_rotation, 0.0),
        (pulse_opcode, None),
        (z_rotation, theta - math.pi),
        (pulse_opcode, None),
        (z_rotation, -math.pi),
    ):
        if angle is None:
            leaves.append(_pulse_leaf(opcode, target=target, metadata=metadata))
        elif angle != 0.0:
            leaves.append(
                Instruction(
                    z_rotation, (target,), params={"theta": angle}, metadata=metadata
                )
            )
    return leaves


def _z_leaf(
    angle: float, *, target: int, z_rotation: str, metadata: Mapping[str, Any]
) -> list[Instruction]:
    """Return `RZ(angle)` on one qubit, or nothing when it is the identity.

    Unlike a magnitude leaf, dropping this one is exact rather than a change of
    global phase: a phase ladder applies exactly one such leaf per branch, so its
    contribution to that branch is the identity when the angle is zero.
    """
    if angle == 0.0:
        return []
    return [
        Instruction(z_rotation, (target,), params={"theta": angle}, metadata=metadata)
    ]


def _magnitude_ladder(
    angles: Sequence[float],
    *,
    target: int,
    controls: Sequence[int],
    flips: Mapping[tuple[int, int], tuple[Instruction, ...]],
    z_rotation: str,
    pulse_opcode: str,
    metadata: Mapping[str, Any],
) -> list[Instruction]:
    """Return one level of the magnitude pass: a uniformly controlled `ry`.

    A branch whose angle is zero and a ladder whose every branch angle is zero are
    both dropped, because both are the identity times one scalar. The ladder is
    the stronger case and is handled first: the flips multiply to the identity, so
    the whole level leaves only a global phase. A lone zero branch keeps its flips,
    which the walk needs to reach the branches around it, but drops its `RY(0)`
    group, which the general Euler form renders as `-i` times the identity --
    measured on this host at `6.1e-17` of diagonality for the off-diagonal entries,
    the same float64 residue every other pulse group in the circuit carries. A
    single zero branch may not drop its flips the same way, because a flip is not a
    scalar and only a whole walk of them cancels.
    """
    alpha = _walsh(angles, len(controls))
    if not any(alpha):
        return []
    if not controls:
        return _ry_euler_leaves(
            alpha[0],
            target=target,
            z_rotation=z_rotation,
            pulse_opcode=pulse_opcode,
            metadata=metadata,
        )
    order, walk = _control_walk(controls)
    emitted: list[Instruction] = []
    for step, branch in enumerate(order):
        if step:
            emitted.extend(flips[(walk[step - 1], target)])
        if alpha[branch] != 0.0:
            emitted.extend(
                _ry_euler_leaves(
                    alpha[branch],
                    target=target,
                    z_rotation=z_rotation,
                    pulse_opcode=pulse_opcode,
                    metadata=metadata,
                )
            )
    emitted.extend(flips[(walk[-1], target)])
    return emitted


def _phase_ladder(
    angles: Sequence[float],
    *,
    target: int,
    controls: Sequence[int],
    flips: Mapping[tuple[int, int], tuple[Instruction, ...]],
    z_rotation: str,
    metadata: Mapping[str, Any],
) -> list[Instruction]:
    """Return one level of the phase pass: a uniformly controlled `rz`.

    A level whose every branch angle is zero is dropped whole and exactly: each
    branch would apply `RZ(0)`, and the flips multiply to the identity, so the
    level is the identity rather than a global phase.
    """
    alpha = _walsh(angles, len(controls))
    if not any(alpha):
        return []
    if not controls:
        return _z_leaf(
            alpha[0], target=target, z_rotation=z_rotation, metadata=metadata
        )
    order, walk = _control_walk(controls)
    emitted: list[Instruction] = []
    for step, branch in enumerate(order):
        if step:
            emitted.extend(flips[(walk[step - 1], target)])
        emitted.extend(
            _z_leaf(
                alpha[branch], target=target, z_rotation=z_rotation, metadata=metadata
            )
        )
    emitted.extend(flips[(walk[-1], target)])
    return emitted


def _control_flips(
    register: Sequence[int],
    *,
    entangler: str,
    z_rotation: str,
    pulse_opcode: str,
    metadata: Mapping[str, Any],
) -> dict[tuple[int, int], tuple[Instruction, ...]] | None:
    """Return every ladder's `CX(control, target)` as the target's own gates.

    None means the target cannot spell the flip at all, which makes the whole
    preparation unavailable rather than partially written.
    """
    table: dict[tuple[int, int], tuple[Instruction, ...]] = {}
    for index, target in enumerate(register):
        for control in register[:index]:
            leaves = synthesize_two_qubit(
                _CONTROL_FLIP,
                qubits=(control, target),
                entangler=entangler,
                z_rotation=z_rotation,
                pulse_opcode=pulse_opcode,
                metadata=metadata,
            )
            if leaves is None:
                return None
            table[(control, target)] = leaves
    return table


def synthesize_state_preparation(
    amplitudes: Any,
    *,
    qubits: Sequence[int] | None = None,
    z_rotation: str = _LADDER_Z_ROTATION,
    pulse_opcode: str = "sx",
    entangler: str = "cx",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Instruction, ...] | None:
    """Return a circuit carrying `|0...0>` onto `amplitudes`, or None.

    `amplitudes` is a one-dimensional sequence of at least two numbers, or any
    object exposing `tolist`, in the same order `flagquantum.Circuit` reads a
    state: entry `k` is the amplitude of the basis state whose `qubits[0]` bit is
    the most significant. It is normalised, so a vector that is not a unit vector
    is prepared with the direction it names. `qubits` names the qubits the
    register occupies, most significant first, and defaults to `0..n-1`.

    `z_rotation` is the z-rotation the target publishes and must be `rz`; `phase`
    and `u1` are refused because a ladder cannot hold a uniform dropped phase over
    them. `pulse_opcode` is the opcode the target publishes for a pi/2 pulse about
    x, and `entangler` a supercontrolled two-qubit opcode from
    `SUPERCONTROLLED_ENTANGLERS`. `metadata` is copied onto every emitted leaf.

    Returns None when the target's vocabulary cannot carry this preparation: a
    z-rotation a ladder cannot hold a uniform phase over, a pulse opcode outside
    `HALF_PI_PULSE_OPCODES`, an entangler outside `SUPERCONTROLLED_ENTANGLERS`, or
    an entangler `synthesize_two_qubit` cannot spell `CX` with. Raises ValueError
    for an amplitude vector that is not one-dimensional, is shorter than two
    entries, is not a power of two long, or is the zero vector, and for a `qubits`
    that does not name as many distinct non-negative qubits as the vector needs.

    The result is equal to the requested state up to one global phase, which
    FlagQuantum IR cannot record and which depends on the basis as well as on the
    state. An empty tuple, for `|0...0>`, is a correct answer.
    """
    if z_rotation not in Z_ROTATION_OPCODES or z_rotation != _LADDER_Z_ROTATION:
        return None
    if pulse_opcode not in HALF_PI_PULSE_OPCODES:
        return None
    if entangler not in SUPERCONTROLLED_ENTANGLERS:
        return None
    data = _normalised(amplitudes)
    n_qubits = len(data).bit_length() - 1
    register = _register(qubits, n_qubits)
    annotations = {} if metadata is None else metadata
    flips = _control_flips(
        register,
        entangler=entangler,
        z_rotation=z_rotation,
        pulse_opcode=pulse_opcode,
        metadata=annotations,
    )
    if flips is None:
        return None
    phases = _phase_angles(data, n_qubits)
    emitted: list[Instruction] = []
    for level in range(n_qubits):
        emitted.extend(
            _magnitude_ladder(
                _magnitude_angles(data, n_qubits, level),
                target=register[level],
                controls=register[:level],
                flips=flips,
                z_rotation=z_rotation,
                pulse_opcode=pulse_opcode,
                metadata=annotations,
            )
        )
    for level, angles in enumerate(phases):
        emitted.extend(
            _phase_ladder(
                angles,
                target=register[level],
                controls=register[:level],
                flips=flips,
                z_rotation=z_rotation,
                metadata=annotations,
            )
        )
    return tuple(emitted)
