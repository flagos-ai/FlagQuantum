"""Collapse a same-wire run of single-qubit gates into one exact instruction.

`pipeline` already removes identities, cancels self-inverse pairs, and adds up
adjacent rotations that share one opcode. None of those touch a run that *mixes*
opcodes, which is the shape most programs have: routing walks a `h` in between
two `rz` gates, a composed rotation lands as `x rz(0.4) x`, and four `t` gates
spell one `z`. This module folds such a run into the shortest sequence that
reproduces it, which is the reduction Qiskit's `Optimize1qGates` performs on its
`u1`/`u2`/`u3` basis.

Four properties make a fold admissible here.

**The fold is exact, including the global phase.** A run's product is
accumulated as a 2x2 matrix and the replacement is emitted from that matrix, so
the replacement equals the run entry for entry. Qiskit may drop a global phase
because its DAG carries `global_phase`; FlagQuantum IR has no such field, and
`examples/compiler_optimize` checks an optimization against a raw statevector, so
a fold here may not lose one. The emitted vocabulary is `u3`, `phase`, and `rz`:
`u3` is the universal single-qubit opcode of `OPERATOR_SCHEMAS`, and `phase` and
`rz` together carry any determinant that a bare `u3` cannot. See
`_convention_phase` for the two opcodes whose runtime matrix is not already in
the `U3` convention `one_qubit_synthesis` tabulates.

**A run is replaced only when the replacement is strictly shorter.** Equal
length is not a win, so a run of one gate and a run whose product needs as many
gates as it had are both left alone. That also makes the pass idempotent, which
`_optimize_to_fixed_point` relies on: folding a replacement again yields the same
instructions because the emission is a function of the product alone.

**A run that a single z-rotation/pulse basis already spells is left to that
basis.** `native_gate_legalization` lowers `u3` into the target's z-rotation and
pi/2 pulse, which costs up to five gates; a run already written as those two
opcodes would therefore come back longer than it started if it were re-spelled
here. `optimize` is target-independent and cannot see the target's basis, so it
declines the one case where its own vocabulary is the wrong one. The fold
targets the runs no single z-rotation/pulse basis can keep, which is where the
reduction is real.

**No trainable angle is folded.** Composition reads numeric amplitudes, and a
trainable angle has no numeric value to read; the same reason
`one_qubit_synthesis` refuses to select a branch on one. A run that carries any
trainable angle is left to `pipeline.merge_adjacent_rotations`, which adds angles
with the caller's own objects and keeps them in the autograd graph.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from ..core.ir import CircuitIR, Instruction
from ..core.operator_schema import canonical_opcode
from .one_qubit_synthesis import (
    _FIXED_EULER_ANGLES,
    _PARAMETERIZED_EULER_ANGLES,
    _PHASE_EPS,
    HALF_PI_PULSE_OPCODES,
    Z_ROTATION_OPCODES,
    _polar_angle,
)

#: A 2x2 matrix of Python complex numbers, rows first.
Matrix = list[list[complex]]

#: Every opcode `one_qubit_synthesis` tabulates a Euler triple for. This is the
#: declared single-qubit unitary set, and it is the set that can join a fold.
_EULER_ANGLES = frozenset(_FIXED_EULER_ANGLES) | frozenset(_PARAMETERIZED_EULER_ANGLES)

#: The general single-qubit opcode a folded run is emitted as.
_GENERAL_OPCODE = "u3"

#: The opcode that carries a determinant a z-rotation cannot, and the z-rotation
#: that carries the rest. Both are declared single-qubit unitary opcodes.
_PHASE_OPCODE = "phase"
_Z_ROTATION_OPCODE = "rz"

#: A rotation this small is float noise from composing the run rather than a
#: rotation the source asked for, so emitting it would spend a gate on nothing.
_ANGLE_EPS = 1.0e-15

_IDENTITY: Matrix = [[1.0 + 0.0j, 0.0j], [0.0j, 1.0 + 0.0j]]


def _convention_phase(opcode: str, params: Mapping[str, Any]) -> float:
    """The global phase the runtime matrix carries over the `U3` convention.

    `one_qubit_synthesis` tabulates a `U3(theta, phi, lam)` triple for every
    declared single-qubit unitary, and for fifteen of the eighteen opcodes that
    triple reproduces the runtime matrix exactly. Three disagree by a global
    phase, which matters here because this module may not drop one:

    ``rz(theta)`` is `exp(-i theta/2)` on one basis state where the same triple
    under `U3` is `phase(theta)`, and `sx` and `sxdg` carry the quarter-turn phase
    of the standard square-root-of-X matrix. `test_compilation_one_qubit_optimization.py`
    pins all three against the runtime gate table, which is the authority.
    """

    if opcode == "rz":
        return -0.5 * float(params["theta"])
    if opcode == "sx":
        return 0.25 * math.pi
    if opcode == "sxdg":
        return -0.25 * math.pi
    return 0.0


def _u3_matrix(theta: float, phi: float, lam: float) -> Matrix:
    """`U3(theta, phi, lam)` under the convention `one_qubit_synthesis` uses."""

    cosine = math.cos(0.5 * theta)
    sine = math.sin(0.5 * theta)
    return [
        [complex(cosine), -cmath.exp(1j * lam) * sine],
        [cmath.exp(1j * phi) * sine, cmath.exp(1j * (phi + lam)) * cosine],
    ]


def _as_complex_matrix(matrix: Any) -> Matrix | None:
    """Return `matrix` as two rows of two complex numbers, or None."""

    try:
        rows = [[complex(entry) for entry in row] for row in matrix]
    except (TypeError, ValueError):
        return None
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        return None
    return rows


def _scalar_angle(value: Any) -> float | None:
    """`value` as one angle, or None when it is not exactly one angle.

    `one_qubit_synthesis._polar_angle` answers "does this value have a number",
    which a one-element array also does. This asks the narrower question a fold
    needs: is this value *the* angle. Only a true scalar passes; anything
    carrying a shape does not, however few elements it holds.
    """

    if getattr(value, "ndim", 0) != 0:
        return None
    return _polar_angle(value)


def _instruction_matrix(instruction: Instruction) -> Matrix | None:
    """One single-qubit instruction's exact matrix, or None.

    None means the instruction cannot take part in a fold: it is not a declared
    single-qubit unitary, one of its angles is trainable and therefore has no
    number to compose with, or one of its angles is a batch of angles rather than
    one angle.
    """

    if instruction.matrix is not None:
        return _as_complex_matrix(
            getattr(instruction.matrix, "tensor", instruction.matrix)
        )
    # A fold has to read the angles, where the passes beside it only have to add
    # them, so a batched parameter is refused here rather than left to the
    # runtime. Reading the single element of a matrix of angles would turn a
    # program the runtime refuses into one that runs with one row broadcast to
    # every batch entry -- a silent result change, not a simplification.
    if any(_scalar_angle(value) is None for value in instruction.params.values()):
        return None
    opcode = canonical_opcode(instruction.name)
    angles = _FIXED_EULER_ANGLES.get(opcode)
    if angles is None:
        source = _PARAMETERIZED_EULER_ANGLES.get(opcode)
        if source is None:
            return None
        angles = source(instruction)
    numbers = tuple(_scalar_angle(angle) for angle in angles)
    if any(number is None for number in numbers):
        return None
    scale = cmath.exp(1j * _convention_phase(opcode, instruction.params))
    matrix = _u3_matrix(*(float(number) for number in numbers))  # type: ignore[arg-type]
    return [[scale * entry for entry in row] for row in matrix]


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    """The product that applies `right` first."""

    return [
        [
            left[row][0] * right[0][column] + left[row][1] * right[1][column]
            for column in range(2)
        ]
        for row in range(2)
    ]


def _run_product(instructions: list[Instruction]) -> Matrix | None:
    """The exact product of one run, or None when any instruction is unreadable."""

    product = _IDENTITY
    for instruction in instructions:
        matrix = _instruction_matrix(instruction)
        if matrix is None:
            return None
        product = _matmul(matrix, product)
    return product


def _split_anchor(matrix: Matrix) -> tuple[float, Matrix]:
    """The trailing z-rotation and the `U3` remainder of one unitary.

    Every unitary factors as ``RZ(anchor) U3(theta, phi, lam)``: the anchor is
    the phase of the leading entry, which a `U3` leaves real and non-negative.
    An anti-diagonal matrix has no leading entry to read, which is exactly the
    case `U3` covers with `theta = pi`, so the anchor is zero there.
    """

    leading = matrix[0][0]
    if abs(leading) < _PHASE_EPS:
        return 0.0, matrix
    phase = cmath.phase(leading)
    return (
        -2.0 * phase,
        [
            [
                cmath.exp(-1j * phase) * matrix[0][0],
                cmath.exp(-1j * phase) * matrix[0][1],
            ],
            [
                cmath.exp(1j * phase) * matrix[1][0],
                cmath.exp(1j * phase) * matrix[1][1],
            ],
        ],
    )


def _u3_angles(matrix: Matrix) -> tuple[float, float, float]:
    """The `U3` triple of a matrix whose leading entry is real and non-negative.

    The two off-diagonal entries give the polar angle and the two azimuths. A
    diagonal matrix has no off-diagonal entry to read an azimuth from — whatever
    survives there is float noise from composing the run — so it is recognized by
    its polar angle and reads the single remaining azimuth off the trailing
    entry instead. Summing the azimuths is what makes the two branches agree, so
    the threshold is on the polar angle rather than on the noise itself.
    """

    leading = matrix[0][0]
    top_right = matrix[0][1]
    bottom_left = matrix[1][0]
    if abs(bottom_left) < _PHASE_EPS and abs(top_right) < _PHASE_EPS:
        return 0.0, 0.0, cmath.phase(matrix[1][1])
    theta = 2.0 * math.atan2(abs(bottom_left), abs(leading))
    phi = cmath.phase(bottom_left)
    lam = cmath.phase(-top_right) if abs(top_right) >= _PHASE_EPS else -phi
    return theta, phi, lam


def _emit(
    matrix: Matrix, *, wire: int, metadata: Mapping[str, Any]
) -> tuple[Instruction, ...]:
    """The shortest exact sequence over `u3`, `phase`, and `rz` for one matrix."""

    anchor, remainder = _split_anchor(matrix)
    theta, phi, lam = _u3_angles(remainder)
    emitted: list[Instruction] = []
    if theta != 0.0:
        emitted.append(
            Instruction(
                _GENERAL_OPCODE,
                (wire,),
                params={"theta": theta, "phi": phi, "lbd": lam},
                metadata=metadata,
            )
        )
    elif abs(phi + lam) > _ANGLE_EPS:
        emitted.append(
            Instruction(
                _PHASE_OPCODE,
                (wire,),
                params={"theta": phi + lam},
                metadata=metadata,
            )
        )
    if abs(anchor) > _ANGLE_EPS:
        emitted.append(
            Instruction(
                _Z_ROTATION_OPCODE,
                (wire,),
                params={"theta": anchor},
                metadata=metadata,
            )
        )
    return tuple(emitted)


def _is_foldable(instruction: Instruction) -> bool:
    """Whether one instruction is a single-qubit unitary that could join a fold.

    False for anything else — a wider gate, a measurement, a barrier, or an
    opcode no Euler table covers — and such an instruction *ends* the run on each
    wire it touches rather than joining it.
    """

    if len(instruction.wires) != 1:
        return False
    if instruction.matrix is not None:
        return True
    return canonical_opcode(instruction.name) in _EULER_ANGLES


def _single_basis_pair(instructions: list[Instruction]) -> tuple[str, str] | None:
    """The one z-rotation/pulse pair a run is entirely spelled in, or None.

    The pair is what `native_gate_legalization` lowers a `u3` into, so a run that
    already matches one is the case this module declines: re-spelling it here
    would only make the target's own lowering put it back. Over eighty circuits
    spelled in `rz` and `sx`, folding them took the intermediate program from 758
    to 371 gates while the target-legal output grew from 758 to 902 — the fold
    was undone by the lowering, at a 19% cost.
    """

    opcodes = [canonical_opcode(item.name) for item in instructions]
    if any(item.matrix is not None for item in instructions):
        return None
    if any(
        opcode not in Z_ROTATION_OPCODES + HALF_PI_PULSE_OPCODES for opcode in opcodes
    ):
        return None
    z_rotations = {opcode for opcode in opcodes if opcode in Z_ROTATION_OPCODES}
    pulses = {opcode for opcode in opcodes if opcode in HALF_PI_PULSE_OPCODES}
    if not pulses or len(z_rotations) > 1 or len(pulses) > 1:
        return None
    return next(iter(z_rotations), _Z_ROTATION_OPCODE), next(iter(pulses))


def _fold_run(instructions: list[Instruction]) -> tuple[Instruction, ...] | None:
    """The exact replacement of one same-wire run, or None to leave it.

    None covers refusal as well as no gain: a run a single z-rotation/pulse
    vocabulary already spells, and a run whose replacement would not be shorter.
    Every instruction here is foldable and readable, so there is no third case.
    """

    if _single_basis_pair(instructions) is not None:
        return None
    product = _run_product(instructions)
    if product is None:
        return None
    replacement = _emit(
        product,
        wire=instructions[0].wires[0],
        metadata=instructions[0].metadata,
    )
    return replacement if len(replacement) < len(instructions) else None


def collapse_one_qubit_runs(ir: CircuitIR) -> CircuitIR:
    """Collapse each same-wire run of single-qubit gates into one exact sequence.

    A run is maximal: it ends at a gate that shares the wire, and it continues
    across gates on other wires, which commute with it. The replacement is
    written at the run's own positions, so the emitted gates keep their order on
    the wire and land in the same place in program order.

    A run is left untouched when it carries a trainable angle, when a single
    z-rotation/pulse vocabulary already spells it, or when folding it would not
    be strictly shorter. A wider gate, a measurement, or a barrier ends the run
    instead of joining it, so the gates on either side of one still fold.
    """

    output: list[Instruction | None] = []
    open_runs: dict[int, list[int]] = {}
    closed_runs: list[list[int]] = []

    def close(wires: Iterable[int]) -> None:
        for touched in wires:
            run = open_runs.pop(touched, None)
            if run is not None:
                closed_runs.append(run)

    for instruction in ir:
        readable = (
            _is_foldable(instruction) and _instruction_matrix(instruction) is not None
        )
        if not readable:
            close(instruction.wires)
            output.append(instruction)
            continue
        open_runs.setdefault(instruction.wires[0], []).append(len(output))
        output.append(instruction)
    closed_runs.extend(open_runs.values())

    replaced: list[Instruction | None] = list(output)
    for run in closed_runs:
        if len(run) < 2:
            continue
        replacement = _fold_run([output[position] for position in run])  # type: ignore[misc]
        if replacement is None:
            continue
        for offset, instruction in enumerate(replacement):
            replaced[run[offset]] = instruction
        for position in run[len(replacement) :]:
            replaced[position] = None
    return replace(
        ir,
        instructions=tuple(item for item in replaced if item is not None),
    )
