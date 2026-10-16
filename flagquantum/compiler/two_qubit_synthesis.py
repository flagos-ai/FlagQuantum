"""Two-qubit unitary synthesis by KAK decomposition over a published entangler.

`gates.native` publishes the gates a target can execute. A supercontrolled
entangler -- one whose Weyl point `(a, b, c)` has `a == pi/4` and `c == 0` --
together with a z-rotation and a pi/2 x-rotation reaches *every* two-qubit
unitary, and the zero-, one-, two- and three-entangler closed forms are exact for
all of them. This module owns that synthesis:

    `U = e^{i phase} (K1l (x) K1r) Ud(a,b,c) (K2l (x) K2r)`

with the local factors fed to `one_qubit_synthesis`, so the leaves are the same
z-rotation plus pulse form the one-qubit path already emits.

**Which entanglers count.** `SUPERCONTROLLED_ENTANGLERS` lists every declared
arity-2 opcode that reaches a supercontrolled Weyl point, together with the angle
it has to be applied at. `cx`, `cz` and `cy` are that point as they stand; `rzz`,
`ryy` and `rxx` reach it at an angle of pi/2 and sit a quarter-turn away from it
at every other angle. `cphase` is deliberately absent: its only supercontrolled
angle is pi, where it *is* `cz`, so it would add a spelling rather than reach.
Listing `rzz` matters because a trapped-ion or flux-tunable-coupler target
publishes a rotation and no `cx` at all, and refusing it would leave those bases
unsynthesizable.

**Index convention.** Every 4x4 matrix here is read with ``index = 2 * bit(qubit
of qubits[0]) + bit(qubit of qubits[1])``: ``qubits[0]`` is the most significant
index bit, which is how `flagquantum.Circuit` orders its statevector
(``fq.Circuit(2).x(0).state().argmax() == 2``). Qiskit orders the other way
round, with its qubit 0 least significant, so Qiskit's ``q0`` is this module's
``qubits[1]``. The 4x4 matrices themselves need no reordering under that match,
but Qiskit's ``K*l`` factors act on its ``q1`` and its ``K*r`` factors on its
``q0``, which is why the factors handed back for a basis use are emitted on
``qubits[0]`` and ``qubits[1]`` in Qiskit's own order.

**Weyl coordinates are computed in the kron convention, then the register is
swapped.** Qiskit's algorithm decomposes a matrix written as ``(L (x) R) E (M (x)
N)`` where ``E`` is the entangler with its control on the *low* index bit. This
module wants the control on ``qubits[0]``, the high bit, so that a caller asking
for ``qubits=(0, 1)`` gets ``Instruction("cx", (0, 1))`` -- control on qubit 0 --
rather than a reversed entangler. Swapping the two index bits is conjugation by
the qubit-exchange permutation ``P``, which maps ``A (x) B`` to ``B (x) A`` and
maps a low-control entangler to a high-control one. So the input is conjugated
with ``P``, decomposed in Qiskit's convention, and the pairs come out in
Qiskit's order over the caller's own qubits. The two conventions agree on the
4x4 matrix itself; only the factor labels and the entangler's control move.

**Global phase is dropped**, exactly as in `one_qubit_synthesis`: FlagQuantum IR
has no field in which to record it, and no basis of a z-rotation, a pulse, and a
supercontrolled entangler can carry the phase of an arbitrary ``SU(4)``. Expect
``|Tr(source^dagger * synthesized)| / 4 == 1`` and a nonzero phase.

**No specialization is applied.** A target that sits exactly on a symmetric
point such as `SWAP` has a canonical shorter form in Qiskit, which rewrites the
factors so that the remaining single-qubit gates cancel against the entangler.
This module always returns the general factors, which are an exact
decomposition of the same matrix. The gate counts for those inputs are
therefore Qiskit's *pulse_optimize=False* counts, not its specialized ones.

**`pulse_optimize` is absent on purpose.** Qiskit's pulse-optimal chooser only
applies to a ``{rz, sx}`` basis around a CNOT and removes one or two single-qubit
gates between entanglers. Dropping it removes a few hundred lines of
float-threshold branching. A gate-count comparison against Qiskit has to pass
``pulse_optimize=False`` on the Qiskit side.

Every number is deterministic: the one randomized step Qiskit has -- the
diagonalization of ``M2`` -- uses the same fixed first attempt here and a
documented golden-angle sweep after it, so no RNG and no LAPACK are involved and
the result cannot depend on the installed BLAS.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..core.ir import Instruction
from .one_qubit_synthesis import (
    HALF_PI_PULSE_OPCODES,
    Z_ROTATION_OPCODES,
    _as_complex_matrix,
    synthesize_one_qubit_matrix,
)

__all__ = ("SUPERCONTROLLED_ENTANGLERS", "synthesize_two_qubit")

Matrix = list[list[complex]]
RealMatrix = list[list[float]]

_PI = math.pi
_PI2 = 0.5 * _PI
_PI4 = 0.25 * _PI
_PI32 = 3.0 * _PI2
_TWO_PI = 2.0 * _PI
_IM = 1j
_FRAC_1_SQRT_2 = 1.0 / math.sqrt(2.0)

#: The entangler spellings this module can synthesize over, in preference order,
#: mapped to the angle each one has to be applied at. A value of None means the
#: opcode carries no parameter. An entangler is usable when its Weyl point has
#: ``a == pi/4`` and ``c == 0``:
#:
#: * ``cx``, ``cz`` and ``cy`` are that point as they stand;
#: * ``rzz``, ``ryy`` and ``rxx`` reach it at an angle of pi/2, and are a
#:   quarter-turn away from it at every other angle.
#:
#: `cphase` is absent on purpose: its only supercontrolled angle is pi, where it
#: *is* `cz`, so listing it would add a spelling rather than reach. An entangler
#: that a target publishes under a name is a basis-translation question, not a
#: Weyl question.
#:
#: The parameter-free spellings come first so that a target publishing one of
#: them never gains a parameter it does not need, and the rotation family follows
#: in decreasing order of how hardware publishes it: `rzz` is the native
#: interaction of both flux-tunable transmons and trapped ions, `rxx` the
#: published form on several ion and superconducting targets, and `ryy` the
#: remainder. The order is fixed rather than taken from a set, so the same target
#: always yields the same program.
SUPERCONTROLLED_ENTANGLERS: Mapping[str, float | None] = MappingProxyType(
    {
        "cx": None,
        "cz": None,
        "cy": None,
        "rzz": _PI2,
        "rxx": _PI2,
        "ryy": _PI2,
    }
)

#: A matrix is accepted as a unitary within this absolute tolerance.
_UNITARY_ATOL = 1.0e-9

#: ``P diag(D) P^T`` must reproduce ``M2`` to this accuracy before ``P`` is
#: accepted, matching the tolerance Qiskit uses.
_M2_ATOL = 1.0e-13
_M2_TRIALS = 100

#: Qiskit's fixed first attempt at mixing ``Re(M2)`` with ``Im(M2)``. Only the
#: ratio matters, so it is kept verbatim; agreement with Qiskit on which valid
#: factors come out depends on it.
_M2_FIRST_TRIAL = (1.2602066112249388, 0.22317849046722027)

#: Golden-angle step for the retry sweep, in radians.
_M2_RETRY_STEP = 2.399963229728653

#: Swaps the two index bits, i.e. conjugates a matrix by the qubit exchange.
_SWAP_INDEX = (0, 2, 1, 3)


# --------------------------------------------------------------- linear algebra


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    size = len(right)
    width = len(right[0])
    return [
        [
            sum((left[row][k] * right[k][column] for k in range(size)), 0j)
            for column in range(width)
        ]
        for row in range(len(left))
    ]


def _transpose(matrix: Matrix) -> Matrix:
    return [list(row) for row in zip(*matrix, strict=True)]


def _dagger(matrix: Matrix) -> Matrix:
    return [[item.conjugate() for item in row] for row in _transpose(matrix)]


def _determinant(matrix: Matrix) -> complex:
    size = len(matrix)
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    total = 0j
    for column in range(size):
        minor = [
            [matrix[row][item] for item in range(size) if item != column]
            for row in range(1, size)
        ]
        total += (-1) ** column * matrix[0][column] * _determinant(minor)
    return total


def _kron(left: Matrix, right: Matrix) -> Matrix:
    rows_left, columns_left = len(left), len(left[0])
    rows_right, columns_right = len(right), len(right[0])
    out: Matrix = [
        [0j] * (columns_left * columns_right) for _ in range(rows_left * rows_right)
    ]
    for row in range(rows_left):
        for column in range(columns_left):
            for inner_row in range(rows_right):
                for inner_column in range(columns_right):
                    out[row * rows_right + inner_row][
                        column * columns_right + inner_column
                    ] = (left[row][column] * right[inner_row][inner_column])
    return out


def _swap_register(matrix: Matrix) -> Matrix:
    return [
        [matrix[_SWAP_INDEX[row]][_SWAP_INDEX[column]] for column in range(4)]
        for row in range(4)
    ]


def _named_entangler_matrix(opcode: str) -> Matrix:
    """The basis matrix of ``opcode`` at its supercontrolled angle.

    Read with ``qubits[0]`` on the most significant index bit, then exchanged into
    the low-control convention the algebra below is derived in -- the same
    exchange `_weyl_decomposition` applies to the target. The parameter-free
    opcodes are written out because their entries are small exact integers or
    ``i``; the rotation family is built from its closed form so that no entry is
    a rounded decimal.

    The compiler layer may not import the runtime's gate matrix table, so these
    are the only gate matrices this module owns, and they are pinned entry-by-
    entry against that table by
    `tests/unit/test_compilation_two_qubit_synthesis.py`.
    """

    angle = SUPERCONTROLLED_ENTANGLERS[opcode]
    if opcode == "cx":
        native: Matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    elif opcode == "cz":
        native = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ]
    elif opcode == "cy":
        native = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -_IM],
            [0.0, 0.0, _IM, 0.0],
        ]
    elif opcode in ("rxx", "ryy", "rzz"):
        # A rotation entangler with no angle in the table would be a table edit
        # that silently changed which gate this module decomposes over.
        if angle is None:  # pragma: no cover - guarded by the mapping above
            raise AssertionError(f"{opcode!r} needs an angle to be a basis gate")
        native = _rotation_entangler_matrix(opcode, angle)
    else:  # pragma: no cover - guarded by the mapping above
        raise ValueError(f"{opcode!r} is not a supercontrolled entangler")
    return _swap_register(native)


def _rotation_entangler_matrix(opcode: str, angle: float) -> Matrix:
    """``exp(-i * angle/2 * P)`` for ``P`` one of ``XX``, ``YY``, ``ZZ``.

    `rzz` is diagonal. `rxx` and `ryy` differ only in the relative sign of the
    two off-diagonal blocks, because `YY` is `XX` with the second qubit
    conjugated by `Z`.
    """

    cosine = math.cos(0.5 * angle)
    sine = math.sin(0.5 * angle)
    if opcode == "rzz":
        diagonal = (cosine - 1j * sine, cosine + 1j * sine)
        return [
            [diagonal[0], 0j, 0j, 0j],
            [0j, diagonal[1], 0j, 0j],
            [0j, 0j, diagonal[1], 0j],
            [0j, 0j, 0j, diagonal[0]],
        ]
    # `rxx` is `exp(-i*angle/2 * XX)` and `ryy` is `exp(-i*angle/2 * YY)`. The two
    # differ only in the relative sign of their off-diagonal blocks, because `YY`
    # is `XX` with the second qubit conjugated by `Z`.
    off_diagonal = 1j * sine if opcode == "ryy" else -1j * sine
    second_block = -off_diagonal if opcode == "ryy" else off_diagonal
    return [
        [cosine, 0j, 0j, off_diagonal],
        [0j, cosine, second_block, 0j],
        [0j, second_block, cosine, 0j],
        [off_diagonal, 0j, 0j, cosine],
    ]


def _eigenvalues(matrix: Matrix, vectors: RealMatrix) -> list[complex]:
    return [
        sum(
            vectors[row][column] * matrix[row][item] * vectors[item][column]
            for row in range(4)
            for item in range(4)
        )
        for column in range(4)
    ]


def _reconstruct(vectors: RealMatrix, diagonal: Sequence[complex]) -> Matrix:
    return [
        [
            sum(
                vectors[row][column] * diagonal[column] * vectors[item][column]
                for column in range(4)
            )
            for item in range(4)
        ]
        for row in range(4)
    ]


def _jacobi_eigh(matrix: RealMatrix) -> tuple[list[float], RealMatrix]:
    """Eigen-decomposition of a real symmetric 4x4 matrix, ascending.

    Cyclic Jacobi rotations: deterministic, no LAPACK, so the result does not
    depend on which BLAS is installed. Returns the eigenvalues and the
    eigenvectors *by column*, so ``vectors[row][column]`` is one entry.
    """

    size = len(matrix)
    work = [list(row) for row in matrix]
    vectors = [
        [1.0 if row == column else 0.0 for column in range(size)] for row in range(size)
    ]
    for _ in range(60):
        off_diagonal = sum(
            work[row][column] ** 2
            for row in range(size)
            for column in range(row + 1, size)
        )
        if off_diagonal <= 1e-16:
            break
        for row in range(size - 1):
            for column in range(row + 1, size):
                entry = work[row][column]
                if abs(entry) < 1e-300:
                    continue
                theta = (work[column][column] - work[row][row]) / (2.0 * entry)
                tangent = math.copysign(1.0, theta) / (
                    abs(theta) + math.sqrt(theta * theta + 1.0)
                )
                cosine = 1.0 / math.sqrt(tangent * tangent + 1.0)
                sine = tangent * cosine
                for item in range(size):
                    if item in (row, column):
                        continue
                    left, right = work[item][row], work[item][column]
                    work[item][row] = cosine * left - sine * right
                    work[item][column] = sine * left + cosine * right
                for item in range(size):
                    if item in (row, column):
                        continue
                    left, right = work[row][item], work[column][item]
                    work[row][item] = cosine * left - sine * right
                    work[column][item] = sine * left + cosine * right
                diagonal_row, diagonal_column = work[row][row], work[column][column]
                work[row][row] = diagonal_row - tangent * entry
                work[column][column] = diagonal_column + tangent * entry
                work[row][column] = work[column][row] = 0.0
                for item in range(size):
                    left, right = vectors[item][row], vectors[item][column]
                    vectors[item][row] = cosine * left - sine * right
                    vectors[item][column] = sine * left + cosine * right
    order = sorted(range(size), key=lambda index: work[index][index])
    values = [work[index][index] for index in order]
    return values, [[vectors[row][index] for index in order] for row in range(size)]


def _diagonalize_m2(m2: Matrix) -> tuple[Matrix, list[complex]]:
    """Return ``P`` in SO(4) and the diagonal ``D`` with ``M2 == P D P^T``.

    ``M2`` is complex symmetric, so it is diagonalized through a *real* symmetric
    linear combination ``rand_a * Re(M2) + rand_b * Im(M2)``: the comments in
    Qiskit's source explain that ``A`` and ``B`` commute and are therefore
    simultaneously diagonalizable. Only the ratio ``rand_a / rand_b`` matters,
    which is why the retry stream sweeps that ratio. Ties and near-degeneracies
    are handled by testing the product rather than the eigensolver's word for it.
    """

    for attempt in range(_M2_TRIALS):
        if attempt == 0:
            rand_a, rand_b = _M2_FIRST_TRIAL
        else:
            angle = attempt * _M2_RETRY_STEP
            rand_a, rand_b = math.cos(angle), math.sin(angle)
        mixed = [
            [
                rand_a * m2[row][column].real + rand_b * m2[row][column].imag
                for column in range(4)
            ]
            for row in range(4)
        ]
        _, vectors = _jacobi_eigh(mixed)
        diagonal = _eigenvalues(m2, vectors)
        rebuilt = _reconstruct(vectors, diagonal)
        if all(
            abs(rebuilt[row][column] - m2[row][column]) <= _M2_ATOL
            for row in range(4)
            for column in range(4)
        ):
            return [[complex(item) for item in row] for row in vectors], diagonal
    raise ValueError(
        "the two-qubit Weyl decomposition could not diagonalize M2 in "
        f"{_M2_TRIALS} attempts"
    )


# ------------------------------------------------------------------ magic basis

#: The magic basis, normalized so it is itself unitary.
_BASIS: Matrix = [
    [_FRAC_1_SQRT_2, _IM * _FRAC_1_SQRT_2, 0j, 0j],
    [0j, 0j, _IM * _FRAC_1_SQRT_2, _FRAC_1_SQRT_2],
    [0j, 0j, _IM * _FRAC_1_SQRT_2, -_FRAC_1_SQRT_2],
    [_FRAC_1_SQRT_2, -_IM * _FRAC_1_SQRT_2, 0j, 0j],
]
_BASIS_DAGGER: Matrix = _dagger(_BASIS)
_I_PAULI_X: Matrix = [[0j, _IM], [_IM, 0j]]
_I_PAULI_Y: Matrix = [[0j, 1.0], [-1.0, 0j]]
_I_PAULI_Z: Matrix = [[_IM, 0j], [0j, -_IM]]


def _into_magic_basis(matrix: Matrix) -> Matrix:
    return _matmul(_matmul(_BASIS, matrix), _BASIS_DAGGER)


def _out_of_magic_basis(matrix: Matrix) -> Matrix:
    return _matmul(_matmul(_BASIS_DAGGER, matrix), _BASIS)


def _split_product_gate(unitary: Matrix) -> tuple[Matrix, Matrix, float]:
    """Return ``(left, right, phase)`` with ``unitary == e^{i phase} left (x) right``.

    The 2x2 blocks of a product gate are proportional to its two factors, so the
    left factor is read off the even rows and columns after the right factor has
    been divided out. A block that is too close to singular cannot be scaled
    safely and is refused rather than silently rescaled.
    """

    right = [row[:2] for row in unitary[:2]]
    determinant = _determinant(right)
    if abs(determinant) < 0.1:
        right = [row[:2] for row in unitary[2:]]
        determinant = _determinant(right)
    if abs(determinant) < 0.1:
        raise ValueError(
            "the two-qubit decomposition produced a product gate whose leading "
            "block is singular and cannot be split"
        )
    scale = cmath.sqrt(determinant)
    right = [[item / scale for item in row] for row in right]
    identity: Matrix = [[1.0 + 0.0j, 0.0j], [0.0j, 1.0 + 0.0j]]
    residual = _matmul(unitary, _kron(identity, _dagger(right)))
    left = [[residual[row][column] for column in (0, 2)] for row in (0, 2)]
    left_determinant = _determinant(left)
    if abs(left_determinant) < 0.9:
        raise ValueError(
            "the two-qubit decomposition produced a product gate whose trailing "
            "block is singular and cannot be split"
        )
    left = [[item / cmath.sqrt(left_determinant) for item in row] for row in left]
    return left, right, cmath.phase(left_determinant) / 2.0


# --------------------------------------------------------------- weyl chamber


@dataclass(frozen=True)
class _Weyl:
    """One Weyl (KAK) decomposition ``U = e^{i phase} K1 Ud(a,b,c) K2``."""

    a: float
    b: float
    c: float
    global_phase: float
    k1l: Matrix
    k1r: Matrix
    k2l: Matrix
    k2r: Matrix


def _weyl_decomposition(unitary: Matrix) -> _Weyl:
    """Return the Weyl decomposition of one 4x4 unitary in the low-control basis."""

    det_u = _determinant(unitary)
    if abs(det_u) < 1.0e-10:
        raise ValueError("the two-qubit unitary is singular")
    scaled = [[item * det_u**-0.25 for item in row] for row in unitary]
    global_phase = cmath.phase(det_u) / 4.0

    in_magic = _out_of_magic_basis(scaled)
    m2 = _matmul(_transpose(in_magic), in_magic)
    p, diagonal = _diagonalize_m2(m2)

    angles = [-cmath.phase(item) / 2.0 for item in diagonal]
    angles[3] = -angles[0] - angles[1] - angles[2]
    coordinates = [((angles[index] + angles[3]) / 2.0) % _TWO_PI for index in range(3)]

    # Move the target into the canonical Weyl chamber: the two largest coordinates
    # are the ones the closed forms need first.
    folded = [min(item % _PI2, _PI2 - (item % _PI2)) for item in coordinates]
    order = sorted(range(3), key=lambda index: folded[index])
    order = [order[1], order[2], order[0]]
    coordinates = [coordinates[order[index]] for index in range(3)]
    angles = [angles[order[0]], angles[order[1]], angles[order[2]], angles[3]]
    p = [
        [p[row][order[column]] if column < 3 else p[row][3] for column in range(4)]
        for row in range(4)
    ]
    if _determinant(p).real < 0.0:
        for row in range(4):
            p[row][3] = -p[row][3]

    phase_matrix: Matrix = [
        [
            cmath.exp(_IM * angles[index]) if index == column else 0j
            for column in range(4)
        ]
        for index in range(4)
    ]
    k1 = _into_magic_basis(_matmul(_matmul(in_magic, p), phase_matrix))
    k2 = _into_magic_basis(_transpose(p))
    k1l, k1r, phase_l = _split_product_gate(k1)
    k2l, k2r, phase_r = _split_product_gate(k2)
    global_phase += phase_l + phase_r

    # Fold the coordinates that fell outside the chamber back in, carrying the
    # factor corrections that keep the identity exact.
    k1l, k1r, k2r, global_phase = _fold_out_of_chamber(
        coordinates, k1l, k1r, k2r, global_phase
    )
    a, b, c = coordinates[1], coordinates[0], coordinates[2]
    return _Weyl(
        a=a,
        b=b,
        c=c,
        global_phase=global_phase,
        k1l=k1l,
        k1r=k1r,
        k2l=k2l,
        k2r=k2r,
    )


def _fold_out_of_chamber(
    coordinates: list[float],
    k1l: Matrix,
    k1r: Matrix,
    k2r: Matrix,
    global_phase: float,
) -> tuple[Matrix, Matrix, Matrix, float]:
    """Fold ``(cs0, cs1, cs2)`` into ``[0, pi/4] x [0, pi/4] x [0, pi/4]``.

    Two of the flips are reflections inside a chamber rather than moves between
    chambers, and they only balance the phase the others add when exactly one of
    them fired, which is what `conjs` counts.
    """
    conjs = 0

    if coordinates[0] > _PI2:
        coordinates[0] -= _PI32
        k1l = _matmul(k1l, _I_PAULI_Y)
        k1r = _matmul(k1r, _I_PAULI_Y)
        global_phase += _PI2
    if coordinates[1] > _PI2:
        coordinates[1] -= _PI32
        k1l = _matmul(k1l, _I_PAULI_X)
        k1r = _matmul(k1r, _I_PAULI_X)
        global_phase += _PI2
    if coordinates[0] > _PI4:
        coordinates[0] = _PI2 - coordinates[0]
        k1l = _matmul(k1l, _I_PAULI_Y)
        k2r = _matmul(_I_PAULI_Y, k2r)
        conjs += 1
        global_phase -= _PI2
    if coordinates[1] > _PI4:
        coordinates[1] = _PI2 - coordinates[1]
        k1l = _matmul(k1l, _I_PAULI_X)
        k2r = _matmul(_I_PAULI_X, k2r)
        conjs += 1
        global_phase += _PI2
        if conjs == 1:
            global_phase -= _PI
    if coordinates[2] > _PI2:
        coordinates[2] -= _PI32
        k1l = _matmul(k1l, _I_PAULI_Z)
        k1r = _matmul(k1r, _I_PAULI_Z)
        global_phase += _PI2
        if conjs == 1:
            global_phase -= _PI
    if conjs == 1:
        coordinates[2] = _PI2 - coordinates[2]
        k1l = _matmul(k1l, _I_PAULI_Z)
        k2r = _matmul(_I_PAULI_Z, k2r)
        global_phase += _PI2
    if coordinates[2] > _PI4:
        coordinates[2] -= _PI2
        k1l = _matmul(k1l, _I_PAULI_Z)
        k1r = _matmul(k1r, _I_PAULI_Z)
        global_phase -= _PI2
    return k1l, k1r, k2r, global_phase


# ------------------------------------------------------------- basis decomposer


def _z_rotation_matrix(theta: float) -> Matrix:
    return [
        [cmath.exp(-0.5j * theta), 0j],
        [0j, cmath.exp(0.5j * theta)],
    ]


class _BasisDecomposer:
    """The zero- to three-entangler closed forms for one supercontrolled basis.

    This is Qiskit's `TwoQubitBasisDecomposer` with `pulse_optimize=False` and
    `euler_basis="ZSX"`, and with the specialized rewrites left out: the
    classification of a target is not used, only its general Weyl angles.
    """

    def __init__(self, entangler: str) -> None:
        if entangler not in SUPERCONTROLLED_ENTANGLERS:
            raise ValueError(f"{entangler!r} is not a supercontrolled entangler")
        self.entangler = entangler
        self.angle = SUPERCONTROLLED_ENTANGLERS[entangler]
        self.basis = _weyl_decomposition(_named_entangler_matrix(entangler))
        if not (
            math.isclose(self.basis.a, _PI4, rel_tol=1.0e-9)
            and math.isclose(self.basis.c, 0.0, rel_tol=1.0e-9, abs_tol=1.0e-12)
        ):
            raise ValueError(
                f"entangler {entangler!r} is not supercontrolled: its Weyl point "
                f"is ({self.basis.a!r}, {self.basis.b!r}, {self.basis.c!r})"
            )

        b = self.basis.b
        exp_neg_ib = cmath.exp(-1j * b)
        exp_pos_ib = cmath.exp(1j * b)
        exp_neg_2ib = cmath.exp(-2j * b)
        exp_pos_2ib = cmath.exp(2j * b)
        cos_2b = math.cos(2.0 * b)
        sin_2b = math.sin(2.0 * b)

        k11l: Matrix = [
            [-_IM * exp_neg_ib, exp_neg_ib],
            [-_IM * exp_pos_ib, -exp_pos_ib],
        ]
        k11l = [[item / (1.0 + _IM) for item in row] for row in k11l]
        k11r: Matrix = [
            [_IM * exp_neg_ib, -exp_neg_ib],
            [exp_pos_ib, -_IM * exp_pos_ib],
        ]
        k11r = [[item * _FRAC_1_SQRT_2 for item in row] for row in k11r]
        k12l: Matrix = [[_IM, _IM], [-1.0, 1.0]]
        k12l = [[item / (1.0 + _IM) for item in row] for row in k12l]
        k12r: Matrix = [[_IM, 1.0], [-1.0, -_IM]]
        k12r = [[item * _FRAC_1_SQRT_2 for item in row] for row in k12r]
        k32l_k21l: Matrix = [
            [1.0 + _IM * cos_2b, _IM * sin_2b],
            [_IM * sin_2b, 1.0 - _IM * cos_2b],
        ]
        k32l_k21l = [[item * _FRAC_1_SQRT_2 for item in row] for row in k32l_k21l]
        k21r: Matrix = [
            [-_IM * exp_neg_2ib, exp_neg_2ib],
            [_IM * exp_pos_2ib, exp_pos_2ib],
        ]
        k21r = [[item / (1.0 - _IM) for item in row] for row in k21r]
        k22l: Matrix = [[1.0, -1.0], [1.0, 1.0]]
        k22l = [[item * _FRAC_1_SQRT_2 for item in row] for row in k22l]
        k22r: Matrix = [[0.0, 1.0], [-1.0, 0.0]]
        k31l: Matrix = [[exp_neg_ib, exp_neg_ib], [-exp_pos_ib, exp_pos_ib]]
        k31l = [[item * _FRAC_1_SQRT_2 for item in row] for row in k31l]
        k31r: Matrix = [
            [_IM * exp_pos_ib, 0j],
            [0j, -_IM * exp_neg_ib],
        ]
        k32r: Matrix = [
            [exp_pos_ib, -exp_neg_ib],
            [-_IM * exp_pos_ib, -_IM * exp_neg_ib],
        ]
        k32r = [[item / (1.0 - _IM) for item in row] for row in k32r]

        k1ld = _dagger(self.basis.k1l)
        k1rd = _dagger(self.basis.k1r)
        k2ld = _dagger(self.basis.k2l)
        k2rd = _dagger(self.basis.k2r)

        self.u0l = _matmul(k31l, k1ld)
        self.u0r = _matmul(k31r, k1rd)
        self.u1l = _matmul(_matmul(k2ld, k32l_k21l), k1ld)
        self.u1ra = _matmul(k2rd, k32r)
        self.u1rb = _matmul(k21r, k1rd)
        self.u2la = _matmul(k2ld, k22l)
        self.u2lb = _matmul(k11l, k1ld)
        self.u2ra = _matmul(k2rd, k22r)
        self.u2rb = _matmul(k11r, k1rd)
        self.u3l = _matmul(k2ld, k12l)
        self.u3r = _matmul(k2rd, k12r)

        self.q0l = _matmul(_dagger(k12l), k1ld)
        self.q0r = _matmul(_matmul(_dagger(k12r), _I_PAULI_Z), k1rd)
        self.q1la = _matmul(k2ld, _dagger(k11l))
        self.q1lb = _matmul(k11l, k1ld)
        self.q1ra = _matmul(_matmul(k2rd, _I_PAULI_Z), _dagger(k11r))
        self.q1rb = _matmul(k11r, k1rd)
        self.q2l = self.u3l
        self.q2r = self.u3r

    def traces(self, target: _Weyl) -> tuple[complex, complex, complex, complex]:
        """The expected ``Tr(source^dagger * synthesized)`` per entangler count."""

        ta, tb, tc = target.a, target.b, target.c
        basis_b = self.basis.b
        return (
            4
            * complex(
                math.cos(ta) * math.cos(tb) * math.cos(tc),
                math.sin(ta) * math.sin(tb) * math.sin(tc),
            ),
            4
            * complex(
                math.cos(_PI4 - ta) * math.cos(basis_b - tb) * math.cos(tc),
                math.sin(_PI4 - ta) * math.sin(basis_b - tb) * math.sin(tc),
            ),
            4 * complex(math.cos(tc), 0.0),
            4 + 0j,
        )

    def entangler_count(self, target: _Weyl) -> int:
        """The fewest entanglers whose expected fidelity is highest.

        A FlagQuantum native gate set is exact, so the expected fidelity of a
        count is `(4 + |trace|^2) / 20` with no per-application weighting, and
        the lowest count wins a tie. That is Qiskit's `np.argmax` over its
        default `basis_fidelity=1.0`, which returns the first maximum.
        """

        expected = [(4.0 + abs(trace) ** 2) / 20.0 for trace in self.traces(target)]
        best = 0
        for count in range(1, 4):
            if expected[count] > expected[best]:
                best = count
        return best

    def decomposition(self, target: _Weyl, count: int) -> tuple[Matrix, ...]:
        """The 2x2 factors for `count` entanglers, in application order.

        Pairs `(2i, 2i + 1)` sit between entangler `i - 1` and entangler `i`, and
        the trailing pair `(2 * count, 2 * count + 1)` is applied last.
        """

        if count == 0:
            return (
                _matmul(target.k1r, target.k2r),
                _matmul(target.k1l, target.k2l),
            )
        if count == 1:
            return (
                _matmul(_dagger(self.basis.k2r), target.k2r),
                _matmul(_dagger(self.basis.k2l), target.k2l),
                _matmul(target.k1r, _dagger(self.basis.k1r)),
                _matmul(target.k1l, _dagger(self.basis.k1l)),
            )
        if count == 2:
            return (
                _matmul(self.q2r, target.k2r),
                _matmul(self.q2l, target.k2l),
                _matmul(
                    _matmul(self.q1ra, _z_rotation_matrix(2.0 * target.b)), self.q1rb
                ),
                _matmul(
                    _matmul(self.q1la, _z_rotation_matrix(-2.0 * target.a)), self.q1lb
                ),
                _matmul(target.k1r, self.q0r),
                _matmul(target.k1l, self.q0l),
            )
        if count == 3:
            return (
                _matmul(self.u3r, target.k2r),
                _matmul(self.u3l, target.k2l),
                _matmul(
                    _matmul(self.u2ra, _z_rotation_matrix(2.0 * target.b)), self.u2rb
                ),
                _matmul(
                    _matmul(self.u2la, _z_rotation_matrix(-2.0 * target.a)), self.u2lb
                ),
                _matmul(
                    _matmul(self.u1ra, _z_rotation_matrix(-2.0 * target.c)), self.u1rb
                ),
                self.u1l,
                _matmul(target.k1r, self.u0r),
                _matmul(target.k1l, self.u0l),
            )
        raise ValueError(f"{count} is not an entangler count this module can emit")


# ------------------------------------------------------------------- public API


def _is_unitary(matrix: Matrix) -> bool:
    product = _matmul(_dagger(matrix), matrix)
    return all(
        abs(product[row][column] - (1.0 if row == column else 0.0)) <= _UNITARY_ATOL
        for row in range(len(matrix))
        for column in range(len(matrix))
    )


def synthesize_two_qubit(
    matrix: Any,
    *,
    qubits: Sequence[int],
    entangler: str,
    z_rotation: str,
    pulse_opcode: str = "sx",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Instruction, ...] | None:
    """Return `matrix` as entanglers plus one-qubit leaves, or None.

    `matrix` is a 4x4 unitary read with `qubits[0]` on the most significant index
    bit, in the same order `flagquantum.Circuit` uses. `entangler` is a
    supercontrolled two-qubit opcode from `SUPERCONTROLLED_ENTANGLERS`,
    `z_rotation` a z-rotation opcode the target publishes, and `pulse_opcode` the
    opcode it publishes for a pi/2 rotation about x. `metadata` is copied onto
    every emitted leaf, so a caller that holds the source instruction can keep
    its annotations.

    Returns None when this module does not apply: an entangler or z-rotation the
    caller did not publish, or a pulse opcode `one_qubit_synthesis` cannot emit.
    Raises ValueError when the input itself is not a two-qubit unitary.

    The result is equal to `matrix` up to one global phase, which FlagQuantum IR
    cannot record. An empty tuple, for the two-qubit identity, is a correct answer.
    A matrix already in the entangler's own class, such as `cx` read with
    `qubits=(0, 1)`, comes back as the entangler alone.
    """

    if entangler not in SUPERCONTROLLED_ENTANGLERS:
        return None
    qubit_list = list(qubits)
    if len(qubit_list) != 2:
        raise ValueError("qubits must name exactly two qubits")
    if any(
        isinstance(qubit, bool) or not isinstance(qubit, int) or qubit < 0
        for qubit in qubit_list
    ):
        raise ValueError("qubits must be non-negative integers")
    if qubit_list[0] == qubit_list[1]:
        raise ValueError("qubits must name two distinct qubits")
    target_matrix = _as_complex_matrix(matrix, 4, what="matrix")
    if not _is_unitary(target_matrix):
        raise ValueError("matrix must be unitary")
    if z_rotation not in Z_ROTATION_OPCODES:
        return None
    if pulse_opcode not in HALF_PI_PULSE_OPCODES:
        return None

    # Decompose in the low-control kron convention, then read the result back on
    # the caller's own qubits; see the module docstring.
    target = _weyl_decomposition(_swap_register(target_matrix))
    decomposer = _BasisDecomposer(entangler)
    count = decomposer.entangler_count(target)
    factors = decomposer.decomposition(target, count)

    left_qubit, right_qubit = qubit_list
    annotations: Mapping[str, Any] = {} if metadata is None else metadata
    entangler_params: Mapping[str, Any] = (
        {} if decomposer.angle is None else {"theta": decomposer.angle}
    )
    leaves: list[Instruction] = []
    for index in range(count):
        leaves.extend(
            _local_leaves(
                factors[2 * index],
                left_qubit,
                z_rotation=z_rotation,
                pulse_opcode=pulse_opcode,
                metadata=annotations,
            )
        )
        leaves.extend(
            _local_leaves(
                factors[2 * index + 1],
                right_qubit,
                z_rotation=z_rotation,
                pulse_opcode=pulse_opcode,
                metadata=annotations,
            )
        )
        leaves.append(
            Instruction(
                entangler,
                (left_qubit, right_qubit),
                params=entangler_params,
                metadata=annotations,
            )
        )
    leaves.extend(
        _local_leaves(
            factors[2 * count],
            left_qubit,
            z_rotation=z_rotation,
            pulse_opcode=pulse_opcode,
            metadata=annotations,
        )
    )
    leaves.extend(
        _local_leaves(
            factors[2 * count + 1],
            right_qubit,
            z_rotation=z_rotation,
            pulse_opcode=pulse_opcode,
            metadata=annotations,
        )
    )
    return tuple(leaves)


def _local_leaves(
    factor: Matrix,
    qubit: int,
    *,
    z_rotation: str,
    pulse_opcode: str,
    metadata: Mapping[str, Any],
) -> tuple[Instruction, ...]:
    """The leaves of one local factor, or an empty tuple for its identity."""

    return (
        synthesize_one_qubit_matrix(
            factor,
            qubit=qubit,
            z_rotation=z_rotation,
            pulse_opcode=pulse_opcode,
            metadata=metadata,
        )
        or ()
    )
