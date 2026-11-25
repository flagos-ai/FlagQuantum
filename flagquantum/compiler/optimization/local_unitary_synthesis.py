"""Lower explicit one/two-qubit local unitaries without discarding scalar phase.

This composes existing Euler/KAK factors and exact scalar/control owners. It
never contracts a register algorithm or imports a numerical execution backend.
Angles remain continuous native rotations, not approximate Clifford+T words.
"""

from collections.abc import Sequence
from math import atan2

import numpy as np
from numpy.typing import NDArray

from ...core.ir import Instruction
from ..registers.bits.bitwise import _validate_wires
from ..registers.primitives.controlled_rotation import controlled_rz_ir
from ..registers.primitives.global_phase import scalar_phase_ir
from .one_qubit_optimization import _emit, _instruction_matrix
from .two_qubit_synthesis import (
    _BasisDecomposer,
    _named_entangler_matrix,
    _swap_register,
    _weyl_decomposition,
)

Matrix = NDArray[np.complex128]


def _exact_one(matrix: list[list[complex]], wire: int) -> tuple[Instruction, ...]:
    result: list[Instruction] = []
    for gate in _emit(matrix, qubit=wire, metadata={}):
        if gate.name != "u3":
            result.append(gate)
            continue
        theta, phi, lam = (float(gate.params[k]) for k in ("theta", "phi", "lbd"))
        result.extend(
            (
                Instruction("rz", (wire,), params={"theta": lam}),
                Instruction("ry", (wire,), params={"theta": theta}),
                Instruction("rz", (wire,), params={"theta": phi}),
                *scalar_phase_ir(wire, (phi + lam) / 2),
            )
        )
    return tuple(result)


def _kak_body(matrix: Matrix) -> tuple[Instruction, ...]:
    target = _weyl_decomposition(_swap_register(matrix.tolist()))
    factors = _BasisDecomposer("cx").decomposition(target, 3)
    result: list[Instruction] = []
    # Use the full KAK form: fidelity-based count selection can round a small
    # nonzero interaction to the identity. Exact phase consumers cannot do so.
    for pair in range(4):
        for wire in range(2):
            result.extend(_exact_one(factors[2 * pair + wire], wire))
        if pair < 3:
            result.append(Instruction("cx", (0, 1)))
    return tuple(result)


def _kak_scalar(matrix: Matrix, body: tuple[Instruction, ...]) -> float:
    """Recover the KAK scalar using only its supplied 4x4 local factors."""
    product = np.eye(4, dtype=np.complex128)
    identity = np.eye(2, dtype=np.complex128)
    cx = np.asarray(_swap_register(_named_entangler_matrix("cx")))
    for gate in body:
        if gate.name == "cx":
            local = cx
        else:
            value = _instruction_matrix(gate)
            if value is None:
                raise ValueError("KAK factor is not a bound single-qubit unitary")
            local = np.asarray(
                (
                    np.kron(value, identity)
                    if gate.wires == (0,)
                    else np.kron(identity, value)
                ),
                dtype=np.complex128,
            )
        product = local @ product
    index = np.unravel_index(np.argmax(np.abs(matrix)), matrix.shape)
    coefficient = matrix[index] / product[index]
    angle = atan2(float(coefficient.imag), float(coefficient.real))
    if not np.allclose(matrix, np.exp(1j * angle) * product, atol=1e-10, rtol=0):
        raise ValueError("KAK factors do not reproduce the supplied local unitary")
    return angle


def _control_body(
    body: tuple[Instruction, ...], control: int
) -> tuple[Instruction, ...]:
    result: list[Instruction] = []
    for gate in body:
        if gate.name == "cx":
            result.append(Instruction("ccx", (control, *gate.wires)))
        elif gate.name == "x":
            result.append(Instruction("cx", (control, *gate.wires)))
        elif gate.name in {"rz", "phase"}:
            angle = float(gate.params["theta"])
            result.extend(controlled_rz_ir(control, gate.wires[0], angle))
            if gate.name == "phase":
                result.append(
                    Instruction("phase", (control,), params={"theta": angle / 2})
                )
        elif gate.name == "ry":
            half = float(gate.params["theta"]) / 2
            cx = Instruction("cx", (control, *gate.wires))
            result.extend(
                (
                    Instruction("ry", gate.wires, params={"theta": half}),
                    cx,
                    Instruction("ry", gate.wires, params={"theta": -half}),
                    cx,
                )
            )
        else:
            raise ValueError("Unexpected gate in exact local factor body")
    return tuple(result)


def local_unitary_ir(
    matrix: Sequence[Sequence[complex]] | Matrix,
    wires: tuple[int, ...],
    *,
    control: int | None = None,
) -> tuple[Instruction, ...]:
    """Synthesize a local matrix, optionally under one positive binary control.

    Wires are ordered MSB first. No quantum work, measurement or classical slots
    are required. The complete scalar phase is retained on an existing wire or
    becomes a phase on the control. The result contains named native gates only.
    Invalid shapes, nonfinite/nonunitary matrices and larger target words raise
    ValueError. Larger algorithms must supply their own decomposed local leaves.
    """
    _validate_wires(wires if control is None else (control, *wires))
    if len(wires) not in (1, 2):
        raise ValueError("Local unitary synthesis requires one or two target wires")
    value = np.array(matrix, dtype=np.complex128, copy=True)
    dimension = 1 << len(wires)
    if (
        value.shape != (dimension, dimension)
        or not np.isfinite(value).all()
        or not np.allclose(
            value.conj().T @ value, np.eye(dimension), atol=1e-10, rtol=0
        )
    ):
        raise ValueError("Local synthesis requires a finite unitary of matching shape")
    if len(wires) == 1:
        body = _exact_one(value.tolist(), wires[0])
        return body if control is None else _control_body(body, control)
    local = _kak_body(value)
    angle = _kak_scalar(value, local)
    body = tuple(
        Instruction(g.name, tuple(wires[w] for w in g.wires), g.params) for g in local
    )
    if control is None:
        return (*body, *scalar_phase_ir(wires[0], angle))
    return (
        *_control_body(body, control),
        Instruction("phase", (control,), params={"theta": angle}),
    )
