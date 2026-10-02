"""One-qubit Euler-angle synthesis into a z-rotation plus a pi/2 pulse basis.

`gates.native` publishes the single-qubit gates a target can execute. Every
single-qubit unitary in FlagQuantum IR is `U3(theta, phi, lam)` up to a global
phase, and

    ``U3(theta, phi, lam) == RZ(lam) SX RZ(theta - pi) SX RZ(phi - pi)``

also up to a global phase. `RX(pi/2)` *is* `SX` exactly, so a basis that
publishes either one reaches the whole one-qubit group with a z-rotation and that
single pulse. `sxdg` is never needed because `SXDG = RZ(-pi) SX RZ(-pi)`.

The angle triples of the parameter-free gates are tabulated rather than solved
numerically, so they stay exact and bit-reproducible. Parameterized gates carry
the caller's own parameter objects through, so a trainable rotation stays
trainable, and a trainable angle never selects a short branch.

**Global phase is dropped.** `RZ` and `SX` generate a determinant group whose
phases are multiples of `i`, so `H`, `S`, `T`, and the rest cannot be matched
exactly, and FlagQuantum IR has no field in which to record the difference. The
caller's exact hand-written rewrites stay ahead of this module, so the phase is
only lost where the target's basis leaves no exact form at all. A global phase
is unobservable in every expectation value, but a consumer that compares raw
statevectors rather than measurement statistics has to know this.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from ..core.ir import Instruction
from ..core.operator_schema import canonical_opcode, get_operator_schema
from .pipeline import _is_zero

_HALF_PI = 0.5 * math.pi
_QUARTER_PI = 0.25 * math.pi

# The opcodes that can carry the pi/2 pulse of the general form, most preferred
# first. `sx` is parameter-free; `rx` is the same pulse spelled as a rotation.
HALF_PI_PULSE_OPCODES: tuple[str, ...] = ("sx", "rx")

# ZYZ angle triples `(theta, phi, lam)` for the parameter-free single-qubit
# unitary opcodes of `OPERATOR_SCHEMAS`, read as
# `opcode == exp(1j * phase) * U3(theta, phi, lam)`.
_FIXED_EULER_ANGLES: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {
        "i": (0.0, 0.0, 0.0),
        "x": (math.pi, 0.0, math.pi),
        "y": (math.pi, _HALF_PI, _HALF_PI),
        "z": (0.0, 0.0, math.pi),
        "h": (_HALF_PI, 0.0, math.pi),
        "s": (0.0, 0.0, _HALF_PI),
        "sdg": (0.0, 0.0, -_HALF_PI),
        "t": (0.0, 0.0, _QUARTER_PI),
        "tdg": (0.0, 0.0, -_QUARTER_PI),
        "sx": (_HALF_PI, -_HALF_PI, _HALF_PI),
        "sxdg": (-_HALF_PI, -_HALF_PI, _HALF_PI),
    }
)


def _rx_angles(instruction: Instruction) -> tuple[Any, Any, Any]:
    # `RX(theta) = RZ(-pi/2) RY(theta) RZ(pi/2)`.
    return (instruction.params["theta"], -_HALF_PI, _HALF_PI)


def _ry_angles(instruction: Instruction) -> tuple[Any, Any, Any]:
    return (instruction.params["theta"], 0.0, 0.0)


def _z_angles(instruction: Instruction) -> tuple[Any, Any, Any]:
    return (0.0, 0.0, instruction.params["theta"])


def _u2_angles(instruction: Instruction) -> tuple[Any, Any, Any]:
    return (_HALF_PI, instruction.params["phi"], instruction.params["lbd"])


def _u3_angles(instruction: Instruction) -> tuple[Any, Any, Any]:
    return (
        instruction.params["theta"],
        instruction.params["phi"],
        instruction.params["lbd"],
    )


# ZYZ angle triples for the opcodes that read their angles from IR parameters.
# `rz`, `phase`, and `u1` differ only by a global phase, so all three delegate to
# whichever of them the target publishes.
_PARAMETERIZED_EULER_ANGLES: Mapping[
    str, Callable[[Instruction], tuple[Any, Any, Any]]
] = MappingProxyType(
    {
        "rx": _rx_angles,
        "ry": _ry_angles,
        "rz": _z_angles,
        "phase": _z_angles,
        "u1": _z_angles,
        "u2": _u2_angles,
        "u3": _u3_angles,
    }
)


def _polar_angle(value: Any) -> float | None:
    """Return `value` as a float for branch selection, or None when it has none.

    A trainable value never selects a branch, for the same reason
    `pipeline._is_zero` refuses to call a trainable angle zero: the short forms
    omit `theta` entirely, which would silently detach a rotation initialized at
    zero from the autograd graph.
    """
    if bool(getattr(value, "requires_grad", False)):
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (RuntimeError, ValueError):
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _pulse_leaf(
    opcode: str,
    instruction: Instruction,
    *,
    angle: Any,
) -> Instruction | None:
    """The pi/2 pulse as `opcode`, or None when the angle has become zero."""
    schema = get_operator_schema(opcode)
    parameters = schema.parameters if schema is not None else ()
    if not parameters:
        return Instruction(opcode, instruction.wires, metadata=instruction.metadata)
    if _is_zero(angle):
        return None
    return Instruction(
        opcode,
        instruction.wires,
        params=dict.fromkeys(parameters, angle),
        metadata=instruction.metadata,
    )


def _leaves(
    theta: Any,
    phi: Any,
    lam: Any,
    *,
    z_rotation: str,
    pulse_opcode: str,
) -> tuple[tuple[str, Any], ...]:
    """Return the gates of one Euler triple, in application order.

    The general form is `RZ(lam) PULSE RZ(theta - pi) PULSE RZ(phi - pi)`, which
    is `U3(theta, phi, lam)` up to a global phase: the sequence a ZSX decomposer
    emits, with `SX` or an equivalent `RX(pi/2)` pulse. A polar angle of zero,
    `+/-pi/2`, or `+/-pi` collapses to fewer gates. Each entry is
    `(opcode, angle)`, where a `None` angle marks the pulse. The angles are the
    caller's own objects, so a leaf stays trainable when the source rotation was.
    """
    polar = _polar_angle(theta)
    if polar == 0.0:
        return ((z_rotation, phi + lam),)
    if polar == _HALF_PI:
        return (
            (z_rotation, lam - _HALF_PI),
            (pulse_opcode, None),
            (z_rotation, phi + _HALF_PI),
        )
    if polar == -_HALF_PI:
        return (
            (z_rotation, lam + _HALF_PI),
            (pulse_opcode, None),
            (z_rotation, phi - _HALF_PI),
        )
    if polar is not None and abs(polar) == math.pi:
        return (
            (pulse_opcode, None),
            (pulse_opcode, None),
            (z_rotation, phi + math.pi - lam),
        )
    return (
        (z_rotation, lam),
        (pulse_opcode, None),
        (z_rotation, theta - math.pi),
        (pulse_opcode, None),
        (z_rotation, phi - math.pi),
    )


def synthesize_one_qubit(
    instruction: Instruction,
    *,
    z_rotation: str,
    pulse_opcode: str = "sx",
) -> tuple[Instruction, ...] | None:
    """Return `instruction` as a z-rotation and pi/2 pulses, or None.

    `z_rotation` is the z-rotation opcode the target publishes and `pulse_opcode`
    the opcode it publishes for a pi/2 rotation about x. None is returned when
    nothing here applies: a multi-wire or non-unitary instruction, an instruction
    the operator schema does not describe, or an instruction that is already that
    same z-rotation, whose shortest form only the caller's native gate set can
    choose.
    """
    opcode = canonical_opcode(instruction.name)
    if opcode == z_rotation:
        return None
    schema = get_operator_schema(opcode)
    if schema is None or not schema.unitary or schema.arity != 1:
        return None
    angles = _FIXED_EULER_ANGLES.get(opcode)
    if angles is None:
        source = _PARAMETERIZED_EULER_ANGLES.get(opcode)
        if source is None:
            return None
        angles = source(instruction)
    leaves = _leaves(*angles, z_rotation=z_rotation, pulse_opcode=pulse_opcode)
    replacement: list[Instruction] = []
    for leaf_opcode, angle in leaves:
        if leaf_opcode == pulse_opcode:
            pulse = _pulse_leaf(leaf_opcode, instruction, angle=_HALF_PI)
            if pulse is not None:
                replacement.append(pulse)
        elif not _is_zero(angle):
            replacement.append(
                Instruction(
                    z_rotation,
                    instruction.wires,
                    params={"theta": angle},
                    metadata=instruction.metadata,
                )
            )
    return tuple(replacement)


__all__ = ("synthesize_one_qubit",)
