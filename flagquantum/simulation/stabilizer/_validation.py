"""Scalar and Pauli-support validation without SDK state."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

from ...errors import ValidationError

# Stim documents a seed as an integer in range(2**64).
_SEED_LIMIT = 2**64


def _validate_seed(seed: Any) -> int | None:
    """Return a seed the engine accepts, refusing anything that is not one.

    A seed that is not an integer denotes no seed, so it is refused rather than
    coerced: `int(2.5)` would sample a different stream than the caller wrote
    while the result reported the requested seed. `bool` is refused for the
    reason the IR count helpers give: it is a flag, not an index.
    """

    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValidationError(
            f"stabilizer sampling seed must be an integer or None, got {seed!r}"
        )
    if not 0 <= seed < _SEED_LIMIT:
        raise ValidationError(
            f"stabilizer sampling seed must be in range(2**64), got {seed}"
        )
    return seed


def _pauli_letters(
    n_wires: int,
    x: Sequence[int],
    y: Sequence[int],
    z: Sequence[int],
) -> str:
    """Return the tracked Pauli, refusing a qubit named twice or off the program.

    A qubit the string does not name carries the identity, and the three axes are
    disjoint by construction, so a qubit appearing twice is a malformed string
    rather than a string two spellings agree about. `bool` is refused for the
    reason the seed check gives: it is a flag, not a qubit index.
    """

    letters = ["I"] * n_wires
    for axis, wires in (("x", x), ("y", y), ("z", z)):
        for wire in wires:
            if isinstance(wire, bool):
                raise ValidationError(
                    f"Pauli readout {axis} wires must be integers, got {wire!r}"
                )
            # `operator.index` is the protocol for "I am exactly an integer", so
            # `0.0` and `'1'` are refused while a NumPy or Torch scalar is not.
            # `int(...)` would also have accepted them, and `0.5` would have
            # truncated to qubit 0 rather than being refused.
            try:
                index = operator.index(wire)
            except TypeError as error:
                raise ValidationError(
                    f"Pauli readout {axis} wires must be integers, got {wire!r}"
                ) from error
            if not 0 <= index < n_wires:
                raise ValidationError(
                    f"Pauli readout {axis} wire {index} is outside a "
                    f"{n_wires}-wire program"
                )
            if letters[index] != "I":
                raise ValidationError(
                    f"Pauli readout names wire {index} on more than one axis; a "
                    "Pauli string carries one letter per wire"
                )
            letters[index] = axis.upper()
    return "".join(letters)


def _validate_shots(shots: Any) -> int:
    """Return the shot count, refusing anything that is not a positive integer."""

    if isinstance(shots, bool) or not isinstance(shots, int):
        raise ValidationError(f"shot count must be an integer, got {shots!r}")
    if shots <= 0:
        raise ValidationError(f"shot count must be positive, got {shots}")
    return shots
