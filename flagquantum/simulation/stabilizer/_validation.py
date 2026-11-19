"""Stabilizer request and symbolic record validation without SDK state."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

from ...errors import CapabilityError, ValidationError

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


def _bounded_stim_expansion(program: Any) -> int:
    """Count repeated instructions before asking the SDK to expand them."""
    count = 0
    pending = [(program, 1)]
    while pending:
        body, multiplier = pending.pop()
        for instruction in body:
            if hasattr(instruction, "repeat_count"):
                pending.append(
                    (instruction.body_copy(), multiplier * instruction.repeat_count)
                )
            else:
                count += multiplier
            if count > 1_000_000:
                raise CapabilityError(
                    "Stim experiment exceeds 1,000,000 expanded instructions"
                )
    return count


def _validate_stim_experiment_source(circuit_text: str) -> None:
    if not isinstance(circuit_text, str):
        raise TypeError("circuit_text must be a string")
    if not circuit_text.strip():
        raise ValueError("circuit_text must not be empty")
    if len(circuit_text.encode("utf-8")) > 1_048_576:
        raise CapabilityError("Stim experiment exceeds 1 MiB of UTF-8 source")


def _validate_stim_reference(circuit: Any) -> None:
    try:
        circuit.without_noise().detector_error_model(allow_gauge_detectors=False)
    except ValueError as error:
        raise CapabilityError(
            "Stim experiment has nondeterministic declared parities"
        ) from error
