# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix Sim::apply.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/sim/sim.h

"""Bit-packed trajectories for computational-basis reversible circuits."""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from random import Random
from typing import Protocol

_TURN = 1 << 128
_SIGN = _TURN >> 1


class _Angle(Protocol):
    @property
    def units(self) -> int: ...


class BasisRecord(Protocol):
    @property
    def opcode(self) -> str: ...
    @property
    def qubits(self) -> tuple[int, ...]: ...
    @property
    def condition_bit(self) -> int | None: ...
    @property
    def result_bit(self) -> int | None: ...
    @property
    def angle(self) -> _Angle | None: ...


@dataclass(frozen=True)
class RegisterInitialization:
    """Write an ordered mixed register from a fixed or sampled integer."""

    members: tuple[tuple[str, int], ...]
    value: int
    randomize: bool = False


@dataclass(frozen=True)
class BasisBatch:
    """Packed wire values, exact per-shot phase and actual operation counts."""

    qubits: dict[int, int]
    bits: dict[int, int]
    phase_units: tuple[int, ...]
    counts: Counter[str]


def _apply_initialization(
    assignments: Sequence[RegisterInitialization],
    qubits: defaultdict[int, int],
    bits: defaultdict[int, int],
    shots: int,
    rng: Random,
) -> None:
    mask = (1 << shots) - 1
    for assignment in assignments:
        width = len(assignment.members)
        value = assignment.value
        samples: list[int] | None = None
        if assignment.randomize:
            if value.bit_length() > width:
                raise ValueError("Random initialization modulus exceeds register width")
            samples = [
                rng.randrange(value) if value else rng.getrandbits(width)
                for _ in range(shots)
            ]
        for offset, (kind, wire) in enumerate(assignment.members):
            state = qubits if kind == "q" else bits
            if samples is None:
                state[wire] = mask if (value >> offset) & 1 else 0
            else:
                state[wire] = sum(
                    ((shot >> offset) & 1) << i for i, shot in enumerate(samples)
                )


def sample_basis_batch(
    records: Iterable[BasisRecord],
    *,
    shots: int,
    rng: Random,
    initialization: Sequence[RegisterInitialization] = (),
    outcome_mask: Callable[[], int] | None = None,
) -> BasisBatch:
    """Run up to 64 input-basis shots, retaining exact signed phase per shot."""
    if not 1 <= shots <= 64:
        raise ValueError("A physical basis batch requires 1 to 64 shots")
    full = (1 << shots) - 1
    q: defaultdict[int, int] = defaultdict(int)
    b: defaultdict[int, int] = defaultdict(int)
    _apply_initialization(initialization, q, b, shots, rng)
    scopes = [full]
    phases = [0] * shots
    counts: Counter[str] = Counter()
    draw = outcome_mask if outcome_mask is not None else lambda: rng.getrandbits(shots)

    for record in records:
        code = record.opcode
        predicate = record.condition_bit
        active = scopes[-1] if predicate is None else scopes[-1] & b[predicate]
        if code == "PUSH_CONDITION":
            scopes.append(active)
        elif code == "POP_CONDITION" and len(scopes) > 1:
            scopes.pop()
        if code in {"PUSH_CONDITION", "POP_CONDITION", "INFO_TAG"}:
            counts[code] += shots
            continue

        counts[code] += active.bit_count()
        if code.startswith("DEBUG_PRINT"):
            continue
        if code in {"BIT_STORE0", "BIT_STORE1", "BIT_INVERT"}:
            result = record.result_bit
            assert result is not None
            if code == "BIT_STORE0":
                b[result] &= ~active
            elif code == "BIT_STORE1":
                b[result] |= active
            else:
                b[result] ^= active
            continue

        if code in {"X", "CX", "CCX"}:
            for control in record.qubits[:-1]:
                active &= q[control]
            q[record.qubits[-1]] ^= active
        elif code in {"SWAP", "CSWAP"}:
            left, right = record.qubits[-2:]
            if code == "CSWAP":
                active &= q[record.qubits[0]]
            toggled = active & (q[left] ^ q[right])
            q[left] ^= toggled
            q[right] ^= toggled
        else:
            phase = _SIGN
            if code in {"R", "HMR"}:
                target = record.qubits[0]
                selected = active
                measured = draw() & selected
                if code == "HMR":
                    result = record.result_bit
                    assert result is not None
                    b[result] = (b[result] & ~active) | measured
                active = measured & q[target]
                q[target] &= ~selected
            elif code in {"Z", "CZ", "CCZ", "Z_POW", "NEG"}:
                for wire in record.qubits:
                    active &= q[wire]
                if record.angle is not None:
                    phase = record.angle.units
            else:
                raise ValueError(f"Unsupported physical basis operation {code!r}")
            while active:
                low = active & -active
                index = low.bit_length() - 1
                phases[index] = (phases[index] + phase) % _TURN
                active ^= low

    return BasisBatch(dict(q), dict(b), tuple(phases), counts)
