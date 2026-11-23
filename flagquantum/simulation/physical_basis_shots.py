# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix Sim::apply.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/sim/sim.h

"""Bit-packed trajectories for computational-basis reversible circuits."""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from random import Random
from typing import Protocol

PHYSICAL_SAMPLE_BATCH_SIZE = 64

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
class BasisInput:
    """Snapshot of the initialized packed wires before circuit execution."""

    qubits: dict[int, int]
    bits: dict[int, int]


@dataclass(frozen=True)
class BasisBatch:
    """Packed wire values, exact per-shot phase and actual operation counts."""

    qubits: dict[int, int]
    bits: dict[int, int]
    phase_units: tuple[int, ...]
    counts: Counter[str]
    sign_mask: int
    before: BasisInput | None


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


@dataclass
class PackedBasisState:
    """Packed persistent state with one exact angle accumulator per lane."""

    qubits: defaultdict[int, int]
    bits: defaultdict[int, int]
    phases: list[int]
    sign: int = 0
    conditions: list[int] = field(default_factory=list)


def apply_packed_basis_records(
    records: Iterable[BasisRecord],
    state: PackedBasisState,
    draw: Callable[[], int],
    *,
    counts: Counter[str] | None = None,
    debug: Callable[[int, bool], None] | None = None,
    observer: Callable[[BasisRecord, int], None] | None = None,
) -> None:
    """Apply physical records to packed state while retaining the caller-owned condition stack."""
    q, b, phases = state.qubits, state.bits, state.phases
    shots = len(phases)
    all_lanes = (1 << shots) - 1
    scopes = state.conditions
    base_condition = scopes[-1] if scopes else all_lanes
    sign = state.sign

    for record in records:
        code = record.opcode
        predicate = record.condition_bit
        active = base_condition
        if predicate is not None:
            active &= b[predicate]
        if observer is not None:
            counted = (
                all_lanes
                if code in {"PUSH_CONDITION", "POP_CONDITION", "INFO_TAG"}
                else active
            )
            observer(record, counted)
        if code == "PUSH_CONDITION":
            base_condition = active
            scopes.append(active)
        elif code == "POP_CONDITION":
            if scopes:
                scopes.pop()
            base_condition = scopes[-1] if scopes else all_lanes
        if code in {"PUSH_CONDITION", "POP_CONDITION", "INFO_TAG"}:
            if counts is not None:
                counts[code] += shots
            continue

        if counts is not None:
            counts[code] += active.bit_count()
        if code.startswith("DEBUG_PRINT"):
            if debug is not None and active & 1:
                if record.qubits:
                    debug(q[record.qubits[0]] & 1, False)
                elif record.result_bit is not None:
                    debug(b[record.result_bit] & 1, False)
                else:
                    debug((phases[0] + (state.sign & 1) * _SIGN) % _TURN, True)
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
            else:
                raise ValueError(f"Unsupported physical basis operation {code!r}")
            if record.angle is None:
                sign ^= active
                continue
            phase = record.angle.units
            while active:
                low = active & -active
                index = low.bit_length() - 1
                phases[index] = (phases[index] + phase) % _TURN
                active ^= low

    state.sign = sign


def sample_basis_batch(
    records: Iterable[BasisRecord],
    *,
    shots: int,
    rng: Random,
    initialization: Sequence[RegisterInitialization] = (),
    outcome_mask: Callable[[], int] | None = None,
    capture_before: bool = False,
    debug: Callable[[int, bool], None] | None = None,
    observer: Callable[[BasisRecord, int], None] | None = None,
) -> BasisBatch:
    """Run up to 64 input-basis shots and report exact combined phase.

    An optional debug observer receives shot zero's value when active and
    a flag identifying angle values. Debug angle reads see the rotation
    accumulator; the source commits the separate sign mask after the scan.
    """
    if not 1 <= shots <= PHYSICAL_SAMPLE_BATCH_SIZE:
        raise ValueError("A physical basis batch requires 1 to 64 shots")
    q: defaultdict[int, int] = defaultdict(int)
    b: defaultdict[int, int] = defaultdict(int)
    _apply_initialization(initialization, q, b, shots, rng)
    before = BasisInput(dict(q), dict(b)) if capture_before else None
    state = PackedBasisState(q, b, [0] * shots)
    counts: Counter[str] = Counter()
    draw = outcome_mask if outcome_mask is not None else lambda: rng.getrandbits(shots)

    apply_packed_basis_records(
        records, state, draw, counts=counts, debug=debug, observer=observer
    )

    angles = tuple(
        (phase + ((state.sign >> i) & 1) * _SIGN) % _TURN
        for i, phase in enumerate(state.phases)
    )
    return BasisBatch(dict(q), dict(b), angles, counts, state.sign, before)
