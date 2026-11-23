# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix py/interactive_simulator.pybind.cc/.h.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Own persistent shot chunks over the shared physical numerical lifecycle."""

from collections import Counter
from typing import Literal

from ._basis_simulation import PackedBasisSimulation, _Circuit
from .physical_basis_rng import _mt19937_64_seed_sequence

WireKind = Literal["q", "b", "bool"]
Wire = tuple[WireKind, int]
_COUNTER_MASK = (1 << 64) - 1
_SIGN = 1 << 127


class BasisTrajectories:
    """Retain numerical chunks, streams and valid-prefix observation semantics."""

    def __init__(
        self, shots: int, seed_words: tuple[int, ...], *, counting: bool = False
    ) -> None:
        if type(shots) is not int or not 0 <= shots < 1 << 64:
            raise ValueError("Shot count must be an unsigned 64-bit integer")
        self.shots = shots
        self.counting = counting
        master = _mt19937_64_seed_sequence(seed_words)
        self.chunks: list[PackedBasisSimulation] = []
        for _ in range(0, shots, 256):
            words: list[int] = []
            for _ in range(4):
                value = next(master)
                words.extend((value & ((1 << 32) - 1), value >> 32))
            self.chunks.append(
                PackedBasisSimulation(
                    256, _mt19937_64_seed_sequence(tuple(words)), counting=counting
                )
            )

    def ensure_wire(self, wire: Wire) -> None:
        kind, index = wire
        for chunk in self.chunks:
            chunk.ensure_big_enough_state_for(
                index + 1 if kind == "q" else 0,
                index + 1 if kind == "b" else 0,
            )

    def apply(self, circuit: _Circuit) -> None:
        for chunk in self.chunks:
            chunk.ensure_big_enough_state_for(circuit.qubits, circuit.bits)
        for chunk in self.chunks:
            chunk.apply(circuit)

    def clear_for_shot(self) -> None:
        for chunk in self.chunks:
            chunk.clear_for_shot()

    def read_across(self, wire: Wire) -> int:
        return sum(
            chunk.safe_read(*wire) << (256 * index)
            for index, chunk in enumerate(self.chunks)
        ) & ((1 << self.shots) - 1)

    def write_across(self, wire: Wire, value: int) -> None:
        self.ensure_wire(wire)
        kind, index = wire
        mask = (1 << 256) - 1
        for position, chunk in enumerate(self.chunks):
            values = chunk.state.qubits if kind == "q" else chunk.state.bits
            values[index] = (value >> (position * 256)) & mask

    def read_bit(self, wire: Wire, shot: int) -> bool:
        if not 0 <= shot < self.shots:
            raise IndexError("Need shot_index < sim.batch_size")
        return bool((self.chunks[shot // 256].safe_read(*wire) >> (shot % 256)) & 1)

    def write_bit(self, wire: Wire, shot: int, value: bool) -> None:
        if not 0 <= shot < self.shots:
            raise IndexError("Need shot_index < sim.batch_size")
        chunk = self.chunks[shot // 256]
        kind, index = wire
        values = chunk.state.qubits if kind == "q" else chunk.state.bits
        mask = 1 << (shot % 256)
        values[index] = (values[index] & ~mask) | (mask if value else 0)

    def read_phase(self, shot: int) -> int:
        if not 0 <= shot < self.shots:
            raise IndexError("Need shot_index < sim.batch_size")
        return self.chunks[shot // 256].read_shot_phase(shot % 256)

    def write_phase(self, shot: int, units: int) -> None:
        if not 0 <= shot < self.shots:
            raise IndexError("Need shot_index < sim.batch_size")
        state = self.chunks[shot // 256].state
        state.phases[shot % 256] = units
        state.sign &= ~(1 << (shot % 256))

    def phase_flipped(self, shot: int) -> bool:
        phase = self.read_phase(shot)
        if phase not in (0, _SIGN):
            raise ValueError(
                "Phase angles other than 0 (unphased) and 1 (phase flipped) "
                "are present. Use `read_shot_phase` instead of "
                "`read_phase_flipped_across_shots`."
            )
        return phase == _SIGN

    def phase_mask(self) -> int:
        return sum(int(self.phase_flipped(shot)) << shot for shot in range(self.shots))

    def counter_snapshot(
        self, shot: int | None = None
    ) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[int, int], ...]]:
        if not self.counting:
            raise ValueError("Operation counting is not enabled on this simulator")
        if shot is not None:
            if not 0 <= shot < self.shots:
                raise IndexError("Need 0 <= shot_index < sim.batch_size")
            return self.chunks[shot // 256].counters.snapshot(shot % 256)
        operations: Counter[str] = Counter()
        levels: Counter[int] = Counter()
        for position, chunk in enumerate(self.chunks):
            for lane in range(min(256, self.shots - position * 256)):
                counted, rotations = chunk.counters.snapshot(lane)
                operations.update(dict(counted))
                levels.update(dict(rotations))
        return (
            tuple((name, value & _COUNTER_MASK) for name, value in operations.items()),
            tuple((level, value & _COUNTER_MASK) for level, value in levels.items()),
        )

    def clear_operation_statistics(self) -> None:
        for chunk in self.chunks:
            chunk.clear_operation_statistics()
