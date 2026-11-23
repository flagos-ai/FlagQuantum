# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix sim/fuzzer.cc and sim/fuzzer.h.
# Revision 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Callback-driven conformance sampling over existing physical circuit owners."""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Literal

from ..compiler.artifacts.phase_angle import PhaseAngle128
from ..compiler.formats.kmb import KmxRegister
from ..compiler.formats.kmx_builder import KmxCircuitBuilder
from ..compiler.formats.kmx_circuit import KmxCircuit
from ..core._fixed_width_integer import FixedWidthInteger
from ..drawer.reversible_phase_diagram import render_circuit_text
from ..simulation._basis_simulation import PackedBasisSimulation
from ..simulation.fixed_width_sampling import sample_fixed_width


class InputSample:
    """Named scalar cells with source-compatible access tracking."""

    def __init__(self, names: dict[str, int], cells: list[FixedWidthInteger]) -> None:
        self._names = names
        self._cells = cells
        self.touched: set[str] = set()

    def __getitem__(self, name: str) -> FixedWidthInteger:
        index = self._names[name]
        self.touched.add(name)
        return self._cells[index]

    def __setitem__(self, name: str, value: FixedWidthInteger | int) -> None:
        old = self[name]
        self._cells[self._names[name]] = (
            FixedWidthInteger(value.width, value.value)
            if isinstance(value, FixedWidthInteger)
            else old.assign_integer(value, signed=value < 1 << 63)
        )


class OutputSample(InputSample):
    """Expected cells and exact global phase, with retained original inputs."""

    def __init__(
        self,
        names: dict[str, int],
        cells: list[FixedWidthInteger],
        original: list[FixedWidthInteger],
    ) -> None:
        super().__init__(names, cells)
        self._original = original
        self.phase_half_turns = PhaseAngle128()

    def old(self, name: str) -> FixedWidthInteger:
        return self._original[self._names[name]]


class CircuitFuzzer:
    """Own 64-lane conformance batches and first-failure diagnostics.

    Use as a context manager or call close() for deterministic unused warnings.
    Circuit construction, scalar arithmetic and numerical execution retain their
    existing domain owners. Sample updates assign immutable successors.
    """

    def __init__(self, draws: Iterator[int]) -> None:
        self.rng = draws
        self._sim = PackedBasisSimulation(64, draws)
        self.circuit = KmxCircuit((), ())
        self.disarmed_warning = False
        self.ignore_global_phase = False
        self.seen_errors = 0
        self.seen_phase_errors = 0
        self.seen_shots = 0
        self.seen_configurations = 0
        self.primitive_counts: Counter[str] = Counter()
        self._builder_sampler: (
            Callable[[KmxCircuitBuilder, Iterator[int]], None] | None
        ) = None
        self._input_sampler: Callable[[InputSample, Iterator[int]], None] | None = None
        self._output_sampler: Callable[[OutputSample], None] | None = None
        self._combined_sampler: (
            Callable[[InputSample, OutputSample, Iterator[int]], None] | None
        ) = None
        self._before = self._sim.register_buffer
        self._after = self._sim.new_register_buffer()
        self._expected = self._sim.expected_buffer
        self._inputs = [InputSample({}, row) for row in self._before]
        self._outputs = [
            OutputSample({}, row, original)
            for row, original in zip(self._expected, self._before, strict=True)
        ]
        self._closed = False

    def __copy__(self) -> "CircuitFuzzer":
        raise TypeError("Circuit fuzzers own their buffers and cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> "CircuitFuzzer":
        raise TypeError("Circuit fuzzers own their buffers and cannot be copied")

    def __enter__(self) -> "CircuitFuzzer":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed and self.seen_shots == 0 and not self.disarmed_warning:
            print("WARNING: a circuit fuzzer didn't do any fuzzing!", file=sys.stderr)
        self._closed = True

    def use_circuit_builder(
        self, callback: Callable[[KmxCircuitBuilder, Iterator[int]], None] | None
    ) -> None:
        self._builder_sampler = callback

    def use_input_sampler(
        self, callback: Callable[[InputSample, Iterator[int]], None] | None
    ) -> None:
        self._input_sampler = callback

    def use_output_sampler(
        self, callback: Callable[[OutputSample], None] | None
    ) -> None:
        self._output_sampler = callback

    def use_input_and_output_sampler(
        self,
        callback: Callable[[InputSample, OutputSample, Iterator[int]], None] | None,
    ) -> None:
        self._combined_sampler = callback

    def switch_to_new_circuit(self) -> None:
        if self._builder_sampler is None:
            raise ValueError("Didn't call 'use_circuit_builder(...)'")
        self.seen_configurations += 1
        builder = KmxCircuitBuilder()
        self._builder_sampler(builder, self.rng)
        built = builder.freeze()
        assigned = {
            wire
            for register in built.registers
            for kind, wire in register.members
            if kind == "q"
        }
        unassigned: tuple[tuple[Literal["q", "b"], int], ...] = tuple(
            ("q", i) for i in range(built.qubits) if i not in assigned
        )
        if unassigned:
            builder.replace_registers(
                (*built.registers, KmxRegister("@unassigned", unassigned))
            )
        self.circuit = builder.freeze()
        self._sim.configure_for(self.circuit)
        self._before = self._sim.register_buffer
        self._after = self._sim.new_register_buffer()
        self._expected = self._sim.expected_buffer
        names = {r.name: i for i, r in enumerate(self.circuit.registers)}
        names = dict(sorted(names.items()))
        self._inputs = [InputSample(names, row) for row in self._before]
        self._outputs = [
            OutputSample(names, row, original)
            for row, original in zip(self._expected, self._before, strict=True)
        ]

    def _sample_lane(self, ins: InputSample, outs: OutputSample) -> None:
        ins.touched.clear()
        for name, index in ins._names.items():
            if name.startswith("@"):
                if name == "@dirty":
                    ins[name] = sample_fixed_width(ins._cells[index].width, self.rng)
                elif name in ("@clean", "@unassigned"):
                    ins[name] = 0
                else:
                    raise ValueError(f"Unrecognized special register name '{name}'")
                outs[name] = outs[name].write_from(ins[name])
        if self._input_sampler is not None and self._combined_sampler is not None:
            raise ValueError(
                "Specified both an input_sampler and an input_and_output_sampler"
            )
        if self._output_sampler is not None and self._combined_sampler is not None:
            raise ValueError(
                "Specified both an output_sampler and an input_and_output_sampler"
            )
        if self._combined_sampler is not None:
            outs.phase_half_turns = PhaseAngle128()
            self._combined_sampler(ins, outs, self.rng)
            self._require_touched(
                ins, "input_and_output_sampler didn't touch the input register named"
            )
            self._require_touched(
                outs, "input_and_output_sampler didn't touch the output register named"
            )
            return
        if self._input_sampler is not None:
            self._input_sampler(ins, self.rng)
            self._require_touched(ins, "input_sampler didn't touch the register named")
        else:
            for name, index in ins._names.items():
                if not name.startswith("@"):
                    ins._cells[index] = sample_fixed_width(
                        ins._cells[index].width, self.rng
                    )
        outs.phase_half_turns = PhaseAngle128()
        for name, index in outs._names.items():
            value = ins._cells[index]
            outs._cells[index] = FixedWidthInteger(value.width, value.value)
        if self._output_sampler is not None:
            self._output_sampler(outs)

    @staticmethod
    def _require_touched(sample: InputSample, message: str) -> None:
        for name in sample._names:
            if name not in sample.touched:
                raise ValueError(f"{message} '{name}'.")

    def _batch(self, requested: int) -> int:
        self._sim.clear_for_shot()
        performed = min(requested, 64)
        for lane in range(performed):
            self._sample_lane(self._inputs[lane], self._outputs[lane])
        self._sim.copy_register_buffer_into_state()
        self._sim.apply(self.circuit, counts=self.primitive_counts)
        self._sim.register_buffer = self._after
        try:
            self._sim.copy_state_into_register_buffer()
        finally:
            self._sim.register_buffer = self._before
        detail = ""
        for lane in range(performed):
            self.seen_shots += 1
            actual_phase = PhaseAngle128._from_units(self._sim.read_shot_phase(lane))
            phase_error = actual_phase != self._outputs[lane].phase_half_turns
            if self._after[lane] != self._expected[lane] or (
                phase_error and not self.ignore_global_phase
            ):
                self.seen_errors += 1
                self.seen_phase_errors += int(phase_error)
                if self.seen_errors == 1:
                    detail = self._failure_detail(lane, actual_phase)
        if self.seen_errors:
            raise ValueError(
                "Fuzz testing failed.\n"
                f"    Circuit configurations: {self.seen_configurations}\n"
                f"    Total shots: {self.seen_shots}\n"
                f"    Total errors: {self.seen_errors}\n"
                f"    Total phase errors: {self.seen_phase_errors}\n\n{detail}"
            )
        return performed

    def fuzz_current_circuit(self, shots: int) -> None:
        if type(shots) is not int or not 0 <= shots < 1 << 64:
            raise ValueError("Shots must fit an unsigned 64-bit count")
        while shots:
            shots -= self._batch(shots)

    def fuzz(self, num_variations: int, num_shots_per_variation: int) -> None:
        if type(num_variations) is not int or not 0 <= num_variations < 1 << 64:
            raise ValueError("Variations must fit an unsigned 64-bit count")
        for _ in range(num_variations):
            self.switch_to_new_circuit()
            self.fuzz_current_circuit(num_shots_per_variation)

    def _failure_detail(self, lane: int, actual_phase: PhaseAngle128) -> str:
        grid = [["", "input", "output", "expected_output", "diff"], [""] * 5]
        if not self.ignore_global_phase:
            expected_phase = self._outputs[lane].phase_half_turns
            grid[1] = [
                "phase",
                "0.0π",
                str(actual_phase),
                str(expected_phase),
                "X" if actual_phase != expected_phase else "",
            ]
        registers = []
        for index, register in enumerate(self.circuit.registers):
            before, actual, expected = (
                buffer[lane][index]
                for buffer in (self._before, self._after, self._expected)
            )
            diff = expected.xor_word(actual)
            grid.append(
                [
                    f"{register.name}(r{index})",
                    before.binary(),
                    actual.binary(),
                    expected.binary(),
                    (
                        ""
                        if not diff
                        else "".join("X" if c == "1" else " " for c in diff.binary())
                    ),
                ]
            )
            quantum = sum(kind == "q" for kind, _ in register.members)
            classical = len(register.members) - quantum
            kinds = "".join(kind for kind, _ in register.members)
            descriptor = f"{quantum}q" if not classical else f"{classical}b"
            if quantum and classical:
                descriptor = kinds
            if not kinds:
                descriptor = "(empty)"
            registers.append(f"    {register.name}(r{index}): {descriptor}\n")
        widths = [max(len(cell.encode()) for cell in column) for column in grid]
        rows = [
            " ".join(
                " " * (width - len(column[row].encode())) + column[row]
                for width, column in zip(widths, grid, strict=True)
            )
            for row in range(5)
        ]
        detail = (
            "\nRegisters:\n"
            + "".join(registers)
            + "\nState:\n"
            + "\n".join(rows)
            + "\n"
        )
        if self.circuit.qubits < 40 and len(self.circuit.records) < 1000:
            detail += "\nCircuit:\n" + render_circuit_text(self.circuit)
        return detail + "\n"
