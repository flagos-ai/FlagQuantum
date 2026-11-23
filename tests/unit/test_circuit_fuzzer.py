"""Compare complete physical fuzzer orchestration with locked source traces."""

import copy
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import pytest

from flagquantum.compiler.artifacts.phase_angle import PhaseAngle128
from flagquantum.compiler.formats.kmb import KmxRegister
from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from flagquantum.core._fixed_width_integer import FixedWidthInteger
from flagquantum.simulation.physical_basis_rng import _mt19937_64
from flagquantum.simulation.physical_basis_shots import (
    BasisRecord,
    PackedBasisState,
    apply_packed_basis_records,
)
from flagquantum.testing import _circuit_fuzzer as module
from flagquantum.testing._circuit_fuzzer import CircuitFuzzer, InputSample, OutputSample

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "circuit_fuzzer.json").read_text()
)


def source_scenario(mode: int) -> str:
    trace: list[str] = []
    fuzzer = CircuitFuzzer(_mt19937_64(123))
    fuzzer.disarmed_warning = True
    inputs, outputs, builds = 0, 0, 0

    def build(builder: KmxCircuitBuilder, rng: Iterator[int]) -> None:
        nonlocal builds
        builds += 1
        trace.append(f"build:{builds}:{next(rng)}\n")
        if mode == 27:
            return
        if mode == 20:
            raise ValueError("builder failure")
        if mode == 14:
            builder.replace_registers(
                (
                    KmxRegister("z", tuple(("q", i) for i in range(5))),
                    KmxRegister("a", tuple(("q", i) for i in range(5, 10))),
                )
            )
            return
        if mode == 7:
            builder.replace_registers(
                (
                    KmxRegister("same", (("q", 0), ("q", 1), ("q", 2))),
                    KmxRegister("same", (("q", 3), ("q", 4), ("q", 5))),
                )
            )
            return
        if mode == 6:
            builder.replace_registers((KmxRegister("@other", (("q", 0),)),))
            return
        width = 65 if mode == 19 else 3
        registers = [KmxRegister("target", tuple(("q", i) for i in range(width)))]
        if mode == 8:
            builder.append_kmx_text("X q5")
        if mode == 17:
            registers.append(KmxRegister("mixed", (("q", 0), ("b", 0), ("q", 2))))
            builder.append_kmx_text("BIT_STORE1 b0")
        if mode in (18, 5):
            registers.append(KmxRegister("@dirty", (("q", 3), ("q", 4))))
        if mode == 18:
            registers.append(KmxRegister("@clean", (("q", 5), ("q", 6))))
        builder.replace_registers(registers)
        if mode in (0, 1, 12, 24, 25, 26):
            builder.append_kmx_text("CCX q0 q1 q2\nCX q0 q1")
            if mode != 26:
                builder.append_kmx_text("X q0")
            if mode == 24:
                builder.append_kmx_text("X q2")
            if mode == 25:
                builder.append_kmx_text("Z q2")
        if mode in (9, 10, 11, 31):
            builder.append_kmx_text("Z q0")
        if mode in (16, 30):
            builder.append_kmx_text("HMR q0 b0")

    def combined(ins: InputSample, outs: OutputSample, rng: Iterator[int]) -> None:
        nonlocal inputs
        inputs += 1
        if mode != 29:
            ins["target"] = 0
        if mode != 4 and (mode != 2 or inputs <= 64):
            outs["target"] = 0
        trace.append(f"both:{inputs}:{next(rng)}\n")

    def input_sample(ins: InputSample, rng: Iterator[int]) -> None:
        nonlocal inputs
        inputs += 1
        trace.append(f"input:{inputs}:{next(rng)}\n")
        if mode == 3:
            return
        ins["target"] = inputs & 7
        if mode == 17:
            ins["mixed"] = 0
        if mode == 18:
            trace.append(f"special:{ins['@dirty'].binary()}:{ins['@clean'].binary()}\n")
            ins["@clean"] = 3

    def output_sample(outs: OutputSample) -> None:
        nonlocal outputs
        outputs += 1
        if mode == 14:
            trace.append(f"lex:{outs.old('z').binary()}:{outs.old('a').binary()}\n")
            return
        trace.append(f"old:{outputs}:{outs.old('target').binary()}\n")
        if mode in (0, 24, 25, 26):
            outs["target"] = outs["target"].add_scalar(1)
        if mode in (1, 12):
            outs["target"] = outs["target"].add_scalar(1, subtract=True)
        if mode == 9:
            outs.phase_half_turns = PhaseAngle128.from_integer_half_turns(
                int(outs["target"].bit(0))
            )
        if mode == 11:
            outs["target"] = outs["target"].add_scalar(1)
        if mode in (16, 30):
            outs["target"] = outs["target"].set_bit(0, False)
        if mode == 17:
            outs["mixed"] = outs["mixed"].set_bit(1)
            outs["target"] = 0

    if mode != 21:
        fuzzer.use_circuit_builder(build)
    if mode in (2, 4, 5, 23, 28, 29):
        fuzzer.use_input_and_output_sampler(combined)
    if mode in (3, 5, 15, 17, 18, 19, 22, 23):
        fuzzer.use_input_sampler(input_sample)
    if mode in (0, 1, 9, 11, 12, 14, 15, 16, 17, 19, 22, 24, 25, 26, 28, 30, 31):
        fuzzer.use_output_sampler(output_sample)
    fuzzer.ignore_global_phase = mode in (10, 11)
    try:
        shots = (
            129 if mode in (2, 15, 19) else 0 if mode == 22 else 64 if mode == 30 else 3
        )
        fuzzer.fuzz(0 if mode == 13 else 1, shots)
        trace.append("ok\n")
    except ValueError as error:
        trace.append(f"error:{error}\n")
    if mode == 12:
        try:
            fuzzer.fuzz_current_circuit(1)
        except ValueError as error:
            trace.append(f"again:{error}\n")

    def next_draw(builder: KmxCircuitBuilder, rng: Iterator[int]) -> None:
        trace.append(f"next:{next(rng)}\n")

    fuzzer.use_circuit_builder(next_draw)
    fuzzer.fuzz(1, 0)
    fuzzer.close()
    return "".join(trace)


@pytest.mark.parametrize("mode", range(32))
def test_original_complete_fuzzer_traces(mode: int) -> None:
    expected = _DATA["observations"][str(mode)]
    if mode == 7:
        # Existing Compiler validation owns native wording for duplicate names.
        expected = expected.replace(
            "Duplicate register name: same", "Duplicate KMB register name"
        )
    assert source_scenario(mode) == expected


def test_sample_cells_immutable_successors_and_original_snapshot() -> None:
    cells = [FixedWidthInteger(3, 2)]
    sample = InputSample({"x": 0}, cells)
    original = sample["x"]
    sample["x"] = original.add_scalar(1)
    assert original.value == 2 and cells[0].value == 3
    out = OutputSample({"x": 0}, cells.copy(), cells)
    out["x"] = 5
    assert out.old("x").value == 3
    with pytest.raises(KeyError):
        sample["missing"]
    sample["x"] = -1
    assert sample["x"].value == 7


def test_fuzzer_explicit_lifecycle_copy_and_invalid_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with CircuitFuzzer(_mt19937_64(0)) as unused:
        unused.fuzz(0, 0)
    unused.close()
    assert (
        capsys.readouterr().err == "WARNING: a circuit fuzzer didn't do any fuzzing!\n"
    )
    with CircuitFuzzer(_mt19937_64(0)) as disarmed:
        disarmed.disarmed_warning = True
    assert capsys.readouterr().err == ""
    for copier in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError, match="cannot be copied"):
            copier(unused)
    for count in (-1, True, 1 << 64):
        with pytest.raises(ValueError, match="Shots"):
            unused.fuzz_current_circuit(count)
        with pytest.raises(ValueError, match="Variations"):
            unused.fuzz(count, 0)


def test_fuzzer_replaces_packed_execution_with_scalar_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def scalar_lanes(
        records: Iterable[BasisRecord],
        state: PackedBasisState,
        draw: Callable[[], int],
        *,
        counts: Counter[str],
    ) -> None:
        captured = tuple(records)
        measurements = [draw() for record in captured if record.opcode in ("HMR", "R")]
        for lane in range(len(state.phases)):
            single = PackedBasisState(
                defaultdict(
                    int, {i: (word >> lane) & 1 for i, word in state.qubits.items()}
                ),
                defaultdict(
                    int, {i: (word >> lane) & 1 for i, word in state.bits.items()}
                ),
                [state.phases[lane]],
                (state.sign >> lane) & 1,
            )
            outcomes = iter((word >> lane) & 1 for word in measurements)
            apply_packed_basis_records(
                captured, single, outcomes.__next__, counts=counts
            )
            for target, source in (
                (state.qubits, single.qubits),
                (state.bits, single.bits),
            ):
                for wire, word in source.items():
                    target[wire] = (target[wire] & ~(1 << lane)) | ((word & 1) << lane)
            state.phases[lane] = single.phases[0]
            state.sign = (state.sign & ~(1 << lane)) | ((single.sign & 1) << lane)

    expected = [source_scenario(mode) for mode in range(32)]
    monkeypatch.setattr(module, "apply_packed_basis_records", scalar_lanes)
    assert [source_scenario(mode) for mode in range(32)] == expected


@pytest.mark.parametrize("outcome", (0, 1, 0xAAAAAAAAAAAAAAAA))
def test_fuzzer_measurement_feedback_exact_phase_workspace_and_tail_counts(
    outcome: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshots: list[tuple[dict[int, int], dict[int, int], tuple[int, ...], int]] = []

    def execution(
        records: Iterable[BasisRecord],
        state: PackedBasisState,
        draw: Callable[[], int],
        *,
        counts: Counter[str],
    ) -> None:
        apply_packed_basis_records(records, state, lambda: outcome, counts=counts)
        snapshots.append(
            (dict(state.qubits), dict(state.bits), tuple(state.phases), state.sign)
        )

    monkeypatch.setattr(module, "apply_packed_basis_records", execution)

    def build(builder: KmxCircuitBuilder, rng: Iterator[int]) -> None:
        builder.replace_registers((KmxRegister("target", (("q", 0), ("q", 1))),))
        builder.append_kmx_text("HMR q0 b0\nZ q1 if b0\nX q1")

    def sample(ins: InputSample, rng: Iterator[int]) -> None:
        ins["target"] = 3

    def expect(outs: OutputSample) -> None:
        assert outs.old("target").value == 3
        outs["target"] = 0

    with CircuitFuzzer(_mt19937_64(0)) as fuzzer:
        fuzzer.use_circuit_builder(build)
        fuzzer.use_input_sampler(sample)
        fuzzer.use_output_sampler(expect)
        fuzzer.fuzz(2, 129)
    assert (
        fuzzer.seen_shots == 258 and fuzzer.seen_errors == fuzzer.seen_phase_errors == 0
    )
    assert fuzzer.primitive_counts == {
        "HMR": 384,
        "X": 384,
        "Z": 6 * (outcome & ((1 << 64) - 1)).bit_count(),
    }
    assert len(snapshots) == 6
    for qubits, bits, phases, sign in snapshots:
        assert not any(qubits.values()) and not sign and not any(phases)
        assert bits[0] == outcome & ((1 << 64) - 1)


def test_default_empty_circuit() -> None:
    fuzzer = CircuitFuzzer(_mt19937_64(0))
    fuzzer.fuzz_current_circuit(1)
    assert fuzzer.seen_shots == 1
    fuzzer.close()


@pytest.mark.parametrize("width,operations", ((3, 1), (40, 1), (3, 1001)))
def test_error_grid_empty_register_and_diagram_threshold(
    width: int, operations: int
) -> None:
    with CircuitFuzzer(_mt19937_64(0)) as fuzzer:

        def build(builder: KmxCircuitBuilder, rng: Iterator[int]) -> None:
            builder.replace_registers(
                (
                    KmxRegister("empty", ()),
                    KmxRegister("target", tuple(("q", i) for i in range(width))),
                )
            )
            builder.append_kmx_text("\n".join(["X q0"] * operations))

        fuzzer.use_circuit_builder(build)
        with pytest.raises(ValueError) as failure:
            fuzzer.fuzz(1, 1)
        assert "empty(r0): (empty)" in str(failure.value)
        assert ("Circuit:\n" in str(failure.value)) == (
            width < 40 and operations < 1000
        )


@pytest.mark.parametrize("mode", range(4))
def test_original_unused_fuzzer_warning(
    mode: int, capsys: pytest.CaptureFixture[str]
) -> None:
    with CircuitFuzzer(_mt19937_64(123)) as fuzzer:
        if mode == 1:
            fuzzer.disarmed_warning = True
        if mode == 2:
            fuzzer.fuzz_current_circuit(0)
        if mode == 3:
            fuzzer.fuzz_current_circuit(1)
    assert capsys.readouterr().err == _DATA["observations"][f"warning:{mode}"]


def test_original_simple_xor_variations_use_existing_integer_updates() -> None:
    widths: list[int] = []

    def build(builder: KmxCircuitBuilder, rng: Iterator[int]) -> None:
        width = next(rng) % 10
        widths.append(width)
        builder.replace_registers(
            (
                KmxRegister("src", tuple(("q", i) for i in range(width))),
                KmxRegister("dst", tuple(("q", i + width) for i in range(width))),
            )
        )
        builder.append_kmx_text(
            "\n".join(f"CX q{i} q{i + width}" for i in range(width))
        )

    def expect(sample: OutputSample) -> None:
        sample["dst"] ^= sample["src"]

    with CircuitFuzzer(_mt19937_64(123)) as fuzzer:
        fuzzer.use_circuit_builder(build)
        fuzzer.use_output_sampler(expect)
        fuzzer.fuzz(6, 256)
    assert fuzzer.seen_configurations == 6 and fuzzer.seen_shots == 1536
    assert fuzzer.primitive_counts == {"CX": 256 * sum(widths)}


def test_phase_reset_follows_input_callback() -> None:
    retained: list[OutputSample] = []
    seen_inputs = 0

    def build(builder: KmxCircuitBuilder, rng: Iterator[int]) -> None:
        builder.replace_registers((KmxRegister("x", (("q", 0),)),))

    def sample(ins: InputSample, rng: Iterator[int]) -> None:
        nonlocal seen_inputs
        if seen_inputs >= 64:
            retained[seen_inputs % 64].phase_half_turns = (
                PhaseAngle128.from_integer_half_turns(1)
            )
        seen_inputs += 1
        ins["x"] = 0

    def expect(outs: OutputSample) -> None:
        if len(retained) < 64:
            retained.append(outs)
        assert not outs.phase_half_turns

    with CircuitFuzzer(_mt19937_64(123)) as fuzzer:
        fuzzer.use_circuit_builder(build)
        fuzzer.use_input_sampler(sample)
        fuzzer.use_output_sampler(expect)
        fuzzer.fuzz(1, 129)
    assert fuzzer.seen_shots == 129
