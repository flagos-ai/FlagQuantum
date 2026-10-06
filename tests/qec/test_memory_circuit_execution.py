"""Integration coverage for the code-driven memory circuit."""

from __future__ import annotations

import pytest

from flagquantum.compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from flagquantum.compiler._hybrid.dynamic_lowering import LoweredDynamicProgram
from flagquantum.qec.circuit import MeasurementRef, MemoryCircuit, build_memory_circuit
from flagquantum.qec.codes import CodeCheck, RepetitionCode
from flagquantum.qec.repetition import _memory_source
from flagquantum.qec.surface import RotatedSurfaceCode
from flagquantum.qec.types import ErrorSchedule
from flagquantum.runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session

pytestmark = pytest.mark.integration


def _lower(source: str, *, checks: int, rounds: int) -> LoweredDynamicProgram:
    program = capture_source(source, (INDEX,))
    return lower_dynamic_program(
        program, (rounds,), max_dynamic_measurements=rounds * checks
    )


def _run(
    memory: MemoryCircuit, *, shots: int
) -> tuple[list[list[int]], list[list[int]]]:
    lowered = _lower(
        memory.source, checks=len(memory.code.checks), rounds=memory.rounds
    )
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=shots, seed=0)
    return execution.classical_bits.tolist(), execution.samples.tolist()


def _with_injected_x(source: str, *, round_index: int, wire: int) -> str:
    anchor = "    for round_index in range(rounds):\n"
    assert source.count(anchor) == 1
    injection = (
        f"        if round_index == {round_index}:\n"
        f"            qp.X(wires={wire})\n"
    )
    return source.replace(anchor, anchor + injection)


def _detector_bits(
    memory: MemoryCircuit, classical: list[int], sample: list[int]
) -> tuple[int, ...]:
    code = memory.code
    checks = len(code.checks)
    ancilla_index = {check.ancilla_wire: check.index for check in code.checks}

    def value(reference: MeasurementRef) -> int:
        if reference.round_index is None:
            return int(sample[reference.wire])
        offset = reference.round_index * checks + ancilla_index[reference.wire]
        return int(classical[offset])

    return tuple(
        sum(value(reference) for reference in detector.parity) % 2
        for detector in memory.detectors.detectors
    )


def _observable_bits(memory: MemoryCircuit, sample: list[int]) -> tuple[int, ...]:
    return tuple(
        sum(int(sample[reference.wire]) for reference in observable.measurement_parity)
        % 2
        for observable in memory.observables.observables
    )


def test_distance_three_matches_the_frozen_circuit_exactly() -> None:
    memory = build_memory_circuit(RepetitionCode(3), rounds=3)

    general = _lower(memory.source, checks=2, rounds=3).circuit
    frozen = _lower(
        _memory_source(ErrorSchedule(), compiled_feedback=False), checks=2, rounds=3
    ).circuit

    assert general.n_wires == frozen.n_wires == 5
    assert general.instructions == frozen.instructions


@pytest.mark.parametrize("distance", (5, 7))
def test_scaled_distances_lower_and_execute_unchanged(distance: int) -> None:
    rounds = distance
    memory = build_memory_circuit(RepetitionCode(distance), rounds=rounds)

    lowered = _lower(memory.source, checks=distance - 1, rounds=rounds)
    assert lowered.circuit.n_wires == 2 * distance - 1

    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=8, seed=0)
    assert execution.samples.shape == (8, 2 * distance - 1)
    assert execution.classical_bits.shape == (8, rounds * (distance - 1))


@pytest.mark.parametrize("distance", (3, 5))
def test_noiseless_run_fires_no_detector_and_no_observable(distance: int) -> None:
    memory = build_memory_circuit(RepetitionCode(distance), rounds=3)
    classical_rows, sample_rows = _run(memory, shots=4)
    expected = (0,) * (distance - 1) * 4

    for classical, sample in zip(classical_rows, sample_rows, strict=True):
        assert _detector_bits(memory, classical, sample) == expected
        assert _observable_bits(memory, sample) == (0,)


def test_one_injected_error_fires_exactly_the_round_zero_boundary_detector() -> None:
    memory = build_memory_circuit(RepetitionCode(3), rounds=3)
    injected = _with_injected_x(memory.source, round_index=0, wire=0)
    lowered = _lower(injected, checks=2, rounds=3)
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=4, seed=0)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        assert _detector_bits(memory, classical, sample) == (1, 0, 0, 0, 0, 0, 0, 0)
        assert _observable_bits(memory, sample) == (1,)


def test_injected_error_on_the_middle_wire_fires_two_detectors() -> None:
    memory = build_memory_circuit(RepetitionCode(3), rounds=3)
    injected = _with_injected_x(memory.source, round_index=0, wire=1)
    lowered = _lower(injected, checks=2, rounds=3)
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=2, seed=0)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        assert _detector_bits(memory, classical, sample) == (1, 1, 0, 0, 0, 0, 0, 0)
        assert _observable_bits(memory, sample) == (1,)


def test_injected_error_in_the_final_round_fires_only_the_final_boundary() -> None:
    memory = build_memory_circuit(RepetitionCode(3), rounds=3)
    injected = _with_injected_x(memory.source, round_index=2, wire=0)
    lowered = _lower(injected, checks=2, rounds=3)
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=4, seed=0)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        assert _detector_bits(memory, classical, sample) == (0, 0, 0, 0, 1, 0, 0, 0)
        assert _observable_bits(memory, sample) == (1,)


def test_injected_error_in_the_middle_round_fires_its_own_boundary_detector() -> None:
    memory = build_memory_circuit(RepetitionCode(3), rounds=3)
    injected = _with_injected_x(memory.source, round_index=1, wire=0)
    lowered = _lower(injected, checks=2, rounds=3)
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=4, seed=0)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        assert _detector_bits(memory, classical, sample) == (0, 0, 1, 0, 0, 0, 0, 0)
        assert _observable_bits(memory, sample) == (1,)


def test_smallest_code_and_single_round_lower_and_execute() -> None:
    memory = build_memory_circuit(RepetitionCode(2), rounds=1)
    lowered = _lower(memory.source, checks=1, rounds=1)
    assert lowered.circuit.n_wires == 3

    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=4, seed=0)
    assert execution.samples.shape == (4, 3)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        assert _detector_bits(memory, classical, sample) == (0, 0)
        assert _observable_bits(memory, sample) == (0,)


def test_smallest_code_reports_an_injected_error() -> None:
    memory = build_memory_circuit(RepetitionCode(2), rounds=1)
    injected = _with_injected_x(memory.source, round_index=0, wire=0)
    lowered = _lower(injected, checks=1, rounds=1)
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=4, seed=0)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        assert _detector_bits(memory, classical, sample) == (1, 0)
        assert _observable_bits(memory, sample) == (1,)


def _with_injected_z(source: str, *, round_index: int, wire: int) -> str:
    """Inject a Z error, which the hybrid gate set expresses as ``H X H``.

    The bounded compiler program admits ``h``, ``x``, ``rx``, ``ry``, ``cx``,
    ``reset`` and ``measure``, so ``Z`` is written as the conjugation that
    produces it rather than added to the gate set for a test.
    """

    anchor = "    for round_index in range(rounds):\n"
    assert source.count(anchor) == 1
    injection = (
        f"        if round_index == {round_index}:\n"
        f"            qp.H(wires={wire})\n"
        f"            qp.X(wires={wire})\n"
        f"            qp.H(wires={wire})\n"
    )
    return source.replace(anchor, anchor + injection)


def _surface_code_memory(*, rounds: int = 3) -> MemoryCircuit:
    """A distance-three rotated surface-code memory experiment.

    Only the distance-three patch is executed here. The dynamic simulator draws
    each shot from the full output distribution, so sampling a 17-wire patch
    costs a 2**17-way draw while a distance-five patch would need 2**49
    categories. Larger patches are reached through the detector error model
    instead of through this simulator, and this file stays at the size the
    simulator can sample.
    """

    return build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=rounds)


def _check_rows(checks: tuple[CodeCheck, ...], *, x_type: bool) -> tuple[int, ...]:
    return tuple(
        index
        for index, check in enumerate(checks)
        if bool(check.stabilizer.x_wires) is x_type
    )


def _fired_round_zero_positions(memory: MemoryCircuit, wire: int) -> set[int]:
    """Positions, inside round zero, of the Z-type detectors an X error flips.

    Round zero declares a detector only for a Z-type check, in check order, so a
    detector's position inside the round is its position within the Z-type rows.
    The expectation is derived from the code's own stabilizer support rather than
    transcribed, because which check sees the error is exactly what is being
    checked.
    """

    z_rows = _check_rows(memory.code.checks, x_type=False)
    return {
        position
        for position, row in enumerate(z_rows)
        if wire in memory.code.checks[row].stabilizer.support
    }


def test_surface_code_noiseless_run_fires_nothing() -> None:
    memory = _surface_code_memory()
    classical_rows, sample_rows = _run(memory, shots=4)
    expected = (0,) * len(memory.detectors.detectors)

    for classical, sample in zip(classical_rows, sample_rows, strict=True):
        assert _detector_bits(memory, classical, sample) == expected
        assert _observable_bits(memory, sample) == (0,)


def test_surface_code_x_type_checks_see_a_z_error_and_the_z_type_checks_do_not() -> (
    None
):
    """A Z error is invisible to a Z-type check and visible to an X-type one.

    A Z error commutes with every Z stabilizer, so no Z-type check changes
    outcome; it anticommutes with the X stabilizers whose support contains the
    data qubit, so those change outcome. An X-type check is deterministic only
    against the round before it, so the change first becomes reportable in round
    one and cannot be reported at all for an error injected before round zero.
    """

    memory = _surface_code_memory()
    checks = memory.code.checks
    z_count = len(_check_rows(checks, x_type=False))
    x_rows = _check_rows(checks, x_type=True)
    z_rows = _check_rows(checks, x_type=False)
    assert len(x_rows) == len(z_rows) == 4

    # Data wire 4 is the interior lattice site, the only data qubit at distance
    # three that touches two X-type ancillas.
    adjacent_x_rows = {row for row in x_rows if 4 in checks[row].stabilizer.support}
    assert len(adjacent_x_rows) == 2

    injected = _with_injected_z(memory.source, round_index=1, wire=4)
    lowered = _lower(injected, checks=len(checks), rounds=memory.rounds)
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=4, seed=0)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        detectors = _detector_bits(memory, classical, sample)
        # Rounds after the first declare a detector for every check, in check
        # order, so an X-type check's detector is at its own row offset.
        fired = {index for index, value in enumerate(detectors) if value}
        assert fired == {z_count + row for row in adjacent_x_rows}
        assert all(detectors[z_count + row] == 0 for row in z_rows)
        assert _observable_bits(memory, sample) == (0,)


def test_surface_code_x_error_fires_the_z_type_checks_on_the_logical_row() -> None:
    """An X error on the logical row flips the declared observable.

    The declared logical operator is ``Z`` on the data row ``j == 0``, so an X
    error anywhere on that row anticommutes with it and flips the readout. A
    Z-type check is deterministic in round zero and at the terminal readout, so
    the error is reported by the round-zero detector of every Z-type check whose
    support contains the wire; the final syndrome and the terminal data readout
    then agree, because both carry the same flipped parity, so no terminal
    detector fires.
    """

    memory = _surface_code_memory()
    checks = memory.code.checks
    z_count = len(_check_rows(checks, x_type=False))
    fired_positions = _fired_round_zero_positions(memory, 0)
    assert len(fired_positions) == 1

    injected = _with_injected_x(memory.source, round_index=0, wire=0)
    lowered = _lower(injected, checks=len(checks), rounds=memory.rounds)
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=4, seed=0)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        detectors = _detector_bits(memory, classical, sample)
        assert {index for index in range(z_count) if detectors[index]} == (
            fired_positions
        )
        assert not any(detectors[z_count:])
        assert _observable_bits(memory, sample) == (1,)


def test_surface_code_x_error_off_the_logical_row_leaves_the_observable_alone() -> None:
    """The complement of the previous test, on the same code and the same round.

    Data wire 4 is the interior lattice site, which is off the declared logical
    row. It touches two Z-type checks rather than the one a corner data qubit
    touches, and it leaves the row observable at zero.
    """

    memory = _surface_code_memory()
    checks = memory.code.checks
    z_count = len(_check_rows(checks, x_type=False))
    fired_positions = _fired_round_zero_positions(memory, 4)
    assert len(fired_positions) == 2

    injected = _with_injected_x(memory.source, round_index=0, wire=4)
    lowered = _lower(injected, checks=len(checks), rounds=memory.rounds)
    execution = execute_hybrid_dynamic_session(lowered.circuit, shots=4, seed=0)

    for classical, sample in zip(
        execution.classical_bits.tolist(), execution.samples.tolist(), strict=True
    ):
        detectors = _detector_bits(memory, classical, sample)
        assert {index for index in range(z_count) if detectors[index]} == (
            fired_positions
        )
        assert not any(detectors[z_count:])
        assert _observable_bits(memory, sample) == (0,)
