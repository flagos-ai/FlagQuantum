"""Integration coverage for the code-driven memory circuit."""

from __future__ import annotations

import pytest

from flagquantum.compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from flagquantum.qec.circuit import MemoryCircuit, build_memory_circuit
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.repetition import _memory_source
from flagquantum.qec.types import ErrorSchedule
from flagquantum.runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session

pytestmark = pytest.mark.integration


def _lower(source: str, *, checks: int, rounds: int):
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

    def value(reference) -> int:
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
