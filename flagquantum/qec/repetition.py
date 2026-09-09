"""Three-qubit repetition-code reference workflow."""

from __future__ import annotations

from typing import Sequence

from ..compiler._hybrid import BOOL, INDEX, capture_source, lower_dynamic_program
from ..runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from .decoders import Decoder, RepetitionLookupDecoder
from .types import (
    DetectionEvent,
    RepetitionMemoryResult,
    RepetitionMemoryShot,
    SyndromeRound,
)

_MEMORY_SOURCE = """
def repetition_memory(inject_error, error_wire, rounds):
    if inject_error:
        qp.X(wires=error_wire)
    left = False
    right = False
    for round_index in range(rounds):
        qp.CNOT(wires=[0, 3])
        qp.CNOT(wires=[1, 3])
        left = qp.measure(wires=3)
        qp.reset(wires=3)
        qp.CNOT(wires=[1, 4])
        qp.CNOT(wires=[2, 4])
        right = qp.measure(wires=4)
        qp.reset(wires=4)
        if left and not right:
            qp.X(wires=0)
        elif left and right:
            qp.X(wires=1)
        elif not left and right:
            qp.X(wires=2)
    return right
"""


def _syndrome_rounds(
    values: Sequence[int], *, rounds: int
) -> tuple[SyndromeRound, ...]:
    if len(values) != rounds * 2:
        raise ValueError("classical register does not match repetition-code rounds")
    previous = (0, 0)
    records = []
    for round_index in range(rounds):
        bits = (int(values[2 * round_index]), int(values[2 * round_index + 1]))
        events = tuple(
            DetectionEvent(round_index, check_index)
            for check_index, (before, after) in enumerate(zip(previous, bits))
            if before != after
        )
        records.append(SyndromeRound(round_index, bits, events))
        previous = bits
    return tuple(records)


def run_repetition_memory_experiment(
    *,
    error_wire: int | None = None,
    rounds: int = 3,
    shots: int = 128,
    seed: int | None = 0,
    strategy: str = "auto",
    analysis_decoder: Decoder | None = None,
) -> RepetitionMemoryResult:
    """Run the bounded three-qubit bit-flip memory reference.

    This is a development-evidence workflow over a noiseless local simulator;
    it does not establish logical error suppression or fault tolerance. The
    optional analysis decoder interprets recorded syndromes; the compiled
    reference lookup remains responsible for executed feedback.
    """

    if error_wire is not None and (
        type(error_wire) is not int or error_wire not in {0, 1, 2}
    ):
        raise ValueError("error_wire must select data wire 0, 1, 2, or None")
    if type(rounds) is not int or rounds <= 0:
        raise ValueError("rounds must be a positive integer")
    if type(shots) is not int or shots <= 0:
        raise ValueError("shots must be a positive integer")
    selected_decoder = (
        RepetitionLookupDecoder() if analysis_decoder is None else analysis_decoder
    )
    if not isinstance(selected_decoder, Decoder):
        raise TypeError("analysis_decoder must implement the Decoder protocol")

    program = capture_source(_MEMORY_SOURCE, (BOOL, INDEX, INDEX))
    lowered = lower_dynamic_program(
        program,
        (error_wire is not None, 0 if error_wire is None else error_wire, rounds),
        max_dynamic_measurements=rounds * 2,
    )
    execution = execute_hybrid_dynamic_session(
        lowered.circuit,
        shots=shots,
        seed=seed,
        strategy=strategy,
    )
    classical_rows = execution.classical_bits.tolist()
    sample_rows = execution.samples.tolist()
    if len(classical_rows) != shots or len(sample_rows) != shots:
        raise RuntimeError("dynamic execution returned an unexpected shot count")
    shot_records = []
    for shot_index, (classical, sample) in enumerate(zip(classical_rows, sample_rows)):
        syndrome_rounds = _syndrome_rounds(classical, rounds=rounds)
        corrections = tuple(
            selected_decoder.decode(record.bits, round_index=record.round_index)
            for record in syndrome_rounds
        )
        final_data = tuple(int(value) for value in sample[:3])
        logical_bit = int(sum(final_data) >= 2)
        shot_records.append(
            RepetitionMemoryShot(
                shot_index=shot_index,
                syndrome_rounds=syndrome_rounds,
                corrections=corrections,
                final_data_bits=final_data,
                logical_bit=logical_bit,
                logical_failure=bool(logical_bit),
            )
        )
    return RepetitionMemoryResult(
        rounds=rounds,
        injected_error_wire=error_wire,
        shot_records=tuple(shot_records),
        execution_semantics=execution.execution_semantics,
    )


__all__ = ("run_repetition_memory_experiment",)
