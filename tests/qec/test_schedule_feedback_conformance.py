"""Approved five-qubit/two-round cross-path and independent-reference checks."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Any

import pytest
import torch

from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import ValidationError
from flagquantum.qec import (
    ErrorEvent,
    ErrorSchedule,
    MeasurementRef,
    MeasurementSamples,
    PauliFrame,
    RepetitionCode,
    SyndromeRound,
    build_memory_circuit,
    run_repetition_memory_experiment,
)
from flagquantum.qec.execution.sampling import _measurement_plan
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from tests.qec.test_memory_circuit_execution import _lower, _with_injected_x

pytestmark = pytest.mark.integration

_ATOL = 1e-12
_FAULTS = (None, 0, 1, 2)
_SINGLE = ((None, None),) + tuple(
    tuple(wire if index == round_index else None for index in range(2))
    for round_index in range(2)
    for wire in range(3)
)
_CORRECTION = {(0, 0): None, (1, 0): 0, (1, 1): 1, (0, 1): 2}


def _syndrome(bits: list[int]) -> tuple[int, int]:
    return bits[0] ^ bits[1], bits[1] ^ bits[2]


def _schedule(faults: tuple[int | None, ...]) -> ErrorSchedule:
    return ErrorSchedule(
        tuple(ErrorEvent(r, wire) for r, wire in enumerate(faults) if wire is not None)
    )


def _direct(
    faults: tuple[int | None, ...],
    *,
    feedback: bool = False,
    encoded_plus: bool = False,
    undo: bool = False,
    omit_reset: bool = False,
    dephase: bool = False,
) -> DynamicCircuit:
    circuit = DynamicCircuit(5, dtype=torch.complex128, device="cpu")
    if encoded_plus:
        circuit.gate("h", 0)
        circuit.gate("cx", (0, 1))
        circuit.gate("cx", (0, 2))
    for r, wire in enumerate(faults):
        if wire is not None:
            circuit.gate("x", wire)
        for j, (left, right, ancilla) in enumerate(((0, 1, 3), (1, 2, 4))):
            circuit.gate("cx", (left, ancilla))
            circuit.gate("cx", (right, ancilla))
            circuit.measure(ancilla, classical_bit=2 * r + j)
            if not omit_reset:
                circuit.reset(ancilla)
        if feedback:
            circuit.conditional("x", 0, conditions={2 * r: 1, 2 * r + 1: 0})
            circuit.conditional("x", 1, conditions={2 * r: 1, 2 * r + 1: 1})
            circuit.conditional("x", 2, conditions={2 * r: 0, 2 * r + 1: 1})
    if dephase:
        circuit.gate("z", 0)
    if undo:
        circuit.gate("cx", (0, 2))
        circuit.gate("cx", (0, 1))
        circuit.gate("h", 0)
    return circuit


@pytest.mark.parametrize("faults", _SINGLE)
def test_declaration_lowering_and_direct_extraction_match_bit_oracle(
    faults: tuple[int | None, ...],
) -> None:
    memory = build_memory_circuit(RepetitionCode(3), rounds=2)
    refs = tuple(MeasurementRef(r, a) for r in range(2) for a in (3, 4)) + tuple(
        MeasurementRef(None, q) for q in range(3)
    )
    assert memory.measurement_refs == refs
    plan = _measurement_plan(memory)
    assert dict(plan.syndrome_columns) == {(0, 3): 0, (0, 4): 1, (1, 3): 2, (1, 4): 3}
    assert dict(plan.terminal_columns) == {0: 4, 1: 5, 2: 6}
    source = memory.source
    bits = [0, 0, 0]
    history: list[int] = []
    for r, wire in enumerate(faults):
        if wire is not None:
            source = _with_injected_x(source, round_index=r, wire=wire)
            bits[wire] ^= 1
        history.extend(_syndrome(bits))
    lowered = _lower(source, checks=2, rounds=2)
    actual = execute_hybrid_dynamic_session(lowered.circuit, shots=2, seed=53)
    direct = run_dynamic(_direct(faults), shots=2, seed=53, strategy="trajectory")
    expected = history + bits
    # Independent parity sets, deliberately not derived from the layouts under test.
    parity_sets = {
        frozenset((0,)),
        frozenset((1,)),
        frozenset((0, 2)),
        frozenset((1, 3)),
        frozenset((2, 4, 5)),
        frozenset((3, 5, 6)),
    }
    column = {ref: index for index, ref in enumerate(refs)}
    layout_sets = {
        frozenset(column[ref] for ref in d.parity) for d in memory.detectors.detectors
    }
    assert layout_sets == parity_sets
    assert len(memory.detectors) == len(parity_sets)
    for execution in (actual, direct):
        assert execution.classical_bits.tolist() == [history, history]
        assert execution.samples[:, :3].tolist() == [bits, bits]
        for measured, sampled in zip(
            execution.classical_bits, execution.samples, strict=True
        ):
            record = measured.tolist() + sampled[:3].tolist()
            assert record == expected
            assert [sum(record[i] for i in row) % 2 for row in layout_sets] == [
                sum(expected[i] for i in row) % 2 for row in layout_sets
            ]
            observable = memory.observables.observables[0].measurement_parity
            assert sum(record[column[ref]] for ref in observable) % 2 == sum(bits) % 2


@pytest.mark.parametrize("faults", tuple(itertools.product(_FAULTS, repeat=2)))
@pytest.mark.parametrize(
    "mode,strategy",
    (
        ("compiled_lookup", "trajectory"),
        ("compiled_lookup", "batched"),
        ("runtime_decoder", "trajectory"),
        ("runtime_pauli_frame", "trajectory"),
    ),
)
def test_online_records_match_independent_round_and_frame_oracle(
    faults: tuple[int | None, ...], mode: str, strategy: str
) -> None:
    data = [0, 0, 0]
    frame = [0, 0, 0]
    histories = []
    corrections = []
    for wire in faults:
        if wire is not None:
            data[wire] ^= 1
        raw = _syndrome(data)
        histories.append(raw)
        prior = _syndrome(frame)
        adjusted = (raw[0] ^ prior[0], raw[1] ^ prior[1])
        correction = _CORRECTION[adjusted]
        corrections.append(correction)
        if correction is not None:
            (frame if mode == "runtime_pauli_frame" else data)[correction] ^= 1
    result = run_repetition_memory_experiment(
        error_schedule=_schedule(faults),
        rounds=2,
        shots=2,
        seed=53,
        strategy=strategy,
        feedback_mode=mode,
    )
    assert result.shot_count == 2 and result.logical_failures == 0
    for shot in result.shot_records:
        assert [record.bits for record in shot.syndrome_rounds] == histories
        assert [item.qubit for item in shot.executed_feedback] == corrections
        assert shot.raw_final_data_bits == tuple(data)
        assert (
            shot.decoded_data_bits
            == tuple(a ^ b for a, b in zip(data, frame, strict=True))
            == (0, 0, 0)
        )
        if mode == "runtime_pauli_frame":
            assert PauliFrame.from_corrections(
                shot.executed_feedback
            ).x_qubits == tuple(q for q in range(3) if frame[q])


@pytest.mark.parametrize("faults", _SINGLE + ((0, 1),))
def test_offline_correction_radius_is_not_online_equivalence(
    faults: tuple[int | None, ...],
) -> None:
    data = [0, 0, 0]
    history = []
    for wire in faults:
        if wire is not None:
            data[wire] ^= 1
        history.append(_syndrome(data))
    correction = _CORRECTION[history[-1]]
    decoded = data.copy()
    if correction is not None:
        decoded[correction] ^= 1
    result = run_repetition_memory_experiment(
        error_schedule=_schedule(faults),
        rounds=2,
        shots=2,
        seed=53,
        strategy="trajectory",
        feedback_mode="offline_pauli_frame",
    )
    for shot in result.shot_records:
        assert [record.bits for record in shot.syndrome_rounds] == history
        assert all(not item.applied for item in shot.executed_feedback)
        assert shot.raw_final_data_bits == tuple(data)
        assert shot.decoded_data_bits == tuple(decoded)
        assert shot.logical_failure == (faults == (0, 1))
    assert result.logical_failures == (2 if faults == (0, 1) else 0)


def _permute(
    state: torch.Tensor, target: int, control: int | None = None
) -> torch.Tensor:
    """Independent big-endian basis permutation, not a Simulation gate helper."""
    result = torch.zeros_like(state)
    for index in range(32):
        active = control is None or bool(index & (1 << (4 - control)))
        result[index ^ (1 << (4 - target)) if active else index] = state[index]
    return result


def _projector_branches(
    wire: int | None,
) -> dict[tuple[int, ...], tuple[float, torch.Tensor | None]]:
    """Enumerate all sixteen joint syndrome outcomes without normalizing zeros."""
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[0] = initial[28] = 1 / math.sqrt(2)
    if wire is not None:
        initial = _permute(initial, wire)
    branches: dict[tuple[int, ...], tuple[float, torch.Tensor | None]] = {}
    for outcomes in itertools.product((0, 1), repeat=4):
        state = initial.clone()
        weight = 1.0
        for r in range(2):
            for j, (left, right, ancilla) in enumerate(((0, 1, 3), (1, 2, 4))):
                state = _permute(_permute(state, ancilla, left), ancilla, right)
                bit = outcomes[2 * r + j]
                projected = torch.tensor(
                    [
                        value if ((index >> (4 - ancilla)) & 1) == bit else 0j
                        for index, value in enumerate(state.tolist())
                    ],
                    dtype=torch.complex128,
                )
                probability = float(projected.abs().square().sum())
                weight *= probability
                if probability == 0:
                    break
                state = projected / math.sqrt(probability)
                if bit:
                    state = _permute(state, ancilla)
            else:
                correction = _CORRECTION[(outcomes[2 * r], outcomes[2 * r + 1])]
                if correction is not None:
                    state = _permute(state, correction)
                continue
            break
        branches[outcomes] = (weight, state if weight > 0 else None)
    return branches


def _density(state: torch.Tensor) -> torch.Tensor:
    return state[:, None] * state.conj()[None, :]


@pytest.mark.parametrize("wire", _FAULTS)
def test_joint_probabilities_conditional_density_and_coherence(
    wire: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branches = _projector_branches(wire)
    assert abs(sum(weight for weight, _ in branches.values()) - 1) <= _ATOL
    positive = [
        (bits, state) for bits, (weight, state) in branches.items() if weight > 0
    ]
    assert len(positive) == 1
    outcomes, expected_state = positive[0]
    assert expected_state is not None
    assert all(state is None for weight, state in branches.values() if weight == 0)
    assert abs(branches[outcomes][0] - 1) <= _ATOL
    probabilities: list[torch.Tensor] = []
    original = torch.multinomial

    def observe(values: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        probabilities.append(values.detach().clone().reshape(-1))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", observe)
    result = run_dynamic(
        _direct((wire, None), feedback=True, encoded_plus=True),
        shots=1,
        seed=53,
        strategy="trajectory",
    )
    assert result.classical_bits.tolist() == [list(outcomes)]
    # Each measurement is followed by a reset measurement; final sampling is last.
    assert len(probabilities) == 9
    for i, bit in enumerate(outcomes):
        expected = torch.tensor([1 - bit, bit], dtype=torch.float64)
        for measured in probabilities[2 * i : 2 * i + 2]:
            torch.testing.assert_close(measured, expected, atol=_ATOL, rtol=0)
    torch.testing.assert_close(
        probabilities[-1], expected_state.abs().square(), atol=_ATOL, rtol=0
    )
    for actual in result.final_states.reshape(-1, 32):
        torch.testing.assert_close(
            _density(actual), _density(expected_state), atol=_ATOL, rtol=0
        )
        assert abs(float(actual.abs().square().sum()) - 1) <= _ATOL
    decoded = run_dynamic(
        _direct((wire, None), feedback=True, encoded_plus=True, undo=True),
        shots=2,
        seed=53,
        strategy="trajectory",
    )
    assert decoded.samples[:, :3].tolist() == [[0, 0, 0], [0, 0, 0]]


def test_relative_phase_and_missing_reset_mutations_are_detected() -> None:
    changed = run_dynamic(
        _direct((None, None), feedback=True, encoded_plus=True, dephase=True),
        shots=2,
        seed=53,
        strategy="trajectory",
    )
    expected = next(
        state for weight, state in _projector_branches(None).values() if weight > 0
    )
    assert expected is not None
    for actual in changed.final_states.reshape(-1, 32):
        # Identical computational-basis probabilities must not hide a relative phase.
        torch.testing.assert_close(
            actual.abs().square(), expected.abs().square(), atol=_ATOL, rtol=0
        )
        assert not torch.allclose(
            _density(actual), _density(expected), atol=_ATOL, rtol=0
        )
    changed_readout = run_dynamic(
        _direct(
            (None, None), feedback=True, encoded_plus=True, dephase=True, undo=True
        ),
        shots=2,
        seed=53,
        strategy="trajectory",
    )
    assert changed_readout.samples[:, :3].tolist() == [[1, 0, 0], [1, 0, 0]]
    bad_reset = run_dynamic(_direct((0, None), omit_reset=True), shots=1, seed=53)
    assert bad_reset.classical_bits.tolist() != [[1, 0, 1, 0]]
    memory = build_memory_circuit(RepetitionCode(3), rounds=2)
    with pytest.raises(ValueError):
        _measurement_plan(
            replace(
                memory,
                source=memory.source.replace(
                    "qp.measure(wires=3)", "qp.measure(wires=4)"
                ),
            )
        )


def test_invalid_dependencies_measurements_frames_and_events_fail_early() -> None:
    with pytest.raises(ValidationError):
        DynamicCircuit(5).conditional("x", 0, classical_bit=3)
    ref = MeasurementRef(0, 3)
    with pytest.raises(ValueError):
        MeasurementSamples((ref, ref), torch.zeros((1, 2), dtype=torch.int8))
    record = MeasurementSamples((ref,), torch.zeros((1, 1), dtype=torch.int8))
    with pytest.raises(ValueError):
        record.outcome(MeasurementRef(1, 3))
    with pytest.raises(ValueError):
        PauliFrame().apply((0, 0))
    with pytest.raises(ValueError):
        PauliFrame().apply((0, 2, 0))
    with pytest.raises(ValueError):
        SyndromeRound(0, (0, 2))
    with pytest.raises(ValueError):
        run_repetition_memory_experiment(
            rounds=2, error_schedule=ErrorSchedule((ErrorEvent(2, 0),))
        )


@pytest.mark.parametrize("mode", ("runtime_decoder", "runtime_pauli_frame"))
def test_runtime_decoder_preserves_explicit_batched_refusal(mode: str) -> None:
    with pytest.raises(
        ValueError, match="decoder feedback is available only with trajectory execution"
    ):
        run_repetition_memory_experiment(
            rounds=2, shots=2, feedback_mode=mode, strategy="batched"
        )
