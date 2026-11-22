"""Close original lane counter projection through the shared execution loop."""

import json
from cmath import exp
from collections import Counter, defaultdict
from math import pi
from pathlib import Path
from random import Random

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats._physical_operation_statistics import (
    PhysicalOperationStatistics,
)
from flagquantum.compiler.formats.kmb import KmxRecord
from flagquantum.compiler.formats.kmx_circuit import read_kmx_circuit
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.physical_basis_shots import (
    PackedBasisState,
    apply_packed_basis_records,
    sample_basis_batch,
)
from flagquantum.simulation.physical_operation_counters import PhysicalOperationCounters
from tests.unit.circuit_asset_fixtures import instructions
from tests.unit.fixtures.generate_operation_statistics import snapshot

pytestmark = pytest.mark.unit
_DATA = json.loads(
    (Path(__file__).with_name("data") / "operation_statistics.json").read_text()
)


@pytest.mark.parametrize("case", _DATA["batches"])
def test_all_source_lanes_chunks_and_aggregate(case: dict[str, object]) -> None:
    size, text, words, expected, lanes = (
        case[name] for name in ("size", "kmx", "words", "total", "lane_indices")
    )
    assert isinstance(size, int) and isinstance(text, str)
    assert isinstance(words, list) and isinstance(lanes, list)
    for word in words:
        assert isinstance(word, int)
    records = read_kmx_circuit(text)
    state = PackedBasisState(
        defaultdict(int), defaultdict(int, enumerate(words)), [0] * size
    )
    plain = PackedBasisState(
        defaultdict(int), defaultdict(int, enumerate(words)), [0] * size
    )
    tally = PhysicalOperationCounters(size)
    aggregate: Counter[str] = Counter()
    apply_packed_basis_records(
        records, state, lambda: 0, counts=aggregate, observer=tally.observe
    )
    apply_packed_basis_records(records, plain, lambda: 0)
    assert state == plain
    counts = PhysicalOperationStatistics.from_counts(*tally.snapshot())
    assert snapshot(counts) == expected
    assert sum(aggregate.values()) == counts.num_instructions
    snapshots = case["lane_snapshots"]
    assert isinstance(snapshots, list)
    for index, snapshot_index in enumerate(lanes):
        assert isinstance(snapshot_index, int)
        expected_lane = snapshots[snapshot_index]
        assert (
            snapshot(PhysicalOperationStatistics.from_counts(*tally.snapshot(index)))
            == expected_lane
        )
    # Caller-owned state replacement has no effect on cumulative counter state.
    state = PackedBasisState(defaultdict(int), defaultdict(int), [0] * size)
    assert (
        snapshot(PhysicalOperationStatistics.from_counts(*tally.snapshot())) == expected
    )
    apply_packed_basis_records(
        read_kmx_circuit("X q0"), state, lambda: 0, observer=tally.observe
    )
    assert (
        PhysicalOperationStatistics.from_counts(*tally.snapshot())["X"]
        == counts["X"] + size
    )
    unchanged = state.qubits.copy()
    tally.clear()
    assert not PhysicalOperationStatistics.from_counts(*tally.snapshot())
    assert state.qubits == unchanged
    assert snapshot(counts) == expected


def test_sample_entry_observer_feedback_and_all_measurement_histories() -> None:
    records = read_kmx_circuit(
        "X q0\nHMR q0 b0\nX q1 if b0\nZ_POW q1 0.25 if b0\nHMR q1 b1\n"
    )
    expected_states: dict[tuple[int, int], torch.Tensor] = {}
    for outcomes in range(4):
        tally = PhysicalOperationCounters(1)
        draws = [outcomes & 1, outcomes >> 1]
        observed = sample_basis_batch(
            records,
            shots=1,
            rng=Random(3),
            outcome_mask=iter(draws).__next__,
            observer=tally.observe,
        )
        plain = sample_basis_batch(
            records, shots=1, rng=Random(3), outcome_mask=iter(draws).__next__
        )
        assert observed == plain
        assert all(value == 0 for value in observed.qubits.values())
        stats = PhysicalOperationStatistics.from_counts(*tally.snapshot())
        assert stats["X"] == 1 + (outcomes & 1)
        assert stats.num_z_pow() == outcomes & 1
        assert stats.num_t_like == outcomes & 1
        assert stats["HMR"] == 2
        first, second = outcomes & 1, outcomes >> 1
        phase = exp(1j * pi * (first + first * second + first * 0.25))
        expected = torch.tensor([phase, 0, 0, 0], dtype=torch.complex128)
        expected_states[first, second] = expected
        actual_phase = exp(2j * pi * observed.phase_units[0] / (1 << 128))
        assert abs(actual_phase - phase) < 1e-12
    dynamic = DynamicCircuit(2, dtype=torch.complex128)
    dynamic.append(FieldOperation(2, instructions(records)))
    result = run_dynamic(dynamic, shots=64, seed=92, strategy="batched")
    actual_states = result.final_states.reshape(64, -1)
    seen = set()
    for index, values in enumerate(result.classical_bits.tolist()):
        branch = (int(values[0]), int(values[1]))
        seen.add(branch)
        torch.testing.assert_close(
            actual_states[index], expected_states[branch], atol=1e-12, rtol=0
        )
    assert seen == set(expected_states)
    assert not torch.count_nonzero(actual_states[:, 1:])


def test_disabled_invalid_bounds_and_wrap() -> None:
    for size in (-1, True):
        with pytest.raises(ValueError, match="shot count"):
            PhysicalOperationCounters(size)
    disabled = PhysicalOperationCounters(1, enabled=False)
    disabled.observe(KmxRecord("X", (0,)), 1)
    assert not disabled.enabled
    with pytest.raises(ValueError, match="not enabled"):
        disabled.snapshot()
    with pytest.raises(AttributeError):
        setattr(disabled, "enabled", True)
    tally = PhysicalOperationCounters(1)
    for index in (-1, 1):
        with pytest.raises(IndexError, match="shot count"):
            tally.snapshot(index)
    for active in (-1, 2):
        with pytest.raises(ValueError, match="mask"):
            tally.observe(KmxRecord("X", (0,)), active)
    tally._operations["X"] = [(1 << 64) - 1]
    tally.observe(KmxRecord("X", (0,)), 1)
    assert not PhysicalOperationStatistics.from_counts(*tally.snapshot())


def test_existing_execution_rejections_with_observer() -> None:
    tally = PhysicalOperationCounters(1)
    with pytest.raises(ValueError, match="Unsupported"):
        sample_basis_batch(
            [KmxRecord("UNKNOWN")], shots=1, rng=Random(0), observer=tally.observe
        )
    for shots in (0, 65):
        with pytest.raises(ValueError, match="1 to 64"):
            sample_basis_batch([], shots=shots, rng=Random(0), observer=tally.observe)
