"""Contract tests for the two-node CUDA tensor-network probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "tools" / "probe_cuda_multinode_tn.py"
_SPEC = importlib.util.spec_from_file_location("cuda_multinode_tn_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_the_two_rank_pair_is_the_only_placement_the_probe_accepts() -> None:
    good = {
        "world_size": 2,
        "local_world_size": 1,
        "node_count": 2,
        "distribution_semantics": "sharded_across_ranks",
        "claim_evidence_type": "production_runtime",
    }

    _MODULE._require_two_node_placement(good, leg="test")

    for wrong in (
        # One node per rank is the claim; two ranks on one node is a different
        # claim that this probe does not make.
        {**good, "local_world_size": 2, "node_count": 1},
        # A single rank would validate the arithmetic and nothing else.
        {**good, "world_size": 1, "node_count": 1},
        # A bespoke semantics string would fail the shared evidence contract.
        {**good, "distribution_semantics": "local_simulated_slice_parallel"},
        # A CPU collective reduces correctly but is not accelerator transport,
        # so an artifact that recorded one would not be accelerator evidence.
        {**good, "claim_evidence_type": "development_smoke"},
    ):
        with pytest.raises(RuntimeError):
            _MODULE._require_two_node_placement(wrong, leg="test")


def test_the_slice_plan_is_the_declared_partition_and_not_a_scale_claim() -> None:
    summary = {
        "slice_tasks": _MODULE.EXPECTED_SLICE_COUNT,
        "slice_labels": list(_MODULE.SLICED_LABELS),
        "tasks_by_rank": {0: _MODULE.EXPECTED_SLICES_PER_RANK}
        | {1: _MODULE.EXPECTED_SLICES_PER_RANK},
        "scalability_claim_allowed": False,
    }

    _MODULE._require_slice_partition(summary, leg="test")

    for wrong in (
        # One slice per rank would mean each rank is a whole contraction, which
        # is the replicated-execution shape the reduction exists to replace.
        {**summary, "slice_tasks": 2},
        # A different cut is a different partition; the artifact declares this
        # one, so a run that took another path is not the recorded evidence.
        {**summary, "slice_labels": [1, 2]},
        # Uneven ownership is a legitimate plan, but not this probe's scope.
        {**summary, "tasks_by_rank": {0: 3, 1: 1}},
        # The probe must never let a two-node pair look like a scale result.
        {**summary, "scalability_claim_allowed": True},
    ):
        with pytest.raises(RuntimeError):
            _MODULE._require_slice_partition(wrong, leg="test")


def test_the_sliced_labels_are_a_cut_carried_by_two_nodes() -> None:
    multiplicities = _MODULE._require_sliced_labels_are_a_cut()

    # Each sliced label has to appear on two nodes, or slicing it splits nothing.
    assert multiplicities == dict.fromkeys(_MODULE.SLICED_LABELS, 2)


def test_the_measured_objective_is_not_degenerate() -> None:
    """A constant expectation would let a broken reduction agree with itself."""

    expectation, gradients = _MODULE._reference_expectation()

    assert 0.0 < abs(expectation) < 1.0
    # Both trainable parameters have to move the objective, or a gradient
    # reduction could drop one rank and still match the reference.
    assert all(value != 0.0 for value in gradients)
    # The observable spans the two wires the entangling gate joins, so the
    # contraction crosses the slice boundary rather than sitting inside a slice.
    assert _MODULE.OBSERVABLE_WIRES[0] == 0
    assert _MODULE.OBSERVABLE_WIRES[1] == _MODULE.N_WIRES - 1


def test_the_checked_bitstrings_are_distinct_and_non_zero() -> None:
    reference = _MODULE._reference_statevector()

    amplitudes = [
        reference[int(bitstring, 2)] for bitstring in _MODULE.CHECKED_BITSTRINGS
    ]
    assert all(value != 0 for value in amplitudes)
    # Four distinct values, so a reduction that returned one rank's slice for
    # every bitstring could not agree with the reference by coincidence.
    assert len({complex(value) for value in amplitudes}) == len(amplitudes)
