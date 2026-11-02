"""Scientific aggregation preserves the frozen corpus and refuses missing data."""

import copy
import gzip
import json
from pathlib import Path

import pytest

from benchmarks.module_execution_matrix import aggregate_process_trials

pytestmark = pytest.mark.benchmark_contract
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "artifacts/development/module_snapshot_cpu_20261102"


def summaries():
    return {
        phase: [
            json.loads(
                gzip.decompress(
                    (EVIDENCE / phase / f"trial-{trial}.json.gz").read_bytes()
                )
            )
            for trial in range(3)
        ]
        for phase in ("before", "after")
    }


def test_existing_frozen_trials_regenerate_without_changing_budgets_or_samples():
    assert aggregate_process_trials(summaries()) == json.loads(
        (EVIDENCE / "aggregate.json").read_text()
    )


@pytest.mark.parametrize("damage", ("case", "demand", "sample", "failure"))
def test_incomplete_or_corrupted_trial_cannot_disappear_from_the_report(damage):
    payload = copy.deepcopy(summaries())
    case = payload["after"][1]["cases"][0]
    if damage == "case":
        payload["after"][1]["cases"].pop()
    elif damage == "demand":
        case["measurements"].pop()
    elif damage == "sample":
        case["measurements"][0]["seconds_per_call"][0] = float("nan")
    else:
        case["status"] = "timeout"
    with pytest.raises(ValueError):
        aggregate_process_trials(payload)
