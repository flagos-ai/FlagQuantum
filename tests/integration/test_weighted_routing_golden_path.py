from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

#: The two routes the example prints. They are pinned because the whole claim of
#: the example is that a hop count cannot tell them apart: both are two hops
#: between qubits 1 and 3 on this device, and both cost the same two SWAPs. Only
#: the couplings they land on differ, so a change that made the hop count decide
#: would change this line without changing any count beside it.
_ROUTES = "routes between qubits 1 and 3: (1, 0, 3) and (1, 2, 3)"

#: What each router emits. The SWAP count is identical; the SWAP links are not.
_HOP_COUNT_PROGRAM = "[('h', (1,)), ('swap', (1, 0)), ('cx', (0, 3)), ('swap', (1, 0))]"
_CALIBRATED_PROGRAM = (
    "[('h', (1,)), ('swap', (1, 2)), ('cx', (2, 3)), ('swap', (1, 2))]"
)


def _run_example() -> str:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "examples.weighted_routing"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_weighted_routing_user_example_runs_end_to_end() -> None:
    stdout = _run_example()

    assert "FlagQuantum weighted routing check passed" in stdout
    assert _ROUTES in stdout
    assert f"connecting couplings only: {_HOP_COUNT_PROGRAM}" in stdout
    assert (
        f"connecting those couplings with their error: {_CALIBRATED_PROGRAM}" in stdout
    )

    # The counts are the point: they are equal, so nothing the router reported
    # before this change could have chosen between the two programs.
    assert "inserted swaps: 2 on both, so a count cannot choose" in stdout
    assert "expected link faults: 0.003 against 0.04 avoided" in stdout
    assert "estimate before routing: 0.003" in stdout
    assert "automatic choice: restore_after_each_gate" in stdout
    assert "minimize_expected_link_faults_then_planned_inserted_swaps" in stdout
    # A refusal is the example's answer for a planner that would not read the
    # calibration, so it is part of the expected output rather than a warning.
    assert "cannot route a device that states a per-link error rate" in stdout
