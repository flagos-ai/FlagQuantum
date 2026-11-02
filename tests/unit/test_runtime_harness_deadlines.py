"""A wrapper deadline must outlast the runtime budget it wraps.

Distributed training harnesses run their scripts under ``torchrun`` inside a
subprocess, and give that subprocess a wall-clock deadline. The script has
its own deadlines, and it asserts its own boundedness from the inside: the
surviving rank of an abrupt loss raises once it has waited longer than the
launcher should ever need, and a training run reports ``collective_timeout``
once its own budget is spent.

Those inner assertions are the claim. They can only speak if the wrapper lets
them, and a wrapper deadline shorter than the inner budget replaces the claim
with ``subprocess.TimeoutExpired`` instead -- which is what a loaded host
produces, and what makes this look like a product defect rather than a harness
one. These tests pin the ordering, so lowering a wrapper deadline below the
budget it wraps fails here rather than as an unexplained timeout under load.
"""

import subprocess
from pathlib import Path

import pytest

from tests.distributed import mps_stability_fault_runtime as mps_runtime
from tests.distributed import statevector_abrupt_failure_runtime as abrupt_runtime
from tests.distributed import statevector_training_runtime as training_runtime
from tests.distributed import test_abrupt_process_loss as abrupt_harness
from tests.distributed import test_mps_stability_faults as mps_harness
from tests.distributed import test_training_runtime as training_harness

pytestmark = pytest.mark.unit


def test_training_wrapper_outlasts_the_script_budget_for_every_mode():
    """Each mode's deadline is its own budget plus the harness's startup slack."""

    for mode in training_harness.MODES:
        inner = training_runtime.mode_inner_budget_seconds(mode)
        outer = training_harness.subprocess_deadline_seconds(mode)

        assert outer >= inner + training_harness.STARTUP_ALLOWANCE_SECONDS, mode


def test_the_startup_allowance_covers_the_measured_startup():
    """The slack is the whole point, so it must not shrink below what was seen.

    Without this the ordering above still holds at an allowance of zero, which
    puts the deadline back on the inner budget and brings the false failure back.
    """

    for harness in (training_harness, abrupt_harness):
        assert (
            harness.STARTUP_ALLOWANCE_SECONDS
            >= harness.MEASURED_STARTUP_UNDER_LOAD_SECONDS
        )


def test_every_mode_has_a_decided_budget():
    """A mode added without deciding its budget must fail, not inherit a guess."""

    with pytest.raises(KeyError):
        training_runtime.mode_inner_budget_seconds("no_such_mode")


def test_abrupt_loss_wrapper_outlasts_the_survivor_wait():
    inner = abrupt_runtime.PEER_LOSS_TIMEOUT_SECONDS
    outer = abrupt_harness.SUBPROCESS_DEADLINE_SECONDS

    assert outer >= inner + abrupt_harness.STARTUP_ALLOWANCE_SECONDS


def test_the_deliberate_timeout_mode_is_the_only_one_with_a_short_budget():
    """`--mode timeout` asks for a timeout; every other mode must not inherit it."""

    short = [
        mode
        for mode in training_harness.MODES
        if training_runtime.mode_inner_budget_seconds(mode)
        <= training_runtime.DELIBERATE_TIMEOUT_SECONDS
    ]

    assert short == ["timeout"]


def test_mps_failure_wrapper_outlasts_startup_setup_and_injected_timeout() -> None:
    """The wrapper must let both startup and the intentional fault speak."""
    assert (
        mps_harness.STARTUP_ALLOWANCE_SECONDS
        >= training_harness.STARTUP_ALLOWANCE_SECONDS
    )
    inner = (
        2 * mps_runtime.SETUP_COLLECTIVE_TIMEOUT_SECONDS
        + mps_runtime.FAULT_COLLECTIVE_TIMEOUT_SECONDS
        + mps_runtime.FAULT_DELAY_SECONDS
    )
    assert (
        inner + mps_harness.STARTUP_ALLOWANCE_SECONDS
        <= mps_harness.PARTICIPANT_FAILURE_DEADLINE_SECONDS
    )


def test_mps_fault_group_keeps_its_short_internal_deadline() -> None:
    """Additional wrapper slack must not extend the deliberate fault itself."""
    assert mps_runtime.SETUP_COLLECTIVE_TIMEOUT_SECONDS == 30
    assert mps_runtime.FAULT_COLLECTIVE_TIMEOUT_SECONDS == 3
    assert mps_runtime.FAULT_DELAY_SECONDS == 5


def test_mps_harness_passes_the_budget_to_the_real_launcher_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    deadlines: list[float] = []

    def capture(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        timeout = kwargs["timeout"]
        assert isinstance(timeout, (int, float))
        deadlines.append(float(timeout))
        return subprocess.CompletedProcess(
            [], 1, stdout="", stderr="injected rank exception"
        )

    monkeypatch.setattr(mps_harness.subprocess, "run", capture)
    mps_harness.test_participant_failure_terminates_launcher("rank_exception", tmp_path)
    assert deadlines == [mps_harness.PARTICIPANT_FAILURE_DEADLINE_SECONDS]
