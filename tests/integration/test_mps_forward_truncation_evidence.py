"""What the distributed MPS forward reports about an unmeasured truncation step.

`FQ_MPS_FIXED_RANK_QR` records a step's discarded weight as `nan` when the
weight cannot be computed, and the rank-owned executor's own split path cannot
produce that today: it splits through one SVD that refuses non-finite factors.
The two sites exercised here still read the weight, and both read it with the
polarity `nan` defeats -- `weight > bound` is false for every bound, so an
unmeasured total passed the budget policy that exists to stop an overspend and
was reported as an exact state. This module pins the reading at the executor's
evidence surface, with the unmeasured weight injected at the split that produces
it, so the real gather, summing, and summary path runs.
"""

from __future__ import annotations

import math
import socket

import pytest

import flagquantum as fq
from flagquantum.runtime.executors.mps import forward
from flagquantum.runtime.executors.mps.forward import (
    execute_torch_distributed_mps_forward,
)

pytestmark = pytest.mark.integration

UNMEASURED = float("nan")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def single_rank_group() -> None:
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("a gloo process group is unavailable")
    if dist.is_initialized():
        pytest.skip("another process group owns this interpreter")
    dist.init_process_group(
        backend="gloo",
        world_size=1,
        rank=0,
        init_method=f"tcp://127.0.0.1:{_free_port()}",
    )
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _workload() -> fq.Circuit:
    """A circuit whose rank-local splits truncate, so every split records."""

    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).cx(1, 2).cx(2, 3)
    return circuit


@pytest.fixture
def unmeasured_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report every rank-local two-site split's weight as never measured.

    This is the value the fixed-rank range QR route writes. Injecting it at the
    producer keeps the gather, the summing, the policy checks, and the summary
    under test, instead of replacing any of them.
    """

    real = forward._apply_rank_local_instruction

    def split_without_a_measurable_weight(*args, **kwargs):
        outputs, split_info = real(*args, **kwargs)
        if split_info is not None:
            split_info = {**dict(split_info), "discarded_weight": UNMEASURED}
        return outputs, split_info

    monkeypatch.setattr(
        forward, "_apply_rank_local_instruction", split_without_a_measurable_weight
    )


def test_enforce_refuses_an_unmeasured_truncation_error(
    single_rank_group: None, unmeasured_split: None
) -> None:
    """`nan > 0.0` is false, so the budget policy used to wave this run through."""

    with pytest.raises(RuntimeError, match="global_error_budget") as refusal:
        execute_torch_distributed_mps_forward(
            _workload(),
            max_bond=1,
            global_error_budget=0.0,
            error_budget_policy="enforce",
        )

    assert "not a finite truncation error" in str(refusal.value)


def test_report_only_records_an_unmeasured_total_without_refusing(
    single_rank_group: None, unmeasured_split: None
) -> None:
    """The reporting policy still reports; it must not start refusing runs."""

    result = execute_torch_distributed_mps_forward(
        _workload(),
        max_bond=1,
        global_error_budget=0.0,
        error_budget_policy="report_only",
    )
    summary = result.summary()

    assert summary["error_budget_policy"] == "report_only"
    assert not math.isfinite(summary["truncation_error"])
    assert summary["error_budget_satisfied"] is False
    assert summary["scalability_claim_allowed"] is False


def test_an_unmeasured_total_is_not_reported_as_an_exact_state(
    single_rank_group: None, unmeasured_split: None
) -> None:
    """`nan > 0.0` is false, so the summary used to claim "exact"."""

    result = execute_torch_distributed_mps_forward(_workload(), max_bond=1)

    assert result.summary()["truncation_semantics"] == "approximate"


def test_exact_truncation_gradients_refuse_an_unmeasured_split(
    single_rank_group: None, unmeasured_split: None
) -> None:
    """An unmeasured weight cannot satisfy a policy against truncated forwards."""

    with pytest.raises(
        RuntimeError, match="exact truncation-gradient policy cannot describe"
    ):
        execute_torch_distributed_mps_forward(
            _workload(), max_bond=1, truncation_gradient_policy="exact"
        )


def test_a_measured_truncation_is_reported_and_not_refused(
    single_rank_group: None,
) -> None:
    """The guard refuses the unmeasured case, not every truncating run."""

    result = execute_torch_distributed_mps_forward(
        _workload(), max_bond=1, global_error_budget=1.0, error_budget_policy="enforce"
    )
    summary = result.summary()

    assert math.isfinite(summary["truncation_error"])
    assert summary["truncation_error"] > 0.0
    assert summary["truncation_semantics"] == "approximate"
    assert summary["error_budget_satisfied"] is True


def test_a_run_without_truncation_still_reports_an_exact_state(
    single_rank_group: None,
) -> None:
    """A total measured at zero is what "exact" is allowed to name."""

    result = execute_torch_distributed_mps_forward(
        _workload(), max_bond=64, global_error_budget=0.0, error_budget_policy="enforce"
    )
    summary = result.summary()

    assert summary["truncation_error"] == 0.0
    assert summary["truncation_semantics"] == "exact"
    assert summary["error_budget_satisfied"] is True
