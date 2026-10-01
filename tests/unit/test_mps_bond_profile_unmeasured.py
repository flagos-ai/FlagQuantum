"""What an MPS bond profile reports about a truncation step it never measured.

The fixed-rank range QR route records `nan` as a step's discarded weight,
because the weight is not computable without materializing the matrix that
route exists to avoid. `planning.py` already reads that sentinel correctly when
it plans bond growth -- an unmeasured bond becomes a growth reason rather than a
cold bond. The profile's maximum was the exception: `max` over a sequence
holding `nan` answers with whichever element it reached first, so a measured
step recorded before an unmeasured one produced a finite maximum and the
unmeasured step vanished from the reported evidence.

The real producer is the opt-in `FQ_MPS_FIXED_RANK_QR` route, and the second
test below drives it rather than stubbing a record, so the claim that this is
reachable is checked by the suite.
"""

from __future__ import annotations

import math

import pytest

import flagquantum as fq
import flagquantum.simulation.mps as fqmps
from flagquantum.simulation.mps.models import MPSTruncationRecord

pytestmark = pytest.mark.unit


def _measured_record(bond: int, discarded_weight: float) -> MPSTruncationRecord:
    return MPSTruncationRecord(
        bond=bond,
        kept_rank=1,
        original_rank=2,
        discarded_weight=discarded_weight,
        max_bond=1,
        cutoff=0.0,
    )


def _unmeasured_record(bond: int) -> MPSTruncationRecord:
    """A record whose weight the fixed-rank range QR route cannot compute."""

    return MPSTruncationRecord(
        bond=bond,
        kept_rank=1,
        original_rank=2,
        discarded_weight=float("nan"),
        max_bond=1,
        cutoff=0.0,
        source="fixed_rank_range_qr_unmeasured",
    )


def _state_with(records: list[MPSTruncationRecord]):
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 1).cx(1, 2).cx(2, 3)
    mps = fqmps.run_mps(circuit, max_bond=1)
    mps.truncation_records = records
    mps.truncation_errors = [record.discarded_weight for record in records]
    return mps


def test_a_measured_step_before_an_unmeasured_one_does_not_hide_it():
    """`max` answers with the first element it reached, so order decided this.

    The measured step comes first precisely because that is the order in which
    it disappeared: with the unmeasured record last, `max` never reached it.
    """

    mps = _state_with([_measured_record(0, 1e-15), _unmeasured_record(1)])

    assert not math.isfinite(mps.bond_profile().max_truncation_error)
    assert not math.isfinite(mps.summary()["max_truncation_error"])


def test_an_unmeasured_step_last_is_still_reported_as_unmeasured():
    """The other order is already non-finite; both orders must agree."""

    mps = _state_with([_unmeasured_record(0), _measured_record(1, 1e-15)])

    assert not math.isfinite(mps.bond_profile().max_truncation_error)
    assert not math.isfinite(mps.summary()["max_truncation_error"])


def test_a_fully_measured_profile_keeps_its_measured_maximum():
    """Reporting "unmeasured" must not turn every measured maximum into `nan`."""

    mps = _state_with([_measured_record(0, 1e-15), _measured_record(1, 3e-15)])

    assert mps.bond_profile().max_truncation_error == 3e-15
    assert mps.summary()["max_truncation_error"] == 3e-15


def test_a_profile_with_no_truncation_step_reports_a_measured_zero():
    mps = _state_with([])

    assert mps.bond_profile().max_truncation_error == 0.0


def test_the_fixed_rank_qr_route_reaches_an_unmeasured_maximum(monkeypatch):
    """The sentinel's real producer, so the reachability claim is checked here.

    This circuit reaches the fixed-rank range QR route, which records a `nan`
    weight for the bonds it cannot measure and a real one for the bonds it can.
    The reported maximum must be non-finite, because one of the steps in that
    list was never measured.
    """

    monkeypatch.setenv("FQ_MPS_FIXED_RANK_QR", "1")
    circuit = fq.Circuit(5)
    circuit.rzz(2, 4, -0.43971031456547105)
    circuit.ry(1, -1.2851345274295358)
    circuit.ry(0, -0.6813140503080439)
    circuit.rzz(1, 0, -0.9327709975806087)
    circuit.cx(2, 1)
    circuit.cx(1, 4)

    mps = fqmps.run_mps(circuit, max_bond=2)

    assert mps.fixed_rank_qr_regions > 0
    recorded = list(mps.truncation_errors)
    assert recorded, "the circuit must record at least one truncation step"
    assert any(
        not math.isfinite(error) for error in recorded
    ), "the circuit must reach an unmeasured step, or it cannot witness this"
    assert not math.isfinite(mps.summary()["max_truncation_error"])
