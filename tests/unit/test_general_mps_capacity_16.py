from __future__ import annotations

import pytest

from benchmarks.internal.evidence.general_mps_capacity_16 import (
    MAX_BOND,
    N_SITES,
    TARGET_WORLD,
    logical_mps_bytes,
    target_boundaries,
    topology_fingerprint,
)


@pytest.mark.unit
def test_two_node_capacity_target_exceeds_two_hundred_gibibytes():
    logical_bytes = logical_mps_bytes(N_SITES, MAX_BOND)
    assert logical_bytes > 200 << 30
    assert logical_bytes // TARGET_WORLD < 16 << 30


@pytest.mark.unit
def test_two_node_capacity_covers_all_fifteen_boundaries():
    boundaries = target_boundaries()
    assert len(boundaries) == 15
    assert boundaries[0] == 1535
    assert boundaries[-1] == 23_039
    assert len(topology_fingerprint()) == 64
