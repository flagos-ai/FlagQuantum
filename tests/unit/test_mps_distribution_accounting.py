"""Cross-rank MPS accounting must report the same numbers it always did.

`global_mps_tensor_bytes`, `global_mps_bond_dimensions` and
`canonicalize_rank_owned_mps` report per-rank tables. They now assemble those
tables on the device from host-known integers instead of writing each field into
a device vector element by element, which changes how the collectives are fed
but must not change what they return. This module pins the returned values
against the same state read directly and pins the two documented accounting
constants, so a rewrite that transposed a column or dropped an unowned wire
fails here rather than in an artifact.

The collective under test is only exercised at world size one, because these
tests run wherever the suite runs. What that still covers is the value mapping:
which wire's footprint lands in which position, and which counter lands in
which column.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist

from flagquantum.runtime.executors.mps.canonicalization import (
    canonicalize_rank_owned_mps,
)
from flagquantum.runtime.executors.mps.distribution import (
    global_mps_bond_dimensions,
    global_mps_tensor_bytes,
    rebalance_mps_if_needed,
)
from flagquantum.runtime.executors.mps.state import (
    RankOwnedMPSState,
    initial_mps_ownership,
)
from flagquantum.simulation.mps.models import MPSConfig
from flagquantum.simulation.mps.rank_local import tensor_nbytes
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit

N_WIRES = 4
MAX_BOND = 4
#: The three collectives `canonicalize_rank_owned_mps` adds to the message
#: count after its sweeps: the metrics all-gather, the norm broadcast and the
#: residual all-reduce.
ACCOUNTING_MESSAGES = 3
#: One int64 per counter, one float64 per state norm and one float64 for the
#: residual, all with batch size one.
ACCOUNTING_BYTES = 3 * 8 + 1 * 8 + 8


@pytest.fixture
def process_group() -> Iterator[None]:
    """A world-size-one group, reused when the suite already made one."""

    if dist.is_initialized():
        yield
        return
    dist.init_process_group(
        backend="gloo", store=dist.HashStore(), rank=0, world_size=1
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _owned_state(*, drop: int | None = None) -> RankOwnedMPSState:
    """One rank owning every site of a four-wire state."""

    generator = torch.Generator().manual_seed(20261003)
    vector = torch.randn(2**N_WIRES, dtype=torch.complex128, generator=generator)
    vector = vector / torch.linalg.vector_norm(vector)
    reference = MPSState.from_statevector(
        vector, N_WIRES, config=MPSConfig(max_bond=MAX_BOND)
    )
    ownership = initial_mps_ownership(N_WIRES, 1)
    sites = {wire: reference.tensors[wire] for wire in ownership[0]}
    if drop is not None:
        del sites[drop]
    return RankOwnedMPSState(
        n_wires=N_WIRES,
        bsz=1,
        rank=0,
        world_size=1,
        config=MPSConfig(max_bond=MAX_BOND),
        local_tensors=sites,
        ownership=ownership,
    )


def test_tensor_bytes_land_on_the_wire_that_owns_them(process_group: None) -> None:
    state = _owned_state()

    assert global_mps_tensor_bytes(state) == tuple(
        tensor_nbytes(state.local_tensors[wire]) for wire in range(N_WIRES)
    )


def test_an_unowned_wire_reports_no_bytes(process_group: None) -> None:
    state = _owned_state(drop=2)

    sizes = global_mps_tensor_bytes(state)

    assert sizes[2] == 0
    assert all(sizes[wire] > 0 for wire in (0, 1, 3))


def test_bond_dimensions_land_on_the_bond_that_owns_them(
    process_group: None,
) -> None:
    state = _owned_state()

    assert global_mps_bond_dimensions(state) == tuple(
        int(state.local_tensors[wire].shape[3]) for wire in range(N_WIRES - 1)
    )


def test_an_unowned_site_reports_no_bond(process_group: None) -> None:
    state = _owned_state(drop=1)

    dimensions = global_mps_bond_dimensions(state)

    assert dimensions[1] == 0
    assert dimensions[0] > 0


def test_an_unbounded_threshold_does_not_rebalance(process_group: None) -> None:
    """An infinite threshold cannot be crossed, so it migrates nothing.

    Compiled site kernels require exactly this threshold, and their buckets are
    sized for the plan, so the decision is settled before any footprint is
    collected. The ownership is compared so a future early return that also
    moved sites would fail here.
    """

    state = _owned_state()
    before = state.ownership

    assert rebalance_mps_if_needed(state, float("inf")) == (False, 0, 0)
    assert state.ownership == before


def test_canonicalization_reports_each_counter_in_its_own_column(
    process_group: None,
) -> None:
    state = _owned_state()

    metrics = canonicalize_rank_owned_mps(state)

    assert metrics.center == N_WIRES - 1
    assert metrics.messages_by_rank == (ACCOUNTING_MESSAGES,)
    assert metrics.bytes_by_rank == (ACCOUNTING_BYTES,)
    # The canonicalization sweeps touch site tensors far larger than the three
    # scalars they report, so the temporary column has to be the largest of the
    # three; a transposed column would put the scalar here instead.
    assert metrics.temporary_bytes_by_rank[0] > metrics.bytes_by_rank[0]
    assert len(metrics.state_norms) == state.bsz
    assert all(value > 0.0 for value in metrics.state_norms)
    assert metrics.residual >= 0.0
