"""Focused contracts for the explicit distributed MPS full-state export.

The export exists so a workload that needs the whole matrix product state can ask
for it by name. These tests pin the three things that make it safe to publish: it
returns the logical state exactly, it reconstructs a sharded state from the
ownership map the ranks published, and it cannot be read as a scaling result.

Ranks are simulated by slicing one reference state's tensors rather than by
launching a process group: what the export has to get right is the placement of
each rank's block and the cost it reports, and both are decided by the block and
the map rather than by the transport that carried them. The real two-host path is
covered by the probe and its team test.
"""

from dataclasses import dataclass

import pytest
import torch

from flagquantum.runtime.executors.mps.errors import MPSFullMaterializationError
from flagquantum.runtime.executors.mps.gather import (
    BLOCKER_FULL_MPS_GATHER_IS_NOT_A_SCALING_RESULT,
    SITE_ORDER_CANONICAL_LOGICAL,
    _owned_wires,
    _place_gathered_sites,
    _site_table,
    _validated_ownership,
    export_distributed_mps,
)
from flagquantum.runtime.executors.mps.records import (
    TorchDistributedMPSForwardResult,
)
from flagquantum.runtime.executors.mps.state import (
    RankOwnedMPSState,
    initial_mps_ownership,
)
from flagquantum.simulation.mps.models import MPSConfig
from flagquantum.simulation.mps.state import MPSState

pytestmark = pytest.mark.unit

N_WIRES = 6


def _reference_state(n_wires: int = N_WIRES) -> MPSState:
    """A reference MPS with distinct, non-trivial site tensors."""

    generator = torch.Generator().manual_seed(20261003)
    state = torch.randn(
        2**n_wires, dtype=torch.complex128, generator=generator
    ).reshape(1, -1)
    state = state / torch.linalg.vector_norm(state)
    return MPSState.from_statevector(state, n_wires, config=MPSConfig(max_bond=8))


@dataclass
class _Placed:
    """A rank-owned state together with the reference it was sliced from."""

    state: RankOwnedMPSState
    ownership: tuple[tuple[int, ...], ...]
    reference: MPSState

    def result(self) -> TorchDistributedMPSForwardResult:
        return TorchDistributedMPSForwardResult(
            shard_state=self.state,
            local_gate_count=0,
            boundary_gate_count=0,
            boundary_messages=0,
            boundary_bytes=0,
            rebalance_messages=0,
            rebalance_bytes=0,
            rebalance_count=0,
            partition_history=(),
            bond_dimensions=tuple(self.reference.bond_dims),
            rank_tensor_bytes=tuple(
                sum(tensor.numel() * tensor.element_size() for tensor in shard)
                for shard in (
                    tuple(
                        self.reference.tensors[wire] for wire in self.ownership[owner]
                    )
                    for owner in range(len(self.ownership))
                )
            ),
            canonicalization=None,  # type: ignore[arg-type]
            truncation_records=(),
            global_error_budget=None,
            error_budget_policy="not_applied",
            truncation_gradient_policy="not_applied",
            backend="gloo",
            local_world_size=len(self.ownership),
        )


def _placed(world_size: int, *, reference: MPSState | None = None) -> _Placed:
    """Slice one reference state into `world_size` contiguous site blocks."""

    reference = reference or _reference_state()
    ownership = initial_mps_ownership(reference.n_wires, world_size)
    return _Placed(
        state=RankOwnedMPSState(
            n_wires=reference.n_wires,
            bsz=reference.bsz,
            rank=0,
            world_size=world_size,
            config=reference.config,
            local_tensors={
                wire: reference.tensors[wire].clone() for wire in ownership[0]
            },
            ownership=ownership,
        ),
        ownership=ownership,
        reference=reference,
    )


def _all_blocks(placed: _Placed) -> tuple[dict[int, torch.Tensor], ...]:
    """Every rank's sites, as the gather would find them on each rank."""

    return tuple(
        {wire: placed.reference.tensors[wire] for wire in wires}
        for wires in placed.ownership
    )


def _padded_blocks(
    blocks: tuple[dict[int, torch.Tensor], ...],
) -> tuple[tuple[tuple[int, tuple[int, ...]], ...], list[torch.Tensor]]:
    """The length tables and padded buffers one all-gather would hand back."""

    tables = tuple(_site_table(sorted(block), block) for block in blocks)
    elements = max(
        sum(int(torch.tensor(shape).prod()) for _, shape in table) for table in tables
    )
    reference = next(iter(blocks[0].values()))
    buffers = []
    for block in blocks:
        flat = torch.cat([block[wire].reshape(-1) for wire in sorted(block)])
        buffer = flat.new_zeros(elements)
        buffer[: flat.numel()] = flat
        buffers.append(buffer)
    assert reference.element_size() > 0
    return tables, buffers


class TestWorldSizeOne:
    """One rank already holds every site; the export must not perturb it."""

    def test_the_exported_state_contracts_to_the_single_device_statevector(
        self,
    ) -> None:
        placed = _placed(1)
        exported = export_distributed_mps(placed.result())
        torch.testing.assert_close(
            exported.state.to_statevector(),
            placed.reference.to_statevector(),
            atol=1e-9,
            rtol=1e-9,
        )

    def test_the_exported_state_carries_the_reference_site_tensors(self) -> None:
        placed = _placed(1)
        exported = export_distributed_mps(placed.result())
        assert exported.state.n_wires == placed.reference.n_wires
        for wire in range(placed.reference.n_wires):
            torch.testing.assert_close(
                exported.state.tensors[wire], placed.reference.tensors[wire]
            )

    def test_the_rank_owned_result_still_refuses_full_state_materialization(
        self,
    ) -> None:
        placed = _placed(1)
        with pytest.raises(MPSFullMaterializationError, match="forbids"):
            placed.result().full_state()

    def test_the_summary_states_the_export_cost_and_not_a_speedup(self) -> None:
        summary = export_distributed_mps(_placed(1).result()).summary()
        assert summary["full_state_materialization"] is True
        assert summary["site_order"] == SITE_ORDER_CANONICAL_LOGICAL
        assert summary["distribution_semantics"] == "replicated_per_rank"
        assert summary["scalability_claim_allowed"] is False
        assert summary["release_gate_allowed"] is False
        assert summary["blockers"] == (BLOCKER_FULL_MPS_GATHER_IS_NOT_A_SCALING_RESULT,)
        # A gather has no speedup to report, so the summary must not carry one.
        assert "speedup" not in " ".join(summary)
        assert summary["operation_semantics"] == "explicit_full_mps_gather"

    def test_the_export_publishes_its_bytes_and_site_ownership(self) -> None:
        placed = _placed(1)
        summary = export_distributed_mps(placed.result()).summary()
        assert summary["world_size"] == 1
        assert summary["node_count"] == 1
        assert summary["local_site_count"] == N_WIRES
        assert summary["site_ownership"] == placed.ownership
        # One rank moves nothing: the gather is over other ranks' blocks.
        assert summary["gather_bytes_per_rank"] == 0
        assert summary["total_gather_bytes"] == 0
        assert summary["full_state_bytes"] == (
            placed.reference.parameter_count
            * placed.reference.tensors[0].element_size()
        )
        assert summary["peak_resident_bytes_per_rank"] == (
            summary["full_state_bytes"] + summary["local_site_bytes"]
        )


class TestMultiRankPlacement:
    """The export must place each rank's block at the wires its map assigns."""

    @pytest.mark.parametrize("world_size", [2, 3, 6])
    def test_placement_reconstructs_the_logical_state(self, world_size: int) -> None:
        placed = _placed(world_size)
        tables, buffers = _padded_blocks(_all_blocks(placed))
        tensors = _place_gathered_sites(tables, buffers, ownership=placed.ownership)
        reconstructed = MPSState(list(tensors), config=placed.reference.config)
        torch.testing.assert_close(
            reconstructed.to_statevector(),
            placed.reference.to_statevector(),
            atol=1e-9,
            rtol=1e-9,
        )

    def test_placement_reads_each_ranks_own_length_table(self) -> None:
        """A smaller block's padding must not be read as one of its sites.

        The wire count is the same on every rank here, but the site shapes are
        not, so a placement that walked a fixed stride would take the next
        rank's padding for a site as soon as the first block was narrower.
        """

        placed = _placed(3)
        blocks = _all_blocks(placed)
        narrower = {
            wire: tensor[..., : max(1, tensor.shape[-1] // 2)]
            for wire, tensor in blocks[0].items()
        }
        blocks = (narrower, *blocks[1:])
        tables, buffers = _padded_blocks(blocks)
        assert tables[0] != tables[1]
        placed_state = _place_gathered_sites(
            tables, buffers, ownership=placed.ownership
        )
        assert len(placed_state) == N_WIRES
        for wire in placed.ownership[0]:
            torch.testing.assert_close(placed_state[wire], narrower[wire])

    def test_a_rank_that_contributed_the_wrong_sites_is_refused(self) -> None:
        placed = _placed(2)
        tables, buffers = _padded_blocks(_all_blocks(placed))
        swapped = (tables[1], tables[0])
        with pytest.raises(MPSFullMaterializationError, match="ownership map gives"):
            _place_gathered_sites(swapped, buffers, ownership=placed.ownership)


class TestRefusals:
    """Every refusal names a state the export cannot reconstruct."""

    def test_an_ownership_map_that_is_not_a_partition_is_refused(self) -> None:
        placed = _placed(2)
        broken = (placed.ownership[0], placed.ownership[0])
        state = _with_ownership(placed.state, broken)
        with pytest.raises(MPSFullMaterializationError, match="not a partition"):
            _validated_ownership(state)

    def test_an_ownership_map_with_an_empty_rank_is_refused(self) -> None:
        placed = _placed(2)
        broken = (placed.ownership[0] + placed.ownership[1], ())
        state = _with_ownership(placed.state, broken)
        with pytest.raises(MPSFullMaterializationError, match="no site"):
            _validated_ownership(state)

    def test_a_rank_holding_sites_the_map_does_not_give_it_is_refused(self) -> None:
        placed = _placed(2)
        state = _with_ownership(placed.state, placed.ownership)
        state.local_tensors = {
            wire: placed.reference.tensors[wire] for wire in placed.ownership[1]
        }
        with pytest.raises(MPSFullMaterializationError, match="ownership map gives"):
            _validated_ownership(state)

    def test_an_empty_shard_is_refused(self) -> None:
        placed = _placed(2)
        state = _with_ownership(placed.state, placed.ownership)
        state.local_tensors = {}
        with pytest.raises(MPSFullMaterializationError, match="owns no MPS site"):
            _owned_wires(state)

    def test_sites_that_disagree_about_their_dtype_are_refused(self) -> None:
        placed = _placed(2)
        state = _with_ownership(placed.state, placed.ownership)
        wire = placed.ownership[0][0]
        state.local_tensors[wire] = state.local_tensors[wire].to(torch.complex64)
        with pytest.raises(MPSFullMaterializationError, match="do not share one dtype"):
            _owned_wires(state)

    def test_a_group_smaller_than_the_state_is_refused(self) -> None:
        """The export must run on the group the state came from.

        Exporting a two-rank state while standing in a one-rank group would move
        no blocks and hand back an incomplete state under a canonical name, so the
        world sizes are compared before anything is exchanged.
        """

        placed = _placed(2)
        with pytest.raises(MPSFullMaterializationError, match="process group holds"):
            export_distributed_mps(placed.result())


def _with_ownership(
    state: RankOwnedMPSState, ownership: tuple[tuple[int, ...], ...]
) -> RankOwnedMPSState:
    """A copy of `state` under a different ownership map, for refusal tests."""

    return RankOwnedMPSState(
        n_wires=state.n_wires,
        bsz=state.bsz,
        rank=state.rank,
        world_size=len(ownership),
        config=state.config,
        local_tensors=dict(state.local_tensors),
        ownership=ownership,
    )
