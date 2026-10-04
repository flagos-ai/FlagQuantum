"""Focused contracts for the explicit distributed statevector full-state gather.

The gather exists so a workload that needs the whole vector can ask for it by
name. These tests pin the two things that makes it safe to publish: it returns
the logical vector exactly, and it cannot be read as a scaling result.
"""

from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import ensure_circuit_ir
from flagquantum.runtime.executors.statevector.errors import (
    FullStateMaterializationError,
)
from flagquantum.runtime.executors.statevector.forward import (
    communication_aware_qubit_layout,
)
from flagquantum.runtime.executors.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)
from flagquantum.runtime.executors.statevector.gather import (
    BLOCKER_FULL_STATE_GATHER_IS_NOT_A_SCALING_RESULT,
    _canonical_basis_order,
    _place_rank_blocks,
    gather_distributed_statevector,
)
from flagquantum.runtime.executors.statevector.local_execution import (
    simulate_distributed_statevector_local,
)
from flagquantum.runtime.executors.statevector.models import (
    StatevectorShard,
    StatevectorShardState,
)
from flagquantum.runtime.executors.statevector.planning import (
    plan_distributed_statevector,
)

pytestmark = pytest.mark.unit


def _unique_amplitude_circuit(n_wires: int) -> fq.Circuit:
    """A circuit whose amplitudes are all distinct, so a wrong order shows up."""

    circuit = fq.Circuit(n_wires)
    for wire in range(n_wires):
        circuit.ry(wire, 0.11 * (wire + 1))
    for wire in range(n_wires - 1):
        circuit.cx(wire, wire + 1)
    circuit.rz(n_wires - 1, -0.29)
    return circuit


def _dense(circuit: fq.Circuit) -> torch.Tensor:
    return circuit.state().reshape(-1).to(torch.complex128)


class TestWorldSizeOne:
    """One rank already holds the whole state; the gather must not perturb it."""

    @pytest.mark.parametrize("qubit_layout", ["canonical", "communication_aware"])
    def test_the_gathered_vector_matches_the_single_device_vector(
        self, qubit_layout: str
    ) -> None:
        circuit = _unique_amplitude_circuit(4)

        result = execute_torch_distributed_statevector(
            circuit,
            device=torch.device("cpu"),
            dtype=torch.complex128,
            qubit_layout=qubit_layout,
        )
        gathered = gather_distributed_statevector(result)

        assert gathered.state.shape == (1, 16)
        assert torch.allclose(gathered.state[0], _dense(circuit), atol=1e-12)

    def test_the_rank_local_result_still_refuses_full_state_materialization(
        self,
    ) -> None:
        """The gather is a separate surface; the executor result stays rank-local."""

        result = execute_torch_distributed_statevector(
            _unique_amplitude_circuit(3),
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )

        with pytest.raises(FullStateMaterializationError, match="rank-local"):
            result.full_state()

    def test_the_summary_states_the_gather_cost_and_not_a_speedup(self) -> None:
        result = execute_torch_distributed_statevector(
            _unique_amplitude_circuit(4),
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )

        summary = gather_distributed_statevector(result).summary()

        assert summary["operation"] == "all_gather"
        assert summary["operation_semantics"] == "explicit_full_state_gather"
        assert summary["distribution_semantics"] == "replicated_per_rank"
        assert summary["full_state_materialization"] is True
        assert summary["amplitude_basis_order"] == "canonical_logical"
        assert summary["scalability_claim_allowed"] is False
        assert summary["release_gate_allowed"] is False
        assert summary["blockers"] == (
            BLOCKER_FULL_STATE_GATHER_IS_NOT_A_SCALING_RESULT,
        )
        assert "speedup" not in " ".join(summary)
        assert "scaling_efficiency" not in " ".join(summary)

    def test_the_gather_publishes_its_bytes_and_rank_ownership(self) -> None:
        result = execute_torch_distributed_statevector(
            _unique_amplitude_circuit(4),
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )

        gathered = gather_distributed_statevector(result)

        assert gathered.full_state_bytes == 16 * 16
        assert gathered.local_block_bytes == 16 * 16
        assert gathered.gather_bytes_per_rank == 0
        assert gathered.total_gather_bytes == 0
        assert gathered.peak_resident_bytes_per_rank == 2 * 16 * 16
        assert gathered.rank_global_index_map == (
            {
                "rank": 0,
                "placement": "contiguous_amplitude_range",
                "global_start": 0,
                "global_end": 16,
                "local_amplitudes": 16,
            },
        )


class TestMultiRankPlacement:
    """The placement math, checked against the in-tree shard simulator.

    A real multi-rank collective needs a launched process group, which a CPU unit
    test does not have. What the collective adds is transport; what it must not
    add is arithmetic. So the arithmetic is checked here against the simulator
    that already reproduces dense execution from the same plan, and the byte
    accounting is checked through the public summary above.
    """

    @pytest.mark.parametrize(
        ("n_wires", "world_size"),
        [(4, 2), (4, 4), (5, 2), (5, 4)],
    )
    def test_placement_reconstructs_the_logical_vector(
        self, n_wires: int, world_size: int
    ) -> None:
        circuit = _unique_amplitude_circuit(n_wires)
        ir, mapping = communication_aware_qubit_layout(
            ensure_circuit_ir(circuit), world_size=world_size, local_world_size=1
        )

        simulated = simulate_distributed_statevector_local(
            ir, world_size=world_size, dtype=torch.complex128
        )
        blocks = [shard.amplitudes for shard in simulated.shards]
        internal = _place_rank_blocks(blocks, simulated.plan)
        state = _canonical_basis_order(internal, tuple(mapping))

        assert simulated.plan.sharded_qubits == tuple(
            sorted(simulated.plan.sharded_qubits)
        )
        assert torch.allclose(state[0], _dense(circuit), atol=1e-12)

    def test_a_sharded_plan_strides_the_rank_address_over_the_global_index(
        self,
    ) -> None:
        """Rank ownership is the plan's, and the placement follows it exactly."""

        circuit = _unique_amplitude_circuit(5)
        ir, mapping = communication_aware_qubit_layout(
            ensure_circuit_ir(circuit), world_size=4, local_world_size=1
        )
        simulated = simulate_distributed_statevector_local(
            ir, world_size=4, dtype=torch.complex128
        )
        plan = simulated.plan

        assert plan.distribution == "qubit_address_sharded"
        assert plan.sharded_qubits == (3, 4)
        assert mapping == (0, 1, 2, 4, 3)

        blocks = [
            torch.full((1, 8), float(rank), dtype=torch.complex128) for rank in range(4)
        ]
        internal = _place_rank_blocks(blocks, plan)

        # Rank r owns every 4th amplitude starting at r, in internal basis order.
        for rank in range(4):
            assert torch.all(internal[0, rank::4] == complex(rank, 0))


class _GatherRefusal:
    """Build a rank-local result whose plan or mapping is deliberately wrong."""

    @staticmethod
    def baseline(n_wires: int = 3) -> tuple[torch.Tensor, object]:
        circuit = _unique_amplitude_circuit(n_wires)
        result = execute_torch_distributed_statevector(
            circuit,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        )
        return _dense(circuit), result


class TestRefusals:
    """A gather that cannot place the amplitudes must fail rather than guess."""

    def test_an_unpublished_qubit_layout_is_refused(self) -> None:
        _, result = _GatherRefusal.baseline()

        with pytest.raises(FullStateMaterializationError, match="qubit layout"):
            gather_distributed_statevector(replace(result, qubit_layout="custom"))

    def test_a_mapping_that_is_not_a_permutation_is_refused(self) -> None:
        _, result = _GatherRefusal.baseline()

        with pytest.raises(FullStateMaterializationError, match="not a permutation"):
            gather_distributed_statevector(
                replace(result, logical_to_physical_qubits=(0, 1, 1))
            )

    def test_a_mapping_of_the_wrong_length_is_refused(self) -> None:
        _, result = _GatherRefusal.baseline()

        with pytest.raises(FullStateMaterializationError, match="not a permutation"):
            gather_distributed_statevector(
                replace(result, logical_to_physical_qubits=(0, 1))
            )

    def test_a_distribution_without_plan_derived_ownership_is_refused(self) -> None:
        _, result = _GatherRefusal.baseline()

        with pytest.raises(FullStateMaterializationError, match="distribution"):
            gather_distributed_statevector(
                replace(result, plan=replace(result.plan, distribution="tensor_sliced"))
            )

    def test_an_uneven_plan_is_refused(self) -> None:
        """World size 3 over 8 amplitudes gives shards of 3, 3 and 2."""

        circuit = _unique_amplitude_circuit(3)
        uneven = plan_distributed_statevector(
            ensure_circuit_ir(circuit), world_size=3, bsz=1, complex_bytes=16
        )
        assert uneven.distribution == "contiguous_amplitude_range"
        assert [shard.local_amplitudes for shard in uneven.shards] == [3, 3, 2]

        _, result = _GatherRefusal.baseline()

        with pytest.raises(FullStateMaterializationError, match="uneven"):
            gather_distributed_statevector(replace(result, plan=uneven))

    def test_a_shard_of_the_wrong_width_is_refused(self) -> None:
        _, result = _GatherRefusal.baseline()

        with pytest.raises(
            FullStateMaterializationError, match="local shard has shape"
        ):
            gather_distributed_statevector(
                replace(
                    result,
                    shard_state=replace(
                        result.shard_state,
                        amplitudes=torch.zeros(1, 4, dtype=torch.complex128),
                    ),
                )
            )

    def test_a_batch_row_count_the_plan_does_not_assign_is_refused(self) -> None:
        _, result = _GatherRefusal.baseline()

        with pytest.raises(
            FullStateMaterializationError, match="local shard has shape"
        ):
            gather_distributed_statevector(
                replace(
                    result,
                    shard_state=replace(
                        result.shard_state,
                        amplitudes=torch.zeros(2, 8, dtype=torch.complex128),
                    ),
                )
            )

    def test_a_shard_of_the_wrong_element_size_is_refused(self) -> None:
        _, result = _GatherRefusal.baseline()

        with pytest.raises(FullStateMaterializationError, match="element_size|byte"):
            gather_distributed_statevector(
                replace(result, plan=replace(result.plan, complex_bytes=8))
            )

    def test_published_indices_that_contradict_the_plan_are_refused(self) -> None:
        """A published index map the plan disagrees with is a real conflict."""

        _, result = _GatherRefusal.baseline()
        wrong = torch.zeros(8, dtype=torch.long)

        with pytest.raises(FullStateMaterializationError, match="global indices"):
            gather_distributed_statevector(
                replace(
                    result,
                    shard_state=replace(result.shard_state, global_indices=wrong),
                )
            )

    def test_published_indices_that_agree_with_the_plan_are_accepted(self) -> None:
        """The empty map means ownership was represented compactly, not lost."""

        dense, result = _GatherRefusal.baseline()
        published = torch.arange(8, dtype=torch.long)

        gathered = gather_distributed_statevector(
            replace(
                result,
                shard_state=replace(result.shard_state, global_indices=published),
            )
        )

        assert torch.allclose(gathered.state[0], dense, atol=1e-12)

    def test_a_process_group_smaller_than_the_plan_is_refused(self) -> None:
        """No process group is initialized here, so the world size reads as one."""

        circuit = _unique_amplitude_circuit(4)
        sharded = plan_distributed_statevector(
            ensure_circuit_ir(circuit), world_size=2, bsz=1, complex_bytes=16
        )
        _, result = _GatherRefusal.baseline(n_wires=4)

        with pytest.raises(FullStateMaterializationError, match="process group holds"):
            gather_distributed_statevector(
                replace(
                    result,
                    plan=sharded,
                    shard_state=StatevectorShardState(
                        rank=0,
                        shard=sharded.shards[0],
                        amplitudes=torch.zeros(1, 8, dtype=torch.complex128),
                        global_indices=torch.empty(0, dtype=torch.long),
                    ),
                )
            )

    def test_a_shard_that_does_not_match_its_plan_shard_is_refused(self) -> None:
        _, result = _GatherRefusal.baseline()
        other = StatevectorShard(
            rank=0,
            world_size=1,
            amplitude_start=0,
            amplitude_end=4,
            local_amplitudes=4,
            local_state_bytes=64,
        )

        with pytest.raises(FullStateMaterializationError, match="global indices"):
            gather_distributed_statevector(
                replace(
                    result,
                    shard_state=replace(
                        result.shard_state,
                        shard=other,
                        global_indices=torch.arange(4, dtype=torch.long),
                    ),
                )
            )
