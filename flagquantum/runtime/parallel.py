"""Canonical batch, observable, data, and state parallel contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..algorithms import HamiltonianTerm


@dataclass(frozen=True)
class ObservableGroup:
    term_indices: tuple[int, ...]
    basis: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class HybridParallelPlan:
    """Orthogonal parallel dimensions for one quantum-AI workload."""

    world_size: int
    data_parallel_size: int
    state_parallel_size: int
    model_parallel_size: int
    input_batch_size: int
    parameter_batch_size: int
    observable_count: int
    observable_group_count: int
    global_quantum_state_bytes: int
    parameter_bytes: int
    input_bytes: int
    observable_bytes: int
    per_rank_quantum_state_bytes: int
    per_rank_input_bytes: int
    state_exchange_bytes_per_gate: int
    ddp_gradient_allreduce_bytes_per_step: int
    observable_reduction_bytes_per_step: int
    data_groups: tuple[tuple[int, ...], ...]
    state_groups: tuple[tuple[int, ...], ...]
    parameter_ownership: str = "ddp_replicated"
    optimizer_state_ownership: str = "ddp_replicated"
    quantum_state_ownership: str = "sharded_across_state_group"
    fsdp_compatibility: str = "parameters_only_state_shard_group_excluded"
    dtensor_compatibility: str = "parameters_only_quantum_state_is_runtime_owned"

    @property
    def distribution_semantics(self) -> str:
        if self.state_parallel_size > 1:
            return "sharded_across_ranks"
        if self.data_parallel_size > 1:
            return "data_parallel_replicated"
        return "single_device_fast_path"

    def summary(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "parallel_dimensions": {
                "input_batch": self.input_batch_size,
                "parameter_batch": self.parameter_batch_size,
                "observable_batch": self.observable_count,
                "data_parallel": self.data_parallel_size,
                "state_parallel": self.state_parallel_size,
                "model_parallel": self.model_parallel_size,
            },
            "distribution_semantics": self.distribution_semantics,
            "memory_accounting": {
                "quantum_state_per_rank": self.per_rank_quantum_state_bytes,
                "parameters_per_rank": self.parameter_bytes,
                "optimizer_state_policy": self.optimizer_state_ownership,
                "inputs_per_data_replica": self.per_rank_input_bytes,
                "observables": self.observable_bytes,
            },
            "communication_accounting": {
                "state_exchange_per_cross_shard_gate": self.state_exchange_bytes_per_gate,
                "ddp_gradient_allreduce_per_step": self.ddp_gradient_allreduce_bytes_per_step,
                "observable_reduction_per_step": self.observable_reduction_bytes_per_step,
            },
            "scalability_claim_allowed": False,
            "blockers": ("measured_hybrid_accelerator_acceptance_pending",),
        }


def group_observables(
    terms: Iterable[HamiltonianTerm],
) -> tuple[ObservableGroup, ...]:
    """Greedily group qubit-wise commuting Pauli terms."""

    groups: list[tuple[list[int], dict[int, str]]] = []
    for index, term in enumerate(terms):
        ops = dict(term.ops)
        for indices, basis in groups:
            if all(
                wire not in basis or basis[wire] == name for wire, name in ops.items()
            ):
                indices.append(index)
                basis.update(ops)
                break
        else:
            groups.append(([index], dict(ops)))
    return tuple(
        ObservableGroup(tuple(indices), tuple(sorted(basis.items())))
        for indices, basis in groups
    )


def plan_hybrid_parallel(
    *,
    world_size: int,
    data_parallel_size: int = 1,
    state_parallel_size: int = 1,
    model_parallel_size: int = 1,
    input_batch_size: int = 1,
    parameter_batch_size: int = 1,
    observable_count: int = 1,
    observable_group_count: int = 1,
    global_quantum_state_bytes: int = 0,
    parameter_bytes: int = 0,
    input_bytes: int = 0,
    observable_bytes: int = 0,
) -> HybridParallelPlan:
    expected = data_parallel_size * state_parallel_size * model_parallel_size
    if world_size != expected:
        raise ValueError(
            "world_size must equal data_parallel_size * state_parallel_size * "
            "model_parallel_size"
        )
    if (
        min(
            world_size,
            data_parallel_size,
            state_parallel_size,
            model_parallel_size,
            input_batch_size,
            parameter_batch_size,
            observable_count,
            observable_group_count,
        )
        <= 0
    ):
        raise ValueError("parallel and batch dimensions must be positive")
    if model_parallel_size != 1:
        raise NotImplementedError(
            "model parallel execution is planned but not implemented"
        )
    state_groups = tuple(
        tuple(range(start, start + state_parallel_size))
        for start in range(0, world_size, state_parallel_size)
    )
    data_groups = tuple(
        tuple(
            replica * state_parallel_size + shard
            for replica in range(data_parallel_size)
        )
        for shard in range(state_parallel_size)
    )
    return HybridParallelPlan(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        state_parallel_size=state_parallel_size,
        model_parallel_size=model_parallel_size,
        input_batch_size=input_batch_size,
        parameter_batch_size=parameter_batch_size,
        observable_count=observable_count,
        observable_group_count=observable_group_count,
        global_quantum_state_bytes=global_quantum_state_bytes,
        parameter_bytes=parameter_bytes,
        input_bytes=input_bytes,
        observable_bytes=observable_bytes,
        per_rank_quantum_state_bytes=(
            global_quantum_state_bytes + state_parallel_size - 1
        )
        // state_parallel_size,
        per_rank_input_bytes=(input_bytes + data_parallel_size - 1)
        // data_parallel_size,
        state_exchange_bytes_per_gate=(
            2
            * (global_quantum_state_bytes + state_parallel_size - 1)
            // state_parallel_size
            if state_parallel_size > 1
            else 0
        ),
        ddp_gradient_allreduce_bytes_per_step=(
            (2 * parameter_bytes * (data_parallel_size - 1) + data_parallel_size - 1)
            // data_parallel_size
        ),
        observable_reduction_bytes_per_step=(
            (2 * observable_bytes * (state_parallel_size - 1) + state_parallel_size - 1)
            // state_parallel_size
            if state_parallel_size > 1
            else 0
        ),
        data_groups=data_groups,
        state_groups=state_groups,
    )


__all__ = [
    "HybridParallelPlan",
    "ObservableGroup",
    "group_observables",
    "plan_hybrid_parallel",
]
