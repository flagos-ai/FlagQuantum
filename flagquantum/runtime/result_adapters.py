"""Small adapters used to normalize backend-native execution metadata."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..core.ir import CircuitIR
    from .module import Module
    from .result import ExecutionResult


class LiveRuntimeSummary(Mapping[str, Any]):
    """Expose mutable native evidence through a stable Mapping contract."""

    def __init__(self, source: Any, overrides: Mapping[str, Any] | None = None) -> None:
        self._source = source
        self._overrides = dict(overrides or {})

    def _snapshot(self) -> dict[str, Any]:
        source = (
            self._source
            if isinstance(self._source, Mapping)
            else self._source.summary()
        )
        return {**dict(source), **self._overrides}

    def __getitem__(self, key: str) -> Any:
        return self._snapshot()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._snapshot())

    def __len__(self) -> int:
        return len(self._snapshot())


def _module_execution_result(
    owner: Module,
    ir: CircuitIR,
    *,
    value: torch.Tensor | None,
    state: torch.Tensor | None,
    inputs: torch.Tensor | None,
    parameters: torch.Tensor | Mapping[str, torch.Tensor],
    qubits: tuple[int, ...],
    runtime: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    state_world_size: int = 1,
) -> ExecutionResult:
    """Keep Module result semantics shared by eager and compiled training."""
    from .result import ExecutionResult

    input_batch_size = 1 if inputs is None or inputs.ndim == 0 else inputs.shape[0]
    parameter_batch_size = (
        parameters.shape[0]
        if isinstance(parameters, torch.Tensor) and parameters.ndim > 1
        else 1
    )
    observable_count = (
        owner.hamiltonian.n_terms
        if owner.policy.observable == "hamiltonian" and owner.hamiltonian is not None
        else len(qubits)
    )
    if owner.policy.observable == "hamiltonian" and owner.hamiltonian is not None:
        from .parallel import group_observables

        observable_group_count = len(group_observables(owner.hamiltonian.terms))
    else:
        observable_group_count = 1
    parallel = (
        owner._parallel_plan.summary()
        if owner._parallel_plan
        else {
            "parallel_dimensions": {
                "input_batch": input_batch_size,
                "parameter_batch": parameter_batch_size,
                "observable_batch": observable_count,
                "data_parallel": 1,
                "state_parallel": state_world_size,
                "model_parallel": 1,
            },
            "distribution_semantics": (
                "sharded_across_ranks"
                if state_world_size > 1
                else "single_device_fast_path"
            ),
        }
    )
    return ExecutionResult(
        value=value,
        state=state,
        metrics={
            "training": owner.training,
            "input_batch_size": input_batch_size,
            "parameter_batch_size": parameter_batch_size,
            "observable_count": observable_count,
            "observable_group_count": observable_group_count,
            "parallelism": parallel,
            "precision": owner.precision.to_dict(),
        },
        provenance={"ir_hash": ir.content_hash},
        runtime=runtime,
        compatibility=compatibility,
    )


__all__ = ("LiveRuntimeSummary",)
