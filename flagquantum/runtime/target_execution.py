"""Target-driven execution that preserves sparse-output semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from ..core.ir import ensure_circuit_ir
from .planner.backend_selection import (
    BackendSelection,
    _validated_output_target,
    select_backend_by_cost,
)


@dataclass(frozen=True)
class TargetExecutionResult:
    values: torch.Tensor
    target: str
    backend: str
    selection: BackendSelection
    execution_summary: Mapping[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "backend": self.backend,
            "shape": tuple(self.values.shape),
            "full_state_materialized": (
                self.backend == "statevector" or self.target == "full_state"
            ),
            "selection": self.selection.summary(),
            "execution": dict(self.execution_summary or {}),
        }


def _statevector_target(
    circuit: Any,
    *,
    target: str,
    bitstrings: Sequence[int | str | Sequence[int]],
    observables: Sequence[Mapping[str, Sequence[int]]],
    options: Mapping[str, Any],
) -> torch.Tensor:
    from .execution import run_native

    state = run_native(circuit, mode="statevector", **dict(options))
    if not isinstance(state, torch.Tensor):
        raise TypeError("statevector execution must return a torch.Tensor")
    if target == "full_state":
        return state
    if target in {"single_amplitude", "few_amplitudes"}:
        from ..simulation.tensor_network.entrypoints import _normalize_bitstring

        n_wires = ensure_circuit_ir(circuit).n_wires
        indices = torch.tensor(
            [
                int("".join(map(str, _normalize_bitstring(bits, n_wires))), 2)
                for bits in bitstrings
            ],
            device=state.device,
        )
        values = state[:, indices]
        return values[:, 0] if target == "single_amplitude" else values
    if target == "local_observables":
        if not hasattr(circuit, "expectation_ps"):
            raise NotImplementedError(
                "statevector local_observables currently requires a Circuit"
            )
        return torch.stack(
            [
                circuit.expectation_ps(
                    x=observable.get("x"),
                    y=observable.get("y"),
                    z=observable.get("z"),
                )
                for observable in observables
            ],
            dim=-1,
        )
    raise ValueError(f"unsupported execution target: {target!r}")


def _target_request(
    target: str,
    *,
    bitstring: int | str | Sequence[int] | None,
    bitstrings: Sequence[int | str | Sequence[int]] | None,
    observables: Sequence[Mapping[str, Sequence[int]]] | None,
) -> tuple[
    tuple[int | str | Sequence[int], ...],
    tuple[Mapping[str, Sequence[int]], ...],
    int,
]:
    targets: tuple[int | str | Sequence[int], ...]
    if target == "single_amplitude":
        if bitstring is None:
            raise ValueError("single_amplitude requires bitstring")
        targets = (bitstring,)
    else:
        targets = tuple(bitstrings or ())
    observable_batch = tuple(observables or ())
    if target == "few_amplitudes" and not targets:
        raise ValueError("few_amplitudes requires at least one bitstring")
    if target == "local_observables" and not observable_batch:
        raise ValueError("local_observables requires at least one observable")
    target_count = (
        len(targets)
        if target in {"single_amplitude", "few_amplitudes"}
        else len(observable_batch) if target == "local_observables" else 1
    )
    return targets, observable_batch, target_count


def _tensor_network_target(
    circuit: Any,
    *,
    target: str,
    bitstrings: tuple[int | str | Sequence[int], ...],
    observables: tuple[Mapping[str, Sequence[int]], ...],
    world_size: int,
    options: Mapping[str, Any],
) -> tuple[torch.Tensor, Mapping[str, Any] | None]:
    if target == "single_amplitude":
        from ..simulation.tensor_network.entrypoints import tensor_network_amplitude

        return tensor_network_amplitude(circuit, bitstrings[0], **dict(options)), None
    if target == "few_amplitudes" and world_size == 1:
        from ..simulation.tensor_network.entrypoints import tensor_network_amplitudes

        return tensor_network_amplitudes(circuit, bitstrings, **dict(options)), None
    if target == "few_amplitudes":
        from .executors.tensor_network.execution import (
            distributed_tensor_network_amplitudes,
        )

        result = distributed_tensor_network_amplitudes(
            circuit, bitstrings, world_size=world_size, **dict(options)
        )
        return result.values, result.summary()
    if target == "local_observables" and world_size == 1:
        from ..simulation.tensor_network.entrypoints import tensor_network_expectations

        return tensor_network_expectations(circuit, observables, **dict(options)), None
    if target == "local_observables":
        from .executors.tensor_network.execution import (
            distributed_tensor_network_expectations,
        )

        result = distributed_tensor_network_expectations(
            circuit,
            observables,
            world_size=world_size,
            **dict(options),
        )
        return result.values, result.summary()
    raise ValueError("tensor_network target execution requires sparse output")


def run_target(
    circuit_or_ir: Any,
    *,
    target: str,
    bitstring: int | str | Sequence[int] | None = None,
    bitstrings: Sequence[int | str | Sequence[int]] | None = None,
    observables: Sequence[Mapping[str, Sequence[int]]] | None = None,
    mode: str = "auto",
    world_size: int = 1,
    require_gradients: bool = False,
    memory_limit_bytes: int | None = None,
    complex_bytes: int = 8,
    max_bond: int | None = None,
    **options: Any,
) -> TargetExecutionResult:
    """Select and execute a backend without turning sparse output into full state."""

    targets, observable_batch, target_count = _target_request(
        target,
        bitstring=bitstring,
        bitstrings=bitstrings,
        observables=observables,
    )
    selection = select_backend_by_cost(
        circuit_or_ir,
        target=_validated_output_target(target),
        target_count=target_count,
        require_gradients=require_gradients,
        world_size=world_size,
        complex_bytes=complex_bytes,
        memory_limit_bytes=memory_limit_bytes,
        max_bond=max_bond,
        requested_backend=mode,
    )
    backend = selection.selected_backend
    execution_summary: Mapping[str, Any] | None = None
    if backend == "statevector":
        if world_size > 1:
            raise NotImplementedError(
                "sparse extraction from distributed statevector shards is pending"
            )
        values = _statevector_target(
            circuit_or_ir,
            target=target,
            bitstrings=targets,
            observables=observable_batch,
            options=options,
        )
    elif backend == "tensor_network":
        values, execution_summary = _tensor_network_target(
            circuit_or_ir,
            target=target,
            bitstrings=targets,
            observables=observable_batch,
            world_size=world_size,
            options=options,
        )
    elif backend == "mps":
        if world_size > 1:
            raise NotImplementedError(
                "distributed MPS sparse-output execution is not production-ready"
            )
        from ..simulation.mps.entrypoints import run_mps

        mps = run_mps(
            circuit_or_ir,
            max_bond=max_bond or selection.estimated_mps_bond,
            cutoff=0.0,
            **options,
        )
        if target == "single_amplitude":
            values = mps.amplitude(targets[0])
        elif target == "few_amplitudes":
            values = mps.amplitudes(targets)
        elif target == "local_observables":
            values = torch.stack(
                [
                    mps.expectation_ps(
                        x=observable.get("x"),
                        y=observable.get("y"),
                        z=observable.get("z"),
                    )
                    for observable in observable_batch
                ],
                dim=-1,
            )
        else:
            raise ValueError("MPS target execution requires sparse output")
        execution_summary = {
            "state_mode": "mps",
            "max_bond_observed": mps.max_bond,
            "estimated_mps_bond": selection.estimated_mps_bond,
            "full_state_materialized": False,
        }
    else:
        raise NotImplementedError(
            f"target execution for selected backend {backend!r} is pending"
        )
    return TargetExecutionResult(
        values=values,
        target=target,
        backend=backend,
        selection=selection,
        execution_summary=execution_summary,
    )


__all__ = ["TargetExecutionResult", "run_target"]
