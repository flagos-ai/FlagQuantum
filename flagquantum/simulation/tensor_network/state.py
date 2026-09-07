"""Native tensor network data structures and contraction runtime."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ..statevector.operations import _bits_from_indices

_CONTRACTION_PROFILE_CACHE: dict[tuple[Any, ...], TensorNetworkContractionProfile] = {}
_CONTRACTION_PATH_CACHE: dict[
    tuple[Any, ...], tuple[tuple[int, int, tuple[int, ...]], ...]
] = {}
_CONTRACTION_STAGE_CACHE: dict[tuple[Any, ...], CompiledTNStagePlan] = {}
_DENSE_Z_OBSERVABLE_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}
_Z_OBSERVABLE_NODE_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], tuple[torch.Tensor, ...]
] = {}


from .models import (  # noqa: E402
    CompiledTNObservableProgram,
    CompiledTNStagePlan,
    TensorNetworkContractionPlan,
    TensorNetworkContractionProfile,
)


class TensorNetworkState:
    """Result object for general circuit tensor-network simulation."""

    def __init__(
        self,
        plan: TensorNetworkContractionPlan,
        *,
        contraction_strategy: str = "greedy",
        max_intermediate_size: int | None = None,
        sliced_labels: Sequence[int] | None = None,
        dense_observable_wires: int = 0,
    ) -> None:
        self.plan = plan
        self.contraction_strategy = contraction_strategy
        self.max_intermediate_size = max_intermediate_size
        self.sliced_labels = tuple(sliced_labels) if sliced_labels is not None else None
        self.dense_observable_wires = int(dense_observable_wires)
        self._state_cache: torch.Tensor | None = None
        self._observable_programs: dict[
            tuple[int, ...], CompiledTNObservableProgram
        ] = {}
        self._last_observable_execution = "not_executed"
        self._last_observable_program_cache_hit = False
        self._last_observable_peak_size: int | None = None

    @property
    def n_wires(self) -> int:
        return self.plan.n_wires

    @property
    def bsz(self) -> int:
        return self.plan.bsz

    def state(
        self,
        *,
        refresh: bool = False,
        strategy: str | None = None,
        max_intermediate_size: int | None = None,
        sliced_labels: Sequence[int] | None = None,
    ) -> torch.Tensor:
        strategy = strategy or self.contraction_strategy
        max_intermediate_size = (
            self.max_intermediate_size
            if max_intermediate_size is None
            else max_intermediate_size
        )
        sliced_labels = self.sliced_labels if sliced_labels is None else sliced_labels
        cacheable = (
            strategy == self.contraction_strategy
            and max_intermediate_size == self.max_intermediate_size
            and tuple(sliced_labels or ()) == tuple(self.sliced_labels or ())
        )
        if cacheable:
            if self._state_cache is None or refresh:
                self._state_cache = self.plan.contract(
                    strategy=strategy,
                    max_intermediate_size=max_intermediate_size,
                    sliced_labels=sliced_labels,
                )
            return self._state_cache
        return self.plan.contract(
            strategy=strategy,
            max_intermediate_size=max_intermediate_size,
            sliced_labels=sliced_labels,
        )

    to_statevector = state
    wavefunction = state

    def probabilities(self) -> torch.Tensor:
        return torch.abs(self.state()) ** 2

    def expectation_z(self, wires: int | Sequence[int] | None = None) -> torch.Tensor:
        if wires is None:
            wire_tuple = tuple(range(self.n_wires))
        elif isinstance(wires, int):
            wire_tuple = (wires,)
        else:
            wire_tuple = tuple(int(wire) for wire in wires)
        if self.n_wires <= self.dense_observable_wires:
            self._last_observable_execution = "dense_state"
            self._last_observable_program_cache_hit = False
            self._last_observable_peak_size = None
            state = self.state()
            probabilities = torch.abs(state) ** 2
            key = (
                self.n_wires,
                wire_tuple,
                str(state.device),
                probabilities.dtype,
            )
            weights = _DENSE_Z_OBSERVABLE_CACHE.get(key)
            if weights is None:
                indices = torch.arange(2**self.n_wires, device=state.device)
                columns = []
                for wire in wire_tuple:
                    shift = self.n_wires - int(wire) - 1
                    columns.append(1 - 2 * ((indices >> shift) & 1))
                weights = torch.stack(columns, dim=-1).to(probabilities.dtype)
                _DENSE_Z_OBSERVABLE_CACHE[key] = weights
            return probabilities @ weights
        observable_strategy = (
            "memory_greedy"
            if self.contraction_strategy == "greedy"
            else self.contraction_strategy
        )
        cache = self.plan.program_cache
        cache_key = (
            "tn_observable",
            self.n_wires,
            self.bsz,
            wire_tuple,
            observable_strategy,
        )
        cached = None if cache is None else cache.get(cache_key)
        self._last_observable_program_cache_hit = cached is not None
        if cached is None:
            program = CompiledTNObservableProgram(self.n_wires, wire_tuple)
            observable_plan = program.bind(self.plan)
            profile = observable_plan.contraction_profile(observable_strategy)
            peak_size = profile.peak_size
            use_direct_batch = peak_size <= 2**24
            if cache is not None:
                cache[cache_key] = (program, peak_size, use_direct_batch)
        else:
            program, peak_size, use_direct_batch = cached
            observable_plan = program.bind(self.plan)
        self._observable_programs[wire_tuple] = program
        self._last_observable_peak_size = peak_size
        if use_direct_batch:
            self._last_observable_execution = "memory_first_direct_batch"
            return torch.real(observable_plan.contract(strategy=observable_strategy))
        self._last_observable_execution = "per_observable_direct_fallback"
        values = [
            tensor_network_expectation_ps(self.plan, z=(wire,)) for wire in wire_tuple
        ]
        return torch.stack(values, dim=-1)

    def expectation_ps(
        self,
        *,
        z: Sequence[int] | None = None,
        x: Sequence[int] | None = None,
        y: Sequence[int] | None = None,
    ) -> torch.Tensor:
        x_set = set(x or ())
        y_set = set(y or ())
        z_set = set(z or ())
        if (x_set & y_set) or (x_set & z_set) or (y_set & z_set):
            raise ValueError("A wire can appear in only one of x, y, or z.")

        return tensor_network_expectation_ps(self.plan, x=x, y=y, z=z)

    def sample(
        self,
        shots: int = 1,
        *,
        generator: torch.Generator | None = None,
        format: str = "bits",
    ) -> torch.Tensor:
        probs = self.probabilities()
        samples = torch.multinomial(
            probs, num_samples=shots, replacement=True, generator=generator
        )
        if format == "index":
            return samples
        if format != "bits":
            raise ValueError("sample format must be 'bits' or 'index'.")
        return _bits_from_indices(samples, self.n_wires)

    def counts(
        self,
        shots: int,
        *,
        generator: torch.Generator | None = None,
        format: str = "bin",
    ) -> list[dict[str | int, int]]:
        samples = self.sample(shots, generator=generator, format="index")
        outputs: list[dict[str | int, int]] = []
        for row in samples:
            unique, counts = torch.unique(row, return_counts=True)
            batch_counts: dict[str | int, int] = {}
            for key, count in zip(unique.tolist(), counts.tolist()):
                if format == "int":
                    out_key: str | int = int(key)
                elif format == "bin":
                    out_key = f"{int(key):0{self.n_wires}b}"
                else:
                    raise ValueError("counts format must be 'bin' or 'int'.")
                batch_counts[out_key] = int(count)
            outputs.append(batch_counts)
        return outputs

    def summary(self) -> dict[str, Any]:
        return {
            **self.plan.summary(),
            "observable_execution": self._last_observable_execution,
            "observable_program_cache_hit": (self._last_observable_program_cache_hit),
            "observable_peak_size": self._last_observable_peak_size,
            "full_state_materialized": self._state_cache is not None,
        }


def tensor_network_expectation_ps(*args: Any, **kwargs: Any) -> torch.Tensor:
    from .entrypoints import tensor_network_expectation_ps as execute

    return execute(*args, **kwargs)
