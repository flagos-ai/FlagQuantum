"""Native tensor network data structures and contraction runtime."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch

from ..statevector.operations import _bits_from_indices
from .contraction import contract_nodes_with_byte_budget
from .models import CompiledTNObservableProgram, TensorNetworkContractionPlan

_DENSE_Z_OBSERVABLE_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}

# Orders the slicing planner can drive. A contraction in one of these families can
# be given a byte budget; the others cannot, because nothing in them lowers a peak.
_SLICED_CONTRACTION_STRATEGIES = frozenset(
    {"sliced", "auto_sliced", "beam_sliced", "quality_sliced"}
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
        max_intermediate_bytes: int | None = None,
    ) -> None:
        self.plan = plan
        self.contraction_strategy = contraction_strategy
        self.max_intermediate_size = max_intermediate_size
        self.max_intermediate_bytes = max_intermediate_bytes
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

    def _state_contraction(
        self,
        strategy: str | None = None,
        *,
        max_intermediate_bytes: int | None = None,
    ) -> tuple[str, int | None]:
        """Resolve the state contraction order against this state's peak budget.

        The state network is built once and its full materialization is what the
        planner's ``state_bytes`` models, so a declared limit is met by slicing:
        a contracted order request becomes the slicing family with the requested
        order as the order *inside* each slice, and the slicing planner lowers the
        peak until it fits the budget or raises. An order already in the slicing
        family keeps its own inner order. With no budget the request passes
        through untouched, so nothing about an unlimited run changes.
        """

        requested = strategy or self.contraction_strategy
        budget = (
            self.max_intermediate_bytes
            if max_intermediate_bytes is None
            else max_intermediate_bytes
        )
        if budget is None:
            return requested, None
        if requested in _SLICED_CONTRACTION_STRATEGIES:
            return requested, budget
        inner = {
            "beam": "beam_sliced",
            "quality_multistart": "quality_sliced",
        }.get(requested)
        return (inner or "quality_sliced"), budget

    def _observable_contraction(self) -> tuple[str, int | None]:
        """Resolve the expectation order and the ceiling that governs it.

        The bra-operator-ket networks behind expectations are a contraction in
        their own right and are built per observable and per marginal, so a
        declared limit governs them too. ``quality_sliced`` is the order that runs
        under a budget because it is the one whose plan is both inside the budget
        and affordable to find: on a bonded eight-wire circuit the unsliced orders
        peak at 524288 bytes (``greedy``), 8388608 (``memory_greedy``) and 2048
        (``quality_multistart``), and the ``greedy`` slicing search costs minutes
        on the 75-node network before it produces a plan. The slice count is then
        what the budget really buys, and it is unbounded — the same network needs
        16 slices per marginal at 512 bytes, 512 at 256, and 131072 at 128 — so
        the economics preflight in
        :func:`flagquantum.simulation.tensor_network.contraction.contract_nodes_with_byte_budget`
        refuses the budgets whose slicing would cost more than the memory it saves
        instead of contracting them one slice at a time.

        With no budget the caller's order stands, so nothing about an unlimited
        run changes.
        """

        if self.max_intermediate_bytes is None:
            return self.contraction_strategy, None
        return "quality_sliced", self.max_intermediate_bytes

    def state(
        self,
        *,
        refresh: bool = False,
        strategy: str | None = None,
        max_intermediate_size: int | None = None,
        sliced_labels: Sequence[int] | None = None,
        max_intermediate_bytes: int | None = None,
    ) -> torch.Tensor:
        strategy = strategy or self.contraction_strategy
        max_intermediate_size = (
            self.max_intermediate_size
            if max_intermediate_size is None
            else max_intermediate_size
        )
        budget = (
            self.max_intermediate_bytes
            if max_intermediate_bytes is None
            else max_intermediate_bytes
        )
        sliced_labels = self.sliced_labels if sliced_labels is None else sliced_labels
        effective_strategy, effective_budget = self._state_contraction(
            strategy, max_intermediate_bytes=budget
        )
        cacheable = (
            strategy == self.contraction_strategy
            and max_intermediate_size == self.max_intermediate_size
            and budget == self.max_intermediate_bytes
            and tuple(sliced_labels or ()) == tuple(self.sliced_labels or ())
        )
        if cacheable:
            if self._state_cache is None or refresh:
                self._state_cache = self.plan.contract(
                    strategy=effective_strategy,
                    max_intermediate_size=max_intermediate_size,
                    max_intermediate_bytes=effective_budget,
                    sliced_labels=sliced_labels,
                )
            return self._state_cache
        return self.plan.contract(
            strategy=effective_strategy,
            max_intermediate_size=max_intermediate_size,
            max_intermediate_bytes=effective_budget,
            sliced_labels=sliced_labels,
        )

    to_statevector = state
    wavefunction = state

    def probabilities(self) -> torch.Tensor:
        return torch.abs(self.state()) ** 2

    def _validate_observable_wires(self, wires: Iterable[int]) -> None:
        """Refuse an observable wire that the state does not have.

        A wire outside ``range(n_wires)`` addresses no open leg of the network,
        and the contraction of an operator on a nonexistent leg still returns a
        finite number instead of failing. The dense sign-weight path is worse
        still: a negative shift makes ``>>`` select the wrong bit or none at
        all, so the reported expectation belongs to a different operator than
        the caller named.
        """
        if any(wire < 0 or wire >= self.n_wires for wire in wires):
            raise ValueError("observable wire index out of range")

    def expectation_z(self, wires: int | Sequence[int] | None = None) -> torch.Tensor:
        """Return the Z expectation on the named wires, one per wire.

        Two paths: below `dense_observable_wires` the dense state is materialized
        and the expectations are a single matmul against cached sign weights; above
        it a bra-ket contraction is built instead, and the strategy recorded on the
        state decides how that contraction is ordered.

        The path taken, whether the weight cache was reused and the peak size of
        the contraction are recorded on the state, so a caller can tell which route
        answered without inferring it from the timing.
        """
        if wires is None:
            wire_tuple = tuple(range(self.n_wires))
        elif isinstance(wires, int):
            wire_tuple = (wires,)
        else:
            wire_tuple = tuple(int(wire) for wire in wires)
        self._validate_observable_wires(wire_tuple)
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
        # The expectation network is a contraction in its own right, so a declared
        # peak budget is a ceiling for it too.
        observable_strategy, observable_budget = self._observable_contraction()
        cache = self.plan.program_cache
        cache_key = (
            "tn_observable",
            self.n_wires,
            self.bsz,
            wire_tuple,
            observable_strategy,
            observable_budget,
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
            if observable_budget is None:
                return torch.real(
                    observable_plan.contract(strategy=observable_strategy)
                )
            direct, _ = contract_nodes_with_byte_budget(
                observable_plan.nodes,
                observable_plan.output_labels,
                max_peak_bytes=observable_budget,
                contraction_strategy=observable_strategy,
            )
            return torch.real(direct)
        self._last_observable_execution = "per_observable_direct_fallback"
        values = [
            tensor_network_expectation_ps(
                self.plan,
                z=(wire,),
                strategy=observable_strategy,
                max_peak_bytes=observable_budget,
            )
            for wire in wire_tuple
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
        self._validate_observable_wires(x_set | y_set | z_set)

        strategy, budget = self._observable_contraction()
        return tensor_network_expectation_ps(
            self.plan,
            x=x,
            y=y,
            z=z,
            strategy=strategy,
            max_peak_bytes=budget,
        )

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
        # Validate the format before sampling. Checking it per sample row would
        # let an unsupported format return an empty histogram for zero shots
        # instead of refusing the request.
        if format not in ("bin", "int"):
            raise ValueError("counts format must be 'bin' or 'int'.")
        samples = self.sample(shots, generator=generator, format="index")
        outputs: list[dict[str | int, int]] = []
        for row in samples:
            unique, counts = torch.unique(row, return_counts=True)
            batch_counts: dict[str | int, int] = {}
            for key, count in zip(unique.tolist(), counts.tolist(), strict=True):
                index = int(key)
                out_key: str | int = (
                    index if format == "int" else f"{index:0{self.n_wires}b}"
                )
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
