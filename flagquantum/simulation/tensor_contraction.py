"""Native tensor network data structures and contraction runtime."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Mapping, Sequence

import torch

from ..core.ir import CircuitIR, ensure_circuit_ir
from .real_imag_kernels import complex_einsum_pair
from .tensor_models import (
    CompiledTNStagePlan,
    PairContractionStep,
    TensorNetworkContractionProfile,
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
)
from .tensor_stages import (
    compile_contraction_stages as _compile_contraction_stages,
)
from .tensor_stages import (
    einsum_pair_by_labels as _einsum_pair_by_labels,
)
from .tensor_stages import (
    einsum_reorder_by_labels as _einsum_reorder_by_labels,
)
from .tensor_stages import (
    execute_contraction_stages as _execute_contraction_stages,
)
from .tensor_stages import (
    execute_pair_steps as _execute_pair_steps,
)
from .tensor_stages import (
    pair_equation as _pair_equation,
)

_CONTRACTION_PROFILE_CACHE: dict[tuple[Any, ...], TensorNetworkContractionProfile] = {}
_CONTRACTION_PATH_CACHE: dict[
    tuple[Any, ...], tuple[tuple[int, int, tuple[int, ...]], ...]
] = {}
_QUALITY_MULTISTART_PATH_CACHE: dict[
    tuple[Any, ...], tuple[PairContractionStep, ...]
] = {}
_CONTRACTION_STAGE_CACHE: dict[tuple[Any, ...], CompiledTNStagePlan] = {}
_DENSE_Z_OBSERVABLE_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}
_Z_OBSERVABLE_NODE_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], tuple[torch.Tensor, ...]
] = {}


@dataclass(frozen=True)
class _DryRunTensor:
    """Shape-only tensor metadata with no storage or PyTorch size limit."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device = torch.device("meta")
    is_cuda: bool = False

    def numel(self) -> int:
        return _product(self.shape)

    def element_size(self) -> int:
        return int(torch.empty((), dtype=self.dtype).element_size())


def _dry_run_tensor(shape: Sequence[int], reference: Any) -> _DryRunTensor:
    return _DryRunTensor(
        shape=tuple(int(dim) for dim in shape),
        dtype=reference.dtype,
    )


def _as_ir(program: Any) -> CircuitIR:
    return ensure_circuit_ir(program)


def _product(values: Any) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return int(out)


def _label_dims(nodes: Sequence[TensorNetworkNode]) -> dict[int, int]:
    dims: dict[int, int] = {}
    for node in nodes:
        for label, dim in zip(node.labels, node.tensor.shape):
            dim = int(dim)
            if label in dims and dims[label] != dim:
                raise ValueError(
                    f"Inconsistent dimension for tensor-network label {label}."
                )
            dims[label] = dim
    return dims


def _label_counts(nodes: Sequence[TensorNetworkNode]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node in nodes:
        for label in node.labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _pair_output_labels(
    left: TensorNetworkNode,
    right: TensorNetworkNode,
    *,
    label_counts: Mapping[int, int],
    final_outputs: set[int],
) -> tuple[int, ...]:
    labels = tuple(dict.fromkeys(left.labels + right.labels))
    return tuple(
        label
        for label in labels
        if not (
            label not in final_outputs
            and (
                label_counts.get(label, 0) == 1
                or (
                    label in left.labels
                    and label in right.labels
                    and label_counts.get(label, 0) == 2
                )
            )
        )
    )


def _estimate_pair(
    left: TensorNetworkNode,
    right: TensorNetworkNode,
    *,
    dims: Mapping[int, int],
    label_counts: Mapping[int, int],
    final_outputs: set[int],
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    output_labels = _pair_output_labels(
        left,
        right,
        label_counts=label_counts,
        final_outputs=final_outputs,
    )
    union_labels = tuple(dict.fromkeys(left.labels + right.labels))
    cost = _product(dims[label] for label in union_labels)
    intermediate_size = _product(dims[label] for label in output_labels)
    contracted = tuple(label for label in union_labels if label not in output_labels)
    return cost, intermediate_size, output_labels, contracted


def _choose_greedy_pair(
    nodes: Sequence[TensorNetworkNode],
    *,
    dims: Mapping[int, int],
    label_counts: Mapping[int, int],
    final_outputs: set[int],
    objective: str = "balanced",
    random_state: random.Random | None = None,
    random_pool_size: int = 1,
) -> tuple[int, int, int, int, tuple[int, ...]]:
    candidates: list[
        tuple[
            tuple[Any, ...],
            tuple[int, int, int, int, tuple[int, ...]],
        ]
    ] = []
    if objective in {"quality", "treewidth"}:
        label_nodes: dict[int, list[int]] = {}
        for node_index, node in enumerate(nodes):
            for label in node.labels:
                if dims[label] <= 1:
                    continue
                label_nodes.setdefault(int(label), []).append(node_index)
        connected_pairs = {
            (left_idx, right_idx)
            for node_indices in label_nodes.values()
            for position, left_idx in enumerate(node_indices)
            for right_idx in node_indices[position + 1 :]
            if left_idx != right_idx
        }
        if connected_pairs:
            candidate_pairs = sorted(
                (min(left_idx, right_idx), max(left_idx, right_idx))
                for left_idx, right_idx in connected_pairs
            )
        else:
            candidate_pairs = [
                (left_idx, right_idx)
                for left_idx in range(len(nodes))
                for right_idx in range(left_idx + 1, len(nodes))
            ]
    else:
        candidate_pairs = [
            (left_idx, right_idx)
            for left_idx in range(len(nodes))
            for right_idx in range(left_idx + 1, len(nodes))
        ]
    for left_idx, right_idx in candidate_pairs:
        cost, intermediate, output_labels, contracted = _estimate_pair(
            nodes[left_idx],
            nodes[right_idx],
            dims=dims,
            label_counts=label_counts,
            final_outputs=final_outputs,
        )
        shared = sum(
            dims[label] > 1
            for label in set(nodes[left_idx].labels) & set(nodes[right_idx].labels)
        )
        contracted_count = len(contracted)
        if objective in {"memory", "treewidth"}:
            score = (
                intermediate,
                cost,
                -contracted_count,
                -shared,
                left_idx,
                right_idx,
                output_labels,
            )
        else:
            score = (
                -contracted_count,
                intermediate,
                cost,
                -shared,
                left_idx,
                right_idx,
                output_labels,
            )
        candidates.append(
            (
                score,
                (left_idx, right_idx, cost, intermediate, output_labels),
            )
        )
    if not candidates:
        raise ValueError("Tensor network requires at least two nodes to contract.")
    candidates.sort(key=lambda item: item[0])
    if random_state is None or random_pool_size <= 1:
        return candidates[0][1]
    pool_size = min(int(random_pool_size), len(candidates))
    rank = min(
        pool_size - 1,
        int((random_state.random() ** 2) * pool_size),
    )
    return candidates[rank][1]


def _slice_nodes(
    nodes: Sequence[TensorNetworkNode],
    assignments: Mapping[int, int],
) -> tuple[TensorNetworkNode, ...]:
    sliced_nodes: list[TensorNetworkNode] = []
    for node in nodes:
        tensor = node.tensor
        labels = list(node.labels)
        for label, value in assignments.items():
            if label not in labels:
                continue
            axis = labels.index(label)
            tensor = tensor.select(axis, int(value))
            labels.pop(axis)
        sliced_nodes.append(
            TensorNetworkNode(
                tensor=tensor,
                labels=tuple(labels),
                name=node.name,
                metadata=node.metadata,
            )
        )
    return tuple(sliced_nodes)


def _canonicalize_unit_extent_nodes(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
) -> tuple[tuple[TensorNetworkNode, ...], tuple[int, ...]]:
    """Remove mathematically inert size-one axes and absorb scalar nodes."""

    normalized: list[TensorNetworkNode] = []
    scalar: torch.Tensor | None = None
    for node in nodes:
        tensor = node.tensor
        labels: list[int] = []
        for axis in range(tensor.ndim - 1, -1, -1):
            if int(tensor.shape[axis]) == 1:
                tensor = tensor.select(axis, 0)
            else:
                labels.insert(0, int(node.labels[axis]))
        if tensor.ndim == 0:
            scalar = tensor if scalar is None else scalar * tensor
            continue
        normalized.append(
            TensorNetworkNode(
                tensor=tensor,
                labels=tuple(labels),
                name=node.name,
                metadata=node.metadata,
            )
        )
    if not normalized:
        if scalar is None:
            raise ValueError("cannot canonicalize an empty tensor network")
        normalized.append(TensorNetworkNode(tensor=scalar, labels=(), name="scalar"))
    elif scalar is not None:
        first = normalized[0]
        normalized[0] = TensorNetworkNode(
            tensor=first.tensor * scalar,
            labels=first.labels,
            name=first.name,
            metadata=first.metadata,
        )
    dims = _label_dims(nodes)
    normalized_output = tuple(
        int(label) for label in output_labels if int(dims[label]) != 1
    )
    return tuple(normalized), normalized_output


def _internal_slice_candidates(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
) -> tuple[int, ...]:
    output_set = set(output_labels)
    counts = _label_counts(nodes)
    return tuple(
        label
        for label, count in counts.items()
        if count >= 2 and label not in output_set
    )


def _cost_for_sliced_labels(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    labels: Sequence[int],
    *,
    contraction_strategy: str = "greedy",
    beam_width: int = 8,
) -> dict[str, int]:
    subnodes = _slice_nodes(nodes, {label: 0 for label in labels})
    dims = _label_dims(subnodes)
    output_size = _product(dims[label] for label in output_labels)
    if contraction_strategy == "beam":
        _, steps = _contract_nodes_beam(
            subnodes, output_labels, dry_run=True, beam_width=beam_width
        )
    elif contraction_strategy == "quality_multistart":
        steps = _contract_nodes_quality_multistart(subnodes, output_labels)
    else:
        _, steps = _contract_nodes_greedy(subnodes, output_labels, dry_run=True)
    return {
        "estimated_cost": sum(step.estimated_cost for step in steps),
        "peak_size": max(
            (step.intermediate_size for step in steps),
            default=output_size,
        ),
    }


def _pair_steps_from_dynamic_path(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    path: Sequence[Sequence[int]],
) -> tuple[PairContractionStep, ...]:
    """Convert a dynamic pair-position path into stable FlagQuantum steps."""

    active = list(nodes)
    final_outputs = set(int(label) for label in output_labels)
    steps: list[PairContractionStep] = []
    for step_index, pair in enumerate(path):
        if len(pair) != 2:
            raise ValueError("contraction path entries require exactly two positions")
        left_index, right_index = (int(pair[0]), int(pair[1]))
        if not (
            0 <= left_index < len(active)
            and 0 <= right_index < len(active)
            and left_index != right_index
        ):
            raise ValueError("contraction path position is out of range")
        dims = _label_dims(active)
        counts = _label_counts(active)
        left = active[left_index]
        right = active[right_index]
        cost, intermediate, pair_outputs, _ = _estimate_pair(
            left,
            right,
            dims=dims,
            label_counts=counts,
            final_outputs=final_outputs,
        )
        output_shape = tuple(dims[label] for label in pair_outputs)
        steps.append(
            PairContractionStep(
                step=step_index,
                left=left.name,
                right=right.name,
                left_labels=left.labels,
                right_labels=right.labels,
                output_labels=pair_outputs,
                output_shape=output_shape,
                estimated_cost=cost,
                intermediate_size=intermediate,
            )
        )
        for index in sorted((left_index, right_index), reverse=True):
            active.pop(index)
        active.append(
            TensorNetworkNode(
                tensor=_dry_run_tensor(output_shape, left.tensor),
                labels=pair_outputs,
                name=f"({left.name},{right.name})",
            )
        )
    if len(active) != 1:
        raise ValueError("contraction path did not reduce the network to one value")
    return tuple(steps)


def _dynamic_path_from_pair_steps(
    nodes: Sequence[TensorNetworkNode],
    steps: Sequence[PairContractionStep],
) -> tuple[tuple[int, int], ...]:
    active = list(nodes)
    path: list[tuple[int, int]] = []
    for step in steps:
        left_index = next(
            (
                index
                for index, node in enumerate(active)
                if node.name == step.left and node.labels == step.left_labels
            ),
            None,
        )
        right_index = next(
            (
                index
                for index, node in enumerate(active)
                if index != left_index
                and node.name == step.right
                and node.labels == step.right_labels
            ),
            None,
        )
        if left_index is None or right_index is None:
            raise ValueError(
                "external contraction path does not match its source graph"
            )
        path.append((left_index, right_index))
        for index in sorted((left_index, right_index), reverse=True):
            active.pop(index)
        active.append(
            TensorNetworkNode(
                tensor=_dry_run_tensor(step.output_shape, nodes[0].tensor),
                labels=step.output_labels,
                name=f"({step.left},{step.right})",
            )
        )
    return tuple(path)


def _reslice_external_slicing_plan(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    slicing: TensorNetworkSlicingPlan,
    *,
    target_slices: int,
) -> TensorNetworkSlicingPlan:
    """Increase path parallelism while retaining the imported contraction order."""

    if target_slices < 1:
        raise ValueError("target_slices must be positive")
    if not slicing.contraction_path:
        raise ValueError("external reslicing requires an embedded contraction path")
    if slicing.n_slices >= int(target_slices):
        return slicing
    if slicing.canonicalize_unit_extent_labels:
        nodes, output_labels = _canonicalize_unit_extent_nodes(nodes, output_labels)
    dims = _label_dims(nodes)
    selected = list(slicing.sliced_labels)
    initial_subnodes = _slice_nodes(nodes, {label: 0 for label in selected})
    dynamic_path = _dynamic_path_from_pair_steps(
        initial_subnodes,
        slicing.contraction_path,
    )
    candidates = set(_internal_slice_candidates(nodes, output_labels)) - set(selected)
    best_steps = slicing.contraction_path
    while candidates and _product(dims[label] for label in selected) < int(
        target_slices
    ):
        best: (
            tuple[tuple[int, int, int], int, tuple[PairContractionStep, ...]] | None
        ) = None
        for label in candidates:
            trial_labels = selected + [label]
            subnodes = _slice_nodes(nodes, {item: 0 for item in trial_labels})
            trial_steps = _pair_steps_from_dynamic_path(
                subnodes,
                output_labels,
                dynamic_path,
            )
            trial_slices = _product(dims[item] for item in trial_labels)
            trial_cost = sum(step.estimated_cost for step in trial_steps) * trial_slices
            trial_peak = max(
                (step.intermediate_size for step in trial_steps),
                default=1,
            )
            score = (trial_cost, trial_peak, int(label))
            if best is None or score < best[0]:
                best = (score, int(label), trial_steps)
        if best is None:
            break
        _, chosen, best_steps = best
        selected.append(chosen)
        candidates.remove(chosen)
    slice_shape = tuple(dims[label] for label in selected)
    n_slices = _product(slice_shape)
    if n_slices < int(target_slices):
        raise ValueError(
            f"external path can expose only {n_slices} slices, below target {target_slices}"
        )
    per_slice_cost = sum(step.estimated_cost for step in best_steps)
    peak_size = max((step.intermediate_size for step in best_steps), default=1)
    return replace(
        slicing,
        sliced_labels=tuple(selected),
        slice_shape=slice_shape,
        n_slices=n_slices,
        per_slice_cost=per_slice_cost,
        total_estimated_cost=per_slice_cost * n_slices,
        peak_size=peak_size,
        recomputation_factor=(
            float(per_slice_cost * n_slices) / float(slicing.baseline_estimated_cost)
            if slicing.baseline_estimated_cost
            else 1.0
        ),
        peak_bytes=peak_size * slicing.element_size_bytes,
        contraction_path=best_steps,
        contraction_path_source=f"{slicing.contraction_path_source}+adaptive_reslice",
    )


def _cotengra_slicing_plan(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    target_peak_elements: int,
    max_repeats: int = 16,
    minimize: str = "write",
    parallel: bool | str = False,
    methods: Sequence[str] | None = None,
    seed: int = 0,
    target_slices: int | None = None,
) -> TensorNetworkSlicingPlan:
    """Jointly search a bounded contraction tree and slicing with cotengra."""

    if int(target_peak_elements) < 1:
        raise ValueError("target_peak_elements must be positive")
    if int(max_repeats) < 1:
        raise ValueError("max_repeats must be positive")
    try:
        import cotengra as ctg
    except ImportError as error:
        raise ImportError(
            "cotengra planning requires the optional 'cotengra' package"
        ) from error
    nodes, output_labels = _canonicalize_unit_extent_nodes(nodes, output_labels)
    dims = _label_dims(nodes)
    inputs = tuple(tuple(int(label) for label in node.labels) for node in nodes)
    output = tuple(int(label) for label in output_labels)
    if tuple(methods or ()) == ("greedy",):
        optimizer = ctg.GreedyOptimizer(temperature=0.0)
        dynamic_path = optimizer(inputs, output, dims)
        tree = ctg.ContractionTree.from_path(
            inputs=inputs,
            output=output,
            size_dict=dims,
            path=dynamic_path,
        )
        if target_slices is not None and int(target_slices) > 1:
            tree.slice_(
                target_slices=int(target_slices),
                temperature=0.0,
                minimize=str(minimize),
                seed=int(seed),
                inplace=True,
            )
        elif tree.max_size() > int(target_peak_elements):
            tree.slice_and_reconfigure_(target_size=int(target_peak_elements))
    else:
        optimizer = ctg.HyperOptimizer(
            methods=None if methods is None else tuple(methods),
            minimize=str(minimize),
            max_repeats=int(max_repeats),
            parallel=parallel,
            progbar=False,
            slicing_reconf_opts={"target_size": int(target_peak_elements)},
            optlib="random",
            optlib_opts={"seed": int(seed)},
        )
        tree = optimizer.search(inputs=inputs, output=output, size_dict=dims)
        if target_slices is not None and tree.nslices < int(target_slices):
            tree.slice_(
                target_slices=int(target_slices),
                temperature=0.0,
                minimize=str(minimize),
                seed=int(seed),
                inplace=True,
            )
    sliced_labels = _normalize_sliced_labels(
        nodes,
        output_labels,
        tuple(int(label) for label in tree.sliced_inds),
    )
    slice_shape = tuple(dims[label] for label in sliced_labels)
    n_slices = _product(slice_shape) if slice_shape else 1
    subnodes = _slice_nodes(nodes, {label: 0 for label in sliced_labels})
    steps = _pair_steps_from_dynamic_path(
        subnodes,
        output_labels,
        tree.get_path(),
    )
    peak_size = max(
        (step.intermediate_size for step in steps),
        default=_product(dims[label] for label in output_labels),
    )
    per_slice_cost = sum(step.estimated_cost for step in steps)
    if peak_size > int(target_peak_elements):
        raise RuntimeError(
            "cotengra returned a contraction path above the requested peak budget"
        )
    element_size = max((int(node.tensor.element_size()) for node in nodes), default=1)
    # Do not fall back to the native planner here: large networks use cotengra
    # precisely because native multi-start planning can dominate or exhaust
    # host memory.  Unslicing the same cotengra tree gives a cheap, topology-
    # matched recomputation baseline.
    baseline_cost = int(tree.unslice_all().total_flops())
    total_cost = per_slice_cost * n_slices
    return TensorNetworkSlicingPlan(
        sliced_labels=sliced_labels,
        slice_shape=slice_shape,
        n_slices=n_slices,
        per_slice_cost=per_slice_cost,
        total_estimated_cost=total_cost,
        peak_size=peak_size,
        target_peak_size=int(target_peak_elements),
        baseline_estimated_cost=baseline_cost,
        recomputation_factor=(
            float(total_cost) / float(baseline_cost) if baseline_cost else 1.0
        ),
        budget_satisfied=True,
        element_size_bytes=element_size,
        peak_bytes=peak_size * element_size,
        target_peak_bytes=int(target_peak_elements) * element_size,
        contraction_path=steps,
        contraction_path_source=f"cotengra:{getattr(ctg, '__version__', 'unknown')}",
        canonicalize_unit_extent_labels=True,
    )


def _profile_from_steps(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    strategy: str,
    steps: Sequence[PairContractionStep],
    n_slices: int = 1,
    sliced_labels: Sequence[int] = (),
) -> TensorNetworkContractionProfile:
    dims = _label_dims(nodes)
    output_size = _product(dims[label] for label in output_labels)
    peak_size = max((step.intermediate_size for step in steps), default=output_size)
    return TensorNetworkContractionProfile(
        strategy=strategy,
        estimated_cost=sum(step.estimated_cost for step in steps) * int(n_slices),
        peak_size=peak_size,
        n_steps=len(steps),
        output_size=output_size,
        n_slices=int(n_slices),
        sliced_labels=tuple(int(label) for label in sliced_labels),
        total_intermediate_size=sum(step.intermediate_size for step in steps)
        * int(n_slices),
    )


def _contraction_profile(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    strategy: str = "greedy",
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    beam_width: int = 8,
) -> TensorNetworkContractionProfile:
    cache_key = _profile_cache_key(
        nodes,
        output_labels,
        strategy=strategy,
        max_intermediate_size=max_intermediate_size,
        max_intermediate_bytes=max_intermediate_bytes,
        sliced_labels=sliced_labels,
        beam_width=beam_width,
    )
    cached = _CONTRACTION_PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if strategy == "einsum":
        dims = _label_dims(nodes)
        profile = TensorNetworkContractionProfile(
            strategy="einsum",
            estimated_cost=_product(dims.values()),
            peak_size=max((node.tensor.numel() for node in nodes), default=0),
            n_steps=max(0, len(nodes) - 1),
            output_size=_product(dims[label] for label in output_labels),
        )
        _CONTRACTION_PROFILE_CACHE[cache_key] = profile
        return profile
    if strategy in {"sliced", "auto_sliced", "beam_sliced", "quality_sliced"}:
        sliced_strategy = {
            "beam_sliced": "beam",
            "quality_sliced": "quality_multistart",
        }.get(strategy, "greedy")
        slicing = _build_slicing_plan(
            nodes,
            output_labels,
            max_intermediate_size=max_intermediate_size,
            max_intermediate_bytes=max_intermediate_bytes,
            sliced_labels=sliced_labels,
            contraction_strategy=sliced_strategy,
            beam_width=beam_width,
        )
        profile = TensorNetworkContractionProfile(
            strategy=strategy,
            estimated_cost=slicing.total_estimated_cost,
            peak_size=slicing.peak_size,
            n_steps=max(0, len(nodes) - 1),
            output_size=_product(_label_dims(nodes)[label] for label in output_labels),
            n_slices=slicing.n_slices,
            sliced_labels=slicing.sliced_labels,
            total_intermediate_size=slicing.peak_size * slicing.n_slices,
        )
        _CONTRACTION_PROFILE_CACHE[cache_key] = profile
        return profile
    if strategy == "beam":
        _, steps = _contract_nodes_beam(
            nodes, output_labels, dry_run=True, beam_width=beam_width
        )
        profile = _profile_from_steps(
            nodes, output_labels, strategy=strategy, steps=steps
        )
        _CONTRACTION_PROFILE_CACHE[cache_key] = profile
        return profile
    if strategy == "quality_multistart":
        steps = _contract_nodes_quality_multistart(nodes, output_labels)
        profile = _profile_from_steps(
            nodes, output_labels, strategy=strategy, steps=steps
        )
        _CONTRACTION_PROFILE_CACHE[cache_key] = profile
        return profile
    if strategy == "quality_reconfigured":
        steps = _contract_nodes_quality_reconfigured(nodes, output_labels)
        profile = _profile_from_steps(
            nodes, output_labels, strategy=strategy, steps=steps
        )
        _CONTRACTION_PROFILE_CACHE[cache_key] = profile
        return profile
    if strategy == "optimal":
        _, steps = _contract_nodes_optimal(nodes, output_labels, dry_run=True)
        profile = _profile_from_steps(
            nodes, output_labels, strategy=strategy, steps=steps
        )
        _CONTRACTION_PROFILE_CACHE[cache_key] = profile
        return profile
    if strategy not in {"greedy", "memory_greedy", "quality_greedy"}:
        raise ValueError(
            "strategy must be 'greedy', 'memory_greedy', 'quality_greedy', 'quality_multistart', 'quality_reconfigured', 'beam', 'optimal', 'einsum', 'sliced', 'auto_sliced', or 'beam_sliced'."
        )
    objective = {
        "greedy": "balanced",
        "memory_greedy": "memory",
        "quality_greedy": "quality",
    }[strategy]
    _, steps = _contract_nodes_greedy(
        nodes, output_labels, dry_run=True, objective=objective
    )
    profile = _profile_from_steps(nodes, output_labels, strategy=strategy, steps=steps)
    _CONTRACTION_PROFILE_CACHE[cache_key] = profile
    return profile


def _profile_cache_key(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    strategy: str,
    max_intermediate_size: int | None,
    max_intermediate_bytes: int | None,
    sliced_labels: Sequence[int] | None,
    beam_width: int,
) -> tuple[Any, ...]:
    node_key = tuple(
        (
            node.labels,
            tuple(int(dim) for dim in node.tensor.shape),
            str(node.tensor.dtype),
        )
        for node in nodes
    )
    return (
        node_key,
        tuple(output_labels),
        str(strategy),
        max_intermediate_size,
        max_intermediate_bytes,
        tuple(sliced_labels or ()),
        int(beam_width),
    )


def _normalize_sliced_labels(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    sliced_labels: Sequence[int],
) -> tuple[int, ...]:
    dims = _label_dims(nodes)
    output_set = set(output_labels)
    normalized = tuple(dict.fromkeys(int(label) for label in sliced_labels))
    missing = [label for label in normalized if label not in dims]
    if missing:
        raise ValueError(f"Cannot slice unknown tensor-network labels: {missing}.")
    public = [label for label in normalized if label in output_set]
    if public:
        raise ValueError(f"Cannot slice output tensor-network labels: {public}.")
    return normalized


def _auto_slice_labels(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    max_intermediate_size: int,
    *,
    contraction_strategy: str = "greedy",
    beam_width: int = 8,
) -> tuple[int, ...]:
    if contraction_strategy == "quality_multistart":
        return _auto_slice_labels_from_peak(
            nodes,
            output_labels,
            max_intermediate_size,
        )
    selected: list[int] = []
    candidates = list(_internal_slice_candidates(nodes, output_labels))
    dims = _label_dims(nodes)
    current = _cost_for_sliced_labels(
        nodes,
        output_labels,
        selected,
        contraction_strategy=contraction_strategy,
        beam_width=beam_width,
    )
    while candidates and current["peak_size"] > max_intermediate_size:
        best: tuple[int, int, int] | None = None
        for label in candidates:
            trial_labels = selected + [label]
            trial = _cost_for_sliced_labels(
                nodes,
                output_labels,
                trial_labels,
                contraction_strategy=contraction_strategy,
                beam_width=beam_width,
            )
            n_slices = _product(dims[item] for item in trial_labels)
            score = (
                trial["peak_size"],
                trial["estimated_cost"] * n_slices,
                label,
            )
            if best is None or score < best:
                best = score
        if best is None:
            break
        chosen = best[-1]
        selected.append(chosen)
        candidates.remove(chosen)
        current = _cost_for_sliced_labels(
            nodes,
            output_labels,
            selected,
            contraction_strategy=contraction_strategy,
            beam_width=beam_width,
        )
    return tuple(selected)


def _parallel_slice_labels(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    world_size: int,
) -> tuple[int, ...]:
    """Choose slice axes that minimize rank-local contraction work."""

    target_slices = max(1, int(world_size))
    if target_slices == 1:
        return ()
    selected: list[int] = []
    candidates = set(_internal_slice_candidates(nodes, output_labels))
    dims = _label_dims(nodes)
    baseline_steps = _contract_nodes_quality_multistart(nodes, output_labels)
    contraction_tree = _tree_from_steps(nodes, baseline_steps)
    current_steps = baseline_steps
    n_slices = 1
    while candidates and n_slices < target_slices:
        best: tuple[tuple[int, int, int, int], int] | None = None
        for label in candidates:
            dimension = dims[label]
            per_slice_cost = sum(
                (
                    (step.estimated_cost + dimension - 1) // dimension
                    if label in step.left_labels or label in step.right_labels
                    else step.estimated_cost
                )
                for step in current_steps
            )
            peak_size = max(
                (
                    (
                        (step.intermediate_size + dimension - 1) // dimension
                        if label in step.output_labels
                        else step.intermediate_size
                    )
                    for step in current_steps
                ),
                default=1,
            )
            trial_slices = n_slices * dimension
            waves = (trial_slices + target_slices - 1) // target_slices
            rank_cost = per_slice_cost * waves
            total_cost = per_slice_cost * trial_slices
            score = (
                rank_cost,
                total_cost,
                peak_size,
                label,
            )
            candidate = (score, label)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            break
        chosen = best[1]
        selected.append(chosen)
        candidates.remove(chosen)
        n_slices *= dims[chosen]
        subnodes = _slice_nodes(nodes, {label: 0 for label in selected})
        current_steps = _linearize_contraction_tree(
            contraction_tree,
            subnodes,
            output_labels,
        )
    if n_slices < target_slices:
        raise ValueError(
            f"cannot create {target_slices} parallel slices from internal TN edges"
        )
    final_nodes = _slice_nodes(nodes, {label: 0 for label in selected})
    _QUALITY_MULTISTART_PATH_CACHE[
        _quality_multistart_cache_key(final_nodes, output_labels)
    ] = current_steps
    return tuple(selected)


def _auto_slice_labels_from_peak(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    max_intermediate_size: int,
) -> tuple[int, ...]:
    """Choose a memory-feasible fixed-tree slice set with bounded overhead."""

    selected: list[int] = []
    internal = set(_internal_slice_candidates(nodes, output_labels))
    original_dims = _label_dims(nodes)
    baseline_steps = _contract_nodes_quality_multistart(nodes, output_labels)
    contraction_tree = _tree_from_steps(nodes, baseline_steps)
    max_sliced_labels = 64
    final_nodes: tuple[TensorNetworkNode, ...] = tuple(nodes)
    final_steps = baseline_steps
    for _ in range(max_sliced_labels + 1):
        subnodes = _slice_nodes(nodes, {label: 0 for label in selected})
        steps = _linearize_contraction_tree(
            contraction_tree,
            subnodes,
            output_labels,
        )
        output_size = _product(_label_dims(subnodes)[label] for label in output_labels)
        peak_size = max(
            (step.intermediate_size for step in steps),
            default=output_size,
        )
        peak_size = max(output_size, peak_size)
        if peak_size <= max_intermediate_size:
            _QUALITY_MULTISTART_PATH_CACHE[
                _quality_multistart_cache_key(subnodes, output_labels)
            ] = steps
            return tuple(selected)

        # Only labels active in an oversized intermediate can improve the
        # current feasibility frontier.  Project each candidate through the
        # same tree, then minimize exact peak first and replicated FLOPs second.
        candidates = {
            label
            for step in steps
            if step.intermediate_size > int(max_intermediate_size)
            for label in step.output_labels
            if (
                label in internal and label not in selected and original_dims[label] > 1
            )
        }
        best: (
            tuple[
                tuple[int, int, int],
                int,
            ]
            | None
        ) = None
        for label in candidates:
            trial_labels = (*selected, label)
            dimension = original_dims[label]
            trial_peak = max(
                (
                    (
                        (step.intermediate_size + dimension - 1) // dimension
                        if label in step.output_labels
                        else step.intermediate_size
                    )
                    for step in steps
                ),
                default=output_size,
            )
            per_slice_cost = sum(
                (
                    (step.estimated_cost + dimension - 1) // dimension
                    if label in step.left_labels or label in step.right_labels
                    else step.estimated_cost
                )
                for step in steps
            )
            n_slices = _product(original_dims[item] for item in trial_labels)
            total_cost = per_slice_cost * n_slices
            candidate = (
                (trial_peak, total_cost, label),
                label,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            break
        _, chosen = best
        selected.append(chosen)
        final_nodes = _slice_nodes(nodes, {label: 0 for label in selected})
        final_steps = _linearize_contraction_tree(
            contraction_tree,
            final_nodes,
            output_labels,
        )
    final_peak = max(
        (step.intermediate_size for step in final_steps),
        default=_product(_label_dims(final_nodes)[label] for label in output_labels),
    )
    if final_peak > max_intermediate_size:
        _QUALITY_MULTISTART_PATH_CACHE[
            _quality_multistart_cache_key(final_nodes, output_labels)
        ] = final_steps
        return tuple(selected)
    _QUALITY_MULTISTART_PATH_CACHE[
        _quality_multistart_cache_key(final_nodes, output_labels)
    ] = final_steps
    return tuple(selected)


def _build_slicing_plan(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    contraction_strategy: str = "quality_multistart",
    beam_width: int = 8,
) -> TensorNetworkSlicingPlan:
    if contraction_strategy not in {"greedy", "beam", "quality_multistart"}:
        raise ValueError(
            "contraction_strategy must be 'greedy', 'beam', or " "'quality_multistart'."
        )
    if max_intermediate_size is not None and int(max_intermediate_size) < 1:
        raise ValueError("max_intermediate_size must be >= 1 element.")
    if max_intermediate_bytes is not None and int(max_intermediate_bytes) < 1:
        raise ValueError("max_intermediate_bytes must be >= 1 byte.")
    element_size_bytes = max(
        (int(node.tensor.element_size()) for node in nodes),
        default=1,
    )
    byte_limited_size = (
        None
        if max_intermediate_bytes is None
        else int(max_intermediate_bytes) // element_size_bytes
    )
    if byte_limited_size is not None and byte_limited_size < 1:
        raise ValueError(
            f"max_intermediate_bytes={int(max_intermediate_bytes)} cannot hold "
            f"one {element_size_bytes}-byte tensor element."
        )
    effective_max_size = max_intermediate_size
    if byte_limited_size is not None:
        effective_max_size = (
            byte_limited_size
            if effective_max_size is None
            else min(int(effective_max_size), byte_limited_size)
        )
    dims = _label_dims(nodes)
    baseline = _cost_for_sliced_labels(
        nodes,
        output_labels,
        (),
        contraction_strategy=contraction_strategy,
        beam_width=beam_width,
    )
    if sliced_labels is None:
        labels = ()
        if effective_max_size is not None:
            labels = _auto_slice_labels(
                nodes,
                output_labels,
                int(effective_max_size),
                contraction_strategy=contraction_strategy,
                beam_width=beam_width,
            )
    else:
        labels = _normalize_sliced_labels(nodes, output_labels, sliced_labels)
    slice_shape = tuple(dims[label] for label in labels)
    n_slices = _product(slice_shape) if slice_shape else 1
    per_slice = _cost_for_sliced_labels(
        nodes,
        output_labels,
        labels,
        contraction_strategy=contraction_strategy,
        beam_width=beam_width,
    )
    total_estimated_cost = per_slice["estimated_cost"] * n_slices
    budget_satisfied = effective_max_size is None or per_slice["peak_size"] <= int(
        effective_max_size
    )
    if not budget_satisfied:
        mode = "automatic" if sliced_labels is None else "explicit"
        raise ValueError(
            f"{mode} tensor-network slicing cannot satisfy "
            f"effective peak budget={int(effective_max_size)} elements; "
            f"best planned peak is {per_slice['peak_size']} elements after "
            f"{n_slices} slices."
        )
    return TensorNetworkSlicingPlan(
        sliced_labels=labels,
        slice_shape=slice_shape,
        n_slices=n_slices,
        per_slice_cost=per_slice["estimated_cost"],
        total_estimated_cost=total_estimated_cost,
        peak_size=per_slice["peak_size"],
        target_peak_size=effective_max_size,
        baseline_estimated_cost=baseline["estimated_cost"],
        recomputation_factor=(
            float(total_estimated_cost) / float(baseline["estimated_cost"])
            if baseline["estimated_cost"]
            else 1.0
        ),
        budget_satisfied=budget_satisfied,
        element_size_bytes=element_size_bytes,
        peak_bytes=per_slice["peak_size"] * element_size_bytes,
        target_peak_bytes=max_intermediate_bytes,
    )


def _contract_nodes_sliced(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    contraction_strategy: str = "greedy",
    beam_width: int = 8,
) -> tuple[torch.Tensor, TensorNetworkSlicingPlan]:
    slicing = _build_slicing_plan(
        nodes,
        output_labels,
        max_intermediate_size=max_intermediate_size,
        max_intermediate_bytes=max_intermediate_bytes,
        sliced_labels=sliced_labels,
        contraction_strategy=contraction_strategy,
        beam_width=beam_width,
    )
    if not slicing.sliced_labels:
        if contraction_strategy == "beam":
            result, _ = _contract_nodes_beam(
                nodes, output_labels, beam_width=beam_width
            )
        elif contraction_strategy == "quality_multistart":
            steps = _contract_nodes_quality_multistart(nodes, output_labels)
            result = _execute_pair_steps(nodes, output_labels, steps)
        else:
            result, _ = _contract_nodes_greedy(nodes, output_labels)
        return result, slicing

    dims = _label_dims(nodes)
    result: torch.Tensor | None = None
    compensation: torch.Tensor | None = None
    value_ranges = [range(dims[label]) for label in slicing.sliced_labels]
    for values in product(*value_ranges):
        assignments = dict(zip(slicing.sliced_labels, values))
        subnodes = _slice_nodes(nodes, assignments)
        if contraction_strategy == "beam":
            subtotal, _ = _contract_nodes_beam(
                subnodes, output_labels, beam_width=beam_width
            )
        elif contraction_strategy == "quality_multistart":
            steps = _contract_nodes_quality_multistart(subnodes, output_labels)
            subtotal = _execute_pair_steps(subnodes, output_labels, steps)
        else:
            subtotal, _ = _contract_nodes_greedy(subnodes, output_labels)
        if result is None:
            result = subtotal
            compensation = torch.zeros_like(subtotal)
        else:
            if compensation is None:
                raise RuntimeError("slice reduction compensation is unavailable")
            corrected = subtotal - compensation
            updated = result + corrected
            compensation = (updated - result) - corrected
            result = updated
    if result is None:
        raise ValueError("Sliced tensor network did not produce any slices.")
    return result, slicing


def _contract_nodes_with_slicing_plan(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    slicing: TensorNetworkSlicingPlan,
) -> torch.Tensor:
    """Execute an already planned native or external sliced contraction."""

    if not slicing.budget_satisfied:
        raise ValueError("cannot execute a slicing plan with an unsatisfied budget")
    source_nodes = tuple(nodes)
    source_outputs = tuple(output_labels)
    original_dims = _label_dims(source_nodes)
    original_output_shape = tuple(original_dims[label] for label in source_outputs)
    if slicing.canonicalize_unit_extent_labels:
        source_nodes, source_outputs = _canonicalize_unit_extent_nodes(
            source_nodes, source_outputs
        )
    dims = _label_dims(source_nodes)
    expected_shape = tuple(dims[label] for label in slicing.sliced_labels)
    if expected_shape != slicing.slice_shape:
        raise ValueError("slicing plan shape does not match the tensor network")
    value_ranges = [range(dims[label]) for label in slicing.sliced_labels]
    assignments = product(*value_ranges) if value_ranges else ((),)
    result: torch.Tensor | None = None
    compensation: torch.Tensor | None = None
    for values in assignments:
        subnodes = _slice_nodes(source_nodes, dict(zip(slicing.sliced_labels, values)))
        if slicing.contraction_path:
            subtotal = _execute_pair_steps(
                subnodes, source_outputs, slicing.contraction_path
            )
        else:
            subtotal, _ = _contract_nodes_greedy(subnodes, source_outputs)
        if result is None:
            result = subtotal
            compensation = torch.zeros_like(subtotal)
        else:
            corrected = subtotal - compensation
            updated = result + corrected
            compensation = (updated - result) - corrected
            result = updated
    if result is None:
        raise ValueError("sliced tensor-network contraction produced no slices")
    return result.reshape(original_output_shape)


def _contract_nodes_greedy(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: bool = False,
    objective: str = "balanced",
    random_state: random.Random | None = None,
    random_pool_size: int = 1,
) -> tuple[torch.Tensor, tuple[PairContractionStep, ...]]:
    if objective not in {"balanced", "memory", "quality", "treewidth"}:
        raise ValueError(
            "objective must be 'balanced', 'memory', 'quality', or 'treewidth'."
        )
    path_key = _profile_cache_key(
        nodes,
        output_labels,
        strategy=f"execute_{objective}",
        max_intermediate_size=None,
        max_intermediate_bytes=None,
        sliced_labels=None,
        beam_width=0,
    )
    cached_path = (
        None if random_state is not None else _CONTRACTION_PATH_CACHE.get(path_key)
    )
    planned_path: list[tuple[int, int, tuple[int, ...]]] = []
    active = list(nodes)
    final_outputs = set(output_labels)
    steps: list[PairContractionStep] = []
    if not active:
        raise ValueError("Cannot contract an empty tensor network.")
    if (
        cached_path is not None
        and not dry_run
        and active[0].tensor.is_cuda
        and len(active) > 1
    ):
        stages = _CONTRACTION_STAGE_CACHE.get(path_key)
        if stages is None:
            stages = _compile_contraction_stages(active, cached_path)
            _CONTRACTION_STAGE_CACHE[path_key] = stages
        final = _execute_contraction_stages(active, stages)
        final_tensor = final.tensor
        if final.labels != tuple(output_labels):
            final_tensor = _einsum_reorder_by_labels(
                final_tensor, final.labels, output_labels
            )
        return final_tensor, ()
    step_index = 0
    while len(active) > 1:
        dims = _label_dims(active)
        counts = _label_counts(active)
        if cached_path is not None:
            left_idx, right_idx, pair_outputs = cached_path[step_index]
            cost, intermediate, expected_outputs, _ = _estimate_pair(
                active[left_idx],
                active[right_idx],
                dims=dims,
                label_counts=counts,
                final_outputs=final_outputs,
            )
            if pair_outputs != expected_outputs:
                raise RuntimeError("cached tensor-network contraction path is stale")
        else:
            left_idx, right_idx, cost, intermediate, pair_outputs = _choose_greedy_pair(
                active,
                dims=dims,
                label_counts=counts,
                final_outputs=final_outputs,
                objective=objective,
                random_state=random_state,
                random_pool_size=random_pool_size,
            )
            planned_path.append((left_idx, right_idx, pair_outputs))
        left = active[left_idx]
        right = active[right_idx]
        output_shape = tuple(dims[label] for label in pair_outputs)
        steps.append(
            PairContractionStep(
                step=step_index,
                left=left.name,
                right=right.name,
                left_labels=left.labels,
                right_labels=right.labels,
                output_labels=pair_outputs,
                output_shape=output_shape,
                estimated_cost=cost,
                intermediate_size=intermediate,
            )
        )
        if dry_run:
            tensor = _dry_run_tensor(output_shape, left.tensor)
        else:
            if cached_path is None and left.tensor.is_cuda:
                tensor = complex_einsum_pair(
                    _pair_equation(left.labels, right.labels, pair_outputs),
                    left.tensor,
                    right.tensor,
                    compile_cuda=False,
                )
            else:
                tensor = _einsum_pair_by_labels(
                    left.tensor,
                    left.labels,
                    right.tensor,
                    right.labels,
                    pair_outputs,
                )
        new_node = TensorNetworkNode(
            tensor=tensor,
            labels=pair_outputs,
            name=f"({left.name},{right.name})",
        )
        for idx in sorted((left_idx, right_idx), reverse=True):
            active.pop(idx)
        active.append(new_node)
        step_index += 1
    if cached_path is None and random_state is None:
        _CONTRACTION_PATH_CACHE[path_key] = tuple(planned_path)
    final = active[0]
    if final.labels != tuple(output_labels):
        if dry_run:
            dims = _label_dims(active)
            final_tensor = _dry_run_tensor(
                tuple(dims[label] for label in output_labels),
                final.tensor,
            )
        else:
            final_tensor = _einsum_reorder_by_labels(
                final.tensor, final.labels, output_labels
            )
    else:
        final_tensor = final.tensor
    return final_tensor, tuple(steps)


def _contract_nodes_quality_multistart(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    repeats: int = 4,
    seed: int = 0,
    random_pool_size: int = 4,
) -> tuple[PairContractionStep, ...]:
    if repeats < 1:
        raise ValueError("quality multistart repeats must be positive")
    if random_pool_size < 1:
        raise ValueError("quality multistart pool size must be positive")
    cache_key = _quality_multistart_cache_key(
        nodes,
        output_labels,
        repeats=repeats,
        seed=seed,
        random_pool_size=random_pool_size,
    )
    cached = _QUALITY_MULTISTART_PATH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    candidates: list[tuple[tuple[int, int, int], tuple[PairContractionStep, ...]]] = []
    _, baseline = _contract_nodes_greedy(
        nodes,
        output_labels,
        dry_run=True,
        objective="treewidth" if len(nodes) >= 64 else "quality",
    )
    baseline_cost = sum(step.estimated_cost for step in baseline)
    baseline_peak = max((step.intermediate_size for step in baseline), default=0)
    candidates.append(
        (
            (
                baseline_peak,
                baseline_cost,
                sum(step.intermediate_size for step in baseline),
            ),
            baseline,
        )
    )
    for repeat in range(int(repeats)):
        _, steps = _contract_nodes_greedy(
            nodes,
            output_labels,
            dry_run=True,
            objective="quality",
            random_state=random.Random(int(seed) + repeat),
            random_pool_size=int(random_pool_size),
        )
        candidates.append(
            (
                (
                    max((step.intermediate_size for step in steps), default=0),
                    sum(step.estimated_cost for step in steps),
                    sum(step.intermediate_size for step in steps),
                ),
                steps,
            )
        )
    best = min(candidates, key=lambda item: item[0])[1]
    _QUALITY_MULTISTART_PATH_CACHE[cache_key] = best
    return best


def _quality_multistart_cache_key(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    repeats: int = 4,
    seed: int = 0,
    random_pool_size: int = 4,
) -> tuple[Any, ...]:
    return (
        _profile_cache_key(
            nodes,
            output_labels,
            strategy="quality_multistart",
            max_intermediate_size=None,
            max_intermediate_bytes=None,
            sliced_labels=None,
            beam_width=0,
        ),
        int(repeats),
        int(seed),
        int(random_pool_size),
    )


@dataclass(frozen=True)
class _TemporaryContractionTree:
    leaves: frozenset[int]
    leaf_index: int | None = None
    left: "_TemporaryContractionTree | None" = None
    right: "_TemporaryContractionTree | None" = None


@dataclass
class _TemporaryTreeValue:
    tree: _TemporaryContractionTree
    node: TensorNetworkNode


def _find_tree_value(
    active: Sequence[_TemporaryTreeValue],
    *,
    name: str,
    labels: tuple[int, ...],
    excluded: int | None = None,
) -> int:
    for index, value in enumerate(active):
        if excluded is not None and index == excluded:
            continue
        if value.node.name == name and value.node.labels == labels:
            return index
    raise ValueError("contraction path cannot be reconstructed as a binary tree")


def _tree_from_steps(
    nodes: Sequence[TensorNetworkNode],
    steps: Sequence[PairContractionStep],
) -> _TemporaryContractionTree:
    active = [
        _TemporaryTreeValue(
            tree=_TemporaryContractionTree(
                leaves=frozenset((index,)),
                leaf_index=index,
            ),
            node=node,
        )
        for index, node in enumerate(nodes)
    ]
    for step in steps:
        left_index = _find_tree_value(
            active,
            name=step.left,
            labels=step.left_labels,
        )
        right_index = _find_tree_value(
            active,
            name=step.right,
            labels=step.right_labels,
            excluded=left_index,
        )
        left = active[left_index]
        right = active[right_index]
        tree = _TemporaryContractionTree(
            leaves=left.tree.leaves | right.tree.leaves,
            left=left.tree,
            right=right.tree,
        )
        for index in sorted((left_index, right_index), reverse=True):
            active.pop(index)
        active.append(
            _TemporaryTreeValue(
                tree=tree,
                node=TensorNetworkNode(
                    tensor=_dry_run_tensor(
                        step.output_shape,
                        left.node.tensor,
                    ),
                    labels=step.output_labels,
                    name=f"({left.node.name},{right.node.name})",
                ),
            )
        )
    if len(active) != 1:
        raise ValueError("contraction path tree did not reduce to one root")
    return active[0].tree


def _linearize_contraction_tree(
    root: _TemporaryContractionTree,
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
) -> tuple[PairContractionStep, ...]:
    active = [
        _TemporaryTreeValue(
            tree=_TemporaryContractionTree(
                leaves=frozenset((index,)),
                leaf_index=index,
            ),
            node=node,
        )
        for index, node in enumerate(nodes)
    ]
    steps: list[PairContractionStep] = []

    def contract(tree: _TemporaryContractionTree) -> _TemporaryTreeValue:
        if tree.leaf_index is not None:
            return next(value for value in active if value.tree.leaves == tree.leaves)
        if tree.left is None or tree.right is None:
            raise ValueError("internal contraction tree node requires two children")
        left = contract(tree.left)
        right = contract(tree.right)
        dims = _label_dims(tuple(value.node for value in active))
        counts = _label_counts(tuple(value.node for value in active))
        cost, intermediate, labels, _ = _estimate_pair(
            left.node,
            right.node,
            dims=dims,
            label_counts=counts,
            final_outputs=set(output_labels),
        )
        result = _TemporaryTreeValue(
            tree=tree,
            node=TensorNetworkNode(
                tensor=_dry_run_tensor(
                    tuple(dims[label] for label in labels),
                    left.node.tensor,
                ),
                labels=labels,
                name=f"({left.node.name},{right.node.name})",
            ),
        )
        steps.append(
            PairContractionStep(
                step=len(steps),
                left=left.node.name,
                right=right.node.name,
                left_labels=left.node.labels,
                right_labels=right.node.labels,
                output_labels=labels,
                output_shape=tuple(dims[label] for label in labels),
                estimated_cost=cost,
                intermediate_size=intermediate,
            )
        )
        active.remove(left)
        active.remove(right)
        active.append(result)
        return result

    contract(root)
    return tuple(steps)


def _tree_nodes_postorder(
    root: _TemporaryContractionTree,
) -> tuple[_TemporaryContractionTree, ...]:
    if root.leaf_index is not None:
        return ()
    if root.left is None or root.right is None:
        raise ValueError("internal contraction tree node requires two children")
    return (
        *_tree_nodes_postorder(root.left),
        *_tree_nodes_postorder(root.right),
        root,
    )


def _replace_tree_subtree(
    root: _TemporaryContractionTree,
    replacement: _TemporaryContractionTree,
) -> _TemporaryContractionTree:
    if root.leaves == replacement.leaves:
        return replacement
    if root.leaf_index is not None:
        return root
    if root.left is None or root.right is None:
        raise ValueError("internal contraction tree node requires two children")
    return _TemporaryContractionTree(
        leaves=root.leaves,
        left=_replace_tree_subtree(root.left, replacement),
        right=_replace_tree_subtree(root.right, replacement),
    )


def _subtree_boundary_labels(
    leaves: frozenset[int],
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
) -> tuple[int, ...]:
    total_counts = _label_counts(nodes)
    inside = [nodes[index] for index in sorted(leaves)]
    inside_counts = _label_counts(inside)
    output = set(output_labels)
    return tuple(
        label
        for node in inside
        for label in node.labels
        if label in output
        or inside_counts[label] < total_counts[label]
        or total_counts[label] == 1
    )


def _path_score(
    steps: Sequence[PairContractionStep],
) -> tuple[int, int, int]:
    return (
        max((step.intermediate_size for step in steps), default=0),
        sum(step.estimated_cost for step in steps),
        sum(step.intermediate_size for step in steps),
    )


def _contract_nodes_quality_reconfigured(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    max_subtree_leaves: int = 7,
    max_reconfiguration_nodes: int = 64,
) -> tuple[PairContractionStep, ...]:
    baseline = _contract_nodes_quality_multistart(nodes, output_labels)
    if len(nodes) > int(max_reconfiguration_nodes):
        return baseline
    root = _tree_from_steps(nodes, baseline)
    current_steps = baseline
    candidate_leaves = tuple(
        tree.leaves
        for tree in _tree_nodes_postorder(root)
        if 3 <= len(tree.leaves) <= int(max_subtree_leaves)
    )
    for leaves in candidate_leaves:
        subnodes = tuple(nodes[index] for index in sorted(leaves))
        boundary = tuple(
            dict.fromkeys(_subtree_boundary_labels(leaves, nodes, output_labels))
        )
        _, exact_steps = _contract_nodes_optimal(
            subnodes,
            boundary,
            dry_run=True,
            max_nodes=max_subtree_leaves,
        )
        replacement_local = _tree_from_steps(subnodes, exact_steps)
        ordered_leaves = tuple(sorted(leaves))

        def remap(tree: _TemporaryContractionTree) -> _TemporaryContractionTree:
            if tree.leaf_index is not None:
                global_index = ordered_leaves[tree.leaf_index]
                return _TemporaryContractionTree(
                    leaves=frozenset((global_index,)),
                    leaf_index=global_index,
                )
            if tree.left is None or tree.right is None:
                raise ValueError("internal contraction tree node requires two children")
            left = remap(tree.left)
            right = remap(tree.right)
            return _TemporaryContractionTree(
                leaves=left.leaves | right.leaves,
                left=left,
                right=right,
            )

        proposal_root = _replace_tree_subtree(root, remap(replacement_local))
        proposal_steps = _linearize_contraction_tree(
            proposal_root,
            nodes,
            output_labels,
        )
        if _path_score(proposal_steps) < _path_score(current_steps):
            root = proposal_root
            current_steps = proposal_steps
    return current_steps


def _contract_nodes_beam(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: bool = False,
    beam_width: int = 8,
) -> tuple[torch.Tensor, tuple[PairContractionStep, ...]]:
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1.")
    if not nodes:
        raise ValueError("Cannot contract an empty tensor network.")
    final_outputs = set(output_labels)
    beams: list[
        tuple[int, int, int, list[TensorNetworkNode], list[PairContractionStep]]
    ] = [(0, 0, 0, list(nodes), [])]
    while len(beams[0][3]) > 1:
        candidates: list[
            tuple[int, int, int, list[TensorNetworkNode], list[PairContractionStep]]
        ] = []
        for total_cost, peak, _, active, steps in beams:
            dims = _label_dims(active)
            counts = _label_counts(active)
            for left_idx in range(len(active)):
                for right_idx in range(left_idx + 1, len(active)):
                    left = active[left_idx]
                    right = active[right_idx]
                    cost, intermediate, pair_outputs, _ = _estimate_pair(
                        left,
                        right,
                        dims=dims,
                        label_counts=counts,
                        final_outputs=final_outputs,
                    )
                    output_shape = tuple(dims[label] for label in pair_outputs)
                    step = PairContractionStep(
                        step=len(steps),
                        left=left.name,
                        right=right.name,
                        left_labels=left.labels,
                        right_labels=right.labels,
                        output_labels=pair_outputs,
                        output_shape=output_shape,
                        estimated_cost=cost,
                        intermediate_size=intermediate,
                    )
                    if dry_run:
                        tensor = _dry_run_tensor(output_shape, left.tensor)
                    else:
                        tensor = _einsum_pair_by_labels(
                            left.tensor,
                            left.labels,
                            right.tensor,
                            right.labels,
                            pair_outputs,
                        )
                    new_node = TensorNetworkNode(
                        tensor=tensor,
                        labels=pair_outputs,
                        name=f"({left.name},{right.name})",
                    )
                    new_active = list(active)
                    for idx in sorted((left_idx, right_idx), reverse=True):
                        new_active.pop(idx)
                    new_active.append(new_node)
                    new_cost = total_cost + cost
                    new_peak = max(peak, intermediate)
                    candidates.append(
                        (new_cost, new_peak, len(steps) + 1, new_active, steps + [step])
                    )
        candidates.sort(key=lambda item: (item[1], item[0], item[2]))
        beams = candidates[: int(beam_width)]
    best = min(beams, key=lambda item: (item[1], item[0], item[2]))
    final = best[3][0]
    if final.labels != tuple(output_labels):
        dims = _label_dims(best[3])
        if dry_run:
            final_tensor = torch.empty(
                tuple(dims[label] for label in output_labels),
                dtype=final.tensor.dtype,
                device=final.tensor.device,
            )
        else:
            final_tensor = _einsum_reorder_by_labels(
                final.tensor, final.labels, output_labels
            )
    else:
        final_tensor = final.tensor
    return final_tensor, tuple(best[4])


def _contract_nodes_optimal(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: bool = False,
    max_nodes: int = 7,
) -> tuple[torch.Tensor, tuple[PairContractionStep, ...]]:
    if len(nodes) > int(max_nodes):
        return _contract_nodes_beam(
            nodes, output_labels, dry_run=dry_run, beam_width=16
        )
    if not nodes:
        raise ValueError("Cannot contract an empty tensor network.")
    if len(nodes) == 1:
        return _contract_nodes_greedy(nodes, output_labels, dry_run=dry_run)
    final_outputs = set(output_labels)

    def search(
        active: tuple[TensorNetworkNode, ...],
        steps: tuple[PairContractionStep, ...],
        cost_so_far: int,
        peak_so_far: int,
    ) -> tuple[
        int, int, tuple[PairContractionStep, ...], tuple[TensorNetworkNode, ...]
    ]:
        if len(active) == 1:
            return peak_so_far, cost_so_far, steps, active
        dims = _label_dims(active)
        counts = _label_counts(active)
        best: (
            tuple[
                int, int, tuple[PairContractionStep, ...], tuple[TensorNetworkNode, ...]
            ]
            | None
        ) = None
        for left_idx in range(len(active)):
            for right_idx in range(left_idx + 1, len(active)):
                left = active[left_idx]
                right = active[right_idx]
                cost, intermediate, pair_outputs, _ = _estimate_pair(
                    left,
                    right,
                    dims=dims,
                    label_counts=counts,
                    final_outputs=final_outputs,
                )
                output_shape = tuple(dims[label] for label in pair_outputs)
                step = PairContractionStep(
                    step=len(steps),
                    left=left.name,
                    right=right.name,
                    left_labels=left.labels,
                    right_labels=right.labels,
                    output_labels=pair_outputs,
                    output_shape=output_shape,
                    estimated_cost=cost,
                    intermediate_size=intermediate,
                )
                if dry_run:
                    tensor = _dry_run_tensor(output_shape, left.tensor)
                else:
                    tensor = _einsum_pair_by_labels(
                        left.tensor,
                        left.labels,
                        right.tensor,
                        right.labels,
                        pair_outputs,
                    )
                new_node = TensorNetworkNode(
                    tensor=tensor,
                    labels=pair_outputs,
                    name=f"({left.name},{right.name})",
                )
                new_active = list(active)
                for idx in sorted((left_idx, right_idx), reverse=True):
                    new_active.pop(idx)
                new_active.append(new_node)
                candidate = search(
                    tuple(new_active),
                    steps + (step,),
                    cost_so_far + cost,
                    max(peak_so_far, intermediate),
                )
                if best is None or (candidate[0], candidate[1], len(candidate[2])) < (
                    best[0],
                    best[1],
                    len(best[2]),
                ):
                    best = candidate
        if best is None:
            raise ValueError("Tensor network requires at least two nodes to contract.")
        return best

    _, _, steps, final_nodes = search(tuple(nodes), (), 0, 0)
    final = final_nodes[0]
    if final.labels != tuple(output_labels):
        dims = _label_dims(final_nodes)
        if dry_run:
            final_tensor = _dry_run_tensor(
                tuple(dims[label] for label in output_labels),
                final.tensor,
            )
        else:
            final_tensor = _einsum_reorder_by_labels(
                final.tensor, final.labels, output_labels
            )
    else:
        final_tensor = final.tensor
    return final_tensor, steps


def _clone_nodes_with_offset(
    nodes: Sequence[TensorNetworkNode],
    *,
    offset: int,
    conjugate: bool = False,
) -> tuple[TensorNetworkNode, ...]:
    out = []
    for node in nodes:
        out.append(
            TensorNetworkNode(
                tensor=torch.conj(node.tensor) if conjugate else node.tensor,
                labels=tuple(label + offset for label in node.labels),
                name=("bra:" if conjugate else "ket:") + node.name,
                metadata=node.metadata,
            )
        )
    return tuple(out)
