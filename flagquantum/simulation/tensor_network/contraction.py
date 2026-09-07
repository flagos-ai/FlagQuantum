"""Native tensor network data structures and contraction runtime."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Mapping, Sequence

import torch

from .models import (
    PairContractionStep,
    TensorNetworkContractionProfile,
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
)
from .path_search import (
    _QUALITY_MULTISTART_PATH_CACHE,
    _contract_nodes_beam,
    _contract_nodes_greedy,
    _contract_nodes_optimal,
    _contract_nodes_quality_multistart,
    _contract_nodes_quality_reconfigured,
    _dry_run_tensor,
    _estimate_pair,
    _label_counts,
    _label_dims,
    _linearize_contraction_tree,
    _product,
    _profile_cache_key,
    _quality_multistart_cache_key,
    _tree_from_steps,
)
from .stages import execute_pair_steps as _execute_pair_steps

_CONTRACTION_PROFILE_CACHE: dict[tuple[Any, ...], TensorNetworkContractionProfile] = {}
_DENSE_Z_OBSERVABLE_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], torch.Tensor
] = {}
_Z_OBSERVABLE_NODE_CACHE: dict[
    tuple[int, tuple[int, ...], str, torch.dtype], tuple[torch.Tensor, ...]
] = {}


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
            "contraction_strategy must be 'greedy', 'beam', or 'quality_multistart'."
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
