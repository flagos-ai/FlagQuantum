"""Native tensor network data structures and contraction runtime."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import torch

from ..core.ir import CircuitIR, ensure_circuit_ir
from .real_imag_kernels import complex_einsum_pair

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


from .tensor_models import (  # noqa: E402
    CompiledTNContractionBucket,
    CompiledTNContractionStage,
    CompiledTNStagePlan,
    PairContractionStep,
    TensorNetworkContractionProfile,
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
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
            label in left.labels
            and label in right.labels
            and label_counts.get(label, 0) == 2
            and label not in final_outputs
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
) -> tuple[int, int, int, int, tuple[int, ...]]:
    best_score: tuple[Any, ...] | None = None
    best_result: tuple[int, int, int, int, tuple[int, ...]] | None = None
    for left_idx in range(len(nodes)):
        for right_idx in range(left_idx + 1, len(nodes)):
            cost, intermediate, output_labels, contracted = _estimate_pair(
                nodes[left_idx],
                nodes[right_idx],
                dims=dims,
                label_counts=label_counts,
                final_outputs=final_outputs,
            )
            shared = len(set(nodes[left_idx].labels) & set(nodes[right_idx].labels))
            contracted_count = len(contracted)
            if objective == "memory":
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
            if best_score is None or score < best_score:
                best_score = score
                best_result = (left_idx, right_idx, cost, intermediate, output_labels)
    if best_result is None:
        raise ValueError("Tensor network requires at least two nodes to contract.")
    return best_result


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
    if contraction_strategy == "beam":
        _, steps = _contract_nodes_beam(
            subnodes, output_labels, dry_run=True, beam_width=beam_width
        )
    else:
        _, steps = _contract_nodes_greedy(subnodes, output_labels, dry_run=True)
    return {
        "estimated_cost": sum(step.estimated_cost for step in steps),
        "peak_size": max((step.intermediate_size for step in steps), default=0),
    }


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
    sliced_labels: Sequence[int] | None = None,
    beam_width: int = 8,
) -> TensorNetworkContractionProfile:
    cache_key = _profile_cache_key(
        nodes,
        output_labels,
        strategy=strategy,
        max_intermediate_size=max_intermediate_size,
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
    if strategy in {"sliced", "auto_sliced", "beam_sliced"}:
        slicing = _build_slicing_plan(
            nodes,
            output_labels,
            max_intermediate_size=max_intermediate_size,
            sliced_labels=sliced_labels,
            contraction_strategy="beam" if strategy == "beam_sliced" else "greedy",
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
    if strategy == "optimal":
        _, steps = _contract_nodes_optimal(nodes, output_labels, dry_run=True)
        profile = _profile_from_steps(
            nodes, output_labels, strategy=strategy, steps=steps
        )
        _CONTRACTION_PROFILE_CACHE[cache_key] = profile
        return profile
    if strategy not in {"greedy", "memory_greedy"}:
        raise ValueError(
            "strategy must be 'greedy', 'memory_greedy', 'beam', 'optimal', 'einsum', 'sliced', 'auto_sliced', or 'beam_sliced'."
        )
    objective = "memory" if strategy == "memory_greedy" else "balanced"
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
    selected: list[int] = []
    candidates = list(_internal_slice_candidates(nodes, output_labels))
    current = _cost_for_sliced_labels(
        nodes,
        output_labels,
        selected,
        contraction_strategy=contraction_strategy,
        beam_width=beam_width,
    )
    while candidates and current["peak_size"] > max_intermediate_size:
        best: tuple[int, int, int, int] | None = None
        for label in candidates:
            trial = _cost_for_sliced_labels(
                nodes,
                output_labels,
                selected + [label],
                contraction_strategy=contraction_strategy,
                beam_width=beam_width,
            )
            score = (trial["peak_size"], trial["estimated_cost"], len(selected), label)
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


def _build_slicing_plan(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    max_intermediate_size: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    contraction_strategy: str = "greedy",
    beam_width: int = 8,
) -> TensorNetworkSlicingPlan:
    dims = _label_dims(nodes)
    if sliced_labels is None:
        labels = ()
        if max_intermediate_size is not None:
            labels = _auto_slice_labels(
                nodes,
                output_labels,
                int(max_intermediate_size),
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
    return TensorNetworkSlicingPlan(
        sliced_labels=labels,
        slice_shape=slice_shape,
        n_slices=n_slices,
        per_slice_cost=per_slice["estimated_cost"],
        total_estimated_cost=per_slice["estimated_cost"] * n_slices,
        peak_size=per_slice["peak_size"],
        target_peak_size=max_intermediate_size,
    )


def _contract_nodes_sliced(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    max_intermediate_size: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    contraction_strategy: str = "greedy",
    beam_width: int = 8,
) -> tuple[torch.Tensor, TensorNetworkSlicingPlan]:
    slicing = _build_slicing_plan(
        nodes,
        output_labels,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=sliced_labels,
        contraction_strategy=contraction_strategy,
        beam_width=beam_width,
    )
    if not slicing.sliced_labels:
        if contraction_strategy == "beam":
            result, _ = _contract_nodes_beam(
                nodes, output_labels, beam_width=beam_width
            )
        else:
            result, _ = _contract_nodes_greedy(nodes, output_labels)
        return result, slicing

    dims = _label_dims(nodes)
    result: torch.Tensor | None = None
    value_ranges = [range(dims[label]) for label in slicing.sliced_labels]
    for values in product(*value_ranges):
        assignments = dict(zip(slicing.sliced_labels, values))
        subnodes = _slice_nodes(nodes, assignments)
        if contraction_strategy == "beam":
            subtotal, _ = _contract_nodes_beam(
                subnodes, output_labels, beam_width=beam_width
            )
        else:
            subtotal, _ = _contract_nodes_greedy(subnodes, output_labels)
        result = subtotal if result is None else result + subtotal
    if result is None:
        raise ValueError("Sliced tensor network did not produce any slices.")
    return result, slicing


_LOCAL_EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _einsum_pair_by_labels(
    left_tensor: torch.Tensor,
    left_labels: Sequence[int],
    right_tensor: torch.Tensor,
    right_labels: Sequence[int],
    output_labels: Sequence[int],
) -> torch.Tensor:
    labels = tuple(
        dict.fromkeys(tuple(left_labels) + tuple(right_labels) + tuple(output_labels))
    )
    if len(labels) > len(_LOCAL_EINSUM_CHARS):
        raise ValueError(
            "Pair contraction rank exceeds local torch.einsum label capacity."
        )
    mapping = {label: _LOCAL_EINSUM_CHARS[index] for index, label in enumerate(labels)}
    equation = (
        "".join(mapping[label] for label in left_labels)
        + ","
        + "".join(mapping[label] for label in right_labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )
    return complex_einsum_pair(equation, left_tensor, right_tensor)


def _batched_pair_equation(equation: str) -> str:
    used = set(equation.replace(",", "").replace("-", "").replace(">", ""))
    batch = next(char for char in reversed(_LOCAL_EINSUM_CHARS) if char not in used)
    inputs, output = equation.split("->")
    left, right = inputs.split(",")
    return f"{batch}{left},{batch}{right}->{batch}{output}"


def _pair_equation(
    left_labels: Sequence[int],
    right_labels: Sequence[int],
    output_labels: Sequence[int],
) -> str:
    labels = tuple(
        dict.fromkeys(tuple(left_labels) + tuple(right_labels) + tuple(output_labels))
    )
    mapping = {label: _LOCAL_EINSUM_CHARS[index] for index, label in enumerate(labels)}
    return (
        "".join(mapping[label] for label in left_labels)
        + ","
        + "".join(mapping[label] for label in right_labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )


def _compile_contraction_stages(
    nodes: Sequence[TensorNetworkNode],
    path: tuple[tuple[int, int, tuple[int, ...]], ...],
) -> CompiledTNStagePlan:
    active = [
        (index, node.labels, tuple(node.tensor.shape), 0)
        for index, node in enumerate(nodes)
    ]
    levels: dict[int, list[tuple[Any, ...]]] = {}
    next_id = len(active)
    for left_idx, right_idx, outputs in path:
        left = active[left_idx]
        right = active[right_idx]
        level = max(left[3], right[3]) + 1
        equation = _pair_equation(left[1], right[1], outputs)
        dimensions = dict(zip(right[1], right[2]))
        dimensions.update(zip(left[1], left[2]))
        output_shape = tuple(dimensions[label] for label in outputs)
        levels.setdefault(level, []).append(
            (left[0], right[0], next_id, equation, left[2], right[2], outputs)
        )
        for index in sorted((left_idx, right_idx), reverse=True):
            active.pop(index)
        active.append((next_id, outputs, output_shape, level))
        next_id += 1
    stages = []
    for level in sorted(levels):
        grouped: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
        for operation in levels[level]:
            grouped.setdefault((operation[3], operation[4], operation[5]), []).append(
                operation
            )
        stages.append(
            CompiledTNContractionStage(
                tuple(
                    CompiledTNContractionBucket(
                        equation=equation,
                        batched_equation=_batched_pair_equation(equation),
                        operations=tuple(operations),
                    )
                    for (equation, _, _), operations in grouped.items()
                )
            )
        )
    return CompiledTNStagePlan(tuple(stages), tuple(path[-1][2]))


def _execute_contraction_stages(
    nodes: Sequence[TensorNetworkNode],
    plan: CompiledTNStagePlan,
) -> TensorNetworkNode:
    values = {index: node.tensor for index, node in enumerate(nodes)}
    for stage in plan.stages:
        for bucket in stage.buckets:
            equation = bucket.equation
            operations = bucket.operations
            if len(operations) == 1:
                left_id, right_id, output_id, _, _, _, _ = operations[0]
                values[output_id] = complex_einsum_pair(
                    equation,
                    values.pop(left_id),
                    values.pop(right_id),
                    compile_cuda=False,
                )
                continue
            left = torch.stack([values[item[0]] for item in operations])
            right = torch.stack([values[item[1]] for item in operations])
            outputs = complex_einsum_pair(bucket.batched_equation, left, right)
            for position, operation in enumerate(operations):
                left_id, right_id, output_id, _, _, _, _ = operation
                values.pop(left_id)
                values.pop(right_id)
                values[output_id] = outputs[position]
    if len(values) != 1:
        raise RuntimeError("compiled TN contraction stages left multiple outputs")
    output_id, tensor = next(iter(values.items()))
    return TensorNetworkNode(tensor, plan.output_labels, name=f"stage:{output_id}")


def _einsum_reorder_by_labels(
    tensor: torch.Tensor,
    labels: Sequence[int],
    output_labels: Sequence[int],
) -> torch.Tensor:
    all_labels = tuple(dict.fromkeys(tuple(labels) + tuple(output_labels)))
    if len(all_labels) > len(_LOCAL_EINSUM_CHARS):
        raise ValueError(
            "Final contraction rank exceeds local torch.einsum label capacity."
        )
    mapping = {
        label: _LOCAL_EINSUM_CHARS[index] for index, label in enumerate(all_labels)
    }
    equation = (
        "".join(mapping[label] for label in labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )
    return torch.einsum(equation, tensor)


def _contract_nodes_greedy(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: bool = False,
    objective: str = "balanced",
) -> tuple[torch.Tensor, tuple[PairContractionStep, ...]]:
    if objective not in {"balanced", "memory"}:
        raise ValueError("objective must be 'balanced' or 'memory'.")
    path_key = _profile_cache_key(
        nodes,
        output_labels,
        strategy=f"execute_{objective}",
        max_intermediate_size=None,
        sliced_labels=None,
        beam_width=0,
    )
    cached_path = _CONTRACTION_PATH_CACHE.get(path_key)
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
            tensor = torch.empty(
                output_shape, dtype=left.tensor.dtype, device=left.tensor.device
            )
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
    if cached_path is None:
        _CONTRACTION_PATH_CACHE[path_key] = tuple(planned_path)
    final = active[0]
    if final.labels != tuple(output_labels):
        if dry_run:
            dims = _label_dims(active)
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
    return final_tensor, tuple(steps)


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
                        tensor = torch.empty(
                            output_shape,
                            dtype=left.tensor.dtype,
                            device=left.tensor.device,
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
                    tensor = torch.empty(
                        output_shape, dtype=left.tensor.dtype, device=left.tensor.device
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
