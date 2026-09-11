"""Tensor-network contraction path search and shape-only planning primitives."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from math import prod
from typing import Any, Literal, Mapping, Sequence, SupportsInt, overload

import torch

from ...core.ir import CircuitIR, ensure_circuit_ir
from ..real_imag_kernels import complex_einsum_pair
from .models import (
    CompiledTNStagePlan,
    PairContractionStep,
    TensorNetworkNode,
)
from .stages import (
    compile_contraction_stages as _compile_contraction_stages,
)
from .stages import (
    einsum_pair_by_labels as _einsum_pair_by_labels,
)
from .stages import (
    einsum_reorder_by_labels as _einsum_reorder_by_labels,
)
from .stages import (
    execute_contraction_stages as _execute_contraction_stages,
)
from .stages import (
    pair_equation as _pair_equation,
)

_CONTRACTION_PATH_CACHE: dict[
    tuple[Any, ...], tuple[tuple[int, int, tuple[int, ...]], ...]
] = {}
_QUALITY_MULTISTART_PATH_CACHE: dict[
    tuple[Any, ...], tuple[PairContractionStep, ...]
] = {}
_CONTRACTION_STAGE_CACHE: dict[tuple[Any, ...], CompiledTNStagePlan] = {}


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
        return self.dtype.itemsize


_SearchTensor = torch.Tensor | _DryRunTensor


@dataclass(frozen=True)
class _SearchNode:
    """Private search intermediate; never passed to numerical kernels directly."""

    tensor: _SearchTensor
    labels: tuple[int, ...]
    name: str = ""


_Node = TensorNetworkNode | _SearchNode


def _real_tensor(value: _SearchTensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("shape-only tensor metadata cannot be executed")
    return value


def _dry_run_tensor(shape: Sequence[int], reference: _SearchTensor) -> _DryRunTensor:
    return _DryRunTensor(
        shape=tuple(int(dim) for dim in shape),
        dtype=reference.dtype,
    )


def _reorder_final_tensor(
    node: _Node,
    output_labels: Sequence[int],
    *,
    dry_run: bool,
) -> _SearchTensor:
    if node.labels == tuple(output_labels):
        return node.tensor
    if dry_run:
        dims = _label_dims((node,))
        return _dry_run_tensor(
            tuple(dims[label] for label in output_labels),
            node.tensor,
        )
    return _einsum_reorder_by_labels(
        _real_tensor(node.tensor), node.labels, output_labels
    )


def _as_ir(program: Any) -> CircuitIR:
    return ensure_circuit_ir(program)


def _product(values: Iterable[SupportsInt]) -> int:
    return prod(int(value) for value in values)


def _label_dims(nodes: Sequence[_Node]) -> dict[int, int]:
    dims: dict[int, int] = {}
    for node in nodes:
        if len(node.labels) != len(node.tensor.shape):
            raise ValueError(
                f"Tensor-network node {node.name!r} has {len(node.labels)} labels "
                f"for {len(node.tensor.shape)} dimensions."
            )
        for label, dim in zip(node.labels, node.tensor.shape):
            dim = int(dim)
            if label in dims and dims[label] != dim:
                raise ValueError(
                    f"Inconsistent dimension for tensor-network label {label}."
                )
            dims[label] = dim
    return dims


def _label_counts(nodes: Sequence[_Node]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node in nodes:
        for label in node.labels:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _pair_output_labels(
    left: _Node,
    right: _Node,
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
    left: _Node,
    right: _Node,
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
    nodes: Sequence[_Node],
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
            tuple[int, int, int, int, int, int, tuple[int, ...]],
            tuple[int, int, int, int, tuple[int, ...]],
        ]
    ] = []
    candidate_pairs: Iterable[tuple[int, int]] = (
        (left_idx, right_idx)
        for left_idx in range(len(nodes))
        for right_idx in range(left_idx + 1, len(nodes))
    )
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
    if random_state is None or random_pool_size <= 1:
        return min(candidates, key=lambda item: item[0])[1]
    candidates.sort(key=lambda item: item[0])
    pool_size = min(int(random_pool_size), len(candidates))
    rank = min(
        pool_size - 1,
        int((random_state.random() ** 2) * pool_size),
    )
    return candidates[rank][1]


def _validate_beam_width(beam_width: int) -> None:
    if type(beam_width) is not int:
        raise ValueError("beam_width must be an integer.")
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1.")


def _profile_cache_key(
    nodes: Sequence[_Node],
    output_labels: Sequence[int],
    *,
    strategy: str,
    max_intermediate_size: int | None,
    max_intermediate_bytes: int | None,
    sliced_labels: Sequence[int] | None,
    beam_width: int,
) -> tuple[Any, ...]:
    if strategy in {"beam", "beam_sliced"}:
        _validate_beam_width(beam_width)
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


@overload
def _contract_nodes_greedy(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: Literal[False] = False,
    objective: str = "balanced",
    random_state: random.Random | None = None,
    random_pool_size: int = 1,
) -> tuple[torch.Tensor, tuple[PairContractionStep, ...]]: ...


@overload
def _contract_nodes_greedy(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: bool,
    objective: str = "balanced",
    random_state: random.Random | None = None,
    random_pool_size: int = 1,
) -> tuple[_SearchTensor, tuple[PairContractionStep, ...]]: ...


def _contract_nodes_greedy(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: bool = False,
    objective: str = "balanced",
    random_state: random.Random | None = None,
    random_pool_size: int = 1,
) -> tuple[_SearchTensor, tuple[PairContractionStep, ...]]:
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
    active: list[_Node] = list(nodes)
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
            stages = _compile_contraction_stages(nodes, cached_path)
            _CONTRACTION_STAGE_CACHE[path_key] = stages
        staged_node = _execute_contraction_stages(nodes, stages)
        staged_tensor = staged_node.tensor
        if staged_node.labels != tuple(output_labels):
            staged_tensor = _einsum_reorder_by_labels(
                staged_tensor, staged_node.labels, output_labels
            )
        return staged_tensor, ()
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
        tensor: _SearchTensor
        if dry_run:
            tensor = _dry_run_tensor(output_shape, left.tensor)
        else:
            if cached_path is None and left.tensor.is_cuda:
                tensor = complex_einsum_pair(
                    _pair_equation(left.labels, right.labels, pair_outputs),
                    _real_tensor(left.tensor),
                    _real_tensor(right.tensor),
                    compile_cuda=False,
                )
            else:
                tensor = _einsum_pair_by_labels(
                    _real_tensor(left.tensor),
                    left.labels,
                    _real_tensor(right.tensor),
                    right.labels,
                    pair_outputs,
                )
        new_node = _SearchNode(
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
    return _reorder_final_tensor(final, output_labels, dry_run=dry_run), tuple(steps)


def _contract_nodes_quality_multistart(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    repeats: int = 4,
    seed: int = 0,
    random_pool_size: int = 4,
) -> tuple[PairContractionStep, ...]:
    if type(repeats) is not int:
        raise ValueError("quality multistart repeats must be an integer")
    if repeats < 1:
        raise ValueError("quality multistart repeats must be positive")
    if type(random_pool_size) is not int:
        raise ValueError("quality multistart pool size must be an integer")
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
    for repeat in range(repeats):
        _, steps = _contract_nodes_greedy(
            nodes,
            output_labels,
            dry_run=True,
            objective="quality",
            random_state=random.Random(int(seed) + repeat),
            random_pool_size=random_pool_size,
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
    nodes: Sequence[_Node],
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
    node: _Node


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
    nodes: Sequence[_Node],
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
                node=_SearchNode(
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
    nodes: Sequence[_Node],
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
            node=_SearchNode(
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
    nodes: Sequence[_Node],
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


@overload
def _contract_nodes_beam(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: Literal[False] = False,
    beam_width: int = 8,
) -> tuple[torch.Tensor, tuple[PairContractionStep, ...]]: ...


@overload
def _contract_nodes_beam(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: bool,
    beam_width: int = 8,
) -> tuple[_SearchTensor, tuple[PairContractionStep, ...]]: ...


def _contract_nodes_beam(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: bool = False,
    beam_width: int = 8,
) -> tuple[_SearchTensor, tuple[PairContractionStep, ...]]:
    _validate_beam_width(beam_width)
    if not nodes:
        raise ValueError("Cannot contract an empty tensor network.")
    final_outputs = set(output_labels)
    beams: list[tuple[int, int, int, list[_Node], list[PairContractionStep]]] = [
        (0, 0, 0, list(nodes), [])
    ]
    while len(beams[0][3]) > 1:
        candidates: list[
            tuple[int, int, int, list[_Node], list[PairContractionStep]]
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
                    tensor: _SearchTensor
                    if dry_run:
                        tensor = _dry_run_tensor(output_shape, left.tensor)
                    else:
                        tensor = _einsum_pair_by_labels(
                            _real_tensor(left.tensor),
                            left.labels,
                            _real_tensor(right.tensor),
                            right.labels,
                            pair_outputs,
                        )
                    new_node = _SearchNode(
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
        beams = candidates[:beam_width]
    best = min(beams, key=lambda item: (item[1], item[0], item[2]))
    final = best[3][0]
    return _reorder_final_tensor(final, output_labels, dry_run=dry_run), tuple(best[4])


@overload
def _contract_nodes_optimal(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: Literal[False] = False,
    max_nodes: int = 7,
) -> tuple[torch.Tensor, tuple[PairContractionStep, ...]]: ...


@overload
def _contract_nodes_optimal(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: bool,
    max_nodes: int = 7,
) -> tuple[_SearchTensor, tuple[PairContractionStep, ...]]: ...


def _contract_nodes_optimal(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    dry_run: bool = False,
    max_nodes: int = 7,
) -> tuple[_SearchTensor, tuple[PairContractionStep, ...]]:
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
        active: tuple[_Node, ...],
        steps: tuple[PairContractionStep, ...],
        cost_so_far: int,
        peak_so_far: int,
    ) -> tuple[int, int, tuple[PairContractionStep, ...], tuple[_Node, ...]]:
        if len(active) == 1:
            return peak_so_far, cost_so_far, steps, active
        dims = _label_dims(active)
        counts = _label_counts(active)
        best: (
            tuple[int, int, tuple[PairContractionStep, ...], tuple[_Node, ...]] | None
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
                tensor: _SearchTensor
                if dry_run:
                    tensor = _dry_run_tensor(output_shape, left.tensor)
                else:
                    tensor = _einsum_pair_by_labels(
                        _real_tensor(left.tensor),
                        left.labels,
                        _real_tensor(right.tensor),
                        right.labels,
                        pair_outputs,
                    )
                new_node = _SearchNode(
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
    return _reorder_final_tensor(final, output_labels, dry_run=dry_run), steps
