"""Dependency-light tensor contraction and staged-execution primitives."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .real_imag_kernels import complex_einsum_pair
from .tensor_models import (
    CompiledTNContractionBucket,
    CompiledTNContractionStage,
    CompiledTNStagePlan,
    PairContractionStep,
    TensorNetworkNode,
)

_LOCAL_EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def kahan_add(
    total: torch.Tensor | None,
    compensation: torch.Tensor | None,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate one tensor with Kahan compensation."""

    if total is None:
        return value, torch.zeros_like(value)
    if compensation is None:
        raise RuntimeError("Kahan compensation is unavailable")
    corrected = value - compensation
    updated = total + corrected
    return updated, (updated - total) - corrected


def execute_pair_steps(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    steps: Sequence[PairContractionStep],
) -> torch.Tensor:
    """Execute a stable sequence of named pair-contraction steps."""

    active = list(nodes)
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
            raise ValueError("planned contraction step does not match active nodes")
        left = active[left_index]
        right = active[right_index]
        tensor = einsum_pair_by_labels(
            left.tensor,
            left.labels,
            right.tensor,
            right.labels,
            step.output_labels,
        )
        for index in sorted((left_index, right_index), reverse=True):
            active.pop(index)
        active.append(
            TensorNetworkNode(
                tensor=tensor,
                labels=step.output_labels,
                name=f"({left.name},{right.name})",
            )
        )
    if len(active) != 1:
        raise ValueError("planned contraction path did not reduce to one tensor")
    result = active[0]
    if result.labels != tuple(output_labels):
        return einsum_reorder_by_labels(
            result.tensor,
            result.labels,
            output_labels,
        )
    return result.tensor


def einsum_pair_by_labels(
    left_tensor: torch.Tensor,
    left_labels: Sequence[int],
    right_tensor: torch.Tensor,
    right_labels: Sequence[int],
    output_labels: Sequence[int],
) -> torch.Tensor:
    """Contract two tensors whose axes are identified by integer labels."""

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


def einsum_pair_by_labels_with_fallback(
    left_tensor: torch.Tensor,
    left_labels: Sequence[int],
    right_tensor: torch.Tensor,
    right_labels: Sequence[int],
    output_labels: Sequence[int],
) -> tuple[torch.Tensor, bool]:
    """Contract a pair and report use of the high-rank native fallback."""

    normalized_left = tuple(int(label) for label in left_labels)
    normalized_right = tuple(int(label) for label in right_labels)
    normalized_output = tuple(int(label) for label in output_labels)
    try:
        return (
            einsum_pair_by_labels(
                left_tensor,
                normalized_left,
                right_tensor,
                normalized_right,
                normalized_output,
            ),
            False,
        )
    except ValueError as error:
        if "supports at most 8 axes per group" not in str(error):
            raise
    unique = tuple(dict.fromkeys(normalized_left + normalized_right))
    remap = {label: index for index, label in enumerate(unique)}
    if len(remap) > 52:
        raise ValueError("high-rank TN einsum fallback exceeds 52 unique labels")
    return (
        torch.einsum(
            left_tensor,
            [remap[label] for label in normalized_left],
            right_tensor,
            [remap[label] for label in normalized_right],
            [remap[label] for label in normalized_output],
        ),
        True,
    )


def batched_pair_equation(equation: str) -> str:
    """Add a free batch label to a two-input einsum equation."""

    used = set(equation.replace(",", "").replace("-", "").replace(">", ""))
    batch = next(char for char in reversed(_LOCAL_EINSUM_CHARS) if char not in used)
    inputs, output = equation.split("->")
    left, right = inputs.split(",")
    return f"{batch}{left},{batch}{right}->{batch}{output}"


def pair_equation(
    left_labels: Sequence[int],
    right_labels: Sequence[int],
    output_labels: Sequence[int],
) -> str:
    """Build a local einsum equation for integer-labelled tensors."""

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


def compile_contraction_stages(
    nodes: Sequence[TensorNetworkNode],
    path: tuple[tuple[int, int, tuple[int, ...]], ...],
) -> CompiledTNStagePlan:
    """Group independent pair contractions into shape-compatible stages."""

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
        equation = pair_equation(left[1], right[1], outputs)
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
                        batched_equation=batched_pair_equation(equation),
                        operations=tuple(operations),
                    )
                    for (equation, _, _), operations in grouped.items()
                )
            )
        )
    return CompiledTNStagePlan(tuple(stages), tuple(path[-1][2]))


def execute_contraction_stages(
    nodes: Sequence[TensorNetworkNode],
    plan: CompiledTNStagePlan,
) -> TensorNetworkNode:
    """Execute a compiled stage plan, batching shape-compatible operations."""

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


def einsum_reorder_by_labels(
    tensor: torch.Tensor,
    labels: Sequence[int],
    output_labels: Sequence[int],
) -> torch.Tensor:
    """Reorder a tensor to a requested integer-label axis order."""

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
