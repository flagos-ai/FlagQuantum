# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 flatten_for_tensor_contraction.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_flattening.py

"""Select existing component expansion using declared custom tensor leaves."""

from collections.abc import Callable

from ...core.ir.program import ProgramRecord
from .component_expansion import expand_component_calls


def expand_components_for_tensor(
    record: ProgramRecord,
    has_custom_tensor: Callable[[str], bool],
    entry: str = "main",
    *,
    full_flatten: bool = True,
    max_depth: int = 1_000,
) -> tuple[ProgramRecord, tuple[int, ...]]:
    """Expand typed calls fully, or stop at declared custom tensor suppliers.

    The input entry is an existing composite body. Tensor providers identify
    custom callees by name; no foreign Python method-name heuristic enters IR.
    Full expansion never queries the classifier. Partial expansion queries it
    in the existing source traversal order, including atomic calls. Classifier
    errors propagate; atomic/mixed bodies remain leaves under the existing
    expansion rules. Retained instance labels, exact outputs, workspace and the
    depth bound are owned by expand_component_calls. The input stays immutable.

    This transforms a graph only: it does not execute tensor callbacks, resolve
    missing decompositions, construct a root wrapper, or select a simulator.
    """
    return expand_component_calls(
        record,
        entry,
        predicate=(
            None
            if full_flatten
            else lambda _instance, operation: not has_custom_tensor(
                operation.callee or ""
            )
        ),
        max_depth=max_depth,
    )
