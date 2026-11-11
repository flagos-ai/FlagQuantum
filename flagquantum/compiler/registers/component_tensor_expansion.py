# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 flatten_for_tensor_contraction and Bloq.as_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_flattening.py
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/bloq.py

"""Select existing component expansion using declared custom tensor leaves."""

from collections.abc import Callable
from dataclasses import replace

from ...core.ir.program import ProgramRecord, verify_program_record
from ...core.ir.program.signatures import Signature
from .component_builder import ComponentProgramBuilder
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


def expand_component_root_for_tensor(
    record: ProgramRecord,
    has_custom_tensor: Callable[[str], bool],
    entry: str = "main",
    *,
    name: str = "tensor_root",
    full_flatten: bool = True,
    max_depth: int = 1_000,
) -> tuple[ProgramRecord, tuple[int, ...]]:
    """Wrap one typed root call, then apply the existing tensor selection policy.

    This is the native equivalent of Bloq.as_composite_bloq followed by flatten.
    Callers explicitly choose this root entry; already-composite entries use
    expand_components_for_tensor without wrapping. Partial mode can retain the
    root itself, while full mode expands its stored typed-call decomposition.
    Primitive bodies remain leaf calls. Register direction, shape, dtype and
    row-major/MSB-first ordering are preserved by ComponentProgramBuilder.

    Original definitions stay shared and canonical_payload is invalidated.
    Invalid programs, unknown roots, physical/untyped entries and name collisions
    raise ValueError. Classifier and expansion failures propagate unchanged.
    No tensor supplier is executed and no gate, work bit or backend is created.
    """
    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid tensor root graph: {diagnostics}")
    functions = {function.name: function for function in record.functions}
    if entry not in functions:
        raise ValueError(f"unknown tensor root {entry!r}")
    root = functions[entry]
    if record.physical_context is not None or root.registers is None:
        raise ValueError("tensor root wrapping requires a typed logical entry")
    builder = ComponentProgramBuilder(
        Signature(root.registers), record.functions, name=name
    )
    wrapped = builder.finalize(builder.call_dict(entry, builder.inputs))
    wrapped = replace(record, canonical_payload=None, functions=wrapped.functions)
    return expand_components_for_tensor(
        wrapped,
        has_custom_tensor,
        name,
        full_flatten=full_flatten,
        max_depth=max_depth,
    )
