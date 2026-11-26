# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Root follows locked Qualtran v0.7.0 optimization/k_xor_sat/load_kxor_instance.py.

"""Compose the original scope lookup; retain bodyless index/RHS declarations."""

from ..compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from ..compiler.registers.layout.component_partition import partition_component_program
from ..compiler.registers.lookup.qrom_component import qrom_program
from ..core.ir.program import FunctionRecord, ProgramRecord
from .kxor_loading import (
    LoadConstraintScopes,
    LoadUniqueScopeIndex,
    PRGAUniqueConstraintRHS,
)


def kxor_loading_program(
    oracle: LoadConstraintScopes | LoadUniqueScopeIndex | PRGAUniqueConstraintRHS,
) -> ProgramRecord:
    """Keep source RIGHT scope ownership and exact Allocate/QROM/Partition calls.

    The packed output is transferred to the caller, not released as private
    scratch. A consumer must supply its exact cleanup when it is no longer
    needed. Index and RHS have source models but no executable source bodies;
    symbolic lookup roots likewise remain declarations.
    """
    if not isinstance(oracle, LoadConstraintScopes) or oracle.inst.is_symbolic():
        return ProgramRecord(
            (FunctionRecord("main", (), registers=tuple(oracle.signature)),)
        )
    table = oracle.qrom
    targets = tuple(r for r in table.signature if r.name.startswith("target"))
    builder = ComponentProgramBuilder(oracle.signature)
    arguments: dict[str, BindingCollection] = {
        r.name: builder.allocate(dtype=r.dtype) for r in targets
    }
    arguments["selection"] = builder.inputs["j"]
    returned = builder.call_program(
        qrom_program(table, reserve_workspace=True), arguments
    )
    index = returned.pop("selection")
    packed = builder.call_program(
        partition_component_program(targets, "pack_scope", reverse=True),
        returned,
        entry="pack_scope",
    )
    return builder.finalize({"j": index, "U": packed["x"]})
