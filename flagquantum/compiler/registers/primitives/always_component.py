# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Always.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/always.py

"""Wrap a native typed child in a single unconditional component call."""

from dataclasses import replace

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def always_component_program(
    record: ProgramRecord, entry: str = "main", name: str = "always"
) -> ProgramRecord:
    """Expose the child's exact signature and call it once with unchanged operands."""
    child = next(
        (function for function in record.functions if function.name == entry), None
    )
    if child is None or child.registers is None or record.physical_context is not None:
        raise ValueError(f"Always requires a typed child definition: {entry!r}")
    builder = ComponentProgramBuilder(
        Signature(child.registers), record.functions, name=name
    )
    wrapped = builder.finalize(builder.call_dict(entry, builder.inputs))
    return replace(record, canonical_payload=None, functions=wrapped.functions)
