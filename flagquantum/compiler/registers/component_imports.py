# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Composition follows Qualtran v0.7.0 BloqBuilder.add and add_from.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.py

"""Prepare immutable component definitions for insertion into a live builder."""

from collections.abc import Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord, verify_program_record


def prepare_component_import(
    record: ProgramRecord,
    entry: str,
    existing: Mapping[str, FunctionRecord],
    caller: str,
    *,
    include_entry: bool,
) -> tuple[ProgramRecord, str]:
    """Reuse equal definitions, isolating conflicting names deterministically."""
    from .component_graph import _namespace_helpers

    diagnostics = verify_program_record(record)
    if diagnostics:
        raise ValueError(f"invalid component import: {diagnostics}")
    functions = {function.name: function for function in record.functions}
    if entry not in functions or functions[entry].registers is None:
        raise ValueError(f"unknown or untyped component entry {entry!r}")

    def conflicts(candidate: ProgramRecord, root: str) -> bool:
        return any(
            (include_entry or function.name != root)
            and (
                function.name == caller
                or function.name in existing
                and existing[function.name] != function
            )
            for function in candidate.functions
        )

    if not conflicts(record, entry):
        return record, entry
    index = 0
    while True:
        prefix = f"__fq_component_{index}"
        candidate = _namespace_helpers(record, prefix)
        root = f"{prefix}_{entry}"
        if not conflicts(candidate, root):
            return candidate, root
        index += 1
