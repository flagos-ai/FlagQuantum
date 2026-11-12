# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 classical_sim_graph.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/classical_sim_graph.py

"""Display explicit classical SSA assignments on existing component edges."""

import json
from collections.abc import Mapping

from ..compiler.registers.component_graph import _call_entry
from ..core.ir.program import ProgramRecord, ValueRef
from .component_score import component_register_score
from .register_graph import RegisterConnection, _export_register_graph_dot


def export_classical_component_graph_dot(
    record: ProgramRecord,
    assignments: Mapping[ValueRef, object],
    entry: str = "main",
) -> str:
    """Label every edge with its producer's already computed classical value.

    The caller supplies Simulation assignments; drawing never evaluates an
    operation. Reuse the component score's finite logical admission, register
    names, array order, instance ordinals and boundary geometry. A THRU lane's
    successive SSA results remain distinct even though its display name is reused.
    Missing producer assignments raise KeyError; value stringification failures
    propagate. Neither assignments nor the program is modified.
    """
    score = component_register_score(record, entry)
    block = _call_entry(record, entry).blocks[0]
    producers: dict[tuple[str, int | None], ValueRef] = {
        (line.register, None): ref
        for line, ref in zip(
            (line for line in score.lines if line.start == -1),
            block.arguments,
            strict=True,
        )
    }
    for column in score.columns:
        for name, ref in zip(
            column.step.outputs,
            block.operations[column.ordinal].results,
            strict=True,
        ):
            producers[name, column.ordinal] = ref

    def attributes(edge: RegisterConnection) -> str:
        value = assignments[producers[edge.register, edge.producer]]
        return (
            f"label={json.dumps(str(value))}, labelfloat=true, fontsize=10, "
            'fontcolor="darkblue", arrowhead="dot", arrowsize=0.25'
        )

    return _export_register_graph_dot(score, attributes)
