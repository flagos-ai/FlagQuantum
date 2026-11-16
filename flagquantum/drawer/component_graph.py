# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PrettyGraphDrawer and TypedGraphDrawer.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/graphviz.py

"""Draw component connectivity with width or typed port annotations."""

import json
from collections.abc import Sequence

from ..core.ir.program import ProgramRecord
from ..core.ir.program.registers import Register, RegisterType
from ..core.ir.program.signatures import Signature
from .component_score import _label, component_register_score
from .register_graph import RegisterConnection, _export_register_graph_dot
from .register_graph_table import pretty_connection_attributes, signature_instance_table
from .register_score import _register_groups


def export_component_graph_dot(
    record: ProgramRecord, entry: str = "main", *, typed: bool = False
) -> str:
    """Draw each owned call instance and preserve both endpoint types."""
    score = component_register_score(record, entry)
    signatures = {
        function.name: Signature(function.registers)
        for function in record.functions
        if function.registers is not None
    }
    root = signatures[entry]
    widths = {line.register: line.width for line in score.lines}

    def types(
        keys: Sequence[str], registers: Sequence[Register]
    ) -> dict[str, RegisterType]:
        return {
            key: register.dtype
            for register, elements in _register_groups(keys, registers, widths)
            for key in elements
        }

    inputs: dict[int | None, dict[str, RegisterType]] = {}
    outputs: dict[int | None, dict[str, RegisterType]] = {}
    if typed:
        outputs[None] = types(
            tuple(
                _label(register.name, index)
                for register in root.lefts()
                for index in register.all_idxs()
            ),
            tuple(root.lefts()),
        )
        inputs[None] = types(score.output_order, tuple(root.rights()))
        for column in score.columns:
            step = column.step
            inputs[column.ordinal] = types(step.inputs, step.input_registers)
            outputs[column.ordinal] = types(step.outputs, step.output_registers)

    def attributes(edge: RegisterConnection) -> str:
        if not typed:
            return pretty_connection_attributes(edge)
        left = outputs[edge.producer][edge.register]
        right = inputs[edge.consumer][edge.register]
        color = "black"
        if left == right:
            label = str(left)
        elif left.num_qubits == 1 and not (left.num_cbits or right.num_cbits):
            label = str(left if left.encoding == "bit" else right)
        else:
            label = f"{left} - {right}"
            color = "red"
        return (
            f"label={json.dumps(label, ensure_ascii=False)}, labelfloat=False, "
            f'fontcolor="{color}", fontsize=10, arrowhead="dot", arrowsize=0.25'
        )

    return _export_register_graph_dot(
        score,
        attributes,
        table=lambda ordinal, step, ports: signature_instance_table(
            signatures[step.name], step, ports, pretty=True
        ),
    )
