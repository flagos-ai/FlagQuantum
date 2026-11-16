# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PrettyGraphDrawer and TypedGraphDrawer.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/graphviz.py

"""Draw component connectivity with width or typed port annotations."""

import json
from collections.abc import Sequence
from dataclasses import replace

from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register, RegisterType
from ..core.ir.program.signatures import Signature
from .component_score import _label, component_register_score
from .register_graph import RegisterConnection, _export_register_graph_dot
from .register_graph_table import (
    _port_label,
    pretty_connection_attributes,
    signature_instance_table,
)
from .register_score import RegisterStep, _register_groups


def _reshaping_label(function: FunctionRecord, signature: Signature) -> str | None:
    """Recognize a pure split or join from the body and directional shapes."""
    lefts, rights = tuple(signature.lefts()), tuple(signature.rights())
    if len(lefts) != 1 or len(rights) != 1 or len(function.blocks) != 1:
        return None
    left, right = lefts[0], rights[0]
    if left.name != right.name:
        return None
    operations = function.blocks[0].operations
    if not operations:
        return None
    first, last = operations[0], operations[-1]
    if (
        not left.shape
        and right.shape == (left.bitsize,)
        and right.dtype.encoding == "bit"
        and first.opcode == "split_register"
        and all(op.opcode == "join_register" for op in operations[1:])
        and first.results == tuple(bit for op in operations[1:] for bit in op.operands)
    ):
        return "Split"
    if (
        not right.shape
        and left.shape == (right.bitsize,)
        and left.dtype.encoding == "bit"
        and last.opcode == "join_register"
        and all(op.opcode == "split_register" for op in operations[:-1])
        and last.operands == tuple(bit for op in operations[:-1] for bit in op.results)
    ):
        return "Join"
    return None


def export_component_graph_dot(
    record: ProgramRecord,
    entry: str = "main",
    *,
    typed: bool = False,
    pretty: bool = True,
) -> str:
    """Draw an owned call graph in base, compact-width or compact-typed form."""
    if typed and not pretty:
        raise ValueError("Typed graph rendering requires compact presentation")
    score = component_register_score(record, entry)
    signatures = {
        function.name: Signature(function.registers)
        for function in record.functions
        if function.registers is not None
    }
    root = signatures[entry]
    reshape = {
        function.name: _reshaping_label(function, signatures[function.name])
        for function in record.functions
        if function.name in signatures
    }
    input_labels = {
        _label(register.name, index): _port_label(register.name, index)
        for register in root.lefts()
        for index in register.all_idxs()
    }
    output_labels = dict(
        zip(
            score.output_order,
            (
                _port_label(register.name, index)
                for register in root.rights()
                for index in register.all_idxs()
            ),
            strict=True,
        )
    )
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
        outputs[None] = types(tuple(input_labels), tuple(root.lefts()))
        inputs[None] = types(score.output_order, tuple(root.rights()))
        for column in score.columns:
            step = column.step
            inputs[column.ordinal] = types(step.inputs, step.input_registers)
            outputs[column.ordinal] = types(step.outputs, step.output_registers)

    def attributes(edge: RegisterConnection) -> str:
        if not typed:
            return (
                pretty_connection_attributes(edge)
                if pretty
                else f'label="{edge.width}"'
            )
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

    def table(ordinal: int, step: RegisterStep, ports: dict[str, str]) -> str:
        signature = signatures[step.name]
        name = reshape[step.name]
        return signature_instance_table(
            signature,
            replace(step, name=name) if name is not None else step,
            ports,
            pretty=pretty,
            reshaping=name is not None,
        )

    return _export_register_graph_dot(
        score,
        attributes,
        table=table,
        input_labels=input_labels,
        output_labels=output_labels,
    )
