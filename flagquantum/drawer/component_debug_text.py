# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Text graph diagnostics follow Qualtran v0.7.0 CompositeBloq.debug_text.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.py

"""Format typed component connections by dependency generation."""

from collections import defaultdict

from ..compiler.registers.component_expansion import _occurrence_labels
from ..compiler.registers.component_graph import _call_entry
from ..core.ir.program import ProgramRecord, ValueRef
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from .component_score import component_register_score


def format_component_summary(record: ProgramRecord, entry: str = "main") -> str:
    """Summarize the immediate component layer, including isolated calls."""
    root = _call_entry(record, entry)
    return f"ComponentProgram([{len(root.blocks[0].operations)} components...])"


def _port_labels(registers: tuple[Register, ...]) -> tuple[str, ...]:
    return tuple(
        register.name + ("[" + ", ".join(map(str, index)) + "]" if index else "")
        for register in registers
        for index in register.all_idxs()
    )


def format_component_debug_text(
    record: ProgramRecord,
    entry: str = "main",
    *,
    instance_ids: tuple[int, ...] | None = None,
) -> str:
    """Render call nodes and both sides of each register connection.

    The finite, unconditional typed-call profile matches component_register_score.
    Explicit instance_ids retain labels returned by selective call expansion.
    """
    score = component_register_score(record, entry)
    functions = {function.name: function for function in record.functions}
    root = functions[entry]
    signature = Signature(root.registers or ())
    block = root.blocks[0]
    labels = _occurrence_labels(root, instance_ids)
    known: dict[ValueRef, tuple[str, str]] = {
        value: ("LeftDangle", port)
        for value, port in zip(
            block.arguments, _port_labels(tuple(signature.lefts())), strict=True
        )
    }
    nodes = {
        position: f"{operation.callee}#{labels[position]}"
        for position, operation in enumerate(block.operations)
    }
    inputs: dict[str, dict[str, list[str]]] = defaultdict(dict)
    outputs: dict[str, dict[str, list[str]]] = defaultdict(dict)

    def connect(value: ValueRef, target: str, target_port: str) -> None:
        source, source_port = known[value]
        inputs[target].setdefault(source, []).append(
            f"  {source}.{source_port} -> {target_port}"
        )
        outputs[source].setdefault(target, []).append(
            f"  {source_port} -> {target}.{target_port}"
        )

    # Connection insertion order controls adjacency grouping within generations.
    for position, operation in enumerate(block.operations):
        callee = functions[operation.callee or ""]
        callee_signature = Signature(callee.registers or ())
        node = nodes[position]
        for value, port in zip(
            operation.operands,
            _port_labels(tuple(callee_signature.lefts())),
            strict=True,
        ):
            connect(value, node, port)
        for value, port in zip(
            operation.results,
            _port_labels(tuple(callee_signature.rights())),
            strict=True,
        ):
            known[value] = (node, port)
    for value, port in zip(
        block.return_values or (), _port_labels(tuple(signature.rights())), strict=True
    ):
        connect(value, "RightDangle", port)

    generations: dict[int, list[str]] = defaultdict(list)
    for column in score.columns:
        node = nodes[column.ordinal]
        lines = generations[column.generation]
        lines.append(node)
        lines.extend(line for group in inputs[node].values() for line in group)
        lines.extend(line for group in outputs[node].values() for line in group)
    return ("\n" + "-" * 20 + "\n").join(
        "\n".join(generations[generation]) for generation in sorted(generations)
    )
