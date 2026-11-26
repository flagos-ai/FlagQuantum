"""Compare source call operands after zero-gate MSB-first register views."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from flagquantum.compiler.registers.component_graph import (
    _elements,
    iter_component_calls,
)
from flagquantum.core.ir.program import ProgramRecord


def assert_source_bit_view_calls(
    original: Any, program: ProgramRecord, entry: str = "main"
) -> None:
    """Retain every allocation, computational stage and free in visitation order.

    Split/Join/Cast/Partition only repack bits, so compare physical bit lineage
    instead of their different word-level SSA handles. Stage outputs receive
    new lineage; identities cannot hide reordered inputs or released work.
    """
    from qualtran._infra.composite_bloq import BloqBuilder

    definitions = {f.name: f for f in program.functions}
    function = definitions[entry]
    native_values: dict[object, tuple[tuple[str, int], ...]] = {}
    source_values: dict[object, tuple[tuple[str, int], ...]] = {}

    def assign(
        values: dict[object, tuple[tuple[str, int], ...]],
        keys: Iterable[tuple[object, int]],
        kind: str,
    ) -> None:
        offset = 0
        for key, width in keys:
            values[key] = tuple((kind, i) for i in range(offset, offset + width))
            offset += width

    assign(
        native_values,
        zip(
            function.blocks[0].arguments,
            (
                int(r.bitsize)
                for r in function.registers or ()
                for _ in r.all_idxs()
                if r.side.value & 1
            ),
            strict=True,
        ),
        "entry",
    )
    _, source_inputs = BloqBuilder.from_signature(original.signature)
    assign(
        source_values,
        (
            (soq, int(soq.reg.bitsize))
            for collection in source_inputs.values()
            for soq in np.asarray(collection).flat
        ),
        "entry",
    )
    actual, expected = [], []
    for index, inputs, outputs in iter_component_calls(program, entry):
        child = definitions[function.blocks[0].operations[index].callee or ""]
        opcodes = {op.opcode for block in child.blocks for op in block.operations}
        tokens = tuple(
            token
            for value in _elements(inputs.values())
            for token in native_values[value.value]
        )
        returned = tuple(_elements(outputs))
        widths = tuple(int(value.register.bitsize) for value in returned)
        if opcodes and opcodes <= {"split_register", "join_register", "cast_register"}:
            offset = 0
            for value, width in zip(returned, widths, strict=True):
                native_values[value.value] = tokens[offset : offset + width]
                offset += width
            assert offset == len(tokens)
            continue
        kind = (
            "allocate"
            if "allocate_register" in opcodes
            else "free" if "free_register" in opcodes else "stage"
        )
        actual.append((kind, tokens, sum(widths)))
        assign(
            native_values,
            (
                (value.value, width)
                for value, width in zip(returned, widths, strict=True)
            ),
            str(len(actual)),
        )
    for instance, inputs, outputs in original.iter_bloqsoqs():
        kind = type(instance.bloq).__name__
        tokens = tuple(
            token
            for reg in instance.bloq.signature.lefts()
            for soq in np.asarray(inputs[reg.name]).flat
            for token in source_values[soq]
        )
        source_returned = tuple(
            soq for collection in outputs for soq in np.asarray(collection).flat
        )
        widths = tuple(int(soq.reg.bitsize) for soq in source_returned)
        if kind in ("Split", "Join", "Cast", "Partition"):
            offset = 0
            for soq, width in zip(source_returned, widths, strict=True):
                source_values[soq] = tokens[offset : offset + width]
                offset += width
            assert offset == len(tokens)
            continue
        label = (
            "allocate" if kind == "Allocate" else "free" if kind == "Free" else "stage"
        )
        expected.append((label, tokens, sum(widths)))
        assign(
            source_values, zip(source_returned, widths, strict=True), str(len(expected))
        )
    assert actual == expected
