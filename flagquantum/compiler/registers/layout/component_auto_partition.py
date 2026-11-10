# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 AutoPartition.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/auto_partition.py

"""Compose source-ordered typed partitions around an existing Program entry."""

from dataclasses import replace

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_verifier import executable_register_width
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_graph import _call_entry, _namespace_helpers
from .auto_partition import AutoPartition


def _partition_program(
    registers: tuple[Register, ...], name: str, reverse: bool
) -> ProgramRecord:
    width = sum(int(r.total_qubits()) for r in registers)
    word = Register(
        "x",
        RegisterType("opaque", width),
        side=RegisterSide.RIGHT if reverse else RegisterSide.LEFT,
    )
    parts = tuple(
        Register(
            r.name,
            r.dtype,
            r.shape,
            RegisterSide.LEFT if reverse else RegisterSide.RIGHT,
        )
        for r in registers
    )
    builder = ComponentProgramBuilder(
        Signature((*parts, word) if reverse else (word, *parts)), name=name
    )
    outputs: dict[str, BindingCollection] = {}
    if reverse:
        joined_bits: list[RegisterBinding] = []
        for register in registers:
            for binding in np.asarray(builder.inputs[register.name]).flat:
                joined_bits.extend(builder.split(binding))
        outputs["x"] = builder.join(joined_bits, word.dtype)
    else:
        packed = builder.inputs["x"]
        assert isinstance(packed, RegisterBinding)
        bits = builder.split(packed)
        offset = 0
        for register in registers:
            values = np.empty(register.shape, dtype=object)
            n = executable_register_width(register.dtype)
            for index in register.all_idxs():
                values[index] = builder.join(bits[offset : offset + n], register.dtype)
                offset += n
            outputs[register.name] = values if register.shape else values[()]
    return _namespace_helpers(builder.finalize(outputs), name)


def auto_partition_component_program(
    record: ProgramRecord,
    layout: AutoPartition,
    entry: str = "main",
    name: str = "packed",
) -> ProgramRecord:
    """Append a packed entry while preserving the underlying child entry.

    Scalar opaque packed words follow layout.groups in MSB-first order. Child
    array elements are row-major. Named unused slices remain live through one
    child call and are repacked unchanged. left_only exposes the child's right
    ports and consumes the packed inputs. The existing layout owns validation
    of names and unused slices. Only concrete admitted quantum types enter
    Program. Invalid layouts, output sets and entry/name collisions raise
    ValueError. No primitive gate or workspace is added by the partitions.
    """
    child = _call_entry(record, entry)
    assert child.registers is not None
    signature = Signature(child.registers)
    lefts = tuple(signature.lefts())
    rights = tuple(signature.rights())
    if len(lefts) != len(child.registers):
        raise ValueError("AutoPartition requires every child declaration on the left")
    widths = layout._inputs()
    if widths != {r.name: int(r.total_qubits()) for r in lefts}:
        raise ValueError("child input names and widths must match the layout")
    if set(layout.output_names) != {r.name for r in rights}:
        raise ValueError("child output names must match the layout")
    if not layout.left_only and any(
        r.side != RegisterSide.THRU for r in child.registers
    ):
        raise ValueError("repacked AutoPartition requires THRU child ports")
    names = {f.name for f in record.functions}
    if name in names:
        raise ValueError("packed entry conflicts with an existing definition")
    index = 0
    while any(n.startswith(f"__fq_partition_{index}_") for n in names):
        index += 1
    prefix = f"__fq_partition_{index}_"
    external = tuple(
        Register(outer, RegisterType("opaque", sum(width for _, width in ports)))
        for outer, ports in layout._concrete_layouts()
    )
    if layout.left_only:
        public = tuple(
            Register(r.name, r.dtype, r.shape, RegisterSide.LEFT) for r in external
        ) + tuple(
            Register(r.name, r.dtype, r.shape, RegisterSide.RIGHT) for r in rights
        )
    else:
        public = external
    registers = {r.name: r for r in lefts}
    definitions = list(record.functions)
    partitions = []
    for i, (_, group) in enumerate(layout._concrete_layouts()):
        parts = tuple(
            (
                registers[port]
                if port in registers
                else Register(port, RegisterType("opaque", width))
            )
            for port, width in group
        )
        partitions.append(parts)
        definitions.extend(
            _partition_program(parts, prefix + f"split{i}", False).functions
        )
        if not layout.left_only:
            definitions.extend(
                _partition_program(parts, prefix + f"join{i}", True).functions
            )
    builder = ComponentProgramBuilder(Signature(public), definitions, name=name)
    inputs: dict[str, BindingCollection] = {}
    unused: dict[str, BindingCollection] = {}
    for i, outer in enumerate(external):
        bound_parts = builder.call_dict(
            prefix + f"split{i}", {"x": builder.inputs[outer.name]}
        )
        for port, value in bound_parts.items():
            (inputs if port in registers else unused)[port] = value
    outputs = builder.call_dict(entry, inputs)
    if not layout.left_only:
        combined = outputs | unused
        outputs = {
            outer.name: builder.call_dict(
                prefix + f"join{i}", {r.name: combined[r.name] for r in partitions[i]}
            )["x"]
            for i, outer in enumerate(external)
        }
    module = builder.finalize(outputs)
    return replace(
        record,
        canonical_payload=None,
        functions=(
            *record.functions,
            *(f for f in module.functions if f.name not in names),
        ),
    )
