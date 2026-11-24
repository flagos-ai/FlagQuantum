# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 AutoPartition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/auto_partition.py

"""Compose source-ordered typed partitions around an existing Program entry."""

from dataclasses import replace

from ....core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register, RegisterSide, RegisterType
from ....core.ir.program.signatures import Signature
from ....core.value_math import ssum
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_graph import _namespace_helpers
from .auto_partition import AutoPartition
from .component_partition import partition_component_program, partition_signature


def auto_partition_signature(
    child: Signature,
    layout: AutoPartition,
    *,
    packed_ports: tuple[Register, ...] | None = None,
) -> Signature:
    """Declare typed outer ports and retain symbolic widths without building IR."""
    lefts = tuple(child.lefts())
    if len(lefts) != len(child):
        raise ValueError("AutoPartition requires every child declaration on the left")
    if layout._inputs() != {reg.name: reg.total_qubits() for reg in lefts}:
        raise ValueError("child input names and widths must match the layout")
    groups = tuple(
        (outer, ssum(width for _, width in segments))
        for outer, segments in layout.groups
    )
    if packed_ports is None:
        packed_ports = tuple(
            Register(name, RegisterType("opaque", width)) for name, width in groups
        )
    elif tuple((reg.name, reg.bitsize) for reg in packed_ports) != groups:
        raise ValueError("packed port names, order and widths must match the layout")

    if not layout.left_only:
        return Signature(packed_ports)
    return Signature(
        (
            *(
                Register(reg.name, reg.dtype, reg.shape_symbolic, RegisterSide.LEFT)
                for reg in packed_ports
            ),
            *(
                Register(reg.name, reg.dtype, reg.shape_symbolic, RegisterSide.RIGHT)
                for reg in child.rights()
            ),
        )
    )


def auto_partition_component_program(
    record: ProgramRecord,
    layout: AutoPartition,
    entry: str = "main",
    name: str = "packed",
    *,
    packed_ports: tuple[Register, ...] | None = None,
) -> ProgramRecord:
    """Append a packed entry while preserving the underlying child entry.

    Scalar packed words follow layout.groups in MSB-first order. Child
    array elements are row-major. Named unused slices remain live through one
    child call and are repacked unchanged. left_only exposes the child's right
    ports and consumes the packed inputs. The existing layout owns validation
    of names and unused slices. Only concrete admitted quantum types enter
    physical binding. Symbolic-width partitions remain typed declarations in
    the stored call graph. Primitive typed bodies are retained without requiring a call decomposition.
    A bodyless child retains its typed declaration.
    Invalid layouts, output sets and entry/name
    collisions raise ValueError. The partitions add no primitive gates or work.
    """
    child = next(
        (function for function in record.functions if function.name == entry), None
    )
    if (
        child is None
        or verify_program_record(record)
        or record.physical_context is not None
        or child.registers is None
    ):
        raise ValueError("AutoPartition requires a valid typed component entry")
    assert child.registers is not None
    signature = Signature(child.registers)
    public = auto_partition_signature(signature, layout, packed_ports=packed_ports)
    if set(layout.output_names) != {r.name for r in signature.rights()}:
        raise ValueError("child output names must match the layout")
    external = tuple(public.lefts()) if layout.left_only else tuple(public)
    if any(reg.shape_symbolic for reg in external):
        raise ValueError("executable AutoPartition requires scalar packed ports")
    if not layout.left_only and any(reg.side != RegisterSide.THRU for reg in external):
        raise ValueError("repacking requires through packed ports")
    groups = layout._layouts()
    names = {f.name for f in record.functions}
    if name in names:
        raise ValueError("packed entry conflicts with an existing definition")
    index = 0
    while any(n.startswith(f"__fq_partition_{index}_") for n in names):
        index += 1
    prefix = f"__fq_partition_{index}_"
    registers = {reg.name: reg for reg in signature.lefts()}
    definitions = list(record.functions)
    partitions = []
    for i, (_, group) in enumerate(groups):
        parts = tuple(
            (
                registers[port]
                if port in registers
                else Register(port, RegisterType("opaque", width))
            )
            for port, width in group
        )
        partitions.append(parts)
        for reverse in ((False,) if layout.left_only else (False, True)):
            partition_name = prefix + f"{'join' if reverse else 'split'}{i}"
            if all(type(register.bitsize) is int for register in parts):
                partition = partition_component_program(parts, partition_name, reverse)
            else:
                partition = ProgramRecord(
                    (
                        FunctionRecord(
                            partition_name,
                            (),
                            registers=tuple(
                                partition_signature(parts, reverse=reverse)
                            ),
                        ),
                    )
                )
            definitions.extend(partition.functions)
    builder = ComponentProgramBuilder(public, definitions, name=name)
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


def auto_partition_component_pair(
    child: ProgramRecord,
    ports: tuple[Register, ...],
    layout: AutoPartition,
    inverse: ProgramRecord | None,
) -> tuple[ProgramRecord, ProgramRecord | None]:
    """Return an isolated source wrapper and its explicitly owned exact inverse."""
    native = _namespace_helpers(child, "partition_child")
    record = auto_partition_component_program(
        native, layout, entry=native.functions[0].name, name="main", packed_ports=ports
    )
    record = replace(
        record,
        functions=tuple(sorted(record.functions, key=lambda f: f.name != "main")),
    )
    if inverse is None:
        return record, None
    registers = {reg.name: reg for reg in child.functions[0].registers or ()}
    pairs = [(child, inverse)]
    for _, group in layout._layouts():
        parts = tuple(
            (
                registers[name]
                if name in registers
                else Register(name, opaque_register_type(width))
            )
            for name, width in group
        )
        partitions = []
        for reverse in (False, True):
            if all(type(register.bitsize) is int for register in parts):
                partition = partition_component_program(parts, "partition", reverse)
            else:
                partition = ProgramRecord(
                    (
                        FunctionRecord(
                            "partition",
                            (),
                            registers=tuple(
                                partition_signature(parts, reverse=reverse)
                            ),
                        ),
                    )
                )
            partitions.append(partition)
        split, join = partitions
        pairs.extend(((split, join), (join, split)))
    return record, adjoint_component_calls(record, tuple(pairs))
