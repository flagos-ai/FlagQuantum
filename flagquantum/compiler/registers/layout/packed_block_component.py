# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Calls follow locked Qualtran v0.7.0 block_encoding/unitary.py and tensor_product.py.

"""Compose packed unitary and disjoint tensor-product block calls."""

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt, ssum
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from .component_partition import partition_component_program
from .packed_oracle_component import packed_oracle_component_program


def packed_block_signature(
    system: SymbolicInt, ancilla: SymbolicInt, resource: SymbolicInt
) -> Signature:
    """Keep source opaque storage and omit concrete zero-width ports."""
    return Signature.build_from_dtypes(
        system=opaque_register_type(system),
        ancilla=opaque_register_type(ancilla),
        resource=opaque_register_type(resource),
    )


def unitary_block_signature(
    child: Signature, *, ancilla: SymbolicInt = 0, resource: SymbolicInt = 0
) -> Signature:
    """Retain the source element-width sum, including its shaped-width limit."""
    if not child.thru_registers_only:
        raise ValueError("Block encoded unitary must have all THRU registers")
    return packed_block_signature(ssum(reg.bitsize for reg in child), ancilla, resource)


def tensor_product_block_signature(
    widths: tuple[tuple[SymbolicInt, SymbolicInt, SymbolicInt], ...],
) -> Signature:
    """Pack nonempty child widths in system, ancilla, resource order."""
    if not widths:
        raise ValueError("Tensor product requires at least one block")
    system, ancilla, resource = (ssum(port) for port in zip(*widths, strict=True))
    return packed_block_signature(system, ancilla, resource)


def unitary_block_component_program(
    child: ProgramRecord,
    *,
    ancilla: SymbolicInt = 0,
    resource: SymbolicInt = 0,
    inverse: ProgramRecord | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Wrap one exact AutoPartition call with untouched auxiliary bystanders."""
    registers = child.functions[0].registers
    if registers is None:
        raise ValueError("Unitary block requires typed child ports")
    signature = unitary_block_signature(
        Signature(registers), ancilla=ancilla, resource=resource
    )
    return packed_oracle_component_program(
        signature,
        (("system", registers),),
        child,
        inverse=inverse,
        adjoint=adjoint,
    )


def tensor_product_block_component_program(
    children: tuple[ProgramRecord, ...],
    *,
    inverses: tuple[ProgramRecord, ...] = (),
    adjoint: bool = False,
) -> ProgramRecord:
    """Keep source group partitions, sequential child calls and exact inverse.

    Every packed port is caller-owned. Private workspace belongs to children.
    Symbolic source tensor products remain declarations with no physical body.
    """
    if not children or type(adjoint) is not bool:
        raise ValueError("Tensor product requires children and a boolean direction")
    if adjoint and len(inverses) != len(children):
        raise ValueError("Tensor product requires one exact inverse per child")
    ports = []
    for child in children:
        registers = child.functions[0].registers
        if registers is None or not Signature(registers).thru_registers_only:
            raise ValueError("Tensor blocks require typed THRU child ports")
        if (
            not registers
            or registers[0].name != "system"
            or any(
                reg.name not in ("system", "ancilla", "resource") for reg in registers
            )
        ):
            raise ValueError("Tensor blocks require source packed block ports")
        ports.append({reg.name: reg for reg in registers})
    widths = tuple(
        (
            regs["system"].bitsize,
            regs["ancilla"].bitsize if "ancilla" in regs else 0,
            regs["resource"].bitsize if "resource" in regs else 0,
        )
        for regs in ports
    )
    signature = tensor_product_block_signature(widths)
    if any(type(reg.bitsize) is not int for reg in signature):
        return ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    parts: dict[str, tuple[Register, ...]] = {}
    words: dict[str, BindingCollection] = {}
    pairs: list[tuple[ProgramRecord, ProgramRecord]] = []
    for name in ("system", "ancilla", "resource"):
        regs = tuple(
            Register(
                f"{name}{i}_",
                ports[i][name].dtype,
                ports[i][name].shape_symbolic,
                ports[i][name].side,
            )
            for i in range(len(children))
            if name in ports[i]
        )
        if not regs:
            continue
        split = partition_component_program(regs, "partition", False)
        join = partition_component_program(regs, "partition", True)
        parts[name] = regs
        pairs.extend(((split, join), (join, split)))
        words.update(
            builder.call_program(
                split, {"x": state.pop(name)}, entry=split.functions[0].name
            )
        )
    for i, child in enumerate(children):
        result = builder.call_program(
            child,
            {name: words.pop(f"{name}{i}_") for name in ports[i]},
            entry=child.functions[0].name,
        )
        words.update((f"{name}{i}_", value) for name, value in result.items())
    for name, regs in parts.items():
        join = partition_component_program(regs, "partition", True)
        state[name] = builder.call_program(
            join,
            {reg.name: words.pop(reg.name) for reg in regs},
            entry=join.functions[0].name,
        )["x"]
    record = builder.finalize(state)
    if adjoint:
        record = adjoint_component_calls(
            record, (*pairs, *zip(children, inverses, strict=True))
        )
    return record
