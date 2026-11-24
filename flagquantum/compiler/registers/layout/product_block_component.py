# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Calls follow locked Qualtran v0.7.0 block_encoding/product.py.

"""Compose source-ordered products with shared words and explicit success flags."""

from typing import cast

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt, smax
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from ..component_graph import _collections, _elements
from ..control.multi_control_component import multi_control_x_program
from .auto_partition import AutoPartition, PartitionGroups
from .component_auto_partition import auto_partition_component_pair
from .component_partition import partition_component_program
from .packed_block_component import packed_block_signature


def product_block_signature(
    system: SymbolicInt,
    ancillas: tuple[SymbolicInt, ...],
    resources: tuple[SymbolicInt, ...],
) -> Signature:
    """Retain shared maxima and leading n-1 flag bits in the ancilla word."""
    if not ancillas or len(ancillas) != len(resources):
        raise ValueError("Product needs matched nonempty child width declarations")
    return packed_block_signature(
        system,
        cast(SymbolicInt, smax(ancillas)) + len(ancillas) - 1,
        cast(SymbolicInt, smax(resources)),
    )


def check_product_resource_partition(
    resources: tuple[SymbolicInt, ...], shared: SymbolicInt
) -> None:
    """Keep the original unequal nonzero resource partition failure explicit."""
    if any(width != 0 and width != shared for width in resources):
        raise ValueError(
            "Locked Product cannot partition unequal nonzero resource widths"
        )


def product_block_component_program(
    children: tuple[ProgramRecord, ...],
    *,
    inverses: tuple[ProgramRecord, ...] = (),
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain reversed constituents, shared bystanders and zero-check flag calls.

    Callers own all system, flag, shared ancilla and resource inputs. Children
    own their private work and exact inverses. Symbolic source products remain
    typed declarations. Identity signal admission belongs to the oracle owner.
    """
    if not children or type(adjoint) is not bool:
        raise ValueError("Product requires children and a boolean direction")
    if adjoint and len(inverses) != len(children):
        raise ValueError("Product requires one exact inverse per child")
    ports = []
    for child in children:
        registers = child.functions[0].registers
        if registers is None or not Signature(registers).thru_registers_only:
            raise ValueError("Product requires typed THRU child ports")
        if (
            not registers
            or registers[0].name != "system"
            or any(
                reg.name not in ("system", "ancilla", "resource") for reg in registers
            )
        ):
            raise ValueError("Product requires original packed block ports")
        ports.append({reg.name: reg for reg in registers})
    system = ports[0]["system"].bitsize
    if any(regs["system"].bitsize != system for regs in ports):
        raise ValueError("Product blocks must share the same system width")
    ancillas = tuple(
        regs["ancilla"].bitsize if "ancilla" in regs else 0 for regs in ports
    )
    resources = tuple(
        regs["resource"].bitsize if "resource" in regs else 0 for regs in ports
    )
    signature = product_block_signature(system, ancillas, resources)
    if any(type(reg.bitsize) is not int for reg in signature):
        return ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
    shared = int(max(ancillas))
    resource = int(max(resources))
    check_product_resource_partition(resources, resource)
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    flag_count = len(children) - 1
    bit = quantum_bit_type()
    flag_regs = (
        *((Register("flag_bits", bit, (flag_count,)),) if flag_count else ()),
        *((Register("ancilla", opaque_register_type(shared)),) if shared else ()),
    )
    pairs: list[tuple[ProgramRecord, ProgramRecord]] = []
    flags: BindingCollection | None = None
    if flag_regs:
        split = partition_component_program(flag_regs, "partition", False)
        join = partition_component_program(flag_regs, "partition", True)
        pairs.extend(((split, join), (join, split)))
        values = builder.call_program(
            split, {"x": state.pop("ancilla")}, entry=split.functions[0].name
        )
        flags = values.pop("flag_bits", None)
        state.update(values)
    for i, index in enumerate(reversed(range(len(children)))):
        groups: list[tuple[str, tuple[tuple[str | None, SymbolicInt], ...]]] = [
            ("system", (("system", system),))
        ]
        public = [Register("system", opaque_register_type(system))]
        width = int(ancillas[index])
        if width:
            groups.append(
                (
                    "ancilla",
                    (
                        ("ancilla", width),
                        *(((None, shared - width),) if shared > width else ()),
                    ),
                )
            )
            public.append(Register("ancilla", opaque_register_type(shared)))
        if resources[index]:
            groups.append(("resource", (("resource", resources[index]),)))
            public.append(Register("resource", opaque_register_type(resources[index])))
        forward, inverse = auto_partition_component_pair(
            children[index],
            tuple(public),
            AutoPartition(cast(PartitionGroups, tuple(groups))),
            inverses[index] if adjoint else None,
        )
        if inverse is not None:
            pairs.append((forward, inverse))
        inputs = {reg.name: state.pop(reg.name) for reg in public}
        state.update(builder.call_program(forward, inputs))
        if width and flag_count and i != flag_count:
            assert flags is not None and not isinstance(flags, RegisterBinding)
            word = state.pop("ancilla")
            assert isinstance(word, RegisterBinding)
            controls = builder.split(word)
            mcx = multi_control_x_program((0,) * width)
            pairs.append((mcx, mcx))
            x = ProgramRecord(
                (
                    bit_gate_function(
                        "x",
                        (Register("q", bit),),
                        CircuitIR(1, (Instruction("x", (0,)),)),
                    ),
                )
            )
            pairs.append((x, x))
            result = builder.call_program(
                mcx,
                _collections(
                    Signature(mcx.functions[0].registers or ()).lefts(),
                    (*controls[:width], flags[i]),
                ),
            )
            flags[i] = builder.call_program(
                x, {"q": result["target"]}, entry=x.functions[0].name
            )["q"]
            state["ancilla"] = builder.join(
                (*_elements((result["controls"],)), *controls[width:]),
                word.register.dtype,
            )
    if flag_regs:
        join = partition_component_program(flag_regs, "partition", True)
        values = {"ancilla": state.pop("ancilla")} if shared else {}
        if flags is not None:
            values["flag_bits"] = flags
        state["ancilla"] = builder.call_program(
            join, values, entry=join.functions[0].name
        )["x"]
    record = builder.finalize(state)
    return adjoint_component_calls(record, tuple(pairs)) if adjoint else record
