# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call order follows locked Qualtran v0.7.0 GeneralizedQSP.decompose_from_registers.

"""Compose the finite QSP call chain from explicit native directional children."""

from collections.abc import Mapping
from dataclasses import replace

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import (
    ControlAdder,
    control_component_program,
    quantum_control_signature,
)
from ..component_graph import _collections, _elements


def controlled_gqsp_component_program(
    record: ProgramRecord, controlled_calls: Mapping[str, ProgramRecord]
) -> ProgramRecord:
    """Control a GQSP root using explicit source-selected child owners.

    Each supplied child has a leading positive control followed by the original
    child's data in the same typed order. Port names may differ; packing views
    belong to those child owners. No controlled semantics are inferred from a
    signature. A bodyless source GQSP remains an unexecutable declaration.
    """
    if not record.functions or record.physical_context is not None:
        raise ValueError("Controlled GQSP requires a logical source Program")
    root = record.functions[0]
    if (
        root.registers is None
        or not root.registers
        or root.registers[0] != Register("signal", quantum_bit_type())
        or not Signature(root.registers).thru_registers_only
    ):
        raise ValueError("Controlled GQSP requires its through signal and data ports")
    if not root.blocks:
        return replace(
            record,
            canonical_payload=None,
            functions=(
                replace(
                    root,
                    registers=tuple(
                        quantum_control_signature(
                            Signature(root.registers), ((quantum_bit_type(), ()),)
                        )
                    ),
                ),
                *record.functions[1:],
            ),
        )
    functions = {function.name: function for function in record.functions}
    adders: dict[str, ControlAdder] = {}
    for name, module in controlled_calls.items():
        original = functions.get(name)
        if original is None or original.registers is None or not module.functions:
            raise ValueError("Controlled GQSP requires explicit typed child owners")
        old_ports = original.registers
        ports = module.functions[0].registers
        if (
            module.physical_context is not None
            or ports is None
            or len(ports) != len(old_ports) + 1
            or ports[0] != Register(ports[0].name, quantum_bit_type())
            or not Signature(ports).thru_registers_only
            or tuple((r.dtype, r.shape, r.side) for r in ports[1:])
            != tuple((r.dtype, r.shape, r.side) for r in old_ports)
        ):
            raise ValueError("Controlled GQSP child must preserve typed data order")

        def add(
            builder: ComponentProgramBuilder,
            control: BindingCollection,
            values: dict[str, BindingCollection],
            *,
            child: ProgramRecord = module,
            original_ports: tuple[Register, ...] = old_ports,
            controlled_ports: tuple[Register, ...] = ports,
        ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
            inputs = {
                new.name: values[old.name]
                for new, old in zip(controlled_ports[1:], original_ports, strict=True)
            }
            inputs[controlled_ports[0].name] = control
            outputs = builder.call_program(child, inputs, entry=child.functions[0].name)
            return outputs[controlled_ports[0].name], tuple(
                outputs[r.name] for r in controlled_ports[1:]
            )

        adders[name] = add
    return control_component_program(record, adders, entry=root.name)


def gqsp_control_view(
    record: ProgramRecord, targets: tuple[Register, ...]
) -> ProgramRecord:
    """Regroup the source .on(control, *targets) elements without physical gates.

    CZPow packs both bits in q; Toffoli packs its two controls together. Their
    existing bodies retain ownership and operation order behind this view.
    """
    if record.physical_context is not None or not record.functions:
        raise ValueError("GQSP controlled view requires a logical native Program")
    child = record.functions[0]
    ports = child.registers
    signature = quantum_control_signature(
        Signature(targets), ((quantum_bit_type(), ()),)
    )
    if ports == tuple(signature):
        return record
    if (
        ports is None
        or not Signature(ports).thru_registers_only
        or tuple(r.bitsize for r in ports for _ in r.all_idxs())
        != tuple(r.bitsize for r in signature for _ in r.all_idxs())
    ):
        raise ValueError(
            "GQSP controlled view requires source-ordered matching elements"
        )
    builder = ComponentProgramBuilder(signature, name=child.name + "_view")
    outputs = builder.call_program(
        record,
        _collections(ports, _elements(builder.inputs.values())),
        entry=child.name,
    )
    return builder.finalize(_collections(tuple(signature), _elements(outputs.values())))


def generalized_qsp_component_program(
    unitary: ProgramRecord,
    inverse: ProgramRecord,
    positive_inverse: ProgramRecord,
    negative_forward: ProgramRecord,
    rotations: tuple[ProgramRecord, ...],
    *,
    negative_power: int = 0,
) -> ProgramRecord:
    """Retain source calls, explicit inverse roles and scalar-carrier ownership.

    A source leaf may be a typed declaration. Logical composition preserves
    that call; physical binding still requires its body. Supplied inverse and
    control roles are the owner's responsibility, not inferred from signatures.
    """
    modules = (unitary, inverse, positive_inverse, negative_forward, *rotations)
    if any(
        module.physical_context is not None or not module.functions
        for module in modules
    ):
        raise ValueError("GQSP requires nonempty logical native child Programs")
    ports = unitary.functions[0].registers
    if (
        ports is None
        or not Signature(ports).thru_registers_only
        or any(port.name == "signal" for port in ports)
        or inverse.functions[0].registers != ports
    ):
        raise ValueError("GQSP requires matching through U pairs and a reserved signal")
    if type(negative_power) is not int or negative_power < 0 or len(rotations) < 2:
        raise ValueError("GQSP requires nonnegative inverse power and two rotations")
    controlled_ports = []
    for child in (positive_inverse, negative_forward):
        registers = child.functions[0].registers
        if (
            registers is None
            or not Signature(registers).thru_registers_only
            or len(registers) != len(ports) + 1
            or registers[0].dtype != quantum_bit_type()
            or registers[0].shape_symbolic
            or registers[1:] != ports
        ):
            raise ValueError(
                "GQSP controlled child requires scalar control and original U ports"
            )
        controlled_ports.append(registers)
    rotation_ports = tuple(Signature.build(q=1))
    if any(rotation.functions[0].registers != rotation_ports for rotation in rotations):
        raise ValueError("GQSP signal rotations require one through q bit")
    builder = ComponentProgramBuilder(
        Signature((Register("signal", quantum_bit_type()), *ports))
    )
    state = dict(builder.inputs)

    def rotate(rotation: ProgramRecord) -> None:
        state["signal"] = builder.call_program(
            rotation, {"q": state["signal"]}, entry=rotation.functions[0].name
        )["q"]

    rotate(rotations[0])
    remaining = negative_power
    for rotation in rotations[1:]:
        index = 0 if remaining > 0 else 1
        child = positive_inverse if remaining > 0 else negative_forward
        registers = controlled_ports[index]
        values = {register.name: state[register.name] for register in ports}
        outputs = builder.call_program(
            child,
            {registers[0].name: state["signal"], **values},
            entry=child.functions[0].name,
        )
        state["signal"] = outputs.pop(registers[0].name)
        state.update(outputs)
        remaining -= int(remaining > 0)
        rotate(rotation)
    for _ in range(remaining):
        state.update(
            builder.call_program(
                inverse,
                {r.name: state[r.name] for r in ports},
                entry=inverse.functions[0].name,
            )
        )
    return builder.finalize(state)
