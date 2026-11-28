# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call order follows locked Qualtran v0.7.0 GeneralizedQSP.decompose_from_registers.

"""Compose the finite QSP call chain from explicit native directional children."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_control import quantum_control_signature
from ..component_graph import _collections, _elements


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
