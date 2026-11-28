# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 qubitization/qubitization_walk_operator.py.

"""Compose source walk calls while children own signal state and private work."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import quantum_control_signature
from ..component_graph import _elements
from ..control.specialized_component_control import multi_controlled_component_pair


def qubitization_walk_signature(
    block_selection: tuple[Register, ...],
    reflection_selection: tuple[Register, ...],
    targets: tuple[Register, ...],
    junk: tuple[Register, ...],
) -> Signature:
    """Keep the existing deterministic native union of complete descriptors.

    Original source constructors use a set. Their adapter must pass the actual
    captured Signature to the composition owner instead of inferring positional
    operands from this deterministic native metadata interface.
    """
    selection = tuple(dict.fromkeys((*block_selection, *reflection_selection)))
    if len({reg.name for reg in selection}) != len(selection):
        raise ValueError("Walk selection union contains unequal same-name registers")
    return Signature((*selection, *targets, *junk))


def doubly_controlled_select_walk_pair(
    signature: Signature,
    block_signature: Signature,
    positive_select_pair: tuple[ProgramRecord, ProgramRecord],
    positive_reflection: ProgramRecord,
    *,
    select_control: str,
    reflection_control: str,
) -> tuple[ProgramRecord, ProgramRecord]:
    """Keep separate packed predicates for source double-positive walk children.

    Source SELECT-block and reflection control adders own their exact port names.
    The source declares its reflection self-adjoint; both directional recipes
    therefore call that same reflection. This does not certify mathematical
    inversion or clean private release for a defective supplied reflection.
    Caller controls are ordered outer then inner, followed by unchanged data.
    Child Programs use the existing multi-control owner's main entry convention.
    """

    def twice(ports: Signature) -> Signature:
        return quantum_control_signature(
            quantum_control_signature(ports, ((quantum_bit_type(), ()),)),
            ((quantum_bit_type(), ()),),
        )

    if not signature.thru_registers_only or not block_signature.thru_registers_only:
        raise ValueError("Double-controlled walk requires THRU data signatures")
    select_ports = positive_select_pair[0].functions[0].registers
    reflection_ports = positive_reflection.functions[0].registers
    if select_ports is None or reflection_ports is None:
        raise ValueError("Double-controlled walk requires typed positive children")
    select_data = tuple(r for r in select_ports if r.name != select_control)
    reflection_data = tuple(r for r in reflection_ports if r.name != reflection_control)
    root_ports = {r.name: r for r in signature}
    if select_data != tuple(block_signature) or any(
        root_ports.get(r.name) != r for r in (*select_data, *reflection_data)
    ):
        raise ValueError("Double-controlled walk child data disagrees with root ports")
    selections = multi_controlled_component_pair(
        *positive_select_pair, select_control, (1, 1)
    )
    reflection = multi_controlled_component_pair(
        positive_reflection, positive_reflection, reflection_control, (1, 1)
    )[0]

    def packed_call(
        builder: ComponentProgramBuilder,
        state: dict[str, BindingCollection],
        child: ProgramRecord,
        packed_name: str,
        controls: tuple[str, str],
    ) -> None:
        ports = child.functions[0].registers
        assert ports is not None
        inputs = {r.name: state.pop(r.name) for r in ports if r.name != packed_name}
        inputs[packed_name] = np.array([state.pop(n) for n in controls], dtype=object)
        outputs = builder.call_program(child, inputs)
        state.update(zip(controls, _elements((outputs.pop(packed_name),)), strict=True))
        state.update(outputs)

    block_ports = twice(block_signature)
    block_controls = tuple(block_ports)[:2]
    blocks = []
    for selected in selections:
        builder = ComponentProgramBuilder(block_ports)
        state = dict(builder.inputs)
        packed_call(
            builder,
            state,
            selected,
            select_control,
            (block_controls[0].name, block_controls[1].name),
        )
        blocks.append(builder.finalize(state))

    ports = twice(signature)
    controls = tuple(ports)[:2]
    result = []
    for inverse in (False, True):
        builder = ComponentProgramBuilder(ports)
        state = dict(builder.inputs)
        for role in (("reflection", "block") if inverse else ("block", "reflection")):
            if role == "reflection":
                packed_call(
                    builder,
                    state,
                    reflection,
                    reflection_control,
                    (controls[0].name, controls[1].name),
                )
            else:
                inputs = {r.name: state.pop(r.name) for r in block_signature}
                inputs.update(
                    (b.name, state.pop(r.name))
                    for b, r in zip(block_controls, controls, strict=True)
                )
                outputs = builder.call_program(blocks[int(inverse)], inputs)
                state.update(
                    (r.name, outputs.pop(b.name))
                    for b, r in zip(block_controls, controls, strict=True)
                )
                state.update(outputs)
        result.append(builder.finalize(state))
    return result[0], result[1]


def qubitization_walk_component_pair(
    signature: Signature,
    block_pair: tuple[ProgramRecord, ProgramRecord],
    reflection_pair: tuple[ProgramRecord, ProgramRecord],
    *,
    controlled: bool = False,
) -> tuple[ProgramRecord, ProgramRecord]:
    """Thread named caller-owned words through block then prepared reflection.

    Reverse recipes are explicit dependencies. Mathematical execution requires
    exact inverse proofs from their owners; literal source capture can retain
    a source constructor defect without claiming that proof. Each dependency
    owns its clean work and feedback. In controlled mode both pairs implement
    their source-positive specializations; the first child control port is
    mapped to the outer first port, independently of their internal spellings.
    No source metadata, gate equivalence or control decomposition is inferred.
    """
    if type(controlled) is not bool or not signature.thru_registers_only:
        raise ValueError("Walk requires a boolean control flag and THRU storage")
    ports = tuple(signature)
    if controlled and (
        not ports
        or ports[0].shape
        or ports[0].total_qubits() != 1
        or ports[0].dtype.classical_bits != 0
    ):
        raise ValueError("Controlled walk requires a leading quantum control bit")
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    for forward, inverse in (block_pair, reflection_pair):
        child_ports = forward.functions[0].registers
        if (
            child_ports is None
            or not Signature(child_ports).thru_registers_only
            or child_ports != inverse.functions[0].registers
        ):
            raise ValueError("Walk dependencies require matching exact THRU pairs")
        if controlled and (
            not child_ports
            or child_ports[0].shape
            or child_ports[0].total_qubits() != 1
            or child_ports[0].dtype.classical_bits != 0
        ):
            raise ValueError("Walk dependency requires its positive control owner")
        data_ports = child_ports[1:] if controlled else child_ports
        if any(reg.name not in state for reg in data_ports):
            raise ValueError("Walk dependency refers to an absent caller-owned word")
        values = {reg.name: state.pop(reg.name) for reg in data_ports}
        if controlled:
            values[child_ports[0].name] = state.pop(ports[0].name)
        result = builder.call_program(forward, values, entry=forward.functions[0].name)
        if controlled:
            state[ports[0].name] = result.pop(child_ports[0].name)
        state.update(result)
    forward = builder.finalize(state)
    return forward, adjoint_component_calls(forward, (block_pair, reflection_pair))
