# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 qubitization/qubitization_walk_operator.py.

"""Compose source walk calls while children own signal state and private work."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder


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
