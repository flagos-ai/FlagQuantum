# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Root order follows locked Qualtran v0.7.0 HamiltonianSimulationByGQSP.

"""Compose owned PREPARE, GQSP and exact PREPARE erasure calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_graph import _collections, _elements


def hamiltonian_gqsp_component_program(
    gqsp: ProgramRecord,
    prepare: ProgramRecord,
    unprepare: ProgramRecord,
    *,
    junk_names: tuple[str, ...],
) -> ProgramRecord:
    """Keep signal caller-owned and release only explicitly owned PREPARE junk.

    The supplied inverse must restore private junk on the accepted input domain.
    Composition records that lifetime; it does not prove numerical cleanup.
    """
    modules = (gqsp, prepare, unprepare)
    if any(
        record.physical_context is not None
        or not record.functions
        or any(not function.blocks for function in record.functions)
        for record in modules
    ):
        raise ValueError("Hamiltonian GQSP requires complete logical child Programs")
    ports = gqsp.functions[0].registers
    preparation = prepare.functions[0].registers
    if (
        ports is None
        or preparation is None
        or not Signature(ports).thru_registers_only
        or not Signature(preparation).thru_registers_only
        or unprepare.functions[0].registers != preparation
        or len(set(junk_names)) != len(junk_names)
    ):
        raise ValueError("Hamiltonian GQSP requires matching exact THRU preparation")
    outer = {register.name: register for register in ports}
    children = {register.name: register for register in preparation}
    if (
        not set(junk_names) <= children.keys()
        or set(junk_names) & outer.keys()
        or any(
            register.name not in junk_names and outer.get(register.name) != register
            for register in preparation
        )
        or any(children[name].dtype.classical_bits for name in junk_names)
    ):
        raise ValueError("PREPARE selection and private junk must have distinct owners")
    builder = ComponentProgramBuilder(Signature(ports))
    state = dict(builder.inputs)
    junk = tuple(register for register in preparation if register.name in junk_names)
    state.update(
        _collections(
            junk,
            (builder.allocate(dtype=r.dtype) for r in junk for _ in r.all_idxs()),
        )
    )
    for record, registers in (
        (prepare, preparation),
        (gqsp, ports),
        (unprepare, preparation),
    ):
        state.update(
            builder.call_program(
                record,
                {register.name: state.pop(register.name) for register in registers},
                entry=record.functions[0].name,
            )
        )
    for binding in _elements(state.pop(register.name) for register in junk):
        builder.free(binding)
    return builder.finalize(state)
