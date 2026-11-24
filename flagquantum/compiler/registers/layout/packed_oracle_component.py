# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 black_box_prepare.py.

"""Own one exact AutoPartition boundary around typed oracle components."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import ssum
from ..component_builder import ComponentProgramBuilder
from .auto_partition import AutoPartition
from .component_auto_partition import auto_partition_component_pair


def packed_oracle_component_program(
    signature: Signature,
    groups: tuple[tuple[str, tuple[Register, ...]], ...],
    prepare: ProgramRecord,
    *,
    inverse: ProgramRecord | None = None,
    adjoint: bool = False,
) -> ProgramRecord:
    """Pack MSB-first oracle words around one AutoPartition call.

    The caller supplies the oracle's exact typed body and, for an adjoint,
    its exact inverse. The inverse reverses the original partition call order.
    Logical junk remains caller-owned, and private work belongs to the child.
    """
    if type(adjoint) is not bool:
        raise ValueError("Packed oracle direction must be boolean")
    layout = AutoPartition(
        tuple(
            (name, tuple((r.name, r.total_qubits()) for r in regs))
            for name, regs in groups
        )
    )
    auto, reverse = auto_partition_component_pair(
        prepare,
        tuple(
            Register(
                name,
                opaque_register_type(ssum(reg.total_qubits() for reg in registers)),
            )
            for name, registers in groups
        ),
        layout,
        inverse if adjoint else None,
    )
    if adjoint:
        if reverse is None:
            raise ValueError("Packed oracle requires an explicit exact inverse")
        auto = reverse
    builder = ComponentProgramBuilder(signature)
    state = dict(builder.inputs)
    inputs = {name: state.pop(name) for name, _ in groups}
    state.update(builder.call_program(auto, inputs))
    return builder.finalize(state)
