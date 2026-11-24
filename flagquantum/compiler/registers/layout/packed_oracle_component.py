# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 black_box_prepare.py.

"""Own one exact AutoPartition boundary around typed oracle components."""

from dataclasses import replace

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..component_graph import _namespace_helpers
from .auto_partition import AutoPartition
from .component_auto_partition import (
    auto_partition_component_program,
)
from .component_partition import (
    partition_component_program,
    partition_signature,
)


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
    child = _namespace_helpers(prepare, "prepare")
    layout = AutoPartition(
        tuple(
            (name, tuple((r.name, r.total_qubits()) for r in regs))
            for name, regs in groups
        )
    )
    auto = auto_partition_component_program(
        child,
        layout,
        entry=child.functions[0].name,
        name="main",
        packed_ports=tuple(signature),
    )
    auto = replace(
        auto, functions=tuple(sorted(auto.functions, key=lambda f: f.name != "main"))
    )
    if adjoint:
        if inverse is None:
            raise ValueError("Packed oracle requires an explicit exact inverse")
        pairs = [(prepare, inverse)]
        for _, registers in groups:
            partitions = []
            for reverse in (False, True):
                if all(type(reg.bitsize) is int for reg in registers):
                    partition = partition_component_program(
                        registers, "partition", reverse
                    )
                else:
                    partition = ProgramRecord(
                        (
                            FunctionRecord(
                                "partition",
                                (),
                                registers=tuple(
                                    partition_signature(registers, reverse=reverse)
                                ),
                            ),
                        )
                    )
                partitions.append(partition)
            split, join = partitions
            pairs.extend(((split, join), (join, split)))
        auto = adjoint_component_calls(auto, tuple(pairs))
    builder = ComponentProgramBuilder(signature)
    return builder.finalize(builder.call_program(auto, builder.inputs))
