# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BlackBoxSelect.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/black_box_select.py

"""Pack SELECT register groups using the existing AutoPartition layout."""

from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from .auto_partition import AutoPartition
from .partition import partition_width


def packed_select_layout(
    selection: tuple[Register, ...],
    target: tuple[Register, ...],
    *,
    controls: tuple[Register, ...] = (),
) -> tuple[Signature, AutoPartition]:
    """Describe the two packed THRU ports and their ordered child bindings.

    The locked packed signature has no control port. A controlled child must
    be bound through its named signature. Array ports use their flattened
    width in row-major order. Symbolic widths are retained in the declaration
    and are validated by AutoPartition when concrete binding is requested.
    """
    if controls:
        raise ValueError("Packed SELECT requires an uncontrolled child")
    for register in (*selection, *target):
        if register.side is not RegisterSide.THRU or register.total_cbits() != 0:
            raise ValueError("Packed SELECT requires quantum THRU registers")

    selection_layout = tuple((reg.name, reg.total_qubits()) for reg in selection)
    target_layout = tuple((reg.name, reg.total_qubits()) for reg in target)
    signature = Signature(
        (
            Register(
                "selection", opaque_register_type(partition_width(selection_layout))
            ),
            Register("system", opaque_register_type(partition_width(target_layout))),
        )
    )
    partition = AutoPartition(
        (("selection", selection_layout), ("system", target_layout))
    )
    return signature, partition
