# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 select_hubbard.py.

"""Store Hubbard spin-up partitioning and inlined coordinate selection."""

import numpy as np

from ...compiler.registers.component_builder import ComponentProgramBuilder
from ...compiler.registers.component_expansion import expand_component_calls_once
from ...compiler.registers.control.unary_wire_calls import adjoint_unary_wire_program
from ...compiler.registers.layout.component_partition import partition_component_program
from ...compiler.registers.lookup.selected_pauli_component import selected_pauli_program
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .hubbard_spin_up import HubbardSpinUpZ


def hubbard_spin_up_program(
    operation: HubbardSpinUpZ, *, adjoint: bool = False
) -> ProgramRecord:
    """Retain y/x selection of the second target half with clean private work.

    Stored negative-control decomposition retains its source failure boundary.
    The separately accepted standalone negative-control operation remains owned
    by its existing physical emitter.
    """
    if type(adjoint) is not bool:
        raise ValueError("Stored Hubbard direction must be boolean")
    if operation.dimension == 1:
        raise ValueError(
            "Original stored spin-up decomposition refuses singleton coordinates"
        )
    if operation.controlled and operation.control_value == 0:
        raise ValueError(
            "Original stored spin-up zero control executes a positive-control body"
        )
    width = (operation.dimension - 1).bit_length()
    coordinate = bounded_unsigned_register_type(width, operation.dimension)
    signature = Signature(
        (
            *(
                (Register("control", quantum_bit_type()),)
                if operation.controlled
                else ()
            ),
            Register("V", bounded_unsigned_register_type(1, 2)),
            Register("x", coordinate),
            Register("y", coordinate),
            Register("target", opaque_register_type(2 * operation.dimension**2)),
        )
    )
    builder = ComponentProgramBuilder(signature)
    values = dict(builder.inputs)
    control = values["V"]
    if operation.controlled:
        control = builder.join(
            np.asarray([control, values["control"]], dtype=object),
            opaque_register_type(2),
        )
    half = operation.dimension**2
    parts = tuple(Register(name, opaque_register_type(half)) for name in ("y1", "y2"))
    split = partition_component_program(parts, "split_target", False)
    join = partition_component_program(parts, "join_target", True)
    target = builder.call_program(split, {"x": values["target"]}, entry="split_target")
    selected = selected_pauli_program(
        (Register("y", coordinate), Register("x", coordinate)),
        control_width=1 + int(operation.controlled),
    )
    result = builder.call_program(
        selected,
        {
            "control": control,
            "y": values["y"],
            "x": values["x"],
            "target": target["y2"],
        },
    )
    target["y2"] = result["target"]
    values["target"] = builder.call_program(join, target, entry="join_target")["x"]
    values.update(x=result["x"], y=result["y"])
    control = result["control"]
    if operation.controlled:
        assert isinstance(control, RegisterBinding)
        values["V"], values["control"] = builder.split(control)
    else:
        values["V"] = control
    root = builder.finalize(values)
    # Source add_from re-inserts the child in connection visitation order.
    root = expand_component_calls_once(
        root, predicate=lambda _, op: op.callee not in {"split_target", "join_target"}
    )[0]
    return (
        adjoint_unary_wire_program(root, ((split, join), (join, split)))
        if adjoint
        else root
    )
