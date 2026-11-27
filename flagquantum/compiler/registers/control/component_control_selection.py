# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _get_ctrl_system_1bit_cv.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/specialized_ctrl.py

"""Select stored single-control specializations before the shared fallback."""

from collections.abc import Callable

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from .component_extension import StructuredControlAdder
from .component_predicate import validate_control_ports
from .control_predicate import ControlRegister
from .specialized_component_control import specialized_component_control_system

ComponentControlSystem = Callable[
    [tuple[Register, ...], tuple[ControlRegister, ...] | None],
    tuple[ProgramRecord, StructuredControlAdder],
]
ControlSpecialization = Callable[[int], tuple[ProgramRecord, str] | None]


def select_component_control_system(
    controls: tuple[Register, ...],
    patterns: tuple[ControlRegister, ...] | None,
    specialization: ControlSpecialization,
    fallback: ComponentControlSystem,
    current_value: int | None = None,
    entry: str = "main",
) -> tuple[ProgramRecord, StructuredControlAdder]:
    """Select a scalar specialization or delegate to the existing control owner.

    A specialization is a positive/negative stored body and its control-port
    name. Positive specialization is required on the branches that query it.
    Existing control combines new then old using the positive specialization.
    Non-single-qubit requests and unavailable negative specializations call
    fallback once without changing the descriptors. That owner validates its
    classical, quantum or unresolved declaration. The direct adder mutates its input
    dictionary as the original callback does. Invalid descriptors or missing
    positive specialization raise ValueError. Adder/builder errors propagate.
    """
    if current_value is not None and (
        type(current_value) is not int or current_value not in (0, 1)
    ):
        raise ValueError("current control activation must be zero, one or None")
    if not controls or (patterns is not None and len(controls) != len(patterns)):
        raise ValueError("control ports require one activation pattern per register")
    if Signature(controls).n_qubits() != 1:
        return fallback(controls, patterns)
    validate_control_ports(controls, patterns)
    if patterns is None:
        raise ValueError("single control requires a concrete activation")
    value = patterns[0].bits()[0]
    selected = specialization(value if current_value is None else 1)
    if selected is None:
        if value == 1 or current_value is not None:
            raise ValueError("positive control specialization must be provided")
        return fallback(controls, patterns)
    module, name = selected
    if current_value is not None:
        module, scalar_adder = specialized_component_control_system(
            module, name, current_value, value, entry
        )

        def add_existing(
            builder: ComponentProgramBuilder,
            added: tuple[BindingCollection, ...],
            values: dict[str, BindingCollection],
        ) -> tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]]:
            (control,) = added
            result, data = scalar_adder(builder, control, values)
            return (result,), data

        return module, add_existing

    def add_direct(
        builder: ComponentProgramBuilder,
        added: tuple[BindingCollection, ...],
        values: dict[str, BindingCollection],
    ) -> tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]]:
        (control,) = added
        values[name] = control
        outputs = builder.call_dict(entry, values)
        return (outputs.pop(name),), tuple(outputs.values())

    return module, add_direct
