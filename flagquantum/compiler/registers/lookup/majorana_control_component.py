# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Generic propagation follows locked Qualtran v0.7.0 controlled.py.

"""Control owned Majorana calls while retaining unavailable directional leaves."""

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import numpy as np

from ....core.ir import CircuitIR, Instruction
from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..component_control import (
    ControlAdder,
    _stored_control_adder,
    control_component_program,
    quantum_control_signature,
)
from ..component_definition_match import matches_component_definition
from ..component_gate_function import bit_gate_function
from ..component_graph import _namespace_helpers
from ..control.component_bookkeeping import bookkeeping_scalar_control_adder
from ..control.conjunction_program import and_component_program
from ..control.specialized_component_control import specialized_component_control_system
from ..control.unary_wire_calls import bit_wire_call
from ..primitives.controlled_y_component import (
    controlled_cirq_y_program,
    controlled_scalar_y_program,
)
from .majorana_component import majorana_program


def controlled_majorana_program(
    selection: tuple[Register, ...],
    pauli: Literal["X", "Y", "Z"] = "Y",
    *,
    control: Register | None = None,
    adjoint: bool = False,
    cy_profile: Literal["cirq", "scalar"] = "cirq",
) -> ProgramRecord:
    """Preserve the original generic scalar-control call graph, not a repair.

    Caller ports retain MSB order and their original predicate. Split/Join and
    lifetimes bypass the new control. Controlled And has its actual directional
    signature but no source body, so physical binding refuses it. The caller
    selects the source CY view at its SDK boundary: Cirq's q-array view has a
    three-call body, whereas scalar CYGate uses its packed-control helper.
    """
    if cy_profile not in ("cirq", "scalar"):
        raise ValueError("Majorana control requires a known source CY profile")
    record = majorana_program(selection, pauli, control=control, adjoint=adjoint)
    functions = {function.name: function for function in record.functions}
    definitions = list(record.functions)
    adders: dict[str, ControlAdder] = {}

    def import_module(module: ProgramRecord) -> FunctionRecord:
        imported = _namespace_helpers(module, f"__fq_majorana_control_{len(adders)}")
        definitions.extend(imported.functions)
        return imported.functions[0]

    def declared_control(function: FunctionRecord) -> FunctionRecord:
        assert function.registers is not None
        signature = quantum_control_signature(
            Signature(function.registers), ((quantum_bit_type(), ()),)
        )
        return import_module(
            ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
        )

    for operation in record.functions[0].blocks[0].operations:
        name = operation.callee
        assert name is not None
        if name in adders:
            continue
        function = functions[name]
        for values in ((1, 0), (1, 1)):
            for erase in (False, True):
                template = and_component_program(values, name="and", uncompute=erase)
                if matches_component_definition(record, name, template):
                    adders[name] = _stored_control_adder(declared_control(function))
                    break
            if name in adders:
                break
        if name in adders:
            continue
        for gate, wires in (
            ("x", (0,)),
            ("cx", (0, 1)),
            ("cy", (0, 1)),
            ("cz", (0, 1)),
        ):
            template = bit_wire_call(Instruction(gate, wires))[0]
            if matches_component_definition(record, name, template):
                adders[name] = _primitive_control(gate, import_module, cy_profile)
                break
        if name not in adders:
            # This validator admits only actual storage/lifetime operations;
            # unknown computational children fail instead of ignoring ctrl.
            adders[name] = bookkeeping_scalar_control_adder(function)
    return control_component_program(
        replace(record, functions=tuple(definitions)), adders
    )


def _primitive_control(
    gate: str,
    import_module: Callable[[ProgramRecord], FunctionRecord],
    cy_profile: Literal["cirq", "scalar"],
) -> ControlAdder:
    if gate == "cz":
        positive = bit_wire_call(Instruction("cz", (0, 1)))[0]
        module, _ = specialized_component_control_system(
            positive, "q1", 1, 1, positive.functions[0].name
        )
    elif gate == "cy" and cy_profile == "cirq":
        module = controlled_cirq_y_program()
    elif gate == "cy":
        module = controlled_scalar_y_program()
    else:
        signature = (
            Signature.build(ctrl=1, target=1)
            if gate == "x"
            else Signature(
                (
                    Register("ctrl", quantum_bit_type(), (2,)),
                    Register("target", quantum_bit_type()),
                )
            )
        )
        count = 2 if gate == "x" else 3
        module = ProgramRecord(
            (
                bit_gate_function(
                    "main",
                    tuple(signature),
                    CircuitIR(
                        count,
                        (
                            Instruction(
                                "cx" if gate == "x" else "ccx", tuple(range(count))
                            ),
                        ),
                    ),
                ),
            )
        )
    function = import_module(module)

    def add(
        builder: ComponentProgramBuilder,
        ctrl: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        if gate == "x":
            result = builder.call_dict(
                function.name, {"ctrl": ctrl, "target": values["q"]}
            )
            return result["ctrl"], (result["target"],)
        if gate in ("cx", "cz"):
            control_name, target_name = (
                ("ctrl", "target") if gate == "cx" else ("q1", "q2")
            )
            result = builder.call_dict(
                function.name,
                {
                    control_name: np.array([ctrl, values[control_name]], dtype=object),
                    target_name: values[target_name],
                },
            )
            controls = np.asarray(result[control_name])
            return controls[0], (controls[1], result[target_name])
        if cy_profile == "cirq":
            result = builder.call_dict(function.name, {"ctrl": ctrl, "q": values["q"]})
            return result["ctrl"], (result["q"],)
        bits = np.asarray(values["q"])
        result = builder.call_dict(
            function.name,
            {"ctrl": np.array([ctrl, bits[0]], dtype=object), "target": bits[1]},
        )
        controls = np.asarray(result["ctrl"])
        return controls[0], (np.array([controls[1], result["target"]], dtype=object),)

    return add
