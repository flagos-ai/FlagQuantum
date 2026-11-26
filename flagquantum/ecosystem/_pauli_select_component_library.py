# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Metadata and controls follow locked Qualtran v0.7.0 SelectPauliLCU,
# mcmt/specialized_ctrl.py and mcmt/controlled_via_and.py.

"""Capture signed source Pauli data without its object-array wire codec."""

from typing import Any, cast

import numpy as np

from ..compiler.formats.legacy_control_specs import (
    control_registers_from_spec,
    deserialize_control_spec,
)
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.component_adjoint import adjoint_component_calls
from ..compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from ..compiler.registers.component_control import (
    ControlAdder,
    control_component_via_predicate,
)
from ..compiler.registers.component_gate_function import bit_gate_function
from ..compiler.registers.control.component_predicate import control_predicate_program
from ..compiler.registers.control.specialized_component_control import (
    multi_controlled_component_pair,
)
from ..compiler.registers.lookup.pauli_select_component import (
    pauli_select_component_pair,
)
from ..core.ir import CircuitIR, Instruction
from ..core.ir.program import ProgramRecord
from ..core.ir.program.register_types import quantum_bit_type
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError
from ._default_linear_combination_oracles import _signature
from ._pauli_select_rows import (
    pauli_select_control,
    pauli_select_selection_width,
    pauli_select_width,
    sign_of_real_pauli_word,
)


def capture_pauli_select_pair(source: object) -> tuple[ProgramRecord, ProgramRecord]:
    """Replace source payload immediately, retaining its sign-of-real convention."""
    from qualtran import Adjoint
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU

    foreign = cast(Any, source)
    if isinstance(source, Adjoint):
        forward, backward = capture_pauli_select_pair(foreign.subbloq)
        return backward, forward
    if isinstance(source, _MultiControlledFromSinglyControlled):
        forward = capture_pauli_select_pair(foreign.ctrl_bloq)[0]
        inverse = capture_pauli_select_pair(foreign.ctrl_bloq.adjoint())[0]
        return multi_controlled_component_pair(
            forward, inverse, foreign.ctrl_reg_name, tuple(foreign.cvs)
        )
    if isinstance(source, ControlledViaAnd):
        return _capture_structured_controls(foreign)
    if not isinstance(source, SelectPauliLCU):
        raise SerializationError("Expected an explicit source Pauli SELECT")
    selection_width = pauli_select_selection_width(foreign.selection_bitsize)
    if type(selection_width) is not int:
        raise ValueError(
            "Pauli SELECT Program capture requires a concrete selection width"
        )
    words: list[str] = []
    for pauli in foreign.select_unitaries:
        try:
            words.append(
                sign_of_real_pauli_word(pauli.pauli_mask, complex(pauli.coefficient))
            )
        except ValueError as error:
            raise SerializationError(
                "Source sign-of-real Pauli leaf is not unitary"
            ) from error
    return pauli_select_component_pair(
        selection_width,
        pauli_select_width(foreign.target_bitsize, "target"),
        tuple(words),
        control_value=pauli_select_control(foreign.control_val),
    )


def _positive_control(source: Any) -> tuple[ProgramRecord, ProgramRecord, ControlAdder]:
    """Convert the original control callback to a closure over owned definitions."""
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.bloqs.multiplexers.select_pauli_lcu import SelectPauliLCU

    if not isinstance(source, (SelectPauliLCU, ControlledViaAnd)):
        raise SerializationError(
            "Pauli SELECT control callback requires an explicit supported child"
        )
    selected, _ = source.get_ctrl_system(CtrlSpec())
    forward = capture_pauli_select_pair(selected)[0]
    inverse = capture_pauli_select_pair(selected.adjoint())[0]
    name = "positive"
    while any(
        f.name == name for record in (forward, inverse) for f in record.functions[1:]
    ):
        name += "_"
    forward, inverse = _rename_program(forward, name), _rename_program(inverse, name)
    names = tuple(reg.name for reg in forward.functions[0].registers or ())
    packed = isinstance(source, SelectPauliLCU) and source.control_val is not None

    def add(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        if packed:
            previous = values.pop("control")
            outputs = builder.call_dict(
                name,
                {"control": np.array([control, previous], dtype=object), **values},
            )
            controls = outputs.pop("control")
            assert isinstance(controls, np.ndarray)
            return controls[0], (controls[1], *outputs.values())
        outputs = builder.call_dict(
            name, dict(zip(names, (control, *values.values()), strict=True))
        )
        new_control = outputs.pop(names[0])
        return new_control, tuple(outputs.values())

    return forward, inverse, add


def _capture_structured_controls(source: Any) -> tuple[ProgramRecord, ProgramRecord]:
    """Reuse the owned predicate root and exact source-specialized child adjoint."""
    from qualtran.serialization.ctrl_spec import ctrl_spec_to_proto

    forward_child, inverse_child, add = _positive_control(source.subbloq)
    builder = ComponentProgramBuilder(
        Signature(_signature(source.subbloq.signature)), forward_child.functions
    )
    ports = _signature(source.signature)
    controls = ports[: source.ctrl_spec.num_ctrl_reg]
    patterns = control_registers_from_spec(
        deserialize_control_spec(
            ctrl_spec_to_proto(source.ctrl_spec).SerializeToString()
        )
    )
    forward = control_component_via_predicate(
        builder.finalize(builder.inputs), {"main": add}, controls, patterns
    )
    inner = tuple(
        Register(f"ctrl_{index}", reg.dtype, reg.shape)
        for index, reg in enumerate(controls)
    )
    compute = control_predicate_program(inner, patterns)
    erase = control_predicate_program(inner, patterns, uncompute=True)
    x = ProgramRecord(
        (
            bit_gate_function(
                "x",
                (Register("q", quantum_bit_type()),),
                CircuitIR(1, (Instruction("x", (0,)),)),
            ),
        )
    )
    backward = adjoint_component_calls(
        forward,
        ((forward_child, inverse_child), (compute, erase), (erase, compute), (x, x)),
    )
    return forward, backward
