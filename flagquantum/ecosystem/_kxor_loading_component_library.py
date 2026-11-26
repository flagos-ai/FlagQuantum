# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Control routes follow locked Qualtran v0.7.0 load_kxor_instance.py.

"""Bind owned instance context without fabricating its absent source codec."""

from dataclasses import replace

import numpy as np

from ..algorithms.kxor_loading import (
    LoadConstraintScopes,
    LoadUniqueScopeIndex,
    PRGAUniqueConstraintRHS,
)
from ..algorithms.kxor_loading_component import kxor_loading_program
from ..compiler.formats.legacy_component_arguments import ControlSpecification
from ..compiler.formats.legacy_controlled_factories import controlled_via_and_factories
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from ..compiler.registers.component_control import (
    ControlAdder,
    _stored_control_adder,
    quantum_control_signature,
)
from ..compiler.registers.control.specialized_component_control import (
    multi_controlled_component_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import quantum_bit_type
from ..errors import SerializationError


def kxor_loading_component(
    oracle: LoadConstraintScopes | LoadUniqueScopeIndex | PRGAUniqueConstraintRHS,
    controls: ControlSpecification | None = None,
) -> ProgramRecord:
    """Retain successful wrappers around the actual source-positive declaration.

    Controlled scope loading has no source non-THRU decomposition. Index and RHS
    remain model-only. Their control roots may be inspected but physical binding
    must refuse the missing positive body; it is never replaced by an invented
    controlled XOR or amplitude rotation.
    """
    base = kxor_loading_program(oracle)
    if controls is None:
        return base
    scalar = (
        len(controls.clauses) == 1
        and controls.clauses[0][0] == quantum_bit_type()
        and controls.clauses[0][1].shape == ()
        and controls.clauses[0][1].item() == 1
    )
    already_controlled = (
        isinstance(oracle, PRGAUniqueConstraintRHS) and oracle.is_controlled
    )
    if already_controlled:
        declaration = FunctionRecord(
            "source_rhs", (), registers=tuple(oracle.signature)
        )
        builder = ComponentProgramBuilder(oracle.signature, (declaration,))
        body = builder.finalize(builder.call_dict(declaration.name, builder.inputs))
        single = (
            len(controls.clauses) == 1
            and controls.clauses[0][0] == quantum_bit_type()
            and controls.clauses[0][1].shape == ()
        )
        if single:
            return multi_controlled_component_program(
                body, "ctrl", (int(controls.clauses[0][1].item()), 1)
            )
        positive = _rename_program(
            multi_controlled_component_program(body, "ctrl", (1, 1)),
            "positive_loader",
        )
    else:
        positive_ports = (
            tuple(replace(oracle, is_controlled=True).signature)
            if isinstance(oracle, PRGAUniqueConstraintRHS)
            else tuple(
                quantum_control_signature(oracle.signature, ((quantum_bit_type(), ()),))
            )
        )
        positive = ProgramRecord(
            (FunctionRecord("positive_loader", (), registers=positive_ports),)
        )
    if scalar:
        return _rename_program(positive, "main")
    root = base.functions[0]

    def owner(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child != root:
            raise SerializationError("Unknown k-XOR loader control child")
        if not already_controlled:
            return positive, _stored_control_adder(positive.functions[0])

        def add(
            builder: ComponentProgramBuilder,
            control: BindingCollection,
            values: dict[str, BindingCollection],
        ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
            old_control = values.pop("ctrl")
            outputs = builder.call_dict(
                positive.functions[0].name,
                {"ctrl": np.array((control, old_control), dtype=object), **values},
            )
            returned = np.asarray(outputs.pop("ctrl"))
            return returned[0], (returned[1], *outputs.values())

        return positive, add

    factory = controlled_via_and_factories(control_system=owner)[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]
    record = factory("main", None, {"subbloq": root, "ctrl_spec": controls})
    return record if isinstance(record, ProgramRecord) else ProgramRecord((record,))
