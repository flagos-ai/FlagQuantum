# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Dependencies follow locked Qualtran v0.7.0 qubitization_walk_operator.py.

"""Capture concrete Pauli walks with caller-owned alias signal environments."""

from math import pi
from typing import Any, cast

from ..compiler.formats.legacy_qubitization_walk_factories import WalkContext
from ..compiler.registers.component_adjoint import adjoint_component_calls
from ..compiler.registers.component_builder import ComponentProgramBuilder
from ..compiler.registers.preparation.block_encoding_component import (
    select_block_component_program,
)
from ..compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from ..compiler.registers.primitives.global_phase_component import scalar_phase_program
from ..core.ir.program import ProgramRecord
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError
from ._alias_preparation_component_library import capture_alias_preparation_pair
from ._default_linear_combination_oracles import _signature
from ._pauli_select_component_library import capture_pauli_select_pair
from ._qubitization_walk_component_library import capture_qubitization_walk_context


def capture_pauli_walk_context(source: object) -> WalkContext:
    """Bind actual Pauli/alias children, retaining all signal words across steps."""
    from qualtran import CtrlSpec
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )

    if not isinstance(source, QubitizationWalkOperator) or not isinstance(
        source.block_encoding, SelectBlockEncoding
    ):
        raise SerializationError(
            "Pauli walk requires an explicit SELECT block encoding"
        )
    foreign = cast(Any, source)
    block, reflection = foreign.block_encoding, foreign.reflect
    preparation = capture_alias_preparation_pair(reflection.prepare_gate)
    selection = capture_pauli_select_pair(block.select)
    # The source SelectPauliLCU declares adjoint() == self. Keep that explicit
    # source child identity inside the generic block inverse recipe.
    blocks = tuple(
        select_block_component_program(
            Signature(_signature(block.signature)),
            selection[0],
            adjoint=inverse,
            inverse=selection[0],
        )
        for inverse in (False, True)
    )
    positive_select = block.select.get_ctrl_system(CtrlSpec())[0]
    positive_selection = capture_pauli_select_pair(positive_select)
    positive_block = block.get_ctrl_system(CtrlSpec())[0]
    builder = ComponentProgramBuilder(Signature(_signature(positive_block.signature)))
    state = dict(builder.inputs)
    control_name = next(iter(builder.inputs))
    returned = builder.call_program(
        positive_selection[0], {"control": state.pop(control_name), **state}
    )
    returned[control_name] = returned.pop("control")
    positive_forward = builder.finalize(returned)
    positive_blocks = positive_forward, adjoint_component_calls(
        positive_forward, ((positive_selection[0], positive_selection[0]),)
    )
    junk = tuple(
        Register("signal_" + reg.name, reg.dtype, shape=reg.shape, side=reg.side)
        for reg in _signature(reflection.prepare_gate.junk_registers)
    )
    names = tuple(reg.name for reg in junk)
    selection_names = tuple(
        reg.name for reg in reflection.prepare_gate.selection_registers
    )

    def reflect(controlled: bool) -> ProgramRecord:
        return prepared_reflection_program(
            *preparation,
            selection_names,
            control_value=1 if controlled else None,
            global_phase=-1,
            scalar_phase=None if controlled else scalar_phase_program(pi),
            signal_junk_names=names,
        )

    ordinary_reflection, positive_reflection = reflect(False), reflect(True)
    return capture_qubitization_walk_context(
        source,
        block_owner=lambda _: (blocks[0], blocks[1]),
        reflection_owner=lambda _: (ordinary_reflection, ordinary_reflection),
        positive_block_owner=lambda _: positive_blocks,
        positive_reflection_owner=lambda _: (positive_reflection, positive_reflection),
        signal_junk=junk,
    )
