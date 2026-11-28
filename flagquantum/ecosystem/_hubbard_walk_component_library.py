# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Dependencies follow locked Qualtran v0.7.0 Hubbard qubitization.

"""Replace actual Hubbard walk dependencies with existing native graph owners."""

from math import pi
from numbers import Integral
from typing import Any, cast

from ..compiler.formats.legacy_qubitization_walk_factories import ExactPair, WalkContext
from ..compiler.registers.component_adjoint import adjoint_component_calls
from ..compiler.registers.component_builder import ComponentProgramBuilder
from ..compiler.registers.control.hubbard_select_component import hubbard_select_program
from ..compiler.registers.preparation.block_encoding_component import (
    select_block_component_program,
)
from ..compiler.registers.preparation.hubbard_prepare_component import (
    hubbard_prepare_program,
)
from ..compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from ..compiler.registers.primitives.global_phase_component import scalar_phase_program
from ..core.ir.program import ProgramRecord
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError
from ._default_linear_combination_oracles import _signature
from ._packed_oracle_component_library import _prepare_angles


def capture_hubbard_walk_context(source: object) -> WalkContext:
    """Retain actual source signatures, control recipes and private temp ownership.

    This is literal source composition. The source's private reflection temp
    can be dirty; capture makes no clean-release or whole-operator certificate.
    Numerical callers needing arbitrary signal inputs must retain their explicit
    caller-owned junk extension instead.
    """
    from qualtran import CtrlSpec
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard,
    )
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard import (
        SelectHubbard,
    )
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )

    if (
        not isinstance(source, QubitizationWalkOperator)
        or not isinstance(source.block_encoding, SelectBlockEncoding)
        or not isinstance(source.block_encoding.select, SelectHubbard)
        or not isinstance(source.reflect.prepare_gate, PrepareHubbard)
    ):
        raise SerializationError("Hubbard walk requires original PREPARE and SELECT")
    foreign = cast(Any, source)
    block, reflection = foreign.block_encoding, foreign.reflect
    prepare, select = reflection.prepare_gate, block.select
    size = prepare.x_dim
    if (
        isinstance(size, bool)
        or not isinstance(size, Integral)
        or size < 2
        or any(value != size for value in (prepare.y_dim, select.x_dim, select.y_dim))
        or select.control_val is not None
        or reflection.control_val is not None
        or reflection.global_phase != -1
    ):
        raise SerializationError("Hubbard context requires an ordinary square walk")
    dimension = int(size)
    angles = _prepare_angles(dimension, prepare.t, prepare.u)
    if angles is None:
        raise SerializationError("Hubbard preparation requires owned source angles")
    preparation = (
        hubbard_prepare_program(dimension, *angles),
        hubbard_prepare_program(dimension, *angles, adjoint=True),
    )

    def selection(controlled: bool) -> ExactPair:
        return (
            hubbard_select_program(dimension, controlled=controlled),
            hubbard_select_program(dimension, controlled=controlled, adjoint=True),
        )

    ordinary = selection(False)
    signature = Signature(_signature(block.signature))
    blocks = (
        select_block_component_program(signature, ordinary[0]),
        select_block_component_program(
            signature, ordinary[0], inverse=ordinary[1], adjoint=True
        ),
    )
    positive_source = block.get_ctrl_system(CtrlSpec())[0]
    positive_select = selection(True)
    builder = ComponentProgramBuilder(Signature(_signature(positive_source.signature)))
    state = dict(builder.inputs)
    control_name = next(iter(state))
    returned = builder.call_program(
        positive_select[0], {"control": state.pop(control_name), **state}
    )
    returned[control_name] = returned.pop("control")
    positive = builder.finalize(returned)
    positive_blocks = positive, adjoint_component_calls(positive, (positive_select,))

    def reflect(controlled: bool) -> ProgramRecord:
        return prepared_reflection_program(
            *preparation,
            tuple(register.name for register in prepare.selection_registers),
            control_value=1 if controlled else None,
            global_phase=-1,
            scalar_phase=None if controlled else scalar_phase_program(pi),
        )

    reflected, positive_reflected = reflect(False), reflect(True)
    source_signature = Signature(_signature(foreign.signature))
    return WalkContext(
        source_signature,
        blocks,
        (reflected, reflected),
        positive_blocks,
        (positive_reflected, positive_reflected),
        source_signature,
    )
