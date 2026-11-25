# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Captures locked Qualtran v0.7.0 sparse chemistry Walk dependencies.

"""Capture sparse walks with an explicit unitary bank and signal extension."""

from math import pi
from numbers import Integral
from typing import Any, cast

import numpy as np

from ..compiler.formats.legacy_qubitization_walk_factories import ExactPair, WalkContext
from ..compiler.registers.component_adjoint import adjoint_component_calls
from ..compiler.registers.component_builder import ComponentProgramBuilder
from ..compiler.registers.lookup.qroam_bank_unitary_component import (
    qroam_bank_unitary_pair,
)
from ..compiler.registers.lookup.source_qroam_clean import QroamCleanTable
from ..compiler.registers.lookup.sparse_chemistry_select_component import (
    sparse_select_component_pair,
)
from ..compiler.registers.preparation.block_encoding_component import (
    select_block_component_program,
)
from ..compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from ..compiler.registers.preparation.sparse_chemistry_prepare_component import (
    sparse_prepare_component_pair,
)
from ..compiler.registers.primitives.global_phase_component import scalar_phase_program
from ..core.ir.program import ProgramRecord
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError
from ._default_linear_combination_oracles import _signature


def capture_sparse_walk_context(source: object) -> WalkContext:
    """Retain actual source order while explicitly extending bank ownership.

    Original source RIGHT preparation banks become caller-owned signal words
    through the independently verified batch-XOR/routing unitary. This native
    context has additional ports and is not the original source constructor
    signature. Default source factories and their model-only inverse remain
    unchanged. No SDK object is retained in the returned context.
    """
    from qualtran import CtrlSpec
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
    from qualtran.bloqs.chemistry.sparse.prepare import PrepareSparse
    from qualtran.bloqs.chemistry.sparse.select_bloq import SelectSparse
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )

    if (
        not isinstance(source, QubitizationWalkOperator)
        or not isinstance(source.block_encoding, SelectBlockEncoding)
        or not isinstance(source.block_encoding.select, SelectSparse)
        or not isinstance(source.reflect.prepare_gate, PrepareSparse)
    ):
        raise SerializationError("Sparse walk requires concrete sparse PREPARE/SELECT")
    foreign = cast(Any, source)
    block, reflection = foreign.block_encoding, foreign.reflect
    prepare = reflection.prepare_gate
    if block.select.control_val is not None or reflection.global_phase != -1:
        raise SerializationError(
            "Sparse walk capture requires the ordinary source root"
        )
    lookup = prepare.build_qrom_bloq()
    dimensions = (
        *lookup.data_shape,
        *lookup.log_block_sizes,
        *lookup.target_bitsizes,
        prepare.num_non_zero,
        block.select.num_spin_orb,
    )
    if any(not isinstance(n, Integral) or isinstance(n, bool) for n in dimensions):
        raise SerializationError("Sparse walk requires concrete integer dimensions")
    if any(
        not isinstance(value, (Integral, np.bool_))
        for data in lookup.data
        for value in data.flat
    ):
        raise SerializationError("Sparse walk lookup requires literal integer data")
    table = QroamCleanTable(
        tuple(int(n) for n in lookup.data_shape),
        tuple(int(n) for n in lookup.log_block_sizes),
        tuple(tuple(int(value) for value in data.flat) for data in lookup.data),
    )
    through = tuple(
        Register(reg.name, reg.dtype, shape=reg.shape)
        for reg in _signature(prepare.signature)
    )
    preparation = sparse_prepare_component_pair(
        Signature(through),
        int(prepare.num_non_zero),
        qroam_bank_unitary_pair(table, tuple(int(n) for n in lookup.target_bitsizes)),
    )
    selection = sparse_select_component_pair(int(block.select.num_spin_orb))
    block_signature = Signature(_signature(block.signature))
    ordinary_blocks: ExactPair = (
        select_block_component_program(block_signature, selection[0]),
        select_block_component_program(
            block_signature, selection[0], adjoint=True, inverse=selection[1]
        ),
    )
    positive_source = block.get_ctrl_system(CtrlSpec())[0]
    positive_selection = sparse_select_component_pair(
        int(block.select.num_spin_orb), control_value=1
    )
    builder = ComponentProgramBuilder(Signature(_signature(positive_source.signature)))
    state = dict(builder.inputs)
    control_name = next(iter(state))
    returned = builder.call_program(
        positive_selection[0], {"control": state.pop(control_name), **state}
    )
    returned[control_name] = returned.pop("control")
    positive_forward = builder.finalize(returned)
    positive_blocks = positive_forward, adjoint_component_calls(
        positive_forward, (positive_selection,)
    )
    names = tuple(reg.name for reg in prepare.selection_registers)
    junk = tuple(
        Register("signal_" + reg.name, reg.dtype, shape=reg.shape)
        for reg in through
        if reg.name not in names
    )

    def reflect(controlled: bool) -> ProgramRecord:
        return prepared_reflection_program(
            *preparation,
            names,
            control_value=1 if controlled else None,
            global_phase=-1,
            scalar_phase=None if controlled else scalar_phase_program(pi),
            signal_junk_names=tuple(reg.name for reg in junk),
        )

    ordinary_reflection, positive_reflection = reflect(False), reflect(True)
    original = Signature(_signature(foreign.signature))
    return WalkContext(
        Signature((*original, *junk)),
        ordinary_blocks,
        (ordinary_reflection, ordinary_reflection),
        positive_blocks,
        (positive_reflection, positive_reflection),
        original,
    )
