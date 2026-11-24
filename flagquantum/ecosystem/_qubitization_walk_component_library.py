# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Metadata follows locked Qualtran v0.7.0 qubitization_walk_operator.py.

"""Capture actual walk signal and source order into native dependency ownership."""

from collections.abc import Callable
from typing import Any, cast

from ..compiler.formats.legacy_qubitization_walk_factories import ExactPair, WalkContext
from ..core.ir.program import ProgramRecord
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError
from ._default_linear_combination_oracles import _signature

ExternalPairOwner = Callable[[object], ExactPair]


def capture_qubitization_walk_context(
    source: object,
    *,
    block_owner: ExternalPairOwner,
    reflection_owner: ExternalPairOwner,
    positive_block_owner: ExternalPairOwner | None = None,
    positive_reflection_owner: ExternalPairOwner | None = None,
    signal_junk: tuple[Register, ...] = (),
) -> WalkContext:
    """Replace actual source dependencies immediately, retaining no SDK object.

    Reflection is taken from source.reflect, not source.prepare: LCU signal
    state is identity even when the latter property exposes original PREPARE.
    The source's actual set-derived signature order is retained unchanged.
    """
    from qualtran import CtrlSpec
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )

    if not isinstance(source, QubitizationWalkOperator):
        raise SerializationError("Expected an explicit source qubitization walk")
    if (positive_block_owner is None) != (positive_reflection_owner is None):
        raise SerializationError("Positive walk owners must be supplied together")
    foreign = cast(Any, source)

    def pair(
        owner: ExternalPairOwner, child: Any, extra: tuple[Register, ...] = ()
    ) -> ExactPair:
        result = owner(child)
        ports = (*_signature(child.signature), *extra)
        if (
            not isinstance(result, tuple)
            or len(result) != 2
            or any(
                not isinstance(record, ProgramRecord)
                or record.physical_context is not None
                or not record.functions
                or record.functions[0].registers != ports
                for record in result
            )
        ):
            raise SerializationError(
                "Walk owner must supply matching native logical pairs"
            )
        return result

    block, reflection = foreign.block_encoding, foreign.reflect
    source_signature = Signature(_signature(foreign.signature))
    if signal_junk:
        original_junk = _signature(reflection.prepare_gate.junk_registers)
        if len(original_junk) != len(signal_junk) or any(
            (original.dtype, original.shape, original.side)
            != (kept.dtype, kept.shape, kept.side)
            for original, kept in zip(original_junk, signal_junk, strict=True)
        ):
            raise SerializationError(
                "Retained signal junk must match every source junk word"
            )
        names = [reg.name for reg in (*tuple(source_signature), *signal_junk)]
        if len(set(names)) != len(names):
            raise SerializationError(
                "Retained signal junk must not alias source walk ports"
            )
    ordinary_block, ordinary_reflection = pair(block_owner, block), pair(
        reflection_owner, reflection, signal_junk
    )
    positive_block = positive_reflection = None
    if positive_block_owner is not None and positive_reflection_owner is not None:
        positive_block = pair(
            positive_block_owner, block.get_ctrl_system(CtrlSpec())[0]
        )
        positive_reflection = pair(
            positive_reflection_owner,
            reflection.get_ctrl_system(CtrlSpec())[0],
            signal_junk,
        )
    return WalkContext(
        Signature((*tuple(source_signature), *signal_junk)),
        ordinary_block,
        ordinary_reflection,
        positive_block,
        positive_reflection,
        source_signature,
    )
