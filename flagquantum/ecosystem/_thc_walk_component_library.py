# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Captures locked Qualtran v0.7.0 chemistry/thc/walk_operator.py dependencies.

"""Capture actual THC source calls without inventing physical rotation bodies."""

from math import pi
from numbers import Integral
from typing import Any, cast

import numpy as np

from ..compiler.formats.legacy_qubitization_walk_factories import ExactPair, WalkContext
from ..compiler.formats.legacy_thc_preparation_factories import (
    thc_preparation_factories,
)
from ..compiler.formats.legacy_thc_selection_factories import _selection
from ..compiler.registers.component_builder import ComponentProgramBuilder
from ..compiler.registers.preparation.block_encoding_component import (
    select_block_component_program,
)
from ..compiler.registers.preparation.prepared_reflection_component import (
    literal_prepared_reflection_program,
)
from ..compiler.registers.primitives.global_phase_component import scalar_phase_program
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError
from ._default_linear_combination_oracles import _signature


def capture_thc_walk_context(source: object) -> WalkContext:
    """Replace actual THC factory children with owned literal directional graphs.

    RIGHT preparation outputs keep original consume/produce/free lifetimes.
    These graphs retain the source's narrow reflection and cleanup defects.
    Opaque rotation, comparator and erasure leaves refuse physical binding.
    No Givens network, corrected sentinel or unitary-bank extension is inferred.
    Positive inverse ownership reverses the stored specialized forward graph.
    The source's direct Controlled.adjoint constructor changes specialization;
    its separate literal block pair is retained without an inverse proof.
    """
    from qualtran import CtrlSpec
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
    from qualtran.bloqs.chemistry.thc.prepare import PrepareTHC
    from qualtran.bloqs.chemistry.thc.select_bloq import SelectTHC
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )

    if (
        not isinstance(source, QubitizationWalkOperator)
        or not isinstance(source.block_encoding, SelectBlockEncoding)
        or not isinstance(source.block_encoding.select, SelectTHC)
        or not isinstance(source.reflect.prepare_gate, PrepareTHC)
    ):
        raise SerializationError(
            "THC walk requires concrete original PREPARE and SELECT"
        )
    foreign = cast(Any, source)
    block, reflection = foreign.block_encoding, foreign.reflect
    prepare, select = reflection.prepare_gate, block.select
    if select.control_val is not None or reflection.global_phase != -1:
        raise SerializationError("THC context requires the ordinary source walk")

    def integer(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise SerializationError("THC source dimensions require exact integers")
        return int(value)

    mu, spin, keep, log = (
        integer(getattr(prepare, key))
        for key in ("num_mu", "num_spin_orb", "keep_bitsize", "log_block_size")
    )
    if mu < 1 or spin < 2 or keep < 1 or log < 0:
        raise SerializationError(
            "THC source dimensions are outside the logical profile"
        )
    rows = mu * (mu + 1) // 2 + spin // 2

    def dataset(key: str) -> tuple[int, ...]:
        values = np.asarray(getattr(prepare, key))
        if values.shape != (rows,) or any(
            not isinstance(value, (Integral, np.bool_)) for value in values.flat
        ):
            raise SerializationError(
                "THC source data require complete literal integer vectors"
            )
        return tuple(int(value) for value in values)

    selection_arguments = {
        key: integer(getattr(select, key))
        for key in (
            "num_mu",
            "num_spin_orb",
            "num_bits_theta",
            "keep_bitsize",
            "kr1",
            "kr2",
        )
    }

    def blocks(original: Any) -> ExactPair:
        controlled = original is not block
        control = 1 if controlled else None
        args = {**selection_arguments, "control_val": control}
        forward = _selection("thc_select", None, args)
        inverse = _selection("thc_select_inverse", None, args, adjoint=True)
        if not isinstance(forward, ProgramRecord) or not isinstance(
            inverse, ProgramRecord
        ):
            raise SerializationError("THC SELECT has no source-success logical body")
        signature = Signature(_signature(original.signature))
        if controlled:
            # The original generic Controlled block calls the specialized
            # SELECT once, mapping its scalar ctrl to SELECT's control port.
            control_port = next(iter(signature)).name

            def wrap(record: ProgramRecord) -> ProgramRecord:
                builder = ComponentProgramBuilder(signature)
                state = dict(builder.inputs)
                value = state.pop(control_port)
                outputs = builder.call_program(
                    record, {"control": value, **state}, entry=record.functions[0].name
                )
                outputs[control_port] = outputs.pop("control")
                return builder.finalize(outputs)

            return wrap(forward), wrap(inverse)
        return (
            select_block_component_program(signature, forward),
            select_block_component_program(
                signature, forward, adjoint=True, inverse=inverse
            ),
        )

    preparation: ExactPair | None = None
    if log:
        owners = thc_preparation_factories()
        forward = owners["qualtran.bloqs.chemistry.thc.prepare.PrepareTHC"](
            "thc_prepare",
            None,
            {
                "num_mu": mu,
                "num_spin_orb": spin,
                "keep_bitsize": keep,
                "log_block_size": log,
                "sum_of_l1_coeffs": float(prepare.sum_of_l1_coeffs),
                **{
                    key: dataset(key)
                    for key in ("alt_mu", "alt_nu", "alt_theta", "theta", "keep")
                },
            },
        )
        assert isinstance(forward, ProgramRecord)
        inverse = owners["qualtran._infra.adjoint.Adjoint"](
            "thc_prepare_inverse", None, {"subbloq": forward.functions[0]}
        )
        assert isinstance(inverse, ProgramRecord)
        preparation = forward, inverse

    def reflect(original: Any) -> ExactPair:
        if preparation is None:
            # The zero-bank source cannot decompose its PREPARE/reflection.
            # Preserve the readable header, never a zero-gate identity body.
            header = ProgramRecord(
                (
                    FunctionRecord(
                        "thc_reflection_unavailable",
                        (),
                        registers=_signature(original.signature),
                    ),
                )
            )
            return header, header
        control = original.control_val
        record = literal_prepared_reflection_program(
            *preparation,
            tuple(reg.name for reg in prepare.selection_registers),
            control_value=control,
            global_phase=original.global_phase,
            scalar_phase=scalar_phase_program(pi) if control is None else None,
        )
        if record.functions[0].registers != _signature(original.signature):
            raise SerializationError(
                "THC reflection source ports differ from native calls"
            )
        # ReflectionUsingPrepare declares return-self even for source defects.
        return record, record

    positive_block = block.get_ctrl_system(CtrlSpec())[0]
    positive_reflection = reflection.get_ctrl_system(CtrlSpec())[0]
    signature = Signature(_signature(foreign.signature))
    positive_blocks = blocks(positive_block)
    generic_select_inverse = _selection(
        "thc_generic_select_inverse",
        None,
        {**selection_arguments, "control_val": None},
        adjoint=True,
        generic_control=True,
    )
    assert isinstance(generic_select_inverse, ProgramRecord)
    source_inverse_builder = ComponentProgramBuilder(
        Signature(_signature(positive_block.adjoint().signature))
    )
    source_inverse = source_inverse_builder.finalize(
        source_inverse_builder.call_program(
            generic_select_inverse,
            source_inverse_builder.inputs,
            entry=generic_select_inverse.functions[0].name,
        )
    )
    return WalkContext(
        signature,
        blocks(block),
        reflect(reflection),
        positive_blocks,
        reflect(positive_reflection),
        source_signature=signature,
        positive_source_block_pair=(positive_blocks[0], source_inverse),
    )
