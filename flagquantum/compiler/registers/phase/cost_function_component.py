# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 rotations/phasing_via_cost_function.py.

"""Compose owned evaluation, variable phase and exact evaluation erasure calls."""

from ....core.ir.program import ProgramRecord
from ....core.ir.program.registers import RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder


def cost_function_phase_program(
    evaluate: ProgramRecord,
    phase: ProgramRecord,
    erase: ProgramRecord,
    *,
    cost_names: tuple[str, ...],
) -> ProgramRecord:
    """Keep source stage order and privately owned RIGHT cost outputs.

    Each supplied Program uses its first function as entry. Erase must be the
    exact inverse on the evaluator's image, including measured feedback.
    Evaluation inputs and phase extras form the outer source signature.
    A phase oracle may have no cost ports; evaluation and erasure still run.
    """
    evaluation = Signature(evaluate.functions[0].registers or ())
    phasing = Signature(phase.functions[0].registers or ())
    erasure = Signature(erase.functions[0].registers or ())
    if len(set(cost_names)) != len(cost_names):
        raise ValueError("Cost phasing requires distinct cost output names")
    for name in cost_names:
        source, target = evaluation.get_right(name), phasing.get_left(name)
        if (
            source.side != RegisterSide.RIGHT
            or target.side != RegisterSide.THRU
            or source.bitsize != target.bitsize
            or source.shape != target.shape
        ):
            raise ValueError("Cost phasing requires matching RIGHT and THRU cost ports")
    extras = tuple(r for r in phasing if r.name not in cost_names)
    builder = ComponentProgramBuilder(Signature((*evaluation.lefts(), *extras)))
    state = dict(builder.inputs)
    for record, signature in (
        (evaluate, evaluation),
        (phase, phasing),
        (erase, erasure),
    ):
        arguments = {r.name: state.pop(r.name) for r in signature.lefts()}
        state.update(
            builder.call_program(record, arguments, entry=record.functions[0].name)
        )
    return builder.finalize(state)
