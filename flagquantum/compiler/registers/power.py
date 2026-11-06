# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Power.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/power.py

"""Literal repetition of an already bound, in-place register operation."""

from ...core._composition import copy_instruction, inverted_instruction
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...core.operator_schema import MAX_POWER_REPEATS


def register_power_ir(
    body: CircuitIR, repetitions: int, *, adjoint: bool = False
) -> tuple[Instruction, ...]:
    """Emit the source's repeated THRU calls over the same bound wires.

    The caller owns the register binding and any clean scratch contract.
    Classical feedback must be defined within each iteration. Scratch
    slots are overwritten by each call and hold the final call's results.
    Terminal result requests cannot be repeated as register operations.

    For a measured operation, bind its explicit inverse as the body when
    repeating the adjoint. Automatic inversion uses the core unitary rules.
    """
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("Register power requires a positive integer")
    if type(adjoint) is not bool:
        raise ValueError("Register power adjoint must be boolean")
    if body.observables or body.measurements:
        raise ValueError(
            "Register power requires an in-place body without result requests"
        )
    if len(body.instructions) * repetitions > MAX_POWER_REPEATS:
        raise ValueError(
            f"Register power exceeds {MAX_POWER_REPEATS} emitted instructions"
        )

    validate_classical_control(body.instructions, n_qubits=body.n_wires)
    instructions = body.instructions
    if adjoint:
        instructions = tuple(
            inverted_instruction(gate) for gate in reversed(instructions)
        )
    if not instructions:
        return ()
    return tuple(
        copy_instruction(gate) for _ in range(repetitions) for gate in instructions
    )
