# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 PhasingViaCostFunction.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/phasing_via_cost_function.py

"""Bind cost evaluation, variable phasing and the original evaluation adjoint."""

from collections.abc import Callable, Mapping

from ...core.ir import Instruction
from .bitwise import _validate_wires

RegisterPorts = Mapping[str, tuple[int, ...]]
BoundStage = Callable[[RegisterPorts], tuple[Instruction, ...]]


def cost_function_phasing_ir(
    inputs: RegisterPorts,
    outputs: RegisterPorts,
    phase_extras: RegisterPorts,
    *,
    cost_names: tuple[str, ...],
    evaluate: BoundStage,
    phase: BoundStage,
    erase: BoundStage,
) -> tuple[Instruction, ...]:
    """Compose three bound emitters with shared named cost outputs.

    Input registers and phase extras survive. The evaluator owns the
    clean output bindings, including scratch that the phase stage cannot
    access. Erase consumes the same physical binding as evaluation and
    must implement its adjoint, including any measurement feedback.
    """
    evaluation_ports = {**inputs, **outputs}
    all_ports = {**evaluation_ports, **phase_extras}
    if len(all_ports) != len(inputs) + len(outputs) + len(phase_extras):
        raise ValueError("Cost phasing port names must be distinct")
    if not cost_names or len(set(cost_names)) != len(cost_names):
        raise ValueError("Cost phasing requires distinct cost output names")
    if any(not outputs.get(name) for name in cost_names):
        raise ValueError("Each phase cost must bind a nonempty evaluation output wires")
    _validate_wires(tuple(wire for word in all_ports.values() for wire in word))
    phase_ports = {name: outputs[name] for name in cost_names} | dict(phase_extras)

    gates: list[Instruction] = []
    for emit, ports in (
        (evaluate, evaluation_ports),
        (phase, phase_ports),
        (erase, evaluation_ports),
    ):
        allowed = {wire for word in ports.values() for wire in word}
        stage = emit(ports.copy())
        if any(wire not in allowed for gate in stage for wire in gate.wires):
            raise ValueError("Cost phasing stage used a wire outside its port binding")
        gates.extend(stage)
    return tuple(gates)
