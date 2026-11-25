# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TwoBitFFFT.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/two_bit_ffft.py

"""Two-qubit fermionic Fourier transform with the source phase convention."""

from math import pi

from ...compiler._component_trace.resource_classification import classify_resource_gate
from ...compiler._component_trace.resource_generalization import (
    CliffordCall,
    PhaseCall,
    RotationCall,
)
from ...compiler._component_trace.resource_model import ComponentCost, GateCost
from ...compiler._component_trace.symbolic_math import SymbolicReal
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.component_builder import ComponentProgramBuilder
from ...compiler.registers.component_gate_function import bit_gate_function
from ...compiler.registers.layout.fermionic_signatures import fermionic_pair_signature
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature

_ADJOINT = {"s": "sdg", "t": "tdg", "tdg": "t"}


def _fourier_angle(k: int, n: int) -> float:
    if type(k) is not int or type(n) is not int or n == 0:
        raise ValueError("Fermionic Fourier transform requires integer k and nonzero n")
    return 2 * pi * k / n


def source_two_bit_fermionic_fourier_signature() -> Signature:
    """Describe the two through bit ports in the source order."""
    return fermionic_pair_signature()


def source_two_bit_fermionic_fourier_calls(
    k: int, n: int, *, epsilon: SymbolicReal = 1e-10
) -> dict[RotationCall | PhaseCall | CliffordCall, int]:
    """Keep the source's precision and its combined T/adjoint-T declaration."""
    return {
        RotationCall("rz", _fourier_angle(k, n), precision=epsilon): 1,
        PhaseCall("s"): 3,
        CliffordCall("h", 1): 6,
        PhaseCall("t"): 2,
        CliffordCall("cx", 2): 3,
    }


def source_two_bit_fermionic_fourier_cost(k: int, n: int) -> ComponentCost:
    """Count the locked Rz, three S, six H, two T and three CNOT calls."""
    phase = classify_resource_gate(RotationCall("rz", _fourier_angle(k, n)))
    return ComponentCost(
        "TwoBitFFFT",
        2,
        GateCost(
            t=2 + int(phase == "t"),
            clifford=12 + int(phase == "clifford"),
            rotation=int(phase == "rotation"),
        ),
    )


def two_bit_fermionic_fourier_ir(
    x: int, y: int, k: int, n: int, *, adjoint: bool = False
) -> tuple[Instruction, ...]:
    """Bind the two-qubit fermionic Fourier transform and its exact inverse."""
    _validate_wires((x, y))
    angle = _fourier_angle(k, n)
    if type(adjoint) is not bool:
        raise ValueError("Fermionic Fourier transform direction must be boolean")

    fixed = (
        Instruction("s", (y,)),
        Instruction("h", (x,)),
        Instruction("h", (y,)),
        Instruction("cx", (x, y)),
        Instruction("h", (x,)),
        Instruction("s", (x,)),
        Instruction("tdg", (y,)),
        Instruction("cx", (x, y)),
        Instruction("h", (x,)),
        Instruction("t", (y,)),
        Instruction("cx", (x, y)),
        Instruction("h", (x,)),
        Instruction("h", (y,)),
        Instruction("s", (x,)),
    )
    if adjoint:
        return (
            *(
                Instruction(_ADJOINT.get(g.name, g.name), g.wires)
                for g in reversed(fixed)
            ),
            Instruction("rz", (x,), {"theta": -angle}),
        )
    return (Instruction("rz", (x,), {"theta": angle}), *fixed)


def two_bit_fermionic_fourier_program(
    k: int, n: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Store the accepted literal sequence and exact inverse as typed bit calls.

    Caller-owned x/y need no work or feedback. The source is_adjoint flag is
    ignored by its decomposition; native adjoint remains the exact inverse.
    """
    builder = ComponentProgramBuilder(source_two_bit_fermionic_fourier_signature())
    wires = [builder.inputs["x"], builder.inputs["y"]]
    for gate in two_bit_fermionic_fourier_ir(0, 1, k, n, adjoint=adjoint):
        names = ("ctrl", "target") if gate.name == "cx" else ("q",)
        function = bit_gate_function(
            "fourier_" + gate.name,
            tuple(Register(name, quantum_bit_type()) for name in names),
            CircuitIR(
                len(names),
                (Instruction(gate.name, tuple(range(len(names))), gate.params),),
            ),
        )
        result = builder.call_program(
            ProgramRecord((function,)),
            dict(zip(names, (wires[i] for i in gate.wires), strict=True)),
            entry=function.name,
        )
        for name, wire in zip(names, gate.wires, strict=True):
            wires[wire] = result[name]
    return builder.finalize(dict(zip(("x", "y"), wires, strict=True)))
