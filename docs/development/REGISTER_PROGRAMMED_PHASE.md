# Programmed phase resource

The locked Qualtran v0.7.0 `ZPowProgrammedAncilla` prepares
`(|0⟩ + exp(iπt)|1⟩)/√2`. FlagQuantum binds the source allocation to
one caller-owned clean wire and emits H followed by the exact phase.
The wire is a live resource output. The preparation uses zero scratch
wires and zero measurements.

Tests check the original allocation and both gate operands on two
physical bindings. They compare full complex amplitudes against the
source tensor contraction, including a coherent spectator. They also
exercise the `circuit.append(op)` binding without Qualtran.

The literal IR has one H and one unsynthesized phase in every tested
case. The source qubit model reports one qubit. Its gate model adds
one Clifford for H and classifies ZPow separately. For example,
the quarter-power case is one Clifford and one T in the source
model, while the exponent −3/8 is one Clifford and one rotation.

The source consuming component `ZPowUsingProgrammedAncilla` supplies
a resource call graph for the selected round count. An executable
measurement and feedback sequence requires its own reference and
acceptance.
