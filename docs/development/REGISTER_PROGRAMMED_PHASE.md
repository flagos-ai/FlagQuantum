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

## Bounded consumption

The native rotation follows Jones et al., Figs. 3–4. Each attempted
round prepares a phase resource, applies CX from data to work, measures
work in Z, and clears a measured one with X. A measured one advances to
twice the exponent. A zero stops the rotation cascade. With final
correction enabled, the all-ones history applies a phase at the next
doubled exponent. The resource and correction tolerances follow the
locked Qualtran call graph. [Jones et al., Figs. 3–4](https://arxiv.org/pdf/1204.0567#page=7)

The original padded route uses public flat IR, which rejects conditional measurement. Later
rounds predicate preparation and CX on all preceding failures, and
measure the idle zero work wire after an earlier success. Zero also
encodes the padded slots. With correction disabled, all ones identifies
the uncorrected failure. Tests project every raw history through three
rounds, including impossible histories and the failure-branch complex
scalar, with a coherent spectator. The native append path is also
checked through the local trajectory executor.

| Three-round resource | Native emitted IR | Locked Qualtran call graph |
| --- | --- | --- |
| Preparation | 3 H, 3 unsynthesized phases | 3 programmed ancillas |
| Consumption | 3 CX, 3 conditional X, 3 unconditional measurements | 3 CNOT, 3 X, 3 MeasureZ |
| Final correction enabled | 1 conditional unsynthesized phase | 1 ZPow at eight times the base exponent |
| Physical work | 1 reusable clean wire | Resource-state preparation counted by call graph |

For an ideal trajectory with (k) attempts, (3-k) measurements are
padding. The preparation and entangling gates execute (k) times, and
X executes once per failed attempt. Literal phase counts are before
synthesis. This records the original padded implementation; the existing
internal guarded-effect route below supersedes that execution limit.


## Physical early exit

`ProgrammedRotationExample` now selects `early_exit=True` in the shared emitter.
It reuses the existing internal `apply_control_guards` and local guarded-effect
execution. Before each measurement after the first, a classical XOR initializes
that round's result slot to zero. The measurement executes only if all earlier
rounds failed; otherwise the initialized zero remains without a projection or
RNG draw. Preparation, CX and reset already carry the same failure predicates.
The work wire remains clean after every attempt and after early success.

The lower-level emitter retains its existing padded default for literal-stream
comparisons. Both routes have identical complete complex branch maps and density
channels; the guarded route uses fewer executed measurements. Result records
retain fixed-width zero suffixes, but these suffixes are classical initialization,
not physical measurements. Three rounds add two classical initializers, no
quantum scratch and no analytical T cost. The original worst-case model remains
three CNOT, three X, three measurements and three resource states.

Acceptance compares every raw history through four rounds, arbitrary outcome
slot order, coherent spectators, exact failure phases and restored work. Local
trajectory and batched execution check the actual measurement count equals the
sum of attempted rounds, with and without terminal correction. Public
conditional measurement, typed conditional-result merges, remote provider and
hybrid-session support are unchanged; no new Core contract is required.
