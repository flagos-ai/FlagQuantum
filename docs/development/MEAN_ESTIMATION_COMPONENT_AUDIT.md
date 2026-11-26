# Mean-estimation dependency composition

Reference: complete unchanged Qualtran v0.7.0
`mean_estimation/mean_estimation_operator.py`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

## Interface recorded before implementation

Use the existing native CodeForRandomVariable descriptor, exact PREPARE pair and
exact encoder pair. Selection descriptors must agree literally; encoder target
names come from the code descriptor in declared order. Encoder private work and
synthesizer junk remain independent child-owned lifetimes. A clean-domain proof
is required for reflection junk: inverse PREPARE alone does not establish it.
An optional signal_junk_names binding reuses the existing reflection owner's
caller-owned THRU mode when clean release is not valid. Those words are explicit
logical inputs/outputs before binding, not private work relabeled after a test.
This preserves the full source physical action but is not the source's smaller
selection-only public ABI or an assertion that its garbage can be discarded.

ComplexPhaseOracle keeps its existing forward Program unchanged. An optional
pair observer requests an exact logical inverse from explicitly owned encoder,
ArcTan and phase pairs. Shared adjoint_component_calls reverses typed calls and
owned views/lifetimes; measurement primitives are never reversed. Without a pair
request, no extra inverse gate construction is performed.

The consumer reuses qubitization_walk_component_pair for phase then prepared
reflection and its reversed exact call composition. Reflection retains the
source coefficient-1 and the existing zero-port scalar owner. No second generic
two-stage builder, Core opcode, public flat binding contract or code serializer
is introduced. The source non-Bloq code argument refuses original serialization
at every checked depth; the owned descriptor supplies explicit live context.

## Verified execution scope

The live context composes the original Bernoulli and Grover dependencies at
precisions 0, 1 and 3 in both directions. Root child order and register operands
match the unchanged source. Full complex operators include a spectator and
reversed physical placement. Every small measurement history is checked for
every basis column at precisions 0 and 1. At precision 3, every history is
checked on a coherent complex input, with two histories per basis column.
All private work returns to zero. SDK-blocked binding, append, IR JSON and
trajectory execution exercise the same emitted instructions.

A measured encoder with independently owned private work also composes with a
synthesizer carrying cleanable junk. Every applicable small measurement history
matches the original dependency tensors and literal reflection action. Explicit
caller junk preserves the full joint source action for an entangled-junk
counterexample, including junk changing from zero to one. All private wires are
still zero. This expanded context does not establish the smaller original ABI.

The observer-free forward path is unchanged: eight complete Programs and their
bound IR, output/event metadata and classical outputs match published
`177ed24ab`. Both phase polarities through bit 31 and both directions match the
source-selected wire streams, angles, feedback and exact matrix payloads.
`any` denotes an explicit quantum matrix primitive, not classical OR.

## Default precision: source stream parity and retained accuracy failure

The actual default-32 instruction stream is executed with the existing test-only
sparse verifier on one logical bit and 36 reserved clean bits, using three
measurement seeds, both basis columns and both directions. This avoids a dense
37-qubit state; it is not a production sparse backend claim. Its independent
reference executes the original Cirq decomposition of each literal phase gate
directly with Cirq matrices, without FlagQuantum conversion. Work returns zero
and the forward/inverse literal phase products are conjugates.

That selected SDK decomposition is not the exact gate operator. For example,
`(cirq.Z ** (1 / 2**17)).controlled(control_values=[0])` has a maximum entrywise
error of `1.1984224905284855e-5` after original Cirq 1.7 decomposition. At the
next bit the decomposition reduces to X, CX, CX, X and loses the phase. The
default-32 Bernoulli example therefore differs from its exact ideal operator by
approximately `1.1572070829341236e-5`. The default test retains this counterexample
and separately checks literal stream parity at `1e-11`, without global-phase
alignment or a relaxed accuracy threshold. Actual Cirq 1.6 and 1.7 profiles are
tested separately. Neither the SDK nor frozen Qualtran source is patched.

This is not an approved upstream-defect waiver. Default-32 mathematical accuracy
and the parent acceptance remain open. Source gate fidelity is not a claim that
the source's numerical decomposition is exact.

## Resources and remaining boundaries

`g2-mean-component-resources-20261126.json` records ten separate actual/model
rows, both directions of Bernoulli/Grover precisions 1/3 and private-clean-junk
precision 1. Actual instructions are respectively 47/99/76/128/158 per direction,
measurements 1/3/2/4/6 and private lease peaks 4/6/4/6/5. Every lease returns.
These include quantum matrix primitives and are not synthesized T counts.

All ten seven-category declaration models and their peak estimates agree with
the original source. Junk width is supplied to the existing reflection cost
owner from the actual PREPARE signature; this does not broaden the older public
selection-only estimator. For Grover precision 3 the model is
`T=144*_c0+1, And=7, Clifford=23, rotation=4, measurement=7, peak=14`, whereas the
executed recipe has four measurements. Opaque ArcTan source costs and explicit
bounded QROM costs are distinct. No equality of unlike primitive/model counts
is claimed.

Original code-object serialization refuses at depths 0/1/2, so no exact shallow
factory is registered. Source encoding controls are absent from the mean root
signature and cannot be silently supplied. Nonclean private junk is not reset
or discarded; callers must either demonstrate clean release or declare it as
logical signal junk. At that earlier batch, wide lookup, generic multiple controls, controlled/powered
models and default-precision accuracy remained outside acceptance. The multiple-control
closure below supersedes only the corresponding live-context execution gap.
Q-0566, Q-0866 and Q-0876 remain open.

## Batch validation

The concentrated batch passes 313 cases without skips in each actual
Cirq 1.7.0/Qiskit 2.5.0 and Cirq 1.6.1/Qiskit 2.0.0 profile. The launcher asserts
Qualtran 0.7.0 and actual Kickmix 0.1 Simulator availability. Local integration
passes 145 cases. Five Python files pass strict mypy, Black and Ruff; both team
scopes, repository language/local links, architecture, internal IR and protected
API checks pass. Five frozen source/test/direct-dependency blobs are verified.
Existing original 12-case mean and independent model evidence is reused after
the locked source hashes are checked.

The Cirq-dependent live constructor belongs to
`ecosystem/cirq/_mean_estimation_library.py`. Compiler only reverses typed
component calls from explicitly supplied inverse owners; numerical execution
and SDK selection remain outside Compiler. No shallow registration is added,
so the constructor census stays 204/303. This batch does not add an accepted
parent, waive another upstream defect, modify pending proposals or merge the PR.


## Multiple-control dependency closure (2026-11-27)

The existing live typed context now also accepts an explicit positive-controlled
encoder pair and a ControlSpecification. The default scalar positive control uses
the source two-stage positive mean root. Other admitted quantum control layouts
reuse the existing ControlledViaAnd owner: CtrlSpecAnd, positive mean, exact
predicate erasure. Controls and selection words are THRU, with MSB-first words
and row-major shaped bits. The predicate and every child own their private work;
no measurement instruction is reversed to construct an inverse.

The positive phase now controls the encoder, bounded ArcTan, each signed phase,
ArcTan erasure and encoder erasure individually, following the actual source
positive child. It does not compute an unconditionally encoded word and only
control its final phases. Source merged S/T/rotation controls and raw controlled
CZPow children are retained. Shared typed views only split and repack MSB-first
bits; normalized recursive comparison checks allocation, stage operands, output
widths and frees after these zero-gate views. Source phase and reflection root
calls also compare directly. Private encoder targets named ctrl are supported
without colliding with the outer control.

The controlled ArcTan is an explicit bounded native QROM replacement for the
opaque source arithmetic leaf. Original Bloq tensor contraction still raises
NotImplementedError, "ArcTan does not support tensor simulation." The independent
full-operator reference is the successful original Cirq controlled-unitary
protocol; these two upstream protocols are not conflated. Dirty controlled-ArcTan
columns are checked separately at widths 0/1/2 and precisions 0/1/3.

At precisions 1 and 3, two positive bits, mixed 0/1 bits, a QUInt(2) word plus
QBit, scalar controls of both polarities and the private-target-name collision
match the original full complex operators in both directions. Every small raw
measurement history is checked on a coherent complex input; each basis column
is checked at the all-zero and all-one histories. The exact erasure certificate
also covers all histories and private work. SDK-blocked append and native
trajectory execution preserve the same operator. Positive and negative integer
squares reuse the same exact forward/inverse Programs, with full complex column
and cleanup checks. Ordinary lookup and mean pairs remain entirely identical
to published 13ebd7dcb at the twelve recorded fixtures.

### Measured execution versus source model

Counts below are actual native instructions, including measurement, reset and
scalar phase instructions. Forward and inverse have equal totals in these
fixtures. They are not a decomposition count for the opaque source ArcTan.

| Precision | Control layout | Physical wires | Peak private bits | Instructions | Measurements |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Two positive bits | 8 | 5 | 106 | 4 |
| 1 | Mixed 0/1 bits | 8 | 5 | 110 | 4 |
| 1 | Two-bit word plus bit | 10 | 6 | 131 | 5 |
| 1 | Scalar positive / negative | 6 | 4 | 89 / 91 | 3 |
| 3 | Two positive bits | 11 | 8 | 197 | 7 |
| 3 | Mixed 0/1 bits | 11 | 8 | 201 | 7 |
| 3 | Two-bit word plus bit | 13 | 9 | 222 | 8 |
| 3 | Scalar positive / negative | 9 | 7 | 180 / 182 | 6 |

The unchanged source QECGatesCost query for these controlled roots fails with
DecomposeTypeError: "Could not build call graph for C[And]: And is atomic."
Direct whole-gate Cirq capture is a different lowering protocol: the first
positive S/T signed stages use 17 instructions without a measured predicate,
whereas retaining the actual source ControlledViaAnd calls emits 22 native
instructions; the later positive rotation emits 39. Direct capture of the
negative signed stages refuses the atomic C[And-adjoint] body. Thus no equality
of whole primitive streams across those protocols is claimed. Comparable typed
calls and operands are checked, while the independent exact Cirq operators
validate their complete complex action.

There is no successful source estimate to relabel as a primitive count. Separate
uncontrolled models and all prior model evidence remain valid in their original
scope. The external resource receipt records the complete measured primitive
histograms and the original exceptions separately.

The live multiple-control execution gap is closed for the concrete fixtures;
Q-0566 remains open for controlled/powered source models, wide/default-precision
accuracy and the original smaller ABI when nonclean signal junk is needed.
The new controlled fixtures do not claim arbitrary caller-junk coverage. Original
code-object serialization still refuses; there is no fabricated shallow factory.
The frozen constructor census stays 257/303 at this batch, and pending Proposal
124 is untouched. Previous full original-source evidence is reused; the source
and direct-dependency blob identities are checked again before publication.


### Multiple-control batch verification

The concentrated six-file regression passes 233 cases per actual Cirq 1.7.0 /
Qiskit 2.5.0 and Cirq 1.6.1 / Qiskit 2.0.0 profile, without skips. Launchers assert
Qualtran 0.7.0 and a real Kickmix 0.1 Simulator. Local integration passes 145
cases. Five Python files pass strict typing, Black and Ruff. Compiler/Ecosystem
scope, English/local links, architecture, multi-level IR and protected API
snapshots pass. Seven frozen source/test/direct-dependency blobs match the
locked manifest. Existing original twelve-test and ordinary default-32 evidence
is reused within its documented scope; concentrated native regressions also
retain the default-32 counterexample.
