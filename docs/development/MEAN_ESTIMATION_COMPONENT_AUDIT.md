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
logical signal junk. Wide lookup, generic multiple controls, controlled/powered
models and default-precision accuracy remain outside this batch's acceptance.
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
