# API Change Proposal 087: Scalable QEC experiments and batch decoding

## Status and decision scope

**Decision A was approved on 2026-10-24. Decision B's complete
[FUNCTIONAL-005 contract](QEC_FUNCTIONAL_005_BATCH_CONTRACT.md) was explicitly
approved by the API owner on 2026-10-26 (Asia/Shanghai). Decision C's complete
[007 contract](QEC_FUNCTIONAL_007_EXPERIMENT_CONTRACT.md) was explicitly approved
for implementation, verification and delivery on the same date.**

The earlier roadmap/proposal approvals did not activate B/C. The later explicit
Decision B approval now authorizes its complete independent-shot contract and
verification, including five concrete decoder types and full existing result
records. Decision C has its separate complete-contract approval; neither decision authorizes new Simulation boundaries or changes
to frozen fixtures, root exports or decoder registration policy. QEC maintainers
own the implementation; exact-head delivery gates remain separate.

This is QEC-FUNCTIONAL-002 from the
[functional roadmap PR #774](https://github.com/flagos-ai/FlagQuantum/pull/774). Review the
three decisions independently: internal Clifford fault propagation, additive
batch decoding, and an additive fixed-shot experiment entry point. A later
implementation PR must cite the decisions actually approved.

## Problem and current authority

The existing circuit-level DEM route derives fault signatures by forced
execution. It is a valuable independent small-system reference, but its full
statevector capacity limits larger patches. A Stim memory sampler already
exists; it does not remove this construction bottleneck.

The existing decoder registry promises `decode(detection_events)` and a result
with `observables`. It intentionally does not require every decoder to expose a
pair graph, a convergence flag or a batch method. Adding a required method to
that protocol would break third-party plugins. The existing streaming decoder
also holds state across rounds, so arbitrary repeated calls must not be
advertised as independent shots.

Existing repetition experiments, noise sweeps, DEM sampling and logical-rate
tests provide components and measured results. They do not provide one general
fixed-shot memory workflow with an explicit statistical estimand and interval.

## Decision A: Internal Clifford fault propagation

Keep the signatures, defaults and result ordering of existing public DEM
constructors unchanged. Keep fault enumeration and DEM assembly in `qec/dem/`
and layout composition in the existing `qec/execution/signatures.py` boundary;
the numerical Clifford primitive belongs to Simulation. Do not add a second
public DEM type or a backend selector to the constructor. The existing execution implementation remains an independent
test reference rather than an undocumented runtime fallback.

Reuse audit: [Simulation's stabilizer engine](../../flagquantum/simulation/stabilizer/README.md)
already owns Clifford images, Pauli conjugation, positioned Pauli-mixture noise
and optional Stim translation. Its `pauli_readout` answers a whole-circuit
observable against the zero state; it is not a reset/measurement-record fault
signature API. Reuse its algebra through a Simulation-owned private primitive
and the QEC execution bridge, with focused owner tests. Do not import an engine's
private helper directly into DEM assembly, duplicate its Clifford image tables,
build another tableau simulator or add another mandatory engine dependency.
Any necessary new public Simulation boundary needs separate API approval.

For the already supported Clifford memory grammar, propagate Pauli faults and
measurement flips through the actual lowered instruction order. Include resets,
X/Y/Z basis rotations, both controlled-gate directions, measurement record
numbering, terminal readout and declared detector/observable parities. Preserve
zero signatures, fault multiplicity, probability merging and error-id grouping
under the current rules. Mixed stabilizers require explicit equivalence tests.

Before replacing the production route, compare every fault signature and final
DEM on bounded reference circuits against execution, including rotated, Steane
and mixed-check cases. Use independently specified Clifford identities and
optional external checks as additional evidence. No unsupported instruction,
non-Pauli channel or conditional operation may silently receive a guessed
signature. Preserve existing supported behavior and documented refusal categories;
expanding the accepted grammar needs separate evidence and review.

Measure operation counts and peak memory as circuit size grows. Passing a
larger patch proves construction capacity only, not decoder quality, logical
suppression, GPU acceleration or a threshold.

QEC-FUNCTIONAL-004 separately extends noise support. Its implementation must
retain mutually exclusive Pauli choices from one location, including correlated
two-qubit products, instead of replacing them with independent Bernoulli faults.
The exact accepted channel grammar and any additions to shared Noise APIs must
be reviewed before that implementation. This proposal does not authorize a
non-Pauli-to-Pauli approximation or a shared Noise contract change.

## Decision B: Additive batch operation

Approved canonical entry point:

```python
from flagquantum.qec.decoders.batch import decode_batch

decode_batch(decoder, detection_events)
```

`detection_events` is a finite iterable of shots; each shot is the sparse
iterable of detector indices already accepted by that decoder's `decode`.
Do not overload a dense bit tensor and sparse indices by inspecting their
length. Dense/native representations stay within the optional adapter.

The operation returns an ordered tuple of existing per-shot result records.
It adds no root `qec` export, common result wrapper, new registry or required
method to `DetectorErrorModelDecoder`. Preserve algorithm-specific result types,
convergence reporting, mechanism numbering and exceptions. An empty batch returns
an empty tuple. Consume each input iterable once and validate before dispatch
where possible; never treat an invalid shot as successful zero correction.

Initial support covers explicitly tested in-tree independent-shot decoders and
the optional PyMatching adapter. Stateful streaming-only decoders are refused
with `CapabilityError`. Do not assume an arbitrary registered plugin is
stateless: it keeps working through `decode`, while batch support requires a
separately reviewed opt-in boundary. Plugin registration and replacement rules
remain unchanged.

The reference batch path uses existing single-shot decoding and makes no speed
claim. The PyMatching optimization must match that adapter's single-shot
observable predictions under its documented tie/precision conventions. If
native batch output does not reconstruct an existing result field faithfully,
do not invent that field or discard it: retain the reference path for that
case and record the optimization's actual scope. No automatic decoder
substitution or missing-extra fallback is authorized.

Validation covers generator inputs, empty batches, ordering, invalid indices,
inconsistent model sizes, unreachable syndromes, configured defect limits,
hyperedges, non-convergence and optional-dependency errors. A plugin implementing
only today's protocol must still register and decode unchanged.

## Decision C: Fixed-shot memory experiment

The owner requested proceeding with the three recommended design choices to
prepare a [complete contract draft](QEC_FUNCTIONAL_007_EXPERIMENT_CONTRACT.md).
The final signature, source path, result, exceptions and support matrix remain
subject to implementation approval. No production API is activated.

A [concrete design refinement](QEC_FUNCTIONAL_007_EXPERIMENT_DESIGN.md) is now
available for review. It proposes a required explicit batch size, default-only
initial decoder configuration and caller-owned evidence without a persistence
schema. These refinements and the complete exception/support contract require
owner approval before implementation. The earlier signature below remains
historical proposal context, not an implemented or approved API.

Proposed canonical entry point, pending approval:

```python
from flagquantum.qec.execution.experiments import run_memory_experiment

run_memory_experiment(memory, *, noise, decoder, shots, seed=None)
```

`memory` is an existing `MemoryCircuit`; `noise` uses its supported noise records.
`decoder` is an explicitly selected DEM decoder name. Algorithm-specific options
must be designed and reviewed before implementation; do not hide unreviewed
keywords in an unrestricted dictionary or choose a decoder automatically.
`shots` is a positive non-boolean integer and `seed` follows existing seed
validation. The initial experiment uses actual memory-circuit sampling; a
declared-model experiment must have an explicit separate source label and may
not masquerade as circuit execution.

Lower, construct the matching DEM, sample the memory, decode and compare each
predicted observable with its sampled observable flip. A shot fails when any
logical observable disagrees; retain per-observable failure counts as well.
For multiple logicals, do not average per-observable rates and call that the
shot-level logical failure probability. Refused or failed decodes terminate
with the decoder's error and are never counted as successful shots or silently
removed from the denominator.

Proposed immutable `MemoryExperimentResult` fields are `shots`,
`logical_failures`, `logical_error_rate`, `observable_failures`,
`confidence_interval`, `seed`, and `sampling_source`. Use integer counts,
`logical_error_rate = logical_failures / shots`, an ordered tuple of counts and
a two-float interval. `sampling_source` is initially the literal
`"memory_circuit"`. The interval is a fixed-shot 95% Wilson score interval for
the shot-level failure event; zero observed failures must yield a positive upper
bound. There is no exact-zero physical-error-probability claim.

Keep configuration capture in an explicit experiment evidence artifact alongside
the numerical result: code/layout, noise, rounds, decoder/options, software
versions, seed, shots and sampling source. Do not require ordinary callers to
construct provenance or scheduler objects. This proposal freezes neither a
portable artifact schema nor pickle compatibility for the new record; define
those explicitly in the implementation review before exposing persistence.

Initial acceptance is fixed-shot only. Adaptive stopping needs an independently
reviewed sequentially valid interval or confidence sequence; an ordinary Wilson
interval after outcome-dependent stopping is not sufficient. Threshold fitting
and its public surface also require a follow-on decision before 008. Neither is
implicitly approved by a roadmap row.

## Compatibility, ownership and dependencies

- Keep the existing 108 ordered root exports, public signatures/defaults,
  registered class identities, surviving pickle paths and seeded results.
- Do not reintroduce any of the 23 retired imports or the duplicate rotated
  class. Do not modify any original restructuring fixture or historical evidence.
- New public names live in their responsibility packages after approval. Do
  not update existing protected snapshots merely to bless accidental changes.
- DEM construction belongs to `dem`; batch semantics to `decoders`; optional
  native translation to `adapters`; experiment composition to `execution`.
  Reuse Runtime/Simulation and existing Noise ownership boundaries.
- No new mandatory package dependency is proposed. Stim and PyMatching remain
  optional, lazy and explicitly selected; missing extras raise the established
  actionable error. No hardware, network or paid task is part of this workflow.
- No asynchronous, accelerator, distributed or realtime claim follows from
  batch decoding. No change to or merge of #731 is included.

## Executable current-user baseline

Decision B is implemented through #904; its contract links its runnable
five-decoder example. Decision C remains a proposal and is not runnable.
The following existing example is runnable and remains the compatibility gate:

```bash
python examples/qec/installed_imports.py
```

Expected output:

```text
Public exports: 108
Registered class: True
Code pickle path: flagquantum.qec.codes
Correction: (0,)
Logical error rate: 0.0
```

Use an installed distribution, or `PYTHONPATH=.` for a source checkout. Future
implementation PRs must add runnable examples for their actual new surface;
this proposal does not present pseudocode as an implemented capability.

## Review and implementation gates

1. Approve or amend decisions A, B and C and the deliberately deferred scopes.
2. Add conformance tests in the owning implementation PR before activation;
   retain old numerical and plugin tests alongside new behavior tests.
3. Deliver 003–006 separately, then the fixed-shot part of 007. Review adaptive
   stopping and threshold additions before the rest of 007–008.
4. Require independent numerical evidence, source/wheel installation checks,
   exact runnable examples and complete required CI at each reviewed head.
   Do not weaken checks, tolerances or refusal semantics.
5. Record actual performance before publishing speed or memory claims. Preserve
   the reference route for correctness tests without treating it as scalable.

Merging this document records a proposal; it does not itself constitute the API
owner's approval, activate implementation, or authorize a release.

## C/007 implementation approval and candidate

Following #924 contract delivery and successful exact-merge Main smoke, the owner
explicitly approved the complete C/007 contract and its implementation,
verification and delivery. The candidate now composes the existing strict Stim
bridge, complete-record 005 operation and shared private Wilson formula in
`flagquantum.qec.execution.experiments`. Earlier design-only/deferred statements
in this proposal describe the publication history, not the current approval.
No root/parent export, registry change, default option, numerical tolerance,
Simulation boundary, dependency or persistence schema is added. Adaptive stopping
and threshold fitting still require separate decisions. Final delivery gates
remain pending; see the [central ledger](../roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md).
