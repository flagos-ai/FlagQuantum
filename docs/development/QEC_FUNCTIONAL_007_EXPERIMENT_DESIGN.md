# QEC-FUNCTIONAL-007: Fixed-shot experiment design review

Status: historical design review delivered in #912. The owner subsequently
approved the [complete 007 contract](QEC_FUNCTIONAL_007_EXPERIMENT_CONTRACT.md)
for implementation, verification and delivery on 2026-10-26. The implementation
candidate is being verified; final delivery gates remain pending. The proposed
shapes and design-only statements below preserve the earlier review context;
the complete approved contract is the current authority.

Decision B is delivered in [#904](https://github.com/flagos-ai/FlagQuantum/pull/904).
Its [complete-result contract](QEC_FUNCTIONAL_005_BATCH_CONTRACT.md) remains a separate
authority. This document refines [Decision C of Proposal
087](API_CHANGE_PROPOSAL_087_QEC_SCALABLE_EXPERIMENTS.md#decision-c-fixed-shot-memory-
experiment). It adds no production module, callable, result type, schema, numerical
helper or experiment run. Permission to merge this design document is not implementation
approval.

## Concrete contract follow-up

The owner requested proceeding with the three recommended design choices. The
[complete contract draft](QEC_FUNCTIONAL_007_EXPERIMENT_CONTRACT.md) now states
the typed signature, ordered result, strict source path, support and exception
tables for final review. This records contract preparation, not implementation
approval. The following sections preserve the original #912 design context.

## Proposed initial scope and signature

One synchronous fixed-shot memory experiment, actual original-circuit sampling, CPU bit
records, no hardware, adaptive stopping, threshold fit, asynchronous execution, plugin
certification or artifact service. Reuse MemoryCircuit, supported
PhenomenologicalNoise/NoiseModel, compiled MemorySampler, the one DEM owner and existing
registry.

Retain the proposed execution.experiments.run_memory_experiment import and require
explicit decoder selection. Proposed refinement for review: add required keyword-only
batch_size to the original memory/noise/decoder/shots/seed signature. An explicit
positive non-boolean integer avoids an unreviewed batch-size default and captures the
seed call schedule. It is a proposed API change and is not approved by permission to
refine design.

Proposed call shape for owner review, **not a runnable API**:

```python
run_memory_experiment(
    memory, *, noise, decoder, shots, batch_size, seed=None
)
```

This intentionally leaves concrete noise annotations to the existing accepted memory-
sampler boundary rather than inventing another noise type. The only canonical location
under review is `flagquantum.qec.execution.experiments.run_memory_experiment`; no root
or parent export is proposed. The required `batch_size` is a refinement of the earlier
signature, so it requires explicit final-contract approval.

The initial decoder argument is an existing registered name, with its existing
constructor defaults only; no unrestricted option dictionary or new options record.
Start with the five Decision-B concrete independent-shot classes. Resolve through
get_decoder, preserve registry replacement, then refuse an actual returned
plugin/subclass outside the certified set. Do not bypass a replacement to force a built-
in implementation. Nondefault algorithm configuration is explicitly outside this first
signature and requires a separate reviewed extension; users retain the current lower-
level API for it. Stateful streams and AI predecoders are not implicitly admitted.

## Configuration and numerical authority

Prepare one immutable supported memory/noise snapshot using existing owners. Pair the
original positioned source sampler with its strict actual-law DEM. Reuse the approved
Simulation bridge for exact DEM analysis if it is the authoritative path; do not
silently substitute nominal-decimal laws, split exclusive groups, enable
approximate_disjoint_errors or add a second mechanism vocabulary. A source the sampler
accepts but the selected decoder/strict DEM cannot represent is a capability refusal,
not a successful approximate experiment. Matching-only transforms never replace source
sampling. Missing SDKs propagate their established errors with no fallback.

Require at least one observable for an experiment result; an empty logical vector must
not become a certified zero-failure experiment. Validate
shots/batch_size/seed/decoder/type-domain and observable shape before consuming RNG
wherever knowable. Define resource support solely through existing source/model bounds;
no new capacity claim.

## Streaming evaluation and failures

Compile once and make exact sample calls of batch_size, followed by the positive tail
only. One seed initializes the existing continuous stream; identical-call-schedule
reproducibility remains version/architecture scoped. Different batch sizes need not
yield identical samples. Never reset seed per batch.

For each batch use the approved complete-record sparse batch route, preserving
algorithm-specific errors and tie conventions. Convert sparse result observables to a
checked [B,L] logical prediction only for aggregation; do not invent correction fields.
A separate use of the existing logical-only PyMatching Tensor method would need explicit
reviewed equivalence for this experiment scope before being selected as an optimization.

A shot fails when any predicted logical differs from its sampled flip. Keep per-
observable counts separately. Count the tail exactly; total completed shots must equal
requested shots. On sample/decode/shape failure, close in a finally/context-manager
boundary and propagate the original error. Return no partial successful result; no
retry, row dropping or denominator adjustment.

## Proposed result and persistence boundary

Retain the seven originally proposed immutable fields: shots, logical_failures,
logical_error_rate, observable_failures (tuple), confidence_interval (two-float tuple),
seed and sampling_source='memory_circuit'. shots is completed and equals the requested
fixed total on success. Rate is failures/shots for the complete experiment, never per-
round normalization.

Freeze a two-sided 95% Wilson score interval without continuity correction. Preserve
exact endpoint inclusion at zero/all failures and a positive upper endpoint at zero
failures. No interval is defined for zero completed shots, which is not an accepted
successful call. Do not reuse the AI assessment Hoeffding interval: it is a different
numerical contract.

Keep persistence outside this initial result contract: no supported save/load,
pickle/JSON portability promise, checkpoint schema or artifact registry. A runnable
example must separately capture the explicit memory/code/layout/rounds/logical support,
noise settings, decoder name/default configuration, shots/batch_size/seed and installed
versions alongside the result. This example evidence dictionary is not a new stable
artifact schema. Before approval, review whether that explicit caller-owned evidence is
sufficient or whether a minimal configuration field must be added; do not silently add
provenance machinery.

## Reuse and independent acceptance

The existing example's Wilson/count code is a candidate authoritative implementation to
relocate privately and reuse from both example and library after approval. The library
must not import examples; retain the independent score-inversion oracle tests and avoid
maintaining two formulas. No numerical helper or production module is created by this
design note.

Acceptance: independently stated multi-observable counts (two wrong bits still one
failed shot); 35 shots in 16+16+3 calls and one compile; controlled zero/all/mixed
counts and analytical score inversion; invalid late inputs before RNG where possible;
sample/decode failure cleanup with no partial result; supported
Repetition/Surface/native source behavior and strict original-law pairing; decoder-
specific complete-record/refusal semantics from 005; actual optional
install/source/wheel/sdist profiles, runnable public Python and exact-head complete
CI/GPU. No new statistical experiment or sampling budget is authorized by these test
descriptions.

## Concrete acceptance matrix for a later approved implementation

| Boundary | Independent expected behavior |
| --- | --- |
| Whole-shot estimand | Two incorrect observable bits in one shot count as one failed shot; per-observable counts retain both. |
| Fixed schedule | 35 shots at batch size 16 cause one compile and exactly 16, 16, 3 sample requests. No zero-length request or seed reset. |
| Input rejection | Reject boolean/noninteger/nonpositive shots and batch sizes, invalid seed/decoder/domain and a zero-observable memory before RNG wherever knowable. Preserve existing owning exception categories; the final contract must enumerate them before activation. |
| Algorithm semantics | Reuse all five certified concrete decoders and their complete results, duplicate policies and unsupported-syndrome refusals from 005. Do not silently replace registry plugins. |
| Failure cleanup | Sample, decode and shape failures close the sampler and propagate the original error; no partial result, retries, denominator change or dropped row. |
| Numerical interval | Independent Wilson score inversion for zero, all and mixed failures; fixed 95% confidence, no continuity correction and no adaptive stopping. |
| Source law | Match the actual positioned source with its strict actual-law DEM. Keep exclusive mechanisms intact and reject unsupported exact translations. |
| Installation/user path | Core/refusal and actual optional-SDK profiles; source, wheel and sdist; runnable public Python and exact-head full CI/GPU followed by normal merge/main acceptance. |

These are future conformance requirements, not evidence that the proposed API exists or
that any experiment has run. They authorize no new hardware, statistical sampling study
or performance claim.

## Final contract review before implementation

The owner has requested proceeding with the recommended required explicit
`batch_size`, default-only registered-name selection, and caller-owned evidence
without an initial persistence schema. These choices now guide the
[complete contract draft](QEC_FUNCTIONAL_007_EXPERIMENT_CONTRACT.md), including
its typed signature, ordered result, strict source path and exception/support
tables. The owner must still approve or amend that complete contract before
implementation. Design delivery and this follow-up do not activate the API.
