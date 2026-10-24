# QEC Stim and PyMatching delivery plan

Status: active delivery plan, 2026-10-24. Core owns QEC composition and adapters;
Simulation owns numerical execution; Runtime owns execution policy and lifecycle.
This document orders work, not approvals of future public APIs or merges.

## Current baseline and user workflow

The structural/retirement baseline is main commit
`d286e4172cb9af4d1553a79199ee71f919af8c7a` (PR #766).
[PR #772](https://github.com/flagos-ai/FlagQuantum/pull/772) introduces the approved
Tensor-only PyMatching batch method. It is a draft, not an accepted prerequisite.
Implementation PRs must start from the actual merged baseline, not assume it.

The workflow #772 supports is:

```bash
pip install -e '.[stim,pymatching]'
PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py
```

It circuit-samples 128 repetition-memory shots in bounded chunks and predicts
logical flips in batches: 8 detector columns, 1 observable, and an observed
logical failure count. The random count is version/architecture scoped.
Stim compilation is still repeated per chunk. No performance claim follows.

Before #772 merges, this command is a branch workflow, not a main capability.
The corresponding main workflow is
`PYTHONPATH=. python examples/qec/memory_layout_and_sampling.py` with the Stim
extra; it exercises the existing measurement layouts, sampling and feedback.

## Immediate action: finish #772

The quality job at head `8d0a0da` failed Black formatting in
`tests/api_contract/test_public_docstring_examples.py` and
`tests/qec/test_restructure_compatibility.py`. This is distinct from the local
default-tier QFT/native-region and Gloo failures reproduced on frozen main.
Apply Black, preserve assertions/fixtures, and rerun the actual quality tools.
Do not classify unrelated remote failures without inspecting their logs.

Update draft validation and publication status; merge only after required gates
and maintainer review. No automatic merge is authorized by this plan.

## Ordered PRs

Letters are planning identifiers, not reserved GitHub numbers.

| Work | Deliverable and dependency | Acceptance and boundary |
| --- | --- | --- |
| A: reusable sampling design | Review existing Runtime/session ownership and propose the narrow lifecycle; #772 interface must be stable. See the [sampling proposal](../development/QEC_REUSABLE_SAMPLING_PROPOSAL.md). | Concrete repeated-batch workflow, RNG and release semantics; no SDK objects in public records; no speculative session platform. New public API requires explicit review. |
| B: compiled sampler reuse | After A approval and prerequisite integration, prepare one private Simulation sampler and compose QEC parity extraction in `execution/sampling.py`. | One lowering/compilation per handle; bounded batch storage; owner isolation; explicit close; immutable configuration snapshot. Existing one-shot APIs retain behavior. |
| C: fixed-shot evaluation | After #772, accumulate requested/completed/failed shots, vector logical failures, 95% intervals and experiment metadata. Reuse existing result concepts first. | Tail batches counted, zero-error upper bound, fixed denominator/stopping rule, all observables checked. Atomic batch failure stays an error; no silently discarded rows. Any new public result requires its own proposal. |
| D: Surface batch evidence | Extend C's workflow using existing rotated code, memory builder and measurement maps; B supplies scalable execution. | X/Z supported readouts; d=3/5/7 correctness and representative d=9/13/17 workloads; initial/final rounds, boundaries and forced faults; same-sample decoder comparisons. No second Surface class. |
| E: annotated Stim experiment bridge | After A/D boundaries stabilize, add QEC annotations to native export through the Simulation seam and accept external Stim experiments at an adapter boundary. | Measurement/detector/observable parity and coordinates preserved; raw DEM separate from decoder conversion; no promise of arbitrary Stim-to-native reverse compilation. |
| F: circuit-level Pauli noise contract | After D/E, define supported locations, single/two-qubit mixtures, preparation/readout faults and explicitly positioned idle faults. | Preserve exclusive branches, original distributions and mechanism identity; state unsupported cases; review any public/schema deltas before G/H. |
| G: consistent noise placement | Implement F in `execution/signatures.py` and `execution/sampling.py`; reuse Simulation Pauli-mixture support. | Forced-fault propagation, exact placement, mutually exclusive depolarizing branches; Y is not independent X/Z. Numerical kernels remain in Simulation. |
| H: matching DEM construction | Implement F using G's location vocabulary in `dem/construction.py`; apply decoder transforms only at the declared boundary. | Original circuit/DEM marginal, selected joint and logical distributions agree; graphlike decomposition never replaces the sampling law; unsupported correlations fail explicitly. |
| I: phase-one acceptance | After B/C/D/E/G/H, certify the two experiment entrances, installability, bounded memory and replacement boundary. | Four actual optional-dependency installation profiles, wheel/sdist checks outside checkout, fixed workload evidence and required gates. Fake SDK absence tests are not four installations. |

Cross-domain implementation changes follow an approved shared contract and land
by owner. G/H must not invent separate fault vocabularies. Existing flat imports
remain retired; frozen restructuring evidence is not regenerated to hide drift.

## Dependency order and exit gates

```text
#772 closes and merges
  +-- A lifecycle proposal -> B reuse -------------------------------+
  +-- C statistics -> D Surface -> E bridge -------------------------+-> I
                               +-> F noise -> G sampling -> H DEM --+
```

C's test scenarios can be prepared while A is reviewed; they do not depend on an
unapproved sampling signature. The graph is a dependency plan, not a request to
spawn parallel agents. Avoid long-lived stacks of unmerged implementation PRs.

Exit gates:

1. B: repeated sampling amortizes preparation with explicit ownership/RNG rules.
2. C/D: reproducible Repetition/Surface logical-failure comparisons, including
   statistical uncertainty, without threshold claims.
3. E/H: native and external experiments enter one evaluator; supported circuit
   noise and DEM remain consistent.
4. I: stop broadening phase one and start data/AI or bottleneck-driven kernels.

## Prepared acceptance cases for work C

These scenarios are ready for the statistics change without depending on A's
unapproved sampler API. They specify behavior, not a new result schema:

- Fixed inputs: 100 completed shots with zero failures must report a positive
  upper confidence bound; 100 failures must report a lower bound below one.
- For a vector of observables, any differing bit counts as one failed shot;
  two differing bits in one shot do not count as two failed shots.
- Requested 35 shots with chunks of 16 must count 16 + 16 + 3. Invalid or failed
  batches must not silently decrease the denominator.
- Use fixed-shot Wilson 95% intervals as the initial candidate. Freeze the
  formula and validate against an independent analytical reference; do not use
  the same helper to compute expected values. Zero completed shots have no
  estimated rate or interval and must be represented explicitly, not as zero.
- Keep logical failure distinct from decoder execution failure. Under #772's
  atomic semantics a failed batch terminates evaluation with its failure
  context; row-level retry/discard is a separate policy change.
- Record seed schedule, batching, code, rounds, noise and actual library
  versions. Do not derive a per-round failure rate by dividing by rounds.

Any new public evaluation entry or result requires a concrete consumer/API
review. The first implementation should reuse the existing example and
statistics utilities where their semantics match.

## Later internal implementations

- GPU DEM sampling consumes the owned mechanism semantics and bit batches.
  Training labels require pre-merge mechanism identity and geometry; logical
  outcomes alone do not reveal actual faults.
- An in-house Clifford/Pauli-frame kernel replaces the Simulation seam. It does
  not require QEC callers to construct vendor objects.
- An in-house matcher uses the existing registry and logical-prediction meaning;
  it must pass the same conformance evidence with an independent reference.
- AI pre-decoding follows three separate changes: labeled data, model training,
  and correction/global-decoder evaluation. Combined correction must preserve
  detector and logical effects, not only reduce event density.

Do not vendor whole libraries or remove independent optional references as an
end in itself. Start replacement work only with a measured bottleneck, hardware
coverage requirement or concrete maintenance benefit.

## Delivery rules and estimates

Each PR contains a runnable user example, expected behavior, failure cases and
appropriate evidence. It owns one coherent step; no second decoder registry,
QEC IR, service database, queue platform or speculative GPU scheduler is added.
Use Black and Ruff as configured by CI; Ruff formatting is not the formatting
gate. Preserve CPU and single-device usability and dependency isolation.

A/F are design reviews; B/G/H are implementation changes after their contracts
are approved. Persistent samplers, common batch protocols and new result types
are not implicitly approved by the roadmap. Publishing PRs and changing release
status follow the user's authorization; merging is a separate decision.

Planning estimate: two QEC/PyTorch contributors, roughly 4-6 weeks for A-I,
including review time. The bridge and circuit-level noise are the largest
uncertainties. Unrelated repository CI repairs and new numerical kernels are
outside that estimate.
