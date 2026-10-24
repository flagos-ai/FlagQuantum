# QEC Stim and PyMatching delivery plan

Status: active delivery plan, 2026-10-24. Core owns QEC composition and adapters;
Simulation owns numerical execution; Runtime owns execution policy and lifecycle.
This document orders work, not approvals of future public APIs or merges.

## Current baseline and user workflow

The structural/retirement baseline is main commit
`d286e4172cb9af4d1553a79199ee71f919af8c7a` (PR #766).
[PR #772](https://github.com/flagos-ai/FlagQuantum/pull/772) merged on 2026-10-24
as `0cb4086270ddc47dc431ee9d5e76984198a2d997`, adding the approved Tensor-only
PyMatching batch method. Its final head was `c1f5a65`; the main CI and Local GPU
Gate workflows both succeeded. Future implementation PRs must start from main
containing this merge and inspect subsequent changes before implementation.

The workflow #772 supports is:

```bash
pip install -e '.[stim,pymatching]'
PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py
```

It circuit-samples 128 repetition-memory shots in bounded chunks and predicts
logical flips in batches: 8 detector columns, 1 observable, and an observed
logical failure count. The random count is version/architecture scoped.
[PR #784](https://github.com/flagos-ai/FlagQuantum/pull/784) merged as
`a5fc00898af53ab85037bec10c2db7579c68823b`, adding compiled sampler reuse and a
continuous seed-53 stream across chunks. Its final head `8c942053` passed all
54 main CI jobs and both local GPU gates. No performance claim follows from the
example.

The existing `PYTHONPATH=. python examples/qec/memory_layout_and_sampling.py`
workflow remains available with the Stim extra; it exercises measurement
layouts, sampling and feedback.

## Immediate action: Surface batch evidence

[PR #788](https://github.com/flagos-ai/FlagQuantum/pull/788) merged as
`47db63d6dd243e5a2fc498440a0a660858eb06a2`. Its final head `9daf326b`
passed all 54 main CI jobs and the single/two-GPU gates. The merge commit's
Main smoke run `40066862426` succeeded; that smoke is not a complete main CI.
The next evidence slice starts at main `4be2375eff47c112ce8d8e24f8206ed5592130d8`,
including subsequent correlated-fault and dynamic-preflight work. It changes
only the existing example, tests and documentation, with no new library API.

The #772 CI repair corrected Black formatting in two tests and registered the
new baseline example in `examples/README.md`. All 54 main CI jobs passed, as
did the single-GPU and required two-GPU jobs. No test assertions or frozen
fixtures were weakened to obtain this result.

The API owner approved the [reusable sampling proposal](../development/QEC_REUSABLE_SAMPLING_PROPOSAL.md)
on 2026-10-24. Work B's two additive QEC exports and lifecycle are now on main.
The first work C slice extends the existing runnable example using those public
APIs. It adds an illustrative JSON report and private example statistics helpers,
without adding a library result class, export or serialized-artifact contract.
The existing proposed library experiment API in
[proposal 087, Decision C](../development/API_CHANGE_PROPOSAL_087_QEC_SCALABLE_EXPERIMENTS.md#decision-c-fixed-shot-memory-experiment)
remains a separate unapproved decision. RepetitionMemoryResult retains all
per-shot reference records, while NoiseSweepPoint carries repetition-specific
noise-event counts; neither expresses this bounded general logical-bit summary
without changing a protected record.

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

C's example statistics use the merged sampling API. The graph is a dependency
plan, not a request to spawn parallel agents. Avoid long-lived stacks of
unmerged implementation PRs.

Exit gates:

1. B: repeated sampling amortizes preparation with explicit ownership/RNG rules.
2. C/D: reproducible Repetition/Surface logical-failure comparisons, including
   statistical uncertainty, without threshold claims.
3. E/H: native and external experiments enter one evaluator; supported circuit
   noise and DEM remain consistent.
4. I: stop broadening phase one and start data/AI or bottleneck-driven kernels.

## Prepared acceptance cases for work C

These scenarios specify behavior, not a new library result schema:

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

## Work C: fixed-shot report semantics

The example retains its `run_baseline(shots=128, batch_size=32)` call and existing
report keys. The statistics change adds `requested_shots`; existing `shots`
means completed shots. Successful evaluations complete the entire request,
including a partial final batch. One shot fails if any logical observable
differs from the sampled flip; `observable_failures` additionally counts each
column. These counts are distinct: two wrong logical bits in a shot still mean
one failed shot.

`logical_error_rate` is `logical_failures / shots`, for one complete memory
experiment. `confidence_interval` is the two-sided fixed-shot Wilson 95% score
interval without continuity correction, following the
[NIST definition](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
The original report makes no uncertainty claim; this slice adds that explicit
interpretation. Zero observed failures have a positive upper bound. Zero
requested shots are refused, and zero completed shots have no estimated rate
or interval. An invalid or failed batch terminates with the original error and
logged batch/completed-shot context, releases the sampling context and returns
no partial success report. No row is retried, dropped or silently counted as
correct.

The report records code class/distance, rounds, the actual uniform Pauli noise
settings, initial seed, continuous-stream schedule, batch size, package/Python
versions and platform. It records `sampling="circuit"` and
`stopping_rule="fixed_shots"`. This is an example report for the existing
repetition workload, not an approved portable artifact schema or a generic
experiment service. It does not divide a failure rate by rounds, apply an
ordinary Wilson interval after adaptive stopping, or claim a threshold.

Verification uses score-test inversion by numerical bisection, an optional
independent [SciPy Wilson reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html),
controlled multi-observable batches, batch failure/cleanup and tail-count tests,
plus actual Stim/PyMatching runs with reproducible identical call sequences.
The statistics tests run without optional SDKs; the real-engine integration
tests retain their dependency guards.

```bash
python -m pytest tests/qec/test_fixed_shot_statistics.py tests/qec/test_stim_pymatching_baseline.py -q
PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py
```

## Work D: bounded Surface evidence

The existing baseline example now also accepts:

```bash
PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py --code surface --distance 5 --basis x --shots 35 --batch-size 16
```

This uses the same code record, builder, compiled stream, PyMatching adapter and
fixed-shot statistics as the repetition example. The Z experiment reads the
declared row operator. The X experiment passes the certified product of X on
**all data qubits**, so preparation and readout both use the uniform X frame.
The code still declares only Z; requesting `readout_basis="x"` without a product
is not added here. Passing a minimal column product would rotate only that
column and would describe a different experiment. The report records the
chosen basis and full logical support to make this distinction inspectable.

Predeclared developer evidence budget: distances 3, 5, 7, 9, 13, 17 in both
frames, exactly three syndrome rounds, 35 shots in chunks 16 + 16 + 3 per cell,
seed 53, independent X/Z faults 0.005/0.007 per data location and syndrome
readout faults 0.009. Y probability is zero; no correlated mechanism is split
for matching. All twelve cells run once with the same call schedule, bounded
to 240 seconds in total. Retain every cell, including failures; do not change
seeds, append shots or relax tolerances after inspecting results. This is
small structural/capacity evidence, not a threshold, precision, GPU throughput
or advantage experiment. Three fixed rounds do not establish distance-scaled
memory performance. A zero count at 35 shots still has a 9.89% upper bound.

The declared twelve-cell run completed on macOS arm64, Python 3.12.14,
Torch 2.13.0, Stim 1.16.0 and PyMatching 2.4.0. Every cell completed all 35
shots and observed zero logical failures; every interval was
`[0.0, 0.09890099232440398]`. These counts are engine/architecture scoped.
This small all-zero sample does not establish a distance trend.

| Distance | Data qubits | Detectors | X failures/shots | Z failures/shots |
| --- | --- | --- | --- | --- |
| 3 | 9 | 24 | 0/35 | 0/35 |
| 5 | 25 | 72 | 0/35 | 0/35 |
| 7 | 49 | 144 | 0/35 | 0/35 |
| 9 | 81 | 240 | 0/35 | 0/35 |
| 13 | 169 | 504 | 0/35 | 0/35 |
| 17 | 289 | 864 | 0/35 | 0/35 |

Committed integration coverage compares raw measurement-map parities with the
compiled sampler at d=3/5/7 in both frames and one/three rounds, verifies
noiseless preparation/readout, and executes forced X/Z data faults at corner
and interior locations and syndrome faults on weight-two/four checks in first
and last rounds. Injected programs are intentionally non-identical by round;
their independent oracle uses the existing Simulation Clifford seam without
weakening the QEC sampler's identical-round placement guard. At d=3, native
matching, PyMatching per-row and PyMatching batch decode exactly the same
zero/single-mechanism syndrome rows. This pins single-fault correction; it
does not require arbitrary tied matchings to choose identical logical classes.

```bash
python -m pytest tests/qec/test_surface_batch_evidence.py -q
```

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
