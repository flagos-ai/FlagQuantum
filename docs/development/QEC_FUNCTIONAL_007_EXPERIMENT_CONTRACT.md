# QEC-FUNCTIONAL-007: Fixed-shot memory experiment contract

## Review status and authority

The API owner explicitly approved the complete contract's implementation,
verification and delivery on 2026-10-26 (Asia/Shanghai), after reviewing the
signature, seven result fields, source path, support/refusal matrix, numerical
interval and exception table. This approval is independent of the earlier
Decision B approval. [#924](https://github.com/flagos-ai/FlagQuantum/pull/924)
delivered the review artifact; its exact merge
`4ec20e1adf4c1b92ef878c49d42977f25ace3318` passed Main smoke `40314835571`.
That is not full main CI. The earlier design-only status is historical.

This implementation candidate adds the two canonical module names below without
root or parent re-exports. Final reviewed-head CI/GPU and merge acceptance remain
pending. Adaptive stopping, threshold fitting and persistence remain deferred.

## Approved complete public surface

Only `flagquantum.qec.execution.experiments` defines and exports these two
names. Add no root `flagquantum`, root `qec` or parent `execution` re-export.
Existing exports, signatures, defaults, plugin contracts and result records stay
unchanged.

```python
# Canonical declarations in flagquantum.qec.execution.experiments.
@dataclass(frozen=True)
class MemoryExperimentResult:
    shots: int
    logical_failures: int
    logical_error_rate: float
    observable_failures: tuple[int, ...]
    confidence_interval: tuple[float, float]
    seed: int | None
    sampling_source: Literal["memory_circuit"]


def run_memory_experiment(
    memory: MemoryCircuit,
    *,
    noise: PhenomenologicalNoise | NoiseModel,
    decoder: str,
    shots: int,
    batch_size: int,
    seed: int | None = None,
) -> MemoryExperimentResult:
    ...
```

The seven result fields have exactly the stated order and no additional fields
or constructor defaults. Returned counts and seed are normalized Python ints;
confidence endpoints are floats. The tuple fields contain immutable scalar
values. Result equality is ordinary dataclass field equality. No portable
pickle/JSON representation, save/load API, checkpoint, artifact registry or
cross-version serialization promise is proposed. Constructor validation is not
an additional public service: these invariants describe records returned by the
experiment function, not a new result-builder workflow.

`shots` and `batch_size` accept `numbers.Integral` values except booleans, reuse
the existing QEC positive-shot validation semantics and normalize once. Both
must be positive; batch size may exceed shots. `seed` accepts None or a
non-boolean Integral, normalizes to int, and must be in `range(2**64)`, preserving
the owning Simulation seed range. Do not coerce strings or floats. No new upper
shot cap, batch default, decoder option dictionary, confidence-level option,
backend selector, source selector or adaptive stopping parameter is introduced.

## Source, preparation and lifecycle

The initial implementation path is explicit and synchronous:

1. Validate scalar types/ranges, memory/noise types, registered decoder name,
   and at least one declared detector and observable before sampling.
2. Export one immutable original positioned memory source through existing
   `memory_to_stim_text(memory, noise=noise)`. Reuse its source/noise snapshot
   boundary; do not add a second NoiseModel or placement implementation.
3. Obtain the strict actual-law model from that exact text using
   `stim_circuit_to_dem_text(text, decompose_errors=False)` followed by the
   existing DEM reader. Do not use nominal-decimal `from_memory_circuit` as an
   interchangeable model when its floating-point factory law differs. Do not
   enable disjoint-error approximation or decompose composite faults.
4. Resolve the requested registered decoder with `get_decoder(name, model)`
   and no options, admit only an exact certified concrete type, and retain all
   current model-construction refusals. Do not bypass a registry replacement.
   Existing exact duplicate-mechanism normalization may be used only where the
   owning decoder already defines it; no new graph/model transform is proposed.
5. Call `compile_stim_sampler` exactly once on the original text, with one seed,
   after all preceding preparation succeeds. This is one compiled stochastic
   sampler, not a promise that export and strict analysis share one SDK parse.
6. Within its owned context, sample `min(batch_size, remaining)` shots until
   exactly `shots` complete. For 35 shots and size 16, requests are 16, 16, 3.
   Convert each binary detector row to sparse indices and call the delivered
   complete-record `decode_batch`. Validate observable predictions and aggregate.
7. Close on success and on failure. No partial successful record is returned.

Both DEM and sampler use the same original text and its reference-normalized
observable flips. A matching graph is never the sampling source. Retain exact
exclusive/correlated mechanism semantics and existing strict conversion limits;
if a source/model/decoder combination cannot represent them, refuse it. No new
private cross-call cache, SDK object escape, global cache or Simulation API is
needed. The library must not import examples.

The initial function requires the existing optional Stim extra for every
successful call; it has no core-only sampling fallback. PyMatching additionally
requires its existing optional dependencies. Dependency presence is checked
lazily by the owning boundary. A refused input consumes no sample shots. Tests
must establish deterministic validation before compilation/RNG where knowable;
no precedence is promised when multiple independent invalid inputs are supplied.

## Supported inputs and decoder matrix

| Input or selection | Initial contract |
| --- | --- |
| Memory | Existing `MemoryCircuit` accepted by the current export/strict-analysis/sampler intersection, with at least one detector and observable. Repetition and Surface examples are required; other native layouts are admitted only through the same existing grammar and bounds. |
| Noise | Existing `PhenomenologicalNoise` or `NoiseModel` whose actually executed channels and placement are accepted by that intersection and the selected decoder. No nominal-law substitution or non-Pauli approximation. |
| `minimum_weight_matching` | Exact `MinimumWeightMatchingDecoder`, constructor defaults; preserve graphlike/model, defect-limit, tie and refusal behavior. |
| `sliding_window_matching` | Exact `SlidingWindowMatchingDecoder`, constructor defaults; independent-shot behavior as certified by 005, not stateful streaming. |
| `belief_propagation` | Exact `BeliefPropagationDecoder`, constructor defaults; preserve complete results and convergence reporting, including a normally returned non-converged result. |
| `belief_propagation_osd` | Exact `BeliefPropagationOsdDecoder`, constructor defaults; preserve its complete results and algorithmic refusals. |
| `pymatching` | Exact `PyMatchingDecoder`, constructor defaults and existing optional dependency errors; use full-record 005, not dense logical-only batch output. |
| Registry replacement/plugin/subclass/stateful decoder | Resolve the existing registry normally. A returned type outside those five exact classes raises `CapabilityError`; never replace it silently with a built-in. Constructor errors propagate before admission. |
| External Stim text, DEM, graph, matrices or decoder instance as top-level inputs | Refused. The function accepts memory/noise and an explicitly selected registered string only. Existing lower-level workflows remain available. |
| Nondefault decoder configuration, AI preprocessing, hardware, GPU, distributed execution | Outside this first contract; no inferred support or automatic substitution. |

The five names are explicit selections, not a promise every model works with every name.
Each retains its owning model and syndrome support. Tests must include both
successful supported combinations and refusal combinations; missing PyMatching
cannot be skipped while claiming five-decoder acceptance.

## Counts, prediction validation and interval

Let B be the actual batch length and L the model's observable count. Require
binary CPU detector/observable tensors of expected two-dimensional shapes,
matching source/model detector and observable dimensions. Each existing result's
sparse observable indices must be non-boolean integers in `[0,L)`, with no
duplicates; malformed backend output raises the existing `ValidationError`.
Build predictions for aggregation only; never synthesize correction edges,
weights or convergence fields.

A shot fails if any of its L predicted bits differs from its sampled flip.
Two wrong bits in one shot count as one logical failure and one failure in each
corresponding observable. Do not average per-observable rates or normalize per
round. A normally returned non-converged decoder record still has its original
prediction and is counted; an algorithm that raises terminates the experiment.
No postselection, dropped row, retry or changed denominator is permitted.

On success, `result.shots == requested_shots`, `logical_failures` is in
`[0,shots]`, `logical_error_rate == logical_failures / shots`,
`observable_failures` has length L and counts in `[0,shots]`, and
`sampling_source == "memory_circuit"`. Seed None remains None; do not invent an
unrecorded reproducible seed. Reproducibility is scoped to the same source,
configuration, versions, architecture and sample-call schedule. A different
batch size need not produce identical samples; never reseed per batch.

Use the existing example's two-sided 95% Wilson score formula with
`z = statistics.NormalDist().inv_cdf(0.975)`, without continuity correction.
For n shots and p = failures/n, compute denominator `1 + z*z/n`, center
`(p + z*z/(2*n))/denominator`, and half-width
`z*sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator`. Clamp to `[0,1]` and set the
lower endpoint exactly zero at zero failures and the upper endpoint exactly one
at all failures. Zero failures retain a positive upper endpoint. Zero shots are
invalid; no adaptive interval, Hoeffding interval, threshold or significance
claim follows. The implementation and existing baseline example reuse one private formula,
preserving independent score-inversion oracle tests.

## Exception and failure contract

| Condition | Exposed exception / effect |
| --- | --- |
| Invalid memory/noise/decoder carrier; boolean or nonintegral shots/batch size/seed | `TypeError`, before sampling. |
| Nonpositive shots/batch size, unknown decoder name, missing detector or observable | `ValueError`, before sampling. |
| Integral seed outside `[0,2**64)` | Existing Simulation `ValidationError` (a `ValueError` subclass), before sampler compilation/RNG. |
| Unsupported source grammar, channel, strict DEM translation, decoder type or algorithmic syndrome | Existing `CapabilityError`; retain owning messages/subclasses. Model construction may retain its existing `ValueError` for an invalid model. |
| Missing Stim | Existing `StabilizerDependencyError`; no fallback. |
| Missing PyMatching | Existing `MatchingDependencyError`; no substitution. |
| Malformed sample shape/bits or decoder observable output | Existing `ValidationError`; close the sampler and return no result. |
| Existing sample/decode/compile failure | Propagate its original exception unchanged; preserve normal non-convergence records rather than inventing a new exception. |
| Cleanup also fails during another failure | Preserve the primary exception, attach cleanup context and release everything the owner can release. On a sole cleanup failure, propagate it and return no result. |

No new exception class or blanket wrapper is proposed. Error strings are
explanatory, not a new frozen literal-message contract. A live sampler is owned
by the creating process/thread; this synchronous function never transfers it.
Returned records retain no sampler, SDK object, source tensor or sampled batch.

## Evidence and acceptance before implementation delivery

Callers retain explicit configuration beside the result: code/layout/rounds and
logical support, exact noise settings, decoder name/default configuration,
shots/batch size/seed, source label and installed versions. The example evidence
dictionary has no stable schema and contains no mandatory private identifiers.
No new raw study, performance claim or hardware budget is authorized here.

An implementation PR must provide the following evidence:

- Independent two-observable counts, including two wrong bits in one shot;
  exact 35-shot 16+16+3 schedule, one compile and no seed reset.
- Controlled zero/all/mixed outcomes and analytical Wilson score-inversion
  oracles; no flaky statistical pass criterion.
- Scalar/seed/domain and zero-observable refusals before RNG; late sample,
  decode, malformed prediction and cleanup failure with no partial result.
- Registry replacement/subclass/stateful refusal while existing single-shot
  plugin behavior stays intact; all five exact classes, complete records,
  convergence and algorithm-specific refusals exercised with actual SDKs.
- Strict actual-law equality for both noise carriers and the original-source
  path; preserve current precision tolerance and exclusive events. Refused
  source/model/decoder intersections must not be weakened to obtain success.
- Core import/refusal plus actual Stim/PyMatching/both profiles through source,
  installed wheel and installed sdist, with asserted import origin; source
  archive and generated package contents must cover the final reviewed change.
- A runnable public Python example with installation/device conditions and real
  output, exact-head complete CI/GPU, current rules/reviews, normal pinned merge
  and actual merge/main verification. Main smoke is not full main CI.

The explicit owner approval covers this complete signature, ordered result
fields, source path, support/refusal matrix, numerical interval and exception
table. Document merge alone was not that approval.
Adaptive stopping, threshold fitting, nondefault decoder options and persistence
remain separate later decisions.

## Candidate user workflow and support limits

Install `python -m pip install -e '.[stim,pymatching]'` and run
`PYTHONPATH=. python examples/qec/memory_experiment.py` on CPU. The example uses
35 shots, explicit batch size 16, seed 53 and constructor-default decoder names.
Each of the five lines reports its name followed by `35 0 (0,)` in the tested
Python 3.12.14 / Torch 2.13 / Stim 1.16 / PyMatching 2.4 environment. This is
small deterministic acceptance evidence, not a logical suppression or throughput
claim. Same-seed reproducibility remains scoped to versions, architecture and
sample-call schedule.

The strict DEM can refuse a source before sampling; decoder constructors can
refuse its model. In particular, zero noise leaves isolated declared detectors
that the existing PyMatching adapter refuses. No padding edge, decomposition or
fallback is added to make that selection pass. The example's nonzero supported
noise avoids this refusal. Both supported phenomenological and gate-bound noise
paths are checked against the original text and independent single-shot records.
