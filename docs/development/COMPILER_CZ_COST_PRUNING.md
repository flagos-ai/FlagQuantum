# Rejected CZ synthesis cost pruning

**Status: rejected for integration.** The production precheck from
`c446adc9802655e40f50b1ad5984054097bf0567` has been restored to the unchanged main
implementation. The final change retains operator and resource-bound tests plus
the positive and adverse evidence below. It adds no compiler optimization.

The initial complete-workflow observations were small, unqualified differences.
Subsequent controls found predictable extra decomposition work in reducing
blocks. That tradeoff does not justify the added private synthesis coupling on
the present evidence. No input-specific exception, threshold adjustment, new
synthesis entry or cache was introduced to rescue the proposal.

## Reducing-block controls and rejection decision

The [additional control record](evidence/cz-cost-pruning-reduction-controls.json)
retains the immutable baseline/candidate sources, exact driver, six inputs,
outputs, timing samples and separate Python allocation observations. A NumPy
operator implementation independent of the Weyl classifier checks each returned
proposal up to one operator-wide phase at absolute tolerance `1e-10`.

| Inspected block | Input / output CZs | Weyl calls, baseline / candidate | Stage median, baseline / candidate |
| --- | ---: | ---: | ---: |
| Refocused pi-X | 2 / 0 | 1 / 2 | 182 / 280 us |
| Refocused pi-X with frame rotations | 2 / 0 | 1 / 2 | 211 / 310 us |
| Quarter-turn Pauli product | 2 / 1 | 1 / 2 | 357 / 462 us |
| Mixed-wire reduction | 3 / 2 | 1 / 2 | 477 / 594 us |
| Generic two-CZ control | 2 / unchanged | 1 / 1 | 203 / 129 us |
| Generic three-CZ control | 3 / unchanged | 1 / 1 | 322 / 204 us |

The four accepted reductions retain identical proposals. The two generic controls
retain identical non-reduction refusals. The precheck helps the latter by
avoiding leaf construction; it repeats the same Weyl work before ordinary
synthesis in the reducing cases. The first two reducing controls also increase
the observed Python allocation peak by 616 bytes each; the other controls have
smaller observed peaks. Every sample is retained rather than selecting one side
of this tradeoff.

These are inspected stage-level development controls on a shared macOS host,
with three initial calls per variant, two alternating warmup rounds and twenty
retained rounds. They are not whole-compilation, qualified Linux, held-out or
cross-SDK performance results. The deterministic extra work and these adverse
observations motivate rejection; no claim that the complete SDK is slower by
these ratios follows. The complete-workflow results below remain unchanged.

The first diagnostic omitted the compiler source path and failed during import.
After restoring that environment, it completed. Two initial labels incorrectly
called reducing examples non-reducing; the output CZ counts established the
mistake. That original record is retained, and the frozen-source rerun uses
descriptive labels and adds the two generic controls without changing inputs or
observations in the original record.

## Historical prototype

Native CZ block analysis can avoid complete leaf synthesis when the existing
Weyl classifier says that the block cannot reduce CZ count. The rejected prototype
uses a conservative raw-size bound to preserve instruction-growth rejection
precedence. It changes no public API, synthesis formula, tolerance, native budget,
routing strategy or accepted rewrite criterion.

This was a `single_device_fast_path` compiler proposal. It reused the existing
Qiskit-derived `_BasisDecomposer.entangler_count` and Weyl engine, whose notices
remain in `optimization/two_qubit_synthesis.py`. It is not a new algorithm port.
Cirq-derived local cleanup, complete-operator residual acceptance and independent
physical-evidence replay continue to own their existing responsibilities.

## Why the resource condition is necessary

The existing candidate generator checks instruction growth before reporting a
non-reducing CZ count. Rejecting from a Weyl count alone would change that
observable rejection reason when resources are tight. A lower bound is also not
proof that a reducing proposal will actually be accepted.

For the existing general CZ/RZ/RX form, a proposal with `k` CZ gates has `k + 1`
layers, two local factors per layer, and at most five leaves per local factor.
Its raw instruction count is therefore at most `k + 10 * (k + 1)`. The classifier
returns at most three entanglers. Two and three entanglers give bounds of 32 and
43 operations respectively. This bound describes the current fixed-pulse general
form before native local cleanup, rather than assuming arbitrary-angle rotations
will always shorten it.

Only blocks with two or three baseline CZs can be rejected by this classifier.
Blocks already needing at least four CZs retain their existing synthesis path,
since the generic exact form can use at most three. If the effective operation
limit is below 32, classification is skipped. Otherwise the product must pass the
existing unitary predicate, then the same Weyl/basis classifier computes `k`.
Early rejection requires both `k >= baseline CZs` and enough resources for the
entire raw-size bound. If either condition fails, the original synthesis and
rejection ordering execute normally. Nonunitary input still reaches the original
synthesizer and its validation failure. No speculative proposal is accepted from
classification alone.

The small precheck deliberately recomputes KAK in the ordinary synthesizer for
candidates it cannot reject. It does not create a second synthesis entry, change
the protected public signature, retain input-dependent caches or bypass a proof.
This extra work is one reason call-count reductions do not translate directly
into equivalent end-to-end gains.

## Complete inspected comparison

Production control: `6db1fbda8a168599e5c0f155b9fb14052cf01342`. The candidate source
hash identifies the measured working-tree implementation at that parent.
SDKBench remains unchanged and clean at
`9aa9260920984e2ca05bcb293abb885fccd9c144`, with already inspected nonce `01`
repeated 32 times. Widths 6/8/12, five application families and four topologies
produce 60 paired complete workflows and 120 independent acceptances.

Every pair retains identical normalized output hashes, evaluator metrics,
endpoint identity layouts, restored SWAP counts, complete native rewrite records
and rejection counts. Six-wire acceptance covers complete operators up to global
phase; larger workflows retain the existing probe scope. This is inspected
development evidence, not unseen qualification or a formal cross-SDK rerun.

Each adapter call creates a newly timestamped target snapshot, so its
snapshot-bound optimization identity is expected to differ across calls. Both
identities and target snapshot IDs are retained as recorded; equality is checked
on the actual rewrite records and rejection counts instead. The original driver
mistakenly compared those time-dependent identities and stopped during timing.
The archive preserves its exact driver, document hash and failure summary. No
SDKBench evaluator or acceptance threshold was changed to correct that diagnostic.

[Raw outputs, evidence records, samples, profiles and exact drivers](evidence/cz-cost-pruning.json)
retain the complete comparison and the preceding rejection-profile observations.
The diagnostic initially named the candidate source hash `runner_sha256`; the
archive labels it `implementation_sha256`, records the actual driver hash as
`runner_sha256`, and retains the original document hash without changing numbers.

## Cost observations and limits

Two warmups and seven alternating retained timing samples per variant, followed
by separate Python allocation observations, give:

| Inspected workflow | Existing median | Pruned median | Python traced peak, existing / pruned |
| --- | ---: | ---: | ---: |
| 6-wire QFT, grid | 114.185 ms | 112.272 ms | 1,973,630 / 1,915,796 bytes |
| 6-wire repetition, ring | 128.838 ms | 127.122 ms | 2,450,987 / 2,470,141 bytes |
| 12-wire QFT, grid | 617.434 ms | 612.036 ms | 10,118,660 / 10,109,323 bytes |

The median differences are approximately 1.7%, 1.3% and 0.9%. Repetition-ring's
traced peak increases by 19,154 bytes. These small differences do not establish a
stable speedup or statistical significance on this shared host. Python traced
allocation is not process RSS. No resource-qualified performance or SDK ranking
claim follows.

Separate profiler observations reduce complete `synthesize_two_qubit` calls from
28 to 5, 23 to 1, and 128 to 38 respectively. Profiling includes instrumentation
overhead and is not substituted for unprofiled samples. The original acceptance
path still executes for every pair. Qualification and full-checkout CI remain
necessary before integration; this does not enable the experimental target-aware
routing selector.

## User workflow and tests

The supported workflow remains native target-artifact compilation through
SDKBench's existing `flagquantum-target-artifact` adapter. Focused tests exercise
its compiler and separate physical-evidence verifier:

```bash
python -m pytest tests/unit/test_compilation_two_qubit_synthesis.py \
  tests/unit/test_compilation_two_qubit_optimization.py \
  tests/unit/test_native_block_optimization.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

The withdrawal selection passed 222 tests before synchronization with main.
After incorporating main `02bacc099cdcd1ac08510c8cf944df6e018c94c9`, all 225 selected tests pass. The retained scenarios preserve rejection priority at
budgets below, at and above the actual raw proposal size; compare seeded general
forms against the runtime's complete operators and the conservative bound; and
verify the four reducing controls against complete operators. The implementation-
specific test requiring the rejected early-return path was removed with that
path; no scientific assertion or resource threshold was weakened. Existing symbolic, annotation,
reversed-operand, topology, layout and tamper scenarios remain unchanged.

The original sparse default-tier selection passed 177 unit tests with 42
integration tests deselected. Final withdrawal checks are recorded in the
additional control record. Neither selection is complete-checkout default-tier
certification; exact-head CI remains required before merging. The independent
CI stack-watchdog repair remains separate from this rejection decision.

The subsequent CI vocabulary gate rejected the newly named `main_commit`
metadata field. It is recorded as the existing `source_commit` field instead;
the revision and numerical evidence are unchanged, and the vocabulary gate is
not expanded.

A Git-object walk of all 1,002 tracked JSON evidence artifacts, with the changed
working artifact substituted, matches all 20 declared revision field names.
This verifies field vocabulary, not full provenance or full CI acceptance.
