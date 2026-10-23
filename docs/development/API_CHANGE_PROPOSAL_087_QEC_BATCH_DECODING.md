# Proposal 087: QEC batch logical prediction

Status: explicitly approved by the user on 2026-10-24; implementation pending
release. Core owns the adapter; Integration reviews the public API delta.
Approval is limited to the Tensor-only method and atomic failure behavior below;
it does not authorize changing exports, serialization or the decoder protocol.
`contracts/qec-batch-prediction-v1.json` records this additive delta. Historical
fixtures remain unchanged; compatibility checks apply the reviewed addition.

## Frozen development baseline

Main at `d286e4172cb9af4d1553a79199ee71f919af8c7a`, cloned on 2026-10-24,
contains the historical-module retirement merged by PR #766. The retirement
contract remains `implemented_pending_release`, targeting 0.4.0; merge does not
establish a published release. Use canonical owners and existing root exports.

Existing capabilities are circuit sampling through the optional Stim boundary,
measurement maps, exact DEM sampling, one decoder registry and single-shot
PyMatching cross-checks. The SDK graph is already reused by a decoder instance.
At that baseline there was no public batch prediction or persistent sampler
session. Do not repeat
the completed package restructuring or restore retired imports.

## First user workflow and executable evidence

```bash
PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py
```

Install the existing stim and pymatching extras first. The example constructs a
distance-three, three-round repetition experiment, samples the original noisy
circuit in bounded chunks and counts complete-shot logical failures. It uses
the approved batch logical-prediction method. Expected invariants are 128 completed shots,
eight detector columns, one observable, and failures between zero and 128.
The exact random count is version/architecture scoped, not a frozen constant.

`tests/qec/test_stim_pymatching_baseline.py` independently recomputes detector and
observable parity from raw measurements, compares the existing decoder with
upstream batch prediction on the same graph, and checks a non-divisible tail
batch. The separate batch conformance suite verifies the new method; neither suite
certifies performance.

## Proposed narrow API delta

The approved implementation adds only this method on `PyMatchingDecoder`:

```python
def decode_batch(self, detection_events: torch.Tensor) -> torch.Tensor:
    ...
```

- Input: CPU, strided-layout, two-dimensional binary bool or integer tensor
  `[B, D]`, where D equals the decoder's detector count. Reject floating,
  complex, sparse, nonbinary and wrong-width inputs before SDK conversion.
  Noncontiguous dense input is accepted with a documented adapter-owned copy.
- Output: CPU bool tensor `[B, L]`, logically owned by the caller, with the
  decoder's existing observable numbering. B=0 returns `[0, L]`; L=0 returns
  `[B, 0]`. Caller input must not be mutated.
- TypeError for wrong carrier/dtype; ValueError for rank, width or nonbinary
  values; CapabilityError if the SDK cannot decode the batch. The whole call
  fails in this first version; no failed row is filled with zeros or discarded.
- Use the existing graph and one upstream `decode_batch` invocation for a
  nonempty batch. No Python single-shot loop and no edge reconstruction in the
  logical-only path. Normalize and validate the SDK answer at the adapter seam.
- No result dataclass, root export, second registry, async operation, GPU claim,
  checkpoint schema, existing protocol change, or full correction output.
  Existing single-shot signatures and behavior remain unchanged.
- Other decoders need not implement this method. A later common batch contract
  requires a second concrete implementation and separate review.

The physical owner is `qec/adapters/pymatching.py`. PyMatching objects remain
private; PyTorch is already a core dependency. Optional SDK import remains lazy.
The future in-house matching implementation must consume the same bit-matrix
meaning and pass the same tests, rather than reproduce SDK graph classes.

## Sampling lifecycle: separate follow-up

The existing sampling API recompiles a sampler on every call. The baseline is
bounded in shots, but does not amortize this work. Persistent compilation is a
separate Simulation/Runtime boundary decision, not an implicit part of this API.

Before proposing a session type, inspect existing Runtime/session ownership and
prove why it cannot cover the use case. Preserve RNG stream semantics, explicit
configuration invalidation and worker ownership. Do not cache mutable samplers
globally by circuit hash or serialize SDK objects.

## Acceptance after approval

1. Compare all output bits against direct upstream batch invocation over the
   same graph; compare the single-shot path with existing tie-aware rules.
2. Cover B=0/1/many, L=0/1/many, all-zero events, noncontiguous input, bad dtype,
   sparse input, wrong width/rank, nonbinary input and impossible syndromes.
3. Verify input immutability, graph reuse, one SDK batch call and zero single
   calls on the fast path. Missing dependency must retain installation guidance.
4. Keep original circuit/DEM sampling separate from decoding decomposition;
   this API does not expand the graph translation's accepted domain.
5. Run the new scenario tests plus existing adapter, registry, restructuring,
   retirement, architecture and public API checks. Do not refresh frozen fixtures.
6. Measure conversion and decoding separately on fixed CPU batches. No speedup
   claim before measurement; throughput inverse is not single-shot latency.
7. Verify installed wheel use outside the checkout with both extras. Separately
   verify that core import and decoder listing work without optional SDKs.

## Implementation sequence

1. This review: freeze baseline, executable single-shot example and independent
   parity/SDK conformance evidence. No library API changes.
2. Review and approve this exact API delta, then implement adapter-only batch
   prediction with focused tests and update API protection through its process.
3. Replace the example's single-shot loop with the approved batch method; add
   complete failure counters and fixed-shot statistical intervals.
4. Design persistent sampler ownership separately, then extend the same proven
   path to Surface memory and explicitly scoped circuit-level Pauli noise.

The user explicitly approved this narrow Tensor-only method and atomic failure
behavior on 2026-10-24. No other public signature is authorized by this proposal.

## S0 validation record (2026-10-24)

Environment: macOS arm64, Python 3.11.16, PyTorch 2.10.0, Stim 1.16.0,
PyMatching 2.4.0. An existing QEC environment was reused without modification;
the initially attempted dedicated environment hit the disk limit and was removed.

- New scenario and existing PyMatching adapter tests: 20 passed.
- Restructure compatibility, package boundaries, decoder relocation and retirement
  contract tests: 61 passed.
- Architecture checker and retired-phase checker: passed (23 retired modules,
  108 exports).
- Ruff checks and formatting on the two Python additions: passed.
- Example: 128 completed shots, 8 detector columns, 1 observable, 0 observed
  logical failures on the recorded environment. This is not a zero-error-rate
  or performance claim.

The required `pr-default` tier was attempted but stopped during collection: the
reused environment lacks setuptools in two unrelated installation/benchmark
imports. It did not execute the selected 15,203 tests and is not a passing gate.
No tests were excluded and no frozen fixtures were updated.

The S0 example used single-shot decoding. The approved follow-up now uses batch
prediction; sampler compilation still occurs per chunk. A persistent sampler
lifecycle remains a separate design task.

## Approved implementation validation (2026-10-24)

The approved method is implemented and the repetition example now calls it.
No frozen fixture, root export, registry entry or existing method was changed.

- 137 focused batch/adapter/registry/restructuring/retirement tests passed.
  The two public doctest gates also passed after registering the new example.
- Architecture, retired-phase, public API migration baseline, Ruff and formatting
  checks passed. The reviewed addition is layered over the immutable historical
  API fixture using the new explicit contract, not a regenerated snapshot.
- A macOS arm64 CPython abi3 wheel built successfully. Installed outside the
  checkout under Python 3.11, it passed binary batch prediction and the 35-shot,
  16-shot-chunk tail-batch example. Installed import/refusal checks also passed
  with import blockers simulating neither SDK, each SDK separately, and both.
  These are availability simulations, not four independently installed envs.
- The required default smoke/unit tier ran with four workers and maxfail=5:
  10,817 passed, 103 skipped, 4 failed and 5 setup errors before stopping. The
  failures concern unchanged QFT pass-count/native-region expectations; errors
  concern local Gloo address resolution. A separate archive of the frozen main
  baseline reproduced all four failures and all five reported setup errors
  (plus another instance of that setup error). The default gate is not green;
  no test exclusion or unrelated numerical change was made to hide this.
- No performance, GPU, threshold or logical-suppression claim is made. Persistent
  Stim sampler ownership is still a follow-up, not part of this batch method.

The local implementation commit is `4a50004`. Publishing the branch and opening
a draft PR require explicit authorization for the GitHub destination after the
automatic push review rejected publication; no push or PR creation succeeded.
