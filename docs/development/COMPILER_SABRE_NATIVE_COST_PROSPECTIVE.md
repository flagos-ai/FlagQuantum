# Prospective check of native-CZ-aware routing

Status: freeze this protocol in Git before drawing the nonce. This is a
`single_device_fast_path` development check of an opt-in experiment, not resource
qualification or permission to change the production compiler default.

## Frozen implementation and cases

Use integrated compiler commit `11745ab11070765974b90f5a8f512bf4c30576b1` and
its unmodified `benchmarks/compiler_sabre_native_cost.py`. The control is that
tree's production selector. The candidate is its experimental `_select`: bounded
existing SABRE trials, call-local lowering and exact matrix-cost reuse, one
shortlisted alternative, and bounded native synthesis of incumbent and candidate.
Selection requires a strict delivered CZ reduction without increased two-body
depth. Retain all independent native/routing verification and artifact emission.
The source and selector hashes must be recorded before execution. No post-draw
implementation change, retuning, fixture exception or redraw is permitted.

Use the unchanged clean SDKBench checkout
`9aa9260920984e2ca05bcb293abb885fccd9c144`, including its generator, target-artifact
adapter, evaluator, acceptance tolerances, compiler seed and existing native
operation budget. Read that budget from the adapter configuration; do not use
the experimental helper's standalone default in place of the declared workflow.
Run every width 6/8/12, application QAOA/Ising/hopping/QFT/repetition and topology
line/ring/grid/sparse combination: 60 complete paired target-artifact workflows.
This study does not compare SDKs or include ordinary compilation.

After committing this document, draw exactly one 32-byte nonce with
`secrets.token_hex(32)`. Before executing any compiled task, retain a manifest
with the freeze commit, compiler and harness revisions, protocol/runner/selector
digests and nonce. Generate and retain source-operation/target-edge fingerprints
separately from probe fingerprints. Compare against both the earlier `01`
repeated nonce and the retained SABRE-depth prospective nonce. Report actual
new source/target pairs and repeated controls; deterministic QFT and probe-only
changes must not be described as new programs.

## Acceptance and decision

Require every existing independent evaluator to accept both variants and their
identity endpoint layouts. Width six uses complete operators up to global phase;
widths eight and twelve check three probes and cannot establish full equivalence.
Compare CZ count and two-body dependency depth separately in every pair. Any
semantic/layout failure, unsupported case or increase of either quality metric
rejects general admission of this candidate. Attempt the whole fixed set, retain
all failures, and require a nonzero runner exit for any failure or regression.
An outer timeout leaves incomplete evidence and cannot be counted as acceptance.

All quality results are descriptive evidence for this frozen set. Even a clean
result cannot establish universal improvement or authorize default admission.
The retained production-control timing losses remain a separate unmet resource
requirement; this study neither replaces nor averages them away. Do not interpret
baseline-first elapsed fields as latency evidence or profiler/allocation results
as process RSS. Formal resource-qualified SDKBench remains user-deferred.

## Execution and supported workflow

Use one numerical thread, a separate supervisor with a 600-second process-group
deadline, and TERM/KILL cleanup. Keep the harness's own timers unchanged. No host
configuration change, kernel upgrade, reboot or service restart is needed.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.compiler_sabre_native_cost \
  --sdkbench-root /path/to/unchanged/sdkbench --nonce DRAWN_NONCE \
  --output /tmp/sabre-native-cost-prospective.json
```

The existing user workflow remains `fq.Circuit` followed by compiler
`compile(..., coupling_map=CouplingMap.line(width))`, with restored logical output
order. The existing native target-artifact workflow additionally legalizes the
target basis and retains independently replayable block evidence. Importing this
experiment installs no production policy. The shortest existing verification is:

```sh
python -m pytest tests/unit/test_compiler_sabre_native_cost.py \
  tests/unit/test_native_block_optimization.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

Expected: numerical preview/restoration, native budget refusal, unsupported-input
refusal and physical-evidence acceptance tests pass. These tests support the
existing workflow; prospective acceptance must come from executing the frozen
cases above, not from this documentation or passing CI alone.
