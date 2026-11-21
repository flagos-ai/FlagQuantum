# Prospective development check of the fixed SABRE depth candidate

Status: frozen protocol before drawing a fresh input nonce. Compiler owns this
`single_device_fast_path` diagnostic. Production selection remains unchanged.
This check can reject a candidate; it cannot qualify resources or authorize
enabling it while the formal SDKBench campaign is deferred.

## Frozen implementation and inputs

The production source is PR #1272 head
`553054f4206ff80c9f4d3005556dd2b3329dd567`, including bounded native identity-prefix
collection. The baseline is that source's unmodified selector. The candidate is
the exact `_select` function in `benchmarks/compiler_sabre_depth.py`: the existing
three seeded and one layer-lookahead trials, equal restored-SWAP cost resolved
by weighted restored depth, and committed pruning bound `best_cost + 1`.
Directed/custom-cost callbacks delegate to the original selector. The diagnostic
is not a new Qiskit/Cirq source port. No candidate or acceptance change is
allowed after the draw within this study. A changed candidate needs a new study.

Freeze this document and the runner in Git before generating one fresh 32-byte
nonce with `secrets.token_hex(32)`. Retain the freeze commit, runner digest,
selector AST digest, this protocol's digest and the drawn nonce in an evidence
manifest before running the workload. Do not redraw in response to results.
Use unchanged SDKBench source
`9aa9260920984e2ca05bcb293abb885fccd9c144`, its application generator, acceptance
cases, seed, caps and adapters. Run all 120 pairs: widths 6/8/12, five application
families, four topologies and both ordinary-restored and target-artifact workflows.
No SDK-specific exception, target ancilla or free final layout is admitted.

A new nonce does not make every application new: QFT operations are deterministic
and its line/ring/grid targets are unchanged. Record source-operation/target-edge
fingerprints separately from probe fingerprints, comparing with the previously
inspected `01`-repeated nonce. Count actual new input pairs and repeated controls;
never label the whole set as unseen. Probe-only changes are not new programs.

## Acceptance and rejection

Apply the existing independent evaluator to both variants and require identity
output layouts. Six-wire cases check full operators up to global phase; larger
cases check three probes and do not establish universal equivalence. Preserve
all failures and attempt the full frozen case set; nonzero exit on any failure.
Outer process termination leaves the report incomplete and is not acceptance.

Compare delivered `(CZ count, CZ dependency depth)` lexicographically within each
workflow/input pair. Any semantic/layout failure, unsupported case or delivered
quality regression rejects general admission of this candidate. Report each
tradeoff separately even when aggregate sums improve. Retain the historical
rejection and later inspected-case results unchanged. No fixture exception or
post-draw retuning is allowed. Passing this small development set is necessary
evidence only, not proof of universal improvement.

## Resource boundary and reproduction

Use one numerical thread and a 600-second outer process-group deadline with
TERM/KILL cleanup. The harness's acceptance remains unchanged; this script does
not implement the formal resource executor. Its baseline-first measurements are
not latency or memory qualification. This development study requires no host
configuration change, driver change or reboot.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.compiler_sabre_depth --suite applications \
  --sdkbench-root /path/to/unchanged/sdkbench --nonce DRAWN_NONCE \
  --output /tmp/sabre-depth-prospective.json
```

The unchanged user workflow remains `fq.Circuit` followed by the existing
compiler `compile(..., coupling_map=CouplingMap.line(width))`, with restored
logical output order. This study changes no public API or default. Subsequent
resource admission still needs predeclared repeated alternating trials, a
qualified host, actual process-memory enforcement and a bounded-search tradeoff
review. Until then, keep the candidate disabled even if all quality checks pass.

Before the draw, the failure-path regression passes: one injected case fails,
all other 79 actual generic cases complete their independent exact GF(2) checks,
the report retains the failure and the process exits nonzero. Historical reports
are unchanged. The runner's new `--nonce` defaults to the earlier development
nonce; new reports explicitly record their nonce and completion state.
