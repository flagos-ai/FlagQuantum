# Native route-preview translation templates

## Scope and mechanism

This rejected `single_device_fast_path` experiment attempted to reduce repeated work in the opt-in
native-CZ route selector. It does not enable that selector in production.
Each preview closure translates and numerically validates each source instruction
once, then remaps the resulting native leaves to each trial's physical operands.
One neutral SWAP template serves all forward and restoration SWAPs. Templates and
physical-placement results are discarded when the closure is released. There is
no cache across compilations or sources. Symbolic, batched, trainable, annotated,
custom-matrix and workspace-adding translations remain outside admission.

The existing basis translator, Qiskit-derived KAK/Weyl implementation, completed
SABRE trials, realized CZ/depth acceptance, growth budgets and independent native
routing replay remain authoritative. See [upstream reuse](COMPILER_UPSTREAM_REUSE.md)
and [trial reuse](COMPILER_SABRE_TRIAL_REUSE.md). No dependency, public API,
default, result field or serialized schema is changed.

## Prospective protocol, November 23, 2026

Publish this protocol and freeze the implementation before any application or
timing observations. The base is `674f83e39a5ffdee14d33d29c61aab791f36c6a2`.
Load its exact experimental preview implementation from Git in the same process;
verify AST equality outside `_native_previews`. Both variants use the existing
completed-trial selector and the same production compiler. Record implementation,
base and SDKBench revisions, source and driver hashes, dependency versions,
configuration, input nonce and every observation. The source tree and SDKBench
must stay clean and unchanged during measurement. Failures remain in evidence.

Use unchanged SDKBench `9aa9260920984e2ca05bcb293abb885fccd9c144`, seed 11,
its configured 20,000-operation native growth budget and
`flagquantum-target-artifact`. The ordinary workflow is unaffected because it
does not invoke the experimental selector; this experiment makes no claim of
improving its output. No participant-specific adapter or evaluator edits apply.

Quality phase: execute all 60 combinations of widths 6/8/12, five existing
application families and four topologies, for both variants. Alternate variant
order. Run the inspected nonce `01` repeated 32 times, then one fresh nonce drawn
and frozen after publication. This is a fresh draw, not an independently blinded
held-out test. Require every workflow to pass the unchanged evaluator, identity
final layout and coupling legality; require byte-identical operation/layout
payloads, identical audits and identical metrics for every pair. Preserve
compile times separately, excluding them from semantic hashes. Width six checks
full operators; widths eight/twelve check three probes with the existing
`1e-8` threshold. Replay all delivered payloads in a fresh process without
importing FlagQuantum, Qiskit or Cirq. Any changed output or rejection blocks
admission even if another input improves.

Only after both quality draws pass, observe complete adapter execution on the
inspected 6-wire QFT/grid, 6-wire repetition/ring and 12-wire QFT/grid controls.
Use two warmups per variant, nine retained rounds in alternating order, and one
separate traced Python peak after garbage collection per variant. Verify every
warmup, timed and traced output and audit against its quality observation.
Run no other local tests during measurements. Bound the quality draws to
600 seconds each, resources to 300 seconds, and each report to 16 MiB. Retain
any timeout, failed attempt or interruption; do not replace unfavorable samples.

Resource admission requires all three median latency ratios (new / old
experimental selector) at most `1.05`, all Python peaks at most `1.10` times the
old peak and at most 128 KiB above it, and at least two controls with latency
ratios at most `0.90`. These are development filters, not statistical significance
or production performance certification. A shared macOS host cannot certify
Linux limits, process RSS or cross-SDK rankings. Formal benchmarking remains
deferred; no reboot, service restart or kernel upgrade is authorized. Passing
this protocol only admits the template implementation in the experimental
selector. Default native-cost routing needs separate complete resource and
quality evidence against production.

## Contributor verification and user workflow

The preview tests check reversed operands, restoration, full operators on small
programs, multiple physical placements, once-per-source translation within a
closure, closure isolation, and unsupported values. Related tests retain bounded
synthesis and independent physical evidence checks.

```sh
python -m pytest tests/unit/test_compilation_sabre_trials.py \
  tests/unit/test_compilation_sabre_dependencies.py \
  tests/unit/test_compiler_sabre_native_cost.py \
  tests/unit/test_native_block_optimization.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

The supported user workflow remains:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

circuit = fq.Circuit(4)
circuit.h(0)
circuit.cx(0, 3)
compiled = compile(circuit, coupling_map=CouplingMap.line(4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

Expected: a topology-legal circuit with restored logical output order. The
experimental change does not alter this default workflow.

Frozen control driver SHA-256: `803bda9c8aa6ddf01f339e6554315e13f06c612b8f51a1e4401df06e126fb869`.
The evidence archive will include its exact source.

## Observed result and withdrawal

The frozen implementation was published at
`1f4719cd49f957fd9b2cbd19c2cd0b128dc8c568` before application observations.
The [inspected archive](evidence/native-preview-templates-inspected-screen.json)
and [fresh archive](evidence/native-preview-templates-fresh-screen.json) retain
120 complete workflow observations each. Their hashed payload chunks reconstruct
the original report bytes exactly. All 120 old/new pairs have identical
operation/layout payloads, audits and evaluator metrics; there are no failures,
unsupported cases or changed payload pairs. This demonstrates preservation, not
improved compilation quality. The fresh draw was frozen after publication and
is not an independently blinded held-out test.

[Producer-free replay](evidence/native-preview-templates-replay.json) accepts all
240 outputs, with maximum semantic residuals `6.072657351995283e-14` and
`5.954448375747785e-14`. The existing full-operator/three-probe scope applies.

[Complete resource observations](evidence/native-preview-templates-resources.json)
retain every warmup, timed and traced output and audit. All memory and per-case
nonregression filters pass, but zero controls meet the predeclared `0.90` median
ratio, below the required two. Do not relax that threshold after observation.

| Inspected control | Old experiment median | Template median | New / old | Old / new Python peak bytes |
| --- | ---: | ---: | ---: | ---: |
| 6-wire QFT, grid | 196.732 ms | 189.307 ms | 0.9623 | 2,268,670 / 2,266,428 |
| 6-wire repetition, ring | 202.667 ms | 198.974 ms | 0.9818 | 2,600,203 / 2,601,809 |
| 12-wire QFT, grid | 558.141 ms | 558.040 ms | 0.9998 | 10,118,214 / 10,118,164 |

These shared-host development observations establish neither significance nor a
ranking. The template implementation is withdrawn; the experimental selector
and all production Python modules are byte-identical to the integrated base.
Retain the expanded rejection checks for batched/trainable inputs and complete
small-operator checks across different trial placements. Do not retain tests
that demand a rejected caching implementation.

Verification before withdrawal: 235 focused tests, Ruff, Black and strict mypy
for both changed Python modules passed. The required full smoke/unit attempt
reached approximately 59 percent at its 600-second deadline; the supervisor
terminated its process group. It is not a passing full-tier result. The initial
two placement tests failed because they used an invalid private `seed` keyword;
they were corrected to `tie_rng` before the focused pass. An initial strict
import and nested-context lint error were corrected without suppressions.

The first PR CI run's CUDA-Q job failed before tests: its bounded dependency
preparation reached exit 124 while downloading CUDA-Q dependencies. This is a
dependency-preparation failure, not executed compiler-test evidence. Preserve
that failed attempt; no protected workflow or timeout is changed here. Full
required checks remain necessary before integration.

After withdrawal, all 234 focused tests pass; Ruff, Black and strict mypy pass
the retained test module. Archive reconstruction from saved payload chunks is
byte-identical to both original reports.
