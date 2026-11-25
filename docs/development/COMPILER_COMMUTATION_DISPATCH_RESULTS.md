# Commutation dispatch cost-control results

Owner: Compiler. Classification: `single_device_fast_path`.

## Decision

Withdraw the executable candidate. The first resource run fails the
[predeclared admission gate](COMPILER_COMMUTATION_DISPATCH.md). Production
commutation code is restored byte-for-byte to integrated main
`46ba1523ed0c47fcf304d64d33b34d3f6c4b9629`. Retain the
[exact candidate patch](evidence/commutation-dispatch-pilot.json), published
implementation `c7cb03e04cf659de46f0675e5de6e093cf09781f`, complete observations
and failures. Do not tune the candidate, loosen the gate or rerun for a better
median in this experiment. This experiment does not establish that reducing
redundant dispatch alone resolves the controlled-commutation overhead.

## Complete unchanged-output checks

The inspected draw and one fresh draw were frozen after source publication,
before either execution. Each covers all 60 application inputs, both complete
workflows and both variants at compiler seed 11. All 480 observations pass the
unchanged SDKBench `9aa9260920984e2ca05bcb293abb885fccd9c144` evaluator. Every
paired normalized output, audit and acceptance metric matches exactly. This
includes 160 full-operator checks at width six and 320 three-probe checks at
widths eight/twelve. Probe acceptance does not prove arbitrary-input equivalence.
CZ count and dependency depth are unchanged throughout.

The [inspected archive](evidence/commutation-dispatch-inspected-screen.json) and
[fresh archive](evidence/commutation-dispatch-fresh-screen.json) retain all rows,
fixtures by nonce and hash, original acceptance compile times, source/configuration
pins, driver source and three hashed payload chunks each. Reconstruct the
original reports in their recorded field order; their exact sizes are 3,708,330
and 3,673,662 bytes. Both reports are independently replayed in a fresh process
without importing any producer SDK; see the
[replay record](evidence/commutation-dispatch-replay.json).

The baseline substitutes main's entire analysis implementation and its captured
consumer aliases, not just its pair predicate. Candidate and baseline use equal
patch lifetimes and fresh adapters. No cross-call cache or new commutation rule
is involved. This is an isolated dispatch refactor, separate from the pending
controlled-gate rule extension.

## First resource observations

The [frozen resource driver](evidence/commutation-dispatch-resources-manifest.json)
was hashed before execution. The [resource record](evidence/commutation-dispatch-resources.json)
retains the complete driver, unchanged protocol, all samples and every sample's
output/audit verification. All six declared pairs completed once, with two
warmups and nine alternating retained samples per variant. There are 144 checked
execution payloads: 24 warmup, 108 timed and 12 traced. Each matches its frozen
quality output and audit outside timing/tracing. The final timed payload per
variant also passes independent acceptance with the exact prior metrics.

| Input | Workflow | Baseline median ms | Candidate median ms | Candidate/baseline | Baseline Python peak bytes | Candidate Python peak bytes | Frozen gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 06/qft/grid | ordinary-restored | 7.650 | 7.269 | 0.950281 | 472055 | 471995 | Fail |
| 06/qft/grid | target-artifact | 218.670 | 219.336 | 1.003048 | 2198358 | 2198242 | Pass |
| 06/repetition/ring | ordinary-restored | 6.339 | 6.157 | 0.971277 | 552542 | 552384 | Fail |
| 06/repetition/ring | target-artifact | 116.349 | 116.962 | 1.005266 | 2566859 | 2566859 | Pass |
| 12/qft/grid | ordinary-restored | 29.711 | 28.089 | 0.945402 | 2113665 | 2114055 | Fail |
| 12/qft/grid | target-artifact | 567.317 | 567.965 | 1.001143 | 10118879 | 10118879 | Pass |

Ordinary QFT6/grid narrowly misses the required ratio of at most 0.95;
ordinary repetition6/ring also misses it. Ordinary QFT12/grid meets the time
threshold but increases the traced Python peak by 390 bytes, failing the
separate no-increase condition. All target-artifact controls pass. No margin is
added after observation. Even a small threshold miss is retained as a failure
under this declaration.

These measurements use one numerical thread on a shared macOS host. Python
allocation peaks cover complete adapter execution, excluding adapter creation;
they are not native allocations, peak RSS or cgroup peak memory. Nine shared-host
samples do not establish significance, broad speedup, a cross-SDK ranking or
formal Linux qualification. No machine or service was restarted.

## Verification and existing workflow

Before the experiment, 135 focused commutation, cancellation, scaling,
optimization-level and physical-evidence tests passed unchanged; the sparse
checkout's default selection passed 227 tests with 42 deselected. Ruff, Black
and strict Linux-targeted mypy passed for the complete candidate module. These
are scoped checks, not the complete repository suite. Withdrawal restores the
production source exactly; no public signature, default, schema or dependency
changes remain.

The existing user workflow remains:

```python
import flagquantum as fq
from flagquantum.compiler import compile

source = fq.Circuit(2).cx(0, 1).rz(0, theta=0.2).cx(0, 1)
compiled = compile(source)
assert [(gate.name, gate.wires) for gate in compiled] == [("rz", (0,))]
```

The retained evidence supports a negative development decision, not admission of
the withdrawn implementation or completion of the larger compiler objective.


The unified diff is encoded inside JSON so its required blank-context spaces
survive the repository trailing-whitespace hook. Decoding its `patch` field
reconstructs the exact original bytes and SHA-256; applying it to the pinned
baseline passes `git apply --check`. This storage correction changes no frozen
source, raw observation or admission decision.
