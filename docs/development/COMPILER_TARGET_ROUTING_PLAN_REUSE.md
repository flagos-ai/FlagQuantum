# Selected routing plans in the target compiler

## Problem and implementation

The target topology legalizer selected an automatic route, discarded its winning
SABRE plan, then searched again to emit the same route. Ordinary compilation
already retains the plan within a call. This change retains the selected plan
through target emission using the existing selector and emitter. It changes no
SABRE ranking, trial count, gate optimization, target admission or public API.
It is classified as `single_device_fast_path`.

Target evidence requires a separate cold topology for emission. An initial
experiment using the ordinary automatic-routing coordinator directly produced
the same instructions but different path-cache metadata and legalization
identities. That approach was rejected. The implementation retains the cold
emission topology and passes only the selected plan into the existing emitter.
No metadata is fabricated or removed, and no global or inter-call cache is added.
Reuse is allowed only when the selected candidate's existing path-cache delta is
zero. SABRE's release valve can access paths during search; if those accesses
would contribute to cold emission evidence, the original replanning path remains
in use. This deliberately limits the optimization rather than changing evidence.
Explicit strategies, directed routing and operation budgets retain their existing
paths. A replaced router still receives the selected strategy and owns emission;
its exceptions still pass through the existing legalization error boundary.
Independent native-block and artifact verification remain enabled.

## Validation

The source implementation is
`7f83b8051d357310a36090fbc71f6a073e3809e9`, based on main
`2d78133a1f5a7d234acbb128724ccf2674e2eeb8`. It is independent of the pending
combined compiler integration in PR #1268; these observations do not measure
the two changes together.

- 209 focused tests passed across topology legalization, routing replacement,
  ordinary routing defaults, v4 evidence, physical evidence and target artifacts.
- 151 additional tests passed across directed topology, layout candidates,
  topology validation, target CZ synthesis and routing estimates.
- Complete routed JSON, result equality and legalization identity match explicit
  selected-strategy emission for line/star devices, cold/warm caller caches and
  successive changed/repeated programs. Caller caches remain untouched.
- Symbolic parameters retain object identity. A SABRE-selected tensor circuit
  preserves its parameter object, state and gradient against the original circuit.
- Search-count tests observe both selection and emission planner boundaries.
  Restoring the original production module makes the two line-device cases fail
  on duplicate planning; both star-device controls pass. The candidate passes all
  four. Tests do not count only one imported alias and miss the second search.
- A forced release-valve regression fails on the initial unconditional reuse
  implementation because serialized cache evidence differs. With the guard,
  complete JSON and legalization identity match the original explicit SABRE
  path, including its path-cache accesses. The final combined focused run passed
  all 360 tests.
- Public API snapshot, Black, Ruff and repository language checks passed.
  The first snapshot attempt lacked the tracked manifest in the sparse checkout;
  restoring it allowed the unchanged check to pass.

Full default/runtime tiers and complete-tree evidence gates remain required CI
checks. Focused local tests do not establish their completion.

## Independent application acceptance and local observations

The [runner](../../benchmarks/compiler_target_routing_plan_reuse.py) uses unchanged
SDKBench at `9aa9260920984e2ca05bcb293abb885fccd9c144` and its existing independent
evaluator and limits. The [raw report](evidence/compiler-target-routing-plan-reuse.json)
retains 60 paired target-artifact cases: five application families, widths
6/8/12, and line/ring/grid/sparse topologies. The inspected nonce is `01` repeated
32 times. All 120 executions passed, with identical normalized operation/layout
digests and evaluator metrics within every pair. Six-qubit checks use complete
operators up to global phase; widths eight/twelve use three probes.

The diagnostic control runs the same source while discarding the selected plan,
forcing the ordinary explicit-strategy router to search again. It reproduces the
redundant work; it is not a separate pristine baseline source process or another
SDK. The runner never changes the SDKBench adapter, reference or acceptance
thresholds. Clock-dependent target identities are not compared byte-for-byte
across separate artifact invocations; deterministic topology evidence is covered
by the focused tests above.

| Target-artifact workflow | Replanning median (ms) | Plan reuse median (ms) |
| --- | ---: | ---: |
| Ising 8 on a line | 205.426 | 205.153 |
| QFT 12 on a line | 875.764 | 866.745 |

Each workflow has two warmup pairs and 21 recorded pairs in alternating order.
The timer includes adapter compilation, native optimization, artifact construction
and serialization, and excludes independent evaluation, fixture setup and startup.
These are small observations on a shared macOS CPU host with one numerical
thread, not a stable speedup or significance result. A short additional local
test run overlapped the preliminary unguarded campaign; host load is not
controlled in either campaign. There is no
memory, held-out, hardware, SDK ranking or scalability conclusion. This change
removes proven duplicate search, but does not resolve broader target compilation
costs or establish parity with Qiskit/Cirq. Formal SDKBench qualification remains
deferred, and this work requires no machine reboot.

The [preliminary report](evidence/compiler-target-routing-plan-reuse-unguarded.json)
is retained separately at `cae2aaa17cdf798b7773bdafe143cba803481b18`. It passed the
same normalized-output acceptance but predates the release-valve regression and
guard, and cannot qualify the final implementation. Its medians were 206.772 /
206.146 ms for Ising 8 and 875.472 / 863.342 ms for QFT 12 (replan / reuse).
The final campaign was rerun because production code changed, not to replace a
disappointing timing outcome. Its timing is still only a development observation.

To reproduce in a development environment with the existing dependencies and an
unchanged pinned SDKBench checkout, run from this source tree on Linux:

```bash
timeout --signal=TERM --kill-after=10s 600s env \
  PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/compiler_target_routing_plan_reuse.py \
  --sdkbench-root /path/to/unchanged/sdkbench \
  --output /tmp/compiler-target-routing-plan-reuse.json
```

A complete report contains 120 application observations and 84 latency
observations. Interrupted checkpoints are not complete evidence. The retained
run used an external 600-second process-group watchdog with cleanup.

## Existing user workflow

Target compilation and offline deployment keep their existing entry points.
The existing offline example exercises local circuit export and target packaging:

```bash
python -m examples.synthesis.synthesis_offline_deployment
```

Expected output includes OpenQASM and
`DeploymentPackage(bell, target=local/offline-line, shots=256)` without credentials
or remote submission. The affected target-artifact path additionally routes a
program against a capability snapshot, optimizes native gates, builds physical
evidence and serializes the deployment artifact. Its complete supported workflow
is exercised without external hardware by:

```bash
python -m pytest -q tests/integration/test_native_block_physical_evidence.py
```

Expected: all 38 physical-evidence scenarios pass, including routed native
optimization, independent verification and rejection of modified evidence.


## Integration after the compiler pipeline merge

PR #1268 merged as main `a7d84a605080c511d4a07037ef2b4f3b90a645d1`
after 243 successful checks and three optional skips. This branch then merged
that exact main revision without conflicts. The combined source is
`9b9fa76a11632d740a6cbdac180da319b9002a50`; earlier independent observations above
are retained with their original scope and are not relabeled as combined runs.

The [combined raw report](evidence/compiler-target-routing-plan-reuse-integrated.json)
uses the unchanged runner and SDKBench pin above. All 60 application pairs pass
independent acceptance with identical normalized output digests and evaluator
metrics. It retains 120 application observations and 84 measured latency
observations. Local replanning/reuse medians are 193.470/193.225 ms for
Ising-8/line and 740.696/734.947 ms for QFT-12/line. They are shared-host
observations, not stable speedups; unrelated focused tests overlapped the
application phase. Numerical acceptance, resource caps and the deferred formal
qualification policy remain unchanged. No machine was reconfigured or rebooted.

The combined focused selection passes 926 tests with four optional tests skipped,
covering compiler routing, classical feedback, IR encoding, native optimization,
artifact evidence, directed topology and the CPU workflow. The first local
attempt lacked sparse-checkout test files and did not collect the intended
selection; restoring the actual tracked fixtures enabled the complete run.
Full repository gates must pass again on the synchronized PR head before merge.
