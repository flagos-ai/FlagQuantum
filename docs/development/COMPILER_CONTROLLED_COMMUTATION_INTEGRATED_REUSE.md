# Controlled commutation with integrated analysis reuse

## Requirement and candidate

Owner: Compiler. Classification: `single_device_fast_path`.

The first controlled-gate extension's resource observation recorded an ordinary
repetition6/ring median ratio of 1.2000 and incomplete per-sample validation.
That evidence remains in [the original results](COMPILER_CONTROLLED_COMMUTATION_RESULTS.md).
Its untimed census found six analysis calls instead of four and 610 predicate
calls instead of 396. Those observations justify examining redundant analysis;
they do not establish the cause or significance of the measured regression.

Integration PR #1306 passed all 246 exact-head checks and merged as
`d8941cea0950447ef4e3bb73ad42a5d5efe4b2ae`. It reuses one structural
analysis within an optimization call and clears it on scope exit, preserving
standalone calls, parameter mutation, failures, nesting and copied-context
lifetimes. It independently passed its declared resources and unchanged SDKBench
acceptance. It is not evidence about this controlled-gate extension.

Synchronize that main into the controlled-gate branch. Do not introduce a new
optimization, alter predicates, add a pair table, change pass composition or
weaken replay. The only executable difference from this baseline is `commute`.
The runner verifies equality of every other module AST node before execution.
The existing private analysis lifecycle and all native target verification run
in both variants. This is a different composed candidate and current-main
baseline, not a repeat of the earlier failed resource observation.

Before this experiment, 374 selected local unit/compiler/integration tests pass,
including the original complete-density thermal case and physical-evidence
checks. Strict Linux-targeted mypy, Ruff and Black pass for the complete
commutation/pipeline modules and the scoped-reuse/controlled-gate test modules.
The separate selection walkthrough still requires Integration PR #1308. Full
exact-head CI and the unresolved original resource evidence remain required.

## Prospective quality and resource protocol

Publish this declaration and candidate before any new observations. Freeze
source hashes, unchanged SDKBench commit
`9aa9260920984e2ca05bcb293abb885fccd9c144`, exact adapter options, versions,
seed 11, runner hashes and each input nonce. No SDK is given a special adapter,
acceptance threshold, backend, budget or failure exemption.

First run all 60 application inputs, two restored-layout workflows and both
variants with inspected nonce `01` repeated 32 times: 240 observations. Then
generate exactly one fresh 32-byte nonce after publication and freeze it before
the second complete 240-observation draw. Do not inspect, redraw, tune or omit
fresh cases. These controlled draws are not independently administered formal
qualification. Store every failure, output, audit, fixture hash and original
compile-wall acceptance value. Alternate variant order by input.

Run the unchanged evaluator on every output. It uses full operators for width
six and three explicitly limited probes for widths eight/twelve, the existing
1e-8 semantic threshold, topology checks, operation caps and 120-second compile
limits. Candidate CZ count and dependency depth may not exceed baseline in any
paired case. Report payload changes separately. Replay both complete reports
in a fresh process without importing FlagQuantum, Qiskit or Cirq.

For the inspected QFT6/grid, repetition6/ring and QFT12/grid inputs, measure both
complete workflows with fresh adapters and equal predicate-patch lifetimes.
Use two warmups, nine alternating retained rounds and one separate traced round
per variant. Independently accept and match the quality hash, metrics and audit
of **all 144** warmup/timed/traced executions outside timing. Save every sample
and output hash; do not check only the last result. Tracing excludes adapter
construction and covers complete adapter execution. Python allocation peaks do
not measure native allocations, RSS or cgroup memory.

Declare admission before observing these resources: each of six candidate/main
median ratios must be at most 1.05; every Python peak must be within both 1.10
times baseline and 128 KiB above baseline. All quality and replay requirements
must pass. This prospective gate governs the composed candidate; it does not
retroactively introduce a gate into the original resource report. If any gate
fails, keep the implementation unpromoted and retain all results. Do not rerun
for a favorable outcome, add allowances or remove a case after observation.

Use one numerical thread, no simultaneous local test suite, a 900-second
process-group budget per draw, 300 seconds for the resource control and
16 MiB per raw report. Output files are created exclusively. Keep raw reports
or exactly reconstructable hashed payload chunks. The source remains immutable
through observations. Formal Linux qualification stays deferred; no host,
service or kernel is restarted or upgraded. No global superiority, latency
ranking, new hardware support or scalability claim follows.

Frozen driver: `fq-controlled-integrated-reuse-20261122.py`, SHA-256
`b34774122ffac3161c24c131749091032ff2e2c71565dfdd5a8ae17bfa1585bc`.
Producer-free replay: `fq-controlled-integrated-reuse-replay-20261122.py`, SHA-256
`9657a3262a8e33b7c53a3ebb7705f7a897d37a2a8641c843ac4841672fcc281f`.
Both complete sources are embedded in their reports.

## Existing supported user workflow

```python
import flagquantum as fq
from flagquantum.compiler import compile

circuit = fq.Circuit(3).cx(0, 1).cx(0, 2).cx(0, 1)
result = compile(circuit)
assert [(gate.name, gate.wires) for gate in result] == [("cx", (0, 2))]
```

The candidate preserves this complete operator while removing the redundant CX
pair. The new experiment tests whether its wider structural coverage can meet
current-main resource bounds. It does not establish improved application routing
quality merely by passing a focused example.


## First observations and decision

The declaration and composed source were published at
`14b3ded9a058664a23c3ac93f1787d851cad3526` before execution. Both complete
240-observation draws pass unchanged acceptance, and a fresh process imports
no producer SDK and replays all 480 observations. The maximum residuals are
6.072657351995283e-14 and 6.753828751317917e-14 against unchanged 1e-8.
Each draw changes twelve ordinary repetition payloads; no paired CZ count or
dependency-depth regression occurs. All other payload pairs are identical.
These observations still do not improve the application entangler/depth gaps.

The sole resource control completes all six cases and independently checks
every warmup, timed and traced output against quality hashes, audits and metrics.
There are 144 checked executions. All Python-memory limits pass.

| Input | Workflow | Main median ms | Candidate median ms | Candidate/main | Main Python peak bytes | Candidate Python peak bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 06/qft/grid | ordinary-restored | 7.280 | 7.307 | 1.003732 | 479969 | 479719 |
| 06/qft/grid | target-artifact | 92.207 | 93.084 | 1.009512 | 2199032 | 2199016 |
| 06/repetition/ring | ordinary-restored | 6.225 | 7.234 | 1.162137 | 565486 | 575692 |
| 06/repetition/ring | target-artifact | 117.038 | 117.050 | 1.000097 | 2567203 | 2567253 |
| 12/qft/grid | ordinary-restored | 28.249 | 28.131 | 0.995835 | 2139919 | 2139241 |
| 12/qft/grid | target-artifact | 567.167 | 579.677 | 1.022057 | 10118164 | 10118214 |

**Reject resource admission.** Ordinary repetition6/ring has ratio 1.162137,
above the prospective 1.05 limit. Integrated analysis reuse alone does not
resolve the extension's resource regression. The original 1.2000 observation
and this different composed-candidate observation are both retained; do not
infer a significant improvement between measurements with different baselines.
No favorable repeat, changed threshold, case omission or source tuning occurred.
The implementation stays under draft review and must not be promoted on this
control's correctness success. A further change must address the remaining
work and receive its own prospective validation.

The [inspected index](evidence/controlled-integrated-inspected-screen.json) and
[fresh index](evidence/controlled-integrated-fresh-screen.json) retain every
observation and hashed payload chunks below the unchanged 1.8 MB artifact limit.
They reconstruct their complete original reports byte for byte, preserving
field order, source/nonce/version/configuration pins, failures and compile-wall
acceptance values. The [producer-free replay](evidence/controlled-integrated-replay.json)
records all 480 independently accepted outputs and its source. The
[complete resource report](evidence/controlled-integrated-resources.json) includes
all samples and their individual verification, full driver and frozen protocol.
Formal Linux qualification, process RSS, statistical significance, cross-SDK
ranking and global superiority remain unproven.
