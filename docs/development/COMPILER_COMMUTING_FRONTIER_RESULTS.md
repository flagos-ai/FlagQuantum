# Commuting-frontier development results

Status: quality diagnostic complete; production admission unproven. Owner:
Compiler. Classification: `single_device_fast_path`.

## Frozen control and acceptance

The [declaration](COMPILER_COMMUTING_FRONTIER.md) and implementation were published
at `4f8bc5388d3ac60daca80da1286265a6fdee35c4` before observation. Its production
source equals integrated main `46ba1523ed0c47fcf304d64d33b34d3f6c4b9629`.
The experiment reuses the existing structural commutation checker and native
SABRE planner; it is not a new upstream code port or an external SDK invocation.

The unchanged SDKBench harness is
`9aa9260920984e2ca05bcb293abb885fccd9c144`. All 60 declared application inputs
ran in both ordinary-restored and target-artifact workflows, each with baseline
and candidate, seed 11 and the previously inspected nonce. All 240 observations
pass unchanged independent acceptance. Eighty observations use complete operators
at width six; 160 use three probes at widths eight/twelve. Those probes do not
establish universal operator equivalence.

The [raw index](evidence/commuting-frontier-screen.json) preserves every input,
row, configuration, compile-wall acceptance value, adapter audit, source hash
and driver. Four hashed payload files retain all 146 distinct normalized outputs.
They reconstruct the original 5,252,058-byte report exactly; no output is dropped
because it repeats. This is one seed and inspected development input, not
independent process replication, held-out qualification or a cross-SDK ranking.

## Paired quality and counterexamples

The [complete analysis](evidence/commuting-frontier-analysis.json) compares CZ
count and two-body dependency depth separately. Dominance requires neither metric
to worsen and at least one to improve; tradeoffs are not hidden in a weighted
score.

| Workflow, 60 inputs each | Candidate dominates | Equal | Candidate dominated | Tradeoff |
| --- | ---: | ---: | ---: | ---: |
| Ordinary restored | 18 | 38 | 0 | 4 |
| Target artifact | 17 | 38 | 0 | 5 |

For example, target-artifact QAOA6/line changes from 70 CZs/depth 41 to 54/38,
QFT6/grid from 65/48 to 53/31, and Ising12/line from 284/146 to 230/105.
No hopping or repetition input changes delivered CZ count or dependency depth
in this draw. A reduced routing cost does not guarantee improved native output:
the complete set of tradeoffs is below.

| Workflow | Input | Baseline CZ/depth | Candidate CZ/depth |
| --- | --- | ---: | ---: |
| Ordinary restored | QFT8/sparse | 182/105 | 164/114 |
| Ordinary restored | QAOA12/line | 270/101 | 252/114 |
| Ordinary restored | QFT12/grid | 408/191 | 330/206 |
| Ordinary restored | QFT12/sparse | 372/167 | 336/206 |
| Target artifact | QFT6/line | 75/52 | 77/50 |
| Target artifact | QFT8/grid | 162/97 | 142/104 |
| Target artifact | QFT8/sparse | 170/96 | 154/106 |
| Target artifact | QFT12/grid | 352/150 | 328/170 |
| Target artifact | QFT12/sparse | 338/158 | 330/190 |

These are complete delivered workflows, not identical intermediate stages.
Native target block boundaries and post-route synthesis can change with gate
order. The first experimental selector accepts strictly lower delivered SWAP
cost only; it is not a Pareto guard on delivered CZ count and depth. Keep it
experimental until quality selection, compilation resources and prospective
validation support a production decision. No comparator is constrained to
match FlagQuantum's missing functionality.

## Resource follow-up declaration

This follow-up is frozen after the complete inspected quality screen and before
resource observation. Preserve the exact experiment from the published freeze;
do not change the run bound, search options, seed or cases after measurement.
Use QFT6/grid, repetition6/ring and QFT12/grid, both existing workflows, baseline
and candidate, nonce `01` repeated 32 times, seed 11. These are the existing
inspected controls, not a new unseen draw.

Each of six case/workflow pairs has two warmups and nine alternating retained
samples per variant. Use equal selector-patch lifetimes and fresh adapters.
Measure `adapter.execute` wall time outside independent acceptance. Record one
separate traced Python-allocation peak per variant after garbage collection,
keeping adapter creation outside that tracing window. Verify final outputs and
layouts against the quality archive and replay independent acceptance outside
the timing window. Save every raw sample and failure; do not rerun to improve a
median or discard an adverse case.

Run one numerical thread, no simultaneous local test suite, existing adapter
120-second deadlines and operation caps, under a 240-second process-group
supervisor. The resource driver is frozen by source hash before execution. The
shared host is unqualified: these are development observations, not cgroup/RSS
measurements, significance, hardware results or latency ranking. No resource
admission threshold is inferred after observing these samples; formal Linux
qualification remains deferred by the user.

## Independent archive replay

A fresh process reconstructed the exact original report and independently
accepted all 240 rows without importing FlagQuantum, Qiskit or Cirq. The
[replay proof](evidence/commuting-frontier-replay.json) retains the replay source,
archive hashes, case completeness checks and exact acceptance results. Maximum
observed semantic residual is 6.918428159859188e-14 against the unchanged 1e-8
threshold. Eighty complete operators and 160 three-probe checks were replayed;
the latter retain their limited coverage.

## Resource observations

The first declared resource run completed within its 240-second supervision
budget, without timeout. The [complete report](evidence/commuting-frontier-resources.json)
preserves every retained sample, the driver and the declaration snapshot above.
All six final output pairs match the frozen quality hashes and adapter audits,
restore output placement, and pass independent acceptance outside timing.

| Input | Workflow | Baseline median ms | Candidate median ms | Candidate/baseline | Baseline Python peak bytes | Candidate Python peak bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 06/qft/grid | ordinary-restored | 7.742 | 6.889 | 0.8898 | 472029 | 448963 |
| 06/qft/grid | target-artifact | 90.053 | 74.000 | 0.8217 | 2198358 | 1822502 |
| 06/repetition/ring | ordinary-restored | 6.343 | 6.531 | 1.0297 | 552669 | 556552 |
| 06/repetition/ring | target-artifact | 115.034 | 116.894 | 1.0162 | 2566909 | 2575059 |
| 12/qft/grid | ordinary-restored | 29.321 | 29.602 | 1.0096 | 2111544 | 1796496 |
| 12/qft/grid | target-artifact | 559.044 | 509.653 | 0.9117 | 10118829 | 8723832 |

These observations include regressions. The unchanged repetition outputs cost
about 3.0% and 1.6% more median execution time, with small traced-peak increases.
QFT12/grid ordinary execution also costs about 1.0% more median time, despite
fewer CZs and a lower traced peak. QFT6/grid and QFT12/grid target execution have
lower observed medians, but the QFT12 outputs still have worse dependency depth.
No significance test or resource admission gate was declared, and nine samples
on this shared host do not establish a latency ranking. Python allocation peaks
exclude native allocations and are not peak RSS or cgroup memory evidence.

## Decision and next requirement

Retain the bounded selector as an explicit development experiment. Do not change
production defaults or declare cross-SDK superiority. The complete inspected
screen supports investigating commutation-aware frontier scheduling, while its
nine quality tradeoffs and resource regressions rule out an unconditional
admission based only on SWAP cost. A subsequent, separately declared candidate
must evaluate delivered CZ count and dependency depth, preserve incumbent
behavior when these conflict, and quantify the extra selection work before
prospective validation. These measurements cannot be relabeled as held-out
results or formal Linux qualification.
