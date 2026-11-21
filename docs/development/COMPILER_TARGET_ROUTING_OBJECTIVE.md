# Delivered target cost across restored routing strategies

Status: completed development inspection on 2026-11-22; exhaustive strategy
selection is not admitted into production. Compiler owns this
`single_device_fast_path` study. No production implementation, public API,
default, evidence limit or acceptance threshold changes here.

## Question and controlled variants

Automatic uncalibrated routing ranks restored SWAP count and instruction count.
The delivered target workflow also legalizes native gates, resynthesizes numeric
CZ blocks, builds physical evidence and serializes an executable artifact.
The [rejected depth experiment](https://github.com/flagos-ai/FlagQuantum/blob/3495f944c92948678d7bd86a31f76b3a445da8e0/docs/development/COMPILER_SABRE_DEPTH.md),
retained in the pending integration branch, demonstrates that a
routing-only improvement can regress the downstream CZ count. Would evaluating
the other existing deployable strategies solve this mismatch cheaply?

The complete production source is main
`2d78133a1f5a7d234acbb128724ccf2674e2eeb8`. Pending pipeline integration
PR #1268 and target-plan reuse PR #1269 are not included. The existing
[Qiskit/Cirq adaptations](COMPILER_UPSTREAM_REUSE.md) remain unchanged. This
inspection is a FlagQuantum control, not a new upstream implementation port.

For each input, the runner first compiles with `auto`, then compiles separately
with the other two strategies from `restore_after_each_gate`,
`persistent_layout` and `sabre`. The default observation supplies its selected
strategy's delivered result; it is not compiled a second time under its explicit
name. All variants require both initial and final identity placement and retain
the same native gate target and budgets. `sabre_layout` is excluded because its
nonidentity initial placement changes the comparison contract.

The diagnostic keeps the best successful complete output by
`(full_cz_count, two_body_dependency_depth)`, preserving the incumbent for an
exact tie. This is an exhaustive development inspection of three restored
strategies, not a proposed new public selector or physical duration objective.

## Independent acceptance and failures

The [runner](../../benchmarks/compiler_target_routing_objective.py) uses SDKBench
`9aa9260920984e2ca05bcb293abb885fccd9c144` fixtures and its unchanged independent
evaluator, targets and caps. A diagnostic wrapper varies the existing native
`routing_strategy` keyword at the artifact compiler boundary. The pinned SDKBench
checkout is unchanged, but this explicit override is **not its registered formal
participant workflow**. These are within-FlagQuantum controls, not cross-SDK
measurements or protocol-qualified results.

The [raw report](evidence/compiler-target-routing-objective.json) retains all
60 cases: five application families, widths 6/8/12, line/ring/grid/sparse
topologies, and inspected nonce `01` repeated 32 times. It records 180 attempts:

| Outcome | Count |
| --- | ---: |
| Complete compilation, serialization and independent acceptance | 165 |
| Compilation rejected by existing v4 record limits | 15 |
| Default automatic workflow accepted | 60 of 60 |
| Cases with all three strategies accepted | 51 |
| Cases with at least one failed alternative | 9 |
| Successful alternative with lower delivered CZ count or equal CZ count and lower depth than default | 0 |

Six-qubit acceptance compares complete operators up to global phase; widths
eight/twelve use three probes. The maximum reported semantic error over accepted
outputs is `6.072657351995283e-14`, below the unchanged `1e-8` threshold.
This is not universal operator proof for the larger widths.

Default selection chooses SABRE in 48 cases, per-gate restoration in nine, and
persistent layout in three. The default is the best delivered count/depth choice
among all three accepted strategies in the 51 complete portfolios. In the other
nine cases it is the best among successful outputs; failures do not supply an
alternative delivered circuit and cannot establish an unqualified optimum.

The failed explicit alternatives are:

| Case | Failed strategies |
| --- | --- |
| QFT 8, line | Per-gate restoration |
| QAOA 12, line | Per-gate restoration |
| Ising 12, line | Per-gate restoration; persistent layout |
| Ising 12, ring | Per-gate restoration; persistent layout |
| Hopping 12, grid | Per-gate restoration |
| QFT 12, line/ring/grid/sparse | Per-gate restoration; persistent layout |

Fourteen failures report `instructions exceeds record limit 4096`; persistent
layout for Ising 12 on a ring reports
`native_input_instructions exceeds record limit 4096`. These occur while building
the v4 physical evidence. A native/routing growth allowance of 20,000 does not
override the separately bounded serialized evidence arrays. The report retains
each exception type, message and traceback. No limit is raised and no alternate
evidence version is substituted to turn these failures into successes.

The first driver stopped on the QFT 8 line per-gate-restoration failure after
32 complete cases. The final driver retains each failed attempt and continues
independent controls, then exits nonzero after all 60 inputs are inspected.
The initial incomplete attempt is not passing evidence. The completed report's
`completed=true` means the inspection finished; its 15 failures still cause
exit status 1. Production source and acceptance thresholds were unchanged between
the attempts.

## Cost and decision

On the 51 portfolios with no failed variant, the sum of the three successful
compilation intervals divided by the default interval ranges from 2.87 to 7.47,
with median 4.55. Each interval includes the adapter's native compilation,
independent native evidence checks, artifact construction and serialization;
it excludes fixture setup, external independent evaluation and process startup.
The sum is an estimate of sequential compiler work, not a measured new integrated
selector. Failed attempts have no successful compilation interval, so the nine
incomplete portfolios are excluded from this ratio and their partial sums are
not treated as full costs.

These are single observations on one shared macOS CPU host with one numerical
thread, no warmups and fixed default-first order. Host drift and caches are not
controlled. The ratios are development observations, not stable latency,
significance, memory, hardware, scalability, held-out or SDK-ranking results.

Do not add this exhaustive strategy portfolio to the default target compiler:
the inspected successful outputs show no quality benefit, while complete
candidate compilation costs substantially more work and some variants cannot
produce an executable artifact. This finding does not prove routing and native
synthesis objectives are universally aligned. In particular, the earlier
same-SWAP depth counterexample varies routes *within* SABRE trials, whereas this
inspection varies the three deployable strategies. They answer different
questions.

The remaining work is to inspect bounded candidates within the existing SABRE
search and model their downstream CZ-block opportunities before expensive full
artifact construction, while retaining the incumbent, independent verification
and time/memory budgets. Any new selector still requires prospective evaluation;
this negative study does not justify fixture-specific exceptions. The v4 capacity
boundary also needs explicit product treatment, documented in
[known limitations](../reference/KNOWN_LIMITATIONS.md#target-compilation-evidence-record-limits),
rather than an unreviewed serialization-contract change.

## Reproduction and supported workflow

Use a disposable source checkout at the production commit above and an unchanged
pinned SDKBench checkout. Copy the runner to an external path, then on Linux run:

```bash
timeout --signal=TERM --kill-after=10s 600s env \
  PYTHONPATH=/path/to/fq-source \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python /path/to/compiler_target_routing_objective.py \
  --sdkbench-root /path/to/unchanged/sdkbench \
  --output /tmp/compiler-target-routing-objective.json
```

The retained execution used an external 600-second process-group watchdog.
A complete inspection has 60 rows, 180 attempts and `completed=true`; the
existing capacity failures deliberately result in nonzero exit. The report pins
the production commit, SDKBench commit and runner digest. Formal resource-qualified
SDKBench execution remains deferred. No machine reconfiguration or reboot occurs.

Normal users keep the existing circuit-to-target path and automatic strategy:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

circuit = fq.Circuit(4).h(0).cx(0, 3)
program = compile(circuit, coupling_map=CouplingMap.line(4))
assert program.metadata["routing"]["mapping_restored"] is True
print("Restored logical output layout")
```

Expected: the message prints. This ordinary workflow is unchanged; the diagnostic
examines the additional target-artifact stages rather than adding a user option.


## Synchronization after the pipeline integration

The measurements above remain bound to their original complete source
`2d78133a1f5a7d234acbb128724ccf2674e2eeb8`. PR #1268 subsequently merged as
main `a7d84a605080c511d4a07037ef2b4f3b90a645d1`, after 243 successful checks
and three optional skips. This study branch merged that exact main revision to
keep its integration checks current. The historical observations have not been
relabeled as measurements of the combined compiler.

The first quality run found the new record-limit note inside the generated
Known Limitations region. Commit
`a0a862f316ab97ec7e684792a31f247c04a70ca3` moved the same text outside it;
the generated region matches unchanged main. An initial coverage worker also
terminated after a five-minute timeout in the existing logical-cat notebook
fixture; that trace alone does not establish its root cause. A subsequent
supply-chain job failed because `pip-audit` timed out while querying PyPI after
60 seconds. That incomplete audit is not a passing security result. Original
runs and failures remain visible. No check, timeout, dependency restriction,
acceptance threshold or evidence requirement is weakened; the synchronized
head requires complete CI again.
