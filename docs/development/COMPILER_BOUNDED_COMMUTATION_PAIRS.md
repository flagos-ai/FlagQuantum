# Bounded commutation pair experiment

## Requirement and ownership

Repeated local optimization sweeps spend time deciding the same structural
commutation questions. The compiler owns this experiment on the
`single_device_fast_path`. It changes no public signature, result schema,
commutation rule, pass composition, routing policy or SDKBench acceptance rule.
The existing Qiskit-inspired block sweep remains authoritative. This bounded
memoization is a native implementation experiment, not a newly ported upstream
algorithm. Existing upstream attribution remains in place.

An untimed census at `50633d8b469969bbfd95bad94d38583cce95a1cd`
observed 34,009 ordinary and 10,037 target predicate queries over 120 complete
application workflows. A simulated 16-entry LRU table covered 16,284 and 5,031
queries respectively. Every predicate still executed; this is evidence of
repetition, not evidence of a speed improvement. The census uses previously
inspected inputs and is not held-out evidence.

## Candidate and invariants

The table is local to one `analyze_commutation` call and contains at most 16
ordered pairs. Each operand key contains canonical opcode, ordered wires,
matrix-presence and parameter-presence. The predicate reads no parameter or
matrix value. Positive and negative answers are cached. Hits refresh recency;
misses at capacity evict the least recent entry. No instruction, tensor or IR
is retained in the table, and no answer survives return or exception.

Representative grouping, first failing member, blocks, indices and logical pair
counters must remain exactly equal to the independent ungrouped sweep. Tests
include trainable batches, symbolic angles, overridden matrices, reversed wires,
more than 16 pairs, repeated barriers, mutable parameter dictionaries between
calls, and exception cleanup. No tensor value is read or detached.

## Prospective complete-workflow control

Freeze and publish the candidate and this protocol before the first resource
observation. Baseline is integrated main
`b6c7ef386955077f6ca91ae00565651d4aba6755`. Load its exact analysis source
alongside the candidate; replace all three existing analysis consumer bindings
for the baseline. Do not disable validation, independent replay or native target
legality checks. The candidate source remains immutable for the entire run.

Use unchanged SDKBench commit `9aa9260920984e2ca05bcb293abb885fccd9c144`,
its two FlagQuantum restored-layout workflows, seed 11, nonce `01` repeated
32 times and all 60 application inputs. The initial 120 candidate executions
must match every previously independently accepted output and audit hash and
pass the unchanged routing evaluator with identical metrics. A mismatch fails
the experiment; retain the failure. These are inspected regression controls,
not a new SDK comparison.

Then measure both complete workflows for 6-qubit grid QFT, 6-qubit ring
repetition and 12-qubit grid QFT. For each of the six cases perform two warmups,
nine retained alternating baseline/candidate rounds and one separately traced
round per variant. Verify all 24 outputs, audits and independent metrics per
case, including warmups and traced runs. Report all samples, medians and
`tracemalloc` peaks. Python peaks are not process RSS or cgroup memory evidence.

Admission requires all six candidate/baseline median ratios at most 1.05,
at least two of the three ordinary ratios at most 0.95, and every Python peak
at most 1.10 times baseline and at most 128 KiB above baseline. A failure
withdraws the production change; preserve the candidate patch and first report.
Do not rerun for a favorable result or change the thresholds after observation.

The first-run driver is `fq-bounded-pairs-control-20261122.py`, SHA-256
`a04f38964831dc854097870aa4ba446c552db419a2428635f50f6aa8d58544b3`.
The report embeds the complete driver and protocol. Run on the existing local
CPU environment with one numerical thread, bytecode disabled, an exclusive
output filename and a 600-second process-group timeout. This is development
resource evidence, not formal Linux qualification, latency ranking or proof of
an entangling-gate or depth improvement. No host is rebooted or upgraded.

## Developer workflow

```python
import flagquantum as fq
from flagquantum.compiler import optimize

circuit = fq.Circuit(2)
circuit.cx(0, 1)
circuit.rz(0, theta=0.2)
circuit.cx(0, 1)
result = optimize(circuit.to_ir())
```

The existing optimizer may cancel the CX pair across the commuting control
rotation. The candidate must produce the same optimized IR and analysis record.
Run `python -m pytest tests/unit/test_compiler_commutation_scaling.py
 tests/unit/test_compilation_commutation.py
 tests/unit/test_compilation_commutation_cancellation.py
 tests/unit/test_compiler_pass_manager.py -q` on one line for focused validation.


## Pre-measurement setup corrections

The first invocation stopped before creating a report because sparse checkout
had not materialized this protocol file. Materializing its unchanged committed
contents fixed that setup error. The next invocation accepted the first 40
quality observations, then stopped at width 8: the driver requested the absent
`ROUTE-LINE-08` bundle case. The existing SDKBench application driver uses
`ROUTE-GRID-08` for this width. No resource observations had begun.

The corrected driver uses precisely that existing case mapping, retains the
original partial report and writes a distinct output file. Candidate production
source, input nonce, metrics, thresholds, resource cases and sample counts are
unchanged. Publish this correction before the first resource observation.
The corrected driver is `fq-bounded-pairs-case-mapping-20261122.py`, SHA-256
`9109aaec86b223ee95b22bd625b2f73e9cc3411eee7cc598f6cd7239a9777bf6`.
The partial report and corrected complete report are separate evidence; the
interrupted attempt is not a complete passing run.


## First complete result: rejected

The corrected protocol ran at frozen commit
`b6c41299f201b657d10b650a28b001e4861a6330`. Its production source is
identical to candidate freeze `c9f2643985f80dec624b9e62f112dd6e9c45edc3`.
All 120 application observations matched the accepted reference payloads,
audits and independent evaluator metrics. All 144 resource-run outputs also
matched, including every warmup and separately traced result. The corrected
run finished normally under its process-group budget in 60.03 seconds.

| Case | Workflow | Candidate / baseline median | Baseline Python peak (bytes) | Candidate Python peak (bytes) |
| --- | --- | ---: | ---: | ---: |
| 06/qft/grid | Ordinary | 1.009826 | 472330 | 473143 |
| 06/qft/grid | Target artifact | 1.001638 | 2198318 | 2199038 |
| 06/repetition/ring | Ordinary | 0.996437 | 552637 | 553400 |
| 06/repetition/ring | Target artifact | 1.017851 | 2566909 | 2566909 |
| 12/qft/grid | Ordinary | 1.002263 | 2109392 | 2114108 |
| 12/qft/grid | Target artifact | 0.978462 | 10118879 | 10118779 |

All six regression and Python-memory gates passed, but **zero** ordinary cases
reached the required 5% improvement; admission required at least two. The
experiment is rejected. Do not infer a speedup, significant slowdown or causal
explanation from these small differences. Query hit rate did not establish a
complete-workflow benefit. No repeat run was used to seek a favorable result.

Production commutation source and its import-boundary test are restored byte
for byte to baseline. Five added scenario regressions remain: repeated barriers,
eviction pressure, mutations between calls, symbolic/batched/overridden gates,
and propagation of predicate failures. These compare behavior independently and
do not require memoization. No public API or numerical reference changed.

Evidence, retained as originally observed:

- [Complete resources and 120 quality observations](evidence/bounded-commutation-pairs-resources.json),
  SHA-256 `3c85fbd48cca4d08b58a74ad4dcce053fb74b01b6d48b1fab764c1262b323513`.
- [Interrupted 40-observation report](evidence/bounded-commutation-pairs-partial.json),
  SHA-256 `5c39ec55b2005892a1ad1bbfd3f11f33e2d19aac4a1e7fe13a0adfed4d4e2a16`.
- [Untimed query census](evidence/bounded-commutation-pairs-census.json),
  SHA-256 `598dcc762ed55992171dcaff7dec5dda3a0ab4758df1ec90c9426aefee915bc0`.
- [Exact withdrawn patch and setup failure records](evidence/bounded-commutation-pairs-pilot.json).

The reports embed the drivers and protocol. The archived patch includes the
candidate-specific predicate-count assertion; the retained regression instead
checks exact blocks, counters and first-failure behavior on the baseline source.
The initial setup failures and partial quality run remain failures, not passing
complete runs. These inspected controls establish no new held-out, SDK ranking,
formal Linux, RSS, CZ-count or depth-improvement claim.

Validation before withdrawal: 141 focused tests and 671 selected local
smoke/unit/integration tests passed; strict mypy, Ruff and Black passed for all
three touched Python files, and the executable optimizer golden path passed.
The first focused attempt retained one failure because the AST import allowlist
had not yet admitted the standard-library `collections` import. The corrected
candidate admitted that import explicitly; restoration returns the test and
source to their exact original contents. This does not weaken the simulation
layer boundary. Final withdrawal checks are recorded with the pull request.

After withdrawal, the same local tier again passed 671 tests. Strict mypy, Ruff
and Black passed for the three applicable Python modules. The archived patch
reconstructs all three candidate files byte for byte; the final production
module and its import-boundary test match baseline exactly. This selected
sparse-checkout tier is not a claim that the entire repository suite ran locally.
