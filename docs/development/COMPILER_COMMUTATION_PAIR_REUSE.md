# Bounded structural commutation reuse

## Problem and implementation boundary

Local development profiles of the already inspected DNN and QAOA programs put
most ordinary-workflow time in the optimization stages before and after routing.
For DNN line, one instrumented call made 5348 structural `commute` evaluations;
QAOA line made 1398. These profiles are shared-macOS diagnostics, not causal
attribution of the qualified Linux performance gap.

The existing sweep already groups equivalent members inside each open commuting
block. It nevertheless asks identical structural questions again after a block
closes and when another wire reaches the same pair. The compiler now retains up
to 512 ordered pair verdicts during one sweep. A class consists of canonical
opcode, exact ordered wires, presence of a matrix override and whether the
parameter dictionary is empty. Both positive and negative verdicts are reusable.
The existing predicate reads no parameter values, matrix contents or metadata.

A miss still calls the same predicate. Once the table is full, new pairs are
computed without insertion. The table is released at return. There is no global
program cache, numeric approximation, new API, changed optimization sequence or
new runtime dependency. Existing logical query counters still count all logical
comparisons, including members represented by an equivalent class. They do not
become counts of actual predicate calls.

## Reviewed upstream mechanisms

The review used pinned Qiskit 2.6.2 primary sources:

- [Commutation analysis](https://github.com/Qiskit/qiskit/blob/2.6.2/crates/transpiler/src/passes/commutation_analysis.rs)
  grows wire-local sets by checking candidate members against the current set.
- [Commutation checker and library](https://github.com/Qiskit/qiskit/blob/2.6.2/crates/transpiler/src/commutation_checker.rs)
  separates prechecks, reusable gate-pair/relative-placement facts and matrix
  fallback. The structural-library concept motivates avoiding repeated questions.

This is a FlagQuantum implementation over its existing exact structural rule
source, not a transcription of Qiskit's Rust code or a port of its numerical
fallback. It retains absolute wire identities and a sweep-local bounded table.
No external source body or new dependency is incorporated. Existing upstream
attribution remains intact. The existing
[upstream reuse inventory](COMPILER_UPSTREAM_REUSE.md) and native pass manager
remain authoritative; no second pass manager or registry is introduced.

## Correctness and resources

`Instruction` parameter dictionaries and matrix operands can be mutable. They
never enter the verdict table: its keys contain only immutable names, wire tuples
and booleans. Each sweep rebuilds its classes, and the table never outlives that
sweep. A later call therefore observes a changed matrix-override flag or changed
parameter presence. Numerical and trainable values remain owned by their
existing optimization and refusal paths. Parameter values do not determine this
structural predicate's answer.

There are at most 512 retained verdicts. Class construction additionally uses one
linear tuple over the input instructions; the existing partition and position
index are also linear in instruction/wire incidences. This is a bounded-entry
cache, not a constant-byte memory claim: wire tuples and opcode strings are part
of the supplied IR. A workload with many distinct pairs can obtain little reuse
and still pays key construction and lookup costs. No production memory or
speedup guarantee is inferred from the entry cap.

Tests compare cache-disabled, saturated and normal sweeps on seeded programs;
retain partitions and logical counters; exercise reversed operand order,
trainable parameters, matrix overrides, mutable parameter presence and predicate
replacement between calls. Existing dense-matrix rule tests and application
integration tests remain unchanged.

## Development measurements and rejected alternatives

All six external tasks are now inspected development inputs. Nine alternating
baseline/candidate pairs per task compared the same ordinary workflow on shared
macOS, with one warmup and exact output-list/layout equality. The baseline sweep
was loaded directly from the unchanged branch-parent source; only the private
sweep function was replaced. This does not constitute a new prospective or
qualified Linux comparison.

| Inspected task | Candidate / baseline median time |
| --- | ---: |
| DNN grid, eight wires | 0.980 |
| DNN line, eight wires | 0.883 |
| QAOA grid, six wires | 0.958 |
| QAOA line, six wires | 0.963 |
| Simon grid, six wires | 0.975 |
| Simon line, six wires | 0.987 |

The [complete local evidence](evidence/compiler-commutation-pair-reuse.json)
retains every sample and exact driver, including both rejected alternatives.
Instrumented ordinary-workflow predicate calls fell from 5348 to 609 on DNN
line and from 1398 to 475 on QAOA line. Logical pair counters remain unchanged.
These are operation counts, not timing measurements.

Sixteen additional seeded controls cover 2, 4, 8 and 12 wires with 10, 50, 200
and 800 source operations. Five alternating pairs per case retained exact output
and layout equality. Their median time ratios ranged from 0.939 to 1.075,
including slower cases. These controls overlapped full local tests and cannot
establish a universal speedup. All raw samples and the seed/driver are retained.

Three alternating allocation probes per external task measured Python
`tracemalloc` peaks around the ordinary workflow. Median candidate/baseline
ratios ranged from 0.9987 to 1.0082, including increases. These are Python-traced
allocations only: they exclude native tensor memory, process RSS and cgroup
accounting, and do not establish a production memory bound. Their complete
samples and driver are included in the same evidence file.

These short observations retain shared-host noise and no confidence interval.
They do not establish an SDK ranking, eliminate the much larger Qiskit gap or
prove a production improvement. Source normalization, synthesis, legalization
and artifact serialization remain separate optimization targets. The protected
source-artifact entry limit is unchanged.

Two preceding local probes were rejected: fixed-gate matrix reuse and skipping
known nonidentity fixed singleton blocks did not produce consistent workflow
improvements. Neither prototype was applied to production. Do not select the
best local sample as new qualification evidence.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

source = fq.Circuit(4).h(0).rz(0, 0.3).cx(0, 3)
compiled = compile(source, coupling_map=CouplingMap.line(4))
assert compiled.metadata["routing"]["mapping_restored"] is True
```

This existing CPU workflow retains its public signature, layout obligation and
numerical semantics. Path classification is `single_device_fast_path`; the
change makes no distributed, accelerator or scalability claim.

## Verification

The focused compiler checks passed 229 tests, including the new reuse scenarios,
existing rule/cancellation/scaling checks, pass-manager tests, one/two-qubit
optimization and ordinary/CZ application integration. The public API migration
baseline, language and documentation-link checks passed.

The complete local default tier ran to completion: 21540 passed, 796 skipped,
9 failed and 21 setup errors in 1082.29 seconds. Failures include transient disk
exhaustion, sandbox loopback-socket denial, a nested pytest inheriting the
initial `PYTEST_ADDOPTS` temporary-directory option, and 100 local provenance
origin-declaration consistency errors. This is not a passing default tier.
After correcting the temporary-directory environment and permitting local
sockets, all 143 tests in the affected transfer, coverage-plan, launch-plan and
MPS-accounting files passed. The original failures remain recorded. The new
evidence artifact's revision and field-name checks pass independently; existing
origin declarations and historical evidence were not modified.

The full local integration tier timed out after 900.338 seconds at approximately
15% progress (exit -15). It is incomplete, not a pass; the recorded process group
and child processes were confirmed absent after termination. Exact-head CI is
required before readiness. Its first attempt exposed an unsupported SDKBench
origin label in this new evidence file, corrected to `producing_host_history`
with the separate checkout identified explicitly. Raw numerical samples are
unchanged. Performance
qualification still requires a separately admitted host and the unchanged
independent SDKBench protocol.
