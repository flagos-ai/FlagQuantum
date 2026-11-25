# Commutation dispatch cost control

Owner: Compiler. Classification: `single_device_fast_path`.

## Requirement and change

The controlled-gate extension exposes additional analysis and fixed-point work.
Before adding a cache or changing those semantics, remove repeated dispatch work
from the existing predicate. `Instruction.__post_init__` already canonicalizes
every opcode. Read that authoritative name directly and resolve each operator
schema once per predicate call. Preserve the existing matrix-override refusal,
fixed-gate and single-generator checks, all structural rules and sweep counters.
Remove the three private helpers that repeatedly normalize and query the same
instructions. No parameter value is read or cached.

This candidate starts from integrated main
`46ba1523ed0c47fcf304d64d33b34d3f6c4b9629`. It does not include the pending
controlled-gate extension, commuting-frontier trial or rejected native-cost
selectors. That separates a general dispatch cost control from an extension's
additional optimization opportunities. It does not establish that the extension's
observed 20% resource regression has been solved.

No public signature, default, analysis result field, serialization, dependency or
pass declaration changes. No cross-call or within-call memoization is introduced.
This is a native implementation simplification, not a new upstream source port.
The Qiskit CommutationChecker documentation inspected on 2026-11-22 states that
its internal cache has been removed and its standard-gate commutation library
remains; this motivates evaluating dispatch before proposing another cache, not
claiming this code is copied from Qiskit.

## Existing supported workflow

```python
import flagquantum as fq
from flagquantum.compiler import compile

source = fq.Circuit(2).cx(0, 1).rz(0, theta=0.2).cx(0, 1)
compiled = compile(source)
assert [(gate.name, gate.wires) for gate in compiled] == [("rz", (0,))]
```

Expected: the same control-axis commuting cancellation as main. Existing complete
operator, symbolic-value, matrix-override, sweep-counter, cancellation, scaling,
optimization-level and native physical-evidence tests must remain unchanged.

## Frozen diagnostic and admission

Publish the implementation and this declaration before observations. Use unchanged
SDKBench `9aa9260920984e2ca05bcb293abb885fccd9c144`, seed 11, both complete
FlagQuantum workflows and all 60 application inputs: widths 6/8/12; QAOA, Ising,
hopping, QFT and repetition; line/ring/grid/sparse. Run one inspected draw with
nonce `01` repeated 32 times and one fresh 32-byte nonce generated after source
publication and frozen before either draw. There are 240 observations per draw.

The baseline uses main's complete commutation predicate and analysis functions,
including their captured consumer aliases. The candidate uses this dispatch
implementation. Use equal patch lifetimes, fresh adapters, unchanged configurations,
one numerical thread, 120-second adapter caps and the unchanged evaluator's
operation budgets and 1e-8 acceptance threshold. Width six has full operators;
widths eight/twelve have three probes, not general operator equivalence. Each
draw has a 900-second process-group budget and a 64 MiB raw-report disk cap.
Retain all inputs, normalized outputs, original compile-wall acceptance values,
audits, failures and source hashes. Alternate order by input. Every paired
normalized output, audit and returned acceptance metric must match exactly;
semantic tolerance alone cannot license an output change in this cost control.
Independently replay both archives in a fresh producer-free process.

After those draws, measure the fixed inspected QFT6/grid, repetition6/ring and
QFT12/grid inputs in both workflows. Each pair has two warmups and nine alternating
retained samples per variant with fresh adapters and equal patch lifetimes.
Measure adapter execution, then outside timing verify every timed output and
audit against its quality hash. Measure one additional Python allocation peak
per variant after garbage collection, excluding adapter creation from tracing,
and validate that traced output too. Independently accept the final timed output
per variant outside timing. Retain every sample and failure, with no rerun for a
better median. Freeze the driver hash before the first execution.

The declared development admission gate requires at least 5% median reduction
on all three ordinary complete workflows and no traced allocation-peak increase.
The three target-artifact controls must not exceed 1.05 times baseline median or
increase their traced peaks. Failure of any gate withdraws the executable pilot;
retain its exact source and observations instead of tuning or weakening gates.
These are fixed development controls on inspected inputs, not formal statistical
significance or a cross-SDK ranking.

Use a 240-second process-group resource budget, one numerical thread and no
simultaneous local tests. Shared macOS observations cannot establish exclusive
Linux core qualification, native allocation/RSS or cgroup peak memory. Formal
host qualification remains deferred. Do not restart machines or services.
