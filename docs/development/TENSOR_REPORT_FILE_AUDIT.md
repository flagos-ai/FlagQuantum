# Tensor report source closure

## Complete source and native ownership

Q-0089 covers complete untouched Qualtran v0.7.0
`dev_tools/qualtran_dev_tools/tensor_report_card.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The acceptance test verifies its
original Git blob against the immutable manifest. Google copyright and the
Apache-2.0 source attribution remain in both native modules.

`tools/development/process_deadlines.py` owns queued process tasks, worker slots,
pipe result collection and per-task termination/cleanup. `_Pending` remains a
private lifecycle record. `tools/development/tensor_report.py` owns the original
report dictionary and stage timing. Callers supply the existing Compiler graph
selection/expansion and Simulation node construction. Width delegates directly
to `Simulation.tensor_network.contraction_width.tensor_contraction_width`.
The tool contains no tensor mathematics, alternative graph representation,
provider registry or physical lowering policy.

## Source comparison

Native and original real process queues compare results, submitted-kwargs
mutation and work counts with one and two workers. Both receive exactly one
complete payload through the `cxn` sending Connection. Numeric report fixtures
compare each existing native tensor against the untouched original network,
then compare exact report fields, insertion order, width and deterministic
stage timings. All three original failure stages preserve completed fields and
the exception message. Arrays, idle registers and empty shaped registers reuse
accepted typed-graph and Simulation tests. The worker integration uses actual
`expand_components_for_tensor`, existing node suppliers and pinned width.

Default platform spawn is exercised through an isolated guarded script. The
initial test runner lacked the required main-entry guard. That test harness was
corrected before acceptance. Explicit fork fixtures isolate source queue logic
from platform startup policy. No frozen source file is edited.

## Explicit process-lifecycle differences

The original empty queue busy-loops indefinitely. A zero-exit task without a
payload blocks its parent in recv because the parent's sending endpoint remains
open. Both defects are reproduced in isolated probes and terminated by the
outer test after a bounded wait. The original also waits for child exit before
receiving, allowing pipe-buffer-sized results to deadlock until timeout, and
terminates overdue workers without joining them.

The native tool closes the parent's sending endpoint immediately after startup,
drains a complete payload before child exit, and joins terminated workers. Empty
work raises StopIteration, missing payloads return None, a one-MiB result succeeds,
and exceptional consumer exit cleans running and queued work. Positive finite
budgets are required. Deadlines use a monotonic clock. A startup failure closes
its pipe and preserves the queued task for a subsequent corrected retry.
These lifecycle differences are intentional native fixes, separate from source
numeric equality. Result-order ties remain scheduler-dependent.

The protocol supports one complete Connection.send payload per task. Raw or
partial pipe framing is outside this developer-task protocol. Startup and parent
unpickling overhead are not strict real-time deadline guarantees. Native valid
result semantics are compared without reproducing the source hangs in-process.

## Resources and inherited boundaries

Durations are measured stage wall times. Width is the existing pinned Cotengra
shape-planner estimate, retaining the accepted single-leaf source boundary.
Neither is a primitive-gate count, an algorithmic resource model or measured RAM.
No new quantum stream is introduced. Complex tensor entries, complete dense
outputs, measurement histories and workspace evidence remain with accepted
Compiler/Simulation tests in the concentrated regression. Register ordering and
ownership are inherited from those existing MSB-first typed-graph suppliers.

No production dependency is added. Multiprocessing is Python standard library.
Cotengra and tensor-node dependencies remain isolated behind their existing
Simulation owner. The caller's flatten/build callbacks are the replacement
boundary, exercised without modifying the report consumer. This closure does
not certify physical support of arbitrary source components. Q-0937 generic
source testing and pending API proposals remain separately tracked.

## Acceptance evidence

The concentrated regression passes 266 cases with zero skips. It covers report
and process behavior, source probes, Compiler expansion, native tensor nodes,
dense and mixed results, tensor providers, width policy, register binding,
component reports and developer subprocess helpers. Strict typing passes three
files. Black, Ruff, Integration scope, architecture, internal IR, protected API
and diff checks pass. Local evidence is in
`ftqc-reference-env/g2-tensor-report-*-20261121.log` outside the repository.
