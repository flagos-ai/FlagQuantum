# Compilation-scoped commutation analysis reuse

Owner: Compiler. Classification: `single_device_fast_path`.

## Verified requirement

An untimed census of all 60 inspected application inputs and both complete
workflows finds 719 commutation-analysis calls. In the ordinary workflow,
144 of 288 calls repeat the immediately previous structural input within the
same fixed-point optimization call. The target-artifact workflow has no such
repeat among its 431 calls. Every census output and audit matches the prior
unchanged-main quality archive. See
[evidence](evidence/commutation-duplicate-census.json). These are work counts,
not a resource result or a universal proportion for quantum programs.

Reuse the existing parameter-blind analysis inside the native optimization
pipeline. Retain one last result per fixed-point call, keyed by every opcode,
ordered qubit tuple, matrix-override presence and parameter-map nonemptiness.
These are precisely the instruction properties the current structural checker
reads; it never reads angle values, matrix entries, metadata or autograd tensors.
Rebuild the key on every call because frozen instructions contain mutable maps.
Also compare checker identity so instrument substitution cannot inherit the old
checker's facts. Changes in any structural key recompute the whole analysis.
Counters and partitions remain exact. No result is reused across optimization
calls, and standalone calls still recompute independently.

The private context manager owns one ContextVar slot rather than adding a new
public analysis contract or altering the generic PassManager. Nested calls get
fresh slots. Exiting clears and deactivates the slot, including exception exits;
a copied context cannot keep it active afterwards. Threads have isolated contexts.
The slot retains no input IR, parameter tensor or matrix; only a structural tuple
and the existing immutable analysis. Release an old result before its replacement
is allocated. Storage is O(instruction count plus analysis size), bounded to one
entry rather than an unbounded key table. This trial does not cache compilation
outputs, bypass independent replay or install the pending controlled-gate rules.

The existing per-qubit analysis follows the documented Qiskit CommutationAnalysis
pattern. This is a native lifecycle improvement, not copied upstream source.
No public signature, default, stage order, result/schema, dependency, capability
maturity or layout contract changes. The generic pass manager keeps its existing
program-dependent invalidation rules.

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler import compile

source = fq.Circuit(2).cx(0, 1).rz(0, theta=0.2).cx(0, 1)
compiled = compile(source)
assert [(gate.name, gate.wires) for gate in compiled] == [("rz", (0,))]
```

Expected: unchanged complete operator and output. Tests also compare a full
compiled workflow with an uncached replacement, preserving symbolic and batched
parameter ownership, matrix overrides, structural mutations, checker replacement,
nesting, expired copied contexts, exceptions, threads and input tensor release.

## Frozen application and resource protocol

Publish the source and this declaration before quality or resource observations.
Baseline is integrated main `5b9cf3878664393e8667ca175ec8669f37fb3bc9`.
Use unchanged SDKBench `9aa9260920984e2ca05bcb293abb885fccd9c144`, seed 11,
all 60 application inputs, both complete workflows and both variants per draw.
Use one inspected nonce (`01` repeated 32 times) and one fresh 32-byte nonce
created after publication and frozen before either execution. Each draw contains
240 observations. The baseline replaces only the new private optimization scope
with a no-op context manager; equal patch lifetimes and fresh adapters apply.
All other production code is identical. Record all failures and unsupported cases.
Paired normalized output, audit and acceptance metrics must match exactly.
Retain original compile-wall acceptance values and producer-free replay all rows.
Width six uses full operators; widths eight/twelve use three probes, not proof
for arbitrary inputs. Keep the unchanged 1e-8 threshold, operation budgets,
120-second adapter deadlines, 900-second process-group budgets and 64 MiB raw cap.

Resource follow-up uses inspected QFT6/grid, repetition6/ring and QFT12/grid in
both complete workflows. Use two warmups and nine alternating retained samples
per variant, fresh adapters, equal patch lifetimes and one numerical thread.
Measure adapter execution only. Outside timing, check every warmup, timed and
traced payload/audit against quality hashes; independently accept the final timed
payload per variant. Measure one additional traced Python peak per variant after
garbage collection, excluding adapter creation. Retain all samples and failures.
Freeze the driver hash before its first and only run, under a 240-second process
group supervisor. No simultaneous local tests, favorable rerun or case removal.

The development admission gate requires a median ratio at most 0.95 on at least
two of the three ordinary workflows, and no ratio above 1.05 on any ordinary or
target-artifact control. Every traced peak must remain within both 1.10 times
baseline and 128 KiB above baseline. The explicit allocation allowance tests the
bounded one-entry storage tradeoff; it does not exempt memory measurement.
Failure withdraws the executable pilot and retains its exact patch and evidence.
Do not tune this candidate or revise gates after observations.

Shared macOS timing and Python allocation peaks are development diagnostics,
not significance, a cross-SDK ranking, peak RSS/native allocation or formal Linux
qualification. The exclusive Linux campaign remains deferred. No machine or
service may be restarted, and no qualification requirement is weakened.
