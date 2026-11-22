# Bounded token-swapping layout restoration proposal

The existing native route restores every logical output to its own wire. Its
final permutation is currently priced using a tree walk or the reversed forward
SWAPs, whichever is shorter. This Compiler-owned `single_device_fast_path`
proposal tests whether shortest-path happy swap chains can use additional graph
edges to reduce that delivered work. Production does not call the candidate.

## Reviewed source and ownership

The prototype adapts the happy-cycle/unhappy-swap mechanism from
[`rustworkx-core/src/token_swapper.rs`](https://github.com/Qiskit/rustworkx/blob/69d412ad9b915d5f86608c15d6277c6d9195ee0a/rustworkx-core/src/token_swapper.rs),
revision `69d412ad9b915d5f86608c15d6277c6d9195ee0a`, SHA-256
`7caf7735be1ee49fc3d76f82f2aa2b0b5f3c7787d705e3d63593fe443ae30edd`.
The Apache-2.0 source cites Miltzow et al., *Approximation and Hardness for Token
Swapping* (2016). Its license SHA-256 is
`5df2a0d87d6c562f0ea11c688ac52532aa28d744cabc7994ff0537f64b3b3320`.
The prototype retains its license notice and identifies the adaptations.

FlagQuantum owns complete layouts, induced device topology, deterministic
traversal, bounded work and source anchors. No upstream graph object, random
trial pool, partial mapping or production dependency is introduced. Existing
distance and coupling implementations remain authoritative. No public API,
serialization, target, budget, native synthesis or independent replay changes.

For each physical wire the directed successor takes its token one hop closer
to home. A nontrivial cycle supplies a happy chain. A path ending at a settled
token supplies an unhappy swap. The candidate returns only a complete restored
permutation within the supplied SWAP budget; exhaustion returns no proposal.
Existing restoration remains the incumbent. The experiment calls the candidate
only for incumbents with at least two SWAPs, using `len(incumbent) - 1` as the
bound. It preserves the incumbent unless a strictly shorter complete restore
exists. Unused device wires cannot supply shortcuts. The surrounding router,
native legalization and independent evidence replay run unchanged.

## Fixed admission before application observations

The baseline is main `7d2e13c4160c0c1546d61a41551189fa11615465`.
SDKBench remains clean at `9aa9260920984e2ca05bcb293abb885fccd9c144`.
Use inspected nonce `01` repeated 32 times, widths 6/8/12,
QAOA/Ising/hopping/QFT/repetition and line/ring/grid/sparse: 60 pairs for each
ordinary-restored and target-artifact workflow, 120 pairs and 240 independent
acceptances. This is inspected development evidence, not held-out qualification.
Record every exception, unsupported case and delivered CZ/depth regression.

Admission requires no delivered CZ-count or two-body dependency-depth regression
on any pair and at least one strict quality improvement. Unchanged SDKBench
acceptance and resource limits apply to every execution. Use symmetric parsed
function replacements with a shared cached replacement-location list, unchanged
garbage collection, Python 3.12.14 and one numerical thread.

For target-artifact QFT6/grid, repetition6/ring and QFT12/grid retain nine
alternating timing samples per variant after two warmups. Separately trace
Python allocations around `adapter.execute` only; fixture preparation and
independent acceptance remain outside that stage. Require each candidate median
to be at most 110% of its baseline and each traced peak at most 105%. These
shared-host observations are not process RSS or qualified latency. Retain the
first complete run without threshold adjustment or favorable reruns. Do not
overlap measurement with local tests or artifact transfers. A 600-second
process-group supervisor owns timeout, TERM/KILL and child cleanup.

Before any production adoption, require exhaustive five-wire permutation
replay against independent breadth-first optimal distances, disconnected and
unused-wire checks, symbolic and annotated end-to-end workflows, and unchanged
native provenance/resource conformance. Finite tests do not prove global
quality, optimality or an approximation ratio on all graphs. If fixed admission
fails, retain the result and keep production unchanged. Formal dedicated-host
qualification remains deferred; no machine or service may be restarted.

## Existing supported workflow

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, compile

program = compile(fq.Circuit(4).h(0).cx(0, 3), coupling_map=CouplingMap.line(4))
assert program.metadata["routing"]["mapping_restored"] is True
```

Expected: legal edges and restored logical output placement. The prototype is
not enabled by this workflow or importing its module.
