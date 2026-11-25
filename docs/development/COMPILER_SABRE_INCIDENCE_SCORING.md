# SABRE endpoint incidence scoring

Status: prospective development control, not a comparative SDK result.
Owner: Compiler. Classification: `single_device_fast_path`.

The existing Qiskit-derived relative scoring implementation constructs a set
union and sorts interaction identifiers for each candidate SWAP. Store the
opposite endpoint and coefficient in each endpoint's incidence list instead.
On the planner's undirected topology, an interaction joining both swapped
endpoints has unchanged distance. Every remaining term needs only the distance
row of its moved endpoint. Repeated interactions retain separate contributions.
The existing absolute-score fallback, tolerance, ordering and tie RNG remain.
This refines FlagQuantum's existing implementation; it is not a new upstream
port. Existing Qiskit/Cirq notices remain with the source.

No public contract, dependency, routing policy, trial budget or restoration
behavior changes. Small devices retain their current full-rescoring path.
The six/eight/twelve-wire SDKBench application grid does not exercise this
scorer. A formal SDKBench campaign remains deferred; no host changes are needed.

## Frozen development protocol

Before observing application timings, fix baseline main
`7d2e13c4160c0c1546d61a41551189fa11615465`, widths 16/24/32 and four
families: Fourier controlled phases, parity ladders, seeded graph-cost layers,
and seeded matching layers. All use four-row grids. The generator and observers
are retained in `benchmarks/compiler_sabre_incidence.py`. These are transparent
development constructions, not held-out application or scientific accuracy
qualification. They represent routing demands from transforms, parity
evolution, graph optimization and layered circuits.

Both implementations use AST-cloned scorer functions and the same patch point.
Measure complete `route_to_topology(..., strategy="sabre")` calls, including
materialization and evidence. One warmup and seven retained alternating samples,
explicit GC before both variants, then one separate full-call traced peak each.
Retain every sample. Require identical routed IR content hashes on every call,
all correctness tests passing, median of the twelve per-case median time ratios at most 0.95,
no case median above 1.10, and no traced peak above 1.05 of baseline.
Resource budget: one CPU process, numerical threads one, 600-second group-owned
wall limit with TERM/KILL cleanup. Local observations are diagnostic: host
exclusivity, Linux memory enforcement and cross-SDK comparisons are not certified.
Reject rather than tune the fixtures or thresholds if these gates fail.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.compiler_sabre_incidence > /tmp/sabre-incidence.json
python -m pytest tests/unit/test_compilation_sabre_scoring.py -q
```

User workflow supported by this internal change:

```python
import flagquantum as fq
from flagquantum.compiler import CouplingMap, route_to_topology

source = fq.Circuit(16).h(0).cx(0, 15).cx(3, 12)
routed = route_to_topology(source, CouplingMap.grid(4, 4), strategy="sabre")
assert routed.n_wires == 16
```

The expected logical state, gradients, topology legality and restored output
layout are unchanged. Timing gains, if qualified, do not imply fewer CZs or
superiority over Qiskit/Cirq. This work does not resolve target-cost search.

The first harness invocation failed before any routing observation because its
IR generator supplied positional parameter tuples instead of the IR parameter
mapping. The corrected driver uses `theta` mappings; tasks, source scorer,
thresholds and observer protocol are unchanged. The failure log is retained
with the eventual observations.

A second setup invocation also stopped before routing: the generator used the
external spelling `cp` instead of the authoritative IR opcode `cphase`. The
corrected generator was validated for all twelve inputs before executing the
unchanged control protocol. Both setup failures remain in the evidence record.

The next invocation routed its first warmup input but stopped while reading its
content hash: the driver called the IR property as a method. No comparison or
retained timing row completed. This driver error is also retained; using the
property changes only observation plumbing.

## First complete observations (2026-11-22)

The published corrected driver at `e4df4fa5c20730f8ef38ddf1ada3c18d5f48a71e`
completed once within its 600-second budget. The prior setup failures are not
performance samples. Every timed and traced routed output matched its baseline
content hash. Seven retained samples per variant and all failure logs are in
[evidence/compiler-sabre-incidence.json](evidence/compiler-sabre-incidence.json).

| Family | Wires | Baseline median ms | Candidate median ms | Time ratio | Traced peak ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| fourier | 16 | 10.527 | 8.237 | 0.7825 | 1.0070 |
| parity_ladder | 16 | 2.757 | 2.426 | 0.8801 | 1.0148 |
| graph_cost | 16 | 12.731 | 9.947 | 0.7813 | 1.0052 |
| matching_layers | 16 | 6.579 | 5.254 | 0.7986 | 1.0101 |
| fourier | 24 | 37.209 | 28.725 | 0.7720 | 1.0000 |
| parity_ladder | 24 | 8.728 | 7.251 | 0.8308 | 1.0070 |
| graph_cost | 24 | 33.700 | 25.778 | 0.7649 | 1.0000 |
| matching_layers | 24 | 17.482 | 13.360 | 0.7642 | 1.0050 |
| fourier | 32 | 85.737 | 65.705 | 0.7664 | 1.0000 |
| parity_ladder | 32 | 18.004 | 14.824 | 0.8233 | 1.0000 |
| graph_cost | 32 | 70.235 | 54.489 | 0.7758 | 1.0000 |
| matching_layers | 32 | 44.388 | 34.534 | 0.7780 | 1.0033 |

Median of case time ratios: 0.7797. Worst traced peak ratio: 1.0148.
All declared performance gates passed on this local development observation.
The result measures complete explicit SABRE routing, including evidence and
materialization, rather than the whole automatic compiler or native lowering.
No dedicated-host, process-memory or inter-SDK performance claim is made.
Required regression tiers and exact-head CI remain integration conditions.
