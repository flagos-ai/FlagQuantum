# Phase 6 PyTorch graph-compilation evidence

Date: 2026-09-09

This report records a correctness-oriented local baseline for the bounded
Phase 6 hybrid quantum region. It is not a performance claim and must not be
used to infer accelerator, distributed, capacity, or production throughput.

## Profile

- macOS 26.0, arm64 CPU
- PyTorch 2.13.0
- four-wire statevector, `float64` parameters, `complex128` state
- eight scalar parameter slots and two unit single-wire Pauli-Z terms
- one positive/negative mixed selected branch
- AOT Eager backend with `fullgraph=True`

## Correctness gates

| Gate | Result |
| --- | --- |
| PyTorch operator schema and mutation check | pass |
| FakeTensor contract | pass |
| Autograd registration | pass |
| AOT dynamic dispatch | pass |
| Central-difference `gradcheck` | pass |
| Eager/compiled forward parity | pass |
| Eager/compiled first-order gradient parity | pass |
| Supported region captured without a graph break | pass |

`torch.library.opcheck` supplied the first four checks. Successful execution of
the operator with `torch.compile(..., backend="aot_eager", fullgraph=True)`
supplied the graph-break gate. The measurement run reported one unique Dynamo
graph containing two captured calls.

## Indicative local timings

Twenty iterations were measured after short warm-up unless marked as a cold
measurement. Values are medians and include Python orchestration. The compiled
step includes the externally requested first-order gradient.

| Measurement | Observed value |
| --- | ---: |
| Source capture | 0.285 ms |
| Specialization/lowering, structure-cache miss | 0.757 ms |
| Specialization/lowering, structure-cache hit | 0.467 ms |
| First compiled forward/backward, including capture | 340.205 ms |
| Eager forward/backward | 3.712 ms |
| Warm compiled forward/backward | 25.177 ms |
| Python allocations, peak across 20 warm compiled steps | 408,294 bytes |

Python allocation tracking does not include native PyTorch allocations, so it
is not a total resident-memory measurement. On this very small opaque quantum
operator the compiled path is slower than eager execution. Phase 6 therefore
establishes graph composability and gradient correctness only; kernel fusion,
state reuse, amortized speedup, and end-to-end model performance remain future
work requiring separate evidence.
