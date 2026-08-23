# Working-set-safe distributed TN production gate

## Frozen workload

The workload is unchanged from the previous capacity test: one complex64
amplitude of a 36-qubit 4x9 grid circuit with four entangling cycles.
Statevector requires 512 GiB, while the estimated MPS bond is 262,144 with
about 36 TiB of MPS storage.

## Change

The previous eight-slice plan limited only the largest intermediate to 16 GiB.
It measured 64 GiB per rank and attempted one additional 16 GiB allocation,
triggering an allocator OOM warning before falling back.

The production policy now accepts an end-to-end working-set budget and applies
a calibrated safety factor of four to the intermediate limit.  The frozen test
therefore uses a 4 GiB intermediate limit, 128 slices, and a measured 16 GiB
rank-local peak.  No allocator OOM warning occurs.

Rank zero now creates and broadcasts both the slicing plan and the complete
representative pair-contraction DAG.  Other ranks do not independently search
for a path.  The same DAG is reused by every assigned slice.

## Strong scaling on A800

| GPUs | Slices/rank | Execution (s) | Speedup | Efficiency | Peak/rank |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 33.978 | 1.00x | 100% | 16.0 GiB |
| 2 | 64 | 18.994 | 1.79x | 89% | 16.0 GiB |
| 4 | 32 | 10.142 | 3.35x | 84% | 16.0 GiB |
| 8 | 16 | 4.682 | 7.26x | 91% | 16.0 GiB |

The eight-GPU result exceeds the predefined 5x production gate.  Amplitudes
agree across world sizes within complex64 precision.

## Interpretation

This passes the execution and memory gates for the frozen workload:

- the problem is outside practical single-device SV and MPS capacity;
- sparse TN does not materialize the full state;
- rank-local memory remains far below device capacity;
- no failed allocation is used as runtime control flow;
- the complete contraction DAG is planned once and shared;
- eight-GPU speedup is greater than 5x.

Cold latency is still dominated by approximately 11--13 seconds of rank-zero
path and slicing search.  Broadcasting eliminates redundant CPU search but
does not eliminate the wall-clock search itself.  Persistent plan caching is
the next latency optimization.

This is one-node, one-workload production evidence.  It supports enabling the
distributed TN path behind the selector and working-set preflight, not a claim
of universal TN scaling across arbitrary topology, gradients, or multiple
nodes.
