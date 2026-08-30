# 36q two-node tensor-network training hard gate

This development hardware bundle combines the frozen 36-qubit 4x9-grid,
four-cycle complex128 expectation-gradient workload across 1, 2, 4, 8, and 16
A800 GPUs. The 16-GPU point uses two nodes with eight GPUs per node.

| GPUs | Nodes | Seconds | Speedup vs 1 GPU |
|---:|---:|---:|---:|
| 1 | 1 | 140.904 | 1.00x |
| 2 | 1 | 73.065 | 1.93x |
| 4 | 1 | 36.742 | 3.83x |
| 8 | 1 | 17.646 | 7.99x |
| 16 | 2 | 8.947 | 15.75x |

The two-node execution completed ten consecutive forward contractions,
explicit sliced reverse passes, and rank-owned SGD steps in 87.51 seconds.
Every step rebuilt the TN from the parameters produced by the previous step.
Each of the 16 ranks owned eight of the 128 contraction slices. The 36
parameter updates were balanced across ranks with one update owner per
parameter; updated parameter values were then broadcast for the next forward
pass. Gradient contributions are reduced directly to those unique parameter
owners, so non-owner ranks do not retain a complete aggregated gradient.
Parameter checksums agreed across all 16 ranks after every step.

The reverse path measured about 15.99 GiB of checkpoint tape and 768
rematerialized operations per rank. Peak CUDA allocation was about 43.64 GiB
and peak reservation about 48.43 GiB per rank. It did not materialize a
statevector, complete unsliced tape, or complete intermediate set.

A complex128 36-qubit statevector requires 1 TiB. The MPS selector estimate for
the same high-width circuit family is bond 262,144 and at least 36 TiB of
storage. Both are outside one 80-GB A800.

The separate 16-rank, 256-MiB NCCL preflight passed collective correctness on
two nodes and measured 6.19 GiB/s algorithmic payload bandwidth. NCCL logs
identified the IB/GDRDMA route with the known-bad `mlx5_101` interface
excluded.

[`evidence.json`](evidence.json) passes the non-release benchmark audit with
all hard-gate checks true. It intentionally remains fail-closed for release:
no signed release artifact has been produced. A release-candidate projection
of the measured payload passes the distributed scalability gate without
metadata errors, but signing requires a release-authorized key and a committed
source identity.
