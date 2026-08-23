# A800 TN working-set calibration

## Scope

This is development hardware calibration, not release evidence. Measurements
use native TN sparse-amplitude contractions on NVIDIA A800-SXM4-80GB GPUs.
World sizes 1/2/4/8 are single-node NVSwitch runs. A two-node, 16-GPU
communication preflight was also performed; it does not create a world-size-16
memory calibration.

The raw measurements demonstrate why a pure measured/planned ratio is
insufficient: CUDA initialization and allocator reservation dominate small
contractions. The versioned calibration therefore uses a conservative affine
envelope:

`predicted reserved = fixed reserved overhead + ceil(tensor peak × safety factor)`

The safety factor includes a 10% margin over the largest measured residual
slope. Calibration identities bind the accelerator name, complex dtype, world
size, topology, measurement digest, and model parameters. Missing or
scope-mismatched calibration fails closed when calibrated memory is required.

## Single-GPU calibration

| Dtype | Qubits | Layers | Planned peak | CUDA allocated peak | CUDA reserved peak |
|---|---:|---:|---:|---:|---:|
| complex64 | 14 | 6 | 8 KiB | 8.23 MiB | 22 MiB |
| complex64 | 18 | 9 | 512 KiB | 9.75 MiB | 22 MiB |
| complex64 | 22 | 12 | 32 MiB | 104.31 MiB | 138 MiB |
| complex128 | 14 | 6 | 16 KiB | 8.25 MiB | 22 MiB |
| complex128 | 18 | 9 | 1 MiB | 11.25 MiB | 24 MiB |
| complex128 | 22 | 12 | 64 MiB | 200.31 MiB | 268 MiB |

The first 36-qubit complex64 grid measurement invalidated the small-only
envelope: a 4 GiB tensor peak produced a 25.85 GiB reserved peak while the
legacy policy predicted only 16 GiB. Including that point yields:

| Scope | Fixed overhead | Safety factor |
|---|---:|---:|
| complex64, world size 1 | 22 MiB | 7.10435 |
| complex128, world size 1 | 22 MiB | 4.22813 |

The independent complex64 closure predicted 28.44 GiB, measured 25.85 GiB,
and correctly passed a 32 GiB budget.

## Single-node distributed matrix

The 36-qubit 4×9 grid, four cycles, and one sparse amplitude were measured at
three distinct tensor peaks. Times below are the slowest-rank median for one
measured iteration.

### complex64

| World size | 2 GiB peak / 512 slices | 4 GiB peak / 128 slices | 8 GiB peak / 32 slices |
|---:|---:|---:|---:|
| 2 | 36.778 s | 17.165 s | 8.433 s |
| 4 | 18.548 s | 8.736 s | 4.406 s |
| 8 | 9.480 s | 4.686 s | 2.411 s |

Reserved memory per rank was 12.86, 25.85, and 51.69 GiB respectively.
All ranks at a given point owned equal task counts and reported equal memory
peaks. The world-size 2/4/8 envelopes use 12.86 GiB fixed overhead and a
5.34048 safety factor.

### complex128

| World size | 1 GiB peak / 2048 slices | 2 GiB peak / 512 slices | 4 GiB peak / 128 slices |
|---:|---:|---:|---:|
| 2 | 115.513 s | 52.634 s | 24.697 s |
| 4 | 57.856 s | 26.442 s | 12.401 s |
| 8 | 29.017 s | 13.310 s | 6.335 s |

Reserved memory per rank was 9.80, 19.57, and 39.32 GiB respectively.
The world-size 2/4/8 envelopes use 9.80 GiB fixed overhead and a 4.05840
safety factor.

These matched-workload single-node points show near-linear slice-parallel
scaling. They do not establish multi-node scalability.

## Independent calibrated closure

Each closure used a 64 GiB working-set budget and a calibration produced from
separate raw records.

| Dtype | World size | Predicted reserved | Measured reserved | Time | Budget result |
|---|---:|---:|---:|---:|---|
| complex64 | 2 | 55.58 GiB | 51.69 GiB | 8.486 s | pass |
| complex64 | 4 | 55.58 GiB | 51.69 GiB | 4.417 s | pass |
| complex64 | 8 | 55.58 GiB | 51.69 GiB | 2.424 s | pass |
| complex128 | 2 | 42.27 GiB | 39.32 GiB | 24.661 s | pass |
| complex128 | 4 | 42.27 GiB | 39.32 GiB | 12.422 s | pass |
| complex128 | 8 | 42.27 GiB | 39.32 GiB | 6.292 s | pass |

## Two-node preflight and blockers

Nodes `p-kt-lc-a800-04` and `p-kt-lc-a800-06` were launched with eight ranks
each. `multinode_preflight_2n16g.json` records:

- world size 16 over NCCL with 16 NVIDIA A800-SXM4-80GB devices;
- unique rank/local-rank/GPU mappings, eight ranks per host, and no placement
  warnings;
- successful barrier, all-reduce sum on every rank, and cross-node object
  gather;
- maximum scalar all-reduce time 2.26 ms;
- NCCL logs selecting nine RoCE HCAs and GDRDMA for cross-node channels.

A high-memory TN attempt then exposed two independent blockers:

1. The RoCE path reported `IBV_WC_RETRY_EXC_ERR` on `mlx5_101` under load.
   The scalar collective preflight alone is therefore insufficient bandwidth
   and stability evidence.
2. With IB disabled to isolate software correctness, all eight GPUs on the
   second node had unrelated processes using about 71.5–73.0 GiB each. GPU 0
   left only 831 MiB free, so the benchmark rank failed a 1 GiB allocation
   without owning that memory.

The benchmark-owned ranks were terminated and the unrelated process was not
touched. No world-size-16 calibration or scalability claim is emitted. A
valid ws16 calibration requires an uncontended node, a repaired/stable RoCE
path, a large-payload collective preflight, three distinct successful tensor
peaks, and an independent calibrated closure.

## Claim boundary

- `claim_evidence_type`: `development_hardware_calibration`
- `scalability_claim_allowed`: `false`
- calibrated scopes: A800-SXM4-80GB complex64/complex128 world sizes 1, 2, 4,
  and 8, with the topology bound in each artifact
- world size 16: communication correctness preflight only; memory calibration
  and performance claims are blocked
- the selector rejects missing, tampered, or scope-mismatched calibration when
  calibrated TN memory is required
