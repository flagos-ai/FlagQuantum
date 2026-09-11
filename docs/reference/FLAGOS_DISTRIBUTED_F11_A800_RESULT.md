# FlagOS distributed F1.1 result on A800

This record captures the FlagQuantum-only F1.1 remediation measured on two
A800 GPUs on `a800-node-1` on 2026-08-25. It supersedes the statevector
precision conclusion in the earlier F1 measurement, while preserving that
measurement as historical evidence.

## Revisions and evidence

- FlagQuantum: `6d20e81daf67e7a9fbb946ddf067225aa9c6ddc5`;
- Torch-FL: `1b19383c24d9aa54637dd23557a5a20ef6d52ec5`;
- PyTorch: `2.11.0+cu130`;
- CUDA runtime: `13.0`;
- driver: `580.126.20`;
- devices: two NVIDIA A800-SXM4-80GB cards;
- ranks: 2 on one node.

The Torch-FL revision is the same public PyTorch-2.11-compatible revision used
by F1, so the before/after comparison changes only FlagQuantum. It is not a
validation of current Torch-FL main, which has a different PyTorch pin.

The machine-readable payload is
[`artifacts/flagos_distributed_conformance_a800_f11_20260825.json`](../../artifacts/flagos_distributed_conformance_a800_f11_20260825.json).
Its SHA-256 digest is
`b305852942c243a1febe817167169c619034b51c71101a73bd77d093acae82a2`.

## Precision remediation

The distributed executor previously maintained a second parameter-to-matrix
implementation. Numeric constants became float32 tensors on the accelerator,
the trigonometric matrix was evaluated at that precision, and only the finished
matrix was promoted to complex128. The information loss was irreversible.

F1.1 removes that duplicate matrix implementation and delegates to
FlagQuantum's precision-aware `gate_matrix` boundary. Non-trainable numeric
constants are evaluated as tiny CPU float64/complex128 matrices and transferred
once to the requested device. Tensor parameters remain on their requested
device, preserving the autograd path.

The same truly sharded three-wire statevector now reports:

- complex64 maximum absolute error: `0.0`;
- complex128 maximum absolute error: `0.0`;
- four local amplitudes out of eight on each rank;
- three distributed gates and two communications;
- no distributed full-state materialization.

The report therefore records
`statevector_workload_conformance_accepted=true`.

## Remaining provider boundary

Eight of ten collective/dtype combinations pass with zero error. Both
complex64 and complex128 `reduce_scatter_tensor` calls remain unsupported by
the measured provider route. The report retains this failure and records
`complex_reduce_scatter_unavailable`.

Consequently:

- `mechanical_conformance_accepted=false`;
- `flagcx_route_verified=false`;
- `communication_claim_allowed=false`;
- `scalability_claim_allowed=false`;
- `release_gate_allowed=false`.

The correct interpretation is: the measured FlagQuantum sharded-statevector
workload now preserves complex128 accuracy on this two-A800 FlagOS route, but
the full collective matrix is not complete and the inner communication
implementation is not attributed. No capability-maturity or
production-support level is promoted.

No Torch-FL or FlagCX source was modified, and no external pull request was
opened.
