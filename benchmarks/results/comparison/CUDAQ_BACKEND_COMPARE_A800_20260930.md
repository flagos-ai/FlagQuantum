# FlagQuantum and CUDA-Q comparison on an A800

This is the first recorded measurement behind `contracts/cudaq-parity-matrix.toml`.
It replaces the scoreboard's previous statement that no CUDA-Q measurement existed
in this repository. The raw payload is the adjacent
`cudaq_backend_compare_a800_20260930.json`; this document reads it.

Recorded: 2026-09-30, on `jp-a800-172`
(`bm-baai-dx-zone1-lc-a800-80g-15-172`), one NVIDIA A800-SXM4-80GB out of eight,
driver 580.126.20, Python 3.12.13, PyTorch 2.13.0+cu130, JAX 0.10.2, CUDA-Q
0.16.0.post1 with the `cuda-quantum-cu12` binary distribution. FlagQuantum ran at
revision `20f0ff7`; the container image digest is recorded in the payload. The
runtime image ships no `git` binary, so the probe could not record a revision
itself; the link to the revision is the SHA-256 of
`benchmarks/cudaq_backend_compare.py`, recorded as `runner_sha256` and identical
in this checkout and in the container that produced the numbers.

## What was measured

Both sides evaluate the same parameterized circuit — a linear hardware-efficient
ansatz with `rx`/`ry`/`rz` per wire and a `cx` ladder per layer — with the same
observable, `complex64` on both sides, and CUDA-Q's `ParameterShift` strategy for
the gradient. Each configuration runs two warmups and ten measured iterations,
with device synchronization inside each measured iteration. The FlagQuantum side
is a JAX kernel behind the PyTorch interface; the CUDA-Q side runs its compiled
program through the `nvidia` target, except for one case that uses `qpp-cpu`.

Every case converged at the declared tolerance (`loss_atol = grad_atol = 1e-4`).
The largest recorded deviation is `8.6e-06` in the loss and `1.8e-05` in the
gradient. On this workload the two frameworks agree numerically, which is the
precondition for any ratio below to mean anything.

## Forward-only execution: FlagQuantum is behind, and the gap widens

Ratios above one mean FlagQuantum was faster.

| Observable | Wires | FlagQuantum JAX | CUDA-Q `nvidia` | CUDA-Q / FlagQuantum |
| ---: | ---: | ---: | ---: | ---: |
| ising | 8 | 1.943 ms | 1.471 ms | 0.76x |
| ising | 14 | 4.449 ms | 2.050 ms | 0.46x |
| ising | 18 | 38.876 ms | 2.509 ms | 0.065x |
| ising | 22 | 355.391 ms | 4.001 ms | 0.011x |
| z_sum | 8 | 1.786 ms | 1.381 ms | 0.77x |
| z_sum | 18 | 33.472 ms | 2.278 ms | 0.068x |
| z_sum | 22 | 284.164 ms | 3.090 ms | 0.011x |

Two readings follow, and both are uncomfortable.

The FlagQuantum JAX statevector path is slower than CUDA-Q's `nvidia` target at
every width measured, by 15x at 18 wires and by 89x at 22 wires. The `nvidia`
target is the cuStateVec path, so this table is the first measured size of the
`statevector_simulation` gap that `contracts/cudaq-parity-matrix.toml` records as
`A_nvidia_proprietary`. The gap is not a constant factor: FlagQuantum costs about
8.7x more per four added wires from 14 to 18 and about 9.1x more from 18 to 22,
while CUDA-Q costs 1.22x and 1.59x. A cost that grows faster per amplitude than
the reference is a structural difference in how the state is evolved, not a
tuning deficit.

The observable is not the cause. Replacing the ising observable's 21 `ZZ` terms,
22 `X` terms, and 22 `Z` terms with `z_sum`'s 22 `Z` terms changes the 22-wire
FlagQuantum time from 355 ms to 284 ms, a 20% move; the state evolution dominates
both. CUDA-Q moves from 4.00 ms to 3.09 ms over the same substitution. The
remaining 89x is therefore attributable to building and applying the circuit, and
that is where the next investigation belongs.

The JIT contribution is measured rather than assumed. The same 14-wire
forward-only configuration with `--no-jax-jit` takes 1830.872 ms against 4.449 ms
with JIT, a factor of 412. Every FlagQuantum number in this document is a
just-in-time-compiled steady-state number, and an interpreted FlagQuantum run is
not competitive with anything here.

## Value and gradient: FlagQuantum is ahead, and the advantage narrows

| Observable | Wires | FlagQuantum JAX | CUDA-Q target | FlagQuantum / CUDA-Q |
| ---: | ---: | ---: | ---: | ---: |
| ising | 8 | 1.948 ms | `nvidia`, 136.197 ms | 69.9x |
| ising | 14 | 4.746 ms | `nvidia`, 337.913 ms | 71.2x |
| ising | 18 | 72.304 ms | `nvidia`, 572.246 ms | 7.9x |
| ising | 6 | 1.049 ms | `qpp-cpu`, 65.821 ms | 62.8x |

These ratios do not mean the FlagQuantum kernel is 70x faster at simulating. The
two sides do not execute the same schedule. CUDA-Q's Python `ParameterShift`
issues one host-side `observe` call per shifted parameter, so its cost grows with
the parameter count and is dominated by per-call overhead rather than by
simulation. The FlagQuantum side calls one compiled XLA program that fuses value
and gradient, so its cost is nearly independent of the parameter count at these
widths. The comparison is real and the ratio is real, but it measures call
structure as much as it measures arithmetic, and it stays inside this repository
as evidence of a training-path advantage rather than of simulation throughput.

The narrowing from 71x at 14 wires to 7.9x at 18 wires is the useful part: once
CUDA-Q's fixed per-call overhead is amortized against a larger state, CUDA-Q's
faster simulation core starts to dominate the comparison. Extending both curves
upward would cross over, and the forward-only table shows roughly where.

## What this does not establish

- **No scalability claim.** One A800 ran one logical workload. Nothing was
  partitioned across ranks, and the payload sets `scalability_claim_allowed` to
  false and `release_gate_allowed` to false.
- **No wall-clock ceiling for CUDA-Q.** The `nvidia` target is measured here at
  8, 14, 18, and 22 wires on one workload shape. Nothing about its multi-GPU
  targets, tensor-network path, or `mqpu` scheduling follows.
- **No gradient-capability conclusion.** The 69.9x ratio is consistent with
  CUDA-Q lacking a fused reverse-mode gradient at this layer, but this
  measurement does not show that, and the companion capability probe records only
  which Python symbols the installed release exports.
- **No claim about the CPU comparison.** The one `qpp-cpu` case shows the same
  call-structure effect against a CPU target and is not a CPU-versus-GPU result:
  the FlagQuantum side ran on the GPU in that case.

## Relationship to the parity scoreboard

`contracts/cudaq-parity-matrix.toml` registers
`benchmarks/cudaq_backend_compare.py` at `status = "runnable"`. This document and
its payload are what a promotion to `status = "measured"` requires. The scoreboard
rows themselves are unchanged by this measurement: no row in the matrix is a
performance comparison, and the matrix's own limitation text says so.

Reproduce with, inside a container carrying CUDA-Q 0.16.0.post1:

```bash
python benchmarks/cudaq_backend_compare.py \
  --device cuda --n-wires 8 --layers 2 --batch-size 1 --observable ising \
  --mode statevector --iters 10 --warmup 2 \
  --cudaq-target nvidia --cudaq-gradient-method parameter-shift \
  --container-digest <image-digest> --json-output gpu.json
```

This is a local comparison result, not scalability or release evidence.
