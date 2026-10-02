# FlagQuantum and CUDA-Q comparison on two A800 hosts

This is the recorded measurement behind `contracts/cudaq-parity-matrix.toml`. It
replaces the scoreboard's previous statement that no CUDA-Q measurement existed in
this repository. The raw payload is the adjacent
`cudaq_backend_compare_a800_20260930.json`; this document reads it.

Recorded: 2026-09-30, on `jp-a800-172`
(`bm-baai-dx-zone1-lc-a800-80g-15-172`) and `jp-a800-171`
(`bm-baai-dx-zone1-lc-a800-80g-15-171`). Each host contributed one NVIDIA
A800-SXM4-80GB out of eight, driver 580.126.20, and CUDA-Q 0.16.0.post1. The two
hosts are deliberately not the same environment. On `172` the runtime is the
existing FlagQuantum development image (`tovx/flagquantum:0.2.0-dev`, Python
3.12.13, PyTorch 2.13.0+cu130, `cuda-quantum-cu12`). On `171` there is no
FlagQuantum image, so the runtime was assembled from a bare
`python:3.12-slim-bookworm` base (Python 3.12.14, PyTorch 2.14.0+cu130, JAX
0.10.2, `cuda-quantum-cu13`). Both pulled CUDA-Q from the Tsinghua mirror, because
the public PyPI index measured 0.01 MB/s from these hosts.

Both ran revision `6cfc4c3708227a4bb2c6e35c9f71e502239a2e63`. Neither runtime
image ships a `git` binary, so no probe could record a revision itself; the link
to the revision is the SHA-256 of `benchmarks/cudaq_backend_compare.py`, recorded
as `runner_sha256` and read back with `sha256sum` inside each container.

## What was measured

Both sides evaluate the same parameterized circuit — a linear hardware-efficient
ansatz with `rx`/`ry`/`rz` per wire and a `cx` ladder per layer — with the same
observable, `complex64` on both sides, and CUDA-Q's `ParameterShift` strategy for
the gradient. The FlagQuantum side is a JAX kernel behind the PyTorch interface;
the CUDA-Q side runs its compiled program through the `nvidia` target, except for
one case that uses `qpp-cpu`. Every case reported here converged at the declared
tolerance (`loss_atol = grad_atol = 1e-4`); the largest recorded deviation is
`1.2e-05` in the loss and `1.8e-05` in the gradient. On this workload the two
frameworks agree numerically, which is the precondition for any ratio below to
mean anything.

The full matrix was captured on both hosts, so every conclusion in this document
rests on two independent runs rather than one.

## Cross-host agreement

Same probe, same revision, same CUDA-Q release, two hosts and two different base
images. This is the reproducibility check: a conclusion that does not survive it
is a measurement of one machine.

| Case | Wires | FlagQuantum `172` | FlagQuantum `171` | Spread | CUDA-Q `172` | CUDA-Q `171` | Spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| forward, ising | 8 | 1.954 ms | 1.957 ms | 0.2% | 1.454 ms | 1.509 ms | 3.7% |
| forward, ising | 14 | 4.642 ms | 4.482 ms | 3.4% | 2.058 ms | 3.046 ms | 32.4% |
| forward, ising | 18 | 39.279 ms | 38.170 ms | 2.8% | 2.499 ms | 2.609 ms | 4.2% |
| forward, ising | 22 | 355.451 ms | 356.229 ms | 0.2% | 3.968 ms | 4.006 ms | 0.9% |
| forward, `z_sum` | 8 | 1.805 ms | 1.835 ms | 1.7% | 1.350 ms | 1.398 ms | 3.5% |
| forward, `z_sum` | 18 | 34.954 ms | 32.627 ms | 6.7% | 2.243 ms | 2.242 ms | 0.0% |
| forward, `z_sum` | 22 | 284.141 ms | 294.620 ms | 3.6% | 3.164 ms | 3.083 ms | 2.6% |
| forward, ising, JIT off | 14 | 1801.812 ms | 1988.257 ms | 9.4% | 2.009 ms | 2.050 ms | 2.0% |
| value+gradient, ising | 14 | 4.647 ms | 4.790 ms | 3.0% | 336.529 ms | 360.972 ms | 6.8% |
| value+gradient, ising | 18 | 37.087 ms | 37.933 ms | 2.2% | 532.943 ms | 543.619 ms | 2.0% |
| value+gradient, `qpp-cpu` | 6 | 1.436 ms | 1.591 ms | 9.8% | 166.883 ms | 156.186 ms | 6.4% |

Every FlagQuantum figure is within 10% across hosts, which is the agreement that
makes the per-host conclusions below portable. The one CUDA-Q outlier is the
14-wire forward case, where `171` reads 3.046 ms against `172`'s 2.058 ms — a 32%
spread in a quantity under 3 ms, i.e. a few hundred microseconds on a shared host
that was running ten containers. At 18 and 22 wires the same quantity agrees to
within 4% and 1%, so the outlier is noise at the small end rather than a
systematic difference, and no conclusion here depends on it.

## Forward-only execution: FlagQuantum is behind, and the gap widens

Ratios above one mean FlagQuantum was faster.

| Observable | Wires | FlagQuantum JAX | CUDA-Q `nvidia` | CUDA-Q / FlagQuantum |
| --- | ---: | ---: | ---: | ---: |
| ising | 8 | 1.954 ms | 1.454 ms | 0.74x |
| ising | 14 | 4.642 ms | 2.058 ms | 0.44x |
| ising | 18 | 39.279 ms | 2.499 ms | 0.064x |
| ising | 22 | 355.451 ms | 3.968 ms | 0.011x |
| z_sum | 8 | 1.805 ms | 1.350 ms | 0.75x |
| z_sum | 18 | 34.954 ms | 2.243 ms | 0.064x |
| z_sum | 22 | 284.141 ms | 3.164 ms | 0.011x |

Two readings follow, and both are uncomfortable.

The FlagQuantum JAX statevector path is slower than CUDA-Q's `nvidia` target at
every width measured, by 16x at 18 wires and by 89x at 22 wires. The `nvidia`
target is the cuStateVec path, so this table is the first measured size of the
`statevector_simulation` gap that `contracts/cudaq-parity-matrix.toml` records as
`A_nvidia_proprietary`. The gap is not a constant factor: FlagQuantum costs about
8.5x more per four added wires from 14 to 18 and about 9.1x more from 18 to 22,
while CUDA-Q costs 1.21x and 1.59x. A cost that grows faster per amplitude than
the reference is a structural difference in how the state is evolved, not a
tuning deficit. This particular conclusion is warmup-insensitive (see below), so
it is not an artifact of the measurement protocol.

The observable is not the cause. Replacing the ising observable's 21 `ZZ` terms,
22 `X` terms, and 22 `Z` terms with `z_sum`'s 22 `Z` terms changes the 22-wire
FlagQuantum time from 355 ms to 284 ms, a 20% move; the state evolution dominates
both. CUDA-Q moves from 3.97 ms to 3.16 ms over the same substitution. The
remaining 89x is therefore attributable to building and applying the circuit, and
that is where the next investigation belongs.

The JIT contribution is measured rather than assumed. The same 14-wire
forward-only configuration with `--no-jax-jit` takes 1801.812 ms against 4.642 ms
with JIT, a factor of 388. Every FlagQuantum number in this document is a
just-in-time-compiled steady-state number, and an interpreted FlagQuantum run is
not competitive with anything here.

## Value and gradient: FlagQuantum is ahead, and the ratio is not yet settled

| Observable | Wires | FlagQuantum JAX | CUDA-Q target | FlagQuantum / CUDA-Q |
| --- | ---: | ---: | ---: | ---: |
| ising | 8 | 1.997 ms | `nvidia`, 134.348 ms | 67.3x |
| ising | 14 | 4.647 ms | `nvidia`, 336.529 ms | 72.4x |
| ising | 18 | 37.087 ms | `nvidia`, 532.943 ms | 14.4x |
| ising | 6 | 1.436 ms | `qpp-cpu`, 166.883 ms | 116.2x |

These ratios do not mean the FlagQuantum kernel is 70x faster at simulating. The
two sides do not execute the same schedule. CUDA-Q's Python `ParameterShift`
issues one host-side `observe` call per shifted parameter, so its cost grows with
the parameter count and is dominated by per-call overhead rather than by
simulation. The FlagQuantum side calls one compiled XLA program that fuses value
and gradient, so its cost is nearly independent of the parameter count at these
widths. The comparison is real and the ratio is real, but it measures call
structure as much as it measures arithmetic, and it stays inside this repository
as evidence of a training-path advantage rather than of simulation throughput.

The 18-wire row deserves a warning rather than a number. An earlier capture of the
same invocation read 72.3 ms for the FlagQuantum side and was reported as 7.9x.
Re-running it here reads 39.2 ms, and sweeping the warmup count with the iteration
count fixed gives 39.2 ms at one warmup, 36.6 ms at three, and 32.6 ms at five —
monotone, and therefore not converged. **The previously reported 7.9x is
withdrawn as an under-warmed reading.** The ratio must be quoted as a range
(13.6x to 16.3x under this protocol) until a longer-warmup protocol is recorded,
and the `protocol_sensitivity` section of the payload carries the sweep.

## What this does not establish

- **No scalability claim.** One A800 per host ran one logical workload. Nothing
  was partitioned across ranks, and both payloads set
  `scalability_claim_allowed` to false and `release_gate_allowed` to false.
- **No wall-clock ceiling for CUDA-Q.** The `nvidia` target is measured here at
  8, 14, 18, and 22 wires on one workload shape. Nothing about its multi-GPU
  targets, tensor-network path, or `mqpu` scheduling follows.
- **No gradient-capability conclusion.** The gradient ratios are consistent with
  CUDA-Q lacking a fused reverse-mode gradient at this layer, but this measurement
  does not show that, and the companion capability probe records only which Python
  symbols the installed release exports.
- **No claim about the CPU comparison.** The one `qpp-cpu` case shows the same
  call-structure effect against a CPU target and is not a CPU-versus-GPU result:
  the FlagQuantum side ran on the GPU in that case.
- **No crossover location.** The two tables above bracket a crossover between 14
  and 18 wires for the gradient path, but neither protocol was run at 15, 16, or
  17 wires, so the crossing point is not measured.

## Relationship to the parity scoreboard

`contracts/cudaq-parity-matrix.toml` registers
`benchmarks/cudaq_backend_compare.py` at `status = "measured"`, naming this
payload and this document as the evidence. The scoreboard rows themselves are
unchanged by the measurement: no row in the matrix is a performance comparison,
and the matrix's own limitation text says so.

Reproduce with, inside a container carrying CUDA-Q 0.16.0.post1:

```bash
python benchmarks/cudaq_backend_compare.py \
  --device cuda --n-wires 8 --layers 2 --batch-size 1 --observable ising \
  --mode statevector --iters 10 --warmup 2 \
  --cudaq-target nvidia --cudaq-gradient-method parameter-shift \
  --container-digest <image-digest> --json-output gpu.json
```

This is a local comparison result, not scalability or release evidence.
