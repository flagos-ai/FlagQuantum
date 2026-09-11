# 512–1024 Qubit MPS Hamiltonian Identification

This project recovers spatially varying couplings from time-resolved local
measurements of a one-dimensional quantum system. The teacher and learned model
use

```text
H = sum_i J_i X_i X_(i+1) + sum_i h_i Y_i
```

and a differentiable brickwork Trotter circuit. Training observes sparse
`Z_i` and `Z_i Z_(i+1)` values at several evolution times and optimizes every
`J_i` and `h_i` through native FlagQuantum MPS autograd. No statevector is
materialized.

## CPU smoke test

Run this before scheduling GPUs:

```bash
conda activate flagquantum-dev
python benchmarks/internal/mps_hamiltonian_identification/train.py \
  --n-qubits 8 --n-initial-states 2 --time-steps 1,2 \
  --observation-stride 2 --max-bond 8 --steps 3 \
  --device cpu --output /tmp/fq-mps-system-id-smoke.json
```

## Eight-A100 run

The default 512-qubit configuration distributes the independent spacetime
probes over eight GPUs:

```bash
conda activate flagquantum-dev
timeout --signal=TERM --kill-after=30s 12h \
  torchrun --standalone --nproc-per-node=8 \
  benchmarks/internal/mps_hamiltonian_identification/train.py \
  --n-qubits 512 --n-initial-states 16 --time-steps 1,2,3,4 \
  --observation-stride 8 --max-bond 64 --steps 50 --probe-batch-size 8 \
  --output benchmarks/results/mps_system_id_512q_8xa100.json
```

For 1024 qubits, start with fewer probes and a coarser observation grid, then
increase them after measuring memory and step time:

```bash
torchrun --standalone --nproc-per-node=8 \
  benchmarks/internal/mps_hamiltonian_identification/train.py \
  --n-qubits 1024 --n-initial-states 8 --time-steps 1,2,3 \
  --observation-stride 16 --max-bond 64 --steps 50 \
  --output benchmarks/results/mps_system_id_1024q_8xa100.json
```

### Measured 1024-qubit configuration

The current 8xA100 target uses 256 time-depth-3 probes so every rank receives
one batch of 32 probes:

```bash
export TORCHINDUCTOR_CACHE_DIR="$PWD/.inductor-cache/mps-system-id-1024q"
torchrun --standalone --nproc-per-node=8 \
  benchmarks/internal/mps_hamiltonian_identification/train.py \
  --n-qubits 1024 --n-initial-states 256 --time-steps 3 \
  --observation-stride 16 --max-bond 128 --cutoff 0 \
  --probe-batch-size 32 --compile-brickwork --steps 50 --lr 0.01 \
  --output benchmarks/results/mps_system_id_1024q_compiled_8xa100.json
```

On the repository's 8xA100-SXM4-40GB host, a warm-cache 10-step run measured
4.87 seconds per training step and 6.25 GiB peak allocated memory per GPU.  The
loss decreased from 1.85e-3 to 2.23e-4.  This short run measures execution and
gradient flow, not converged parameter recovery.

A subsequent convergence run used a 1000-step upper bound, learning rate 0.01,
target loss 1e-7, patience 50, and minimum improvement 1e-8.  It reached the
target at step 127 in 449.0 training seconds (3.54 seconds per completed step):

- observation MSE: 1.85e-3 to 9.98e-8;
- coupling relative error: 21.4%, with true/learned Pearson correlation 0.27;
- field relative error: 34.2%, with true/learned Pearson correlation -0.01;
- mean SM utilization across the eight GPUs: 41.7% to 48.0%;
- median utilization: 19% to 21%, and p95 utilization: 100%;
- peak `nvidia-smi` memory: 8078 MiB per GPU.

This is convergence of the sparse-observation objective, not unique recovery of
the teacher Hamiltonian.  A single time depth and stride-16 Z/ZZ observations
leave the inverse problem poorly conditioned: distinct spatial parameter
vectors can reproduce the measured local dynamics.  The result supports the
claim that FlagQuantum can optimize a differentiable 1024-qubit MPS model, but
does not support a scientific claim of accurate Hamiltonian identification.
Parameter-recovery claims require a multi-time observation design, denser or
staggered spatial coverage, and validation on held-out initial states/times.

For long chains the compiled path sends only shape buckets with at least eight
sites or bonds through Inductor.  Small boundary buckets remain eager.  This
reduces the dynamic specialization family observed at 1024 qubits from roughly
16 to 6 graphs per rank while retaining Triton kernels for the large interior
buckets.  Use a persistent cache: the first forward and backward still include
autotuning and must not be reported as steady-state step time.

## Site-sharded learned MPS

Pass `--site-sharded` under `torchrun` to place one batched learned MPS across
contiguous rank-owned site intervals. This path uses boundary-tensor P2P
exchange, explicit reverse VJPs, and owner-sharded Adam state, without
reconstructing the learned MPS:

```bash
torchrun --standalone --nproc-per-node=8 \
  benchmarks/internal/mps_hamiltonian_identification/train.py \
  --site-sharded --n-qubits 1024 --n-initial-states 8 \
  --time-steps 1,2,3 --observation-stride 16 \
  --max-bond 128 --cutoff 0 --steps 50 --lr 0.01 \
  --output benchmarks/results/mps_system_id_1024q_site_sharded.json
```

Different time depths are represented by masked RY/RXX batch layers. Teacher,
learned-model, and held-out observations all use the same rank-owned MPS path.
A single pair of distributed environment sweeps produces every requested Z and
adjacent ZZ value; the training sweep also forms the complete MSE adjoint in
that pass. No rank reconstructs a complete teacher or learned MPS.

`--probe-batch-size` is a true memory-bounding microbatch control in this mode.
Gradients are weighted by each microbatch's probe count and accumulated across
all microbatches; owner-sharded Adam executes exactly once per global step,
then broadcasts the two updated parameter vectors. Early stopping operates on
the global probe MSE. Every improvement writes one owner checkpoint per rank,
and the best parameters are restored before optional held-out validation.

The v2 result JSON records microbatch and optimizer-step counts, best checkpoint
paths, held-out MSE, per-rank owned tensor/peak GPU memory, and logical boundary
communication bytes. These are execution evidence, not by themselves a
capacity claim: a production run still needs an attached single-GPU OOM or
memory-budget baseline before promotion.

### Compiled rank-local site kernels

Use `--site-sharded --compile-site-kernels --compile-observables` to compile
only work wholly owned by one rank. Consecutive RY sites are bucketed by
batch/left-bond/right-bond/dtype. Equal-shape local RXX bonds compile their
contraction and then use the same exact-QR or truncated-SVD split as eager
execution. Cross-rank bonds always retain the NCCL P2P path.

Z/ZZ transfers use real/imaginary channels and fixed-shape compiled graphs
inside each rank. Result JSON separates `cold_step_seconds` from
`warm_step_seconds` and reports Dynamo graph, generated Triton-kernel, call,
and compile-time counters per rank. Compilation fails closed rather than
silently falling back. Use a persistent `TORCHINDUCTOR_CACHE_DIR`; cold compile
and autotuning time is not steady-state performance.

Before scheduling a multi-step training run, gate it with one bounded optimizer
step. The approximate-gradient contract is explicit: truncation is permitted,
but the sum of discarded weights across all microbatches in that optimizer step
must remain within the stated tolerance. The command also fails unless backward
completes on every rank, owned gradients are finite and nonzero, both parameter
vectors change, every rank stays below the memory ceiling, and a persistent
Inductor cache is configured.

```bash
export TORCHINDUCTOR_CACHE_DIR="$PWD/.inductor-cache/mps-hamiltonian-identification-acceptance"
timeout --signal=TERM --kill-after=30s 20m \
  torchrun --standalone --nproc-per-node=8 \
  benchmarks/internal/mps_hamiltonian_identification/train.py \
  --site-sharded --compile-site-kernels --compile-observables \
  --single-step-acceptance --steps 1 \
  --n-qubits 24 --n-initial-states 1 --time-steps 1 \
  --observation-stride 3 --probe-batch-size 1 \
  --max-bond 1 --cutoff 0 \
  --gradient-policy approximate --discarded-weight-tolerance 0.1 \
  --peak-memory-budget-gib 38 \
  --output benchmarks/results/local/mps_system_id_single_step_acceptance_8xa100.json
```

Do not extrapolate cold-step timing from this gate. Reuse the same cache and
identical shape family for later runs, and increase `--steps` only after
`single_step_acceptance.passed` is true. Exact-gradient runs instead require
`--gradient-policy exact --discarded-weight-tolerance 0`; a positive cutoff or
an active `max_bond` truncation then fails closed.

```bash
torchrun --standalone --nproc-per-node=8 \
  benchmarks/internal/mps_hamiltonian_identification/train.py \
  --site-sharded --compile-site-kernels --compile-observables \
  --n-qubits 1024 --n-initial-states 8 --time-steps 1,2,3 \
  --probe-batch-size 1 --max-bond 128 --cutoff 0 \
  --steps 1000 --lr 0.01 --log-every 10
```

## Interpretation and claim boundary

The eight-GPU mode is `observable_term_parallel`: every rank handles distinct
initial-state/time probes and gradients are summed before one optimizer update.
Each individual probe still stores a complete MPS on one GPU. This is useful
multi-GPU quantum-AI training, but it is not rank-sharded MPS capacity evidence.
The JSON says so explicitly and keeps `scalability_claim_allowed=false`.

Probes with the same time depth execute as one batched MPS. Set
`--probe-batch-size` between 8 and 32 and make `n-initial-states` at least
`8 * probe-batch-size` to give every GPU a full batch. Local Z and adjacent ZZ
terms share one pair of environment sweeps instead of contracting the chain
once per observable.

### Measured utilization boundary

On the repository's 8xA100-SXM4-40GB NVSwitch host, 512-qubit, time-depth-3
runs measured the following with 500 ms `nvidia-smi` samples:

| probe batch/GPU | peak memory/GPU | mean SM range | conclusion |
| --- | ---: | ---: | --- |
| 8 | 1.81 GiB | 15.5%–24.4% | batched path works; launch-bound |
| 32 | 3.69 GiB | 14.2%–22.5% | batching alone does not remove launch gaps |

Both configurations reached occasional 99%–100% samples, but neither achieved
the proposed 60%–80% sustained target. These measurements predate the compiled
fast path below; do not reuse them as compiled-kernel utilization evidence.

## Compiled brickwork fast path

Add `--compile-brickwork` to bypass Circuit instruction interpretation. The
specialized path:

- packs equal-shape sites and spatially disjoint bonds into larger batches;
- represents complex tensors as real/imaginary channels because PyTorch 2.13
  Inductor cannot code-generate the required complex operators correctly;
- compiles batched RY and fused RXX contraction/exact-split graphs;
- autotunes real-valued Triton BMM kernels for the bounded family of active
  bond shapes;
- converts to complex MPS tensors only after the sweep for observables.

Example with a persistent compilation cache:

```bash
export TORCHINDUCTOR_CACHE_DIR="$PWD/.inductor-cache/mps-system-id"
torchrun --standalone --nproc-per-node=8 \
  benchmarks/internal/mps_hamiltonian_identification/train.py \
  --n-qubits 512 --n-initial-states 256 --time-steps 3 \
  --observation-stride 16 --max-bond 128 --cutoff 0 \
  --probe-batch-size 32 --compile-brickwork --steps 50 \
  --output benchmarks/results/mps_system_id_512q_compiled_8xa100.json
```

The first run includes shape-specialization and Triton autotuning. Use the same
cache directory and identical shapes for steady-state measurements. The fast
path fails closed for positive cutoffs or a `max-bond` smaller than the active
exact rank; truncated SVD is not silently substituted.

Primary quality metrics are final observation MSE and relative errors of the
recovered coupling and field vectors. Always compare several bond dimensions
and cutoffs; agreement across truncation settings is stronger evidence than a
low training loss alone.

The training default is `--cutoff 0`, which selects FlagQuantum's exact
QR/autograd split when `max-bond` covers the active rank. Positive truncation
cutoffs currently pass through PyTorch complex-SVD backward and can fail at
degenerate singular values; use them only as an explicitly monitored
approximate-gradient experiment.
