# Eight-GPU Variable-Bond MPS Capacity Example

This example evolves and differentiates one open-boundary MPS. It does not use
a large batch, data-parallel replicas, or full-MPS reconstruction.

The capacity profile contains 12,288 physical sites, batch size one, and a
maximum bond dimension of 512. Its dense complex64 MPS tensors require about
48 GiB before reverse-tape and optimizer memory, so an A100 40GB cannot hold
the logical state. With eight GPUs, each rank owns one continuous interval of
1,536 sites and about 6 GiB of the initial MPS.

First run the bounded smoke profile:

```bash
timeout --signal=TERM --kill-after=30s 10m \
  torchrun --standalone --nproc-per-node=8 \
  examples/distributed_mps/variable_bond_capacity_8gpu.py \
  --profile smoke --output /tmp/fq-mps-smoke.json
```

Measure the single-A100 capacity failure:

```bash
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 30m \
  torchrun --standalone --nproc-per-node=1 \
  examples/distributed_mps/variable_bond_capacity_8gpu.py \
  --profile capacity --output /tmp/fq-mps-capacity-1gpu.json
```

Run the same single-state workload on eight A100 40GB GPUs:

```bash
timeout --signal=TERM --kill-after=30s 2h \
  torchrun --standalone --nproc-per-node=8 \
  examples/distributed_mps/variable_bond_capacity_8gpu.py \
  --profile capacity --output /tmp/fq-mps-capacity-8gpu.json
```

The JSON remains fail-closed: it records measured capacity behavior but does
not enable a release or general scalability claim by itself.

## Measured reference

On 2026-07-12 the capacity profile used 51,469,702,464 logical MPS bytes. One
A100 40GB failed with CUDA OOM at 41,859,156,992 peak allocated bytes. The
eight-A100 run completed one forward/backward/Adam step in 47.1–47.3 seconds;
all ranks reported useful work and peak allocation was 25.7–25.8 GB per rank.
The compact measured record is stored in
`results/capacity_a100_8gpu_summary.json`.

For the general entangling capacity gate with dynamic bond truncation and all
seven rank boundaries active, run:

```bash
timeout --signal=TERM --kill-after=30s 2h \
  torchrun --standalone --nproc-per-node=8 \
  benchmarks/internal/evidence/general_mps_capacity.py \
  --profile capacity \
  --single-gpu-artifact /tmp/issue092-capacity-1gpu.json \
  --output /tmp/issue092-capacity-8gpu.json
```

Its compact audited result is
`benchmarks/development/issue092_general_mps_capacity.json`.

## Scientific VQE limit probe

`benchmarks/distributed_mps_heisenberg_vqe.py` trains an open-boundary
antiferromagnetic XXZ Heisenberg chain from a configurable Neel-like or random
isometric MPS warm start with an even/odd Hamiltonian variational ansatz.  The
linear Hamiltonian energy uses one five-channel distributed MPO scan for all
on-site Z and adjacent XX/YY/ZZ terms, rather than one full MPS scan per Pauli
term.  It reports qubits, requested and realized bond
dimension, ansatz depth, actual two-qubit gate depth, energy trajectory,
truncation, time, memory, communication, and rank ownership.  Each sweep point
must use a fresh `torchrun`, since continuing after a CUDA OOM can leave an
NCCL process group in an unreliable state.

For example, one 8-GPU point is:

```bash
timeout --signal=TERM --kill-after=30s 30m \
  torchrun --standalone --nproc-per-node=8 \
  benchmarks/distributed_mps_heisenberg_vqe.py \
  --n-sites 256 --depth 16 --max-bond 512 --steps 3 \
  --output /tmp/fq-heisenberg-n256-p16-chi512.json
```

Sweep one axis at a time (for example sites `64,128,256,512,1024`, bond
`64,128,256,512,1024`, and HVA depth `2,4,8,16,32`) before refining the first
failing interval.  A point is capacity evidence only when every rank completes;
a scalability claim additionally needs a matched one-GPU OOM baseline and the
repository release audit.
