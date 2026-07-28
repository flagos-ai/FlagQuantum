# MPS Top-Tier Roadmap

## Purpose

This document is the execution plan for moving FlagQuantum MPS from
development evidence to a top-tier PyTorch-native, differentiable,
rank-owned, multi-GPU MPS training system.

“Top tier” does not mean fastest on every workload. It means that FlagQuantum:

1. is numerically trustworthy across its declared dtype and training matrix;
2. reaches the performance Pareto frontier in important MPS workloads;
3. completes useful workloads that do not fit on one accelerator;
4. scales predictably across one node and then multiple nodes;
5. provides a broad MPS/MPO algorithm surface;
6. publishes reproducible, signed and independently auditable evidence.

Claims must remain capability-specific and fail closed. A plan, preflight,
CPU test or unsigned development result is not release evidence.

## Current baseline

Status captured on 2026-07-28:

- PyTorch-native exact and approximate MPS gradients are implemented.
- Rank-owned forward, reverse, optimizer and checkpoint paths exist.
- Single-node NCCL development measurements pass on 1/2/4/8 A800 GPUs.
- The frozen workload is 16 qubits, four layers, maximum bond 64, Adam,
  exact gradients, five warmups and twenty retained steps.
- The development artifacts pass the structural evidence audit but are
  unsigned and explicitly non-release evidence.

| GPUs | Mean seconds/step | Speedup over 1 GPU | Parallel efficiency |
| ---: | ---: | ---: | ---: |
| 1 | 0.699 | 1.00× | 100% |
| 2 | 0.506 | 1.38× | 69% |
| 4 | 0.453 | 1.54× | 39% |
| 8 | 0.499 | 1.40× | 18% |

The current small workload is fastest on four GPUs. The eight-GPU result is
about 10% slower than the four-GPU result, so it does not support a strong
scaling claim.

Retained development evidence:

- `benchmarks/results/smoke/release_candidates/mps_single_node/`
- source identity: `f4242786c17419669b32ddc287f97da8acbd0827`
- audit result: four valid development artifacts, zero release-claimable
  artifacts

## Top-tier acceptance gates

### Numerical correctness

- [ ] complex64 forward/value tolerance is at most `3e-5`.
- [ ] complex64 gradient tolerance is at most `3e-5`.
- [ ] complex128 value and gradient tolerance is at most `1e-10`.
- [ ] Exact, truncated and approximate policies report distinct semantics.
- [ ] SGD, Adam and L-BFGS complete the declared Cartesian matrix.
- [ ] Checkpoint/restart matches uninterrupted training.
- [ ] Truncation error and discarded weight reconcile with the global budget.

### Performance

- [ ] At least one predeclared, useful workload leads public baselines on the
      same hardware and numerical error budget.
- [ ] Other primary workloads are no worse than `1.25×` the best measured
      baseline unless a documented product tradeoff explains the gap.
- [ ] Large workloads achieve at least `1.5×` speedup from four to eight GPUs.
- [ ] Performance comparisons include latency, throughput, peak memory,
      communication, compilation and numerical error.
- [ ] Results include confidence intervals and prohibit cherry-picked points.

### Capacity and scaling

- [ ] A matched workload fails from capacity on one GPU and completes on eight.
- [ ] A second workload fails on four GPUs and completes on eight.
- [ ] Each rank retains only owned state, tape and optimizer shards.
- [ ] No hidden full-MPS or statevector materialization occurs.
- [ ] Single-node crossover is calibrated for qubits, bond and batch.
- [ ] Multi-node forward, backward and optimizer ownership are certified.

### Stability and recovery

- [ ] 1,000-step SGD and Adam tests show a bounded memory plateau.
- [ ] A representative 10,000-step scientific workload completes.
- [ ] Rank failure and communication timeout perform bounded cleanup.
- [ ] Ten checkpoint/restart cycles preserve parameters, optimizer state and
      loss within the declared tolerance.
- [ ] No CPU fallback occurs in accelerator-certified execution.

### Algorithms and product surface

- [ ] Circuit MPS and variable-bond training are stable.
- [ ] MPO expectation and MPO×MPS are stable.
- [ ] One-site and two-site DMRG are available.
- [ ] TEBD and one-site/two-site TDVP are available.
- [ ] Conditional sampling and marginals are available.
- [ ] Dynamic measurement/reset can use an MPS trajectory backend.
- [ ] Unsupported combinations fail closed through the planner.

### Release evidence

- [ ] `FQ_EVIDENCE_SIGNING_KEY` is held by controlled CI or a secrets manager.
- [ ] Artifacts bind source commit, workload hash, command, GPU UUIDs,
      topology, rank placement and environment.
- [ ] Strict release audit accepts every promoted artifact.
- [ ] Third-party reproduction instructions are complete.
- [ ] Claims in documentation match the promoted capability level.

## Frozen benchmark matrix

The benchmark matrix must be declared before optimization results are viewed.

| Axis | Required values |
| --- | --- |
| Qubits/sites | 16, 32, 64, 128, 256, 1024 |
| Maximum bond | 32, 64, 128, 256, 512, 1024 |
| Batch | 1, 4, 16, 64 |
| Dtype | complex64, complex128 |
| GPU count | 1, 2, 4, 8 |
| Optimizer | SGD, Adam, L-BFGS |
| Entanglement | low, medium, high |
| Workload | circuit training, Heisenberg/XXZ, capacity, DMRG/TDVP |

Full Cartesian execution is not required for every pull request. The manifest
must define a statistically useful covering array for scheduled tests and a
complete release matrix.

External baselines should include, where compatible:

- NVIDIA cuTensorNet/CUDA-Q MPS;
- PennyLane Lightning Tensor;
- ITensor/ITensorTDVP;
- the previous FlagQuantum release and local fast path.

All comparisons must use the same accelerator model, dtype, workload,
truncation/error budget, warmup policy and retained sample count.

## Workstreams

### W1: Profiling and crossover calibration

- Capture rank-aligned host and CUDA timelines.
- Separate forward, reverse, optimizer, factorization, NCCL and idle time.
- Record message sizes, bucket counts, synchronization and rank imbalance.
- Sweep qubit, bond and batch independently.
- Produce a decision surface for one, two, four and eight GPUs.

Deliverable: a predeclared crossover report explaining why eight GPUs win or
lose in every selected region.

### W2: Communication and ownership

- Bucket owner gradients.
- Pack adjacent boundary messages.
- Overlap P2P communication with independent site work.
- Reuse descriptors, payload buffers and communicators.
- Add topology-aware rank placement.
- Rebalance site and bond ownership using measured work rather than site count.
- Separate node-local and inter-node communication policies.

Deliverable: large-workload four-to-eight-GPU speedup of at least `1.5×`
without changing numerical semantics.

### W3: Factorization and contraction kernels

- Calibrate QR/SVD drivers by shape and dtype.
- Pool factorization workspaces.
- Implement batched two-site contraction.
- Fuse contraction, truncation and decomposition where safe.
- Provide small-bond Triton and medium/large-bond vendor-library paths.
- Add CUDA Graph capture for steady-state training.
- Investigate mixed precision with residual refinement.

Deliverable: shape-indexed kernel policy with correctness and cold/warm
performance evidence.

### W4: Memory and checkpointing

- Report owned tensors, reverse tape, optimizer, communication and
  factorization workspaces separately.
- Add checkpoint/rematerialization policies for the reverse tape.
- Eliminate repeated allocation in steady-state execution.
- Bound allocator fragmentation and high-water growth.
- Support atomic rank-shard checkpoints and validated restart manifests.
- Add optional world-size-changing reshard as an experimental feature.

Deliverable: single-GPU and four-GPU capacity failures matched to successful
eight-GPU completion.

### W5: Algorithm completeness

- Stabilize MPO expectation and Hamiltonian lowering.
- Add one-site/two-site DMRG with sweep and convergence records.
- Add TEBD and one-site/two-site TDVP.
- Add MPS sampling, marginals and reduced density matrices.
- Add symmetry-aware block-sparse tensors after dense correctness freezes.
- Add noisy and dynamic MPS trajectories as experimental capabilities.

Deliverable: frozen scientific reference workloads with energy, variance,
conserved quantities and truncation-error checks.

### W6: Multi-node and portability

- Validate GPUDirect RDMA and topology discovery.
- Implement hierarchical intra-node/inter-node communication.
- Test timeout, rank failure and bounded cleanup.
- Certify NVIDIA first; keep FlagOS and other accelerators fail closed until
  native kernels, collectives and no-fallback evidence exist.

Deliverable: a multi-node workload whose capacity exceeds one node, with
forward/backward/optimizer ownership preserved.

### W7: Planner and productization

- Feed measured calibration into runtime selection.
- Predict memory, bond growth, communication and expected error.
- Select local/MPS/TN and one/two/four/eight GPUs transparently.
- Emit an explanation and blockers with every decision.
- Freeze stable public APIs only after conformance and migration tests pass.

Deliverable: reproducible planner decisions backed by versioned calibration,
never by hard-coded marketing thresholds.

## Milestones

### M1 — FlagOS 2.2 / 2026-08-31

Goal: trustworthy single-node development capability.

- [x] Fail-closed 8×A800 preflight.
- [x] Auditable 1/2/4/8-GPU Adam development matrix.
- [ ] Configure controlled evidence signing.
- [ ] Complete crossover sweep for bond 64/128/256/512.
- [ ] Add complex128 and SGD scheduled coverage.
- [ ] Explain and reduce the current four-to-eight-GPU regression.

Promotion boundary: remain `development_evidence`.

### M2 — FlagOS 2.3 / 2026-11-30

Goal: single-node performance leadership in declared regions.

- [ ] Communication buckets and overlap.
- [ ] Shape-calibrated QR/SVD policy and workspace pool.
- [ ] CUDA Graph steady-state path.
- [ ] Matched competitor benchmark suite.
- [ ] One-GPU and four-GPU OOM to eight-GPU completion gates.
- [ ] 1,000-step Adam/SGD stability.

Promotion boundary: production support may be considered only for the exact
single-node envelope that passes every gate.

### M3 — FlagOS 2.4 / 2027-02-28

Goal: broad algorithms and genuine multi-node training.

- [ ] Multi-node rank-owned forward/backward/optimizer.
- [ ] Checkpoint/recovery and fault injection.
- [ ] DMRG, TDVP and stable MPO workflows.
- [ ] Planner calibration across local and distributed MPS.
- [ ] Signed release evidence and third-party reproduction.

Promotion boundary: multi-node and new algorithms are promoted separately;
passing one capability does not promote the others.

## Immediate sprint backlog

### Sprint 1 — Explain eight-GPU regression

- [ ] Run qubit/bond/batch crossover sweep on 8×A800.
- [x] Separate constant-sites-per-rank profiling from fixed-problem strong
      scaling; freeze the corrected 64-site, bond-64 4/8-GPU matrix in
      `benchmarks/manifests/mps_crossover_sprint1_v1.json`.
- [x] Capture four-GPU and eight-GPU PyTorch profiler traces; NSYS is not
      installed on the current A800 host and remains a follow-up.
- [ ] Reconcile CUDA, NCCL, host and idle time with an NCCL-visible trace.
- [x] Quantify per-rank site/bond work and boundary-message communication.
- [x] Publish the first non-release crossover diagnosis report.

### Sprint 2 — Communication and balance

- [x] Implement and certify dtype/owner gradient buckets; the 8×A800,
      1000-parameter development run reduces collectives from 1000 to 8 with
      exact loss, gradient, SGD and Adam parity.
- [x] Pack compatible boundary messages and reuse packed send/receive buffers;
      8×A800 microbenchmark evidence records 50% fewer physical messages
      without logical-byte growth.
- [x] Overlap boundary-halo communication with independent layer work; frozen
      A/B evidence improves eight-GPU mean step time by 38.9% and four-GPU by
      1.2%, while preserving finite losses and phase reconciliation.
- [x] Add measured communication-aware ownership balancing; the frozen A/B
      reduces eight-GPU boundary messages by 46.2% and boundary bytes by
      64.4%, improving mean step time by 84.8% at the cost of more compilation.
- [x] Re-run the frozen matrix without changing acceptance thresholds; the
      correctness gate passes but the final 4→8 speedup is only 0.861× versus
      the 1.5× target, so Sprint 2 closes without performance promotion.

### Sprint 3 — Large-bond kernels

- [x] Calibrate bond 128/256/512 SVD drivers on A800: retain `gesvd` for
      strict mode and `gesvda` only for explicit approximate mode; portability
      remains required before automatic production selection.
- [x] Calibrate the no-truncation reduced-QR path for the same bond matrix;
      full MPS transfer adds below 5% over direct QR and preserves reconstruction
      and orthogonality errors below 2e-6 on A800.
- [x] Pool application-controlled RXX factorization staging workspaces with
      bounded shape/device/dtype/role leases and CUDA event-safe reuse.
- [x] Fail closed on non-finite compiled-kernel and SVD outputs, with strict
      isolated-CUDA/CPU-LAPACK recovery for solver failures.
- [x] Initially quarantine the batched two-site path when `max_bond >= 128`; an 8×A800,
      64-wire, depth-4 run now passes with zero allocator retry/OOM. This is a
      correctness recovery, not a performance promotion.
- [x] Narrow the large-bond quarantine: restore compiled batched contraction,
      isolate only the unstable batched factorization at `max_bond >= 128`,
      and retain strict per-matrix SVD. The same 8×A800 workload passes with
      zero solver fallback, allocator retry, or OOM; staging-pool reuse remains
      quarantined and this single run is not a performance promotion.
- [x] Investigate solver-native workspace control below `torch.linalg`: the
      CUDA 13 runtime exposes classic complex `gesvd` and 64-bit `Xgesvd`
      device/host workspace APIs, while PyTorch 2.13 exposes no public
      caller-owned workspace contract. Production integration therefore
      requires a compiled, stream-aware C++/CUDA extension; direct `ctypes`
      execution and private ATen ABI coupling are explicitly rejected.
- [ ] Fuse common contraction/decomposition regions.
- [ ] Add steady-state CUDA Graph evaluation.

### Sprint 4 — Capacity and comparison

- [ ] Produce matched one-GPU/four-GPU failure and eight-GPU completion.
- [ ] Run cuTensorNet/CUDA-Q/PennyLane comparisons.
- [ ] Report error, throughput, memory and cost together.
- [ ] Seal and audit eligible candidates.

## Progress protocol

Each completed item must reference:

1. the implementation commit;
2. the exact command and workload manifest;
3. raw measured artifacts;
4. the audit or correctness result;
5. the supported claim and remaining blockers.

Update this table as work lands:

| Date | Milestone | Item | Commit | Evidence | Result | Blockers |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-28 | M1 | 1/2/4/8 A800 Adam matrix | `bd816c6` | `benchmarks/results/smoke/release_candidates/mps_single_node/` | Development audit valid | Unsigned; 8 GPU slower than 4 GPU |
| 2026-07-28 | M1 | Fixed 64-site bond-64 4/8 GPU profile | `0de6174` | `benchmarks/results/smoke/release_candidates/mps_crossover_sprint1/` | Development audit valid; 4→8 speedup 0.906×; boundary messages grow 2.6× | NCCL-visible transport breakdown not captured; boundary transport optimization required |
| 2026-07-28 | M2 | Packed boundary transport and buffer reuse | `05154c4` | `benchmarks/results/smoke/release_candidates/mps_boundary_transport/` | 8×A800 development audit valid; physical messages −50%; logical bytes +0%; total rank P2P wait −18.7% | Microbenchmark only; frozen end-to-end matrix must be repeated |
| 2026-07-28 | M2 | Boundary-halo overlap A/B | `4991374` | `benchmarks/results/smoke/release_candidates/mps_halo_overlap/` | 8-GPU mean step time improves 38.9%; 4-GPU improves 1.2%; correctness checks pass | Overlap-enabled 4→8 speedup remains 0.923×; single unsigned A/B run |
| 2026-07-28 | M2 | Communication-aware ownership A/B | `35bdd03` | `benchmarks/results/smoke/release_candidates/mps_communication_aware_ownership/` | 4-GPU improves 6.4%; 8-GPU improves 84.8%; messages −46.2%; bytes −64.4% | Aware 4→8 speedup remains below 1×; compile setup grows about 3.1× |
| 2026-07-28 | M2 | Owner gradient buckets | `357856c` | `benchmarks/results/smoke/release_candidates/mps_gradient_buckets/` | 8×A800 collectives 1000→8; measured backward 3.45× faster; exact optimizer parity | Specialized microbenchmark; single unsigned run |
| 2026-07-28 | M2 | Final optimized frozen matrix | `9898286` | `benchmarks/results/smoke/release_candidates/mps_sprint2_closeout/` | Correctness passes; 4 GPU 0.487s, 8 GPU 0.565s | 4→8 is 0.861×, below 1.5× target; repeated trials and Sprint 3 kernels required |
| 2026-07-28 | M3 | A800 SVD driver calibration | `8eb8657` | `benchmarks/results/smoke/release_candidates/mps_svd_policy/` | Strict: `gesvd`; explicit approximate: `gesvda`, 6.7×–11.1× faster with measured residual below 1e-6 | A800-only; QR matrix and cross-device portability remain |
| 2026-07-28 | M3 | A800 no-truncation QR calibration | `876decd` | `benchmarks/results/smoke/release_candidates/mps_qr_policy/` | Bond 128/256/512 full update 0.815/2.01/5.24 ms; errors below 2e-6 | Batch-1 complex64 A800 only; portability remains |
| 2026-07-28 | M3 | Factorization staging workspace pool | `a9e9ec2` | `benchmarks/results/smoke/release_candidates/mps_workspace_pool/` | 8×A800 stable-shape run: every rank reuses buffers; plateau passes; zero allocator retry/OOM | Solver workspace remains internal; bond-128 depth-4 `gesvd` convergence blocker observed |
| 2026-07-28 | M3 | Large-bond correctness quarantine | `889bf31` | `benchmarks/results/smoke/release_candidates/mps_large_bond_quarantine/` | 8×A800, 64 wires, depth 4, bond 128 passes plateau with zero allocator retry/OOM and no SVD fallback | Non-release smoke only; batched two-site execution and staging-pool reuse remain quarantined at `max_bond >= 128` |
| 2026-07-28 | M3 | Hybrid large-bond contraction/factorization | `3914274` | `benchmarks/results/smoke/release_candidates/mps_large_bond_hybrid/` | Compiled batched contraction restored; per-matrix strict factorization; 8×A800 plateau passes with zero fallback/retry/OOM | Non-release single run; batched large-bond factorization and staging-pool reuse remain quarantined |

## Claim policy

- Never report qubit count without bond, depth, batch, error and memory.
- Never compare runtimes with different numerical error budgets.
- Never call replicated execution sharded scaling.
- Never promote a cherry-picked point without the frozen matrix.
- Never edit measured artifacts to satisfy a schema; regenerate them.
- Never put signing keys in source, artifacts, shell history or documentation.
- Never infer multi-node readiness from single-node NCCL.
- Prefer an explicit blocker over an unsupported success claim.
