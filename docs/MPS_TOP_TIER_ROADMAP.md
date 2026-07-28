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

- [ ] Implement gradient buckets.
- [ ] Pack boundary messages.
- [ ] Overlap communication and independent factorization.
- [ ] Add measured ownership balancing.
- [ ] Re-run the frozen matrix without changing acceptance thresholds.

### Sprint 3 — Large-bond kernels

- [ ] Calibrate bond 128/256/512 QR/SVD drivers.
- [ ] Pool workspaces.
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

## Claim policy

- Never report qubit count without bond, depth, batch, error and memory.
- Never compare runtimes with different numerical error budgets.
- Never call replicated execution sharded scaling.
- Never promote a cherry-picked point without the frozen matrix.
- Never edit measured artifacts to satisfy a schema; regenerate them.
- Never put signing keys in source, artifacts, shell history or documentation.
- Never infer multi-node readiness from single-node NCCL.
- Prefer an explicit blocker over an unsupported success claim.
