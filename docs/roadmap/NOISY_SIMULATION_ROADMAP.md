# FlagQuantum Noisy Simulation Implementation Roadmap

## 1. Decision and Positioning

FlagQuantum should begin developing noisy simulation now. Existing `KrausChannel`,
`NoiseModel`, exact density matrices, MPS trajectories, and SV/MPS/TN planning and
multi-GPU execution provide a foundation.

The primary goal extends beyond adding noise channels:

> Build a trustworthy, scalable, verifiable quantum trajectory system that reaches
> target accuracy on multiple GPUs with explicit confidence intervals.

Use exact density matrices as small-system correctness references, distributed SV
trajectories as the general scaling path, and MPS trajectories for low-entanglement
structure. Develop MPO and noisy TN after those capabilities stabilize.

First target API:

```python
result = fq.run(
    circuit,
    noise_model=noise,
    observables=hamiltonian,
    mode="auto",
    trajectories="auto",
    target_standard_error=1e-3,
    seed=42,
)
```

The system should:

1. Lower the noise model to unified Channel IR.
2. Select density matrix, SV trajectory, or MPS trajectory execution.
3. Determine trajectory counts from target statistical error.
4. Schedule trajectories across GPUs.
5. Return expectations, variance, standard error, confidence intervals, truncation error.
6. Record random streams, noise model identity, and execution audits.
7. Cross-check against small-system density-matrix execution.

## 2. Overall Architecture

```text
DeviceNoiseProfile / NoiseModel
                 │
                 ▼
          Unified Channel IR
                 │
                 ▼
        Noisy execution planner
          ├─ exact_density
          ├─ statevector_trajectory
          ├─ mps_trajectory
          └─ future
              ├─ mpo_density
              └─ tensor_network_trajectory
                 │
                 ▼
       statistics and error accounting
```

### 2.1 Noise Description Layer

Gradually organize noise implementation as:

```text
flagquantum/noise/
├── channels.py
├── model.py
├── device_profile.py
├── lowering.py
├── validation.py
└── serialization.py
```

Descriptions must be backend-independent. One `NoiseModel` should serve density
matrices, SV, MPS, and future MPO/TN.

Noise placements include:

- Before and after gates.
- Duration-dependent idle noise.
- Initialization, reset, and readout noise.
- Coherent over-rotation.
- Later crosstalk, leakage, and temporally correlated noise.

### 2.2 Unified Channel IR

Channel instructions retain origin and timing alongside Kraus matrices:

```python
ChannelInstruction(
    channel_type="kraus",
    wires=(0,),
    parameters={...},
    placement="after_gate",
    source_gate_id="gate-17",
    duration_ns=35.0,
)
```

Minimum instruction set:

- `KrausChannel`
- `LindbladSegment` (later)
- `ReadoutChannel`
- `ResetChannel`
- `ClassicalCondition`

### 2.3 Execution Plan

`NoisyExecutionPlan` records at least:

- Trajectory count and batch size.
- GPU/rank allocation.
- State/workspace device-memory estimates.
- Target standard error and maximum trajectory count.
- MPS bond/cutoff and truncation budget.
- Random-stream policy.
- Checkpoint policy.
- Candidate backend rejection reasons.

## 3. Phased Implementation

## Phase 0: Freeze Semantics and Result Contracts

Estimate: 3–5 working days.

Work:

- Define channel/gate application order.
- Define batch, wire, dtype, and device propagation.
- Fix random-number and cross-world-size reproducibility semantics.
- Separate physical channel noise from measurement sampling fluctuations.
- Define versioned result schemas and model identity.

Proposed result type:

```python
@dataclass(frozen=True)
class NoisyExecutionResult:
    expectation: Tensor
    variance: Tensor
    standard_error: Tensor
    confidence_interval: Tensor
    trajectory_count: int
    effective_sample_size: float
    truncation_error_bound: float | None
    noise_model_identity: str
    execution_summary: dict
```

Acceptance:

- Identical seeds reproduce exactly.
- Different seeds agree within statistical tolerances.
- NoiseModel serialization and identity are stable.
- Identity or measurement-summary tampering fails checks.
- Non-CPTP or incorrectly dimensioned channels fail closed.

## Phase 1: Correctness Golden Path

Estimate: one week.

Initial channels:

- Bit flip.
- Phase flip.
- Depolarizing.
- Amplitude damping.
- Phase damping.
- Thermal relaxation.
- Reset error.
- Readout confusion matrices.
- Coherent over-rotation.

Strengthen density execution for arbitrary one-/two-qubit Kraus operators,
batches, arbitrary wires, observables, sampling, complex64/complex128, and required
autograd behavior.

Verify:

```math
\rho=\rho^\dagger,\qquad
\operatorname{Tr}(\rho)=1,\qquad
\rho\succeq0.
```

Test 1–8 qubits, probability boundaries, entangled circuits, and multiple
observables against analytic solutions and independent implementations.

Acceptance:

- Trace, Hermiticity, and positivity meet dtype-specific tolerances.
- Small-system analytic results match for every built-in channel.
- Trajectory means cover density references within appropriate confidence intervals.

## Phase 2: Production-Grade MPS Trajectories

Estimate: 1–2 weeks.

Extend existing MPS trajectories with:

- Stable global trajectory IDs and seed derivation.
- Online Welford means/variances.
- Adaptive stopping.
- Trajectory checkpoint/resume.
- Separate cumulative truncation-error reporting.
- Multiple observables per execution.
- Per-trajectory failure isolation.

Target interface:

```python
result = fq.run_noisy_mps(
    circuit,
    noise,
    observables=observables,
    trajectories=1024,
    target_standard_error=1e-3,
    max_trajectories=8192,
    max_bond=256,
    cutoff=1e-8,
    seed=7,
)
```

After each trajectory batch, update:

```math
\mathrm{SE}=\frac{s}{\sqrt{N}}.
```

Stop when all target observables meet absolute/relative error thresholds. Use
`min_trajectories` to avoid accidental zero variance from too few samples.

Acceptance:

- Deterministic channels match density execution.
- Stochastic confidence intervals cover density references.
- Adaptive stopping does not systematically underestimate error.
- Checkpoint restoration agrees with uninterrupted execution.
- Statistical and MPS truncation errors are reported separately.

## Phase 3: Batched SV Trajectories

Estimate: two weeks; the most important initial performance phase.

Progress (2026-08-06): the first verifiable milestone includes actual trajectory
batch layouts, arbitrary unitaries, batched sampling/normalization for single-/
multiqubit Kraus channels, Pauli fast paths, amplitude damping without branch-state
materialization, readout-Z postprocessing, global-ID random streams, memory-budget
automatic selection, and density cross-checks. CUDA peaks, batched throughput, and
multi-GPU performance evidence remain incomplete, so the phase has not passed
full acceptance.

State layout:

```text
[trajectory_batch, circuit_batch, 2^n]
```

Implement:

- Batched single-qubit Kraus channels.
- Batched two-qubit Kraus channels.
- Fused probability, sampling, application, and normalization.
- Pauli-channel fast paths.
- Amplitude-damping fast paths.
- Readout-error postprocessing.

Sample Pauli noise and reuse X/Y/Z kernels instead of generic matrix multiplication.

Device-memory planning:

```math
M \approx
B_{\mathrm{trajectory}}
\times B_{\mathrm{circuit}}
\times 2^n
\times \text{complex bytes}
+ M_{\mathrm{workspace}},
```

The first four factors multiply. Choose trajectory batch size from available
memory; reduce the batch or fail closed when the budget is exceeded.

Acceptance:

- Batched and individual execution match for identical trajectory IDs.
- Clear throughput improvement over Python per-trajectory loops.
- CUDA reserved peak stays within the predicted budget.
- Measured time, throughput, and memory on 1/2/4/8 GPUs.

## Phase 4: Distributed Trajectory Scheduling

Estimate: 1–2 weeks.

Progress (2026-08-06): statevector trajectories support rank-local round-robin
global-ID ownership, world-size-independent seeds, and result merging with
completeness, duplicate-ID, and noise-model identity validation. Single-process
1/2/4-rank semantic tests match retained single-rank states trajectory by trajectory.
NCCL count/sum/sum² collectives are integrated into automatic public `fq.run`.
Fixed-workload 1/2/4/8-card strong-scaling measurements on 8×A800 reached 7.29×
speedup and 91.1% efficiency at eight cards.

Distributed adaptive stopping evaluates global standard error per batch. An 8×A800
run stopped uniformly at 128/512 trajectories; the initial version requires the
trajectory ceiling to be divisible by world size × batch. Atomic statevector
statistics checkpoints bind circuit/model/seed/schedule/rank ownership. An
eight-card run resumed from interruption at 128/512 to 512/512. Batch failures
fall back to individual trajectory retries; successful samples remain in statistics,
and failed IDs enter results/checkpoints. Injecting failure at global ID 3 on
eight cards completed 127/128 with clean collective exit. Core Phase 4 semantics
are complete; stronger fault tolerance still needs process/GPU/NCCL failure coverage.

The first layer uses trajectory parallelism. Each rank handles a subset of global
IDs, reducing only:

- `count`
- `sum(x)`
- `sum(x²)`
- Failure count.
- Maximum truncation error.

Use a world-size-independent counter-based RNG schema:

```math
\operatorname{seed}_t =
H(\operatorname{global\ seed},\operatorname{trajectory\ id}).
```

The second layer supports hybrid parallelism:

```text
trajectory groups × statevector shard ranks = world size
```

Eight GPUs may use 8×1, 4×2, 2×4, or 1×8. The planner searches combinations using
per-state size, communication cost, and target trajectory count.

Checkpoints retain:

- Next global trajectory ID.
- Count/sum/sum-square.
- RNG schema and seed.
- NoiseModel identity.
- Circuit digest.
- MPS truncation statistics.
- World-size-independent scheduling state.

Acceptance:

- Consistent statistics at world sizes 1/2/4/8.
- World-size changes do not alter a given trajectory ID's computation.
- Resumed results match uninterrupted execution.
- Hardware evidence for communication correctness, throughput, and tail latency.

## Phase 5: Noise-Aware Automatic Selection

Estimate: one week.

Progress (2026-08-06): density matrix, batched SV, and MPS trajectories share a
candidate report with lowered noise-event count, maximum Kraus rank, memory
estimates, exact/sampling/truncation error semantics, eligibility, and rejection
reasons. Existing auto dispatch uses it and fails closed if every candidate
exceeds budget. Versioned selection snapshots for measured A800 workloads are saved.

The first A800 calibration set covers 8/10/12/16 qubits, depths 4/8, and two noise
densities. SV time strongly correlates with lowered channel count; small-system
MPS throughput is substantially below batched SV. Versioned calibration artifacts
are optional cost inputs only when circuit digest, noise identity, channel count,
trajectory batch, and related fields match exactly; otherwise analytic policy
remains the fallback. An 8-qubit calibrated A800 decision snapshot is saved.

Target statistical error enters time-to-solution evidence. Without pilot variance,
the explicit `Var(Pauli) <= 1` bound estimates trajectory needs and whether the
ceiling suffices; actual global standard error still controls stopping. Fixed-seed/
trajectory-ID pilot variance records per-observable means, variances, and identities.
A 32-trajectory single-A800 pilot point estimate implied 121 trajectories. The
new 95% bounded-variance upper confidence bound, Bonferroni-corrected for eight
observables, still required 2,500, correctly selecting 3.40 s exact density over
estimated 31.41 s SV trajectories.

The single-A800 32/128/512/2048 pilot-size curve yielded upper bounds
1.0/0.5745/0.2394/0.1104. A 512 pilot already cost 6.51 s, exceeding 3.40 s exact
density, so every point selected density. This positions pilots for cases where
density is infeasible due to scale or memory. Formal 512-trajectory scaling data
became world-size-bound 1/2/4/8×A800 selector calibrations with matched times
11.07/5.28/2.87/1.52 s; all four decisions record `exact_device_calibration`.

Audit found that MPS supports only manual rank-local partitioning and post-hoc
merge primitives, without public rank inference/collective reduction. Therefore
`world_size>1` explicitly rejects `distributed_collective_not_implemented`, and
multicard noisy_mps requests without explicit ranks fail closed. Manual rank-local
execution plus merging remains. Next implement public distributed MPS statistics
aggregation, then collect cost evidence.

Candidate cost models:

- Density matrix.
- SV trajectories.
- MPS trajectories.
- Future MPO.
- Future TN trajectories.

Decision inputs:

- Qubits, batch, noise-event count.
- Mean/maximum Kraus rank.
- Target statistical error.
- Estimated MPS bond dimension.
- Output count/type.
- World size and device-memory limits.
- Exact/approximate policy.
- Gradient requirements.

Initial policy:

```text
Small system with exact_noise=True          -> density_matrix
Low entanglement/locality; approximation OK -> mps_trajectory
Other general circuits                     -> statevector_trajectory
Single-device memory insufficient           -> state sharding or try MPS
Every candidate exceeds budget              -> fail closed
```

Selections must include rejection reasons, memory predictions, trajectory-count
estimates, and error policy.

## Phase 6: Noisy Training and Backward

Start after forward execution stabilizes; estimate: 2–3 weeks.

Initial scope:

- Circuit-parameter gradients.
- Fixed noise parameters.
- Trajectory autograd.
- Common random numbers.
- Multi-GPU gradient-statistics reductions.

Defer unbiased gradients for arbitrary Kraus parameters and higher-order gradients.
For parameter perturbations or paired execution, reuse trajectory IDs and Kraus
random streams. Report gradient mean, standard error, signal-to-noise ratio, and
trajectory count.

Acceptance:

- Small-circuit gradients match density/autograd references.
- Common random numbers demonstrably reduce paired-gradient variance.
- Distributed gradients match single-card references.
- Training checkpoints include trajectory state.

## Phase 7: MPO and Noisy TN

Not blockers for the first release.

MPO priorities:

- One-dimensional local open systems.
- Avoiding Monte Carlo variance.
- Controlled operator entanglement.

Noisy TN priorities:

- Few local observables.
- Prunable causal cones.
- Low/moderate treewidth.
- Low Kraus rank.
- Sparse noise events.

Start with trajectory TN, reusing contraction, slicing, calibration, and multi-GPU
task parallelism. Avoid immediately building full doubled density networks that
may sharply increase effective treewidth.

## 4. Tests and Evidence

### 4.1 Correctness Evidence Chain

```text
Analytic solutions
  ↕
density matrix
  ↕
SV trajectory
  ↕
MPS trajectory
  ↕
future TN trajectory
```

Trajectory correctness requires confidence-interval coverage and repeated trials,
not fixed absolute-error checks alone.

### 4.2 Performance Metrics

Each backend records:

- Trajectories/s.
- Noisy gate-events/s.
- Time-to-target-error.
- Peak allocated/reserved memory.
- Trajectory batch size.
- GPU utilization.
- Communication time.
- Final standard error.

The primary metric is time to the same target accuracy:

```math
\text{Time To Target Error},
```

not individual trajectory forward time alone.

### 4.3 Physical Realism Evidence

After importing hardware calibration, compare:

- Ideal simulation.
- Simplified noise.
- Calibration-driven noise.
- Real hardware results.

Report unexplained residuals, missing calibration parameters, and independent-noise
assumptions. A simple depolarizing fit is not a complete reproduction of a device.

## 5. Noisy Simulation v1 Scope

First milestone:

> **Noisy Simulation v1: verifiable, parallel quantum trajectories**

Includes:

- Nine basic noise families.
- Unified NoiseModel/Channel IR.
- Density-matrix golden path.
- MPS trajectories.
- Batched SV trajectories.
- Single-node 1/2/4/8-GPU trajectory parallelism.
- Adaptive stopping and confidence intervals.
- Checkpoint/resume.
- Automatic backend selection.
- Forward API.

Excludes:

- Multinode production claims.
- MPO.
- Noisy TN doubled networks.
- Leakage.
- Non-Markovian noise.
- Complete noise-parameter gradients.
- Comprehensive error mitigation.

## 6. Priorities and Risk Controls

Implementation priorities:

1. Correctness and result schemas.
2. MPS trajectory statistics/recovery.
3. Batched SV trajectories.
4. Multi-GPU trajectory parallelism.
5. Adaptive error control.
6. Automatic selection.
7. Backward training.
8. Hardware calibration import.
9. MPO/noisy TN.

| Risk | Control |
| --- | --- |
| Statistical trajectory error mistaken for numerical error | Report variance, standard error, and truncation separately |
| World-size changes break reproducibility | Derive random streams from global trajectory IDs |
| Trajectory batches cause OOM | Reserved-memory preflight and automatic batch reduction |
| MPS truncation bias masks noise effects | Density cross-checks and separate cumulative truncation accounting |
| More channels without product value | Accept using time-to-target-error and real workloads |
| Premature noisy TN/MPO investment | Defer until v1 forward/distributed trajectories stabilize |
| Simplified models mistaken for hardware reproduction | Audit model identity, calibration time, and assumptions |

## 7. Definition of Done

Noisy Simulation v1 requires all of:

1. Correctness: density/SV/MPS trajectory cross-validation passes.
2. Statistics: repeated trials calibrate interval coverage and adaptive stopping.
3. Reproducibility: stable seeds, trajectory IDs, checkpoints, and world-size semantics.
4. Resource safety: memory preflight and fail-closed behavior on every GPU path.
5. Performance: real 1/2/4/8-GPU time-to-target-error evidence.
6. Observability: model identity, execution plan, error decomposition, runtime summary.
7. Claim boundaries: single-node evidence is not generalized to multinode or
   hardware realism claims.
