# Noisy simulation

FlagQuantum uses one backend-neutral `NoiseModel` for exact density-matrix
evolution, batched statevector trajectories, and MPS quantum trajectories. The
exact path is the small-system correctness oracle; trajectory paths report
sampling statistics, and the MPS path additionally reports truncation data.

## Quick start

```python
import flagquantum as fq
import flagquantum.runtime as fqr
import flagquantum.noise as fqn
from flagquantum.runtime.planner import plan_noise_execution_selection

circuit = fq.Circuit(2).h(0).cx(0, 1)
noise = (
    fqn.NoiseModel()
    .add("h", fqn.thermal_relaxation_channel(
        t1=50_000,
        t2=70_000,
        duration=35,
    ))
    .add("cx", fqn.depolarizing_channel(0.01))
    .add_readout(0, fqn.ReadoutError(((0.98, 0.02), (0.07, 0.93))))
)

exact = fq.run(
    circuit,
    noise_model=noise,
    options=fq.ExecutionOptions(mode="density_matrix"),
    outputs=fq.expectation(fq.Z(0) + fq.Z(1)),
)
exact_z = exact.expectation()

sampled = fqr.run_noisy_mps(
    circuit,
    noise,
    trajectories=4096,
    min_trajectories=128,
    target_standard_error=1e-3,
    seed=42,
    retain_trajectories=False,
)

print(noise.identity)
print(exact_z)
print(sampled.expectation_z_mean)
print(sampled.statistics.standard_error)
print(sampled.converged, sampled.stopped_early)
```

## CPU counts beyond the dense-memory limit

The stable `fq.run(...)` entry point can select noisy MPS trajectories when the
exact density matrix exceeds an explicit memory budget. The caller must opt in
to approximation; otherwise planning fails rather than silently changing
semantics. This example is directly runnable on CPU:

```python
import flagquantum as fq
import flagquantum.noise as fqn

n_qubits = 24
circuit = fq.Circuit(n_qubits).h(0)
for wire in range(n_qubits - 1):
    circuit.cx(wire, wire + 1)

noise = (
    fqn.NoiseModel()
    .add("cx", fqn.depolarizing_channel(0.01))
    .add_readout(0, fqn.ReadoutError(((0.98, 0.02), (0.03, 0.97))))
)

result = fq.run(
    circuit,
    options=fq.ExecutionOptions(
        mode="auto",
        device="cpu",
        shots=1_000,
        seed=42,
        memory_limit_bytes=512 * 1024**2,
        allow_approximate=True,
    ),
    outputs=fq.counts(),
    noise_model=noise,
)

print(result.counts[0])
print(result.plan.noisy_execution_plan.summary())
print(result.measurement("counts").statistics)
```

The planner keeps exact density-matrix evolution when it fits. If it does not,
the memory budget determines a bounded MPS bond dimension and the serialized
plan records MPS representation, quantum-trajectory evolution, the noise-model
identity, and the trajectory policy. Restoring the plan from JSON executes the
same route without replanning.

`shots` and trajectory count are different error sources. Shots control the
final measurement histogram. The stable auto route currently uses 32 noise
trajectories and samples the empirical mixture of their retained MPS states;
measurement statistics report `sampling_semantics`, `trajectory_count`, and
`retained_trajectory_count`, plus the observed maximum bond, maximum trajectory
truncation error, and maximum discarded weight. Use the expert
`run_noisy_mps(...)` interface when the trajectory count, adaptive stopping
threshold, bond dimension, or cutoff must be controlled directly.

### Writing circuits for MPS

Users keep the same circuit API, but circuit structure matters after the
planner selects MPS:

- Put qubits that interact frequently next to each other in logical-wire order.
  Nearest-neighbour two-qubit gates are the most MPS-friendly; distant gates
  require internal swaps and increase work.
- Prefer shallow, local-entangling layers. One-dimensional hardware-efficient
  ansatzes, product states, and GHZ chains can remain low-bond at large width.
- Treat deep random circuits, dense all-to-all QAOA layers, and unoptimised QFT
  patterns as high-risk. They can create volume-law entanglement and make the
  required bond dimension exponential.
- Do not infer difficulty from gate count alone. The important quantity is the
  entanglement crossing every cut in the chosen wire ordering; reordering
  logical qubits can change MPS cost substantially without changing the
  algorithm.
- `shots` only reduces final measurement noise. It does not reduce trajectory
  error or MPS truncation error. Increasing `memory_limit_bytes` permits a
  larger bond; the expert API separately controls trajectories, `max_bond`,
  and `cutoff`.

Always inspect the returned diagnostics:

```python
stats = result.measurement("counts").statistics
print("bond:", stats["observed_max_bond"], "/", stats["configured_max_bond"])
print("cutoff:", stats["configured_cutoff"])
print("trajectory truncation:", stats["max_trajectory_truncation_error"])
print("discarded weight:", stats["max_discarded_weight"])
```

Reaching the configured bond is not by itself a failure, but reaching it while
discarded weight is non-zero means truncation occurred. There is no universal
acceptable threshold: rerun important workloads with a larger memory budget
or expert `max_bond` and require the observable or counts distribution to be
stable within the application's tolerance. Use an exact density-matrix result
on a reduced circuit as a correctness oracle when possible.

This raises capacity only for workloads whose entanglement remains compressible
at the selected bond dimension. It is not a promise that every 24-qubit noisy
circuit will be fast or accurate: highly entangled circuits may require a larger
memory budget, and trajectory plus truncation errors must be evaluated for the
workload.

### Capacity and limits

There is no fixed MPS qubit constant in the counts path. Capacity is controlled
by the bond dimension `chi`, not by qubit count alone. For the stable route,
which retains 32 trajectories so it can produce shot samples, the planned MPS
state memory is approximately:

```text
32 * batch_size * n_qubits * 2 * chi**2 * complex_bytes
```

`complex_bytes` is 8 for `complex64` and 16 for `complex128`. The planner
chooses the largest integer `chi` whose estimate fits `memory_limit_bytes`.
For example, with one circuit, `complex64`, 100 qubits, and a 512 MiB limit,
the current policy caps the bond at 102. Shot sampling has a separate bounded
workspace of at most 4,000,000 complex elements (about 30.5 MiB for
`complex64` or 61 MiB for `complex128`), plus PyTorch and factorization
workspace. The declared MPS-state estimate is therefore not a process-RSS
upper bound.

The implementation now verifies full-register counts on a 100-qubit CPU
product-state circuit without materializing an integer basis index. Explicit
`format="index"` samples remain limited to 63 qubits because they use signed
`int64`; ordinary `fq.counts()` and `format="bits"` do not have that artificial
limit. The checked-in performance benchmark provides noisy, entangled evidence
from 16 through 1,000 qubits for a low-bond GHZ chain with bond cap 8.

The 1,000-qubit check is a width test, not a general 1,000-qubit performance or
accuracy guarantee. Required Schmidt rank can grow as `2**(n_qubits / 2)` for
highly entangled circuits. Once the required rank exceeds `chi`, MPS truncates;
the result reports maximum observed bond, truncation error, and discarded
weight. Dominant two-qubit split work can grow cubically with `chi`, while
memory grows quadratically. Runtime also grows with circuit depth, non-local
gate routing, trajectory count, and shots. The stable route uses one CPU process
and 32 trajectories; public distributed noisy-MPS reduction is not implemented.

For a managed `quafu:<device>-sim` target, the usable qubit count is additionally
bounded by the logical wires covered by the selected physical-device
calibration and by the service's admission policy. FlagQuantum rejects a
profile that does not cover every logical wire. Consequently the honest
capacity statement is: up to 1,000 qubits are timed for this low-bond noisy GHZ
workload; every other circuit must be admitted by memory and calibration checks
and judged from its reported bond dimension and truncation evidence.

The 2026-09-24 arm64 CPU smoke run used a GHZ chain, `cx`
depolarizing probability 0.01, four trajectories, bond cap 8, cutoff `1e-10`,
and 1,000 output shots. Median end-to-end times over three runs were:

| Qubits | Median time | Observed max bond | Counts |
|---:|---:|---:|---:|
| 16 | 0.053 s | 2 | 1,000 |
| 20 | 0.062 s | 2 | 1,000 |
| 24 | 0.099 s | 2 | 1,000 |
| 50 | 0.183 s | 2 | 1,000 |
| 100 | 0.298 s | 2 | 1,000 |
| 200 | 0.553 s | 2 | 1,000 |
| 500 | 1.408 s | 2 | 1,000 |
| 1,000 | 3.026 s | 2 | 1,000 |

These timings include planning, noisy evolution, and counts. No truncation was
observed for this particular rank-2 workload. They are workload-specific
capacity evidence, not a general performance promise. The raw record is
[`cpu_noisy_mps_counts_20260924.json`](../../benchmarks/results/local/cpu_noisy_mps_counts_20260924.json).

For dense circuits that fit statevector memory, trajectories can be processed
in true tensor batches:

```python
sampled_sv = fqr.run_noisy_statevector(
    circuit,
    noise,
    trajectories=4096,
    trajectory_batch_size=64,
    seed=42,
)
print(sampled_sv.expectation_z)
print(sampled_sv.standard_error)
```

The state layout is
`[trajectory_batch, circuit_batch, 2**n_wires]`. Seeded random streams belong
to global trajectory IDs, so changing `trajectory_batch_size` preserves every
sampled trajectory. Pauli channels use a state-independent branch fast path;
amplitude damping uses a branch-state-free specialized kernel, and other Kraus
channels use batched probability, sampling, application, and normalization.
Multi-wire Kraus channels are supported by the generic path.

With `mode="auto"`, specifying `trajectories` selects this path when its
trajectory block satisfies the supplied memory budget. `max_bond` or `cutoff`
selects MPS, and an over-budget statevector block falls back to MPS.

## Auditable backend selection

`plan_noise_execution_selection` evaluates density matrix, batched
statevector trajectory, and MPS trajectory together. Every candidate reports
representation, evolution semantics, memory estimate, exact/sampling/
truncation error flags, eligibility, reasons, and explicit rejection reasons.
If every candidate exceeds policy or memory constraints, selection fails
instead of silently ignoring the budget.

```python
selection = plan_noise_execution_selection(
    circuit,
    noise,
    trajectories=512,
    trajectory_batch_size=64,
    world_size=8,
    memory_limit_bytes=80 * 1024**3,
)
print(selection.selected_mode)
print(selection.summary())
```

For the documented A800 workload, the selector counted 184 lowered channel
events with maximum Kraus rank 4 and selected `noisy_statevector`. The exact
density candidate was rejected because trajectories were explicitly
requested; MPS remained eligible but carried truncation-error semantics. The
reproducible snapshot is
`benchmarks/results/local/noise_selector_a800_20260806.json`.

The broader A800 calibration artifact covers 8/10/12/16 qubits, depth 4/8,
and local-only versus full gate-noise placement. Batched statevector timings
track lowered channel count closely. At 16 qubits and 248 channel events, 128
trajectories completed in 3.91 s with about 457 MiB peak allocation. On the
8-qubit, depth-8, 120-channel case, statevector completed 128 trajectories in
1.61 s, exact density took 3.40 s, and MPS took 13.32 s for only 8
trajectories. These results demonstrate why memory size alone is not a speed
model; MPS remains valuable for capacity and low-entanglement regimes.

![A800 selector calibration](../../benchmarks/results/local/noise_selector_calibration_a800_20260806.png)

Raw measurements are in
`benchmarks/results/local/noise_selector_calibration_a800_20260806.json`.

The artifact can be supplied as an optional selector cost input:

```python
selection = plan_noise_execution_selection(
    circuit,
    noise,
    trajectories=128,
    trajectory_batch_size=64,
    calibration="benchmarks/results/local/noise_selector_calibration_a800_20260806.json",
)
```

A timing is used only when the circuit digest, noise-model identity, wire and
lowered-channel counts, backend mode, trajectory batch size, and world size
match. If both eligible trajectory backends match, the measured-time estimate decides; an
incomplete or mismatched calibration falls back to the analytic policy. The
8-qubit single-A800 full-noise decision and all candidate evidence are preserved in
`benchmarks/results/local/noise_selector_a800_calibrated_20260806.json`. Public
`flagquantum.runtime.run_native(..., noise_performance_calibration=...)` uses
the same policy.

For adaptive runs, the selector also reports a time-to-target trajectory
estimate when both a standard-error target and a hard trajectory ceiling are
provided:

```python
selection = plan_noise_execution_selection(
    circuit,
    noise,
    trajectories=4096,
    min_trajectories=128,
    target_standard_error=0.02,
)
```

Without a workload-specific variance pilot, it uses the explicit conservative
planning assumption `Var(Pauli) <= 1`, giving
`ceil(1 / target_standard_error**2)` trajectories, bounded below by
`min_trajectories`. The report states whether that estimate fits inside the
requested ceiling. This is a cost-planning estimate, not a convergence
guarantee; runtime stopping still uses the observed global standard error.

A reproducible pilot can replace the worst-case variance assumption:

```bash
python benchmarks/export_noise_pilot_selection.py \
  --pilot-trajectories 32 --trajectory-cap 4096 \
  --trajectory-batch-size 64 --target-standard-error 0.02 \
  --seed 20260806 \
  --calibration benchmarks/results/local/noise_selector_calibration_a800_20260806.json \
  --output benchmarks/results/local/noise_selector_pilot_a800_20260806.json
```

On the documented single-A800 workload, the fixed-ID pilot measured maximum
sample variance 0.04815 across the eight Z observables. Its point estimate is
121 trajectories for standard error 0.02. However, the 95% bounded-variance
upper confidence limit with Bonferroni correction across eight observables is
still 1.0 at only 32 pilot samples, requiring the conservative 2,500
trajectories. Calibrated time-to-target is therefore 31.41 s for statevector,
while exact density takes 3.40 s and is selected because it has no sampling
error.
The raw per-observable pilot means, variances, IDs, seed, circuit/model
identities, confidence level, simultaneous observable count, and decision are
stored in the JSON artifact. Runtime stopping still uses observed error.

### Pilot-size study

The fixed-workload A800 curve tests 32, 128, 512, and 2,048 pilot trajectories:

![A800 pilot-size curve](../../benchmarks/results/local/noise_selector_pilot_curve_a800_20260806.png)

The simultaneous variance bound tightened from 1.0 to 0.5745, 0.2394, and
0.1104. The smallest conservative target count was 599 at a 512-trajectory
pilot; the 2,048-point estimate is floored by the pilot already spent. More
importantly, that 512 pilot took 6.51 s, already exceeding the calibrated
3.40 s exact-density solution. Density was therefore selected at every tested
point. Pilot planning is useful primarily after exact density has become
ineligible by memory or scale, rather than as mandatory overhead for small
systems. Raw evidence is in
`benchmarks/results/local/noise_selector_pilot_curve_a800_20260806.json`; the plotting
script exports PNG, PDF, and SVG.

### Distributed selector calibration

The formal 512-trajectory scaling measurements are also packaged as strict
world-size-specific selector calibrations under
the external evidence workspace. Exact workload
matching reports 11.07, 5.28, 2.87, and 1.52 s on 1, 2, 4, and 8 A800 GPUs,
respectively, and every corresponding decision records
`selection_basis="exact_device_calibration"`.

These distributed artifacts currently contain only the noisy-statevector
candidate. For `world_size > 1`, the selector marks noisy MPS ineligible with
`distributed_collective_not_implemented`: its low-level runner can produce
explicit rank-local results for later `merge_noisy_mps_results`, but public
`fq.run`/`run_native` does not yet infer ranks and collectively reduce MPS
statistics. Distributed `noisy_mps` requests without an explicit rank fail
closed; explicit rank-local execution remains available for manual merging. Thus these
snapshots do not claim a calibrated multi-GPU MPS comparison. An artifact for
one world size is rejected for every other world size. They can be regenerated
from the sealed raw benchmark JSON files with:

```bash
python benchmarks/build_distributed_noise_selector_calibrations.py \
  --inputs benchmarks/results/local/noisy_statevector_a800_{1,2,4,8}gpu_20260806.json \
  --output-dir artifacts/development/distributed_selector_calibrations
```

## Reproducible throughput benchmark

The same-workload benchmark compares trajectory block sizes while keeping the
circuit, noise model, seed, and global trajectory IDs fixed:

```bash
python benchmarks/benchmark_noisy_statevector_trajectories.py \
  --n-wires 12 --depth 8 --trajectories 256 \
  --batch-sizes 1 8 32 64 --repeats 3 --device cuda \
  --output benchmarks/results/noisy-statevector.json
```

It refuses to report results if a seeded expectation changes with batch size.
On CUDA it also records peak allocated and reserved memory. CPU smoke results
are useful for regression testing but are not GPU performance evidence.

The 2026-08-06 A800 evidence run used 512 fixed global trajectories on a
12-qubit, depth-8 workload with trajectory batch 64. Measured throughput was
46.26, 96.96, 178.27, and 337.01 trajectories/s on 1, 2, 4, and 8
A800-SXM4-80GB GPUs. The 8-GPU result is 7.29× the single-GPU throughput
(91.1% strong-scaling efficiency). Timings include the public `fq.run` planner,
runtime, and NCCL statistics reduction. These are medians of two executions;
raw JSON is retained under `benchmarks/results/` and should not be generalized
to other circuits or noise densities.

![A800 noisy-statevector scaling](../../benchmarks/results/local/noisy_statevector_a800_scaling_20260806.png)

Rank-local semantic execution is available through `rank` and `world_size`.
`merge_noisy_statevector_results` validates compatible plans, disjoint global
IDs, complete ownership, and noise-model identity before merging Welford
statistics. When torch.distributed is initialized, `fq.run` automatically
infers rank/world size, partitions global trajectory IDs, and collectively
reduces count/sum/sum². Retained trajectory states intentionally fail closed
in collective mode rather than triggering an implicit all-gather.

`min_trajectories` and `target_standard_error` also work collectively. Every
rank evaluates the same global standard error after a trajectory batch and
stops on the same round. The first implementation requires `trajectories` to
be divisible by `world_size * trajectory_batch_size`; unsupported schedules
fail before execution rather than risk mismatched collective ordering. An
8×A800 validation requested 512 trajectories with minimum 128 and target
standard error 0.03; it converged and stopped globally after 128 trajectories.
The raw result is
`benchmarks/results/local/noisy_statevector_a800_8gpu_adaptive_20260806.json`.

Statevector statistics checkpoints are atomic and tensor-only. They bind the
requested count, base seed, completed global IDs, Welford moments, circuit
digest, NoiseModel identity, trajectory batch size, rank, and world size. A
distributed checkpoint path must contain `{rank}` so ranks never overwrite one
another. Resume refuses changed circuits, models, seeds, schedules, or foreign
trajectory ownership. Retained quantum states are not checkpointed.

An 8×A800 interruption test wrote 8 rank-local checkpoints after 128/512
global trajectories and completed 512/512 after a new torchrun launch. Against
the uninterrupted execution, maximum expectation and variance differences
were `1.79e-7` and `1.86e-8`. Raw resumed output is
`benchmarks/results/local/noisy_statevector_a800_8gpu_checkpoint_resume_20260806.json`.

With `continue_on_error=True`, a failed vectorized batch is retried one global
trajectory at a time. Successful trajectories still contribute to moments;
failed IDs retain exception type, message, and retryability in both results
and checkpoints. Collective completion gathers explicit ID lists, so holes are
never inferred from counts. An 8×A800 injected-failure run isolated global ID
3 on rank 3, completed 127/128 trajectories, reduced global statistics, and
exited all ranks without a collective hang. Raw evidence is
`benchmarks/results/local/noisy_statevector_a800_8gpu_failure_isolation_20260806.json`.

The complete executable version is
[`examples/noisy_simulation_v1.py`](../../examples/noisy_simulation_v1.py).

## Built-in channels

The current built-ins are:

- `bit_flip_channel(probability)`;
- `phase_flip_channel(probability)`;
- `depolarizing_channel(probability)`;
- `amplitude_damping_channel(gamma)`;
- `phase_damping_channel(gamma)`;
- `reset_error_channel(probability_zero, probability_one=0)`;
- `thermal_relaxation_channel(t1, t2, duration, excited_population=0)`;
- `coherent_overrotation_channel(angle, axis="x")`.

Every `KrausChannel` is validated at construction. Operators must be finite,
square, equal-sized, act on a power-of-two Hilbert space, and satisfy

```math
\sum_k K_k^\dagger K_k = I.
```

Invalid channels fail before compilation or execution.

## Reproducibility and identity

`NoiseModel.to_dict()` returns the versioned
`flagquantum.noise_model.v1` schema. `NoiseModel.from_dict()` validates and
restores it, while `NoiseModel.identity` is a SHA-256 digest of the canonical
device-independent representation.

MPS trajectories derive each random stream from `(base_seed,
global_trajectory_id)`. Checkpoint/resume and rank-local trajectory ownership
therefore preserve the trajectory IDs and random streams.

For a single-rank run, `target_standard_error` enables adaptive stopping after
at least `min_trajectories` have completed. The result records `converged` and
`stopped_early`; reaching the trajectory ceiling without meeting the target is
reported as `converged=False`. Distributed adaptive stopping fails closed until
a collective statistics reduction is available.

`ReadoutError` is a classical true-to-observed confusion matrix. It is kept
separate from quantum Kraus evolution. `run_noisy_mps` applies configured
readout rules to its per-wire Z statistics, while
`NoiseModel.apply_readout_probabilities` can transform an explicit ideal
probability distribution.

## Calibration-driven timing and idle noise

`DeviceNoiseProfile` stores timestamped per-wire T1/T2/readout calibration and
gate durations in one declared time unit:

```python
profile = fqn.DeviceNoiseProfile(
    qubits=(
        fqn.QubitNoiseCalibration(
            0,
            t1=50_000,
            t2=70_000,
            readout_error=fqn.ReadoutError(((0.98, 0.02), (0.07, 0.93))),
        ),
        fqn.QubitNoiseCalibration(1, t1=48_000, t2=65_000),
    ),
    gate_durations=(
        fqn.GateDuration("h", 35),
        fqn.GateDuration("cx", 280),
    ),
    source="device-calibration-export",
    captured_at="2026-08-06T12:00:00+08:00",
    time_unit="ns",
)
noise = fqn.NoiseModel.from_device_profile(profile)
```

Lowering uses an ASAP wire-clock schedule. It inserts per-wire thermal
relaxation for gate duration, idle gaps before synchronization gates, and
terminal idle time. Every generated channel records its placement, duration,
time unit, source gate, and `device_profile_identity`. Missing gate duration or
qubit calibration fails closed rather than silently assuming zero noise.

This is an explicit Markovian timing approximation. A profile import does not
by itself prove that the model reproduces real hardware.

## Hardware agreement and reliability

**Current status: numerical correctness validated and workload-specific QPU
agreement measured; device-wide hardware fidelity remains unvalidated.** The
exact-density comparison tests show that the MPS engine implements the declared
noise channels within sampling and truncation error. The live evidence below
measures three circuits on one physical edge and one calibration snapshot per
device; it does not show that arbitrary circuits reproduce a Baihua, Shenglian,
or Dongling QPU. Do not interpret `device_profile_identity` or a `<device>-sim`
target name as a hardware-accuracy guarantee.

Hardware agreement belongs to the complete tuple `(device, calibration
snapshot, physical mapping, compiled circuit, noise-model version, workload)`;
there is no honest device-wide percentage that can be inferred from MPS width
or bond dimension. Report distribution agreement with total-variation distance
(TVD):

```text
TVD(p, q) = 0.5 * sum_x abs(p[x] - q[x])
agreement = 1 - TVD
```

For a claim about one managed simulator release, freeze a prospective circuit
suite and, for every circuit:

1. Capture the complete calibration snapshot used to build the noise model.
2. Preserve the same physical-qubit mapping and final compiled circuit for the
   simulator and QPU comparison.
3. Submit at least two independent QPU jobs so QPU-to-QPU repeatability is
   measured rather than assumed.
4. Record MPS-to-QPU TVD, ideal-to-QPU TVD, QPU-to-QPU TVD, shot count and a
   finite-shot uncertainty bound, plus MPS bond/truncation diagnostics.
5. Predeclare an application-specific maximum TVD. Accept the model only when
   MPS-to-QPU TVD meets that limit, improves on the ideal baseline, and is not
   materially larger than QPU repeatability plus finite-shot uncertainty.

Use several circuit families and mappings; one Bell circuit validates only that
exact circuit on those physical qubits at that calibration time. More shots
reduce histogram uncertainty but do not repair an incomplete noise model, stale
calibration, too few quantum trajectories, or MPS truncation.

The 2026-09-24 paired run used `|00>`, `|11>`, and Bell circuits, one fixed
physical edge per device, 8,192 simulator shots, and two independent 1,024-shot
QPU repetitions. Local MPS used 4,096 trajectories, bond cap 4, and zero
cutoff; observed bond was at most 2 and truncation error was zero. Values below
are TVD averaged over the three circuits and both QPU observations:

| Device | Local MPS to QPU | Online `<device>-sim` to QPU | QPU repeatability | Ideal to QPU |
|---|---:|---:|---:|---:|
| Dongling | 4.20% | 6.39% | 2.51% | 16.10% |
| Shenglian | 2.48% | 2.55% | 2.08% | 8.25% |
| Baihua | 2.08% | 2.28% | 0.94% | 4.46% |
| Mean | 2.92% | 3.74% | 1.84% | 9.60% |

This is positive evidence: the local MPS model improved substantially over the
ideal baseline on this suite. It is not a device-wide acceptance result. Its
mean MPS-to-QPU distance remains above mean QPU repeatability, Dongling `|00>`
was about 5.26% from QPU while QPU repeatability was about 0.68%, and the
distribution-free 95% TV radius for one 1,024-shot four-outcome histogram is
about 5.25%. More circuits, mappings, shots, and repetitions are required for a
tighter claim.

The public Task API snapshots contain T1/T2 and gate/readout fidelities but no
gate durations. The local MPS evidence therefore uses one-qubit/CZ depolarizing
channels plus independent readout confusion and excludes timing-derived T1/T2
relaxation. It also uses the expert 4,096-trajectory route, not the stable
managed route's current 32-trajectory policy. The deployed online simulators
are reported separately and are not assumed to run this PR's code. Raw counts,
task IDs, calibration IDs and hashes, mappings, per-circuit TVD, and all claim
boundaries are preserved in
[`quafu_noisy_mps_qpu_validation_20260924.json`](../../artifacts/development/quafu_noisy_mps_qpu_validation_20260924.json).

An older Dongling task record that lacks its complete calibration payload is
not used in these numbers. The paired run instead restores each submission's
exact snapshot through `/api/v1/calibrations/{calibration_id}`; comparing with a
newer “current” calibration would mix device drift into the result.

FlagQuantum's QPU Twin validation tools already report Twin-to-QPU TVD,
ideal-to-QPU TVD, QPU repeatability, and simultaneous finite-shot TV bounds; see
[QPU Digital Twin](QPU_DIGITAL_TWIN.md). Managed-target evidence reuses those
definitions rather than introducing a second notion of “accuracy.”

## Current support boundary

- Density-matrix evolution is exact but requires exponential memory.
- Batched statevector trajectories support arbitrary gates and single- or
  multi-wire Kraus channels; large target arity remains exponentially costly.
- Single-wire MPS channels sample Kraus branches directly in MPS form.
- Multi-wire MPS channels sample the correct Kraus branch but currently use an
  explicitly dense statevector correctness fallback before rebuilding the MPS.
- Rank-local trajectory partitioning is semantic parallel ownership, not
  evidence of production multi-GPU scalability.
- Pulse overlap, crosstalk, leakage, calibration-provider adapters,
  distributed adaptive stopping, noisy gradients, and production multi-GPU
  scheduling are not yet supported.

See [Known limitations](../reference/KNOWN_LIMITATIONS.md) and the
[Noisy simulation roadmap](../roadmap/NOISY_SIMULATION_ROADMAP.md).
