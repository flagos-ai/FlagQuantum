# Noise Code Organization and Architecture Review

## 1. Conclusion

FlagQuantum's top-level layering is sound, but complexity needs active management.
Noise development must not concentrate more functionality in a single Simulation
facade, `runtime/execution.py`, or the existing planner, repeating TN module growth
and internal-interface proliferation.

Keep three orthogonal dimensions:

- Noise is public physical semantics and compiler input.
- Quantum trajectories are a shared execution strategy.
- SV, density matrix, MPS, and TN are state-representation backends.

Noise is not another state backend alongside SV/MPS/TN. Do not express every
execution combination through an expanding set of mode strings.

## 2. Current Structure Assessment

### 2.1 Layers Worth Retaining

These ownership directions are sound:

```text
core/               IR, contracts, parameters
compilation/        Analysis, selection, execution plans
runtime/            Scheduling, execution, results, audits
runtime/executors/   SV, density, MPS, TN, JAX backends
ops/                Gate semantics, matrices, low-level lowering
testing/            Correctness and capability certification
benchmarking/       Formal benchmark protocols
docs/               Capabilities, architecture, claim boundaries
```

Continue organizing `runtime/executors/statevector`, `mps`, and `tensor_network`
by state representation.

### 2.2 P0: Low-Level Execution Depends on High-Level Circuit

The historical reverse dependency in `simulation/noise.py` has been removed.
`simulation/mps/entrypoints.py` still depends on high-level paths. Runtime TN/MPS
backends also import private kernels from Simulation modules.

The incorrect direction is:

```text
backend/simulation -> Circuit
```

The intended direction is:

```text
Circuit
   |
   v
Core IR
   |
   v
Compilation
   |
   v
Runtime/backend
   |
   v
Numerical kernels
```

Shared gate-matrix lowering belongs to `ops` or another low-level module. Backends
must not import private Circuit implementation.

### 2.3 P0: Noise Execution Semantics Are Lost in Plans

`mps_trajectory` and `noisy_mps` currently normalize to plain `mps`. Plans cannot
fully express:

- Single versus multiple trajectories.
- Trajectory count and batch size.
- RNG schema.
- Target standard error.
- MPS truncation error.
- Trajectory parallelism versus state sharding.
- Checkpoint state.

Future plans need orthogonal dimensions:

```text
StateRepresentation
    statevector | density_matrix | mps | tensor_network

EvolutionSemantics
    pure | exact_mixed | quantum_trajectory

ParallelStrategy
    local | trajectory_parallel | state_sharded | hybrid

DifferentiationStrategy
    autograd | adjoint | parameter_shift | paired_trajectory
```

### 2.4 P1: Central Execution Entry Point Keeps Growing

`runtime/execution.py` currently handles mode dispatch, parameter cleanup, noise
lowering, plan construction, execution, and result adaptation. More Noise modes
would add many `elif` branches and parameter combinations.

The target entry point is:

```python
request = normalize_request(...)
plan = planner.build(request)
executor = executor_registry.resolve(plan)
raw = executor.execute(plan, request)
return result_adapter.normalize(raw, plan)
```

Migrate existing branches individually; a complete rewrite is not required.

### 2.5 P1: TN Internal Interfaces and Files Keep Growing

Current pressure points include:

- `simulation/tensor_network/contraction.py` previously exceeded 1800 lines.
- `runtime/executors/tensor_network/execution.py` exceeds 1000 lines.
- The TN backend facade exposes many internal types and kernels.
- Several TN backend modules contain 600–800 lines.

During Noise development:

- Do not add noisy TN to `tensor_network/contraction.py`.
- Do not add internal exports to the TN `__init__.py`.
- Put new pair kernels in modules with explicit ownership.
- Do not begin noisy TN before SV/MPS trajectories stabilize.
- Gradually split `runtime/executors/tensor_network/execution.py` by execution
  responsibility without adding public concepts.

### 2.6 P2: Noise Modules Mix Responsibilities

Historical `simulation/noise.py` contained:

- Noise domain models.
- Channel factories.
- NoiseModel-to-IR lowering.
- Density-matrix kernels.
- Density-matrix execution.
- Expectation computation.

Although manageable at that stage, adding relaxation, readout, device profiles,
trajectories, statistics, and hardware imports would quickly create another monolith.

### 2.7 P2: NoiseModel Is Not Yet a Long-Term Public Model

The current model is mutable. Kraus channels retain device-bound Torch tensors
and lack:

- Stable schemas and identities.
- Explicit CPTP validation.
- Duration, idle, readout, and reset semantics.
- Hardware calibration provenance.
- Separation between specifications and compiled channels.

The long-term model should store serializable device-independent specifications.
Compilation then produces dtype/device-specific Kraus tensors.

## 3. Target Noise Layout

### 3.1 Public Noise Semantics

```text
flagquantum/noise/
├── __init__.py
├── channels.py
├── model.py
├── rules.py
├── device_profile.py
├── validation.py
└── serialization.py
```

Responsibilities:

- `channels.py`: immutable, serializable channel specifications.
- `model.py`: `NoiseModel`, rules, and placement.
- `rules.py`: gate, wire, idle, and readout matching.
- `device_profile.py`: hardware calibration and provenance.
- `validation.py`: probabilities, dimensions, CPTP, and composition rules.
- `serialization.py`: schemas, digests, identities, and round trips.

### 3.2 Noise Compilation

```text
flagquantum/compiler/noise.py
```

`noise.py` is the sole entry point transforming `CircuitIR + NoiseModel` into IR
with ChannelInstructions. It may depend only on Core, the noise domain, and ops
schemas, not Circuit, Runtime, Simulation, or concrete backends.

Noise backend selection and device calibration are execution policy. Their
authoritative entry points are `flagquantum/runtime/planner/noise_selection.py`
and `noise_calibration.py`. Subplans such as `NoisyExecutionPlan` temporarily
remain with protected execution-plan products in `flagquantum/compilation/models.py`
and are assembled by `runtime/planner`. They are not Compiler execution policy.

### 3.3 Density-Matrix Numerics

```text
flagquantum/simulation/density_matrix.py
```

Simulation owns `expand_operator`, `apply_unitary_density`, `apply_kraus_density`,
`density_matrix_from_ir`, and density expectations. `runtime/noise_registry.py`
coordinates noise lowering, execution-plan validation, and executor dispatch.
Numerical implementations do not depend on Runtime.

### 3.4 Shared Trajectory Runtime

```text
flagquantum/runtime/trajectories/
├── __init__.py
├── models.py
├── rng.py
├── statistics.py
├── scheduler.py
├── checkpoint.py
├── distributed.py
└── result.py
```

It centrally owns:

- Global trajectory IDs.
- World-size-independent RNG.
- Welford statistics.
- Adaptive stopping.
- Rank ownership.
- Checkpoint/resume.
- Failure accounting.
- Confidence intervals.

It does not own state-evolution kernels.

### 3.5 Backend-Specific Trajectories

```text
runtime/executors/statevector/
├── trajectory.py
└── noisy_kernels.py

runtime/executors/mps/
├── trajectory.py
└── noisy_operations.py

runtime/executors/tensor_network/
└── trajectory.py       # Later phase
```

Each backend advances one trajectory from compiled ChannelInstructions. Do not
reimplement statistics, scheduling, RNG, or checkpointing per backend.

## 4. Mandatory Dependency Rules

```text
noise
  -> core
  -> no runtime/simulation/circuit dependency

compiler.noise
  -> core + noise + ops schema
  -> no runtime/simulation/provider/backend dependency

runtime.trajectories
  -> core contracts + compilation plans
  -> no concrete SV/MPS/TN backend dependency

backend
  -> core + compiled plan + numerical kernel
  -> no private Circuit imports

api/circuit
  -> runtime public entry point
  -> must not be imported by backends
```

Gradually eliminate:

```python
from flagquantum.circuit import _gate_matrix
from flagquantum.simulation.tensor_network.contraction import _einsum_pair_by_labels
from flagquantum.simulation.mps.factorization import _split_pair_matrix
```

## 5. Architecture Preparation for Noise Development

### A1: Extract Gate-Matrix Lowering

Move `_gate_matrix` from high-level Circuit into low-level `ops`. Circuit,
density, MPS, and SV share it, eliminating reverse Noise/MPS-to-Circuit dependencies.

### A2: Establish the Noise Domain Package

Models, channel factories, lowering, and density-matrix numerics have moved to
their authoritative directories. The `simulation/noise.py` compatibility facade
was deleted after canonical-path replacement verification.

### A3: Establish a Structured NoisyExecutionPlan

Include at least:

```python
@dataclass(frozen=True)
class NoisyExecutionPlan:
    representation: StateRepresentation
    evolution: EvolutionSemantics
    trajectory: TrajectoryPlan | None
    parallel: ParallelPlan
    error_budget: NoiseErrorBudget
    memory: MemoryPlan
```

### A4: Introduce an Executor Registry

Support new Noise executors first, then migrate existing modes:

```python
executor_registry.register(
    representation="mps",
    evolution="quantum_trajectory",
    executor=MPSQuantumTrajectoryExecutor,
)
```

### A5: Limit Further Growth

- New Noise files cannot request large-file exceptions.
- Do not add combined-mode branches to `runtime/execution.py`.
- Do not duplicate RNG, statistics, checkpointing, or distributed ownership.
- Defer noisy TN until the shared trajectory runtime stabilizes.

## 6. Test Organization

```text
tests/noise/
├── test_channels.py
├── test_model.py
├── test_serialization.py
├── test_lowering.py
├── test_density_correctness.py
├── test_trajectory_statistics.py
├── test_trajectory_reproducibility.py
├── test_backend_selection.py
└── test_result_contract.py

tests/distributed/noise/
├── test_trajectory_ownership.py
├── test_world_size_invariance.py
├── test_checkpoint_resume.py
└── trajectory_runtime.py
```

Constraints:

- Domain tests do not import Runtime.
- Lowering tests do not initialize CUDA.
- Backend correctness uses the density path as reference.
- Distributed tests verify scheduling and statistics only.
- Hardware benchmarks do not enter ordinary unit tests.

## 7. Module Size Guidelines

| Type | Recommended maximum |
| --- | ---: |
| Domain/model module | 300–500 lines |
| Backend execution module | 600–800 lines |
| `__init__.py` facade | 150 lines |
| One public `__all__` | 30–40 symbols |
| Central Runtime entry point | 400–500 lines |
| One test file | 800–1000 lines |

Existing oversized files may temporarily have legacy exceptions with an owner and
removal version. New Noise modules must not depend on exceptions from the outset.

## 8. Recommended Sequence

```text
Architecture Preparation
    |
    v
Noise domain + serialization
    |
    v
Channel IR lowering
    |
    v
Exact density backend
    |
    v
Shared trajectory runtime
    |
    v
MPS trajectory migration
    |
    v
Batched SV trajectory
    |
    v
Distributed trajectory
    |
    v
Automatic selection
    |
    v
Gradients
    |
    v
MPO / noisy TN
```

Do not implement each backend independently and unify later. SV/MPS/TN must share
trajectory IDs, RNG, statistics, checkpointing, and error accounting.
