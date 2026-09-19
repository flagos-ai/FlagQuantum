# FlagQuantum Capabilities

Choose a supported workflow by user goal, runtime, hardware, and evidence level.
This catalog is generated from the machine-validated
[`capability-maturity.toml`](../../capability-maturity.toml) source of truth.

> Maturity applies only to the scope stated in each row. A local, replicated,
> sliced, or planned execution path is not distributed scalability evidence.

## How to read maturity

| Level | Meaning |
| --- | --- |
| **Release certified** | Release-gated with audited, reproducible evidence and no unresolved release blocker. |
| **Production supported** | Supported path with compatibility, operational guidance, and target-hardware evidence. |
| **Development evidence** | Executable and tested development result; not a production or general scalability claim. |
| **Experimental** | Research surface without compatibility or production guarantees. |

## Find a capability by goal

| Goal | Capability | Maturity | Start here |
| --- | --- | --- | --- |
| Build portable quantum circuits | Unified circuit API and FlagQuantum IR | Release certified | [Run example](../../examples/quick_start.py) |
| Compile or export a circuit | Unified circuit API and FlagQuantum IR | Release certified | [Run example](../../examples/quick_start.py) |
| Inspect a stable circuit representation | Unified circuit API and FlagQuantum IR | Release certified | [Run example](../../examples/quick_start.py) |
| Simulate a small or medium circuit exactly | Local statevector simulation and training | Production supported | [Run example](../../examples/single_machine_quantum_ai/01_vqe_statevector.py) |
| Train a parameterized quantum circuit | Local statevector simulation and training | Production supported | [Run example](../../examples/single_machine_quantum_ai/01_vqe_statevector.py) |
| Run local VQE and quantum machine learning | Local statevector simulation and training | Production supported | [Run example](../../examples/single_machine_quantum_ai/01_vqe_statevector.py) |
| Train one statevector workload across multiple ranks | Sharded statevector training | Production supported | [Run example](../../examples/distributed_statevector_topologies/run.sh) |
| Plan distributed statevector ownership | Sharded statevector training | Production supported | [Run example](../../examples/distributed_statevector_topologies/run.sh) |
| Inspect communication and sharding semantics | Sharded statevector training | Production supported | [Run example](../../examples/distributed_statevector_topologies/run.sh) |
| Check FlagQuantum and Torch-FL integration | FlagOS local statevector CUDA reference | Development evidence | [Run example](../../docs/reference/ACCELERATOR_PLATFORM_RUNTIME.md) |
| Audit the statevector operator profile | FlagOS local statevector CUDA reference | Development evidence | [Run example](../../docs/reference/ACCELERATOR_PLATFORM_RUNTIME.md) |
| Compare complex numerical behavior with a CPU complex128 reference | FlagOS local statevector CUDA reference | Development evidence | [Run example](../../docs/reference/ACCELERATOR_PLATFORM_RUNTIME.md) |
| Audit FlagOS distributed statevector workload support | FlagOS distributed statevector workloads | Development evidence | [Run example](../../docs/reference/FLAGOS_DISTRIBUTED_CONFORMANCE.md) |
| Run sharded FlagOS statevector forward workloads | FlagOS distributed statevector workloads | Development evidence | [Run example](../../docs/reference/FLAGOS_DISTRIBUTED_CONFORMANCE.md) |
| Run bounded sharded FlagOS training trajectories | FlagOS distributed statevector workloads | Development evidence | [Run example](../../docs/reference/FLAGOS_DISTRIBUTED_CONFORMANCE.md) |
| Audit matched FlagOS statevector capacity expansion | FlagOS statevector capacity expansion | Development evidence | [Run example](../../docs/reference/FLAGOS_STATEVECTOR_CAPACITY_F5.md) |
| Run one full-width complex128 statevector across eight ranks | FlagOS statevector capacity expansion | Development evidence | [Run example](../../docs/reference/FLAGOS_STATEVECTOR_CAPACITY_F5.md) |
| Inspect fail-closed single, replicated, and sharded capacity evidence | FlagOS statevector capacity expansion | Development evidence | [Run example](../../docs/reference/FLAGOS_STATEVECTOR_CAPACITY_F5.md) |
| Audit observable FlagOS collective behavior | FlagOS distributed transport observability | Development evidence | [Run example](../../docs/reference/FLAGOS_TRANSPORT_OBSERVABILITY_F6.md) |
| Inspect bounded profiler evidence for complex collectives | FlagOS distributed transport observability | Development evidence | [Run example](../../docs/reference/FLAGOS_TRANSPORT_OBSERVABILITY_F6.md) |
| Distinguish FlagOS validation from unverified FlagCX route attribution | FlagOS distributed transport observability | Development evidence | [Run example](../../docs/reference/FLAGOS_TRANSPORT_OBSERVABILITY_F6.md) |
| Train a large low-entanglement system | Differentiable and sharded MPS training | Development evidence | [Run example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py) |
| Distribute one MPS across several GPUs | Differentiable and sharded MPS training | Development evidence | [Run example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py) |
| Inspect variable-bond MPS capacity | Differentiable and sharded MPS training | Development evidence | [Run example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py) |
| Evaluate software-extended precision on FP32 hardware | Double-Single FP32 numerical primitives | Experimental | [Run example](../../docs/reference/DOUBLE_SINGLE_FP32.md) |
| Measure cancellation error against a float64 reference | Double-Single FP32 numerical primitives | Experimental | [Run example](../../docs/reference/DOUBLE_SINGLE_FP32.md) |
| Prepare a precision provider without changing the runtime | Double-Single FP32 numerical primitives | Experimental | [Run example](../../docs/reference/DOUBLE_SINGLE_FP32.md) |
| Evaluate statevector forward execution on FP32-only PyTorch devices | Split real/imag FP32 local statevector P0 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P0.md) |
| Validate Torch-FL flagos logical-device residency | Split real/imag FP32 local statevector P0 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P0.md) |
| Compare split FP32 numerical error with a CPU complex128 reference | Split real/imag FP32 local statevector P0 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P0.md) |
| Evaluate Pauli expectation values on FP32-only PyTorch devices | Split real/imag FP32 observable and parameter-shift P1 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P1.md) |
| Compute explicit parameter-shift gradients for named rotation parameters | Split real/imag FP32 observable and parameter-shift P1 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P1.md) |
| Compare training primitives with a CPU complex128 reference | Split real/imag FP32 observable and parameter-shift P1 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P1.md) |
| Survive cancellation-sensitive Hamiltonian reductions on FP32-only devices | Selective Double-Single split statevector precision P2 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P2_PRECISION.md) |
| Request an auditable selective precision plan | Selective Double-Single split statevector precision P2 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P2_PRECISION.md) |
| Fail closed when requested accuracy exceeds certified evidence | Selective Double-Single split statevector precision P2 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P2_PRECISION.md) |
| Evaluate full Double-Single state evolution on FP32-only PyTorch devices | Full Double-Single split statevector P3 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P3_DOUBLE_SINGLE.md) |
| Break the FP32 state-evolution error floor | Full Double-Single split statevector P3 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P3_DOUBLE_SINGLE.md) |
| Measure state, norm, expectation, and gradient errors against complex128 | Full Double-Single split statevector P3 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P3_DOUBLE_SINGLE.md) |
| Avoid CPU complex128 gate encoding on FP32-only PyTorch devices | Device-generated Double-Single split statevector P4 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P4_DEVICE_GATES.md) |
| Evaluate bounded Double-Single trigonometry and state evolution | Device-generated Double-Single split statevector P4 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P4_DEVICE_GATES.md) |
| Measure state, norm, expectation, and gradient errors against complex128 diagnostics | Device-generated Double-Single split statevector P4 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P4_DEVICE_GATES.md) |
| Train through a conventional PyTorch scalar loss on CPU | CPU PyTorch autograd bridge over Double-Single P5 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P5_AUTOGRAD_OPTIMIZER_CONTRACT.md) |
| Retain Double-Single arithmetic inside forward and parameter-shift evaluation | CPU PyTorch autograd bridge over Double-Single P5 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P5_AUTOGRAD_OPTIMIZER_CONTRACT.md) |
| Audit the exact boundary where PyTorch receives one FP32 gradient word | CPU PyTorch autograd bridge over Double-Single P5 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P5_AUTOGRAD_OPTIMIZER_CONTRACT.md) |
| Preserve parameter updates below one FP32 ULP | Single-device precision-preserving Double-Single SGD P5 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P5_AUTOGRAD_OPTIMIZER_CONTRACT.md) |
| Compare Double-Single and FP32-master optimization against complex128 trajectories | Single-device precision-preserving Double-Single SGD P5 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P5_AUTOGRAD_OPTIMIZER_CONTRACT.md) |
| Run auditable precision SGD without claiming torch.optim compatibility | Single-device precision-preserving Double-Single SGD P5 | Experimental | [Run example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P5_AUTOGRAD_OPTIMIZER_CONTRACT.md) |
| Reproduce small open-chain imaginary-time TEBD studies | Constrained local MPS TEBD | Experimental | [Run example](../../docs/guides/TEBD.md) |
| Cross-check MPS evolution against an independent oracle | Constrained local MPS TEBD | Experimental | [Run example](../../docs/guides/TEBD.md) |
| Inspect truncation and normalization evidence | Constrained local MPS TEBD | Experimental | [Run example](../../docs/guides/TEBD.md) |
| Evaluate a circuit with tensor-network contraction | Tensor-network execution and training | Experimental | [Run example](../../examples/vqe_switch_sv_mps_tn.py) |
| Compare statevector, MPS, and tensor-network modes | Tensor-network execution and training | Experimental | [Run example](../../examples/vqe_switch_sv_mps_tn.py) |
| Validate small noisy circuits exactly | Exact and trajectory-based noisy simulation | Experimental | [Run example](../../examples/noisy_simulation_v1.py) |
| Evaluate low-entanglement noisy circuits with MPS trajectories | Exact and trajectory-based noisy simulation | Experimental | [Run example](../../examples/noisy_simulation_v1.py) |
| Resume reproducible trajectory ensembles | Exact and trajectory-based noisy simulation | Experimental | [Run example](../../examples/noisy_simulation_v1.py) |
| Package a trained parameterized circuit | Circuit packaging and cloud deployment | Development evidence | [Run example](../../examples/train_parameterized_circuit_then_deploy.py) |
| Export a circuit for a provider | Circuit packaging and cloud deployment | Development evidence | [Run example](../../examples/train_parameterized_circuit_then_deploy.py) |
| Run a circuit through a deployment abstraction | Circuit packaging and cloud deployment | Development evidence | [Run example](../../examples/train_parameterized_circuit_then_deploy.py) |
| Predict a mapped QPU measurement distribution | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/twin_connected_region.py) |
| Compare a frozen prediction with later QPU counts | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/twin_connected_region.py) |
| Reject circuits outside validated topology and depth | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/twin_connected_region.py) |
| Check explicit circuit mappings against connected validated cells | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/twin_connected_region.py) |
| Discover registered interoperability adapters | Interoperability adapter contract | Experimental | [Run example](../../docs/reference/API.md) |
| Implement a framework adapter without changing FlagQuantum core | Interoperability adapter contract | Experimental | [Run example](../../docs/reference/API.md) |
| Certify round-trip and fail-closed adapter behavior | Interoperability adapter contract | Experimental | [Run example](../../docs/reference/API.md) |
| Handle conversion diagnostics consistently | Interoperability adapter contract | Experimental | [Run example](../../docs/reference/API.md) |
| Import a supported PennyLane QuantumScript | PennyLane QuantumScript interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Export static FlagQuantum IR to PennyLane | PennyLane QuantumScript interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Audit semantic loss at the boundary | PennyLane QuantumScript interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Import a supported Qiskit circuit | Qiskit IR interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Export FlagQuantum IR to Qiskit | Qiskit IR interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Audit semantic loss at a framework boundary | Qiskit IR interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Prototype mid-circuit measurement and feed-forward | Dynamic circuits and backend assessment | Experimental | [Run example](../../docs/reference/API.md) |
| Assess backend support before execution | Dynamic circuits and backend assessment | Experimental | [Run example](../../docs/reference/API.md) |
| Exercise a fixed-round QEC control workflow | Repetition-code memory experiment | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Inspect syndrome and detection-event records | Repetition-code memory experiment | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Prototype a decoder against a typed contract | Repetition-code memory experiment | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Prototype a FlagQuantum extension | Extension SDK | Experimental | [Run example](../../docs/guides/COMPILER_PLUGINS.md) |
| Register custom framework behavior | Extension SDK | Experimental | [Run example](../../docs/guides/COMPILER_PLUGINS.md) |
| Install an external circuit compiler | Extension SDK | Experimental | [Run example](../../docs/guides/COMPILER_PLUGINS.md) |
| Convert a QUBO problem into a Hamiltonian | QUBO to Ising mapping | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Recover the QUBO form from a Hamiltonian | QUBO to Ising mapping | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Evaluate a QUBO objective on a candidate assignment | QUBO to Ising mapping | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Prepare a uniform superposition | Quantum state preparation | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Prepare a state matching a given amplitude vector | Quantum state preparation | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Cross-check a prepared state against the target amplitudes | Quantum state preparation | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Flip a target qubit only when every control is set | Oracle building blocks | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Mark the states where one bit string is greater than another | Oracle building blocks | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Compose reversible classical logic into a circuit | Oracle building blocks | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Mark the states satisfying a predicate with a phase | Truth-table oracle synthesis | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Write a predicate's value onto an output qubit | Truth-table oracle synthesis | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| List the states a predicate marks | Truth-table oracle synthesis | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Search for the states a predicate marks | Grover search | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Amplify the marked states' amplitudes | Grover search | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Read the most likely marked state from samples | Grover search | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Estimate the amplitude a marking operator selects | Quantum amplitude estimation | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Read an amplitude off the counting register | Quantum amplitude estimation | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Compare an estimate against a known amplitude | Quantum amplitude estimation | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Estimate the spectrum of a density matrix | Quantum principal component analysis | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Read an eigenvalue off a counting register | Quantum principal component analysis | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Compare a read-out eigenvalue against the density matrix's own spectrum | Quantum principal component analysis | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Assign each point to its nearest centroid | Quantum k-medians | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Read an assignment off sampled searches | Quantum k-medians | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Update centroid positions to the medians of the points assigned to them | Quantum k-medians | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |

## Build and compile

### Unified circuit API and FlagQuantum IR

Build, validate, serialize, compile, and inspect quantum circuits through the stable FlagQuantum interface.

- **Maturity:** Release certified
- **Public API:** `fq.Circuit`, `fq.CircuitIR`, `flagquantum.compiler.compile`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../examples/quick_start.py)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** IR v1; incompatible schema changes require an explicit migration.


## Simulation and training

### Local statevector simulation and training

Run exact circuits and differentiable quantum workloads on a CPU or one GPU.

- **Maturity:** Production supported
- **Public API:** `fq.Circuit`, `fq.run`, `fq.Module`, `fq.train`
- **Runtime modes:** `statevector`
- **Hardware:** `cpu`, `single_gpu`
- **Gradient support:** `exact`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../examples/single_machine_quantum_ai/01_vqe_statevector.py)
- **Documentation:** [guide](../../examples/single_machine_quantum_ai/README.md)
- **Known boundary:** Capacity is bounded by one device; distributed capacity claims use the sharded capability.

### FlagOS local statevector CUDA reference

Exercise the local differentiable statevector path through Torch-FL's logical flagos device on a locked CUDA reference environment.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.runtime.resolve_device`, `fq.run`
- **Runtime modes:** `statevector`
- **Hardware:** `nvidia_a100_cuda_reference`
- **Gradient support:** `development_evidence`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../docs/reference/ACCELERATOR_PLATFORM_RUNTIME.md)
- **Documentation:** [guide](../../docs/reference/STATEVECTOR_OPERATOR_PROFILES.md)
- **Known boundary:** CUDA-backed development reference only. It does not certify a domestic accelerator, prove absence of Torch-FL host fallback, establish production performance, or authorize a scalability claim.

### Double-Single FP32 numerical primitives

Use residual-preserving pairs of float32 tensors for bounded real and split-complex arithmetic experiments on PyTorch devices.

- **Maturity:** Experimental
- **Public API:** Not exposed; internal development evidence only
- **Runtime modes:** `numerical_primitive_conformance`
- **Hardware:** `cpu`, `device_generic_pytorch`
- **Gradient support:** `experimental_composed_primitives`
- **Distribution semantics:** `single_device_primitive_only`
- **Start:** [quick example](../../docs/reference/DOUBLE_SINGLE_FP32.md)
- **Documentation:** [guide](../../docs/reference/DOUBLE_SINGLE_FP32.md)
- **Known boundary:** Pure FP32 real and split-complex eager primitives, selective split-statevector P2 reductions, full-state P3, and bounded device-generated-gate P4 experiments are available. P4 removes CPU float64/complex128 gate encoding for its certified gate and angle scope, but optimized kernels, compiled execution, distributed collectives, decompositions, optimizer state, provider-owned Torch-FL route auditing, domestic accelerators, performance, convergence, and production use remain uncertified. Double-Single retains FP32 exponent range and is not generally equivalent to FP64 or complex128.

### Split real/imag FP32 local statevector P0

Execute a bounded forward-only statevector using two device-resident float32 tensors without requiring accelerator complex dtypes.

- **Maturity:** Experimental
- **Public API:** `fq.experimental.numerics.execute_split_real_imag_statevector`
- **Runtime modes:** `split_real_imag_statevector_p0`
- **Hardware:** `cpu`, `single_cuda_reference`, `device_generic_pytorch`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P0.md)
- **Documentation:** [guide](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P0.md)
- **Known boundary:** Explicit experimental forward-only executor for a bounded built-in gate set using separate FP32 real and imaginary tensors. It is not selected by the default runtime. Custom matrices, gradients, optimizer steps, sampling and observables APIs, compiled execution, distributed execution, Double-Single storage, provider-internal route auditing, domestic-hardware certification, performance, and production use remain unsupported. CUDA or CUDA-backed flagos evidence is portability evidence only.

### Split real/imag FP32 observable and parameter-shift P1

Evaluate bounded Pauli Hamiltonians and explicit parameter-shift gradients using two device-resident float32 state tensors without accelerator complex dtypes.

- **Maturity:** Experimental
- **Public API:** `fq.experimental.numerics.execute_split_real_imag_expectation`, `fq.experimental.numerics.parameter_shift_split_real_imag_gradient`
- **Runtime modes:** `split_real_imag_statevector_p1`
- **Hardware:** `cpu`, `single_cuda_reference`, `device_generic_pytorch`
- **Gradient support:** `parameter_shift_experimental`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P1.md)
- **Documentation:** [guide](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P1.md)
- **Known boundary:** Explicit experimental batch-one Pauli expectation and occurrence-wise two-term parameter-shift gradients for direct named scalar parameters on RX, RY, RZ, RXX, RYY, and RZZ. It is not native autograd and provides no optimizer integration. Parameter expressions, trainable coefficients, custom matrices or states, sampling, compilation, distributed execution, automatic runtime selection, performance, convergence, provider-internal route auditing, and domestic-hardware certification remain unsupported. CUDA or CUDA-backed flagos evidence is portability evidence only.

### Selective Double-Single split statevector precision P2

Retain FP32 state and gates while upgrading Pauli inner products, Hamiltonian sums, and parameter-shift accumulation to explicit Double-Single high/low reductions.

- **Maturity:** Experimental
- **Public API:** Not exposed; internal development evidence only
- **Runtime modes:** `split_real_imag_statevector_p2_precision`
- **Hardware:** `cpu`, `single_cuda_reference`, `device_generic_pytorch`
- **Gradient support:** `parameter_shift_selective_double_single_experimental`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P2_PRECISION.md)
- **Documentation:** [guide](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P2_PRECISION.md)
- **Known boundary:** Explicit experimental selective precision path only. State storage, gate generation, and gate application remain split FP32; only Pauli inner products, Hamiltonian term sums, and parameter-shift accumulation retain Double-Single high/low words. Full Double-Single statevectors, native autograd, optimizer integration, residual checkpoints, decomposition, compilation, distributed execution, automatic selection, convergence certification, provider-internal route auditing, domestic-hardware certification, performance, and production use remain unsupported.

### Full Double-Single split statevector P3

Store every complex amplitude as four FP32 high/low words and retain residuals through gate application, periodic normalization, observables, and parameter-shift gradients.

- **Maturity:** Experimental
- **Public API:** Not exposed; internal development evidence only
- **Runtime modes:** `split_real_imag_statevector_p3_double_single`
- **Hardware:** `cpu`, `single_cuda_reference`, `device_generic_pytorch`
- **Gradient support:** `parameter_shift_full_double_single_experimental`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P3_DOUBLE_SINGLE.md)
- **Documentation:** [guide](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P3_DOUBLE_SINGLE.md)
- **Known boundary:** Explicit correctness-first full Double-Single state experiment. CPU float64/complex128 parameter and gate encoding is required before four FP32 words are transferred to the execution device; state evolution itself has no host fallback. Device-only Double-Single trigonometry, optimized/fused kernels, native autograd, optimizer integration, compilation, distributed execution, automatic selection, algorithmic convergence certification, provider-internal route auditing, domestic-hardware certification, performance, and production use remain unsupported.

### Device-generated Double-Single split statevector P4

Generate bounded fixed and rotation gates with device-resident FP32 Double-Single arithmetic, then retain four FP32 words through state evolution, observables, and parameter-shift gradients.

- **Maturity:** Experimental
- **Public API:** Not exposed; internal development evidence only
- **Runtime modes:** `split_real_imag_statevector_p4_device_double_single`
- **Hardware:** `cpu`, `single_cuda_reference`, `device_generic_pytorch`
- **Gradient support:** `parameter_shift_device_double_single_experimental`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P4_DEVICE_GATES.md)
- **Documentation:** [guide](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P4_DEVICE_GATES.md)
- **Known boundary:** Explicit correctness-first P4 path for built-in gates and direct scalar parameters within |angle| <= 1024. Python scalars are host-ingested as FP32 values; device-resident FP32 or Double-Single parameters remain on device. Parameter expressions, float64 parameter tensors, custom matrices, unbounded angles, native autograd, optimizer integration, compilation, distributed execution, automatic selection, algorithmic convergence certification, provider-internal route auditing, domestic-hardware certification, performance, and production use remain unsupported.

### CPU PyTorch autograd bridge over Double-Single P5

Expose a bounded first-order PyTorch autograd path whose forward and parameter-shift backward use P4 Double-Single arithmetic before an explicit FP32 tensor delivery boundary.

- **Maturity:** Experimental
- **Public API:** Not exposed; internal development evidence only
- **Runtime modes:** `split_real_imag_statevector_p5_autograd_bridge`
- **Hardware:** `cpu`
- **Gradient support:** `first_order_parameter_shift_internal_double_single_float32_delivery_experimental`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P5_AUTOGRAD_OPTIMIZER_CONTRACT.md)
- **Documentation:** [guide](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P5_AUTOGRAD_OPTIMIZER_CONTRACT.md)
- **Known boundary:** Explicit CPU-only first-order PyTorch autograd bridge for direct named scalar FP32 parameters and bounded Pauli expectations. Forward and backward use P4 Double-Single arithmetic internally, but the scalar loss and Tensor.grad are one-word FP32 delivery boundaries and no end-to-end Double-Single gradient claim is made. A separate explicit Double-Single SGD lane is available; higher-order and compiled autograd, CUDA, Torch-FL flagos, FlagCX, distributed execution, automatic selection, convergence, hardware certification, performance, and production use remain unsupported.

### Single-device precision-preserving Double-Single SGD P5

Consume explicit P4 high/low parameter-shift gradients and return updated high/low master parameters without crossing the one-word PyTorch Tensor.grad boundary.

- **Maturity:** Experimental
- **Public API:** Not exposed; internal development evidence only
- **Runtime modes:** `split_real_imag_statevector_p5_double_single_sgd`
- **Hardware:** `cpu`, `single_cuda_reference`, `torch_fl_flagos_cuda_reference`
- **Gradient support:** `explicit_parameter_shift_double_single_optimizer_experimental`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P5_AUTOGRAD_OPTIMIZER_CONTRACT.md)
- **Documentation:** [guide](../../docs/reference/SPLIT_REAL_IMAG_STATEVECTOR_P5_AUTOGRAD_OPTIMIZER_CONTRACT.md)
- **Known boundary:** Explicit single-device functional high/low master-parameter SGD using P4 high/low parameter-shift gradients. CPU, native CUDA on A800, and CUDA-backed Torch-FL flagos:0 portability trajectories are recorded; the A800 routes do not certify a domestic accelerator or provider internals. It is not torch.optim compatible and has no momentum, weight decay, loss scaling, Adam-family algorithm, checkpoint/state-dict compatibility, higher-order autograd, FlagCX, distributed execution, automatic selection, convergence certification, hardware certification, performance, or production claim.

### Constrained local MPS TEBD

Evolve open-chain local Pauli Hamiltonians with fail-closed second-order imaginary-time TEBD.

- **Maturity:** Experimental
- **Public API:** `fq.experimental.simulation.run_tebd`
- **Runtime modes:** `mps_tebd`
- **Hardware:** `cpu`, `single_gpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../docs/guides/TEBD.md)
- **Documentation:** [guide](../../docs/guides/TEBD.md)
- **Known boundary:** Static real one-site and adjacent two-site Pauli terms on an open chain, batch one, second-order imaginary-time evolution, and product initial states only. Real-time evolution, periodic and nonlocal terms, gradients, TDVP, distributed execution, and production or scalability claims are unsupported.

### Tensor-network execution and training

Execute tensor-network circuit paths and evaluate experimental contraction and gradient workflows.

- **Maturity:** Experimental
- **Public API:** `flagquantum.simulation.tensor_network.run_tensor_network`
- **Runtime modes:** `tensor_network`
- **Hardware:** `cpu`, `single_gpu`
- **Gradient support:** `experimental`
- **Distribution semantics:** `manual_sliced_tensor_contraction`
- **Start:** [quick example](../../examples/vqe_switch_sv_mps_tn.py)
- **Documentation:** [guide](../../docs/reference/KNOWN_LIMITATIONS.md)
- **Known boundary:** General reverse contraction and production distributed transport are not certified.

### Exact and trajectory-based noisy simulation

Lower validated Kraus noise models into FlagQuantum IR and execute exact density-matrix or MPS quantum-trajectory paths.

- **Maturity:** Experimental
- **Public API:** `flagquantum.noise.NoiseModel`, `flagquantum.noise.noisy_density_matrix`, `flagquantum.runtime.run_noisy_mps`, `flagquantum.runtime.run_noisy_statevector`
- **Runtime modes:** `density_matrix`, `noisy_mps`
- **Hardware:** `cpu`, `single_gpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_device_fast_path_or_rank_local_trajectory_partition`
- **Start:** [quick example](../../examples/noisy_simulation_v1.py)
- **Documentation:** [guide](../../docs/guides/NOISY_SIMULATION.md)
- **Known boundary:** Validated Markovian Kraus channels, timestamped DeviceNoiseProfile input, ASAP gate/idle thermal lowering, classical readout confusion, exact density execution, and reproducible MPS trajectories with single-rank adaptive stopping are available. Pulse overlap, crosstalk, leakage, provider calibration adapters, distributed adaptive stopping, batched statevector trajectories, production multi-GPU scheduling, and noisy gradients remain unsupported. Multi-wire MPS channels use an explicitly dense correctness fallback.

### Repetition-code memory experiment

Run a bounded three-data-qubit memory experiment with timed errors or circuit-location bit-flip/readout noise, compiled feedback, per-round Runtime decoding, or Pauli-frame correction, including a two-round temporal reference decoder.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.run_repetition_memory_experiment`, `flagquantum.qec.run_repetition_memory_noise_sweep`, `flagquantum.qec.RepetitionNoiseProfile`, `flagquantum.qec.ErrorSchedule`, `flagquantum.qec.Decoder`, `flagquantum.qec.StreamingDecoder`, `flagquantum.qec.RepetitionTemporalDecoder`
- **Runtime modes:** `local_statevector_compiled_feedback`, `local_statevector_runtime_decoder`, `local_statevector_runtime_pauli_frame`, `local_statevector_runtime_temporal_decoder`, `local_statevector_runtime_temporal_pauli_frame`, `local_statevector_offline_pauli_frame`, `local_statevector_noisy_trajectory`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/README.md)
- **Documentation:** [guide](../../flagquantum/qec/README.md)
- **Known boundary:** A synchronous local reference for one fixed three-data-qubit repetition-code profile. It supports bounded deterministic X-error schedules, replaceable per-round trajectory decoding with physical-X or Pauli-frame-X actions, and a two-round temporal rule that rejects an isolated readout excursion. Confirmable data errors require a following round; terminal-round onsets remain unconfirmed. The circuit-location stochastic profile contains independent bit flips after parity-check CNOTs and independent syndrome/final-readout confusion. The middle data wire has two CNOT noise opportunities per round while edge wires have one. Feedback traces separate true and observed bits, actions, and frame evolution. Sweeps report finite-shot observations only, not logical suppression or thresholds. The temporal rule is not maximum-likelihood decoding and repeated readout faults may mimic data errors. Batched decoder feedback, general channels/codes, correlated or timing noise, hard-real-time/provider control, gradients, distributed execution, capacity, performance, and fault-tolerance claims remain unsupported. The feedback records are private subinterfaces and the namespace is not exported from the stable package root.

### QUBO to Ising mapping

Express a quadratic unconstrained binary optimization problem as an Ising Hamiltonian for the existing variational workflows.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.qubo`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** A polynomial classical transformation with no advantage of its own: any advantage a caller observes belongs to the solver that consumes the Hamiltonian. Only Z-basis objectives are representable, so a Hamiltonian outside the Z basis is rejected. The mapping carries the constant as an identity term and recovers it on the way back, so a caller comparing the two forms sees identical values; a caller who strips the identity term loses that constant. It certifies no solver, convergence, performance, or hardware behavior.

### Quantum state preparation

Prepare a uniform or arbitrary quantum state from a classical amplitude vector using uniformly controlled rotations.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.primitives.state_preparation`
- **Runtime modes:** `local_statevector`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** Demonstration scale. Computing the rotation angles requires a classical pass over all 2**n amplitudes and a 2**n by 2**n linear solve, so the input is already exponential in size: this unit shows that a state can be prepared efficiently given its amplitudes, not that preparing a state is cheaper than its classical description. It makes no advantage, performance, convergence, or hardware claim, and is not selected by the default runtime. The preparation is exact only to the working precision of the solve.

### Oracle building blocks

Multi-controlled X and a reversible bit-string comparator: the reversible classical logic the oracle units are built from.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.primitives.oracle.append_multi_controlled_x`, `flagquantum.algorithms.primitives.oracle.append_comparator`
- **Runtime modes:** `local_statevector`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** Reversible classical logic of O(n) Toffoli-style cost with no advantage premise of its own. A multi-controlled X above two controls needs len(controls) - 2 caller-supplied ancillas, each of which must be in |0> on entry: measured, a dirty ancilla makes the target silently wrong on a large fraction of inputs (8 of 16 at three controls, 32 of 96 at four) while never corrupting the ancilla itself, so the failure is invisible from the ancilla. The comparator restores every wire it is given except the target; its len(lhs) + 1 equality flags and its scratch wire must all be in |0> on entry too, and a dirty equality[0] makes the target silently wrong on a large fraction of the operand patterns (6 of 16 at two bits) before the ladder restores the flag. The comparator's published record is semi-verified: its venue is not indexed by Crossref, DBLP or INSPIRE, so its volume and page numbers are reported by citing works rather than index-confirmed. It makes no advantage, performance, or hardware claim.

### Truth-table oracle synthesis

Turn a classical predicate into a phase oracle or a bit oracle by enumerating its truth table.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.primitives.oracle.marked_states`, `flagquantum.algorithms.primitives.oracle.phase_oracle`, `flagquantum.algorithms.primitives.oracle.append_phase_oracle`, `flagquantum.algorithms.primitives.oracle.bit_oracle`, `flagquantum.algorithms.primitives.oracle.append_bit_oracle`
- **Runtime modes:** `local_statevector`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** Synthesis enumerates all 2**n inputs of the truth table classically, so it carries no advantage of its own at any scale beyond demonstration and its cost is exponential in the register width. Only a truth table is accepted: there is no boolean-expression parser and no other predicate form. A phase oracle is capped at three wires, because a multi-controlled Z above that needs ladder ancillas a standalone circuit does not have; the in-place append form takes them from the caller. The bit oracle's output is XORed rather than assigned, and it restores every wire it allocates. No performance, convergence, or hardware claim is made.

### Grover search

Search a classical predicate's truth table by amplitude amplification over an evaluation register.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.grover`
- **Runtime modes:** `local_statevector`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** The improvement is in query complexity, against an oracle this unit synthesizes from a truth table at a classical cost of 2**n. No end-to-end advantage follows at any scale beyond demonstration, and no qRAM, block encoding or amplitude encoding is assumed. The search is bounded at three evaluation wires, because a phase oracle above that needs ladder ancillas a register of exactly that width does not have. The register width makes the classical truth-table enumeration exponential, which is the honest limit of the unit. It makes no performance, convergence, or hardware claim.

### Quantum amplitude estimation

Estimate the probability a state-preparation unitary's marked subspace carries, by phase estimation over the Grover operator.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.amplitude_estimation`
- **Runtime modes:** `local_statevector`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** The quadratic speedup over classical sampling is real only given that the state-preparation unitary A is free: a real distribution needs QRAM, and this unit does not supply one, so it does not show that any Monte Carlo integral is estimated faster than classically. The estimate lies on the amplitude grid sin^2(pi j / 2**(m+1)) and is accurate to about one grid step; that is a resolution, not a coverage-calibrated confidence interval, and no confidence interval is reported. The operator must be able to apply A, its adjoint, the marking operator and the zero-state reflection each under control, which excludes state-preparation circuits that cannot be controlled. It makes no performance, convergence, or hardware claim.

### Quantum principal component analysis

Estimate the eigenvalues of a data matrix's density matrix by phase estimation over its exponential, reading them off a counting register.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.pca`
- **Runtime modes:** `local_statevector`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** Demonstration scale. The density matrix is materialized classically and its exponential is built with a dense matrix exponential, so the unit does not reproduce quantum PCA's input model: it never avoids forming rho and never uses the O(1/eps^3) state copies the algorithm is built on. The purified input is supplied as all 2**n amplitudes. Meaningful only for low effective rank.

### Quantum k-medians

Assign points to their nearest centroids with a Grover-style minimum search over a centroid index register, then move each centroid to the classical coordinate-wise median of its cluster.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.kmedians`
- **Runtime modes:** `local_statevector`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** The advantage premise is Grover's oracle model, and it is not met: the oracle is not free here. The search's oracle is synthesized from the predicate's truth table at O(2**n) cost, so no end-to-end advantage follows at this scale, and the distance table the predicate compares is computed classically, one point at a time, before any circuit is built -- the register is capped at three wires, which is also what keeps the distances out of it. Nothing here reads a qRAM or runs an adiabatic evolution, so no conclusion that rests on either applies to this unit. Demonstration scale: the search is bounded at three evaluation wires, so at most eight centroids, and the assignment is sampled rather than read out, which is why a small sample can stop a point's search short. It makes no performance, convergence, or hardware claim.


## Distributed execution

### Sharded statevector training

Partition one logical statevector workload across ranks while preserving differentiable training semantics.

- **Maturity:** Production supported
- **Public API:** `fq.plan`, `flagquantum.experimental.distributed.train_distributed_statevector`
- **Runtime modes:** `distributed_statevector`
- **Hardware:** `multi_gpu`, `multi_node`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../examples/distributed_statevector_topologies/run.sh)
- **Documentation:** [guide](../../examples/distributed_statevector_topologies/README.md)
- **Known boundary:** Multi-node release certification remains dependent on promoted audited hardware evidence.

### FlagOS distributed statevector workloads

Run sharded statevector forward and bounded training workloads through the public FlagOS boundary on the locked CUDA development reference.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.experimental.distributed.train_distributed_statevector`
- **Runtime modes:** `distributed_statevector`
- **Hardware:** `nvidia_a800_cuda_reference`, `single_node_2_4_8_gpu`
- **Gradient support:** `development_evidence_exact_autograd`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../docs/reference/FLAGOS_DISTRIBUTED_CONFORMANCE.md)
- **Documentation:** [guide](../../docs/reference/FLAGOS_WORKLOAD_CAPABILITY_F4.md)
- **Known boundary:** Development evidence only for complex64 and complex128 on one CUDA-backed A800 node at 2, 4, and 8 cards. The inner communication route and host staging remain unattributed; complex reduce_scatter_tensor is unsupported in the tested full collective matrix; multi-node behavior, single-device capacity failure, performance, convergence, production support, scalability, and release certification are not established.

### FlagOS statevector capacity expansion

Demonstrate one matched complex128 statevector that fails on a single device and completes when sharded across eight devices through the public FlagOS boundary.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.experimental.distributed.train_distributed_statevector`
- **Runtime modes:** `distributed_statevector`
- **Hardware:** `nvidia_a800_cuda_reference`, `single_node_8_gpu`
- **Gradient support:** `forward_only_development_evidence`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../docs/reference/FLAGOS_STATEVECTOR_CAPACITY_F5.md)
- **Documentation:** [guide](../../docs/reference/FLAGOS_STATEVECTOR_CAPACITY_F5.md)
- **Known boundary:** One exact 32-qubit complex128 forward workload on one CUDA-backed eight-A800 node. The single-device and replicated paths measured OOM while the eight-rank sharded path completed. The inner communication route and host staging remain unattributed; determinism replay, backward and optimizer capacity, performance, convergence, multi-node behavior, production support, general scalability, and release certification are not established.

### FlagOS distributed transport observability

Observe a fixed multi-rank complex collective matrix through the public FlagOS boundary while keeping inner-route and host-staging claims fail-closed.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.experimental.distributed.train_distributed_statevector`
- **Runtime modes:** `distributed_transport_observation`
- **Hardware:** `nvidia_a800_cuda_reference`, `single_node_2_4_8_gpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `rank_local_collective_observation`
- **Start:** [quick example](../../docs/reference/FLAGOS_TRANSPORT_OBSERVABILITY_F6.md)
- **Documentation:** [guide](../../docs/reference/FLAGOS_TRANSPORT_OBSERVABILITY_F6.md)
- **Known boundary:** One CUDA-backed A800 node at 2, 4, and 8 ranks for four collectives with complex64 and complex128. Correctness and logical FlagOS residency passed, but CUPTI device activity capture was incomplete. The inner communication route, FlagCX use, absence of host staging, performance, multi-node behavior, scalability, production support, and release certification are not established.

### Differentiable and sharded MPS training

Train low-entanglement quantum systems with local or rank-owned matrix product states.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.simulation.mps.run_mps`, `flagquantum.experimental.distributed.train_distributed_mps`
- **Runtime modes:** `mps`, `distributed_mps`
- **Hardware:** `cpu`, `single_gpu`, `multi_gpu`, `multi_node`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py)
- **Documentation:** [guide](../../examples/distributed_mps/README.md)
- **Known boundary:** Single-node and dual-node execution plus matched checkpoint/restart have development evidence. The only public capacity measurement is emitted from the validated claim below; it is one exact-workload result, not general scalability or release evidence. Boundary instructions still execute serially by owner, and layer-parallel contraction/SVD, capacity multi-step soak, a sealed fault matrix, repeated evidence, and the release payload remain incomplete.


## Deployment and extension

### Circuit packaging and cloud deployment

Package trained circuits, export provider formats, and route them through deployment provider abstractions.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.deployment.create_deployment_package`, `flagquantum.deployment.deploy_circuit`
- **Runtime modes:** `provider`
- **Hardware:** `provider_dependent`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `provider_dependent`
- **Start:** [quick example](../../examples/train_parameterized_circuit_then_deploy.py)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Provider support and credential/runtime behavior vary; no provider is release-certified by this matrix.

### Evidence-qualified QPU digital twins

Predict calibration-conditioned measurement distributions, qualify them against identity-bound hardware evidence, and compose connected structural coverage.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.twin`
- **Runtime modes:** `offline_density_prediction`, `offline_evidence_assessment`, `explicit_provider_submission`
- **Hardware:** `cpu`, `quafu_development_evidence`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../examples/twin_connected_region.py)
- **Documentation:** [guide](../../docs/guides/QPU_DIGITAL_TWIN.md)
- **Known boundary:** Frozen mapped models, validation series, calibration and validation histories, prospective candidate comparisons, topology- and depth-qualified evidence, and connected structural cell composition are available. Agreement is total-variation agreement for classical measurement distributions, not quantum-state fidelity. Region composition checks topology and conservative common limits only; it never combines local bounds into region accuracy. Evidence remains specific to declared circuits, operations, mappings, physical couplers, depth, calibration snapshots, and confidence bounds. Automatic calibration collection, scheduling, arbitrary-circuit generalization, region-level statistical inference, trust policy, model promotion, global publication, and release-certified provider support remain outside the framework capability.

### Interoperability adapter contract

Implement and certify optional external-framework conversion behind one immutable lazy registry and framework-neutral, loss-aware result contract.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem`
- **Runtime modes:** `control_plane_conversion`
- **Hardware:** `cpu_control_plane`
- **Gradient support:** `adapter_defined`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/reference/API.md)
- **Documentation:** [guide](../../docs/development/INTEROP_ADAPTERS.md)
- **Known boundary:** The framework-neutral protocol is candidate-stable pending API-owner approval; PennyLane and Qiskit implementations remain experimental. Common conformance does not install dependencies, sandbox third-party Python, certify provider hardware or numerical equivalence, or permit external objects to enter runtime and accelerator layers.

### PennyLane QuantumScript interoperability

Translate supported immutable PennyLane QuantumScript programs to versioned FlagQuantum IR and back through an isolated, loss-aware control-plane adapter.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem.pennylane.from_pennylane`, `flagquantum.ecosystem.pennylane.to_pennylane`
- **Runtime modes:** `control_plane_conversion`
- **Hardware:** `cpu_control_plane`
- **Gradient support:** `bound_parameters_only`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/reference/API.md)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Certified with PennyLane 0.44.1 and 0.45.1 on Python 3.11 or newer for static QuantumScript conversion and complex128 numerical semantics. QNode, device execution, shots, measurements, trainable parameters, arbitrary wire labels without explicit lossy flattening, and idle wire extents are outside v1. PennyLane objects never enter FlagQuantum runtime, Torch-FL, CUDA, vendor accelerator, or QPU layers.

### Qiskit IR interoperability

Translate supported Qiskit circuits to versioned FlagQuantum IR and export FlagQuantum IR through an isolated, loss-aware control-plane adapter.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem.qiskit.from_qiskit`, `flagquantum.ecosystem.qiskit.to_qiskit`
- **Runtime modes:** `control_plane_conversion`
- **Hardware:** `cpu_control_plane`
- **Gradient support:** `symbolic_parameters_only`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/reference/API.md)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Certified against Qiskit 2.0.x and 2.5.x with Aer 0.17.x through an executable operation, wire-order, statevector, classical-bit, and round-trip contract. Qiskit control flow and arbitrary ParameterExpression import are rejected; named or multiple registers require explicit lossy flattening; custom multi-qubit unitary matrices remain blocked until basis ordering is specified. Conversion does not make Qiskit a runtime dependency or certify any provider hardware.

### Dynamic circuits and backend assessment

Execute dynamic circuits locally, including a bounded bit-flip/readout-noise profile, and assess whether a backend can support their required features.

- **Maturity:** Experimental
- **Public API:** `flagquantum.dynamic.DynamicCircuit`, `flagquantum.experimental.dynamic.assess_dynamic_backend`, `flagquantum.experimental.dynamic.run_dynamic`
- **Runtime modes:** `local_statevector_trajectory`, `local_statevector_noisy_trajectory`, `backend_assessment`
- **Hardware:** `cpu`, `provider_profiles_unverified`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/reference/API.md)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** DynamicCircuit construction is candidate-stable pending API-owner approval; dynamic execution and backend assessment remain experimental. Local dynamic noise is limited to one-wire bit-flip channels after matching executed gates and independent readout confusion on explicit measurements and final sampling. Other Kraus channels, correlated readout, device-profile timing noise, noisy gradients, and provider-noise execution fail closed. Routing, deployment packaging, dialect export and provider integration are internal workflows rather than public experimental APIs. Provider-neutral conformance passes locally and on Qiskit Aer, but no real IQM QPU task was used.

### Extension SDK

Build and qualify optional extensions through the Ecosystem extension protocol.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem.extensions`
- **Runtime modes:** `extension_defined`
- **Hardware:** `extension_defined`
- **Gradient support:** `extension_defined`
- **Distribution semantics:** `extension_defined`
- **Start:** [quick example](../../docs/guides/COMPILER_PLUGINS.md)
- **Documentation:** [guide](../../docs/reference/EXTENSION_SDK.md)
- **Known boundary:** The migrated SDK protocol is approved but not frozen; individual extensions remain experimental until separately qualified. Compiler plugins currently exchange CircuitIR only; pulse and native-binary artifacts are not supported.


## Validated public performance claims

Every value below is read from a hash-bound raw artifact. Missing or changed
evidence makes the source-of-truth check fail closed.

| Claim | Maturity and scope | Artifact-derived result | Recorded environment | Evidence identity |
| --- | --- | --- | --- | --- |
| **Sharded MPS exact-workload capacity**<br>`mps-capacity-131072-chi768-20260806` | `development_evidence`<br>One batch-one, complex64, χ768 MPS training step for the checked-in all-rank and all-boundary workload. This is not arbitrary statevector capacity, fixed-plan strong scaling, or release evidence. | **Sites:** 131,072<br>**Logical MPS state:** 1,236,780,012,864 bytes (1,151.84 GiB)<br>**Maximum elapsed time:** 367.37 s<br>**Maximum peak allocated memory per rank:** 77,745,407,488 bytes (72.41 GiB)<br>**Cumulative discarded weight:** 8.39e-06 | **Ranks:** 16<br>**Reported device memory per rank:** 85,093,777,408 bytes (79.25 GiB)<br>**CUDA allocator policy:** expandable_segments:True<br>**Topology fingerprint:** c74a91e3a224a4dd414cfbfbcb31da7570880d42e6aa4eaccecf4b85427940a5<br>**Metadata boundary:** The artifact records rank count, per-rank device memory, topology fingerprint, and CUDA allocator policy. It does not record the exact GPU model or Python, PyTorch, CUDA, NCCL, driver, host, and operating-system versions, so the claim is restricted to the recorded environment fields. | [raw JSON](../../benchmarks/results/local/mps_capacity_131072q_chi768_16xa800_repeat_complete_20260806.json)<br>SHA-256 `df8c19b74cc173799e3894025fb8e72de72a786d4542a9aa5e67687298f482f5`<br>code `9d56a6ecd78b06f11b9ee6e8aadcbe9644f2c708` |
