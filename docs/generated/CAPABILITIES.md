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
| Prototype mid-circuit measurement and feed-forward | Dynamic circuits and IQM Braket preflight | Experimental | [Run example](../../examples/braket_iqm_dynamic_preflight.py) |
| Export dynamic OpenQASM 3 | Dynamic circuits and IQM Braket preflight | Experimental | [Run example](../../examples/braket_iqm_dynamic_preflight.py) |
| Preflight an IQM Braket task without submitting it | Dynamic circuits and IQM Braket preflight | Experimental | [Run example](../../examples/braket_iqm_dynamic_preflight.py) |
| Prototype a FlagQuantum extension | Extension SDK | Experimental | [Run example](../../examples/extensions/reference_extensions.py) |
| Register custom framework behavior | Extension SDK | Experimental | [Run example](../../examples/extensions/reference_extensions.py) |

## Build and compile

### Unified circuit API and FlagQuantum IR

Build, validate, serialize, compile, and inspect quantum circuits through the stable FlagQuantum interface.

- **Maturity:** Release certified
- **Public API:** `fq.Circuit`, `fq.CircuitIR`, `flagquantum.compiler.compile_for_backend`
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
- **Public API:** `flagquantum.backends.resolve_device`, `fq.run`
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
- **Public API:** `fq.experimental.numerics.run_double_single_conformance`
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
- **Public API:** `fq.experimental.numerics.execute_split_real_imag_statevector`, `fq.experimental.numerics.run_split_real_imag_conformance`
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
- **Public API:** `fq.experimental.numerics.execute_split_real_imag_expectation`, `fq.experimental.numerics.parameter_shift_split_real_imag_gradient`, `fq.experimental.numerics.run_split_real_imag_training_conformance`
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
- **Public API:** `fq.experimental.numerics.execute_split_real_imag_precision_expectation`, `fq.experimental.numerics.parameter_shift_split_real_imag_precision_gradient`, `fq.experimental.numerics.run_split_real_imag_precision_conformance`, `fq.experimental.numerics.split_real_imag_p2_precision_plan`, `fq.experimental.numerics.split_real_imag_p2_accuracy_envelope`
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
- **Public API:** `fq.experimental.numerics.execute_split_real_imag_double_single_statevector`, `fq.experimental.numerics.execute_split_real_imag_double_single_expectation`, `fq.experimental.numerics.parameter_shift_split_real_imag_double_single_gradient`, `fq.experimental.numerics.run_split_real_imag_double_single_conformance`, `fq.experimental.numerics.split_real_imag_p3_precision_plan`, `fq.experimental.numerics.split_real_imag_p3_accuracy_envelope`
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
- **Public API:** `fq.experimental.numerics.execute_split_real_imag_device_double_single_statevector`, `fq.experimental.numerics.execute_split_real_imag_device_double_single_expectation`, `fq.experimental.numerics.parameter_shift_split_real_imag_device_double_single_gradient`, `fq.experimental.numerics.run_split_real_imag_device_double_single_conformance`, `fq.experimental.numerics.split_real_imag_p4_precision_plan`, `fq.experimental.numerics.split_real_imag_p4_accuracy_envelope`
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
- **Public API:** `fq.experimental.numerics.split_real_imag_device_double_single_autograd_expectation`, `fq.experimental.numerics.split_real_imag_p5_autograd_bridge_summary`, `fq.experimental.numerics.run_split_real_imag_autograd_conformance`
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
- **Public API:** `fq.experimental.numerics.SplitRealImagDoubleSingleSGDState`, `fq.experimental.numerics.SplitRealImagDoubleSingleSGDStepResult`, `fq.experimental.numerics.initialize_split_real_imag_double_single_sgd`, `fq.experimental.numerics.double_single_sgd_step`, `fq.experimental.numerics.split_real_imag_double_single_sgd_step`, `fq.experimental.numerics.run_split_real_imag_optimizer_conformance`
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
- **Public API:** `fq.experimental.simulation.run_tebd`, `fq.experimental.simulation.TEBDResult`
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
- **Public API:** `flagquantum.backends.run_tensor_network`, `flagquantum.experimental.planning.plan_runtime_selection`
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
- **Public API:** `flagquantum.noise.NoiseModel`, `flagquantum.noise.noisy_density_matrix`, `flagquantum.backends.run_noisy_mps`
- **Runtime modes:** `density_matrix`, `noisy_mps`
- **Hardware:** `cpu`, `single_gpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_device_fast_path_or_rank_local_trajectory_partition`
- **Start:** [quick example](../../examples/noisy_simulation_v1.py)
- **Documentation:** [guide](../../docs/guides/NOISY_SIMULATION.md)
- **Known boundary:** Validated Markovian Kraus channels, timestamped DeviceNoiseProfile input, ASAP gate/idle thermal lowering, classical readout confusion, exact density execution, and reproducible MPS trajectories with single-rank adaptive stopping are available. Pulse overlap, crosstalk, leakage, provider calibration adapters, distributed adaptive stopping, batched statevector trajectories, production multi-GPU scheduling, and noisy gradients remain unsupported. Multi-wire MPS channels use an explicitly dense correctness fallback.


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
- **Public API:** `flagquantum.backends.run_mps`, `flagquantum.experimental.distributed.train_distributed_mps`, `flagquantum.experimental.mps.plan_production_mps`
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

### Interoperability adapter contract

Implement and certify optional external-framework conversion behind one immutable lazy registry and framework-neutral, loss-aware result contract.

- **Maturity:** Experimental
- **Public API:** `flagquantum.interop`
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
- **Public API:** `fq.experimental`
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
- **Public API:** `flagquantum.interop.qiskit.from_qiskit`, `flagquantum.interop.qiskit.to_qiskit`
- **Runtime modes:** `control_plane_conversion`
- **Hardware:** `cpu_control_plane`
- **Gradient support:** `symbolic_parameters_only`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/reference/API.md)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Certified against Qiskit 2.0.x and 2.5.x with Aer 0.17.x through an executable operation, wire-order, statevector, classical-bit, and round-trip contract. Qiskit control flow and arbitrary ParameterExpression import are rejected; named or multiple registers require explicit lossy flattening; custom multi-qubit unitary matrices remain blocked until basis ordering is specified. Conversion does not make Qiskit a runtime dependency or certify any provider hardware.

### Dynamic circuits and IQM Braket preflight

Execute dynamic circuits locally and prepare sealed IQM OpenQASM 3 programs with fail-closed hardware preflight.

- **Maturity:** Experimental
- **Public API:** `fq.experimental`
- **Runtime modes:** `local_statevector_trajectory`, `provider_preflight`
- **Hardware:** `cpu`, `amazon_braket_iqm_unverified`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../examples/braket_iqm_dynamic_preflight.py)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Provider-neutral conformance vectors pass on the FlagQuantum trajectory runtime and Qiskit Aer. IQM dialect serialization, SDK Program construction, sealed packaging, and mocked provider submission are tested. No AWS account or real IQM QPU task was used, so device availability, published qubit groups, billing, credentials, and hardware results remain unverified.

### Extension SDK

Build against the frozen extension protocol while qualifying each extension independently.

- **Maturity:** Experimental
- **Public API:** `flagquantum.extensions`
- **Runtime modes:** `extension_defined`
- **Hardware:** `extension_defined`
- **Gradient support:** `extension_defined`
- **Distribution semantics:** `extension_defined`
- **Start:** [quick example](../../examples/extensions/reference_extensions.py)
- **Documentation:** [guide](../../docs/reference/EXTENSION_SDK.md)
- **Known boundary:** The SDK protocol contract is frozen; individual extensions remain experimental until separately qualified.


## Validated public performance claims

Every value below is read from a hash-bound raw artifact. Missing or changed
evidence makes the source-of-truth check fail closed.

| Claim | Maturity and scope | Artifact-derived result | Recorded environment | Evidence identity |
| --- | --- | --- | --- | --- |
| **Sharded MPS exact-workload capacity**<br>`mps-capacity-131072-chi768-20260806` | `development_evidence`<br>One batch-one, complex64, χ768 MPS training step for the checked-in all-rank and all-boundary workload. This is not arbitrary statevector capacity, fixed-plan strong scaling, or release evidence. | **Sites:** 131,072<br>**Logical MPS state:** 1,236,780,012,864 bytes (1,151.84 GiB)<br>**Maximum elapsed time:** 367.37 s<br>**Maximum peak allocated memory per rank:** 77,745,407,488 bytes (72.41 GiB)<br>**Cumulative discarded weight:** 8.39e-06 | **Ranks:** 16<br>**Reported device memory per rank:** 85,093,777,408 bytes (79.25 GiB)<br>**CUDA allocator policy:** expandable_segments:True<br>**Topology fingerprint:** c74a91e3a224a4dd414cfbfbcb31da7570880d42e6aa4eaccecf4b85427940a5<br>**Metadata boundary:** The artifact records rank count, per-rank device memory, topology fingerprint, and CUDA allocator policy. It does not record the exact GPU model or Python, PyTorch, CUDA, NCCL, driver, host, and operating-system versions, so the claim is restricted to the recorded environment fields. | [raw JSON](../../benchmarks/results/local/mps_capacity_131072q_chi768_16xa800_repeat_complete_20260806.json)<br>SHA-256 `df8c19b74cc173799e3894025fb8e72de72a786d4542a9aa5e67687298f482f5`<br>code `9d56a6ecd78b06f11b9ee6e8aadcbe9644f2c708` |
