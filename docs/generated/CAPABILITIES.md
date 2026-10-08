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
| Optimize a program without changing its numerical result | Program optimization and target-aware compilation | Production supported | [Run example](../../examples/compiler_optimize.py) |
| Compile a program onto a constrained connectivity | Program optimization and target-aware compilation | Production supported | [Run example](../../examples/compiler_optimize.py) |
| Inspect schedule layers and routing decisions before execution | Program optimization and target-aware compilation | Production supported | [Run example](../../examples/compiler_optimize.py) |
| Rewrite a gate a target cannot run into gates it publishes | Target-basis gate and state-preparation synthesis | Development evidence | [Run example](../../examples/compiler_synthesis.py) |
| Synthesize a one-qubit or two-qubit unitary into a target basis | Target-basis gate and state-preparation synthesis | Development evidence | [Run example](../../examples/compiler_synthesis.py) |
| Prepare a state from its amplitude vector in a target basis | Target-basis gate and state-preparation synthesis | Development evidence | [Run example](../../examples/compiler_synthesis.py) |
| Find out whether a target basis can carry a program at all | Target-basis gate and state-preparation synthesis | Development evidence | [Run example](../../examples/compiler_synthesis.py) |
| Inspect a circuit diagram in a terminal | Circuit drawing | Experimental | [Run example](../../flagquantum/drawer/README.md) |
| Render a figure for a document or notebook | Circuit drawing | Experimental | [Run example](../../flagquantum/drawer/README.md) |
| Check a program's wiring and gate order by eye | Circuit drawing | Experimental | [Run example](../../flagquantum/drawer/README.md) |
| Draw with Circuit.draw or the flagquantum.drawer entry points | Circuit drawing | Experimental | [Run example](../../flagquantum/drawer/README.md) |
| Place a reusable block on chosen qubits of a wider circuit | Construction-time program composition | Development evidence | [Run example](../../docs/reference/API.md) |
| Undo a program by taking its inverse | Construction-time program composition | Development evidence | [Run example](../../docs/reference/API.md) |
| Apply a program a whole number of times | Construction-time program composition | Development evidence | [Run example](../../docs/reference/API.md) |
| Run a block only when a set of control qubits is all set | Construction-time program composition | Development evidence | [Run example](../../docs/reference/API.md) |
| Build a subroutine library and compose it into one circuit | Construction-time program composition | Development evidence | [Run example](../../docs/reference/API.md) |
| Simulate a small or medium circuit exactly | Local statevector simulation and training | Production supported | [Run example](../../examples/single_machine_quantum_ai/01_vqe_statevector.py) |
| Train a parameterized quantum circuit | Local statevector simulation and training | Production supported | [Run example](../../examples/single_machine_quantum_ai/01_vqe_statevector.py) |
| Run local VQE and quantum machine learning | Local statevector simulation and training | Production supported | [Run example](../../examples/single_machine_quantum_ai/01_vqe_statevector.py) |
| Differentiate a circuit without importing an internal gradient helper | One gradient entry point with a reported method | Production supported | [Run example](../../examples/gradient_methods/README.md) |
| Find out which gradient method a program can use | One gradient entry point with a reported method | Production supported | [Run example](../../examples/gradient_methods/README.md) |
| Train a circuit in statevector, MPS, or tensor-network mode | One gradient entry point with a reported method | Production supported | [Run example](../../examples/gradient_methods/README.md) |
| Train one statevector workload across multiple ranks | Sharded statevector training | Production supported | [Run example](../../examples/distributed_statevector_topologies/run.sh) |
| Plan distributed statevector ownership | Sharded statevector training | Production supported | [Run example](../../examples/distributed_statevector_topologies/run.sh) |
| Inspect communication and sharding semantics | Sharded statevector training | Production supported | [Run example](../../examples/distributed_statevector_topologies/run.sh) |
| Train one statevector workload across two nodes | Two-node statevector training | Release certified | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Resume a multi-node statevector run from its checkpoint | Two-node statevector training | Release certified | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Inspect inter-node statevector communication and rank placement | Two-node statevector training | Release certified | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
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
| Train one MPS workload across two nodes | Two-node MPS training | Release certified | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Resume a multi-node MPS run from its checkpoint | Two-node MPS training | Release certified | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Inspect inter-node site-boundary communication and rank placement | Two-node MPS training | Release certified | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
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
| Train one tensor-network workload across two nodes | Two-node tensor-network training | Production supported | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Resume a multi-node tensor-network run from its checkpoint | Two-node tensor-network training | Production supported | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Inspect inter-node slice-boundary communication and rank placement | Two-node tensor-network training | Production supported | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Validate small noisy circuits exactly | Exact and trajectory-based noisy simulation | Experimental | [Run example](../../examples/noisy_simulation_v1.py) |
| Write a channel directly in a circuit | Exact and trajectory-based noisy simulation | Experimental | [Run example](../../examples/noisy_simulation_v1.py) |
| Evaluate low-entanglement noisy circuits with MPS trajectories | Exact and trajectory-based noisy simulation | Experimental | [Run example](../../examples/noisy_simulation_v1.py) |
| Resume reproducible trajectory ensembles | Exact and trajectory-based noisy simulation | Experimental | [Run example](../../examples/noisy_simulation_v1.py) |
| Simulate driven amplitude damping | Continuous-time Lindblad density-matrix evolution | Production supported | [Run example](../../examples/lindblad_evolution.py) |
| Inspect time-resolved populations and observables | Continuous-time Lindblad density-matrix evolution | Production supported | [Run example](../../examples/lindblad_evolution.py) |
| Audit trace drift and numerical provenance | Continuous-time Lindblad density-matrix evolution | Production supported | [Run example](../../examples/lindblad_evolution.py) |
| Package a trained parameterized circuit | Circuit packaging and cloud deployment | Development evidence | [Run example](../../examples/train_parameterized_circuit_then_deploy.py) |
| Export a circuit for a provider | Circuit packaging and cloud deployment | Development evidence | [Run example](../../examples/train_parameterized_circuit_then_deploy.py) |
| Run a circuit through a deployment abstraction | Circuit packaging and cloud deployment | Development evidence | [Run example](../../examples/train_parameterized_circuit_then_deploy.py) |
| Predict a mapped QPU measurement distribution | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/remote/quafu_twin_region_holdout.py) |
| Compare a frozen prediction with later QPU counts | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/remote/quafu_twin_region_holdout.py) |
| Reject circuits outside validated topology and depth | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/remote/quafu_twin_region_holdout.py) |
| Distinguish exact released circuits from structurally compatible but unvalidated regional circuits | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/remote/quafu_twin_region_holdout.py) |
| Compose compatible local Twin cells into a connected regional prediction model | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/remote/quafu_twin_region_holdout.py) |
| Prospectively validate one exact regional circuit with repeated QPU tasks | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/remote/quafu_twin_region_holdout.py) |
| Validate a fixed regional circuit suite under one simultaneous confidence level | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/remote/quafu_twin_region_holdout.py) |
| Compare predeclared reference and holdout regional circuits | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/remote/quafu_twin_region_holdout.py) |
| Track the same fixed regional suite across calibration snapshots | Evidence-qualified QPU digital twins | Development evidence | [Run example](../../examples/remote/quafu_twin_region_holdout.py) |
| Discover registered interoperability adapters | Interoperability adapter contract | Experimental | [Run example](../../docs/reference/API.md) |
| Implement a framework adapter without changing FlagQuantum core | Interoperability adapter contract | Experimental | [Run example](../../docs/reference/API.md) |
| Certify round-trip and fail-closed adapter behavior | Interoperability adapter contract | Experimental | [Run example](../../docs/reference/API.md) |
| Handle conversion diagnostics consistently | Interoperability adapter contract | Experimental | [Run example](../../docs/reference/API.md) |
| Import a supported PennyLane QuantumScript | PennyLane QuantumScript interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Export static FlagQuantum IR to PennyLane | PennyLane QuantumScript interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Audit semantic loss at the boundary | PennyLane QuantumScript interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Execute FlagQuantum code on PennyLane Lightning | PennyLane Lightning execution bridge | Experimental | [Run example](../../docs/guides/PENNYLANE_LIGHTNING_EXECUTION.md) |
| Compare an external simulator without changing the native default | PennyLane Lightning execution bridge | Experimental | [Run example](../../docs/guides/PENNYLANE_LIGHTNING_EXECUTION.md) |
| Inspect external-backend provenance and fallback status | PennyLane Lightning execution bridge | Experimental | [Run example](../../docs/guides/PENNYLANE_LIGHTNING_EXECUTION.md) |
| Execute FlagQuantum code on Cirq Simulator | Cirq Simulator execution bridge | Experimental | [Run example](../../docs/guides/CIRQ_SIMULATOR_EXECUTION.md) |
| Compare an external simulator without changing the native default | Cirq Simulator execution bridge | Experimental | [Run example](../../docs/guides/CIRQ_SIMULATOR_EXECUTION.md) |
| Inspect external-backend provenance and fallback status | Cirq Simulator execution bridge | Experimental | [Run example](../../docs/guides/CIRQ_SIMULATOR_EXECUTION.md) |
| Import a supported Cirq circuit | Cirq circuit interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Export FlagQuantum IR to Cirq | Cirq circuit interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Audit semantic loss at a framework boundary | Cirq circuit interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Import a supported Amazon Braket circuit | Amazon Braket circuit interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Export FlagQuantum IR to Amazon Braket | Amazon Braket circuit interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Audit semantic loss at a framework boundary | Amazon Braket circuit interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Export FlagQuantum IR to a CUDA-Q kernel | CUDA-Q kernel export | Experimental | [Run example](../../docs/reference/API.md) |
| Audit rejected operations before kernel construction | CUDA-Q kernel export | Experimental | [Run example](../../docs/reference/API.md) |
| Check CUDA-Q wire-order assumptions at the boundary | CUDA-Q kernel export | Experimental | [Run example](../../docs/reference/API.md) |
| Choose among simulators measured for an exact FlagQuantum circuit | Evidence-based simulator advisor | Experimental | [Run example](../../docs/guides/SIMULATOR_ADVISOR.md) |
| Calibrate an arbitrary circuit under an explicit budget | Evidence-based simulator advisor | Experimental | [Run example](../../docs/guides/SIMULATOR_ADVISOR.md) |
| Inspect timing, stability, correctness, circuit identity, and confidence | Evidence-based simulator advisor | Experimental | [Run example](../../docs/guides/SIMULATOR_ADVISOR.md) |
| Reject unsupported inference outside measured evidence | Evidence-based simulator advisor | Experimental | [Run example](../../docs/guides/SIMULATOR_ADVISOR.md) |
| Import a supported Qiskit circuit | Qiskit IR interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Export FlagQuantum IR to Qiskit | Qiskit IR interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Audit semantic loss at a framework boundary | Qiskit IR interoperability | Experimental | [Run example](../../docs/reference/API.md) |
| Save a circuit as OpenQASM text | OpenQASM 2 and OpenQASM 3 interchange | Production supported | [Run example](../../flagquantum/compiler/README.md) |
| Load OpenQASM text back into FlagQuantum | OpenQASM 2 and OpenQASM 3 interchange | Production supported | [Run example](../../flagquantum/compiler/README.md) |
| Know exactly why a piece of OpenQASM was refused | OpenQASM 2 and OpenQASM 3 interchange | Production supported | [Run example](../../flagquantum/compiler/README.md) |
| Execute FlagQuantum code on Qiskit Aer | Qiskit Aer execution bridge | Experimental | [Run example](../../docs/guides/QISKIT_AER_EXECUTION.md) |
| Compare an external simulator without changing the native default | Qiskit Aer execution bridge | Experimental | [Run example](../../docs/guides/QISKIT_AER_EXECUTION.md) |
| Inspect external-backend provenance and fallback status | Qiskit Aer execution bridge | Experimental | [Run example](../../docs/guides/QISKIT_AER_EXECUTION.md) |
| Prototype mid-circuit measurement and feed-forward | Dynamic circuits and backend assessment | Experimental | [Run example](../../docs/reference/API.md) |
| Assess backend support before execution | Dynamic circuits and backend assessment | Experimental | [Run example](../../docs/reference/API.md) |
| Exercise a fixed-round QEC control workflow | Repetition-code memory experiment | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Inspect syndrome and detection-event records | Repetition-code memory experiment | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Prototype a decoder against a typed contract | Repetition-code memory experiment | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Declare a stabilizer code with its distance, qubits, checks and logical observables | Stabilizer-code memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Build the circuit source and detection layouts of a memory experiment from a code | Stabilizer-code memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Trace a detection event back to the check and round that produced it | Stabilizer-code memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Read one recorded measurement by the handle that names it rather than by its column index | Stabilizer-code memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Build a code record from the parity-check and logical matrices you already hold, rather than from a code class | Stabilizer-code memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Reach a code record by name, and read its X-type and Z-type ancilla bands and the two matching stabilizer counts rather than only their total | Stabilizer-code memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Build the detector error model a memory circuit defines under phenomenological noise | Detector error model and stim text interchange | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Build the model of a code whose parity-check and logical matrices are known, without writing a circuit | Detector error model and stim text interchange | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Read a stim detector error model and sample its detection events | Detector error model and stim text interchange | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Emit stim text for an external decoder or matcher | Detector error model and stim text interchange | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Decode a detection-event syndrome with a detector error model | Minimum-weight matching decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Inspect the weighted decoding graph a model defines | Minimum-weight matching decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Cross-check the matcher against an independent implementation over the same graph | Minimum-weight matching decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Compare a matcher's prediction against the observables a model sampled | Minimum-weight matching decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Select a decoder by name instead of by import | Minimum-weight matching decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Decode a syndrome of a model the matcher refuses | Belief-propagation decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Read a mechanism of three or more detectors as one variable | Belief-propagation decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Tell whether the exchange settled or the fallback answered | Belief-propagation decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Select the decoder by name through the registry | Belief-propagation decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Cut a detector error model into windows that each cover a few detector rounds | Round-window decomposition of a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Read which detector rows a window shares with its neighbours | Round-window decomposition of a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Rejoin the windows and recover the model exactly | Round-window decomposition of a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Tell whether a window size tiles the rounds the model has | Round-window decomposition of a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Decode a syndrome round by round instead of holding the whole block | Sliding-window decoder over a chunk decomposition | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Read one window's detector, observable and parity-check projections | Sliding-window decoder over a chunk decomposition | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Get the logical-frame prediction of a streamed decode | Sliding-window decoder over a chunk decomposition | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Change the window size or the inner decoder without touching the model | Sliding-window decoder over a chunk decomposition | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Sample detection events from a noisy memory experiment | Sampled detection events from a memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Reproduce a sampling run from a seed | Sampled detection events from a memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Decode a sampled syndrome with the model built from the same circuit | Sampled detection events from a memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Read one recorded measurement of a sampled run by the handle that names it | Sampled detection events from a memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Bind a noise channel to a gate the program executes and sample that fault | Sampled detection events from a memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
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
| Estimate the kernel matrix of a set of feature vectors | Quantum kernel estimation and kernel ridge classification | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Read a kernel entry off sampled swap tests | Quantum kernel estimation and kernel ridge classification | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Classify held-out rows with a kernel ridge classifier | Quantum kernel estimation and kernel ridge classification | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Build the binary objective of a feature-selection instance | Feature selection as a QUBO | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Read the objective's value off an assignment | Feature selection as a QUBO | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Map the objective to an Ising Hamiltonian for a solver to consume | Feature selection as a QUBO | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Estimate the fraction of a database's items whose support meets a threshold | Frequent-item fractions by amplitude estimation | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Read that fraction off an amplitude estimation counting register | Frequent-item fractions by amplitude estimation | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Compare the estimate against an enumerated frequent fraction | Frequent-item fractions by amplitude estimation | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Estimate a matrix's singular values from a counting register | Singular values by phase estimation over the Hermitian embedding | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Read one singular value off the register's mode | Singular values by phase estimation over the Hermitian embedding | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Compare a read-out value against the matrix's own decomposition | Singular values by phase estimation over the Hermitian embedding | Experimental | [Run example](../../docs/guides/ALGORITHMS.md) |
| Extrapolate an observable to zero noise from a model whose gate error strengths can be scaled | Zero-noise extrapolation over scaled noise models | Development evidence | [Run example](../../examples/algorithms/error_mitigation.py) |
| Read the residual and the variance amplification that qualify the estimate | Zero-noise extrapolation over scaled noise models | Development evidence | [Run example](../../examples/algorithms/error_mitigation.py) |
| Compare the unmitigated measurement against the mitigated estimate | Zero-noise extrapolation over scaled noise models | Development evidence | [Run example](../../examples/algorithms/error_mitigation.py) |
| Optimize an objective whose only accessible value is a sample | Simultaneous perturbation stochastic approximation | Development evidence | [Run example](../../examples/algorithms/spsa_optimizer.py) |
| Spend a fixed two evaluations per step instead of one per parameter | Simultaneous perturbation stochastic approximation | Development evidence | [Run example](../../examples/algorithms/spsa_optimizer.py) |
| Replay a seeded run from the call site | Simultaneous perturbation stochastic approximation | Development evidence | [Run example](../../examples/algorithms/spsa_optimizer.py) |
| Sample a wide Clifford circuit that has no representable amplitude store | Clifford circuit sampling by Pauli stabilizer tracking | Development evidence | [Run example](../../examples/stabilizer_sampling.py) |
| Reproduce a sampling run from a seed | Clifford circuit sampling by Pauli stabilizer tracking | Development evidence | [Run example](../../examples/stabilizer_sampling.py) |
| Measure a chosen subset of qubits in a chosen output order | Clifford circuit sampling by Pauli stabilizer tracking | Development evidence | [Run example](../../examples/stabilizer_sampling.py) |

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

### Program optimization and target-aware compilation

Transform one FlagQuantum IR program without executing it: canonical optimization, logical layer scheduling, and lowering onto an explicit coupling topology.

- **Maturity:** Production supported
- **Public API:** `flagquantum.compiler.optimize`, `flagquantum.compiler.compile`, `flagquantum.compiler.route_to_topology`, `flagquantum.compiler.schedule_layers`, `flagquantum.compiler.CouplingMap`, `flagquantum.compiler.plan_trivial_layout`, `flagquantum.compiler.plan_dense_layout`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../examples/compiler_optimize.py)
- **Documentation:** [guide](../../flagquantum/compiler/README.md)
- **Known boundary:** In-process transformation of one FlagQuantum IR v1 program on CPU; the compiler never executes a program or certifies hardware behavior. optimize() removes identity gates, merges self-inverse runs, and merges adjacent rotations to a fixed point. compile() runs that same fixed point, optionally routes onto one explicit CouplingMap, optimizes again, and records post-routing optimization. Routing offers four caller-selected strategies: restore_after_each_gate, persistent_layout, sabre, and sabre_layout. All four restore the identity logical order at the program output, so a caller cannot ask for an unrestored final permutation, and the cost-estimate automatic selection covers only restore_after_each_gate and persistent_layout because the two SABRE strategies plan SWAPs instead of estimating them; the topology is always caller-supplied and never discovered. Initial placement is planned separately: plan_trivial_layout places logical qubit i on physical qubit i, and plan_dense_layout places the program on the most densely connected window of device qubits a breadth-first walk from every physical qubit finds, which is the only placement that can help on a device with unequal connectivity and the only one of the two that can reach a program the lowest qubits cannot hold at all. That walk is a heuristic and not an exact optimum: over 55 device and program-width combinations it matched an exhaustive search of every connected window 45 times and returned a window with one fewer internal connection the other 10 times. Both return a placement as a tuple of physical qubits, one per logical qubit, which is what route_to_directed_topology takes as initial_layout; final_layout reports the assignment a routed program ended on as a Layout, and a route that allocates workspace on a device wider than the program reports the device width there with its idle slots as None in physical_to_logical. Both placements collapse to the identity on a device no wider than the program. Neither reaches route_to_topology, which refuses an initial_layout because a plain CouplingMap routes only over qubits the program owns; a non-identity placement requires a DirectedCouplingMap, whose physical workspace owns the idle slots and is cleaned by the inverse routing SWAPs, and it is passed to route_to_directed_topology or to legalize_circuit_topology. remove_layout_restore drops the appended logical-restore phase from a program that was already routed, halving the SWAP count over a batch of random programs for persistent_layout because that strategy's restore is a full reverse replay of its forward SWAPs and leaving restore_after_each_gate unchanged because its restores are interleaved with the program, and its result is a program for a local simulator or a provider that accepts per-physical-qubit results rather than deployable routing evidence: the deployment routing contract rejects it and the deployment entry points treat it as unrouted and route it again. Window selection is greedy over device connectivity and never estimates routing cost, so it can cost more SWAPs than the trivial placement on a program whose interactions prefer a sparser window, and it is not noise- or calibration-aware. Scheduling constructs deterministic logical ASAP layers with explicit qubit and classical-data dependencies; it is not target timing or pulse scheduling. There is no directed acyclic graph intermediate, no composable pass manager, no pass analysis or preservation metadata, and no in-repository pass extension point: canonicalization functions are pipeline internals, and an installed external compiler is selected by name through the extension SDK rather than composed as a pass. Pulse-level, calibration-aware, and parameter-aware compilation, dynamic-circuit control flow, and noise-aware optimization beyond lower_noise_model are unsupported. The structured hybrid program slice under flagquantum/compiler/_hybrid is private and absent from public exports. Compilation is single-process; no distributed or multi-device compilation path exists.

### Target-basis gate and state-preparation synthesis

Spell gates and states a target cannot express in the gates that target publishes, and report a basis that cannot carry them instead of approximating it.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.compiler.native_gate_legalization.legalize_native_gates`, `flagquantum.compiler.one_qubit_synthesis.synthesize_one_qubit`, `flagquantum.compiler.two_qubit_synthesis.synthesize_two_qubit`, `flagquantum.compiler.state_preparation_synthesis.synthesize_state_preparation`, `flagquantum.compiler.basis_translation.translate`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../examples/compiler_synthesis.py)
- **Documentation:** [guide](../../flagquantum/compiler/README.md)
- **Known boundary:** In-process target-basis rewriting of one FlagQuantum IR v1 program on CPU; it never executes a program, chooses a target, or certifies hardware behavior, and no performance, fidelity, or hardware claim is made. The synthesis entry points are deliberately not stable flagquantum.compiler exports: synthesize_one_qubit, synthesize_two_qubit and synthesize_state_preparation are reached by module path, and only legalize_native_gates applies them as a whole-program pass. A synthesized rewrite equals its source only up to one global phase, which FlagQuantum IR has no field to record, so every entry point refuses rather than approximating. The refusals are per entry point rather than uniform: synthesize_one_qubit declines z_rotation='u3' and pulse_opcode='ry' while it does accept z_rotation='phase' for a single gate, and the state-preparation ladder declines 'phase' and 'u1' at the same keyword because a ladder needs an entrywise-exact rz. synthesize_two_qubit declines any entangler outside the supercontrolled table, 'ecr' among them, and it is handed a matrix rather than an instruction because the matrix of a named two-qubit gate belongs to flagquantum.simulation, which this layer must not import; it holds only the three parameter-free entangler matrices and the closed form of the rotation family, so a named two-qubit opcode reaches it only after the equivalence table has rewritten it or has reported the gate the target is missing. That table declares eighteen identities from named opcodes to named opcodes and composes them recursively, because a named gate carries no matrix on this layer; an entry that is not itself exact therefore does not stay exact under further expansion, and cphase drops a global phase. For ccx the table's T-form was measured against Qiskit's MCXGrayCode statement and kept: the Gray-code form declares seven leaves against fifteen but costs eight two-qubit gates on the interaction basis against six, and its further expansion runs through cphase, the one table entry that is not exact. Multi-controlled decomposition above three controls is unsupported: FlagQuantum IR declares no arity above three, Qiskit builds MCXGrayCode from an ancilla chain only from five controls, and the V-chain and recursive forms that do cover four controls and above use fewer entanglers than Gray code but need caller-supplied ancillas the IR does not model. The one-qubit angle form and the two-qubit block form are exact only to the working precision of their own arithmetic, and a rewrite whose leaves leave the native set is reported as the opcode the target is missing rather than as the shortest near miss. Synthesis is single-process CPU work with no distributed, multi-device, pulse-level, calibration-aware, or noise-aware path.

### Circuit drawing

Render one circuit as terminal text or a Matplotlib figure through one shared layout, without executing it.

- **Maturity:** Experimental
- **Public API:** `flagquantum.drawer.draw`, `flagquantum.drawer.draw_text`, `flagquantum.drawer.use_style`, `flagquantum.drawer.available_styles`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../flagquantum/drawer/README.md)
- **Documentation:** [guide](../../flagquantum/drawer/README.md)
- **Known boundary:** Presentation only. The drawer owns layout, labels, gate symbols, and styles; it does not execute, compile, validate, or route a circuit, and it changes no numerical result. The text renderer depends only on the standard library and Core, while Matplotlib is optional and is imported only by the figure renderer, so draw_mpl is absent from flagquantum.drawer when Matplotlib is not installed. The rendered text layout is not a frozen contract: no compatibility guarantee is made for exact spacing, gate symbols, column placement, or option names, and the only tests assert renderer self-consistency rather than a golden diagram. Rendering is single-process CPU work with no streaming or interactive backend. Structured hybrid programs, pulse-level schedules, and target timing diagrams are not drawable. The package README is the only user-facing documentation.

### Construction-time program composition

Build a larger program out of smaller ones in place: place one program on chosen qubits, invert a program, apply a program a whole number of times, and run a program only when added control qubits are all set, without introducing a second IR.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.Circuit.compose`, `flagquantum.Circuit.adjoint`, `flagquantum.Circuit.power`, `flagquantum.Circuit.control`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/reference/API.md)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** All four methods are Stable Core additions to the fq.Circuit type, approved by docs/api-changes/FQ-CIRCUIT-COMPOSITION-20261002.md, docs/api-changes/FQ-CIRCUIT-ADJOINT-20261003.md, docs/api-changes/FQ-CIRCUIT-POWER-20261021.md and docs/api-changes/FQ-CIRCUIT-CONTROL-20261022.md, and the four-member family is complete: every operation it names has an approved proposal. A noise channel, a dynamic operation, a classically conditioned instruction, a matrix without a conjugate transpose and an unknown opcode without a matrix are each refused with CapabilityError rather than copied forward. Composition, inversion, control and repetition are rewrites of qubit labels and not a second program description: none of the four changes IR_VERSION, each emits instructions the IR already describes, and none adds a root export, so no existing program changes meaning. Circuit.power returns a new circuit that applies the receiver's program an integer number of times, and it is exact for every instruction: the whole program is appended in order, so a gate with no angle, a custom operation carrying a matrix and a noise channel all work, and a negative exponent is adjoint().power(-k) so that the inverse rule keeps one implementation. One rewrite shortens the common case -- a receiver that is exactly one instruction declaring exactly one parameter has that parameter multiplied by the exponent instead of the gate being repeated -- and the rule belongs to the gate, declared as a power_rule on every opcode and measured through the dense operator by tools/check_circuit_composition_contract.py: 12 of the 35 registered opcodes take the closed form and the other 23 repeat, with u2 and u3 repeating because scaling all of their angles is a different operator. power(1) copies rather than multiplying by one, so a symbolic parameter or a trainable angle stays the same object, and power(0) is the empty program of the same shape, which keeps p.power(a).power(b) equal to p.power(a * b) at a = 0. A fractional exponent is an owned gap: the IR has no instruction form for a matrix root, so power(0.5) raises TypeError rather than approximating, while PennyLane 0.45.1 constructs and runs qml.Hadamard(0) ** 0.5; closing it needs a new Instruction form and is therefore an IR change rather than a composition change. Requests that would emit more than 4096 instructions are refused with ValueError naming the count. Circuit.control returns a new circuit in which every gate runs only when the control qubits it added are all set. The control qubits are added rather than borrowed, so each label must lie outside the receiver's own range and the result's width covers them, and an input state set on the receiver is carried into the wider register with every added qubit in |0>. The controlled form belongs to the opcode and is declared once in flagquantum/core/operator_schema.py next to the inversion and shift rules: 31 of the 35 registered opcodes declare one, the four that do not are the noise channels, and a channel, a dynamic operation, a classically conditioned instruction or an instruction carrying its own matrix is refused with CapabilityError rather than approximated. A gate the registry already has in controlled form is used directly, so x under one control is cx and swap under one control is cswap, and every wider control is emitted as an ancilla-free phase ladder of registered gates, which keeps a symbolic angle symbolic and the result inside the autograd graph. The price is depth and it is published: a w-qubit block under k controls needs a ladder of level k + w - 1, which emits on the order of 3 ** level instructions, MAX_LADDER_LEVEL is 10, and a request past the ceiling is refused by name. The surface is single-process CPU work on one program with no execution, compilation, routing, noise handling, or scalability claim, and it is not differentiable by itself because composition records instructions rather than evaluating them. contracts/circuit-composition-contract.toml is the authoritative list of these guarantees and of every refusal, and tests/unit/test_circuit_composition_contract.py reads it back against the implementation."

### OpenQASM 2 and OpenQASM 3 interchange

Write one FlagQuantum program as OpenQASM text and read that text back as the same program, with every other text refused by name.

- **Maturity:** Production supported
- **Public API:** `fq.from_openqasm`, `flagquantum.compiler.openqasm.emit_openqasm`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `bound_parameters_only`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../flagquantum/compiler/README.md)
- **Documentation:** [guide](../../flagquantum/compiler/README.md)
- **Known boundary:** Text interchange only; the importer never executes a program and does not make FlagQuantum an OpenQASM front end. Import accepts exactly the canonical subset the emitter writes and verifies that claim by re-emitting the parsed program and requiring statement equivalence, so text that denotes something the emitter would write differently is refused rather than interpreted approximately, and arbitrary third-party OpenQASM is out of scope. Both 2.0 and 3.0 are read; the version and its matching include statement are mandatory, so a program declaring any other version is refused as unsupported_version. One qreg or qubit register and one creg or bit register may be declared, and a program declaring more is malformed_statement. Because OpenQASM 3 spells the three-angle gate as U and has no two-angle gate, U is read as u3 unconditionally: u2(phi, lbd) and U(pi/2, phi, lbd) denote the same unitary and the emitter writes the same text for both. The inverse of sx is the one power accepted, spelled pow(-1) @ sx q[i]; every other power is unknown_gate. Parameters must be bound numbers, so rx(theta) q[0]; is unbound_parameter. barrier and reset are not opcodes and are refused. The measurement block must be terminal, must define every classical bit exactly once, each bit must read a distinct qubit, and a classical register wider than the quantum register or a partly defined measurement block is incomplete_measurement. Comments, blank lines, and spacing carry no meaning; any other textual difference from the canonical emission is not_canonical_text. Eleven refusal reasons form a closed vocabulary, each carried on OpenQASMImportError.issue_code, which is the first issue_code in the package. Import returns an OpenQASMImport rather than a Circuit because a Circuit cannot carry a measurement, and IR_VERSION stays 1.0 because import happens at construction time.


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

### One gradient entry point with a reported method

Differentiate a parameterized circuit through any exact execution mode without choosing an executor-specific route, and read which method produced the derivative.

- **Maturity:** Production supported
- **Public API:** `fq.gradient`, `fq.jacobian`, `fq.jvp`, `fq.vjp`
- **Runtime modes:** `statevector`, `mps`, `tensor_network`
- **Hardware:** `cpu`, `single_gpu`
- **Gradient support:** `exact`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../examples/gradient_methods/README.md)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Only statevector, MPS, and tensor-network mode can serve the expectation value a differentiable program needs; stabilizer mode samples outcomes and fails closed instead. method='autograd' and method='parameter_shift' are exact; method='finite_difference' and method='spsa' are declared approximations whose result reports exact=False together with the displacement actually used. method='adjoint' is refused, because FlagQuantum has no standalone adjoint entry point: the reversible sweep is reachable only as the backward pass behind PyTorch autograd, and no result reports whether backward used adjoint replay. method='auto' measures the program rather than declaring a route, so it probes for an autograd graph before it resorts to a shift rule or a difference. The displacement method='finite_difference' defaults to is the cube root of the machine epsilon of the dtype the loss delivers, not of the dtype the parameters carry, so a program that casts inside its loss gets the step its own arithmetic justifies. fq.jacobian, fq.jvp, and fq.vjp exist for a program that returns several values, so that no single scalar gradient is defined; all three are autograd-only -- there is no shift rule for a vector output and no reportable displacement -- so a program whose output carries no PyTorch graph is refused rather than answered with a zero derivative. fq.jvp needs the program differentiated twice and is refused with CapabilityError where an executor cannot build a graph of a graph; fq.jacobian needs one backward pass per output element, and fq.vjp needs one backward pass in total whatever the parameter count. Every result is detached, so the derivative of a derivative requires an explicit second route rather than nesting these calls.

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
- **Known boundary:** General reverse contraction and production distributed transport are not certified. Noise channel instructions are unsupported in this mode and fail closed instead of falling back to statevector. Execution spanning more than one host is the separate two-node capability, whose evidence covers one declared cut on one recorded pair and does not extend back to arbitrary host counts, automatic slicing, or wider cuts.

### Exact and trajectory-based noisy simulation

Write a channel into a circuit or lower a validated Kraus noise model into FlagQuantum IR, then execute exact density-matrix or MPS quantum-trajectory paths.

- **Maturity:** Experimental
- **Public API:** `flagquantum.noise.NoiseModel`, `flagquantum.noise.channel_from_parameters`, `flagquantum.noise.noisy_density_matrix`, `flagquantum.runtime.run_noisy_mps`, `flagquantum.runtime.run_noisy_statevector`
- **Runtime modes:** `density_matrix`, `noisy_mps`
- **Hardware:** `cpu`, `single_gpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_device_fast_path_or_rank_local_trajectory_partition`
- **Start:** [quick example](../../examples/noisy_simulation_v1.py)
- **Documentation:** [guide](../../docs/guides/NOISY_SIMULATION.md)
- **Known boundary:** Validated Markovian Kraus channels, timestamped DeviceNoiseProfile input, ASAP gate/idle thermal lowering, classical readout confusion, exact density execution, and reproducible MPS trajectories with single-rank adaptive stopping are available. Four channel opcodes (bit_flip, phase_flip, depolarizing, amplitude_damping) declare the scalar their factory takes, so they can be written directly into a circuit as well as attached to a gate through a NoiseModel rule; the two routes lower to the same instruction, and a program is planned as noisy because it carries a channel, not because a model was passed. That same reading of the program decides the trajectory representation under mode='auto' and restores it from a saved plan, so an inline channel and an equivalent NoiseModel rule select the same route. A representation that holds amplitudes -- mps, tensor_network, and statevector -- refuses a channel instead of returning the noiseless number, and mode='stabilizer' names its own obstacle. A channel instruction's matrix field is the Kraus tuple, so consumers read it as a sequence rather than as one tensor. Only those four opcodes are reachable by name: the remaining flagquantum.noise callables, including thermal relaxation, readout error, and coherent overrotation, are NoiseModel rules only. Channel parameters are bound real scalars; a trainable channel probability has no gradient path. Lowering refuses a rule naming a qubit outside the program width and refuses a model whose rules match no instruction at all, so a misspelled gate name cannot yield a clean result; a model that matches some instructions stays legal. Pulse overlap, crosstalk, leakage, provider calibration adapters, distributed adaptive stopping, batched statevector trajectories, production multi-GPU scheduling, and noisy gradients remain unsupported. Multi-qubit MPS channels use an explicitly dense correctness fallback.

### Continuous-time Lindblad density-matrix evolution

Evolve small Markovian open systems on explicit time grids with physical collapse rates and time-resolved diagnostics.

- **Maturity:** Production supported
- **Public API:** `flagquantum.lindblad.run`, `flagquantum.lindblad.plan`, `flagquantum.lindblad.LindbladPlan`, `flagquantum.lindblad.amplitude_damping`
- **Runtime modes:** `continuous_time_density_matrix`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../examples/lindblad_evolution.py)
- **Documentation:** [guide](../../flagquantum/simulation/README.md)
- **Known boundary:** Time-independent dense Hamiltonians, finite strictly increasing grids, Markovian Lindblad collapse operators, and CPU complex64/complex128 execution only. The explicit grid controls fixed-step fourth-order Runge-Kutta accuracy. Non-Markovian environments, stochastic trajectories, gradients, sparse solvers, GPU/distributed execution, hardware submission, and arbitrary time-dependent generators are unsupported.

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
- **Known boundary:** A synchronous local reference for one fixed three-data-qubit repetition-code profile. It supports bounded deterministic X-error schedules, replaceable per-round trajectory decoding with physical-X or Pauli-frame-X actions, and a two-round temporal rule that rejects an isolated readout excursion. Confirmable data errors require a following round; terminal-round onsets remain unconfirmed. The circuit-location stochastic profile contains independent bit flips after parity-check CNOTs and independent syndrome/final-readout confusion. The middle data qubit has two CNOT noise opportunities per round while edge qubits have one. Feedback traces separate true and observed bits, actions, and frame evolution. Sweeps report finite-shot observations only, not logical suppression or thresholds. The temporal rule is not maximum-likelihood decoding and repeated readout faults may mimic data errors. Batched decoder feedback, general channels/codes, correlated or timing noise, hard-real-time/provider control, gradients, distributed execution, capacity, performance, and fault-tolerance claims remain unsupported. The feedback records are private subinterfaces and the namespace is not exported from the stable package root.

### Stabilizer-code memory circuit

Declare a stabilizer code record -- directly, by name, or from the parity-check and logical matrices a caller already holds -- and turn it into a memory-experiment source with the detector and observable layouts that experiment implies, for the repetition code, the rotated surface code and the Steane code.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.build_memory_circuit`, `flagquantum.qec.RepetitionCode`, `flagquantum.qec.RotatedSurfaceCode`, `flagquantum.qec.SteaneCode`, `flagquantum.qec.css_code_matrices`, `flagquantum.qec.CssCodeMatrices`, `flagquantum.qec.CssCode`, `flagquantum.qec.CodeCheck`, `flagquantum.qec.Pauli`, `flagquantum.qec.MemoryCircuit`, `flagquantum.qec.DetectorLayout`, `flagquantum.qec.ObservableLayout`, `flagquantum.qec.MeasurementRef`, `flagquantum.qec.MeasurementSamples`, `flagquantum.qec.sample_memory_measurements`, `flagquantum.qec.ancilla_bands`, `flagquantum.qec.get_code`, `flagquantum.qec.code_names`, `flagquantum.qec.register_code`
- **Runtime modes:** `local_statevector_dynamic_hybrid`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/README.md)
- **Documentation:** [guide](../../flagquantum/qec/IMPLEMENTATION.md)
- **Known boundary:** A code-independent declaration layer, not a decoder and not an experiment runner. build_memory_circuit emits bounded hybrid source text plus detector and observable layouts; executing that source is the Runtime's dynamic hybrid path and carries that path's own limits. Three code records exist: the repetition code, the rotated surface code and the Steane code, the last of which is the first record carrying both a Z-type and an X-type logical observable. Each record is reachable by name through get_code, listed by code_names and added with register_code, which checks the protocol members at registration and refuses a duplicate name unless replace is set; the names are repetition, rotated_surface and steane, and the registry is the factory half of upstream's get_code and get_available_codes rather than a plugin boundary. A record declares its ancilla total and also the X-type and Z-type bands, and num_x_stabilizers and num_z_stabilizers beside them, which is the full accessor list upstream's code record carries as pure virtuals; the two bands are derived from the checks by ancilla_bands rather than declared beside them, and the two stabilizer counts read those same bands rather than recounting the checks, because each of the three code classes upstream ships answers the same number to a stabilizer count as to the band count of that basis: the builder refuses a record whose stated counts disagree with the bands its own checks define, so a record cannot report a split that contradicts it, while a flag or idle ancilla measures neither basis and is in neither band, leaving the two bands free not to cover the total. The colour code is not among the records, and it is not a gap: upstream's headers declare no colour code and no concrete code type at all, only the abstract code interface and the three families its own factory names, Steane, repetition and surface, of which this repository now carries one of each. A check states one ancilla and one CNOT direction chosen by the check's type, so a mixed X-and-Z stabilizer is refused; a code needing a second ancilla per stabilizer requires the record to grow first. The readout basis is a field of the record and defaults to the Z one, so one code record can describe two experiments: the basis states which of the code's logical observables the terminal readout measures, and it is checked against the observables the code declares rather than assumed from them, so a code declaring no logical observable in the requested basis is refused rather than measured under premises that do not hold for it. Upstream derives the same basis from the preparation kernel it is handed, where here it is stated on the record, because the Steane code declares a logical observable of each type and the code alone would not say which of the two experiments was built. Upstream's separate x_ and z_ model entry points stay absent deliberately, because a basis is a component of the record here rather than a second way to ask for one thing. Detector counts follow the two check classes, the readout basis' own class contributing rounds + 1 and the other rounds - 1 -- z_checks * (rounds + 1) + x_checks * (rounds - 1) for the default Z basis -- and are validated against the code's checks and the configured rounds, but the validation is membership-based: it does not check that a detector names the same check in every round, that a terminal detector's data qubits are the support of the check it belongs to, or that the emitted source is the program the layouts describe, so a hand-built layout can be semantically wrong while passing every check. A detection event is a parity of recorded measurements, and the record is readable at the level underneath it: MemoryCircuit.measurement_refs is every measurement the experiment records -- one handle per check per round in round-major order, then one per data qubit for the terminal readout -- read off the code and the round count rather than collected from the layouts, and sample_memory_measurements returns those outcomes with MeasurementSamples.outcome, .vector and .integer reading one handle, a chosen sub-vector as booleans in the order asked for, and that same order packed with the first handle least significant. The two sets need not agree in either direction, and the reason the handle layer exists is the direction where the layouts are right to be smaller: round zero of a patch measures the check class its preparation does not stabilize -- an X-type check under the all-zero preparation, a Z-type check under the |+> one -- whose outcome is not deterministic, so no detector may name it, and it is still measured and still readable. A detector's parity recomputed from the handles that detector names equals the detector's parity shot for shot, which the tests assert per detector and per observable rather than assuming. The layer makes no claim about which handles are deterministic, and a test measures the border instead of stating it: a noiseless run reads the readout basis' own check class' ancilla as 0 in every round and the other class' ancilla as unbiased, the two swapping places under an X-basis preparation, and an individual terminal data qubit as unbiased, while every detector and every observable reads 0. A rate read off a single handle is therefore a rate read off the state and not a channel rate, and a detector is a parity and not a measurement. A reading that does not fit is refused rather than approximated: a handle the experiment does not record raises naming the handle, an empty reading raises, a reading addressed by anything but a MeasurementRef raises as a type error, and a record whose tensor is not two-dimensional or whose column count does not match its handle vector raises rather than being truncated. What the layer is not is a kernel annotation form -- detector, detectors and logical_observable as kernel calls are not implemented here, and identity is set in a layout record beside the source rather than in a kernel body the compiler would have to carry into IR. A code also does not have to be declared as a class: a caller holding the four CSS blocks -- hz, hx, lz and lx -- builds a code record out of them, and the data-qubit count, the ancilla total with its two bands, the checks with their CNOT direction, the stabilizers and the logical observables are read off those blocks, so an arbitrary parity-check matrix has a route in without a class for it and a code no class here declares, the nine-qubit Shor code for instance, reaches a memory circuit and a detector error model. Two things about that route are deliberate. Its distance is stated by the caller rather than derived, because a minimum-weight-codeword search is exponential in general and would refuse exactly the large matrices the route exists for, while the one direction a stated logical operator does prove is enforced: an operator of weight w bounds the distance above by w, so a record claiming more than the lightest operator it states is refused and one that understates its distance is accepted. And the blocks are held to the algebra a matrix record normally leaves unchecked: two check blocks that do not commute, a logical operator meeting the opposite basis' checks on an odd number of qubits, a logical operator inside its own checks' span, two dependent rows in one logical block, and a check or logical row with no support are each refused with the row and the reason named, which is a strengthening of the baseline rather than a copy of it. What the route still does not carry is identity: a record built this way is not registered by name, because its identity is the matrix rather than a string, so get_code reaches the declared three and not a matrix-built record. Nor is a code here a map from an operation to the kernel that performs it, which is upstream's shape: a record declares its checks and one source is built from them, so a code carrying its own preparation, measurement or logical-operation kernels would need a second shape, and an arbitrary non-CSS stabilizer list has no route in for the same reason a mixed X-and-Z stabilizer is refused, because the matrices route is CSS-shaped by construction. The rotated surface code's declared logical observable is Z on one data row; which weight-d logical operator a patch uses is a convention, and this record states the one it uses rather than deriving all of them. Agreement with stim's generated rotated-memory circuit is on the declared detector and observable count at distances two through eight and rounds one through four, which fixes the shape and not a mechanism-level correspondence. Execution evidence covers the distance-three patch only, because the dynamic hybrid simulator draws each shot from the full output distribution and a distance-five patch needs 2**49 categories. No decoding, no threshold, no logical-suppression, no fault-tolerance, no real-time and no hardware-feedback claim follows from a declared code, a detector layout, or a syndrome trace.

### Detector error model and stim text interchange

Build the exact Pauli-noise detector error model of a memory circuit or of a code-capacity CSS record, sample it, and exchange it with stim as text in both directions.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.DetectorErrorModel`, `flagquantum.qec.DemError`, `flagquantum.qec.DemMergeRule`, `flagquantum.qec.DemSample`, `flagquantum.qec.css_code_matrices`, `flagquantum.qec.decoder_context_from_memory_circuit`, `flagquantum.qec.DecoderContext`, `flagquantum.qec.DecoderInputs`, `flagquantum.qec.MeasurementMap`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/IMPLEMENTATION.md)
- **Documentation:** [guide](../../flagquantum/qec/IMPLEMENTATION.md)
- **Known boundary:** An exact construction for Pauli noise over the reference Clifford gate set, and a text reader for the stim detector error model format. Construction does not sample: a mechanism's signature comes from one forced execution whose two shots must agree on the detector and observable flips they produce, and a non-Pauli channel is refused with a stated reason rather than approximated. The agreement is required of the flip set and not of the raw register: an X-type ancilla is prepared in |+>, so its round-zero outcome is a coin toss, while the steady-state X-type detector it feeds is deterministic because the first round projected both compared rounds into the same eigenstate. Demanding register agreement refused every code with an X-type check, so the rotated surface code could not be modelled at all. Because the signature comes from an execution, construction is bounded by the statevector amplitude ceiling: a distance-2 patch is 7 qubits and a distance-3 patch is 17, both of which build in seconds, while distance 4 is 31 qubits (2**31 amplitudes) and did not complete in 45 minutes, and distance 5 is 49 qubits and fails on the allocator outright. The modelled rotated-surface distance is therefore 3 on the circuit route, and that ceiling is a property of forcing a signature through an execution rather than of the construction contract: reading a matrix support costs one pass over its nonzero entries per round, so a distance-5 or distance-7 patch reaches a model as easily as a distance-2 one. A second construction route reads a code-capacity experiment rather than a memory circuit. DetectorErrorModel.from_code_matrices takes a CssCodeMatrices record -- the four CSS blocks hz, hx, lz and lx -- and derives every mechanism combinatorially, and css_code_matrices lifts a code record into that record, refusing a logical observable that is neither pure X nor pure Z rather than half-reading a CSS code. Its geometry is not the memory circuit's and the difference is pinned rather than glossed: a fault in round r reaches the detector band of round r and of round r + 1, the final round has no band after it, and there is no terminal data readout, so the detector count is num_rounds * num_checks where a memory circuit gains one terminal detector per check of the readout basis it is built in. Both geometries are asserted against their own stim transcription, and the same suite asserts that the two counts differ. The matrix route is compared to stim one readout basis at a time, because lz and lx anti-commute and no one state has both as a deterministic value: the reference experiment is run twice against the model restricted to the matching basis, and each run's observable rows are the ones the Pauli character of that fault family predicts. A fault that is still in the data at the end of the run is still seen by the logical readout, so a physical fault spans one detector band while the model gives it two; that single divergence is pinned as the exact relation between the two mechanism sets rather than as a tolerance. The fault family is the three single-qubit Pauli faults plus a check's syndrome bit flipped at readout, so an X fault flips Z-type checks, a Z fault flips X-type checks and a Y fault flips both, and the X detector band and the lx observable rows exist on the matrix route. The rate record is no longer what is narrower than upstream's. PhenomenologicalNoise states upstream CssNoise's four families as four scalars -- data_flip, phase_flip, both_flip and measurement_flip, which are its px, pz, py and pm -- and the same four per element as data_flip_per_qubit, phase_flip_per_qubit, both_flip_per_qubit and measurement_flip_per_check, which are its px_per_qubit, pz_per_qubit, py_per_qubit and pm_per_check. A stated vector replaces its scalar wholesale rather than mixing with it, a vector that does not name every data qubit or every check is refused naming both counts instead of being partially applied, and an element whose effective rate is zero enumerates no mechanism at all, so a quiet location between two noisy ones is expressible and costs nothing to sample. The per-qubit vectors are indexed in the code's own data_qubits order, which is the column order of every matrix, and the per-check vector by matrix row with the Z-type checks first; a code may declare its checks in any other order -- the rotated surface code interleaves the two types -- so the two orders are related by one named translation that all three readers share. What remains different from upstream is the carrier rather than the rates: upstream folds the record into the argument its matrix entry point takes, so the vectors are validated against matrices that arrive with it, where here the record is a standalone frozen dataclass and its vectors are resolved against a count the reader supplies. The sampler reads all four families, so a rate reaches a sampled record and a model alike: the engine executes one channel, so the Z and Y data families are placed as that bit flip conjugated by h and by sdg/s, one draw at that family's rate rather than a pair of independent bit flips. A matrix that is not binary, is not two-dimensional, or is not indexed by the same data qubits as another non-empty block is refused rather than coerced. The matrix route is graphlike only while it stays inside one round; at two rounds the middle data qubit of the distance-3 repetition code flips the same check in both bands, which is a four-detector mechanism, and the matching decoder refuses it as a hyperedge. The model is built on the memory circuit without in-circuit feedback, because it describes the noise-to-detection mapping a decoder inverts, so the frozen profile's compiled-feedback path is unmodelled. Detector rates are cross-checked against rates sampled from the circuit simulator, but the cross-check shares the injection helper with construction, so it is not an independent re-derivation of the signatures. An independent check does exist and runs against the optional stim distribution: the memory circuit is transcribed into a stim circuit and stim's own error analysis builds the model, which agrees with this one on shape, mechanism count and detector marginals at distance 2 and 3. That check shares nothing but the circuit's gate sequence and the noise model. The text reader accepts everything stim 1.16.0 wrote in a 240-model developer-time sweep, repeat blocks and # comments included, and refuses a declaration that skips an index, an error mechanism that flips nothing, a malformed line, a block that never closes and a closing brace with no block open, naming each rather than dropping it. Sixty of the swept models carry a repeat block, and reading one is an expansion rather than a second statement of the format, so the block's meaning is compared rather than assumed: this reader's model of a block-carrying text equals its model of the same text after stim's own flattened(), and both equal stim's reading of the block, across the repetition-code and rotated-surface-code shapes stim was measured to write a block in. The shift a block contains advances once per iteration rather than once per block, which is what places the copies of a repeated mechanism at successive detectors, and repeat 0 is legal and contributes nothing. A line's ^ separators are read the way stim means them -- the signature is the symmetric difference of the targets, so a detector named twice cancels and the groups are not retained -- and the decomposition the separators suggest is available under upstream's own flag name as from_stim_text(text, use_decomp_suggestions=True), which returns one mechanism per component at the probability the line states and turns 536 mechanisms into 1042 on a distance-5, two-round rotated surface code. That second reading is a different distribution from the line rather than an equivalent statement of it, and it is measurably so: against stim's own sampler on the same text the default reading's worst marginal misses by 0.0016, inside the 0.004 tolerance, while the expanded reading misses the observable marginals by 0.048, twelve times outside it, which is what makes the tolerance evidence rather than decoration. A component that cancels to nothing is refused by the expanded reading alone, with the default reading named as the route that states the line. The circuit route states the same kind of named, lossy second reading of its own. DetectorErrorModel.from_memory_circuit takes decompose_composite_faults, which enumerates a composite data fault -- the Y family, an X flip and a Z flip at one location -- as those two parts, each forced through the same injector at the parent's round and qubit and read by the same signature reader the single-Pauli families use, rather than as one mechanism whose signature is their exclusive-or. It exists because a composite fault is the one hyperedge a memory circuit states: on a distance-three rotated surface code at two rounds the combined model states four mechanisms of three detectors and one of four and the matching decoder refuses it, while the decomposed model states none above two and the decoder builds on it. The claim it makes is bounded and stated with it: a part is graphlike exactly where the corresponding single-Pauli fault at that location is and no further, the two readings are the same joint law only in the marginals -- each detector's and each observable's rate is unchanged to floating-point rounding, measured at 5.6e-17 worst case, while the mechanism sets are asserted to differ -- and the default is the combined reading, so no earlier caller's model changes and a repetition code or a one-round surface experiment, where the parts are already separate, reaches the same model either way. The code-capacity route deliberately takes no such keyword: its detector bands make one fault span two of them, so a part there still reaches four detectors at two rounds on rotated surface codes at distances three, five and seven, and the keyword would promise a graphlike model that route cannot deliver. The text writer is exact on this side and stim's width is the platform's: stim prints at numeric_limits<long double>::digits10 + 1, so nineteen significant digits where long double is the x86 80-bit extended type (the Linux CI runners, where a reprint returned every swept probability unchanged) and sixteen where it is a double (arm64 macOS, where about a quarter of the same sample came back changed); seventeen digits name a double uniquely, so a round trip through stim must compare against a re-read of the text rather than against the in-memory model exactly where the width is the narrower one, and the drift there is bounded by one part in 10**15 and measured at 5.4e-16 worst case, while tests/qec/test_dem_stim_text_precision.py reads the width off stim's own output and pins the digit count, the direction of the loss and the bound against it. A CodeCheck states one ancilla and one CNOT direction fixed by the check's type, so a code with both check types is expressible while a mixed X-and-Z stabilizer is refused; the detector grammar is therefore the readout basis' own class contributing rounds + 1 and the other rounds - 1 -- z_checks * (rounds + 1) + x_checks * (rounds - 1) in the default Z basis -- rather than one detector per check per round, and a code family needing a second ancilla per stabilizer still requires that record to grow first. dem_sampling reports a DEM-sampled rate and is labelled as one wherever it is reported; no decoder, no threshold, no logical-suppression, no real-time or hardware-feedback, and no performance claim follows from a detector error model or from any rate it reports. DemSample carries tensors and is deliberately unhashable. The merge is a stated operation and not only a construction step: DemMergeRule names its two rules for what they compute, INDEPENDENT_PARITY being the probability that an odd number of a duplicated group fires and CLAMPED_LINEAR_SUM the sum of the group clamped at one, mechanisms_are_unique and require_unique_mechanisms are the predicate and the refusal, and the parity rule is exact rather than approximate because a detector's rate is a product of 1 - 2p factors over the mechanisms touching it, so a group's combined prior is the single p whose factor is that product and regrouping the factors cannot change any detector or observable rate; the sum rule does not preserve them and is offered only for a caller who means it. A model may also state that mechanisms are alternatives rather than independent, which is the one statement its parity matrices cannot carry, because two columns of a parity matrix are independent by construction: DemError.error_id is an optional non-negative label, the mechanisms sharing one are mutually exclusive, and DetectorErrorModel.error_ids projects upstream's optional vector back out, None when no mechanism states an id, which is upstream's nullopt and the state stim's text always describes. Exclusivity is given a distribution rather than left as a bare flag: the members of a group are disjoint pieces of one shot and each keeps the probability it states, the left-over mass is the group firing none of them, and a group whose probabilities sum above one is refused rather than renormalized, because renormalizing would change every rate the caller read; a group summing to exactly one is admitted, and a lone id excludes nothing and behaves as if it were unstated. The model's own arithmetic honours the groups, so the marginal rate of a target sums the group members that touch it instead of composing them by parity -- 0.26 for two independent mechanisms of 0.1 and 0.2 against 0.30 for the same two stated as alternatives -- and sampling draws one uniform per group and lets at most one member fire, so a model with no ids, which is every model the construction routes here produce, keeps the arithmetic and the draws it had before the field existed. DetectorErrorModel.error_rates is the parallel rate column the matrices are read against: entry i is the rate mechanism i itself states rather than a marginal, its length is num_errors and therefore the column count of detector_error_matrix and observables_flips_matrix, and its order is the model's own normalized mechanism order rather than the order a caller stated its mechanisms in, so a reader takes a column's support and that column's weight from one index. It is deliberately not a distribution, so a grouped model reports each member's own rate and the folding stays in detector_rates and observable_rates, which are different quantities again. The statement is not transcribable, and the operations that would silently drop it refuse instead: to_stim_text names the ids it would lose rather than printing an independent model under the same probabilities, and merge_duplicate_mechanisms names them because either rule combines mechanisms by assuming independence and would invent a shot in which two alternatives both fired. What remains absent is an operation that folds a group under its exclusivity rather than merging by independence, which is the gap the canonicalize field row records. A decoder is handed both halves of what it needs, not only the model. detector_context_from_memory_circuit builds the model once and returns a DecoderContext whose full_component, x_component and z_component each return a DecoderInputs -- the model together with both maps -- and whose num_measurements states the width those maps index. A MeasurementMap is one row per detector or observable holding the measurements whose parity it is: dense() projects to the (rows, measurements) orientation upstream stores as D and O, and flattened() to the -1-terminated sparse form a realtime decoder configuration takes, the terminator being what keeps a row that reads nothing visible to a sparse reader. The map is closed rather than conventional -- a row is sorted because a parity is a set, it may not name one measurement twice because two reads of one measurement cancel, and an index outside the buffer is refused. The basis split is a projection of the model as built rather than a second construction: a detector belongs to the check whose ancilla it reads, so the component of the experiment's own readout basis carries the terminal detectors and the union is the model itself, with no boundary-aware variant needed because the layout states its own boundary detectors. Two refusals keep the projection a reading rather than a rewrite: a component of a model that states error ids is refused, because projecting a group of alternatives would either drop the correlation or merge two of its members, and a component whose basis has no detector is refused rather than returned empty. The numbering the maps use is derived from the circuit's declaration and asserted equal to the numbering the sampler derives from the lowered program, so the two readings of one record convention are pinned to each other by measurement rather than by a shared constant. The context does not execute the circuit, costs one construction and no simulation, and adds no canonicalization and no chunk or seam helper of its own. test_dem_signatures.py, test_dem_sampling.py, test_dem_cross_check.py, test_dem_from_memory_circuit.py, test_dem_public_surface.py, test_dem_merge.py, test_dem_error_ids.py, test_dem_rate_overrides.py, test_dem_rates.py, test_decoder_context.py, and tests/qec/test_dem_code_matrices.py, whose stim cross-check is tests/qec/test_dem_code_matrices_stim.py.

### Minimum-weight matching decoder over a detector error model

Turn the detector error model of a memory circuit into a weighted graph and decode a syndrome exactly by minimum-weight matching, reporting the mechanisms selected and the logical observables they flip.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.DecodingGraph`, `flagquantum.qec.DecodingGraphEdge`, `flagquantum.qec.MinimumWeightMatchingDecoder`, `flagquantum.qec.MatchingDecodeResult`, `flagquantum.qec.PyMatchingDecoder`, `flagquantum.qec.get_decoder`, `flagquantum.qec.register_decoder`, `flagquantum.qec.decoder_names`, `flagquantum.qec.DetectorErrorModelDecoder`, `flagquantum.qec.AUTHORITY_NAME`, `flagquantum.qec.CROSS_CHECK_NAME`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/IMPLEMENTATION.md)
- **Documentation:** [guide](../../flagquantum/qec/IMPLEMENTATION.md)
- **Known boundary:** One decoder, for graphlike detector error models only, on one detector error model at a time, with no threshold, no logical-suppression, no real-time, and no performance claim. A decoding graph is built from the model's mechanisms and nothing else: a mechanism becomes an edge whose weight is log((1 - p) / p), a mechanism flipping one detector becomes an edge to a boundary node one past the last detector, a mechanism flipping three or more detectors is a hyperedge and is refused with a stated reason rather than projected onto a pair, a mechanism of probability zero contributes no edge because its weight would be infinite, and two mechanisms that share a detector pair but disagree on their observable labels stay two edges because merging them would either lose a logical flip or invent a weight neither mechanism has. Two mechanisms that agree on both are the different case and are refused: one fault stated twice would become two parallel edges and the matcher would charge the cheaper of them, log 4 for mechanisms of 0.1 and 0.2 where the fault's own combined parity 0.26 has weight log(0.74 / 0.26), so the matcher would prefer a longer chain of other mechanisms over the mechanism that actually fired. The decoder therefore requires a unique signature before it builds a graph, names both mechanisms and the merge that resolves the duplicate, and leaves the graph's parallel edges intact so the refusal is a decoder decision and not a representation one. A model that states two mechanisms are alternatives is refused for the neighbouring reason: one weight per mechanism is the weight of a fault that fires alone, while a group of alternatives is one fault whose weight is the negative log-likelihood of the group, so the graph refuses such a model and names the groups rather than weighing each member as though the others could fire with it. Whatever the model states is what the graph holds, so the decoder inherits every scope limit of the detector error model it consumes, including the modelled rotated-surface distance of 3. The matcher is exact and self-implemented: it searches each defective detector for the cheapest chain of mechanisms to every other detector and to the boundary, and then enumerates the ways to pair the defects and to route any number of them to the boundary, which is a minimum-weight perfect matching on the metric closure. The boundary is a sink rather than a waypoint, which is what makes a syndrome's parity irrelevant, and chains may pass through defective detectors because edges two chains share cancel in the symmetric difference. Because the pairing is enumerated, the decoder accepts at most twenty detection events per syndrome by default and refuses a larger syndrome as a capability boundary rather than returning a pairing that only looks cheapest; the budget is a constructor argument because the cost is in that enumeration, so a distance-three patch at ordinary noise decodes while a much larger syndrome is declined. Detectors that no chain of mechanisms connects are refused with a stated reason rather than answered partially. The prediction is the exclusive-or of the selected mechanisms' observable labels, so it is a decision about which cheapest explanation the decoder adopts and not an estimate of the probability that the observable flipped, and it is not calibrated: two mechanisms that share a detector pair and disagree on their label are indistinguishable from the syndrome alone, so the decoder cannot always recover the more likely value. Its failure rate is therefore at or above the optimal decoder's on every model and strictly above it once rounds exceed one. A single-round repetition model has no such pair, and there the matcher reaches the exhaustive optimum exactly, which is the only optimality claim made here. Absent: a union-find or belief-finding decoder, and the plugin protocol the registry is not. A sliding-window decoder is no longer among them: it is a decoder of its own, described in the capability detector_error_sliding_window_decoder, and it reaches this matcher or the belief-propagation decoder as the inner decoder of each of its windows. Belief propagation with ordered statistics decoding is no longer among them: it is a decoder of its own, described in the capability that follows, and it reads the hyperedges this one refuses rather than projecting them. The hyperedge-decomposition front end that would widen the graphlike scope has landed on the construction side rather than here, because a hyperedge is a statement the model makes and the decoder can only refuse it: DetectorErrorModel.from_memory_circuit takes decompose_composite_faults, which reads the circuit route's one composite fault -- the Y family, an X flip and a Z flip at one location -- as those two parts at the parent's rate instead of as one mechanism whose signature is their exclusive-or. On a distance-three rotated surface code at two rounds the composite model states four mechanisms of three detectors and one of four and this decoder refuses it as a hyperedge, while the decomposed model states none above two and the same decoder builds on it and returns corrections for the syndromes those mechanisms explain. The option is not a relabelling and is not offered as one: the parts always fire together in the composite fault and independently when read apart, so each detector's and each observable's marginal rate is unchanged to floating-point rounding while the joint law is not, and the default stays the composite reading so no earlier caller's model changes. The matrix route states no such keyword deliberately, because there a fault's part spans two detector bands and still reaches four detectors at two rounds; a hyperedge from any other source is still refused by name. Batching, streaming, and a per-detector error-rate vector are also absent, and the decoder is not connected to the stim sampling path. The matrix route that DetectorErrorModel.from_code_matrices provides is inside this decoder's graphlike domain only while it stays inside one round: at two rounds the middle data qubit of the distance-3 repetition code flips the same check in both detector bands, which is a four-detector mechanism, and the decoder refuses it as a hyperedge rather than projecting it onto a pair. The observable prediction has no consumer yet: no logical-error-rate estimator, no threshold scan, and no integration with the memory-experiment result records. The result record is deliberately not the repetition-code DecodeResult, whose correction carries a qubit in {0, 1, 2} and an X basis only and therefore cannot express a surface-code correction. A second decoder exists behind the pymatching extra and is a cross-check rather than the authority: flagquantum.qec.adapters translates the same decoding graph, reports the same two records, and is imported lazily, so the core install and the decoder neither need nor load it. What the comparison establishes is stated narrowly, because the two instruments are not equal: PyMatching reports 3.9020747171643912 for a mechanism this package states as 3.9020746947749574, so the graph it minimizes over is not exactly this one and two explanations closer than that difference can be ordered differently on the two sides. The two decoders are therefore compared on the cheapest weight of every syndrome with a tolerance of one single-precision rounding per selected mechanism, on the observables wherever the cheapest explanation is unique, and on their refusals as the same set; a tie is uncomparable rather than evidence against either implementation, and one tie of the distance-three repetition code is pinned by hand so that the clause is a demonstrated fact rather than a place a disagreement could hide. The translation refuses two graphs instead of approximating them, and both are graphs the matcher itself answers: a detector pair carrying two mechanisms, which PyMatching's independent merge strategy would collapse and thereby lose a logical-label difference, and a detector that no mechanism flips, which PyMatching cannot represent because it infers its detector count from its edges. The cross-check's domain is therefore narrower than the authority's and never the reverse. The matcher is checked against brute force on syndromes small enough to enumerate, against the exhaustive enumeration of a model's own mechanism distribution, and against that second decoder over the same graph, so its correctness rests on the detector error model all three read and on the authority of no external decoder. A name reaches a decoder through flagquantum.qec.get_decoder, and the name is not a promise about the implementation: the registry returns the class a name is registered against and never prefers one because it is available, so the authority is returned for its own name wherever the optional extra happens to be installed. The registry is narrower than the CUDA-Q QEC decoder surface it is aligned to. Its source argument takes the three carriers a caller can hold a model in -- the detector error model, stim's text for one, and the decoding graph the model defines -- and not a parity-check matrix, because lifting a matrix would also mean choosing the noise model and round count that DetectorErrorModel.from_code_matrices reads rather than defaults. It holds the detector-error-model family alone: the repetition-code decoders take an ordered syndrome history rather than detection events, so registering both under one name space would make a name mean one of two things, and they stay directly constructed. It fails closed rather than falling back: an unregistered name raises and lists the registered names, a class missing decode, from_detector_error_model or from_decoding_graph is refused while the registering module is imported rather than at the first caller, a source that is not one of the three carriers is refused by type, and a second registration of one name is refused unless the caller asks to replace it. The optional implementation is registered whether or not PyMatching is installed, so its name is part of this package's surface rather than the extra's and asking for it without the extra raises the error naming the extra. The registry selects among decoders; the second one it selects is the belief-propagation decoder described in the capability that follows, and the hyperedge-decomposition front end is a construction option rather than a registry entry, so the decoder side now holds the matcher, the cross-check, the belief-propagation decoder and the sliding window over a chunk decomposition, and what remains unimplemented there is the plugin boundary, which upstream ships as a precedent rather than as a protocol here.

### Belief-propagation decoder over a detector error model

Decode a detector error model's syndrome by exchanging beliefs between its mechanisms and its detectors, reading a mechanism of three or more detectors rather than refusing it, and answer an exchange that does not settle by solving an ordered-statistics information set exactly.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.BeliefPropagationDecoder`, `flagquantum.qec.BeliefPropagationDecodeResult`, `flagquantum.qec.BELIEF_PROPAGATION_NAME`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/IMPLEMENTATION.md)
- **Documentation:** [guide](../../flagquantum/qec/IMPLEMENTATION.md)
- **Known boundary:** One decoder over one detector error model at a time, on the CPU, with no threshold, no logical-suppression, no real-time and no performance claim, and no batching. The factor graph is the model as written and not a projection of it: a mechanism is one variable, a detector is one check, a mechanism touching three detectors joins three checks, and a mechanism of rate zero is refused rather than given a finite edge, because its log-likelihood ratio is not a number and a mechanism no shot can select is not evidence. The prior of a variable is the log-likelihood ratio log((1 - p) / p) of that mechanism's own rate, and not a marginal or a folded group rate. The exchange is sum-product in the log domain with each check's own syndrome bit in the update, so a check answers the complement of what it would answer with the bit left out, and the message a check sends a variable excludes that variable's own report, which is what separates the check's evidence about the variable from what the variable already said. A run that reaches an iterate whose hard decision explains the syndrome returns it and reports converged, and the empty syndrome is answered from the priors alone before any message is sent, with an iteration count of zero. Off a tree the exchange is approximate and this is stated rather than hidden: a model whose factor graph has cycles can settle on an explanation that flips the syndrome and is heavier than the cheapest one, so a caller who needs the cheapest explanation on a graphlike model wants the matcher and a caller who needs a mechanism of three detectors read needs this decoder and a heavier correction than the minimum. A syndrome that no set of mechanisms can produce is refused by name, because the mechanisms span a subspace of the detector space and a syndrome outside it did not come from this model; returning the least-bad set would be a correction that does not explain what it was asked about. A model that states its mechanisms are alternatives is refused for the neighbouring reason the matcher refuses one: the exchange weighs each variable as an independent fault, while a group of alternatives is one fault whose members cannot fire together, so reading the group as independent would invent shots in which two of its members both fired. When the exchange does not settle, the fallback orders the mechanisms by their posterior belief, takes a greedily chosen independent set of their columns as the information set, solves the reduced system over it, fixes every other mechanism at its belief decision, and refuses an inconsistent residual rather than returning a set that does not explain the syndrome; a caller who needs the exchange's own answer rather than a solve can turn the fallback off, and then an unsettled run is refused instead. The result record is the decoder's own and deliberately not the baseline's DecoderResult: it carries the convergence flag, the selected mechanisms, the logical observables they flip, the total weight and the iteration count, and it has no batch form, no async form and no optional-results channel, so a caller who needs the baseline's batch record is not served here. The decoder does not implement the sliding window itself, which lives in the capability detector_error_sliding_window_decoder: this decoder reads one model and one syndrome at a time, and a windowed decode over a chunk decomposition reaches it as the inner decoder of each window rather than through this record, so a long-lived experiment whose syndrome history exceeds one shot wants that decoder and not this one. It reads one model and does not sample: no logical-error-rate estimator, no threshold scan, no decoder configuration record and no consumer of the observables beyond the caller.

### Round-window decomposition of a detector error model

Cut a detector error model into windows that each cover a few detector layers and share one boundary layer with the next, name each window's boundary seams, and stitch or close the windows back into the model they were cut from.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.SeamId`, `flagquantum.qec.PhaseId`, `flagquantum.qec.DemSeam`, `flagquantum.qec.ChunkLayout`, `flagquantum.qec.DemChunk`, `flagquantum.qec.DemChunkSpec`, `flagquantum.qec.DemChunksSpec`, `flagquantum.qec.dem_chunk_from_spec`, `flagquantum.qec.dem_chunks_from_spec`, `flagquantum.qec.dem_stitch`, `flagquantum.qec.dem_stitch_all`, `flagquantum.qec.dem_stitch_merged`, `flagquantum.qec.dem_close`, `flagquantum.qec.dem_close_all`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/IMPLEMENTATION.md)
- **Documentation:** [guide](../../flagquantum/qec/IMPLEMENTATION.md)
- **Known boundary:** One model at a time, cut in memory on the CPU, with no decoder, no threshold, no streaming and no performance or memory claim. This is the substrate a sliding-window decoder slides over and not that decoder: nothing here decodes, and the chunk-scoped projections of a window live beside the decoder that reads them in flagquantum/qec/sliding_window.py, so a caller who wants one window's detector matrix takes the whole model's through flagquantum.qec.dem_close_all and projects it with the model's own detector_error_matrix, or reads the sequence through flagquantum.qec.dem_chunks_to_pcm. What a window is, is fixed by the model rather than chosen by the caller. A layer is the set of detectors with one round index, so the layer widths are the round sizes in detector order and their sum is the model's detector count; the layout that states them is read off a memory circuit by ChunkLayout.from_memory_circuit, which is also where the numbering is pinned: a terminal detector, whose parity references the terminal data readout and no round, is placed in the last layer rather than given a round of its own, and a numbering that reaches a layer after skipping one is refused rather than closed up, because renumbering the rounds after a missing one would silently move every window that follows it. A window is a contiguous run of layers and its bands are fixed rather than configurable: the leading boundary layer of its predecessor, its own interior, and the trailing boundary layer its successor shares, so a window of w layers spans w - 2 interior layers and the two end windows of a decomposition span one layer fewer than a middle one. The stride is fixed at the window width minus one, which is what makes the windows share exactly one layer each and therefore tile: a caller who wants to advance by less reads overlapping windows and simply does not stitch them. A window of one layer, a window wider than the layout, and a layer count that does not tile into windows at that stride are all refused with the layer count named, so a decomposition is a statement the caller can check rather than a best effort. Ownership makes the windows a partition rather than a cover. A mechanism spans exactly two adjacent layers, because a fault is placed at one location and flips that round's detectors and the round before's, and a mechanism whose detectors lie inside one layer does not exist here; so the first window that contains a mechanism whole owns it, every mechanism is owned exactly once, and the windows are disjoint in what they hold. A mechanism that no window contains whole is refused and never split, because splitting it would state two weaker faults where the model stated one; a mechanism that flips no detector is refused because an observable-only fault has no round to be placed in and choosing one would be inventing placement. Two operations follow from the fixed bands. A stitch contracts the shared layer of two adjacent windows and lays it out once, between their interiors, which is why the result spans one layer fewer than the sum of its parts; a close lays every window of a decomposition out at its own place in the model's own numbering, so a close of a decomposition is the model again and the two round trips are asserted as equalities in tests/qec/test_dem_chunks.py rather than assumed. A seam here is identified by its name text rather than by a hash of one, which is a deliberate difference from the baseline and is recorded in the alignment checklist: there is no name registry for a diagnostic to consult and no hash that can collide. A seam's rows carry the global detector index, not a position within the seam, so a stitch refuses two bands that merely have the same width and compares identities instead. A chunk may name only the two standard boundary seams, at most once each, ordered by row, with the leading band starting at local row zero and the trailing band ending at the last local row and the bands not overlapping; a stitch additionally refuses a non-adjacent pair, an observable count that differs between the two sides, a boundary either side does not carry, and a close of a sequence refuses a window count or an observable count that disagrees window by window. dem_stitch_merged is the stitch followed by the model's own merge under a stated rule, which is the only place in this capability where a probability is combined, and a close does not merge: closing a decomposition gives back the model it was cut from, duplicates and all. The spec layer is linear and not a phase graph. init, bulk and final label a window's position, and a single-window decomposition is bulk alone; a window carries no prev_round seam when no layer precedes it and no next_round seam when none follows, so an init window's leading edge and a final window's trailing edge are the cut rather than a seam. A caller may state a decomposition's phases directly instead of taking the derived sequence. The baseline's sparse spec shorthand and its separate matrix per seam have no form here because a chunk is cut from a model whose rows are already laid out in layers, so there is nothing to expand against a round count supplied at expansion time; and the baseline's per-seam tags and its straddle flags are absent, while its dem_chunks_to_d_sparse / dem_chunks_to_o_sparse / dem_chunks_to_pcm projections have landed with the sliding-window decoder that reads them, in flagquantum/qec/sliding_window.py, and are described in the capability detector_error_sliding_window_decoder. Nothing here samples, nothing here decodes, and no window is offered as a smaller model a decoder may be run on in place of the whole: a window carries its boundary seams precisely because its own arithmetic at a shared layer is incomplete without its neighbour.

### Sliding-window decoder over a chunk decomposition

Decode a detector error model a few detector rounds at a time by sliding a window over the model's chunk decomposition, committing each window's correction as the next window opens, and project the sequence onto one shared fault column space.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.SlidingWindowDecoder`, `flagquantum.qec.SlidingWindowDecodeResult`, `flagquantum.qec.dem_chunks_to_d_sparse`, `flagquantum.qec.dem_chunks_to_o_sparse`, `flagquantum.qec.dem_chunks_to_pcm`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/README.md)
- **Documentation:** [guide](../../flagquantum/qec/IMPLEMENTATION.md)
- **Known boundary:** One block at a time, one window at a time, on the CPU, with no threshold, no logical-suppression, no real-time and no performance or memory claim, and no batching beyond the one stream a decoder holds. The decoder is built from a DemChunksSpec rather than from a model, because the windows are what it slides over, and it is deliberately not a name the registry holds: a registered decoder is built from a model or from the graph one defines, and a window geometry is not what a model states, so a caller reaches it by constructing it and names only the registry decoder each window is built from. Two entry points replace the baseline's one. decode takes a whole block in the model's detector numbering and splits it into the layout's rounds, and decode_round takes one round in that round's own numbering, which is the numbering the baseline's per-round vector already has; the baseline's decision between the two paths by comparing a length is the kind of overloading a stated entry point does without. A whole block cannot be decoded into a block already in progress, and a refused round leaves the stream exactly where it was. What an inner decoder must report is the substantive narrowing. A window's syndrome is its own detector rows less the support of the mechanisms the window before it committed, because two adjacent windows share the boundary round's detectors and one event may not be explained twice; forming that residue needs the selected mechanisms, so an inner record must state them as mechanisms in that window's own column order, which BeliefPropagationDecodeResult does and MatchingDecodeResult does not, since that record states a correction as graph edges. A window built with the authority matcher therefore refuses at the first commit by name rather than silently dropping the correction, and that is recorded as a property of that record rather than worked around here. A window that cannot explain its residue refuses rather than guesses, and that is a limit rather than a defect: a window's residue is a subsystem the window need not span, so an arbitrary detector subset is not a syndrome any window promises to read, and a narrow window can abandon a block a wider one decodes. Measured on the distance-three repetition code at four rounds, a two-round window abandons the terminal mechanisms' syndromes while a three-round window answers every mechanism of the same model, and the tests assert that directly by driving the model's own mechanisms rather than random subsets. An inner decoder's refusal resets the stream and re-raises the inner reason, which is what makes the next round start a new block instead of continuing a window whose residue is now unexplained. The result record states the committed faults and the observables that vector flips, so the two are one statement read twice rather than two answers free to disagree; a caller who projects a window's mechanisms independently gets the same observables. This is a deliberate divergence from the baseline, which returns an empty vector until the final window because its result vector is indexed by a column space in which the shared columns are still open, where no column is shared here: dem_chunks_from_spec already cut the matrices, so a window's correction is final when it is committed and there is no column offset to resolve. The complete flag is what the baseline's empty-vector return encodes, and the converged field is the inner decoders' flags anded, or None when none of them reports one, because MatchingDecodeResult has no such flag and crediting a decoder that said nothing with settling would be worse than saying nothing. The three projections are read through dem_close_all, which refuses a sequence that is not a decomposition of one model before any column is named, and a numbering past what a realtime decoder can address is refused rather than wrapped. dem_chunks_to_pcm and dem_chunks_to_o_sparse index one shared fault column space in the order dem_close_all states, so each is the closed model's own detector or observable matrix in sparse form and neither is ever built dense; dem_chunks_to_pcm is deliberately the trail-row form rather than a dense matrix product, because a long model's mechanism count makes the dense product the thing to avoid and the row and column index spaces are what a caller reads either way. dem_chunks_to_d_sparse is the only one of the three that meets a third numbering, the measurement bit of a flat rounds-by-d buffer, so it is round-major and its precondition -- every round of the sequence the same d detectors wide -- is checked rather than assumed, which is what makes it refuse a surface code's boundary rounds by name: a distance-three surface code's boundary rounds are four rows wide where its interior rounds are eight, so its detectors are not pairs of uniform rounds and its correspondence comes from the circuit's measurement handles through MeasurementMap instead. The decoder reads one model and does not sample: no logical-error-rate estimator, no threshold scan, no decoder configuration record, and no consumer of the observables beyond the caller. Nothing here establishes fault tolerance, and the window size that decodes one model need not decode another.

### Sampled detection events from a memory circuit

Sample a memory experiment's detection events and observable flips from the circuit itself under a phenomenological noise record or a gate-bound noise model, so a decoding claim has a route that does not depend on the statevector ceiling.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.sample_memory_circuit`, `flagquantum.qec.sample_memory_measurements`, `flagquantum.qec.MeasurementSamples`, `flagquantum.qec.MeasurementRef`, `flagquantum.simulation.stabilizer.sample_noisy_measurements`
- **Runtime modes:** `stabilizer`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/README.md)
- **Documentation:** [guide](../../flagquantum/qec/README.md)
- **Known boundary:** One experiment shape and two noise grammars, selected by the record the caller states rather than by a second entry point. The circuit must be a MemoryCircuit, so the sampler is reachable from a code record and not from an arbitrary annotated circuit. A PhenomenologicalNoise states round boundaries, so a data fault opens a round and a measurement flip corrupts a check's readout, and each of those locations can be given its own rate through the record's per-qubit and per-check vectors, so a quiet qubit between two noisy ones is sampled as stated. A NoiseModel states gates, so a rule's fault follows the gate its rule matched and is placed in every round that gate appears in, which is one location per ruled gate rather than one per round boundary; a rule whose gate the program does not carry places nothing, because a record stated over a gate set says nothing about a gate outside it. All three Pauli data families are placed on both grammars, the Z and Y ones as the engine's single bit-flip channel conjugated by h or by sdg/s at a round boundary and by h,x,h after a named gate, which is one draw at that family's rate rather than a pair of independent bit flips, so a family the conjugation could not express would be refused rather than sampled as a nearby Pauli. A depolarizing or damping channel still has no location on either grammar, because a mixture is not one Pauli fault; a gate-bound rule naming a readout or a preparation, a one-qubit channel bound to two wires, and a fault that would follow the program's last instruction are each refused by name; and the per-check vector is resolved against the code's check count before any placement is derived, so a profile that names the wrong number of checks is refused rather than silently read against the wrong one. Placement is derived from the lowered program rather than written into the experiment, because the bounded hybrid capture refuses a channel call in the source; the derivation requires the program to lower to the experiment's own readout rotation around `rounds` identical round blocks that measure every check once in the code's declared check order, and refuses a program that does not, so an experiment whose round structure differs is declined rather than sampled. The sampler executes on the stabilizer engine, so it requires the optional `stim` distribution and inherits the engine's opcode set: exactly the thirteen Clifford opcodes, with a non-Clifford gate refused by name. It reads the record's readout basis, so an experiment prepared and read out in |+> is sampled here as directly as a Z-basis one: the basis contributes one rotation instruction per data qubit at each end of the lowered program, and the plan states that offset beside the round block rather than folding the two together, because the rotation belongs to the experiment and not to a round. Sampling is not differentiable and returns no gradients. Two entry points read one run: sample_memory_circuit returns the detection events and observable flips as a DemSample, and sample_memory_measurements returns the recorded measurements as a MeasurementSamples whose handle vector is the MemoryCircuit's measurement_refs, so a caller can read a bit that no detector names. A handle is addressed as a MeasurementRef rather than by column index, and reading one is refused with the handle named when the experiment does not record it, so the two entry points cannot disagree about which column is which. measurement_refs covers every measurement the experiment records rather than only the subset the layouts name, which is the wider set: round zero of a patch measures the check class its preparation does not stabilize, which is not deterministic, so no detector may name it and it is still readable. The measurements entry point therefore makes no claim that a handle is deterministic, and a noiseless run reads the check class its preparation does not stabilize as unbiased and an individual terminal data qubit as unbiased while every detector and observable reads 0, so a rate read off a single handle is a rate read off the state rather than a channel rate. The result is a DemSample like the model's own sampler returns, and the two are deliberately different code paths: the model derives a mechanism's signature from the source while this sampler derives its instruction position from the lowered program, and the tests pin the two to each other rather than having one call the other. A seed reproduces a run on one engine version and one machine instruction set; it does not pin a bit pattern across machines or engine versions. No threshold, no logical-suppression, no fault-tolerance, no real-time and no performance claim follows. The rate this sampler reports is a circuit-sampled rate, and the model's own sampler reports a DEM-sampled one; neither may be reported as the other.

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
- **Known boundary:** Reversible classical logic of O(n) Toffoli-style cost with no advantage premise of its own. A multi-controlled X above two controls needs len(controls) - 2 caller-supplied ancillas, each of which must be in |0> on entry: measured, a dirty ancilla makes the target silently wrong on a large fraction of inputs (8 of 16 at three controls, 32 of 96 at four) while never corrupting the ancilla itself, so the failure is invisible from the ancilla. The comparator restores every qubit it is given except the target; its len(lhs) + 1 equality flags and its scratch qubit must all be in |0> on entry too, and a dirty equality[0] makes the target silently wrong on a large fraction of the operand patterns (6 of 16 at two bits) before the ladder restores the flag. The comparator's published record is semi-verified: its venue is not indexed by Crossref, DBLP or INSPIRE, so its volume and page numbers are reported by citing works rather than index-confirmed. It makes no advantage, performance, or hardware claim.

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
- **Known boundary:** Synthesis enumerates all 2**n inputs of the truth table classically, so it carries no advantage of its own at any scale beyond demonstration and its cost is exponential in the register width. Only a truth table is accepted: there is no boolean-expression parser and no other predicate form. A phase oracle is capped at three qubits, because a multi-controlled Z above that needs ladder ancillas a standalone circuit does not have; the in-place append form takes them from the caller. The bit oracle's output is XORed rather than assigned, and it restores every qubit it allocates. No performance, convergence, or hardware claim is made.

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
- **Known boundary:** The improvement is in query complexity, against an oracle this unit synthesizes from a truth table at a classical cost of 2**n. No end-to-end advantage follows at any scale beyond demonstration, and no qRAM, block encoding or amplitude encoding is assumed. The search is bounded at three evaluation qubits, because a phase oracle above that needs ladder ancillas a register of exactly that width does not have. The register width makes the classical truth-table enumeration exponential, which is the honest limit of the unit. It makes no performance, convergence, or hardware claim.

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
- **Known boundary:** The advantage premise is Grover's oracle model, and it is not met: the oracle is not free here. The search's oracle is synthesized from the predicate's truth table at O(2**n) cost, so no end-to-end advantage follows at this scale, and the distance table the predicate compares is computed classically, one point at a time, before any circuit is built -- the register is capped at three qubits, which is also what keeps the distances out of it. Nothing here reads a qRAM or runs an adiabatic evolution, so no conclusion that rests on either applies to this unit. Demonstration scale: the search is bounded at three evaluation qubits, so at most eight centroids, and the assignment is sampled rather than read out, which is why a small sample can stop a point's search short. It makes no performance, convergence, or hardware claim.

### Quantum kernel estimation and kernel ridge classification

Estimate a kernel matrix by swap test over an angle-encoded feature map, and fit a classical kernel ridge classifier on the estimated entries.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.quantum_kernel`
- **Runtime modes:** `local_statevector`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** The advantage premise is the data-access model, and this unit does not meet it. The kernel-matrix circuit's cost statement assumes the two feature states are available, reached through a qRAM or an amplitude-encoding unitary whose cost the estimate does not count; here each feature state is built gate by gate from the classical feature vector on every run, so that cost is paid rather than assumed away and no end-to-end advantage follows. The sampling cost is the paper's own -- O(eps**-2) shots per kernel entry and O(m**2 / eps**2) for an m by m kernel matrix -- and no error bound, confidence interval or repetition scheme is computed or reported anywhere in this unit. Every entry is a sampled estimate, so the classifier's coefficients and its predictions inherit the sample, and an estimate of a near-zero overlap can come back slightly negative because the readout is 1 - 2 * share and is not clamped. The classical hardness of estimating these kernel entries is a conjecture in Havlicek et al. and not a theorem, and the rigorous speed-up results for quantum kernel methods require a fault-tolerant quantum computer (Liu, Arunachalam and Temme 2021). Demonstration scale: the feature map is a three-feature angle encoding of this package's own, and the classifier is classical kernel ridge regression whose only quantum part is the kernel. It makes no performance, convergence, or hardware claim.

### Feature selection as a QUBO

Build the binary objective of a feature-selection instance -- a subset's relevance and pairwise redundancy, scored with a penalty on the subset's size -- and evaluate or map it for a solver.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.feature_selection`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** This unit does not solve, and the repository has no annealer: it builds the objective of one feature-selection instance and evaluates that objective at an assignment the caller supplies. Which subset a solver returns, and at what cost, belongs to the solver the problem is handed to, so any advantage such a solver observes is the solver's and this construction carries none of its own -- the QUBO form and the Ising form are a polynomial classical transformation with no advantage of their own. Both scores are the caller's data: this unit defines no relevance measure and no redundancy measure and puts no interpretation on either. The penalty weight is the caller's too, with no default here and no weight at which the target size starts to bind computed or predicted, so a weight small enough against the scores can leave a subset of another size cheapest. Demonstration scale: the instance is a set of feature scores the caller brings, and the objective is an ordinary quadratic binary form whose quadratic terms are the pairwise scores folded together with the size penalty. It certifies no solver, convergence, performance, or hardware behavior, and it selects no runtime.

### Frequent-item fractions by amplitude estimation

Estimate the fraction of a binary incidence matrix's items whose support meets a threshold, with a support register the circuit fills one controlled increment per transaction-item membership.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.qarm`
- **Runtime modes:** `local_statevector`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** The advantage premise is coherent entry-wise database access, and this unit does not meet it. The cited paper's speed-up is measured in calls to an oracle that returns one database entry per call, and it reaches the candidate itemset superpositions it prepares through a qRAM; neither is exercised here: the transactions are iterated over classically, one controlled increment of the support register per transaction-item membership, emitted from the incidence matrix as Python reads it, so the access cost is paid explicitly rather than assumed away and no end-to-end advantage follows. The improvement the paper claims is quadratic in the number of database queries and is stated conditionally, for the case M_f^(k) << M_c^(k); it is not exponential, and that wording appears only in an earlier arXiv listing of the same work. The support register must be wide enough to hold the largest support any database of that transaction count could produce: the increment is a permutation of the register's own values, so a narrower register wraps a support into another value and the readout can come back wrong with nothing raised, and a width that cannot hold every support is therefore refused rather than left to wrap. The item register is addressed by one qubit per item-index bit, so the item count is a power of two. Demonstration scale: at most eight items and seven transactions, and the marking operator enumerates the support values at or above the threshold. It makes no performance, convergence, or hardware claim.

### Singular values by phase estimation over the Hermitian embedding

Estimate a matrix's singular values from the phase of the Hermitian matrix that carries it as its off-diagonal block, with a private block encoding of that embedding under a stated subnormalisation.

- **Maturity:** Experimental
- **Public API:** `flagquantum.algorithms.svd`
- **Runtime modes:** `local_statevector`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/ALGORITHMS.md)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** The advantage premise is the input model, and this unit does not meet it. The cited algorithm's cost is counted in queries to a structure that returns the matrix's entries, and against that count the state the estimation is applied to is assumed to be preparable; neither is present here. The matrix is held as an ordinary tensor, its embedding is formed and exponentiated as a dense matrix, and the input state is built from the singular vectors a classical torch.linalg.svd returns -- the very decomposition the readout estimates -- so the access and the preparation are paid explicitly rather than assumed away and no end-to-end advantage follows. Dequantization is recorded rather than glossed over: Tang's classical algorithm for the recommendation problem removes the exponential speed-up and is only polynomially slower, its bound containing eps**-12, which the author calls a large slowdown in some exponents; it is not a classical algorithm that matches the quantum runtime. Arrazola et al. record the practical conditions the dequantized algorithms need, and Gharibian-Le Gall dequantize the quantum singular value transformation for sparse matrices at constant precision; their hardness result is for a different task, estimating a local Hamiltonian's ground-state energy at inverse-polynomial precision given a state close to the ground state. The block encoding is not free to read: a readout that post-selects the ancilla succeeds with probability ||(A/alpha)|psi>||**2 on a normalised input, whose greatest value over inputs is (||A||/alpha)**2, and where alpha is much larger than ||A|| that probability is exponentially small. The readout is the counting register's mode, and the unit does not claim that the mode's value is the largest singular value: the readout is a grid value at the register's own resolution, and within() is the whole of the accuracy contract. The counting width is the caller's and is at least one; at one qubit the register can only return alpha itself, and a run whose mode falls in the register's lower half is refused rather than returning a value above alpha -- an outcome of the sample rather than a precondition on the arguments. The input state is prepared from the singular vectors, so the classical work includes the very decomposition the unit estimates. Demonstration scale: at most four rows and four columns, with the embedding, its exponential and the state preparation all classical. It makes no performance, convergence, or hardware claim.

### Zero-noise extrapolation over scaled noise models

Measure an observable at several error strengths by scaling the one parameter each noise channel declares, continue the curve to zero noise by polynomial least squares or Richardson extrapolation, and report the residual and the variance amplification that qualify the estimate.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.algorithms.error_mitigation`, `flagquantum.algorithms.run_zne`, `flagquantum.algorithms.scale_noise_model`
- **Runtime modes:** `density_matrix`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../examples/algorithms/error_mitigation.py)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** Zero-noise extrapolation over exactly simulated state expectations. Noise is scaled by multiplying the single error-probability parameter a channel declares, so bit_flip, phase_flip, depolarizing, two_qubit_depolarizing, amplitude_damping and phase_damping are scalable, while coherent_overrotation, reset_error and thermal_relaxation are refused by name: multiplying an angle or two independent reset probabilities does not scale the noise, and thermal_relaxation's factory does not take its duration as one leading parameter. A caller who needs another scaling supplies a callable and owns the claim that the scaled model differs from the original only in noise strength, and the result records which of the two claims it rests on. The estimated quantity is the exact expectation Tr(O rho) of each scaled model, so no shot is consumed, no confidence interval is reported, and the estimate carries no measured uncertainty. variance_amplification is the factor by which the fitted weights would amplify the variance of a shot-based estimate of the same points; it is a property of the scale grid and the degree alone, and it is not a measured variance. A model that declares a readout rule is refused rather than measured without it, because classical readout confusion is applied after measurement and is not part of rho: extrapolating a curve that omits it would return a state-preparation estimate under the name of a measured one, and readout-error mitigation is absent. Probabilistic error cancellation, Clifford data regression, circuit folding and gate-folding scale factors are absent. The method's one assumption is that the measured curve is a polynomial of degree at most the fit order in the scale factor; nothing checks it and it is not checkable from the measurements alone. The diagnostic that exposes a violated assumption is the largest absolute residual the fit left. At one cx gate with depolarizing noise at 0.05, the four-point curve at scale factors 1, 3, 5 and 7 leaves 1.8e-2 at degree one and 4.2e-9 at degree two, while a coherent over-rotation over the same scale factors leaves 8.9e-2 at degree one and 2.9e-2 at degree two and its estimate stays 1.1e-1 away from the noiseless value, so a non-polynomial family is visible in the residual rather than silently extrapolated. A square fit's residual is zero by construction and is therefore not evidence, which the record enforces by reporting max_residual as absent when no degree of freedom is left rather than as an arithmetic zero. The observable must be a single number, so the circuit's batch size is one, and the channel parameters keep the runtime's precision, so against the analytic value 1.0 a single-precision run recovers it to 1.3e-7 where a double-precision run reaches 3.0e-9; the exactly simulated noiseless read that a comparison is usually taken against is itself 6.0e-8 away from that analytic value at the default precision, so a distance measured against it is dominated by that offset rather than by the extrapolation's error, and every distance recorded here names the precision and the reference it is measured from. Demonstration scale: two-qubit circuits on the CPU density-matrix path, with at most seven scale factors and a fitted degree in the low single digits. No performance, scaling, hardware or fault-tolerance claim follows from it.

### Simultaneous perturbation stochastic approximation

Minimize a scalar objective with no gradient, from two objective evaluations per step whatever the parameter count, on a recursion whose direction is one randomly perturbed finite difference.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.algorithms.spsa`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../examples/algorithms/spsa_optimizer.py)
- **Documentation:** [guide](../../docs/guides/ALGORITHMS.md)
- **Known boundary:** The estimate this optimizer builds is biased for every finite perturbation and is not a gradient: its expectation reaches the objective's gradient only as the perturbation shrinks, so a single estimate is not a descent direction and the update it produces is an approximation of a gradient step. An objective that has an exact gradient is served more cheaply and more accurately by autograd or parameter shift, and this unit says so in its own module docstring rather than presenting itself as a preferred optimizer. What it supports is a cost claim and not an accuracy claim: one step spends two objective evaluations whether the objective has one parameter or 128, measured at parameter counts 1, 2, 8, 32 and 128, where a parameter-shift gradient spends 2*n. The optimizer executes nothing itself -- it calls a callable the caller supplies and never inspects it, so no runtime mode is selected, no circuit is built, no observable is measured, and the caller's objective owns whatever execution it performs. The perturbation is drawn from a torch.Generator the caller passes, which makes a seeded run replay on the parameters' own device; without one an internal generator is created for the parameter device and the draw is then not reproducible from the call site. Parameter names are domain names rather than the published single letters: parameter_gain is a, perturbation is c, stability is A, parameter_gain_exponent is alpha and perturbation_exponent is gamma, with Spall's defaults 0.602 and 0.101. Averaging the estimates, a per-coordinate or blockwise perturbation scale, constraint handling, and resuming an optimizer from serialized state are all absent, and each is a second algorithm with its own convergence conditions rather than a knob on this one. No convergence rate, iteration bound or confidence interval is computed or reported: the numbers a run produces are a trajectory of one objective and the spread across seeds is the estimator's variance rather than a tolerance. It makes no performance, convergence, scalability or hardware claim, and it adds no root-level fq name.

### Clifford circuit sampling by Pauli stabilizer tracking

Sample computational-basis outcomes from a Clifford circuit whose qubit count puts a dense amplitude store out of reach.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.simulation.stabilizer.require_clifford_program`, `flagquantum.simulation.stabilizer.sample_stabilizer`, `flagquantum.simulation.stabilizer.sample_noisy_measurements`
- **Runtime modes:** `stabilizer`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../examples/stabilizer_sampling.py)
- **Documentation:** [guide](../../flagquantum/simulation/stabilizer/README.md)
- **Known boundary:** Requires the optional `stim` distribution (Apache-2.0), installed with `pip install 'flagquantum[stim]'`; without it the entry point raises StabilizerDependencyError naming that command rather than falling back to an amplitude path. Accepted gates are exactly the thirteen Clifford opcodes in flagquantum.core.operator_schema -- i, x, y, z, h, s, sdg, sx, sxdg, cx, cy, cz, swap -- and every other opcode is refused with a CapabilityError naming it and the accepted set. ccx and cswap are refused because neither normalizes the Pauli group, so they are not Clifford gates despite being permutations; t, tdg, the rotation and controlled-rotation family, phase, u1, u2 and u3 are refused for the same reason. Circuits passed to sample_stabilizer must be noiseless: the four noise channels are refused rather than sampled as unitaries, so no readout error, depolarizing, damping or trajectory behavior is available. The second entry point, sample_noisy_measurements, is the narrow exception, and it is not a widening of that refusal: it executes a bit-flip channel at an explicit position a caller has already chosen, so the frame tracking is the caller's placement and not an inference the engine makes, and a channel whose Kraus operators are not the two-element bit-flip pair is still refused by name. It reads no lowered measurement node, returns only the sampled bits with no detection events or observable flips, and is not reachable from the planner or the executor, so `mode='stabilizer'` keeps refusing a noisy program. Only computational-basis measurement sampling is supported; expectation values, detector error models, decoding, error-corrected logical operations and hardware-backed sampling are absent. Sampling is not differentiable and returns no gradients. The execution route is explicit: `fq.ExecutionOptions(mode='stabilizer')` reaches the engine through `fq.run`, the planner publishes a plan whose `state_mode` is `stabilizer` and whose `state_bytes` is the packed tableau rather than an amplitude count, and the result reports `execution_path='local_stabilizer'`. Automatic selection never chooses this mode, so no existing plan changes meaning. Planning refuses a non-Clifford gate, a noise channel, a batch, a non-CPU device, a gradient request, a non-sampling target, a sharded world size and a program without shots; sampling refuses a lowered request it cannot reconcile with its own arguments and postselection, and a memory limit below the tableau. None of those refusals is a fallback, and none of them is a scalability claim. Runs are CPU-only single-device work with no GPU or distributed sharding. A seed reproduces a run on one engine version and one machine instruction set; it does not pin a bit pattern across machines or engine versions. The capacity statement is representational -- memory grows with the qubit count squared rather than exponentially -- and no throughput, latency or scaling claim is made.


## Distributed execution

### Sharded statevector training

Partition one logical statevector workload across ranks while preserving differentiable training semantics.

- **Maturity:** Production supported
- **Public API:** `fq.plan`, `flagquantum.experimental.distributed.train_distributed_statevector`
- **Runtime modes:** `distributed_statevector`
- **Hardware:** `cpu`, `multi_gpu`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../examples/distributed_statevector_topologies/run.sh)
- **Documentation:** [guide](../../examples/distributed_statevector_topologies/README.md)
- **Known boundary:** Target hardware is one host: the required two-GPU gate covers forward, backward, and training on two local devices with NCCL. A run spanning more than one host is the separate two-node capability, whose evidence does not extend back to arbitrary host counts.

### Two-node statevector training

Shard one logical statevector workload across two hosts and keep gradient, optimizer, and checkpoint semantics that a single host would produce.

- **Maturity:** Release certified
- **Public API:** `fq.plan`, `flagquantum.experimental.distributed.train_distributed_statevector`
- **Runtime modes:** `distributed_statevector`
- **Hardware:** `multi_node`, `nvidia_a800_sxm4_80gb`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../docs/guides/MULTINODE_RUNBOOK.md)
- **Documentation:** [guide](../../docs/development/TESTING.md)
- **Known boundary:** Certified scope is the recorded pair: two A800 hosts, up to four devices per host, and a matched statevector training workload of 22, 24 and 26 wires at depths 8, 12 and 16 with cross-shard gate fractions up to one half. The release gate evaluates the payloads promoted into benchmarks/results/scalability alongside the declared single-device baseline, and the checked-in result audit verifies their signatures and claim evidence; each promoted payload reports release_gate_allowed and scalability_claim_allowed true with an empty error list, and the matched pair leg reports a 5.73x speedup whose 0.95 confidence interval lies wholly above one. Inter-node shard exchange, the exact adjoint gradient, an owner-sharded optimizer step, a checkpoint resumed by a restarted run, and a full-state export all executed on that pair and are recorded by the hardware artifact. The whole amplitude vector is exported as the product of a leg of its own, reassembled by the runtime's own gather, so the readout is a materialization the workload needs rather than a gather taken to check an answer; that observation closed the validation-only full-state-gather blocker. The staging audit profiles the measured region and finds no explicit host transfer in it, which closed the host-staging blocker by measurement rather than by relabelling the region. What the certificate does not cover: the pair is the widest topology any measurement here reaches, and no third host was available, so two_node_pair_only_no_wider_topology stands as the pair's own scope and no measurement on this hardware can retract it; the hardware artifact's own latency leg is recorded at the narrowest rung of the workload this certificate freezes -- twenty-two wires over eight layers, one hundred and seventy-six bound parameters, read from the manifest the artifact names -- so the toy-circuit blocker that once applied to that leg is retired by that observation rather than by withdrawing the word; and GPU clocks, power state, temperature, host memory and fabric congestion are unrecorded, so the numbers describe this pair at the recorded revision and nothing wider. The pair's release world sizes are two, four and eight, all single-host except the matched pair leg, so behaviour as further hosts are added is not established.

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
- **Hardware:** `cpu`, `single_gpu`, `multi_gpu`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py)
- **Documentation:** [guide](../../examples/distributed_mps/README.md)
- **Known boundary:** Single-node and multi-GPU execution plus matched checkpoint/restart have recorded evidence. A run spanning more than one host is the separate two-node capability, whose evidence does not extend back to arbitrary host counts. The only public capacity measurement is emitted from the validated claim below; it is one exact-workload result, not general scalability or release evidence. Boundary instructions still execute serially by owner, and layer-parallel contraction/SVD, capacity multi-step soak, a sealed fault matrix, repeated evidence, and the release payload remain incomplete.

### Two-node MPS training

Shard one logical matrix product state by site across two hosts and keep gradient, optimizer, and checkpoint semantics that a single host would produce.

- **Maturity:** Release certified
- **Public API:** `fq.plan`, `flagquantum.experimental.distributed.train_distributed_mps`
- **Runtime modes:** `distributed_mps`
- **Hardware:** `multi_node`, `nvidia_a800_sxm4_80gb`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../docs/guides/MULTINODE_RUNBOOK.md)
- **Documentation:** [guide](../../docs/development/TESTING.md)
- **Known boundary:** Certified scope is the frozen release topology: two A800 hosts of eight devices each, sixteen ranks over the inter-node RoCE fabric with the NCCL backend, and one device left free on each host. The release gate evaluates the payloads promoted into benchmarks/results/scalability alongside the declared single-device baseline, the checked-in result audit verifies their signatures and claim evidence, and each promoted payload reports release_gate_allowed and scalability_claim_allowed true with an empty error list. The capacity leg runs the frozen 131072-site bond-768 workload for one sharded training step across those sixteen ranks, and it fails closed unless the backward-execution contract it asserts actually executed: the boundary adjoint exchange, the boundary gradient routes, the owner-sharded parameter, gradient and optimizer-update ownership, the per-rank backward memory plan and the per-rank backward communication plan are all reported production-measured or production-executed for that step, and the memory plan carries the per-rank byte vectors the readiness gate reads rather than a peak alone. The speed leg times the frozen matched-speed ladder at one device and at sixteen over the same pair, and the promoted payload reports a speedup of 1.1231339460338892 whose 0.95 confidence interval [1.099641001349927, 1.15942039054475] lies wholly above one. That ratio is measured against the workload's own shardable fraction and not against the world size: the frozen ladder's measured serial fraction is 0.8146523972125517, so sixteen ranks can at best return 1.2103070390553239 times one device, and the measured 1.1231339460338892 is 0.9279743980589623 of that ceiling. The efficiency against the world size is 0.07019587162711807 and is disclosed rather than claimed: the workload is dominated by work no rank can share, so what is certified is that sharding a workload this serial still gains about twelve per cent with a significant interval, not that the ratio approaches the rank count. The capacity premise is established by two signed measured_production_run envelopes at this contract's own digests: the same frozen workload exhausting one device with a recorded CUDA out-of-memory reason of 20.00 MiB refused against a 79.25 GiB device at a 75657923584-byte peak, and the same workload completing on sixteen ranks at a 77745422336-byte peak. What the certificate does not cover: the pair is the widest topology any measurement here reaches and no third host was available, so two_node_pair_only_no_wider_topology stands as the pair's own scope, and because the frozen release world of sixteen devices is exactly this pair's width, behaviour as further hosts are added is not established. The premise's two halves are two signed runs rather than one, and the single-device baseline's single_gpu_expected_oom is the release contract's expectation rather than a second measurement of the same run. The profiled measurement region of the layer-halo path still issues explicit host transfers that keep a control-plane readback blocker of their own -- fourteen load-balancing cost probes, five point-to-point control descriptors read back to size a receive buffer, four truncation record exchanges and five end-of-forward reporting reads, with no rank payload staged among them -- and this certification does not retire that blocker, because the release legs measure the training step rather than profile that region. The layer halo crosses the host boundary only on the compiled site-kernel path, which the checkpointed training legs deliberately do not take. GPU clocks, power state, temperature, host CPU and memory, fabric congestion and the driver version are unrecorded, so the numbers describe this pair at the recorded revision and nothing wider. Historical provenance this certificate supersedes but does not delete: the earlier eight-rank attempt at the same ladder is retained under benchmarks/results/smoke/release_candidates/mps_matched_speed_4x2 as a measurement the release contract refused, it sets no claim flag, and its payload still carries the plan shapes the strict audit rejected -- an absent optimizer_update_semantics, a backward memory plan without the per-rank vectors, and a backward communication plan without the boundary edge record shape -- so it must be read as superseded provenance rather than as the certified evidence. The former production_supported scope is likewise retained as measurement rather than erased: on the recorded two-A800 pair with one device per host, one six-wire complex128 circuit and a bond limit that truncates nothing, site-boundary exchange, an exact reverse gradient with the layer halo prefetched over the inter-node transport, an owner-sharded optimizer step and a checkpoint resumed by a restarted run all executed, the route was observed on the RoCE fabric, the cut width was swept rather than declared across five placements at widths two and four, and the whole MPS was exported as the product of a leg of its own through the runtime's own gather. The four-devices-per-host forward latency of one six-wire circuit at the ladder's 8192-site acceptance rung is recorded in benchmarks/results/local/multinode_two_node_performance_20261001.json, whose own blockers and false claim flags say it is a latency of one circuit and not a scaling, capacity or release claim.

### Two-node tensor-network training

Slice one logical tensor-network contraction across two hosts and keep gradient, optimizer, and checkpoint semantics that a single host would produce.

- **Maturity:** Production supported
- **Public API:** `flagquantum.simulation.tensor_network.run_tensor_network`, `flagquantum.experimental.distributed.train_distributed_tensor_network`
- **Runtime modes:** `distributed_tensor_network`
- **Hardware:** `multi_node`, `nvidia_a800_sxm4_80gb`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../docs/guides/MULTINODE_RUNBOOK.md)
- **Documentation:** [guide](../../docs/development/TENSOR_NETWORK_DISTRIBUTED_CONTRACT.md)
- **Known boundary:** Scope is the recorded pair: two A800 hosts with one device per host, one fourteen-wire complex128 circuit bound to the frozen ladder's own narrowest matched-speed rung -- eighty-four parameters over six layers -- and a cut of exactly two labels declared by the workload. Slice-boundary reduction, the exact slice-gradient sum, an owner-sharded optimizer step, and a checkpoint generation resumed by a restarted run all executed on that pair, and the run reports scalability_claim_allowed and release_gate_allowed false with one blocker attached. The route was observed on the RoCE fabric and five synchronized amplitude samples were recorded after two warmups, so the fabric and a repeated measurement are evidence rather than assumptions. The cut width was swept across both labels of the declared cut, which retired the blockers that named an unswept cut and a fixed slice count. The cut is still declared rather than chosen by the automatic slicer, which selects by peak memory and would here pick a label carried only by state-copy nodes whose partial is zero on every rank but one; slicing such a label would report sharded execution while one rank did the arithmetic. Congestion and capacity were not measured, the profiled measurement region issues no explicit host transfer, the whole state the workload produces is exported as a leg of its own rather than gathered to check an answer, and topologies wider than one device per host run on the lane but are not recorded, so this evidence does not establish behaviour as the cut widens or as nodes are added. This capability does not reach release_certified, and the reason is measured rather than inferred. Its release contract freezes a matched-speed acceptance configuration -- 180 trainable angles over eighteen wires and ten layers at six sliced labels -- and that configuration was measured on one A800 and exhausts it: 66.9 GiB peak and a further 16 GiB refused in 50.7 s, while the two narrower rungs of the same frozen ladder complete at 0.79 GiB and 8.53 GiB. The pair does not rescue it, because the reverse-mode peak is not slice-owned: the pair fails the rung with the same 58.9 GiB resident and the same 16 GiB refused. No matched-speed baseline or pair leg can be sealed from the frozen ladder while its reverse tape is unbounded, which is what 'missing_statistically_significant_speedup_artifact' reported for as long as every leg was run that way. Bounding the tape changes the peak without changing the arithmetic: one baseline leg at world one and one sharded leg at world two, both under a 4294967296-byte reverse-tape bound with the shape, slicing, parameters, steps and optimizer untouched, complete all three frozen rungs and seal as benchmarks/results/smoke/release_candidates/tensor_network_matched_speed_pair/tensor_network_matched_speed_pair.json, where the acceptance rung's interval [1.9821866944895041, 1.98656634565854] is wholly above one and the other two rungs report intervals that exclude one as well. The sealed candidate carries no claim flag, so the gate reads it as production evidence rather than as a release payload: with the four claim keys added it reports 'capacity_premise_not_established' alone, and without them it reports that blocker beside the four a claim-free payload cannot satisfy. The claim was measured rather than asserted -- a probe envelope sealed with --release-payload from the same two legs at the same revision passes the strict audit with no errors -- so a reader can reproduce the comparing verdict from the committed candidate by adding the claim keys the candidate deliberately omits, and tests/benchmark_contract/test_tensor_network_release_gate.py reads the committed artifact to assert both verdicts: that the claim-free candidate reports the four blockers a claim would answer beside the premise, and that the same evidence carrying those keys reports the premise alone while the strict audit accepts it with no errors. Those tests also rebuild the artifact's unsigned envelope and require the committed content_sha256 to be reproduced, so the file's own digest is what makes the comparison an observation rather than a transcription. The bound is a measurement condition the frozen speed_workload does not name: the payload records it as checkpoint_budget_bytes and its provenance command records it, but the release gate does not read it, so the candidate is evidence about this pair under a recorded bound rather than a certification of an unbounded ladder. Freezing that bound now would describe a protocol chosen to fit the hardware it was measured on, so the manifest is left naming no budget and the condition is disclosed here instead. The contract's capacity premise is unestablished for the same structural reason -- no measured shape both exhausts one device and completes sharded at one slicing -- so 'capacity_premise_not_established' cannot be cleared either. Both findings are recorded in the manifest rather than corrected by re-freezing the ladder after the measurement, which would let the freeze describe a protocol chosen to fit the hardware it was measured on. The host-staging blocker was retired by measurement rather than by relabelling: the audited amplitude region resolves its contraction plan on every rank instead of receiving it through an object collective, and it builds its basis projectors by indexing a device-resident identity instead of lifting the target table, so the region the probe profiles now reports no explicit host transfer at all -- twenty-three host-to-device copies at the revision this note previously described, then three, then none, each step read from the probe's own profiler sweep. The validation-only-gather blocker is retracted by measurement rather than by declaration: the probe runs an export leg of its own that materializes the whole distributed state as the product the workload needs, it refuses a state it did not materialize, and it asserts the placement that leg ran under, so the sliced full state is no longer gathered only to check an answer. Two later measurements widen the capacity finding rather than narrow it. First, the frozen capacity workload does not contract at its own default slicing at the revision the manifest declares: the executor's default slice-label selection picks the lowest-numbered contracted labels, which for the frozen 4x6x2 grid are its deepest, and the resulting operands exceed the twenty-five-dimension ceiling shared by torch.einsum and PyTorch's copy machinery, so the producer's capacity-failure role raises 'tensor has too many (>25) dims' before it measures any allocation. Spreading the label argument across the network instead completes the same rung in 4099927552 bytes, so the rung the ladder records as completing at 51191433216 bytes was measured with an explicit label argument the manifest does not name, and the ladder is not reproducible from the freeze. Second, every rung in the ladder was measured with an unbounded rematerialization tape, so the rungs that exhaust a device record the budget they ran under rather than the shape: bounding the reverse tape at 4294967296 bytes and holding shape, slicing, parameters and steps fixed turns the two-sliced-label 4x6x2 rung from 80930380288 bytes out-of-memory into a completion at 17160570368 bytes, and turns both 4x6x3 out-of-memory rungs into completions at 30072620032 and 27926225920 bytes. The budget is not monotone -- the four-sliced-label 4x6x2 rung completes unbounded and raises once bounded -- so neither the label choice nor the tape budget can be left implicit by a manifest that claims to have frozen a capacity premise. Both findings are recorded in the manifest's freeze_note and premise_established_note and in the runbook rather than folded into measured_ladder, which stays a record of what was run.


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

Predict calibration-conditioned measurement distributions, prospectively validate fixed and held-out regional suites, and align comparable evidence over time.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.twin`
- **Runtime modes:** `offline_density_prediction`, `offline_evidence_assessment`, `explicit_provider_submission`
- **Hardware:** `cpu`, `quafu_development_evidence`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../examples/remote/quafu_twin_region_holdout.py)
- **Documentation:** [guide](../../docs/guides/QPU_DIGITAL_TWIN.md)
- **Known boundary:** Frozen mapped models, validation series, calibration and validation histories, prospective candidate comparisons, topology- and depth-qualified evidence, connected structural cell composition, fail-closed regional model composition, repeated exact-circuit validation, fixed regional suite validation, predeclared reference/holdout comparison with simultaneous confidence, and longitudinal fixed-suite regional histories are available. Agreement is total-variation agreement for classical measurement distributions, not quantum-state fidelity. Region-model composition never infers cross-cell correlated noise or combines local bounds into region accuracy. Holdout evidence applies only to the prospectively declared circuits and does not establish arbitrary-circuit accuracy or training generalization. Regional suite histories compare only the same ordered fixed suite, mapping, and full topology; they do not define a trust window. Evidence remains specific to declared circuits, operations, mappings, physical couplers, depth, calibration snapshots, and confidence bounds. Automatic calibration collection, scheduling, arbitrary-circuit generalization, region-wide statistical inference, trust policy, model promotion, global publication, and release-certified provider support remain outside the framework capability.

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

### PennyLane Lightning execution bridge

Execute a FlagQuantum circuit explicitly on local PennyLane lightning.qubit while preserving FlagQuantum-owned result and evidence contracts.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem.pennylane.run`
- **Runtime modes:** `external_local_statevector`, `external_local_shots`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/PENNYLANE_LIGHTNING_EXECUTION.md)
- **Documentation:** [guide](../../docs/guides/PENNYLANE_LIGHTNING_EXECUTION.md)
- **Known boundary:** The flagquantum.ecosystem.pennylane bridge executes one fully bound, single-batch FlagQuantum circuit on local CPU lightning.qubit and returns an owned ExecutionResult. It supports exact complex64 and complex128 statevectors and computational-basis samples or counts with explicit qubit order and seed. Explicit device wires preserve idle FlagQuantum qubit extent. It does not support gradients, QNodes, noise models, dynamic circuits, automatic routing, GPU, alternative PennyLane devices, or fallback. Native fq.run remains unchanged.

### Cirq Simulator execution bridge

Execute a FlagQuantum circuit explicitly on local Cirq Simulator while preserving FlagQuantum-owned result and evidence contracts.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem.cirq.run`
- **Runtime modes:** `external_local_statevector`, `external_local_shots`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/CIRQ_SIMULATOR_EXECUTION.md)
- **Documentation:** [guide](../../docs/guides/CIRQ_SIMULATOR_EXECUTION.md)
- **Known boundary:** The flagquantum.ecosystem.cirq bridge executes one fully bound, single-batch FlagQuantum circuit on local CPU Cirq Simulator and returns an owned ExecutionResult. It supports exact complex64 and complex128 statevectors and computational-basis samples or counts with explicit qubit order and seed. Explicit qubit order preserves idle FlagQuantum qubit extent. It does not support gradients, noise models, dynamic circuits, automatic routing, device selection, or fallback. Native fq.run remains unchanged.

### Cirq circuit interoperability

Translate supported Cirq circuits to versioned FlagQuantum IR and back through an isolated, loss-aware control-plane adapter.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem.cirq.from_cirq`, `flagquantum.ecosystem.cirq.to_cirq`, `flagquantum.ecosystem.cirq.import_cirq`, `flagquantum.ecosystem.cirq.export_cirq`
- **Runtime modes:** `control_plane_conversion`
- **Hardware:** `cpu_control_plane`
- **Gradient support:** `symbolic_parameters_only`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/reference/API.md)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Certified against Cirq 1.6.1 and 1.7.0 on Python 3.11 or newer for static circuit conversion with complex128 numerical semantics, bidirectional and fail-closed unless allow_lossy is set. Moment structure is flattened and not preserved, and the contiguous-line-qubit mapping rejects non-line and non-contiguous line qubits. Arbitrary qid dimensions, CircuitOperation blocks, classical controls, MatrixGate, operation tags, and global-phase operations are unsupported, as are idle-qubit extents and global phase in representation. Nineteen FlagQuantum opcodes have no Cirq lowering. The parameter subset is FlagQuantum add/multiply/negate after Cirq symbolic simplification with cirq.parameter_symbols discovery and cirq.resolve_parameters binding equivalence; function, power, complex-constant, and division-by-unbound-parameter nodes are rejected before an operation is created. Measurements convert only as terminal MeasurementGate instructions with distinct non-empty keys, and invert masks, confusion maps, and non-terminal measurements are not representable. Execution, gradients, noise, and hardware submission are out of scope; Cirq objects never enter Core, Compiler, Runtime, or Simulation, and the separate local Cirq Simulator bridge is declared as cirq_simulator_execution.

### Amazon Braket circuit interoperability

Translate supported Amazon Braket circuits to versioned FlagQuantum IR and back through an isolated, loss-aware control-plane adapter.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem.braket.from_braket`, `flagquantum.ecosystem.braket.to_braket`, `flagquantum.ecosystem.braket.import_braket`, `flagquantum.ecosystem.braket.export_braket`
- **Runtime modes:** `control_plane_conversion`
- **Hardware:** `cpu_control_plane`
- **Gradient support:** `bound_parameters_only`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/reference/API.md)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Certified against amazon-braket-sdk 1.117.0 and 1.127.1 on Python 3.11 or newer for bidirectional static circuit conversion, fail-closed unless allow_lossy is set. The Braket integer qubit index equals the FlagQuantum qubit, and explicit identity preserves unreferenced FlagQuantum qubits. Only bound finite real scalars are accepted; free parameters are rejected instead of converted, so no symbolic parameter expression survives the boundary. Global phase and measurement instructions are not represented, and measurements belong to the execution plan rather than to static circuit IR. Compiler directives, gate calibrations, noise instructions, pulse gates, verbatim boxes, result types, and qubit-extent preservation are unsupported, and nine FlagQuantum opcodes have no Braket lowering. Core import, runtime execution, cloud submission, and autograd bridging are all prohibited at this boundary, and Braket objects never enter FlagQuantum runtime layers. docs/reference/API.md documents Braket through the adapter contract and the provider execution path only; it has no static-conversion section, so the contract file is the authoritative statement of scope.

### CUDA-Q kernel export

Export supported FlagQuantum IR to a CUDA-Q kernel through an isolated, fail-closed control-plane adapter. Reverse conversion is not offered.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem.cudaq.to_cudaq`, `flagquantum.ecosystem.cudaq.export_cudaq`
- **Runtime modes:** `control_plane_conversion`
- **Hardware:** `cpu_control_plane`
- **Gradient support:** `bound_parameters_only`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/reference/API.md)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Certified against CUDA-Q 0.15.1 and 0.16.0.post1 on Python 3.11 or newer, and only on linux_x86_64 and linux_aarch64. The boundary is export-only: FlagQuantum IR becomes a cudaq.Kernel built with cudaq.make_kernel_dynamic_builder, and reverse conversion is rejected with reverse_conversion_not_supported rather than approximated. CUDA-Q orders wire zero as the least significant bit, so an explicit conformance reordering is required and is enforced by the export contract. Only bound finite real scalars are accepted; symbolic parameters and kernel arguments are rejected, so no parameter expression crosses the boundary. Measurements are rejected from static unitary export. Arbitrary or decorated kernel import, classical control flow, kernel composition, qudit operations, state initialization, noise channels, and remote or hardware execution are unsupported, and twenty-three FlagQuantum opcodes have no CUDA-Q lowering. Core import and autograd bridging are prohibited, and CUDA-Q objects never enter FlagQuantum runtime layers.

### Evidence-based simulator advisor

Rank explicit local simulator choices from exact checked-in evidence or opt-in live calibration without changing runtime routing.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem.simulators.recommend`
- **Runtime modes:** `offline_evidence_advice`, `opt_in_live_cpu_calibration`
- **Hardware:** `recorded_cpu_environment`, `current_local_cpu`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `not_applicable`
- **Start:** [quick example](../../docs/guides/SIMULATOR_ADVISOR.md)
- **Documentation:** [guide](../../docs/guides/SIMULATOR_ADVISOR.md)
- **Known boundary:** The default path is non-executing: circuit-bound advice requires the canonical FlagQuantum CircuitIR hash, workload fields, and recorded environment to match versioned non-release evidence exactly. A profile-only query is not circuit-bound. For an unknown circuit, a caller may explicitly provide a positive calibration budget; this executes native and installed external statevector bridges, verifies outputs against native, filters unavailable, incorrect, undersampled, and unstable engines, applies the native tie margin, and caches the exact result in-process. The budget is soft because an in-flight framework import or backend call is not interrupted. The advisor does not infer performance from qubit or gate count, automatically route, install dependencies, guarantee production performance, or establish scalability. The initial packaged evidence is limited to five exact complex128 hardware-efficient circuits on one ARM64 CPU environment, and no similarity model is claimed from that narrow corpus.

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
- **Known boundary:** Certified against Qiskit 2.0.x and 2.5.x with Aer 0.17.x through an executable operation, wire-order, statevector, classical-bit, arithmetic-parameter-expression, custom-unitary, and round-trip contract. Six fixed-seed differential programs exercise both conversion directions across three to five qubits, mixed one- to three-qubit operations, reordered qubits, and asymmetric custom unitaries. ParameterExpression import supports the FlagQuantum v1 add/multiply/negate arithmetic subset after Qiskit symbolic simplification; functions, powers, and other operations fail closed. One- to three-qubit custom unitary matrices are converted with explicit local basis-order normalization and validated before export. Qiskit control flow is rejected; named or multiple registers require explicit lossy flattening; custom unitary matrices above three qubits are rejected. Conversion does not make Qiskit a runtime dependency or certify any provider hardware.

### Qiskit Aer execution bridge

Execute a FlagQuantum circuit explicitly on local Qiskit Aer while preserving FlagQuantum-owned result and evidence contracts.

- **Maturity:** Experimental
- **Public API:** `flagquantum.ecosystem.qiskit.run`, `flagquantum.ecosystem.extensions.ExecutionBackendExtension`
- **Runtime modes:** `external_local_statevector`, `external_local_shots`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../docs/guides/QISKIT_AER_EXECUTION.md)
- **Documentation:** [guide](../../docs/guides/QISKIT_AER_EXECUTION.md)
- **Known boundary:** The flagquantum.ecosystem.qiskit Aer bridge executes one fully bound, single-batch FlagQuantum circuit on local CPU Aer and returns an owned ExecutionResult. It supports exact statevectors and computational-basis samples or counts with explicit qubit order, seed, and CPU thread controls. It does not support gradients, noise models, dynamic circuits, automatic routing, provider hardware, or fallback. Native fq.run remains unchanged.

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
- **Known boundary:** DynamicCircuit construction is candidate-stable pending API-owner approval; dynamic execution and backend assessment remain experimental. Local dynamic noise is limited to one-qubit bit-flip channels after matching executed gates and independent readout confusion on explicit measurements and final sampling. Other Kraus channels, correlated readout, device-profile timing noise, noisy gradients, and provider-noise execution fail closed. Routing, deployment packaging, dialect export and provider integration are internal workflows rather than public experimental APIs. Provider-neutral conformance passes locally and on Qiskit Aer, but no real IQM QPU task was used.

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
- **Known boundary:** The migrated SDK protocol is approved but not frozen; individual extensions remain experimental until separately qualified. Compiler plugins currently exchange CircuitIR only; pulse and native-binary artifacts are not supported. An execution backend is admitted in-process through the host-side admission module and executes FlagQuantum IR; there is no stable C ABI, no out-of-process backend, no per-execution discovery, and no numerical equivalence certification between an admitted backend and the built-in engine.


## Validated public performance claims

Every value below is read from a hash-bound raw artifact. Missing or changed
evidence makes the source-of-truth check fail closed.

| Claim | Maturity and scope | Artifact-derived result | Recorded environment | Evidence identity |
| --- | --- | --- | --- | --- |
| **Two-node statevector matched-speed release**<br>`multinode-statevector-matched-speed-release-20261003` | `release_certified`<br>One logical sharded statevector training workload -- 26 wires, depth 16, one half of the gates crossing a shard boundary -- run as a matched pair on two A800 hosts with four devices per host, against the same workload on a single device. Speedup is the geometric mean over the matched configurations the manifest freezes, timed for one full forward-and-reverse training step, and the 0.95 confidence interval must exclude 1.0. The released set also carries world 2, world 4 and world 8 single-host legs and a capacity-completion leg; the declared single-device baseline is provenance for those legs and is never itself a release payload, because one device has no ranks to shard across. | **Geometric-mean speedup over the matched configurations:** 5.73e+00<br>**Lower 0.95 confidence bound on the speedup:** 5.71e+00<br>**Scaling efficiency:** 7.16e-01<br>**Fraction of the step spent in inter-node communication:** 9.09e-01 | **Processes in the release workload:** 8<br>**Devices per host:** 4<br>**Hosts:** 2<br>**Collective backend:** nccl<br>**Warmup iterations before sampling:** 5<br>**Measured iterations:** 30<br>**Metadata boundary:** Each sealed payload records the revision it was produced from, the command, the rank-to-device mapping with device UUIDs, the collective backend, the topology fingerprint, the warmup and measured iteration counts, the seeds, the raw log digest and any fallback events, plus world, local world and node counts, per-rank ownership, memory and communication. It does not record GPU clocks, power state, temperature, host CPU or memory, fabric congestion, or the driver version. The recorded fabric is RoCE over mlx5_101 through mlx5_108 on ens22f0 with the NCCL backend, so the numbers describe this pair at this revision and nothing wider. | [raw JSON](../../benchmarks/results/scalability/statevector_matched_speed_pair.json)<br>SHA-256 `17c80b5b655c990c4fe7ff40fa54706c67a2d6b77a9f7a1a161b5b0a57d849a9`<br>code `e339c225bd1b246c63994bb17d015b36f0106908` |
| **Sharded MPS exact-workload capacity**<br>`mps-capacity-131072-chi768-20260806` | `development_evidence`<br>One batch-one, complex64, χ768 MPS training step for the checked-in all-rank and all-boundary workload. This is not arbitrary statevector capacity, fixed-plan strong scaling, or release evidence. | **Sites:** 131,072<br>**Logical MPS state:** 1,236,780,012,864 bytes (1,151.84 GiB)<br>**Maximum elapsed time:** 367.37 s<br>**Maximum peak allocated memory per rank:** 77,745,407,488 bytes (72.41 GiB)<br>**Cumulative discarded weight:** 8.39e-06 | **Ranks:** 16<br>**Reported device memory per rank:** 85,093,777,408 bytes (79.25 GiB)<br>**CUDA allocator policy:** expandable_segments:True<br>**Topology fingerprint:** c74a91e3a224a4dd414cfbfbcb31da7570880d42e6aa4eaccecf4b85427940a5<br>**Metadata boundary:** The artifact records rank count, per-rank device memory, topology fingerprint, and CUDA allocator policy. It does not record the exact GPU model or Python, PyTorch, CUDA, NCCL, driver, host, and operating-system versions, so the claim is restricted to the recorded environment fields. | [raw JSON](../../benchmarks/results/local/mps_capacity_131072q_chi768_16xa800_repeat_complete_20260806.json)<br>SHA-256 `df8c19b74cc173799e3894025fb8e72de72a786d4542a9aa5e67687298f482f5`<br>code `9d56a6ecd78b06f11b9ee6e8aadcbe9644f2c708` (produced on a host whose history this repository does not contain) |
| **Two-node MPS matched-speed release**<br>`multinode-mps-matched-speed-release-20261009` | `release_certified`<br>One logical sharded MPS training workload -- the frozen matched-speed ladder's acceptance rung of 8192 sites at maximum bond 64, thirty-one trainable parameters -- run as a matched pair on two A800 hosts with eight devices per host, against the same workload on a single device. Speedup is the ratio of the two legs' measured per-rung medians over the ladder the manifest freezes, timed for one full forward-and-reverse training step, and the 0.95 confidence interval must exclude 1.0. Efficiency is measured against the workload's own shardable fraction -- the ceiling the same legs' serial fraction implies -- and not against the world size; the linear efficiency is reported beside it and is not the claim. The promoted set also carries the sixteen-rank capacity completion of the frozen 131072-site bond-768 workload; the declared single-device capacity baseline is provenance for that leg and is never itself a release payload, because one device has no ranks to shard across and its own run exhausted the device. | **Matched-pair speedup over the frozen ladder:** 1.12e+00<br>**Lower 0.95 confidence bound on the speedup:** 1.10e+00<br>**Scaling efficiency against the workload's shardable ceiling:** 9.28e-01<br>**Ceiling the measured serial fraction implies at this world size:** 1.21e+00<br>**Fraction of the step spent in boundary communication:** 9.99e-01 | **Processes in the release workload:** 16<br>**Devices per host:** 8<br>**Hosts:** 2<br>**Collective backend:** nccl<br>**Boundary transport scope:** multi_node_production_transport<br>**Maximum bond dimension:** 64<br>**Metadata boundary:** Each sealed payload records the revision it was produced from, the command, the rank-to-device mapping with device UUIDs, the collective backend, the topology fingerprint, the warmup and measured iteration counts, the seeds, the raw log digest and any fallback events, plus world, local world and node counts, per-rank ownership, memory and communication. It does not record GPU clocks, power state, temperature, host CPU or memory, fabric congestion, or the driver version. The recorded fabric is RoCE over mlx5_101 through mlx5_108 on ens22f0 with the NCCL backend, so the numbers describe this pair at this revision and nothing wider. No third host was available, so behaviour as further hosts are added is not established. | [raw JSON](../../benchmarks/results/scalability/mps_matched_speed_2n16g.json)<br>SHA-256 `b4d3c190b0f4aaecd8eff02577668901fafbc7b05696854a4ac9e163eb76f128`<br>code `eea46ed81f8d7b1d4299c0b8fac68e7000c1adf4` |
| **Two-node sliced amplitude latency**<br>`multinode-tensor-network-amplitude-latency-20261001` | `production_supported`<br>One sharded forward pass of the recorded two-node workload, timed between CUDA synchronizations on two A800 hosts with one device per host and a RoCE fabric route read from the NCCL debug log. This is a latency of one fourteen-wire circuit bound to the frozen ladder's own narrowest matched-speed rung -- eighty-four parameters over six layers -- and not a scaling, throughput, or capacity claim; the artifact's own blockers say so and both of its claim flags are false. | **Minimum amplitude latency (s):** 4.51e-02<br>**Median amplitude latency (s):** 4.52e-02<br>**Maximum amplitude latency (s):** 4.53e-02 | **Warmup iterations before sampling:** 2<br>**Synchronized samples:** 5<br>**Observed transport route:** infiniband<br>**Devices per node:** 1<br>**Metadata boundary:** The artifact records the accelerator, node count, world and local world size, dtype, route, NCCL, CUDA runtime, PyTorch and Python versions, and the per-sample latencies. It does not record GPU clocks, power state, temperature, host CPU or memory, fabric congestion, or the driver version, so the numbers describe this pair at the recorded revision and nothing wider. | [raw JSON](../../benchmarks/results/local/multinode_two_node_performance_20261001.json)<br>SHA-256 `82788f14f3fbbcc7db7743393958fca0426c52304a5d564b795c96bef63d048e`<br>code `be04365f7a78c5361635b712184f8abe53b47f51` |
