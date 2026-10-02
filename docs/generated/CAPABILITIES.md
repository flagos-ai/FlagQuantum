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
| Inspect a circuit diagram in a terminal | Circuit drawing | Experimental | [Run example](../../flagquantum/drawer/README.md) |
| Render a figure for a document or notebook | Circuit drawing | Experimental | [Run example](../../flagquantum/drawer/README.md) |
| Check a program's wiring and gate order by eye | Circuit drawing | Experimental | [Run example](../../flagquantum/drawer/README.md) |
| Draw with Circuit.draw or the flagquantum.drawer entry points | Circuit drawing | Experimental | [Run example](../../flagquantum/drawer/README.md) |
| Simulate a small or medium circuit exactly | Local statevector simulation and training | Production supported | [Run example](../../examples/single_machine_quantum_ai/01_vqe_statevector.py) |
| Train a parameterized quantum circuit | Local statevector simulation and training | Production supported | [Run example](../../examples/single_machine_quantum_ai/01_vqe_statevector.py) |
| Run local VQE and quantum machine learning | Local statevector simulation and training | Production supported | [Run example](../../examples/single_machine_quantum_ai/01_vqe_statevector.py) |
| Differentiate a circuit without importing an internal gradient helper | One gradient entry point with a reported method | Production supported | [Run example](../../examples/gradient_methods/README.md) |
| Find out which gradient method a program can use | One gradient entry point with a reported method | Production supported | [Run example](../../examples/gradient_methods/README.md) |
| Train a circuit in statevector, MPS, or tensor-network mode | One gradient entry point with a reported method | Production supported | [Run example](../../examples/gradient_methods/README.md) |
| Train one statevector workload across multiple ranks | Sharded statevector training | Production supported | [Run example](../../examples/distributed_statevector_topologies/run.sh) |
| Plan distributed statevector ownership | Sharded statevector training | Production supported | [Run example](../../examples/distributed_statevector_topologies/run.sh) |
| Inspect communication and sharding semantics | Sharded statevector training | Production supported | [Run example](../../examples/distributed_statevector_topologies/run.sh) |
| Train one statevector workload across two nodes | Two-node statevector training | Production supported | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Resume a multi-node statevector run from its checkpoint | Two-node statevector training | Production supported | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Inspect inter-node statevector communication and rank placement | Two-node statevector training | Production supported | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
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
| Train one MPS workload across two nodes | Two-node MPS training | Production supported | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Resume a multi-node MPS run from its checkpoint | Two-node MPS training | Production supported | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
| Inspect inter-node site-boundary communication and rank placement | Two-node MPS training | Production supported | [Run example](../../docs/guides/MULTINODE_RUNBOOK.md) |
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
| Declare a stabilizer code with its distance, wires, checks and logical observables | Stabilizer-code memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Build the circuit source and detection layouts of a memory experiment from a code | Stabilizer-code memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Trace a detection event back to the check and round that produced it | Stabilizer-code memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Build the detector error model a memory circuit defines under phenomenological noise | Detector error model and stim text interchange | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Build the model of a code whose parity-check and logical matrices are known, without writing a circuit | Detector error model and stim text interchange | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Read a stim detector error model and sample its detection events | Detector error model and stim text interchange | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Emit stim text for an external decoder or matcher | Detector error model and stim text interchange | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Decode a detection-event syndrome with a detector error model | Minimum-weight matching decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Inspect the weighted decoding graph a model defines | Minimum-weight matching decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Cross-check the matcher against an independent implementation over the same graph | Minimum-weight matching decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Compare a matcher's prediction against the observables a model sampled | Minimum-weight matching decoder over a detector error model | Development evidence | [Run example](../../flagquantum/qec/IMPLEMENTATION.md) |
| Sample detection events from a noisy memory experiment | Sampled detection events from a memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Reproduce a sampling run from a seed | Sampled detection events from a memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
| Decode a sampled syndrome with the model built from the same circuit | Sampled detection events from a memory circuit | Development evidence | [Run example](../../flagquantum/qec/README.md) |
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
| Sample a wide Clifford circuit that has no representable amplitude store | Clifford circuit sampling by Pauli stabilizer tracking | Development evidence | [Run example](../../examples/stabilizer_sampling.py) |
| Reproduce a sampling run from a seed | Clifford circuit sampling by Pauli stabilizer tracking | Development evidence | [Run example](../../examples/stabilizer_sampling.py) |
| Measure a chosen subset of wires in a chosen output order | Clifford circuit sampling by Pauli stabilizer tracking | Development evidence | [Run example](../../examples/stabilizer_sampling.py) |

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
- **Known boundary:** In-process transformation of one FlagQuantum IR v1 program on CPU; the compiler never executes a program or certifies hardware behavior. optimize() removes identity gates, merges self-inverse runs, and merges adjacent rotations to a fixed point. compile() runs that same fixed point, optionally routes onto one explicit CouplingMap, optimizes again, and records post-routing optimization. Routing offers four caller-selected strategies: restore_after_each_gate, persistent_layout, sabre, and sabre_layout. All four restore the identity logical order at the program output, so a caller cannot ask for an unrestored final permutation, and the cost-estimate automatic selection covers only restore_after_each_gate and persistent_layout because the two SABRE strategies plan SWAPs instead of estimating them; the topology is always caller-supplied and never discovered. Initial placement is planned separately: plan_trivial_layout places logical wire i on physical wire i, and plan_dense_layout places the program on the most densely connected window of device wires a breadth-first walk from every physical wire finds, which is the only placement that can help on a device with unequal connectivity and the only one of the two that can reach a program the lowest wires cannot hold at all. That walk is a heuristic and not an exact optimum: over 55 device and program-width combinations it matched an exhaustive search of every connected window 45 times and returned a window with one fewer internal connection the other 10 times. Both return a placement as a tuple of physical wires, one per logical wire, which is what route_to_directed_topology takes as initial_layout; final_layout reports the assignment a routed program ended on as a Layout, and a route that allocates workspace on a device wider than the program reports the device width there with its idle slots as None in physical_to_logical. Both placements collapse to the identity on a device no wider than the program. Neither reaches route_to_topology, which refuses an initial_layout because a plain CouplingMap routes only over wires the program owns; a non-identity placement requires a DirectedCouplingMap, whose physical workspace owns the idle slots and is cleaned by the inverse routing SWAPs, and it is passed to route_to_directed_topology or to legalize_circuit_topology. remove_layout_restore drops the appended logical-restore phase from a program that was already routed, halving the SWAP count over a batch of random programs for persistent_layout because that strategy's restore is a full reverse replay of its forward SWAPs and leaving restore_after_each_gate unchanged because its restores are interleaved with the program, and its result is a program for a local simulator or a provider that accepts per-physical-wire results rather than deployable routing evidence: the deployment routing contract rejects it and the deployment entry points treat it as unrouted and route it again. Window selection is greedy over device connectivity and never estimates routing cost, so it can cost more SWAPs than the trivial placement on a program whose interactions prefer a sparser window, and it is not noise- or calibration-aware. Scheduling constructs deterministic logical ASAP layers with explicit wire and classical-data dependencies; it is not target timing or pulse scheduling. There is no directed acyclic graph intermediate, no composable pass manager, no pass analysis or preservation metadata, and no in-repository pass extension point: canonicalization functions are pipeline internals, and an installed external compiler is selected by name through the extension SDK rather than composed as a pass. Pulse-level, calibration-aware, and parameter-aware compilation, dynamic-circuit control flow, and noise-aware optimization beyond lower_noise_model are unsupported. The structured hybrid program slice under flagquantum/compiler/_hybrid is private and absent from public exports. Compilation is single-process; no distributed or multi-device compilation path exists.

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
- **Public API:** `fq.gradient`
- **Runtime modes:** `statevector`, `mps`, `tensor_network`
- **Hardware:** `cpu`, `single_gpu`
- **Gradient support:** `exact`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../examples/gradient_methods/README.md)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Only statevector, MPS, and tensor-network mode can serve the expectation value a differentiable program needs; stabilizer mode samples outcomes and fails closed instead. method='autograd' and method='parameter_shift' are exact; method='finite_difference' and method='spsa' are declared approximations whose result reports exact=False together with the displacement actually used. method='adjoint' is refused, because FlagQuantum has no standalone adjoint entry point: the reversible sweep is reachable only as the backward pass behind PyTorch autograd, and no result reports whether backward used adjoint replay. method='auto' measures the program rather than declaring a route, so it probes for an autograd graph before it resorts to a shift rule or a difference.

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
- **Known boundary:** Validated Markovian Kraus channels, timestamped DeviceNoiseProfile input, ASAP gate/idle thermal lowering, classical readout confusion, exact density execution, and reproducible MPS trajectories with single-rank adaptive stopping are available. Four channel opcodes (bit_flip, phase_flip, depolarizing, amplitude_damping) declare the scalar their factory takes, so they can be written directly into a circuit as well as attached to a gate through a NoiseModel rule; the two routes lower to the same instruction, and a program is planned as noisy because it carries a channel, not because a model was passed. That same reading of the program decides the trajectory representation under mode='auto' and restores it from a saved plan, so an inline channel and an equivalent NoiseModel rule select the same route. A representation that holds amplitudes -- mps, tensor_network, and statevector -- refuses a channel instead of returning the noiseless number, and mode='stabilizer' names its own obstacle. A channel instruction's matrix field is the Kraus tuple, so consumers read it as a sequence rather than as one tensor. Only those four opcodes are reachable by name: the remaining flagquantum.noise callables, including thermal relaxation, readout error, and coherent overrotation, are NoiseModel rules only. Channel parameters are bound real scalars; a trainable channel probability has no gradient path. Lowering refuses a rule naming a wire outside the program width and refuses a model whose rules match no instruction at all, so a misspelled gate name cannot yield a clean result; a model that matches some instructions stays legal. Pulse overlap, crosstalk, leakage, provider calibration adapters, distributed adaptive stopping, batched statevector trajectories, production multi-GPU scheduling, and noisy gradients remain unsupported. Multi-wire MPS channels use an explicitly dense correctness fallback.

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
- **Known boundary:** A synchronous local reference for one fixed three-data-qubit repetition-code profile. It supports bounded deterministic X-error schedules, replaceable per-round trajectory decoding with physical-X or Pauli-frame-X actions, and a two-round temporal rule that rejects an isolated readout excursion. Confirmable data errors require a following round; terminal-round onsets remain unconfirmed. The circuit-location stochastic profile contains independent bit flips after parity-check CNOTs and independent syndrome/final-readout confusion. The middle data wire has two CNOT noise opportunities per round while edge wires have one. Feedback traces separate true and observed bits, actions, and frame evolution. Sweeps report finite-shot observations only, not logical suppression or thresholds. The temporal rule is not maximum-likelihood decoding and repeated readout faults may mimic data errors. Batched decoder feedback, general channels/codes, correlated or timing noise, hard-real-time/provider control, gradients, distributed execution, capacity, performance, and fault-tolerance claims remain unsupported. The feedback records are private subinterfaces and the namespace is not exported from the stable package root.

### Stabilizer-code memory circuit

Declare a stabilizer code record and turn it into a memory-experiment source with the detector and observable layouts that experiment implies, for the repetition code, the rotated surface code and the Steane code.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.build_memory_circuit`, `flagquantum.qec.RepetitionCode`, `flagquantum.qec.RotatedSurfaceCode`, `flagquantum.qec.SteaneCode`, `flagquantum.qec.css_code_matrices`, `flagquantum.qec.CssCodeMatrices`, `flagquantum.qec.CodeCheck`, `flagquantum.qec.Pauli`, `flagquantum.qec.MemoryCircuit`, `flagquantum.qec.DetectorLayout`, `flagquantum.qec.ObservableLayout`
- **Runtime modes:** `local_statevector_dynamic_hybrid`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/README.md)
- **Documentation:** [guide](../../flagquantum/qec/IMPLEMENTATION.md)
- **Known boundary:** A code-independent declaration layer, not a decoder and not an experiment runner. build_memory_circuit emits bounded hybrid source text plus detector and observable layouts; executing that source is the Runtime's dynamic hybrid path and carries that path's own limits. Three code records exist: the repetition code, the rotated surface code and the Steane code, the last of which is the first record carrying both a Z-type and an X-type logical observable. A check states one ancilla and one CNOT direction chosen by the check's type, so a mixed X-and-Z stabilizer is refused; a code needing a second ancilla per stabilizer requires the record to grow first. Both the initial state and the terminal data readout are in the Z basis, so a code whose declared logical observable is not Z-type is refused rather than measured under premises that do not hold for it. Detector counts follow the two check classes, z_checks * (rounds + 1) + x_checks * (rounds - 1), and are validated against the code's checks and the configured rounds, but the validation is membership-based: it does not check that a detector names the same check in every round, that a terminal detector's data wires are the support of the check it belongs to, or that the emitted source is the program the layouts describe, so a hand-built layout can be semantically wrong while passing every check. The rotated surface code's declared logical observable is Z on one data row; which weight-d logical operator a patch uses is a convention, and this record states the one it uses rather than deriving all of them. Agreement with stim's generated rotated-memory circuit is on the declared detector and observable count at distances two through eight and rounds one through four, which fixes the shape and not a mechanism-level correspondence. Execution evidence covers the distance-three patch only, because the dynamic hybrid simulator draws each shot from the full output distribution and a distance-five patch needs 2**49 categories. No decoding, no threshold, no logical-suppression, no fault-tolerance, no real-time and no hardware-feedback claim follows from a declared code, a detector layout, or a syndrome trace.

### Detector error model and stim text interchange

Build the exact Pauli-noise detector error model of a memory circuit or of a code-capacity CSS record, sample it, and exchange it with stim as text in both directions.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.DetectorErrorModel`, `flagquantum.qec.DemError`, `flagquantum.qec.DemMergeRule`, `flagquantum.qec.DemSample`, `flagquantum.qec.css_code_matrices`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/IMPLEMENTATION.md)
- **Documentation:** [guide](../../flagquantum/qec/IMPLEMENTATION.md)
- **Known boundary:** An exact construction for Pauli noise over the reference Clifford gate set, and a text reader for the stim detector error model format. Construction does not sample: a mechanism's signature comes from one forced execution whose two shots must agree on the detector and observable flips they produce, and a non-Pauli channel is refused with a stated reason rather than approximated. The agreement is required of the flip set and not of the raw register: an X-type ancilla is prepared in |+>, so its round-zero outcome is a coin toss, while the steady-state X-type detector it feeds is deterministic because the first round projected both compared rounds into the same eigenstate. Demanding register agreement refused every code with an X-type check, so the rotated surface code could not be modelled at all. Because the signature comes from an execution, construction is bounded by the statevector amplitude ceiling: a distance-2 patch is 7 wires and a distance-3 patch is 17, both of which build in seconds, while distance 4 is 31 wires (2**31 amplitudes) and did not complete in 45 minutes, and distance 5 is 49 wires and fails on the allocator outright. The modelled rotated-surface distance is therefore 3 on the circuit route, and that ceiling is a property of forcing a signature through an execution rather than of the construction contract: reading a matrix support costs one pass over its nonzero entries per round, so a distance-5 or distance-7 patch reaches a model as easily as a distance-2 one. A second construction route reads a code-capacity experiment rather than a memory circuit. DetectorErrorModel.from_code_matrices takes a CssCodeMatrices record -- the four CSS blocks hz, hx, lz and lx -- and derives every mechanism combinatorially, and css_code_matrices lifts a code record into that record, refusing a logical observable that is neither pure X nor pure Z rather than half-reading a CSS code. Its geometry is not the memory circuit's and the difference is pinned rather than glossed: a fault in round r reaches the detector band of round r and of round r + 1, the final round has no band after it, and there is no terminal data readout, so the detector count is num_rounds * num_checks where a memory circuit gains one terminal detector per Z-type check. Both geometries are asserted against their own stim transcription, and the same suite asserts that the two counts differ. The matrix route is compared to stim one readout basis at a time, because lz and lx anti-commute and no one state has both as a deterministic value: the reference experiment is run twice against the model restricted to the matching basis, and each run's observable rows are the ones the Pauli character of that fault family predicts. A fault that is still in the data at the end of the run is still seen by the logical readout, so a physical fault spans one detector band while the model gives it two; that single divergence is pinned as the exact relation between the two mechanism sets rather than as a tolerance. The fault family is the three single-qubit Pauli faults plus a check's syndrome bit flipped at readout, so an X fault flips Z-type checks, a Z fault flips X-type checks and a Y fault flips both, and the X detector band and the lx observable rows exist on the matrix route. The rate record is what remains narrower than upstream's: it states uniform scalars, so an independent px/py/pz/pm split and per-qubit or per-check rates are not expressible. A matrix that is not binary, is not two-dimensional, or is not indexed by the same data qubits as another non-empty block is refused rather than coerced. The matrix route is graphlike only while it stays inside one round; at two rounds the middle data wire of the distance-3 repetition code flips the same check in both bands, which is a four-detector mechanism, and the matching decoder refuses it as a hyperedge. The model is built on the memory circuit without in-circuit feedback, because it describes the noise-to-detection mapping a decoder inverts, so the frozen profile's compiled-feedback path is unmodelled. Detector rates are cross-checked against rates sampled from the circuit simulator, but the cross-check shares the injection helper with construction, so it is not an independent re-derivation of the signatures. An independent check does exist and runs against the optional stim distribution: the memory circuit is transcribed into a stim circuit and stim's own error analysis builds the model, which agrees with this one on shape, mechanism count and detector marginals at distance 2 and 3. That check shares nothing but the circuit's gate sequence and the noise model. The text reader accepts everything stim 1.16.0 wrote in a 240-model developer-time sweep except a repeat block, a # comment, a declaration that skips an index, and a malformed line; str(detector_error_model.flattened()) is a complete route around the repeat refusal. A line's ^ separators are read the way stim means them -- the signature is the symmetric difference of the targets, so a detector named twice cancels and the groups are not retained -- and the decomposition the separators suggest is available under upstream's own flag name as from_stim_text(text, use_decomp_suggestions=True), which returns one mechanism per component at the probability the line states and turns 536 mechanisms into 1042 on a distance-5, two-round rotated surface code. That second reading is a different distribution from the line rather than an equivalent statement of it, and it is measurably so: against stim's own sampler on the same text the default reading's worst marginal misses by 0.0016, inside the 0.004 tolerance, while the expanded reading misses the observable marginals by 0.048, twelve times outside it, which is what makes the tolerance evidence rather than decoration. A component that cancels to nothing is refused by the expanded reading alone, with the default reading named as the route that states the line. The text writer is exact on this side and stim's width is the platform's: stim prints at numeric_limits<long double>::digits10 + 1, so nineteen significant digits where long double is the x86 80-bit extended type (the Linux CI runners, where a reprint returned every swept probability unchanged) and sixteen where it is a double (arm64 macOS, where about a quarter of the same sample came back changed); seventeen digits name a double uniquely, so a round trip through stim must compare against a re-read of the text rather than against the in-memory model exactly where the width is the narrower one, and the drift there is bounded by one part in 10**15 and measured at 5.4e-16 worst case, while tests/qec/test_dem_stim_text_precision.py reads the width off stim's own output and pins the digit count, the direction of the loss and the bound against it. A CodeCheck states one ancilla and one CNOT direction fixed by the check's type, so a code with both check types is expressible while a mixed X-and-Z stabilizer is refused; the detector grammar is therefore z_checks * (rounds + 1) + x_checks * (rounds - 1) rather than one detector per check per round, and a code family needing a second ancilla per stabilizer still requires that record to grow first. dem_sampling reports a DEM-sampled rate and is labelled as one wherever it is reported; no decoder, no threshold, no logical-suppression, no real-time or hardware-feedback, and no performance claim follows from a detector error model or from any rate it reports. DemSample carries tensors and is deliberately unhashable. The merge is a stated operation and not only a construction step: DemMergeRule names its two rules for what they compute, INDEPENDENT_PARITY being the probability that an odd number of a duplicated group fires and CLAMPED_LINEAR_SUM the sum of the group clamped at one, mechanisms_are_unique and require_unique_mechanisms are the predicate and the refusal, and the parity rule is exact rather than approximate because a detector's rate is a product of 1 - 2p factors over the mechanisms touching it, so a group's combined prior is the single p whose factor is that product and regrouping the factors cannot change any detector or observable rate; the sum rule does not preserve them and is offered only for a caller who means it. Construction is additionally covered by tests/qec/test_dem_records.py, test_dem_rates.py, test_dem_signatures.py, test_dem_sampling.py, test_dem_cross_check.py, test_dem_from_memory_circuit.py, test_dem_public_surface.py, test_dem_merge.py, and tests/qec/test_dem_code_matrices.py, whose stim cross-check is tests/qec/test_dem_code_matrices_stim.py.

### Minimum-weight matching decoder over a detector error model

Turn the detector error model of a memory circuit into a weighted graph and decode a syndrome exactly by minimum-weight matching, reporting the mechanisms selected and the logical observables they flip.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.DecodingGraph`, `flagquantum.qec.DecodingGraphEdge`, `flagquantum.qec.MinimumWeightMatchingDecoder`, `flagquantum.qec.MatchingDecodeResult`, `flagquantum.qec.PyMatchingDecoder`
- **Runtime modes:** `not_applicable`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/IMPLEMENTATION.md)
- **Documentation:** [guide](../../flagquantum/qec/IMPLEMENTATION.md)
- **Known boundary:** One decoder, for graphlike detector error models only, on one detector error model at a time, with no threshold, no logical-suppression, no real-time, and no performance claim. A decoding graph is built from the model's mechanisms and nothing else: a mechanism becomes an edge whose weight is log((1 - p) / p), a mechanism flipping one detector becomes an edge to a boundary node one past the last detector, a mechanism flipping three or more detectors is a hyperedge and is refused with a stated reason rather than projected onto a pair, a mechanism of probability zero contributes no edge because its weight would be infinite, and two mechanisms that share a detector pair but disagree on their observable labels stay two edges because merging them would either lose a logical flip or invent a weight neither mechanism has. Two mechanisms that agree on both are the different case and are refused: one fault stated twice would become two parallel edges and the matcher would charge the cheaper of them, log 4 for mechanisms of 0.1 and 0.2 where the fault's own combined parity 0.26 has weight log(0.74 / 0.26), so the matcher would prefer a longer chain of other mechanisms over the mechanism that actually fired. The decoder therefore requires a unique signature before it builds a graph, names both mechanisms and the merge that resolves the duplicate, and leaves the graph's parallel edges intact so the refusal is a decoder decision and not a representation one. Whatever the model states is what the graph holds, so the decoder inherits every scope limit of the detector error model it consumes, including the modelled rotated-surface distance of 3. The matcher is exact and self-implemented: it searches each defective detector for the cheapest chain of mechanisms to every other detector and to the boundary, and then enumerates the ways to pair the defects and to route any number of them to the boundary, which is a minimum-weight perfect matching on the metric closure. The boundary is a sink rather than a waypoint, which is what makes a syndrome's parity irrelevant, and chains may pass through defective detectors because edges two chains share cancel in the symmetric difference. Because the pairing is enumerated, the decoder accepts at most twenty detection events per syndrome by default and refuses a larger syndrome as a capability boundary rather than returning a pairing that only looks cheapest; the budget is a constructor argument because the cost is in that enumeration, so a distance-three patch at ordinary noise decodes while a much larger syndrome is declined. Detectors that no chain of mechanisms connects are refused with a stated reason rather than answered partially. The prediction is the exclusive-or of the selected mechanisms' observable labels, so it is a decision about which cheapest explanation the decoder adopts and not an estimate of the probability that the observable flipped, and it is not calibrated: two mechanisms that share a detector pair and disagree on their label are indistinguishable from the syndrome alone, so the decoder cannot always recover the more likely value. Its failure rate is therefore at or above the optimal decoder's on every model and strictly above it once rounds exceed one. A single-round repetition model has no such pair, and there the matcher reaches the exhaustive optimum exactly, which is the only optimality claim made here. Absent: belief propagation with ordered statistics decoding, a sliding-window decoder, a union-find or belief-finding decoder, a decoder registry or plugin protocol, and any hyperedge-decomposition front end that would widen the graphlike scope. Batching, streaming, a per-detector error-rate vector, and per-mechanism identifiers are also absent, and the decoder is not connected to the stim sampling path. The matrix route that DetectorErrorModel.from_code_matrices provides is inside this decoder's graphlike domain only while it stays inside one round: at two rounds the middle data wire of the distance-3 repetition code flips the same check in both detector bands, which is a four-detector mechanism, and the decoder refuses it as a hyperedge rather than projecting it onto a pair. The observable prediction has no consumer yet: no logical-error-rate estimator, no threshold scan, and no integration with the memory-experiment result records. The result record is deliberately not the repetition-code DecodeResult, whose correction carries a wire in {0, 1, 2} and an X basis only and therefore cannot express a surface-code correction. A second decoder exists behind the pymatching extra and is a cross-check rather than the authority: flagquantum.qec.adapters translates the same decoding graph, reports the same two records, and is imported lazily, so the core install and the decoder neither need nor load it. What the comparison establishes is stated narrowly, because the two instruments are not equal: PyMatching reports 3.9020747171643912 for a mechanism this package states as 3.9020746947749574, so the graph it minimizes over is not exactly this one and two explanations closer than that difference can be ordered differently on the two sides. The two decoders are therefore compared on the cheapest weight of every syndrome with a tolerance of one single-precision rounding per selected mechanism, on the observables wherever the cheapest explanation is unique, and on their refusals as the same set; a tie is uncomparable rather than evidence against either implementation, and one tie of the distance-three repetition code is pinned by hand so that the clause is a demonstrated fact rather than a place a disagreement could hide. The translation refuses two graphs instead of approximating them, and both are graphs the matcher itself answers: a detector pair carrying two mechanisms, which PyMatching's independent merge strategy would collapse and thereby lose a logical-label difference, and a detector that no mechanism flips, which PyMatching cannot represent because it infers its detector count from its edges. The cross-check's domain is therefore narrower than the authority's and never the reverse. The matcher is checked against brute force on syndromes small enough to enumerate, against the exhaustive enumeration of a model's own mechanism distribution, and against that second decoder over the same graph, so its correctness rests on the detector error model all three read and on the authority of no external decoder.

### Sampled detection events from a memory circuit

Sample a memory experiment's detection events and observable flips from the circuit itself under phenomenological noise, so a decoding claim has a route that does not depend on the statevector ceiling.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.qec.sample_memory_circuit`, `flagquantum.simulation.stabilizer.sample_noisy_measurements`
- **Runtime modes:** `stabilizer`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../flagquantum/qec/README.md)
- **Documentation:** [guide](../../flagquantum/qec/README.md)
- **Known boundary:** One experiment shape and one noise grammar. The circuit must be a MemoryCircuit, so the sampler is reachable from a code record and not from an arbitrary annotated circuit; the noise must be a PhenomenologicalNoise, so a data flip at a round boundary and a measurement flip at a check's readout are the only two locations that exist, and a channel placed after a named gate, an asymmetric X/Z rate, a per-qubit or per-check rate, and a depolarizing or damping channel have no location to be placed at. Placement is derived from the lowered program rather than written into the experiment, because the bounded hybrid capture refuses a channel call in the source; the derivation requires the program to lower to `rounds` identical round blocks that measure every check once in the code's declared check order, and refuses a program that does not, so an experiment whose round structure differs is declined rather than sampled. The sampler executes on the stabilizer engine, so it requires the optional `stim` distribution and inherits the engine's opcode set: exactly the thirteen Clifford opcodes, with a non-Clifford gate refused by name. It shares the code record's Z-memory limit, so an X-type logical observable has no route here. Sampling is not differentiable and returns no gradients. The result is a DemSample like the model's own sampler returns, and the two are deliberately different code paths: the model derives a mechanism's signature from the source while this sampler derives its instruction position from the lowered program, and the tests pin the two to each other rather than having one call the other. A seed reproduces a run on one engine version and one machine instruction set; it does not pin a bit pattern across machines or engine versions. No threshold, no logical-suppression, no fault-tolerance, no real-time and no performance claim follows. The rate this sampler reports is a circuit-sampled rate, and the model's own sampler reports a DEM-sampled one; neither may be reported as the other.

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
- **Known boundary:** The advantage premise is coherent entry-wise database access, and this unit does not meet it. The cited paper's speed-up is measured in calls to an oracle that returns one database entry per call, and it reaches the candidate itemset superpositions it prepares through a qRAM; neither is exercised here: the transactions are iterated over classically, one controlled increment of the support register per transaction-item membership, emitted from the incidence matrix as Python reads it, so the access cost is paid explicitly rather than assumed away and no end-to-end advantage follows. The improvement the paper claims is quadratic in the number of database queries and is stated conditionally, for the case M_f^(k) << M_c^(k); it is not exponential, and that wording appears only in an earlier arXiv listing of the same work. The support register must be wide enough to hold the largest support any database of that transaction count could produce: the increment is a permutation of the register's own values, so a narrower register wraps a support into another value and the readout can come back wrong with nothing raised, and a width that cannot hold every support is therefore refused rather than left to wrap. The item register is addressed by one wire per item-index bit, so the item count is a power of two. Demonstration scale: at most eight items and seven transactions, and the marking operator enumerates the support values at or above the threshold. It makes no performance, convergence, or hardware claim.

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
- **Known boundary:** The advantage premise is the input model, and this unit does not meet it. The cited algorithm's cost is counted in queries to a structure that returns the matrix's entries, and against that count the state the estimation is applied to is assumed to be preparable; neither is present here. The matrix is held as an ordinary tensor, its embedding is formed and exponentiated as a dense matrix, and the input state is built from the singular vectors a classical torch.linalg.svd returns -- the very decomposition the readout estimates -- so the access and the preparation are paid explicitly rather than assumed away and no end-to-end advantage follows. Dequantization is recorded rather than glossed over: Tang's classical algorithm for the recommendation problem removes the exponential speed-up and is only polynomially slower, its bound containing eps**-12, which the author calls a large slowdown in some exponents; it is not a classical algorithm that matches the quantum runtime. Arrazola et al. record the practical conditions the dequantized algorithms need, and Gharibian-Le Gall dequantize the quantum singular value transformation for sparse matrices at constant precision; their hardness result is for a different task, estimating a local Hamiltonian's ground-state energy at inverse-polynomial precision given a state close to the ground state. The block encoding is not free to read: a readout that post-selects the ancilla succeeds with probability ||(A/alpha)|psi>||**2 on a normalised input, whose greatest value over inputs is (||A||/alpha)**2, and where alpha is much larger than ||A|| that probability is exponentially small. The readout is the counting register's mode, and the unit does not claim that the mode's value is the largest singular value: the readout is a grid value at the register's own resolution, and within() is the whole of the accuracy contract. The counting width is the caller's and is at least one; at one wire the register can only return alpha itself, and a run whose mode falls in the register's lower half is refused rather than returning a value above alpha -- an outcome of the sample rather than a precondition on the arguments. The input state is prepared from the singular vectors, so the classical work includes the very decomposition the unit estimates. Demonstration scale: at most four rows and four columns, with the embedding, its exponential and the state preparation all classical. It makes no performance, convergence, or hardware claim.

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

### Clifford circuit sampling by Pauli stabilizer tracking

Sample computational-basis outcomes from a Clifford circuit whose wire count puts a dense amplitude store out of reach.

- **Maturity:** Development evidence
- **Public API:** `flagquantum.simulation.stabilizer.require_clifford_program`, `flagquantum.simulation.stabilizer.sample_stabilizer`, `flagquantum.simulation.stabilizer.sample_noisy_measurements`
- **Runtime modes:** `stabilizer`
- **Hardware:** `cpu`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../examples/stabilizer_sampling.py)
- **Documentation:** [guide](../../flagquantum/simulation/stabilizer/README.md)
- **Known boundary:** Requires the optional `stim` distribution (Apache-2.0), installed with `pip install 'flagquantum[stim]'`; without it the entry point raises StabilizerDependencyError naming that command rather than falling back to an amplitude path. Accepted gates are exactly the thirteen Clifford opcodes in flagquantum.core.operator_schema -- i, x, y, z, h, s, sdg, sx, sxdg, cx, cy, cz, swap -- and every other opcode is refused with a CapabilityError naming it and the accepted set. ccx and cswap are refused because neither normalizes the Pauli group, so they are not Clifford gates despite being permutations; t, tdg, the rotation and controlled-rotation family, phase, u1, u2 and u3 are refused for the same reason. Circuits passed to sample_stabilizer must be noiseless: the four noise channels are refused rather than sampled as unitaries, so no readout error, depolarizing, damping or trajectory behavior is available. The second entry point, sample_noisy_measurements, is the narrow exception, and it is not a widening of that refusal: it executes a bit-flip channel at an explicit position a caller has already chosen, so the frame tracking is the caller's placement and not an inference the engine makes, and a channel whose Kraus operators are not the two-element bit-flip pair is still refused by name. It reads no lowered measurement node, returns only the sampled bits with no detection events or observable flips, and is not reachable from the planner or the executor, so `mode='stabilizer'` keeps refusing a noisy program. Only computational-basis measurement sampling is supported; expectation values, detector error models, decoding, error-corrected logical operations and hardware-backed sampling are absent. Sampling is not differentiable and returns no gradients. The execution route is explicit: `fq.ExecutionOptions(mode='stabilizer')` reaches the engine through `fq.run`, the planner publishes a plan whose `state_mode` is `stabilizer` and whose `state_bytes` is the packed tableau rather than an amplitude count, and the result reports `execution_path='local_stabilizer'`. Automatic selection never chooses this mode, so no existing plan changes meaning. Planning refuses a non-Clifford gate, a noise channel, a batch, a non-CPU device, a gradient request, a non-sampling target, a sharded world size and a program without shots; sampling refuses a lowered request it cannot reconcile with its own arguments and postselection, and a memory limit below the tableau. None of those refusals is a fallback, and none of them is a scalability claim. Runs are CPU-only single-device work with no GPU or distributed sharding. A seed reproduces a run on one engine version and one machine instruction set; it does not pin a bit pattern across machines or engine versions. The capacity statement is representational -- memory grows with the wire count squared rather than exponentially -- and no throughput, latency or scaling claim is made.


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

- **Maturity:** Production supported
- **Public API:** `fq.plan`, `flagquantum.experimental.distributed.train_distributed_statevector`
- **Runtime modes:** `distributed_statevector`
- **Hardware:** `multi_node`, `nvidia_a800_sxm4_80gb`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../docs/guides/MULTINODE_RUNBOOK.md)
- **Documentation:** [guide](../../docs/development/TESTING.md)
- **Known boundary:** Scope is the recorded pair: two A800 hosts with one device per host and one five-wire complex128 circuit. Inter-node shard exchange, the exact adjoint gradient, an owner-sharded optimizer step, and a checkpoint resumed by a restarted run all executed on that pair, and the run reports scalability_claim_allowed and release_gate_allowed false with four blockers attached. The route was observed on the RoCE fabric, and the probe recorded five synchronized forward samples after two warmups, so the fabric and a repeated measurement are evidence rather than assumptions; congestion and capacity were still not measured. That measurement is of a five-wire circuit and therefore says nothing about production scale, which stays blocked as a toy circuit. The profiled measurement region contains explicit host transfers, which keeps a blocker of its own: part of each sample is host staging rather than device work. Topologies wider than one device per host run on the lane but are not recorded, and the tiny full state is gathered for validation only.

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

- **Maturity:** Production supported
- **Public API:** `fq.plan`, `flagquantum.experimental.distributed.train_distributed_mps`
- **Runtime modes:** `distributed_mps`
- **Hardware:** `multi_node`, `nvidia_a800_sxm4_80gb`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../docs/guides/MULTINODE_RUNBOOK.md)
- **Documentation:** [guide](../../docs/development/TESTING.md)
- **Known boundary:** Scope is the recorded pair: two A800 hosts with one device per host, one six-wire complex128 circuit, and a bond limit that truncates nothing. Site-boundary exchange, an exact reverse gradient with the layer halo prefetched over the inter-node transport, an owner-sharded optimizer step, and a checkpoint resumed by a restarted run all executed on that pair, and the run reports scalability_claim_allowed and release_gate_allowed false with four blockers attached. The route was observed on the RoCE fabric and five synchronized forward samples were recorded after two warmups, so the fabric and a repeated measurement are evidence rather than assumptions; congestion and capacity were still not measured. The cut width was swept rather than declared: the recorded workload's Schmidt profile is flat at rank two, so the sweep binds a circuit of its own whose profile is not, and re-runs the same reverse under every site plan the launched shape can hold with a boundary moved, comparing each leg with the exact complex128 expectation and gradient. At two ranks that is five placements across widths two and four, each of which carried bytes over the inter-node transport; at a wider shape the same leg walks each rank boundary in turn, but a plan at more than two ranks carries several boundaries at once and is named by the heaviest of them, so no wider shape is recorded here and no leg claims a bond it did not carry. The tiny full MPS is gathered for validation only, the profiled measurement region contains explicit host transfers that keep a blocker of their own, and topologies wider than one device per host run on the lane but are not recorded, so this evidence does not establish behaviour as nodes are added. The layer halo crosses the host boundary only on the compiled site-kernel path, which the checkpointed training legs deliberately do not take, and the sweep therefore measures the compiled path.

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
- **Known boundary:** Scope is the recorded pair: two A800 hosts with one device per host, one five-wire complex128 circuit with toy parameters, and a cut of exactly two labels declared by the workload. Slice-boundary reduction, the exact slice-gradient sum, an owner-sharded optimizer step, and a checkpoint generation resumed by a restarted run all executed on that pair, and the run reports scalability_claim_allowed and release_gate_allowed false with four blockers attached. The route was observed on the RoCE fabric and five synchronized amplitude samples were recorded after two warmups, so the fabric and a repeated measurement are evidence rather than assumptions. The cut width was swept across both labels of the declared cut, which retired the blockers that named an unswept cut and a fixed slice count. The cut is still declared rather than chosen by the automatic slicer, which selects by peak memory and would here pick a label carried only by state-copy nodes whose partial is zero on every rank but one; slicing such a label would report sharded execution while one rank did the arithmetic. Congestion and capacity were not measured, the profiled measurement region contains explicit host transfers that keep a blocker of their own, the sliced full state is gathered for validation only, and topologies wider than one device per host run on the lane but are not recorded, so this evidence does not establish behaviour as the cut widens or as nodes are added.


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
- **Known boundary:** The flagquantum.ecosystem.pennylane bridge executes one fully bound, single-batch FlagQuantum circuit on local CPU lightning.qubit and returns an owned ExecutionResult. It supports exact complex64 and complex128 statevectors and computational-basis samples or counts with explicit wire order and seed. Explicit device wires preserve idle FlagQuantum wire extent. It does not support gradients, QNodes, noise models, dynamic circuits, automatic routing, GPU, alternative PennyLane devices, or fallback. Native fq.run remains unchanged.

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
- **Known boundary:** The flagquantum.ecosystem.cirq bridge executes one fully bound, single-batch FlagQuantum circuit on local CPU Cirq Simulator and returns an owned ExecutionResult. It supports exact complex64 and complex128 statevectors and computational-basis samples or counts with explicit wire order and seed. Explicit qubit order preserves idle FlagQuantum wire extent. It does not support gradients, noise models, dynamic circuits, automatic routing, device selection, or fallback. Native fq.run remains unchanged.

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
- **Known boundary:** Certified against Cirq 1.6.1 and 1.7.0 on Python 3.11 or newer for static circuit conversion with complex128 numerical semantics, bidirectional and fail-closed unless allow_lossy is set. Moment structure is flattened and not preserved, and the contiguous-line-qubit mapping rejects non-line and non-contiguous line qubits. Arbitrary qid dimensions, CircuitOperation blocks, classical controls, MatrixGate, operation tags, and global-phase operations are unsupported, as are idle-wire extents and global phase in representation. Nineteen FlagQuantum opcodes have no Cirq lowering. The parameter subset is FlagQuantum add/multiply/negate after Cirq symbolic simplification with cirq.parameter_symbols discovery and cirq.resolve_parameters binding equivalence; function, power, complex-constant, and division-by-unbound-parameter nodes are rejected before an operation is created. Measurements convert only as terminal MeasurementGate instructions with distinct non-empty keys, and invert masks, confusion maps, and non-terminal measurements are not representable. Execution, gradients, noise, and hardware submission are out of scope; Cirq objects never enter Core, Compiler, Runtime, or Simulation, and the separate local Cirq Simulator bridge is declared as cirq_simulator_execution.

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
- **Known boundary:** Certified against amazon-braket-sdk 1.117.0 and 1.127.1 on Python 3.11 or newer for bidirectional static circuit conversion, fail-closed unless allow_lossy is set. The Braket integer qubit index equals the FlagQuantum wire, and explicit identity preserves unreferenced FlagQuantum wires. Only bound finite real scalars are accepted; free parameters are rejected instead of converted, so no symbolic parameter expression survives the boundary. Global phase and measurement instructions are not represented, and measurements belong to the execution plan rather than to static circuit IR. Compiler directives, gate calibrations, noise instructions, pulse gates, verbatim boxes, result types, and wire-extent preservation are unsupported, and nine FlagQuantum opcodes have no Braket lowering. Core import, runtime execution, cloud submission, and autograd bridging are all prohibited at this boundary, and Braket objects never enter FlagQuantum runtime layers. docs/reference/API.md documents Braket through the adapter contract and the provider execution path only; it has no static-conversion section, so the contract file is the authoritative statement of scope.

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
- **Known boundary:** Certified against Qiskit 2.0.x and 2.5.x with Aer 0.17.x through an executable operation, wire-order, statevector, classical-bit, arithmetic-parameter-expression, custom-unitary, and round-trip contract. Six fixed-seed differential programs exercise both conversion directions across three to five wires, mixed one- to three-wire operations, reordered wires, and asymmetric custom unitaries. ParameterExpression import supports the FlagQuantum v1 add/multiply/negate arithmetic subset after Qiskit symbolic simplification; functions, powers, and other operations fail closed. One- to three-qubit custom unitary matrices are converted with explicit local basis-order normalization and validated before export. Qiskit control flow is rejected; named or multiple registers require explicit lossy flattening; custom unitary matrices above three qubits are rejected. Conversion does not make Qiskit a runtime dependency or certify any provider hardware.

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
- **Known boundary:** The flagquantum.ecosystem.qiskit Aer bridge executes one fully bound, single-batch FlagQuantum circuit on local CPU Aer and returns an owned ExecutionResult. It supports exact statevectors and computational-basis samples or counts with explicit wire order, seed, and CPU thread controls. It does not support gradients, noise models, dynamic circuits, automatic routing, provider hardware, or fallback. Native fq.run remains unchanged.

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
- **Known boundary:** The migrated SDK protocol is approved but not frozen; individual extensions remain experimental until separately qualified. Compiler plugins currently exchange CircuitIR only; pulse and native-binary artifacts are not supported. An execution backend is admitted in-process through the host-side admission module and executes FlagQuantum IR; there is no stable C ABI, no out-of-process backend, no per-execution discovery, and no numerical equivalence certification between an admitted backend and the built-in engine.


## Validated public performance claims

Every value below is read from a hash-bound raw artifact. Missing or changed
evidence makes the source-of-truth check fail closed.

| Claim | Maturity and scope | Artifact-derived result | Recorded environment | Evidence identity |
| --- | --- | --- | --- | --- |
| **Two-node statevector forward latency**<br>`multinode-statevector-forward-latency-20261001` | `production_supported`<br>One sharded forward pass of the recorded two-node workload, timed between CUDA synchronizations on two A800 hosts with one device per host and a RoCE fabric route read from the NCCL debug log. This is a latency of a five- or six-wire toy circuit and not a scaling, throughput, or capacity claim; the artifact's own blockers say so and both of its claim flags are false. | **Minimum forward latency (s):** 2.98e-03<br>**Median forward latency (s):** 3.19e-03<br>**Maximum forward latency (s):** 3.21e-03 | **Warmup iterations before sampling:** 2<br>**Synchronized samples:** 5<br>**Observed transport route:** infiniband<br>**Devices per node:** 1<br>**Metadata boundary:** The artifact records the accelerator, node count, world and local world size, dtype, route, NCCL, CUDA runtime, PyTorch and Python versions, and the per-sample latencies. It does not record GPU clocks, power state, temperature, host CPU or memory, fabric congestion, or the driver version, so the numbers describe this pair at the recorded revision and nothing wider. | [raw JSON](../../benchmarks/results/local/multinode_two_node_performance_20261001.json)<br>SHA-256 `22c88fa84b1e0518efa416da49e1187a01e447ef6886f133f5be60e5183f084e`<br>code `591401fac230213aa73a3b253a89aa2ef919e510` |
| **Sharded MPS exact-workload capacity**<br>`mps-capacity-131072-chi768-20260806` | `development_evidence`<br>One batch-one, complex64, χ768 MPS training step for the checked-in all-rank and all-boundary workload. This is not arbitrary statevector capacity, fixed-plan strong scaling, or release evidence. | **Sites:** 131,072<br>**Logical MPS state:** 1,236,780,012,864 bytes (1,151.84 GiB)<br>**Maximum elapsed time:** 367.37 s<br>**Maximum peak allocated memory per rank:** 77,745,407,488 bytes (72.41 GiB)<br>**Cumulative discarded weight:** 8.39e-06 | **Ranks:** 16<br>**Reported device memory per rank:** 85,093,777,408 bytes (79.25 GiB)<br>**CUDA allocator policy:** expandable_segments:True<br>**Topology fingerprint:** c74a91e3a224a4dd414cfbfbcb31da7570880d42e6aa4eaccecf4b85427940a5<br>**Metadata boundary:** The artifact records rank count, per-rank device memory, topology fingerprint, and CUDA allocator policy. It does not record the exact GPU model or Python, PyTorch, CUDA, NCCL, driver, host, and operating-system versions, so the claim is restricted to the recorded environment fields. | [raw JSON](../../benchmarks/results/local/mps_capacity_131072q_chi768_16xa800_repeat_complete_20260806.json)<br>SHA-256 `df8c19b74cc173799e3894025fb8e72de72a786d4542a9aa5e67687298f482f5`<br>code `9d56a6ecd78b06f11b9ee6e8aadcbe9644f2c708` (produced on a host whose history this repository does not contain) |
| **Two-node MPS forward latency**<br>`multinode-mps-forward-latency-20261001` | `production_supported`<br>One sharded forward pass of the recorded two-node workload, timed between CUDA synchronizations on two A800 hosts with one device per host and a RoCE fabric route read from the NCCL debug log. This is a latency of a five- or six-wire toy circuit and not a scaling, throughput, or capacity claim; the artifact's own blockers say so and both of its claim flags are false. | **Minimum forward latency (s):** 2.04e-02<br>**Median forward latency (s):** 2.15e-02<br>**Maximum forward latency (s):** 2.24e-02 | **Warmup iterations before sampling:** 2<br>**Synchronized samples:** 5<br>**Observed transport route:** infiniband<br>**Devices per node:** 1<br>**Metadata boundary:** The artifact records the accelerator, node count, world and local world size, dtype, route, NCCL, CUDA runtime, PyTorch and Python versions, and the per-sample latencies. It does not record GPU clocks, power state, temperature, host CPU or memory, fabric congestion, or the driver version, so the numbers describe this pair at the recorded revision and nothing wider. | [raw JSON](../../benchmarks/results/local/multinode_two_node_performance_20261001.json)<br>SHA-256 `22c88fa84b1e0518efa416da49e1187a01e447ef6886f133f5be60e5183f084e`<br>code `591401fac230213aa73a3b253a89aa2ef919e510` |
| **Two-node sliced amplitude latency**<br>`multinode-tensor-network-amplitude-latency-20261001` | `production_supported`<br>One sharded forward pass of the recorded two-node workload, timed between CUDA synchronizations on two A800 hosts with one device per host and a RoCE fabric route read from the NCCL debug log. This is a latency of a five- or six-wire toy circuit and not a scaling, throughput, or capacity claim; the artifact's own blockers say so and both of its claim flags are false. | **Minimum amplitude latency (s):** 4.19e-03<br>**Median amplitude latency (s):** 4.21e-03<br>**Maximum amplitude latency (s):** 4.56e-03 | **Warmup iterations before sampling:** 2<br>**Synchronized samples:** 5<br>**Observed transport route:** infiniband<br>**Devices per node:** 1<br>**Metadata boundary:** The artifact records the accelerator, node count, world and local world size, dtype, route, NCCL, CUDA runtime, PyTorch and Python versions, and the per-sample latencies. It does not record GPU clocks, power state, temperature, host CPU or memory, fabric congestion, or the driver version, so the numbers describe this pair at the recorded revision and nothing wider. | [raw JSON](../../benchmarks/results/local/multinode_two_node_performance_20261001.json)<br>SHA-256 `22c88fa84b1e0518efa416da49e1187a01e447ef6886f133f5be60e5183f084e`<br>code `591401fac230213aa73a3b253a89aa2ef919e510` |
