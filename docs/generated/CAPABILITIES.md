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
| Train a large low-entanglement system | Differentiable and sharded MPS training | Development evidence | [Run example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py) |
| Distribute one MPS across several GPUs | Differentiable and sharded MPS training | Development evidence | [Run example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py) |
| Inspect variable-bond MPS capacity | Differentiable and sharded MPS training | Development evidence | [Run example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py) |
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
- **Public API:** `fq.Circuit`, `fq.CircuitIR`, `fq.compile_for_backend`
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
- **Public API:** `fq.resolve_device`, `fq.run`
- **Runtime modes:** `statevector`
- **Hardware:** `nvidia_a100_cuda_reference`
- **Gradient support:** `development_evidence`
- **Distribution semantics:** `single_device_fast_path`
- **Start:** [quick example](../../docs/reference/ACCELERATOR_PLATFORM_RUNTIME.md)
- **Documentation:** [guide](../../docs/reference/STATEVECTOR_OPERATOR_PROFILES.md)
- **Known boundary:** CUDA-backed development reference only. It does not certify a domestic accelerator, prove absence of Torch-FL host fallback, establish production performance, or authorize a scalability claim.

### Constrained local MPS TEBD

Evolve open-chain local Pauli Hamiltonians with fail-closed second-order imaginary-time TEBD.

- **Maturity:** Experimental
- **Public API:** `fq.experimental.run_tebd`, `fq.experimental.TEBDResult`
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
- **Public API:** `fq.run_tensor_network`, `fq.plan_runtime_selection`
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
- **Public API:** `fq.NoiseModel`, `fq.noisy_density_matrix`, `fq.run_noisy_mps`
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
- **Public API:** `fq.plan`, `fq.train_distributed_statevector`
- **Runtime modes:** `distributed_statevector`
- **Hardware:** `multi_gpu`, `multi_node`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../examples/distributed_statevector_topologies/run.sh)
- **Documentation:** [guide](../../examples/distributed_statevector_topologies/README.md)
- **Known boundary:** Multi-node release certification remains dependent on promoted audited hardware evidence.

### Differentiable and sharded MPS training

Train low-entanglement quantum systems with local or rank-owned matrix product states.

- **Maturity:** Development evidence
- **Public API:** `fq.run_mps`, `fq.train_distributed_mps`, `fq.plan_production_mps`
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
- **Public API:** `fq.create_deployment_package`, `fq.deploy_circuit`
- **Runtime modes:** `provider`
- **Hardware:** `provider_dependent`
- **Gradient support:** `not_applicable`
- **Distribution semantics:** `provider_dependent`
- **Start:** [quick example](../../examples/train_parameterized_circuit_then_deploy.py)
- **Documentation:** [guide](../../docs/reference/API.md)
- **Known boundary:** Provider support and credential/runtime behavior vary; no provider is release-certified by this matrix.

### Qiskit IR interoperability

Translate supported Qiskit circuits to versioned FlagQuantum IR and export FlagQuantum IR through an isolated, loss-aware control-plane adapter.

- **Maturity:** Experimental
- **Public API:** `fq.experimental.from_qiskit`, `fq.experimental.to_qiskit`
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

Add gates, transformations, runtime hooks, and providers through the experimental extension surface.

- **Maturity:** Experimental
- **Public API:** `fq.experimental`
- **Runtime modes:** `extension_defined`
- **Hardware:** `extension_defined`
- **Gradient support:** `extension_defined`
- **Distribution semantics:** `extension_defined`
- **Start:** [quick example](../../examples/extensions/reference_extensions.py)
- **Documentation:** [guide](../../docs/reference/EXTENSION_SDK.md)
- **Known boundary:** Extension compatibility is not guaranteed before stabilization.


## Validated public performance claims

Every value below is read from a hash-bound raw artifact. Missing or changed
evidence makes the source-of-truth check fail closed.

| Claim | Maturity and scope | Artifact-derived result | Recorded environment | Evidence identity |
| --- | --- | --- | --- | --- |
| **Sharded MPS exact-workload capacity**<br>`mps-capacity-131072-chi768-20260806` | `development_evidence`<br>One batch-one, complex64, χ768 MPS training step for the checked-in all-rank and all-boundary workload. This is not arbitrary statevector capacity, fixed-plan strong scaling, or release evidence. | **Sites:** 131,072<br>**Logical MPS state:** 1,236,780,012,864 bytes (1,151.84 GiB)<br>**Maximum elapsed time:** 367.37 s<br>**Maximum peak allocated memory per rank:** 77,745,407,488 bytes (72.41 GiB)<br>**Cumulative discarded weight:** 8.39e-06 | **Ranks:** 16<br>**Reported device memory per rank:** 85,093,777,408 bytes (79.25 GiB)<br>**CUDA allocator policy:** expandable_segments:True<br>**Topology fingerprint:** c74a91e3a224a4dd414cfbfbcb31da7570880d42e6aa4eaccecf4b85427940a5<br>**Metadata boundary:** The artifact records rank count, per-rank device memory, topology fingerprint, and CUDA allocator policy. It does not record the exact GPU model or Python, PyTorch, CUDA, NCCL, driver, host, and operating-system versions, so the claim is restricted to the recorded environment fields. | [raw JSON](../../benchmarks/results/local/mps_capacity_131072q_chi768_16xa800_repeat_complete_20260806.json)<br>SHA-256 `efb0d34c741d1196a7f7c89404cbf033c655f14bf5d575470bbe605bb93b0cbc`<br>code `9d56a6ecd78b06f11b9ee6e8aadcbe9644f2c708` |
