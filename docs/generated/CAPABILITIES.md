# FlagQuantum Capabilities

Do not edit. Source: `capability-maturity.toml`. Maturity describes the exact documented scope; it does not turn local, replicated, or planned execution into distributed scalability evidence.

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
| Train a large low-entanglement system | Differentiable and sharded MPS training | Development evidence | [Run example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py) |
| Distribute one MPS across several GPUs | Differentiable and sharded MPS training | Development evidence | [Run example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py) |
| Inspect variable-bond MPS capacity | Differentiable and sharded MPS training | Development evidence | [Run example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py) |
| Evaluate a circuit with tensor-network contraction | Tensor-network execution and training | Experimental | [Run example](../../examples/vqe_switch_sv_mps_tn.py) |
| Compare statevector, MPS, and tensor-network modes | Tensor-network execution and training | Experimental | [Run example](../../examples/vqe_switch_sv_mps_tn.py) |
| Package a trained parameterized circuit | Circuit packaging and cloud deployment | Development evidence | [Run example](../../examples/train_parameterized_circuit_then_deploy.py) |
| Export a circuit for a provider | Circuit packaging and cloud deployment | Development evidence | [Run example](../../examples/train_parameterized_circuit_then_deploy.py) |
| Run a circuit through a deployment abstraction | Circuit packaging and cloud deployment | Development evidence | [Run example](../../examples/train_parameterized_circuit_then_deploy.py) |
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
- **Documentation:** [guide](../../docs/API.md)
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

### Tensor-network execution and training

Execute tensor-network circuit paths and evaluate experimental contraction and gradient workflows.

- **Maturity:** Experimental
- **Public API:** `fq.run_tensor_network`, `fq.plan_runtime_selection`
- **Runtime modes:** `tensor_network`
- **Hardware:** `cpu`, `single_gpu`
- **Gradient support:** `experimental`
- **Distribution semantics:** `manual_sliced_tensor_contraction`
- **Start:** [quick example](../../examples/vqe_switch_sv_mps_tn.py)
- **Documentation:** [guide](../../docs/KNOWN_LIMITATIONS.md)
- **Known boundary:** General reverse contraction and production distributed transport are not certified.


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
- **Hardware:** `cpu`, `single_gpu`, `multi_gpu`
- **Gradient support:** `exact`
- **Distribution semantics:** `sharded_across_ranks`
- **Start:** [quick example](../../examples/distributed_mps/variable_bond_capacity_8gpu.py)
- **Documentation:** [guide](../../examples/distributed_mps/README.md)
- **Known boundary:** Single-node 2/4/8-GPU forward and boundary transport plus a 100-step 8-GPU SGD soak are validated; other optimizers, multi-node operation, and release payload evidence remain incomplete.


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
- **Documentation:** [guide](../../docs/API.md)
- **Known boundary:** Provider support and credential/runtime behavior vary; no provider is release-certified by this matrix.

### Dynamic circuits and IQM Braket preflight

Execute dynamic circuits locally and prepare sealed IQM OpenQASM 3 programs with fail-closed hardware preflight.

- **Maturity:** Experimental
- **Public API:** `fq.experimental`
- **Runtime modes:** `local_statevector_trajectory`, `provider_preflight`
- **Hardware:** `cpu`, `amazon_braket_iqm_unverified`
- **Gradient support:** `unsupported`
- **Distribution semantics:** `single_process`
- **Start:** [quick example](../../examples/braket_iqm_dynamic_preflight.py)
- **Documentation:** [guide](../../docs/API.md)
- **Known boundary:** Local trajectory execution, IQM dialect serialization, SDK Program construction, sealed packaging, and mocked provider submission are tested. No AWS account or real IQM QPU task was used, so device availability, published qubit groups, billing, credentials, and hardware results remain unverified.

### Extension SDK

Add gates, transformations, runtime hooks, and providers through the experimental extension surface.

- **Maturity:** Experimental
- **Public API:** `fq.experimental`
- **Runtime modes:** `extension_defined`
- **Hardware:** `extension_defined`
- **Gradient support:** `extension_defined`
- **Distribution semantics:** `extension_defined`
- **Start:** [quick example](../../examples/extensions/reference_extensions.py)
- **Documentation:** [guide](../../docs/EXTENSION_SDK.md)
- **Known boundary:** Extension compatibility is not guaranteed before stabilization.
