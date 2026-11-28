# Independent migration path map

Main baseline: `7278efd00f6768fbbacff6a41eb4dec57a59783e`.
Initial #1084 snapshot: `9b1a8cd196f9d8b8427b382c404f66705cc57840`.
Example batch snapshot: `11fd9fa27ec31e74f03277010762e8d25d17c105`; see the [batch record](example-structure-migration.md).

Each row names the authoritative owner. Partial rows move only the listed definitions. Public and #1084-dependent retained facades are individually justified in the [retirement inventory](internal-forwarder-retirement.md); retired internal/command entries are absent.

| Group | Previous path | Authoritative path | Scope |
| --- | --- | --- | --- |
| compute | `flagquantum/compute/_memory.py` | `flagquantum/compute/local/_memory.py` | entry removed; callers updated |
| compute | `flagquantum/compute/contracts.py` | `flagquantum/compute/local/contracts.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| compute | `flagquantum/compute/registry.py` | `flagquantum/compute/local/registry.py` | entry removed; callers updated |
| compute | `flagquantum/compute/pytorch.py` | `flagquantum/compute/local/pytorch.py` | entry removed; callers updated |
| compute | `flagquantum/compute/flagos.py` | `flagquantum/compute/local/flagos.py` | entry removed; callers updated |
| compute | `flagquantum/compute/flaggems.py` | `flagquantum/compute/local/flaggems.py` | entry removed; callers updated |
| compute | `flagquantum/compute/cpu_target_capabilities.py` | `flagquantum/compute/local/cpu_target_capabilities.py` | entry removed; callers updated |
| compute | `flagquantum/compute/cuda_target_capabilities.py` | `flagquantum/compute/local/cuda_target_capabilities.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/flagtree.py` | `flagquantum/simulation/kernels/flagtree.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/__init__.py` | `flagquantum/simulation/kernels/__init__.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/provenance.py` | `flagquantum/simulation/kernels/provenance.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/catalog/semantics.py` | `flagquantum/simulation/kernels/catalog/semantics.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/catalog/__init__.py` | `flagquantum/simulation/kernels/catalog/__init__.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/catalog/matching.py` | `flagquantum/simulation/kernels/catalog/matching.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/catalog/implementations.py` | `flagquantum/simulation/kernels/catalog/implementations.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/catalog/evidence.py` | `flagquantum/simulation/kernels/catalog/evidence.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/catalog/schema.py` | `flagquantum/simulation/kernels/catalog/schema.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/catalog/validation.py` | `flagquantum/simulation/kernels/catalog/validation.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/two_qubit_pauli_tangent.py` | `flagquantum/simulation/kernels/triton/two_qubit_pauli_tangent.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/statevector_pauli_rotation.py` | `flagquantum/simulation/kernels/triton/statevector_pauli_rotation.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/hva_forward_tangent.py` | `flagquantum/simulation/kernels/triton/hva_forward_tangent.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/mps_observable_adjoint.py` | `flagquantum/simulation/kernels/triton/mps_observable_adjoint.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/statevector_ry_fp64.py` | `flagquantum/simulation/kernels/triton/statevector_ry_fp64.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/statevector_reversible_3q.py` | `flagquantum/simulation/kernels/triton/statevector_reversible_3q.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/statevector_diagonal.py` | `flagquantum/simulation/kernels/triton/statevector_diagonal.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/mps_wire_probabilities.py` | `flagquantum/simulation/kernels/triton/mps_wire_probabilities.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/mps_one_site.py` | `flagquantum/simulation/kernels/triton/mps_one_site.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/__init__.py` | `flagquantum/simulation/kernels/triton/__init__.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/mps_sampling_collapse.py` | `flagquantum/simulation/kernels/triton/mps_sampling_collapse.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/single_qubit_loop.py` | `flagquantum/simulation/kernels/triton/single_qubit_loop.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/mps_two_site.py` | `flagquantum/simulation/kernels/triton/mps_two_site.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/statevector_adjoint.py` | `flagquantum/simulation/kernels/triton/statevector_adjoint.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/statevector_gates.py` | `flagquantum/simulation/kernels/triton/statevector_gates.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/complex_bmm.py` | `flagquantum/simulation/kernels/triton/complex_bmm.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/mps_environment.py` | `flagquantum/simulation/kernels/triton/mps_environment.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/statevector_measurement.py` | `flagquantum/simulation/kernels/triton/statevector_measurement.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/statevector_swap.py` | `flagquantum/simulation/kernels/triton/statevector_swap.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/statevector_cx_fp64.py` | `flagquantum/simulation/kernels/triton/statevector_cx_fp64.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/_jit.py` | `flagquantum/simulation/kernels/triton/_jit.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/extensions/__init__.py` | `flagquantum/simulation/kernels/triton/extensions/__init__.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/extensions/tle/__init__.py` | `flagquantum/simulation/kernels/triton/extensions/tle/__init__.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/extensions/tle/statevector_transport.py` | `flagquantum/simulation/kernels/triton/extensions/tle/statevector_transport.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/extensions/tle/statevector_adjoint.py` | `flagquantum/simulation/kernels/triton/extensions/tle/statevector_adjoint.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/triton/extensions/tle/statevector_gates.py` | `flagquantum/simulation/kernels/triton/extensions/tle/statevector_gates.py` | entry removed; callers updated |
| kernels | `flagquantum/kernels/README.md` | `flagquantum/simulation/kernels/README.md` | entry removed; callers updated |
| remote | `flagquantum/remote/_job_receipts.py` | `flagquantum/compute/remote/_job_receipts.py` | entry removed; callers updated |
| remote | `flagquantum/remote/__init__.py` | `flagquantum/compute/remote/__init__.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/jobs.py` | `flagquantum/compute/remote/jobs.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/_quafu_collection.py` | `flagquantum/compute/remote/_quafu_collection.py` | entry removed; callers updated |
| remote | `flagquantum/remote/qpu/iqm.py` | `flagquantum/compute/remote/qpu/iqm.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/qpu/quafu_dynamic_legality.py` | `flagquantum/compute/remote/qpu/quafu_dynamic_legality.py` | entry removed; callers updated |
| remote | `flagquantum/remote/qpu/execution.py` | `flagquantum/compute/remote/qpu/execution.py` | entry removed; callers updated |
| remote | `flagquantum/remote/qpu/quafu.py` | `flagquantum/compute/remote/qpu/quafu.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/qpu/_quafu_calibration.py` | `flagquantum/compute/remote/qpu/_quafu_calibration.py` | entry removed; callers updated |
| remote | `flagquantum/remote/qpu/azure.py` | `flagquantum/compute/remote/qpu/azure.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/qpu/__init__.py` | `flagquantum/compute/remote/qpu/__init__.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/qpu/declaration.py` | `flagquantum/compute/remote/qpu/declaration.py` | entry removed; callers updated |
| remote | `flagquantum/remote/qpu/quantinuum.py` | `flagquantum/compute/remote/qpu/quantinuum.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/qpu/_quafu_physical.py` | `flagquantum/compute/remote/qpu/_quafu_physical.py` | entry removed; callers updated |
| remote | `flagquantum/remote/qpu/quafu_dynamic.py` | `flagquantum/compute/remote/qpu/quafu_dynamic.py` | entry removed; callers updated |
| remote | `flagquantum/remote/qpu/quafu_dynamic_submission.py` | `flagquantum/compute/remote/qpu/quafu_dynamic_submission.py` | entry removed; callers updated |
| remote | `flagquantum/remote/qpu/http.py` | `flagquantum/compute/remote/qpu/http.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/qpu/_quafu_sdk.py` | `flagquantum/compute/remote/qpu/_quafu_sdk.py` | entry removed; callers updated |
| remote | `flagquantum/remote/qpu/contracts.py` | `flagquantum/compute/remote/qpu/contracts.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/qpu/result_parsing.py` | `flagquantum/compute/remote/qpu/result_parsing.py` | entry removed; callers updated |
| remote | `flagquantum/remote/qpu/ionq.py` | `flagquantum/compute/remote/qpu/ionq.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/qpu/calibration.py` | `flagquantum/compute/remote/qpu/calibration.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/qpu/braket.py` | `flagquantum/compute/remote/qpu/braket.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/kaiwu/sdk.py` | `flagquantum/compute/remote/kaiwu/sdk.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/kaiwu/_credentials.py` | `flagquantum/compute/remote/kaiwu/_credentials.py` | entry removed; callers updated |
| remote | `flagquantum/remote/kaiwu/client.py` | `flagquantum/compute/remote/kaiwu/client.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/kaiwu/__init__.py` | `flagquantum/compute/remote/kaiwu/__init__.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/kaiwu/jobs.py` | `flagquantum/compute/remote/kaiwu/jobs.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/kaiwu/contracts.py` | `flagquantum/compute/remote/kaiwu/contracts.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/compute/_resource_discovery.py` | `flagquantum/compute/remote/compute/_resource_discovery.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_jiuding_images.py` | `flagquantum/compute/remote/compute/_jiuding_images.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_job_payload.py` | `flagquantum/compute/remote/compute/_job_payload.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_program_job.py` | `flagquantum/compute/remote/compute/_program_job.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_workspace_executor.py` | `flagquantum/compute/remote/compute/_workspace_executor.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_workspace_results.py` | `flagquantum/compute/remote/compute/_workspace_results.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/execution.py` | `flagquantum/compute/remote/compute/execution.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_program_submission.py` | `flagquantum/compute/remote/compute/_program_submission.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_distributed_job.py` | `flagquantum/compute/remote/compute/_distributed_job.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/__init__.py` | `flagquantum/compute/remote/compute/__init__.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/compute/_jiuding_credentials.py` | `flagquantum/compute/remote/compute/_jiuding_credentials.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_native_job.py` | `flagquantum/compute/remote/compute/_native_job.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_workspace_discovery.py` | `flagquantum/compute/remote/compute/_workspace_discovery.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_managed_program.py` | `flagquantum/compute/remote/compute/_managed_program.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/_job_results.py` | `flagquantum/compute/remote/compute/_job_results.py` | entry removed; callers updated |
| remote | `flagquantum/remote/compute/jiuding.py` | `flagquantum/compute/remote/compute/jiuding.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/compute/_worker.py` | `flagquantum/compute/remote/compute/_worker.py` | entry removed; callers updated |
| remote | `flagquantum/realtime/service.py` | `flagquantum/compute/realtime/service.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/realtime/protocol.py` | `flagquantum/compute/realtime/protocol.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/realtime/session.py` | `flagquantum/compute/realtime/session.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/realtime/__init__.py` | `flagquantum/compute/realtime/__init__.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/realtime/payload.py` | `flagquantum/compute/realtime/payload.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/realtime/channels.py` | `flagquantum/compute/realtime/channels.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/emulation.py` | `flagquantum/compute/remote/emulation.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/README.md` | `flagquantum/compute/remote/README.md` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/AGENTS.md` | `flagquantum/compute/remote/AGENTS.md` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/qpu/README.md` | `flagquantum/compute/remote/qpu/README.md` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/kaiwu/README.md` | `flagquantum/compute/remote/kaiwu/README.md` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/remote/compute/README.md` | `flagquantum/compute/remote/compute/README.md` | implementation moved; public or #1084-dependent facade justified in inventory |
| remote | `flagquantum/realtime/README.md` | `flagquantum/compute/realtime/README.md` | implementation moved; public or #1084-dependent facade justified in inventory |
| dynamics | `flagquantum/simulation/lindblad.py` | `flagquantum/simulation/dynamics/lindblad.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| dynamics | `flagquantum/simulation/lindblad_adjoint.py` | `flagquantum/simulation/dynamics/adjoint.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| dynamics | `flagquantum/simulation/lindblad_integrators.py` | `flagquantum/simulation/dynamics/integrators.py` | entry removed; callers updated |
| dynamics | `flagquantum/simulation/lindblad_request.py` | `flagquantum/simulation/dynamics/problem.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| dynamics | `flagquantum/simulation/lindblad_generator.py` | `flagquantum/simulation/dynamics/generator.py` | entry removed; callers updated |
| runtime | `flagquantum/runtime/training.py` | `flagquantum/runtime/training/__init__.py` | same import identity; module converted to package |
| runtime | `flagquantum/runtime/module.py` | `flagquantum/runtime/training/module.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| runtime | `flagquantum/runtime/training_state.py` | `flagquantum/runtime/checkpointing.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| runtime | `flagquantum/runtime/training_compilation.py` | `flagquantum/runtime/training/compilation.py` | entry removed; callers updated |
| runtime | `flagquantum/runtime/result.py` | `flagquantum/runtime/results/__init__.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| runtime | `flagquantum/runtime/result_adapters.py` | `flagquantum/runtime/results/adapters.py` | entry removed; callers updated |
| runtime | `flagquantum/runtime/mps_result.py` | `flagquantum/runtime/results/mps.py` | entry removed; callers updated |
| runtime | `flagquantum/runtime/records.py` | `flagquantum/runtime/results/records.py` | entry removed; callers updated |
| runtime | `flagquantum/runtime/distributed/training_profile.py` | `flagquantum/runtime/profiles/training.py` | entry removed; callers updated |
| runtime | `flagquantum/runtime/distributed/scale_profile.py` | `flagquantum/runtime/profiles/scale.py` | entry removed; callers updated |
| runtime | `flagquantum/runtime/distributed/capacity_profile.py` | `flagquantum/runtime/profiles/capacity.py` | entry removed; callers updated |
| core | `flagquantum/core/operator_schema.py` | `flagquantum/core/operators/schema.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/core/algebra/pauli.py` | `flagquantum/core/operators/pauli.py` | entry removed; callers updated |
| core | `flagquantum/operators/registry.py` | `flagquantum/core/operators/registry.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/operators/superoperator.py` | `flagquantum/core/operators/superoperator.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/noise/channels.py` | `flagquantum/core/operators/channels.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/noise/model.py` | `flagquantum/core/noise.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/noise/device_profile.py` | `flagquantum/core/noise_profile.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/observables/__init__.py` | `flagquantum/core/operators/observable.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/observables/fermion.py` | `flagquantum/core/operators/fermion.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/observables/boson.py` | `flagquantum/core/operators/boson.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/observables/__init__.py` | `flagquantum/core/measurement.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/operators/registry.py` | `flagquantum/compiler/construction/operator_registration.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| core | `flagquantum/simulation/density_matrix.py` | `flagquantum/core/operators/embedding.py` | partial: `_basis_bits`, `_bits_to_index`, `_complex_dtype`, `_validate_qubits`, `expand_operator` |
| core | `flagquantum/simulation/entropy.py` | `flagquantum/core/measurement.py` | partial: `log_base_divisor` |
| evolution | `flagquantum/lindblad/__init__.py` | `flagquantum/runtime/evolution/__init__.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| evolution | `flagquantum/lindblad/_plan.py` | `flagquantum/runtime/evolution/_plan.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| evolution | `flagquantum/lindblad/_native_plan.py` | `flagquantum/runtime/evolution/_native_plan.py` | entry removed; callers updated |
| evolution | `flagquantum/lindblad/README.md` | `flagquantum/runtime/evolution/README.md` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/_diagonal_objective.py` | `flagquantum/algorithms/optimization/_diagonal_objective.py` | entry removed; callers updated |
| algorithms | `flagquantum/algorithms/amplitude_estimation.py` | `flagquantum/algorithms/estimation/amplitude_estimation.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/arithmetic.py` | `flagquantum/algorithms/primitives/arithmetic.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/cdr.py` | `flagquantum/algorithms/error_mitigation/cdr.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/chemistry.py` | `flagquantum/algorithms/chemistry/__init__.py` | same import identity; module converted to package |
| algorithms | `flagquantum/algorithms/chemistry_integrals.py` | `flagquantum/algorithms/chemistry/molecular/integrals.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/cobyla.py` | `flagquantum/algorithms/optimizers/cobyla.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/core.py` | `flagquantum/algorithms/primitives/hamiltonian.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/data_encoding.py` | `flagquantum/algorithms/quantum_ml/encoding.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/error_mitigation.py` | `flagquantum/algorithms/error_mitigation/__init__.py` | same import identity; module converted to package |
| algorithms | `flagquantum/algorithms/feature_selection.py` | `flagquantum/algorithms/quantum_ml/feature_selection.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/folding.py` | `flagquantum/algorithms/error_mitigation/folding.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/grover.py` | `flagquantum/algorithms/search/grover.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/kmedians.py` | `flagquantum/algorithms/quantum_ml/kmedians.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/molecular.py` | `flagquantum/algorithms/chemistry/molecular/__init__.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/nelder_mead.py` | `flagquantum/algorithms/optimizers/nelder_mead.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/optimization.py` | `flagquantum/algorithms/optimization/__init__.py` | same import identity; module converted to package |
| algorithms | `flagquantum/algorithms/pca.py` | `flagquantum/algorithms/linear_algebra/pca.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/pec.py` | `flagquantum/algorithms/error_mitigation/pec.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/qarm.py` | `flagquantum/algorithms/quantum_ml/qarm.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/quantum_kernel.py` | `flagquantum/algorithms/quantum_ml/kernels.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/qubo.py` | `flagquantum/algorithms/optimization/qubo.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/readout_mitigation.py` | `flagquantum/algorithms/error_mitigation/readout.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/solvers.py` | `flagquantum/algorithms/optimization/solvers.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/spsa.py` | `flagquantum/algorithms/optimizers/spsa.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/svd.py` | `flagquantum/algorithms/linear_algebra/svd.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/trotter.py` | `flagquantum/algorithms/many_body/trotter.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/variational.py` | `flagquantum/algorithms/optimization/qaoa.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/chemistry_domain/__init__.py` | `flagquantum/algorithms/chemistry/models/__init__.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/chemistry_domain/orbital_rotation.py` | `flagquantum/algorithms/chemistry/models/orbital_rotation.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/chemistry_domain/active_space.py` | `flagquantum/algorithms/chemistry/models/active_space.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/chemistry_domain/problem.py` | `flagquantum/algorithms/chemistry/models/problem.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/models.py` | `flagquantum/algorithms/quantum_ml/models.py` | entry removed; callers updated |
| algorithms | `flagquantum/algorithms/core.py` | `flagquantum/algorithms/primitives/ansatz.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/core.py` | `flagquantum/algorithms/many_body/hva.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/core.py` | `flagquantum/algorithms/optimization/qaoa_circuits.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| algorithms | `flagquantum/algorithms/core.py` | `flagquantum/algorithms/optimization/vqe.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| frontends | `flagquantum/compiler/_hybrid/rewrites.py` | `flagquantum/compiler/frontends/_structured/rewrites.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/transforms.py` | `flagquantum/compiler/frontends/_structured/transforms.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/analysis.py` | `flagquantum/compiler/frontends/_structured/analysis.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/diagnostics.py` | `flagquantum/compiler/frontends/_structured/diagnostics.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/lowering.py` | `flagquantum/compiler/frontends/_structured/lowering.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/__init__.py` | `flagquantum/compiler/frontends/_structured/__init__.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| frontends | `flagquantum/compiler/_hybrid/model.py` | `flagquantum/compiler/frontends/_structured/model.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| frontends | `flagquantum/compiler/_hybrid/capture.py` | `flagquantum/compiler/frontends/_structured/capture.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/dynamic_lowering.py` | `flagquantum/compiler/frontends/_structured/dynamic_lowering.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/schemas.py` | `flagquantum/compiler/frontends/_structured/schemas.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/unitary_effects.py` | `flagquantum/compiler/frontends/_structured/unitary_effects.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/runtime_inputs.py` | `flagquantum/compiler/frontends/_structured/runtime_inputs.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/specialize.py` | `flagquantum/compiler/frontends/_structured/specialize.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/passes.py` | `flagquantum/compiler/frontends/_structured/passes.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/verifier.py` | `flagquantum/compiler/frontends/_structured/verifier.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| frontends | `flagquantum/compiler/_hybrid/dynamic/expressions.py` | `flagquantum/compiler/frontends/_structured/dynamic/expressions.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/dynamic/__init__.py` | `flagquantum/compiler/frontends/_structured/dynamic/__init__.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/dynamic/program.py` | `flagquantum/compiler/frontends/_structured/dynamic/program.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/frontend/diagnostics.py` | `flagquantum/compiler/frontends/_structured/frontend/diagnostics.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/frontend/control.py` | `flagquantum/compiler/frontends/_structured/frontend/control.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/frontend/expressions.py` | `flagquantum/compiler/frontends/_structured/frontend/expressions.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/frontend/__init__.py` | `flagquantum/compiler/frontends/_structured/frontend/__init__.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/frontend/program.py` | `flagquantum/compiler/frontends/_structured/frontend/program.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/frontend/quantum.py` | `flagquantum/compiler/frontends/_structured/frontend/quantum.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/frontend/source.py` | `flagquantum/compiler/frontends/_structured/frontend/source.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_torch/__init__.py` | `flagquantum/compiler/frontends/torch/__init__.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_torch/module_builder.py` | `flagquantum/compiler/frontends/torch/_module_builder.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_torch/training.py` | `flagquantum/compiler/frontends/torch/_training.py` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/IMPLEMENTATION.md` | `flagquantum/compiler/frontends/_structured/IMPLEMENTATION.md` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/README.md` | `flagquantum/compiler/frontends/_structured/README.md` | implementation moved; public or #1084-dependent facade justified in inventory |
| frontends | `flagquantum/compiler/_hybrid/dynamic/README.md` | `flagquantum/compiler/frontends/_structured/dynamic/README.md` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_hybrid/frontend/README.md` | `flagquantum/compiler/frontends/_structured/frontend/README.md` | entry removed; callers updated |
| frontends | `flagquantum/compiler/_torch/README.md` | `flagquantum/compiler/frontends/torch/README.md` | entry removed; callers updated |
| services | `flagquantum/services/synthesis.py` | `flagquantum/synth/_workflow.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| services | `flagquantum/services/_synthesis_registers.py` | `flagquantum/synth/_program.py` | entry removed; callers updated |
| services | `flagquantum/services/_synthesis_readout.py` | `flagquantum/synth/_readout.py` | entry removed; callers updated |
| services | `flagquantum/services/_synthesis_execution.py` | `flagquantum/runtime/planner/synthesis.py` | entry removed; callers updated |
| services | `flagquantum/services/_synthesis_preparation.py` | `flagquantum/compiler/synthesis/programs/preparation.py` | entry removed; callers updated |
| services | `flagquantum/services/_qmod_workflow.py` | `flagquantum/ecosystem/_qmod/workflow.py` | entry removed; callers updated |
| services | `flagquantum/services/ground_state.py` | `flagquantum/algorithms/optimization/ground_state.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| services | `flagquantum/services/research_capabilities.py` | `flagquantum/algorithms/optimization/_research_capabilities.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| services | `flagquantum/services/research_evidence_shape.py` | `flagquantum/algorithms/optimization/_research_evidence_shape.py` | entry removed; callers updated |
| services | `flagquantum/services/research_execution.py` | `flagquantum/algorithms/optimization/_research_execution.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| services | `flagquantum/services/research_options.py` | `flagquantum/algorithms/optimization/_research_options.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| services | `flagquantum/services/research_providers.py` | `flagquantum/algorithms/optimization/_research_providers.py` | entry removed; callers updated |
| services | `flagquantum/services/research_schemas.py` | `flagquantum/algorithms/optimization/_research_schemas.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| services | `flagquantum/services/preflight.py` | `flagquantum/runtime/planner/preflight.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| services | `flagquantum/services/managed_simulator.py` | `flagquantum/runtime/managed_simulator.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| services | `flagquantum/services/preflight.py` | `flagquantum/_api.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| tools/validation | `tools/_qmod_basis_mps.py` | `tools/validation/_qmod_basis_mps.py` | entry removed; callers updated |
| tools/release | `tools/artifact_manifest.py` | `tools/release/artifact_manifest.py` | entry removed; callers updated |
| tools/validation | `tools/audit_flagos_workload_capability.py` | `tools/validation/audit_flagos_workload_capability.py` | entry removed; callers updated |
| tools/validation | `tools/audit_qmod_corpus.py` | `tools/validation/audit_qmod_corpus.py` | entry removed; callers updated |
| tools/validation | `tools/calibrate_performance_crossover.py` | `tools/validation/calibrate_performance_crossover.py` | entry removed; callers updated |
| tools/architecture | `tools/census_wire_vocabulary.py` | `tools/architecture/census_wire_vocabulary.py` | entry removed; callers updated |
| tools/architecture | `tools/check_architecture.py` | `tools/architecture/check_architecture.py` | entry removed; callers updated |
| tools/validation | `tools/check_braket_interop_contract.py` | `tools/validation/check_braket_interop_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_capability_maturity.py` | `tools/validation/check_capability_maturity.py` | entry removed; callers updated |
| tools/validation | `tools/check_circuit_composition_contract.py` | `tools/validation/check_circuit_composition_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_construction_acceptance_contract.py` | `tools/validation/check_construction_acceptance_contract.py` | entry removed; callers updated |
| tools/ci | `tools/check_coverage.py` | `tools/ci/check_coverage.py` | entry removed; callers updated |
| tools/validation | `tools/check_cudaq_export_contract.py` | `tools/validation/check_cudaq_export_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_density_matrix_output_contract.py` | `tools/validation/check_density_matrix_output_contract.py` | entry removed; callers updated |
| tools/docs | `tools/check_docs_links.py` | `tools/docs/check_docs_links.py` | entry removed; callers updated |
| tools/validation | `tools/check_domestic_single_card_contract.py` | `tools/validation/check_domestic_single_card_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_double_single_contract.py` | `tools/validation/check_double_single_contract.py` | entry removed; callers updated |
| tools/release | `tools/check_evidence_retention.py` | `tools/release/check_evidence_retention.py` | entry removed; callers updated |
| tools/release | `tools/check_evidence_revisions.py` | `tools/release/check_evidence_revisions.py` | entry removed; callers updated |
| tools/validation | `tools/check_gradient_methods_contract.py` | `tools/validation/check_gradient_methods_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_import_time.py` | `tools/validation/check_import_time.py` | entry removed; callers updated |
| tools/validation | `tools/check_interop_capability_gap_matrix.py` | `tools/validation/check_interop_capability_gap_matrix.py` | entry removed; callers updated |
| tools/architecture | `tools/check_legacy_root_api_usage.py` | `tools/architecture/check_legacy_root_api_usage.py` | entry removed; callers updated |
| tools/release | `tools/check_main_smoke_qec_wheel.py` | `tools/release/check_main_smoke_qec_wheel.py` | entry removed; callers updated |
| tools/validation | `tools/check_metric_tensor_contract.py` | `tools/validation/check_metric_tensor_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_mps_single_node_certification_environment.py` | `tools/validation/check_mps_single_node_certification_environment.py` | entry removed; callers updated |
| tools/validation | `tools/check_native_gradients_contract.py` | `tools/validation/check_native_gradients_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_opcode_gradient_exactness.py` | `tools/validation/check_opcode_gradient_exactness.py` | entry removed; callers updated |
| tools/validation | `tools/check_openqasm_import_contract.py` | `tools/validation/check_openqasm_import_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_parameter_shift_coverage_contract.py` | `tools/validation/check_parameter_shift_coverage_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_parameter_shift_hessian_contract.py` | `tools/validation/check_parameter_shift_hessian_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_pennylane_interop_contract.py` | `tools/validation/check_pennylane_interop_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_primitives_admission_contract.py` | `tools/validation/check_primitives_admission_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_qec_cudaq_alignment.py` | `tools/validation/check_qec_cudaq_alignment.py` | entry removed; callers updated |
| tools/release | `tools/check_qec_installed_profiles.py` | `tools/release/check_qec_installed_profiles.py` | entry removed; callers updated |
| tools/validation | `tools/check_qec_retirement.py` | `tools/validation/check_qec_retirement.py` | entry removed; callers updated |
| tools/validation | `tools/check_qiskit_interop_contract.py` | `tools/validation/check_qiskit_interop_contract.py` | entry removed; callers updated |
| tools/architecture | `tools/check_qubit_vocabulary.py` | `tools/architecture/check_qubit_vocabulary.py` | entry removed; callers updated |
| tools/validation | `tools/check_realtime_messaging_contract.py` | `tools/validation/check_realtime_messaging_contract.py` | entry removed; callers updated |
| tools/release | `tools/check_release_evidence_environment.py` | `tools/release/check_release_evidence_environment.py` | entry removed; callers updated |
| tools/architecture | `tools/check_repository_hygiene.py` | `tools/architecture/check_repository_hygiene.py` | entry removed; callers updated |
| tools/architecture | `tools/check_repository_language.py` | `tools/architecture/check_repository_language.py` | entry removed; callers updated |
| tools/validation | `tools/check_split_real_imag_contract.py` | `tools/validation/check_split_real_imag_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_split_real_imag_p1_contract.py` | `tools/validation/check_split_real_imag_p1_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_split_real_imag_p2_precision_contract.py` | `tools/validation/check_split_real_imag_p2_precision_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_split_real_imag_p3_double_single_contract.py` | `tools/validation/check_split_real_imag_p3_double_single_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_split_real_imag_p4_device_double_single_contract.py` | `tools/validation/check_split_real_imag_p4_device_double_single_contract.py` | entry removed; callers updated |
| tools/validation | `tools/check_split_real_imag_p5_autograd_optimizer_contract.py` | `tools/validation/check_split_real_imag_p5_autograd_optimizer_contract.py` | entry removed; callers updated |
| tools/architecture | `tools/check_team_scope.py` | `tools/architecture/check_team_scope.py` | entry removed; callers updated |
| tools/validation | `tools/check_vn_entropy_output_contract.py` | `tools/validation/check_vn_entropy_output_contract.py` | entry removed; callers updated |
| tools/ci | `tools/ci_coverage_plan.py` | `tools/ci/ci_coverage_plan.py` | entry removed; callers updated |
| tools/ci | `tools/ci_cpu_budget.py` | `tools/ci/ci_cpu_budget.py` | entry removed; callers updated |
| tools/ci | `tools/ci_pip_wheel_cache.py` | `tools/ci/ci_pip_wheel_cache.py` | entry removed; callers updated |
| tools/ci | `tools/ci_tier.py` | `tools/ci/ci_tier.py` | entry removed; callers updated |
| tools/validation | `tools/collect_ftqc_mixed_continuation.py` | `tools/validation/collect_ftqc_mixed_continuation.py` | entry removed; callers updated |
| tools/validation | `tools/collect_ftqc_phase_cost.py` | `tools/validation/collect_ftqc_phase_cost.py` | entry removed; callers updated |
| tools/validation | `tools/configure_a100_entry_pool.py` | `tools/validation/configure_a100_entry_pool.py` | entry removed; callers updated |
| tools/validation | `tools/convert_q_atlas_twin_evidence.py` | `tools/validation/convert_q_atlas_twin_evidence.py` | entry removed; callers updated |
| tools/validation | `tools/correctness_certification.py` | `tools/validation/correctness_certification.py` | entry removed; callers updated |
| tools/ci | `tools/coverage_map.py` | `tools/ci/coverage_map.py` | entry removed; callers updated |
| tools/validation | `tools/diagnose_flagos_statevector_reverse.py` | `tools/validation/diagnose_flagos_statevector_reverse.py` | entry removed; callers updated |
| tools/docs | `tools/docs_source_of_truth.py` | `tools/docs/docs_source_of_truth.py` | entry removed; callers updated |
| tools/validation | `tools/evaluate_performance_artifact.py` | `tools/validation/evaluate_performance_artifact.py` | entry removed; callers updated |
| tools/release | `tools/evidence_provenance.py` | `tools/release/evidence_provenance.py` | entry removed; callers updated |
| tools/docs | `tools/fetch_example_assets.py` | `tools/docs/fetch_example_assets.py` | entry removed; callers updated |
| tools/validation | `tools/ftqc_monitor_evidence.py` | `tools/validation/ftqc_monitor_evidence.py` | entry removed; callers updated |
| tools/release | `tools/hardware_run_manifest.py` | `tools/release/hardware_run_manifest.py` | entry removed; callers updated |
| tools/validation | `tools/hardware_telemetry.py` | `tools/validation/hardware_telemetry.py` | entry removed; callers updated |
| tools/validation | `tools/materialize_qmod_revision.py` | `tools/validation/materialize_qmod_revision.py` | entry removed; callers updated |
| tools/validation | `tools/multinode_launch_plan.py` | `tools/validation/multinode_launch_plan.py` | entry removed; callers updated |
| tools/validation | `tools/normalize_non_release_benchmarks.py` | `tools/validation/normalize_non_release_benchmarks.py` | entry removed; callers updated |
| tools/validation | `tools/observe_flagos_transport.py` | `tools/validation/observe_flagos_transport.py` | entry removed; callers updated |
| tools/validation | `tools/observe_ftqc_phase_cost.py` | `tools/validation/observe_ftqc_phase_cost.py` | entry removed; callers updated |
| tools/docs | `tools/operator_manifest.py` | `tools/docs/operator_manifest.py` | entry removed; callers updated |
| tools/architecture | `tools/parity_matrix.py` | `tools/architecture/parity_matrix.py` | entry removed; callers updated |
| tools/architecture | `tools/parity_plan_ledger.py` | `tools/architecture/parity_plan_ledger.py` | entry removed; callers updated |
| tools/ci | `tools/partition_failures_against_base.py` | `tools/ci/partition_failures_against_base.py` | entry removed; callers updated |
| tools/ci | `tools/pre_push.py` | `tools/ci/pre_push.py` | entry removed; callers updated |
| tools/validation | `tools/probe_cuda_multinode_mps.py` | `tools/validation/probe_cuda_multinode_mps.py` | entry removed; callers updated |
| tools/validation | `tools/probe_cuda_multinode_statevector.py` | `tools/validation/probe_cuda_multinode_statevector.py` | entry removed; callers updated |
| tools/validation | `tools/probe_cuda_multinode_tn.py` | `tools/validation/probe_cuda_multinode_tn.py` | entry removed; callers updated |
| tools/validation | `tools/probe_cuda_target_capabilities.py` | `tools/validation/probe_cuda_target_capabilities.py` | entry removed; callers updated |
| tools/release | `tools/promote_release_candidates.py` | `tools/release/promote_release_candidates.py` | entry removed; callers updated |
| tools/architecture | `tools/public_api_snapshot.py` | `tools/architecture/public_api_snapshot.py` | entry removed; callers updated |
| tools/ci | `tools/pytest_shard.py` | `tools/ci/pytest_shard.py` | entry removed; callers updated |
| tools/ci | `tools/pytest_stack_watchdog.py` | `tools/ci/pytest_stack_watchdog.py` | entry removed; callers updated |
| tools/release | `tools/qec_installed_smoke.py` | `tools/release/qec_installed_smoke.py` | entry removed; callers updated |
| tools/validation | `tools/run_mps_capacity_series.py` | `tools/validation/run_mps_capacity_series.py` | entry removed; callers updated |
| tools/ci | `tools/run_multinode_watchdog.py` | `tools/ci/run_multinode_watchdog.py` | entry removed; callers updated |
| tools/validation | `tools/run_qmod_catalog.py` | `tools/validation/run_qmod_catalog.py` | entry removed; callers updated |
| tools/validation | `tools/run_qmod_cscope.py` | `tools/validation/run_qmod_cscope.py` | entry removed; callers updated |
| tools/ci | `tools/run_with_watchdog.py` | `tools/ci/run_with_watchdog.py` | entry removed; callers updated |
| tools/release | `tools/runtime_contract_schema.py` | `tools/release/runtime_contract_schema.py` | entry removed; callers updated |
| tools/release | `tools/sanitize_public_evidence.py` | `tools/release/sanitize_public_evidence.py` | entry removed; callers updated |
| tools/release | `tools/seal_runtime_evidence.py` | `tools/release/seal_runtime_evidence.py` | entry removed; callers updated |
| tools/docs | `tools/synthesis_catalog.py` | `tools/docs/synthesis_catalog.py` | entry removed; callers updated |
| tools/validation | `tools/validate_domestic_single_card.py` | `tools/validation/validate_domestic_single_card.py` | entry removed; callers updated |
| tools/validation | `tools/validate_flagos_cuda_reference.py` | `tools/validation/validate_flagos_cuda_reference.py` | entry removed; callers updated |
| tools/validation | `tools/validate_flagos_distributed_conformance.py` | `tools/validation/validate_flagos_distributed_conformance.py` | entry removed; callers updated |
| tools/validation | `tools/validate_flagos_reference_evidence.py` | `tools/validation/validate_flagos_reference_evidence.py` | entry removed; callers updated |
| tools/validation | `tools/validate_flagos_statevector_capacity.py` | `tools/validation/validate_flagos_statevector_capacity.py` | entry removed; callers updated |
| tools/validation | `tools/validate_flagos_statevector_scale.py` | `tools/validation/validate_flagos_statevector_scale.py` | entry removed; callers updated |
| tools/validation | `tools/validate_flagos_statevector_training.py` | `tools/validation/validate_flagos_statevector_training.py` | entry removed; callers updated |
| tools/validation | `tools/validate_native_source_states.py` | `tools/validation/validate_native_source_states.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_binding_scopes.py` | `tools/validation/validate_qmod_binding_scopes.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_controlled_evolution.py` | `tools/validation/validate_qmod_controlled_evolution.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_decision_predicates.py` | `tools/validation/validate_qmod_decision_predicates.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_exact_amplification.py` | `tools/validation/validate_qmod_exact_amplification.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_grover.py` | `tools/validation/validate_qmod_grover.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_integer_composition.py` | `tools/validation/validate_qmod_integer_composition.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_iterations.py` | `tools/validation/validate_qmod_iterations.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_knapsack.py` | `tools/validation/validate_qmod_knapsack.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_large_pauli.py` | `tools/validation/validate_qmod_large_pauli.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_permutations.py` | `tools/validation/validate_qmod_permutations.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_phase_factors.py` | `tools/validation/validate_qmod_phase_factors.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_predicate_oracle.py` | `tools/validation/validate_qmod_predicate_oracle.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_quantum_callbacks.py` | `tools/validation/validate_qmod_quantum_callbacks.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_records.py` | `tools/validation/validate_qmod_records.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_sparse_phase.py` | `tools/validation/validate_qmod_sparse_phase.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_static_qpe.py` | `tools/validation/validate_qmod_static_qpe.py` | entry removed; callers updated |
| tools/validation | `tools/validate_qmod_swap_test.py` | `tools/validation/validate_qmod_swap_test.py` | entry removed; callers updated |
| tools/ci | `tools/validate_required_checks.py` | `tools/ci/validate_required_checks.py` | entry removed; callers updated |
| tools/release | `tools/validate_sbom.py` | `tools/release/validate_sbom.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_device_double_single_evidence.py` | `tools/validation/validate_split_real_imag_device_double_single_evidence.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_device_double_single_flagos.py` | `tools/validation/validate_split_real_imag_device_double_single_flagos.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_double_single_evidence.py` | `tools/validation/validate_split_real_imag_double_single_evidence.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_double_single_flagos.py` | `tools/validation/validate_split_real_imag_double_single_flagos.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_flagos.py` | `tools/validation/validate_split_real_imag_flagos.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_optimizer_accelerator.py` | `tools/validation/validate_split_real_imag_optimizer_accelerator.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_optimizer_evidence.py` | `tools/validation/validate_split_real_imag_optimizer_evidence.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_precision_evidence.py` | `tools/validation/validate_split_real_imag_precision_evidence.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_precision_flagos.py` | `tools/validation/validate_split_real_imag_precision_flagos.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_training_evidence.py` | `tools/validation/validate_split_real_imag_training_evidence.py` | entry removed; callers updated |
| tools/validation | `tools/validate_split_real_imag_training_flagos.py` | `tools/validation/validate_split_real_imag_training_flagos.py` | entry removed; callers updated |
| tools/validation | `tools/verify_embedded_stabilizer_install.py` | `tools/validation/verify_embedded_stabilizer_install.py` | entry removed; callers updated |
| tools/validation | `tools/verify_installed_sparse_blossom.py` | `tools/validation/verify_installed_sparse_blossom.py` | entry removed; callers updated |
| tools/validation | `tools/verify_reproducible_builds.py` | `tools/validation/verify_reproducible_builds.py` | entry removed; callers updated |
| tools/validation | `tools/write_audit_requirements.py` | `tools/validation/write_audit_requirements.py` | entry removed; callers updated |
| repository | `architecture.toml` | `config/architecture.toml` | entry removed; callers updated |
| repository | `team-ownership.toml` | `config/team-ownership.toml` | entry removed; callers updated |
| repository | `environment.yml` | `config/environments/environment.yml` | entry removed; callers updated |
| repository | `workshops/flagos2026/SETUP.md` | `docs/guides/workshops/flagos2026/SETUP.md` | entry removed; callers updated |
| repository | `workshops/flagos2026/CHALLENGES.md` | `docs/guides/workshops/flagos2026/CHALLENGES.md` | entry removed; callers updated |
| repository | `workshops/flagos2026/VALIDATION.md` | `docs/guides/workshops/flagos2026/VALIDATION.md` | entry removed; callers updated |
| repository | `workshops/flagos2026/README.md` | `docs/guides/workshops/flagos2026/README.md` | entry removed; callers updated |
| repository | `workshops/flagos2026/.gitignore` | `examples/workshops/flagos2026/.gitignore` | entry removed; callers updated |
| repository | `workshops/flagos2026/CLOUD_JOBS.md` | `docs/guides/workshops/flagos2026/CLOUD_JOBS.md` | entry removed; callers updated |
| repository | `workshops/flagos2026/INSTRUCTOR.md` | `docs/guides/workshops/flagos2026/INSTRUCTOR.md` | entry removed; callers updated |
| repository | `workshops/flagos2026/FIELD_NOTES.md` | `docs/guides/workshops/flagos2026/FIELD_NOTES.md` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks_cn/04_one_problem_three_simulators.ipynb` | `examples/workshops/flagos2026/notebooks_cn/04_one_problem_three_simulators.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks_cn/05_jiuding_jobs.ipynb` | `examples/workshops/flagos2026/notebooks_cn/05_jiuding_jobs.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks_cn/06_quafu_hardware.ipynb` | `examples/workshops/flagos2026/notebooks_cn/06_quafu_hardware.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks_cn/02_quantum_training.ipynb` | `examples/workshops/flagos2026/notebooks_cn/02_quantum_training.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks_cn/01_first_circuit.ipynb` | `examples/workshops/flagos2026/notebooks_cn/01_first_circuit.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks_cn/03_hybrid_model.ipynb` | `examples/workshops/flagos2026/notebooks_cn/03_hybrid_model.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/basics/02_cpu_gpu.ipynb` | `examples/workshops/flagos2026/reference_notebooks/basics/02_cpu_gpu.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/basics/09_shots.ipynb` | `examples/workshops/flagos2026/reference_notebooks/basics/09_shots.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/training/14_gradients.ipynb` | `examples/workshops/flagos2026/reference_notebooks/training/14_gradients.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/simulation/11_mps.ipynb` | `examples/workshops/flagos2026/reference_notebooks/simulation/11_mps.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/simulation/10_noise.ipynb` | `examples/workshops/flagos2026/reference_notebooks/simulation/10_noise.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/cloud/16_cloud_sweep.ipynb` | `examples/workshops/flagos2026/reference_notebooks/cloud/16_cloud_sweep.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/cloud/06_compare_results.ipynb` | `examples/workshops/flagos2026/reference_notebooks/cloud/06_compare_results.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/cloud/13_trained_hardware.ipynb` | `examples/workshops/flagos2026/reference_notebooks/cloud/13_trained_hardware.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/applications/15_maxcut.ipynb` | `examples/workshops/flagos2026/reference_notebooks/applications/15_maxcut.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/applications/08_vqe.ipynb` | `examples/workshops/flagos2026/reference_notebooks/applications/08_vqe.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/compilation/17_extension.ipynb` | `examples/workshops/flagos2026/reference_notebooks/compilation/17_extension.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_notebooks/compilation/12_compilation.ipynb` | `examples/workshops/flagos2026/reference_notebooks/compilation/12_compilation.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_results/quafu_bell_replay.json` | `examples/workshops/flagos2026/reference_results/quafu_bell_replay.json` | entry removed; callers updated |
| repository | `workshops/flagos2026/reference_results/README.md` | `examples/workshops/flagos2026/reference_results/README.md` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/cloud_lesson.py` | `examples/workshops/flagos2026/scripts/cloud_lesson.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/verify_inline_plots.py` | `examples/workshops/flagos2026/scripts/verify_inline_plots.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/bell.py` | `examples/workshops/flagos2026/scripts/bell.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/workshop_startup.py` | `examples/workshops/flagos2026/scripts/workshop_startup.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/jiuding_job.py` | `examples/workshops/flagos2026/scripts/jiuding_job.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/verify_notebooks.py` | `examples/workshops/flagos2026/scripts/verify_notebooks.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/quafu_job.py` | `examples/workshops/flagos2026/scripts/quafu_job.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/train.py` | `examples/workshops/flagos2026/scripts/train.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/jiuding_paths.py` | `examples/workshops/flagos2026/scripts/jiuding_paths.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/verify_local.py` | `examples/workshops/flagos2026/scripts/verify_local.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/compare.py` | `examples/workshops/flagos2026/scripts/compare.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/scripts/preflight.py` | `examples/workshops/flagos2026/scripts/preflight.py` | entry removed; callers updated |
| repository | `workshops/flagos2026/environment/requirements.txt` | `examples/workshops/flagos2026/environment/requirements.txt` | entry removed; callers updated |
| repository | `workshops/flagos2026/environment/README.md` | `examples/workshops/flagos2026/environment/README.md` | entry removed; callers updated |
| repository | `workshops/flagos2026/assets/README.md` | `examples/workshops/flagos2026/assets/README.md` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks/04_one_problem_three_simulators.ipynb` | `examples/workshops/flagos2026/notebooks/04_one_problem_three_simulators.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks/05_jiuding_jobs.ipynb` | `examples/workshops/flagos2026/notebooks/05_jiuding_jobs.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks/06_quafu_hardware.ipynb` | `examples/workshops/flagos2026/notebooks/06_quafu_hardware.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks/02_quantum_training.ipynb` | `examples/workshops/flagos2026/notebooks/02_quantum_training.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks/01_first_circuit.ipynb` | `examples/workshops/flagos2026/notebooks/01_first_circuit.ipynb` | entry removed; callers updated |
| repository | `workshops/flagos2026/notebooks/03_hybrid_model.ipynb` | `examples/workshops/flagos2026/notebooks/03_hybrid_model.ipynb` | entry removed; callers updated |
| deployment | `flagquantum/deployment/cloud.py` | `flagquantum/compute/remote/packages.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| deployment | `flagquantum/deployment/cloud.py` | `flagquantum/runtime/planner/deployment.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| deployment | `flagquantum/deployment/cloud.py` | `flagquantum/runtime/results/counts.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| deployment | `flagquantum/deployment/routing_evidence.py` | `flagquantum/compiler/artifacts/routing_evidence.py` | entry removed; callers updated |
| deployment | `flagquantum/deployment/dynamic_rematch.py` | `flagquantum/runtime/dynamic/routing_rematch.py` | entry removed; callers updated |
| deployment | `flagquantum/deployment/dynamic_sealing.py` | `flagquantum/runtime/dynamic/remote_sealing.py` | entry removed; callers updated |
| deployment | `flagquantum/deployment/artifact_dry_run.py` | `flagquantum/compute/remote/validation.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/statevector/__init__.py` | `flagquantum/simulation/statevector/jax/__init__.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/statevector/README.md` | `flagquantum/simulation/statevector/jax/README.md` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/statevector/kernels.py` | `flagquantum/simulation/statevector/jax/kernels.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/mps/__init__.py` | `flagquantum/simulation/mps/jax/__init__.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/mps/README.md` | `flagquantum/simulation/mps/jax/README.md` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/mps/kernels.py` | `flagquantum/simulation/mps/jax/kernels.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/mps/batched.py` | `flagquantum/simulation/mps/jax/batched.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/mps/pullbacks.py` | `flagquantum/simulation/mps/jax/pullbacks.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/tensor_network/models.py` | `flagquantum/simulation/tensor_network/jax/models.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/tensor_network/__init__.py` | `flagquantum/simulation/tensor_network/jax/__init__.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/tensor_network/README.md` | `flagquantum/simulation/tensor_network/jax/README.md` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/tensor_network/kernels.py` | `flagquantum/simulation/tensor_network/jax/kernels.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/tensor_network/contraction.py` | `flagquantum/simulation/tensor_network/jax/contraction.py` | entry removed; callers updated |
| simulation/jax | `flagquantum/simulation/jax/primitives.py` | `flagquantum/simulation/numerics/jax.py` | entry removed; callers updated |
| algorithms | `flagquantum/algorithms/core.py` | `flagquantum/algorithms/many_body/spin.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| runtime/telemetry-and-trajectories | `flagquantum/runtime/trajectories/result.py` | `flagquantum/runtime/results/trajectories.py` | implementation moved; public or #1084-dependent facade justified in inventory |
| runtime/telemetry-and-trajectories | `flagquantum/runtime/executors/mps/profiling.py` | `flagquantum/runtime/observability/mps.py` | entry removed; callers updated |
| simulation/numerics | `flagquantum/runtime/executors/statevector/reverse_adjoint_cx.py` | `flagquantum/simulation/statevector/cx_adjoint.py` | partial: `apply_cpu_compact_cx_adjoint_segment`, `apply_cpu_cx_adjoint_index`, `apply_cpu_cx_adjoint_segment`, `cpu_cx_permutation_index` |
| simulation/numerics | `flagquantum/runtime/executors/statevector/forward.py` | `flagquantum/simulation/statevector/adjoint.py` | partial: `_is_diagonal_instruction` |
| simulation/numerics | `flagquantum/runtime/executors/statevector/reverse_adjoint_kernels.py` | `flagquantum/simulation/statevector/adjoint.py` | partial: `_complex_cross`, `_fused_single_process_cpu_reversible_vjp`, `_pair_components`, `_single_process_cpu_pauli_rotation_gradient` |
| simulation/numerics | `flagquantum/runtime/executors/statevector/reverse_observable.py` | `flagquantum/simulation/statevector/adjoint.py` | partial: `seed_adjoint` |
| compiler-interface-retirement | `flagquantum/compiler/module_builder.py` | `flagquantum/compiler/frontends/torch/_module_builder.py` | entry removed; callers updated |
| compiler-interface-retirement | `flagquantum/compiler/training.py` | `flagquantum/compiler/frontends/torch/_training.py` | entry removed; callers updated |
| existing-compiler-alias-retirement | `flagquantum/compiler/basis_translation.py` | `flagquantum/compiler/lowering/basis_translation.py` | retired internal aliases: `_gate` |
| existing-compiler-alias-retirement | `flagquantum/compiler/layout.py` | `flagquantum/compiler/topology/layout.py` | retired internal aliases: `_routing_swap_phases` |
| existing-compiler-alias-retirement | `flagquantum/compiler/noise.py` | `flagquantum/compiler/lowering/noise.py` | retired internal aliases: `_encode_channel_instruction` |
| existing-compiler-alias-retirement | `flagquantum/compiler/openqasm.py` | `flagquantum/compiler/formats/openqasm.py` | retired internal aliases: `_format_number` |
| existing-compiler-alias-retirement | `flagquantum/compiler/openqasm_import.py` | `flagquantum/compiler/formats/openqasm_import.py` | retired internal aliases: `_ISSUE_CODES` |
| existing-compiler-alias-retirement | `flagquantum/compiler/pipeline.py` | `flagquantum/compiler/optimization/canonicalization.py` | retired internal aliases: `_ROTATION_PARAM`, `_SELF_INVERSE`, `_WireLocalProgram`, `_add_values`, `_is_zero`, `_replace_param`, `_rotation_table`, `_self_inverse_table` |
| existing-compiler-alias-retirement | `flagquantum/compiler/pipeline.py` | `flagquantum/compiler/optimization/pass_registry.py` | retired internal aliases: `_BUILTIN_ANALYSES`, `_BUILTIN_PASS_SPECS`, `_cancel_commuting_self_inverse`, `_collapse_one_qubit_runs`, `_collapse_two_qubit_blocks`, `_commutation_blocks`, `_merge_commuting_rotations`, `_merge_inverse_pairs`, `_remove_diagonal_gates_before_measure`, `_rotation_parameters`, `_self_inverse_opcodes` |
| existing-compiler-alias-retirement | `flagquantum/compiler/pipeline.py` | `flagquantum/compiler/optimization/pipeline.py` | retired internal aliases: `_as_ir`, `_with_recorded_level` |
| existing-compiler-alias-retirement | `flagquantum/compiler/qcis.py` | `flagquantum/compiler/formats/qcis.py` | retired internal aliases: `_decompose` |
| existing-compiler-alias-retirement | `flagquantum/compiler/resource_estimation.py` | `flagquantum/compiler/optimization/resource_estimation.py` | retired internal aliases: `_T_FAMILY_OPCODES` |
| existing-compiler-alias-retirement | `flagquantum/compiler/routing.py` | `flagquantum/compiler/topology/routing.py` | retired internal aliases: `_routing_metadata` |
| existing-compiler-alias-retirement | `flagquantum/compiler/pipeline.py` | `flagquantum/compiler/optimization/pipeline.py` | retired unused wrapper: `_registry_admitting` |
| examples | `examples/absorptive_photon_counting.py` | `examples/simulation/absorptive_photon_counting.py` | entry removed; callers updated |
| examples | `examples/ascend_910c_training.py` | `examples/training/ascend_910c_training.py` | entry removed; callers updated |
| examples | `examples/benchmark_static_program.py` | `examples/training/benchmark_static_program.py` | entry removed; callers updated |
| examples | `examples/boson_interference.py` | `examples/simulation/boson_interference.py` | entry removed; callers updated |
| examples | `examples/carried_operand_execution.py` | `examples/simulation/carried_operand_execution.py` | entry removed; callers updated |
| examples | `examples/compiler_optimize.py` | `examples/circuits/compiler/compiler_optimize.py` | entry removed; callers updated |
| examples | `examples/compiler_synthesis.py` | `examples/circuits/compiler/compiler_synthesis.py` | entry removed; callers updated |
| examples | `examples/constrained_basis_reference.py` | `examples/simulation/constrained_basis_reference.py` | entry removed; callers updated |
| examples | `examples/cpu_statevector.py` | `examples/simulation/cpu_statevector.py` | entry removed; callers updated |
| examples | `examples/cudaq_bell_export.py` | `examples/ecosystem/cudaq_bell_export.py` | entry removed; callers updated |
| examples | `examples/density_matrix_execution.py` | `examples/simulation/density_matrix_execution.py` | entry removed; callers updated |
| examples | `examples/driven_three_level.py` | `examples/simulation/driven_three_level.py` | entry removed; callers updated |
| examples | `examples/driven_three_level_relaxation.py` | `examples/simulation/driven_three_level_relaxation.py` | entry removed; callers updated |
| examples | `examples/dynamic_density_matrix.py` | `examples/simulation/dynamic_density_matrix.py` | entry removed; callers updated |
| examples | `examples/dynamic_disjunctive_condition.py` | `examples/circuits/dynamic_disjunctive_condition.py` | entry removed; callers updated |
| examples | `examples/gate_coverage_batch_b.py` | `examples/circuits/gate_coverage_batch_b.py` | entry removed; callers updated |
| examples | `examples/hybrid_jax_torch_training.py` | `examples/training/hybrid_jax_torch_training.py` | entry removed; callers updated |
| examples | `examples/hybrid_model_acceptance.py` | `examples/training/hybrid_model_acceptance.py` | entry removed; callers updated |
| examples | `examples/ising_braid_execution.py` | `examples/simulation/ising_braid_execution.py` | entry removed; callers updated |
| examples | `examples/ising_braid_source.py` | `examples/circuits/ising_braid_source.py` | entry removed; callers updated |
| examples | `examples/ising_fusion_measurement.py` | `examples/simulation/ising_fusion_measurement.py` | entry removed; callers updated |
| examples | `examples/kaiwu_matrix_boundary.py` | `examples/compute/kaiwu_matrix_boundary.py` | entry removed; callers updated |
| examples | `examples/lindblad_evolution.py` | `examples/simulation/lindblad_evolution.py` | entry removed; callers updated |
| examples | `examples/lindblad_output_names.py` | `examples/simulation/lindblad_output_names.py` | entry removed; callers updated |
| examples | `examples/matrix_expectation_dynamics.py` | `examples/simulation/matrix_expectation_dynamics.py` | entry removed; callers updated |
| examples | `examples/mixed_dimension_dynamics.py` | `examples/simulation/mixed_dimension_dynamics.py` | entry removed; callers updated |
| examples | `examples/neutral_atom_pair_reference.py` | `examples/simulation/neutral_atom_pair_reference.py` | entry removed; callers updated |
| examples | `examples/noisy_simulation_v1.py` | `examples/simulation/noisy_simulation_v1.py` | entry removed; callers updated |
| examples | `examples/quantum_transformer.py` | `examples/training/quantum_transformer.py` | entry removed; callers updated |
| examples | `examples/quick_start.py` | `examples/training/quick_start.py` | entry removed; callers updated |
| examples | `examples/routed_artifact_preflight.py` | `examples/circuits/compiler/routed_artifact_preflight.py` | entry removed; callers updated |
| examples | `examples/rydberg_native_dynamics.py` | `examples/simulation/rydberg_native_dynamics.py` | entry removed; callers updated |
| examples | `examples/rydberg_program_geometry.py` | `examples/circuits/rydberg_program_geometry.py` | entry removed; callers updated |
| examples | `examples/save_experiment.py` | `examples/training/save_experiment.py` | entry removed; callers updated |
| examples | `examples/shared_dynamics_workflow.py` | `examples/simulation/shared_dynamics_workflow.py` | entry removed; callers updated |
| examples | `examples/spin_mode_convergence.py` | `examples/simulation/spin_mode_convergence.py` | entry removed; callers updated |
| examples | `examples/spin_mode_thermal_bath.py` | `examples/simulation/spin_mode_thermal_bath.py` | entry removed; callers updated |
| examples | `examples/stabilizer_sampling.py` | `examples/simulation/stabilizer_sampling.py` | entry removed; callers updated |
| examples | `examples/superoperator_channel_algebra.py` | `examples/simulation/superoperator_channel_algebra.py` | entry removed; callers updated |
| examples | `examples/synthesis_offline_deployment.py` | `examples/synthesis/synthesis_offline_deployment.py` | entry removed; callers updated |
| examples | `examples/target_aware_compilation.py` | `examples/circuits/compiler/target_aware_compilation.py` | entry removed; callers updated |
| examples | `examples/terminal_instrument_reference.py` | `examples/simulation/terminal_instrument_reference.py` | entry removed; callers updated |
| examples | `examples/terminal_instrument_sampling.py` | `examples/simulation/terminal_instrument_sampling.py` | entry removed; callers updated |
| examples | `examples/train_boson_coupling.py` | `examples/training/train_boson_coupling.py` | entry removed; callers updated |
| examples | `examples/train_parameterized_circuit_then_deploy.py` | `examples/training/train_parameterized_circuit_then_deploy.py` | entry removed; callers updated |
| examples | `examples/train_piecewise_drive.py` | `examples/training/train_piecewise_drive.py` | entry removed; callers updated |
| examples | `examples/train_rydberg_controls.py` | `examples/training/train_rydberg_controls.py` | entry removed; callers updated |
| examples | `examples/triton_statevector_local_1q.py` | `examples/simulation/triton_statevector_local_1q.py` | entry removed; callers updated |
| examples | `examples/twin_calibration_history.py` | `examples/twin/twin_calibration_history.py` | entry removed; callers updated |
| examples | `examples/twin_candidate_fit.py` | `examples/twin/twin_candidate_fit.py` | entry removed; callers updated |
| examples | `examples/twin_circuit_support.py` | `examples/twin/twin_circuit_support.py` | entry removed; callers updated |
| examples | `examples/twin_connected_region.py` | `examples/twin/twin_connected_region.py` | entry removed; callers updated |
| examples | `examples/twin_evidence_freshness.py` | `examples/twin/twin_evidence_freshness.py` | entry removed; callers updated |
| examples | `examples/twin_evolution_cycle.py` | `examples/twin/twin_evolution_cycle.py` | entry removed; callers updated |
| examples | `examples/twin_evolution_history.py` | `examples/twin/twin_evolution_history.py` | entry removed; callers updated |
| examples | `examples/twin_parameter_estimation.py` | `examples/twin/twin_parameter_estimation.py` | entry removed; callers updated |
| examples | `examples/twin_region_model.py` | `examples/twin/twin_region_model.py` | entry removed; callers updated |
| examples | `examples/twin_region_model_persistence.py` | `examples/twin/twin_region_model_persistence.py` | entry removed; callers updated |
| examples | `examples/twin_region_release_assessment.py` | `examples/twin/twin_region_release_assessment.py` | entry removed; callers updated |
| examples | `examples/twin_validation_history.py` | `examples/twin/twin_validation_history.py` | entry removed; callers updated |
| examples | `examples/visualize_rydberg_transport.py` | `examples/simulation/visualize_rydberg_transport.py` | entry removed; callers updated |
| examples | `examples/vqe_switch_sv_mps_tn.py` | `examples/training/vqe_switch_sv_mps_tn.py` | entry removed; callers updated |
| examples | `examples/weighted_routing.py` | `examples/circuits/compiler/weighted_routing.py` | entry removed; callers updated |
| matrix-free-numerics | `flagquantum/simulation/matrix_free_autograd.py` | `flagquantum/simulation/numerics/matrix_free/autograd.py` | entry removed; callers updated |
| matrix-free-numerics | `flagquantum/simulation/matrix_free_exponential.py` | `flagquantum/simulation/numerics/matrix_free/exponential.py` | entry removed; callers updated |
| matrix-free-numerics | `flagquantum/simulation/matrix_free_hamiltonian.py` | `flagquantum/simulation/numerics/matrix_free/hamiltonian.py` | two proven public-plan type locators only |
| matrix-free-numerics | `flagquantum/simulation/matrix_free_linear_solve.py` | `flagquantum/simulation/numerics/matrix_free/linear_solve.py` | entry removed; callers updated |
| physical-dynamics | `flagquantum/simulation/rydberg_model.py` | `flagquantum/simulation/dynamics/rydberg.py` | entry removed; callers updated |
| physical-dynamics | `flagquantum/simulation/rydberg_presence.py` | `flagquantum/simulation/dynamics/rydberg_presence.py` | entry removed; callers updated |
| physical-dynamics | `flagquantum/simulation/ising_braid.py` | `flagquantum/simulation/dynamics/ising_braid.py` | entry removed; callers updated |
| shared-kernel-primitives | `flagquantum/simulation/kernel_dispatch.py` | `flagquantum/simulation/kernels/dispatch.py` | entry removed; callers updated |
| shared-kernel-primitives | `flagquantum/simulation/complex_bmm_dispatch.py` | `flagquantum/simulation/kernels/complex_bmm.py` | entry removed; callers updated |
| shared-kernel-primitives | `flagquantum/simulation/real_imag_kernels.py` | `flagquantum/simulation/kernels/real_imag.py` | entry removed; callers updated |
| experimental-owner-closure | `flagquantum/experimental/controlled_pauli.py` | `flagquantum/core/operators/pauli.py` | `decode_controlled_pauli`; public experimental facade only |
| experimental-owner-closure | `flagquantum/experimental/controlled_pauli.py` | `flagquantum/simulation/pauli.py` | `controlled_pauli`; public experimental facade only |
| experimental-owner-closure | `flagquantum/experimental/qec.py` | `flagquantum/qec/execution/experiments.py` | `run_stim_matching_experiment`; public experimental facade only |
| shared-gradient-numerics | `flagquantum/core/finite_differences.py` | `flagquantum/simulation/numerics/finite_differences.py` | internal file removed; no compatibility |
| runtime-state-numerics | `flagquantum/runtime/dynamic/execution.py` | `flagquantum/simulation/statevector/measurement_ops.py` | `_measure_wire`, `_measure_wires_batched`; no compatibility |
| runtime-state-numerics | `flagquantum/runtime/dynamic/execution.py` | `flagquantum/simulation/statevector/instruction.py` | `_apply_instruction`; no compatibility |
| density-instrument-numerics | `flagquantum/runtime/measurements.py` | `flagquantum/simulation/density_matrix.py` | `_draw_probability_indices`, `_DensityMatrixTarget`; no compatibility |
| density-instrument-numerics | `flagquantum/runtime/_terminal_instruments.py` | `flagquantum/simulation/density_matrix.py` | `_instrument_probabilities`; no compatibility |
| dense-probability-numerics | `flagquantum/runtime/measurements.py` | `flagquantum/simulation/numerics/probabilities.py` | `_reduced_joint_marginal`, `_normalize_joint_marginal`, `_ideal_probabilities`, `_probabilities_from_parity_expectations`; unchanged `_TRACE_TOLERANCE`; no compatibility |
| dynamics-normalizer-retirement | `flagquantum/simulation/dynamics/lindblad.py` | `flagquantum/simulation/dynamics/problem.py` | partial: 15 private normalizer reexports removed; callers use actual owner; seven public evolution exports unchanged |
| simulation-private-binding-retirement | `flagquantum/simulation/stabilizer/engine.py` | `flagquantum/simulation/stabilizer/_circuit_text.py` | partial: retire `_annotated_stim_source`, `_annotated_stim_text` imported aliases; definitions unchanged |
| simulation-private-binding-retirement | `flagquantum/simulation/statevector/local.py` | `flagquantum/simulation/statevector/measurement_ops.py` | partial: retire `_expectation_from_operators`, `_expectation_pauli_string`, `_sample_statevector` imported aliases; definitions unchanged |
| simulation-private-binding-retirement | `flagquantum/simulation/statevector/operations.py` | `flagquantum/simulation/statevector/dense_fusion_cpu.py` | partial: retire `_apply_swap_sequence`, `_cpu_cx_rzz_swap_fusion_enabled`, `_cpu_swap_sequence_fusion_enabled` imported aliases; definitions unchanged |
| simulation-private-binding-retirement | `flagquantum/simulation/statevector/operations.py` | `flagquantum/simulation/statevector/diagonal_cpu.py` | partial: retire `_apply_cross_wire_diagonal_cpu`, `_apply_disjoint_diagonal_regions_cpu` imported aliases; definitions unchanged |
| simulation-private-binding-retirement | `flagquantum/simulation/statevector/operations.py` | `flagquantum/simulation/statevector/diagonal_matrix_dispatch.py` | partial: retire `_apply_diagonal_matrix`, `_apply_diagonal_matrix_reference` imported aliases; definitions unchanged |
| simulation-private-binding-retirement | `flagquantum/simulation/statevector/operations.py` | `flagquantum/simulation/statevector/program.py` | partial: retire `_StatevectorCXSequenceRZZSwapStep`, `_StatevectorSwapSequenceStep` imported aliases; definitions unchanged |
| qec-private-binding-retirement | `flagquantum/qec/codes/__init__.py` | `flagquantum/qec/codes/_css_helpers.py` | partial: retire `_DISTANCE_SEARCH_WEIGHT` imported aliases; frozen QEC bindings unchanged |
| qec-private-binding-retirement | `flagquantum/qec/codes/__init__.py` | `flagquantum/qec/codes/steane.py` | partial: retire `_STEANE_CHECK_SUPPORTS`, `_STEANE_LOGICAL_SUPPORT` imported aliases; frozen QEC bindings unchanged |
| qec-private-binding-retirement | `flagquantum/qec/dem/construction.py` | `flagquantum/qec/codes/matrices.py` | partial: retire `_binary_matrix` imported aliases; frozen QEC bindings unchanged |
| qec-private-binding-retirement | `flagquantum/qec/dem/construction.py` | `flagquantum/qec/dem/matrices.py` | partial: retire `_code_matrix_entries`, `_column_supports` imported aliases; frozen QEC bindings unchanged |
| qec-private-binding-retirement | `flagquantum/qec/dem/construction.py` | `flagquantum/qec/execution/signatures.py` | partial: retire `_FAULT_KIND_BY_LETTER` imported aliases; frozen QEC bindings unchanged |
| qec-private-binding-retirement | `flagquantum/qec/dem/matrices.py` | `flagquantum/qec/codes/matrices.py` | partial: retire `_binary_matrix` imported aliases; frozen QEC bindings unchanged |
| qec-private-binding-retirement | `flagquantum/qec/execution/signatures.py` | `flagquantum/simulation/stabilizer/_fault_signatures.py` | partial: retire `_pauli_channel_terms` imported aliases; frozen QEC bindings unchanged |
| compiler-runtime-private-binding-retirement | `flagquantum/compiler/topology/routing.py` | `flagquantum/compiler/lowering/operand_semantics.py` | partial: retire `_require_multi_qubit_device_local` imported aliases; definitions unchanged |
| compiler-runtime-private-binding-retirement | `flagquantum/compiler/topology/routing.py` | `flagquantum/compiler/topology/routing_cost.py` | partial: retire `_TopologyCostModel`, `_count_sabre_orientation_swaps`, `_source_routing_counts` imported aliases; definitions unchanged |
| compiler-runtime-private-binding-retirement | `flagquantum/compiler/topology/routing.py` | `flagquantum/compiler/topology/routing_emission.py` | partial: retire `_remap_instruction`, `_routing_metadata`, `_swap_instruction` imported aliases; definitions unchanged |
| compiler-runtime-private-binding-retirement | `flagquantum/compiler/topology/routing.py` | `flagquantum/compiler/topology/routing_strategy.py` | partial: retire `_estimate_routing_cost`, `_estimate_swap_plan`, `_plan_sabre_for_topology` imported aliases; definitions unchanged |
| compiler-runtime-private-binding-retirement | `flagquantum/compiler/topology/routing.py` | `flagquantum/compiler/topology/sabre.py` | partial: retire `_plan_sabre_trial`, `_select_sabre_trials` imported aliases; definitions unchanged |
| compiler-runtime-private-binding-retirement | `flagquantum/compiler/topology/routing_cost.py` | `flagquantum/compiler/topology/link_error_rates.py` | partial: retire `_quantized` imported aliases; definitions unchanged |
| compiler-runtime-private-binding-retirement | `flagquantum/compiler/topology/sabre.py` | `flagquantum/compiler/topology/layout_restore.py` | partial: retire `_restore_path` imported aliases; definitions unchanged |
| compiler-runtime-private-binding-retirement | `flagquantum/runtime/dynamic/__init__.py` | `flagquantum/runtime/dynamic/_conditions.py` | partial: retire `_classical_width`, `_instruction_conditions` imported aliases; definitions unchanged |

| runtime-readout-sampling-numerics | `flagquantum/runtime/measurements.py` | `flagquantum/simulation/numerics/probabilities.py` | partial: Pauli probability clamp/normalization, draw and bit reconstruction; existing orchestration/public method remains |
| runtime-readout-sampling-numerics | `flagquantum/runtime/dynamic/execution.py` | `flagquantum/simulation/statevector/measurement_ops.py` | partial: Single and batched final trajectory draws; existing orchestration/public method remains |
| runtime-readout-sampling-numerics | `flagquantum/runtime/results/trajectories.py` | `flagquantum/simulation/mps/sampling.py` | partial: Empirical retained noisy-MPS mixture sampling; existing orchestration/public method remains |

| runtime-observable-numerics | `flagquantum/runtime/executors/jax/statevector/records.py` | `flagquantum/simulation/numerics/probabilities.py` | partial: Raw probabilities and ordered dense Z readout; public result/cache/validation remain |
| runtime-observable-numerics | `flagquantum/runtime/executors/jax/tensor_network/records.py` | `flagquantum/simulation/numerics/probabilities.py` | partial: Same raw dense readout after TensorNetwork reconstruction; public result identity remains |
| runtime-observable-numerics | `flagquantum/runtime/_ising_plan.py` | `flagquantum/simulation/dynamics/ising_braid.py` | partial: Physical Ising output contraction only; admission/device/result policy remains |
| runtime-observable-numerics | `flagquantum/runtime/_rydberg_presence.py` | `flagquantum/simulation/dynamics/rydberg_presence.py` | partial: Physical presence trajectory output contraction only; replay/RNG/result policy remains |
| runtime-observable-numerics | `flagquantum/runtime/executors/statevector/reverse_adjoint.py` | `flagquantum/simulation/statevector/adjoint.py` | partial: Existing native/eager scalar evaluation; weight-cache admission/ownership remains Runtime |
| runtime-observable-numerics | `flagquantum/runtime/executors/statevector/reverse_adjoint.py` | `flagquantum/simulation/native_cpu/adjoint.py` | partial: retire only the old imported native function binding; published provider/definition unchanged |

| runtime-mps-local-numerics | `flagquantum/runtime/executors/jax/mps/backward.py` | `flagquantum/simulation/mps/jax/pullbacks.py` | partial: Actual RY sites, local Z contractions and local derivative computation; Runtime keeps rank exchange/timing |
| runtime-mps-local-numerics | `flagquantum/runtime/executors/jax/mps/canonicalization.py` | `flagquantum/simulation/mps/jax/pullbacks.py` | partial: Direct RY Z/derivative for actual owner-rank optimizer updates; Runtime keeps SGD and collective |
| runtime-mps-local-numerics | `flagquantum/runtime/executors/jax/mps/pullbacks.py` | `flagquantum/simulation/mps/jax/pullbacks.py` | partial: Actual local scores that feed received VJP cotangents; evidence-only tensor allocations stay Runtime |

The [Simulation direct-relay retirement](simulation-direct-relay-retirement.md) removes seven redundant internal definitions. Their callers use the existing local, observable, and native permutation owners directly; the surviving tensor-network modules remain authoritative for their other operations.

## Development Compose location

| Group | Previous path | Authoritative path | Scope |
| --- | --- | --- | --- |
| development-compose-location | `compose.dev.yaml` | `docker/compose.dev.yaml` | Unchanged YAML; old root file removed, explicit repository project directory preserves all relative-path and project behavior; workflow filters and ownership synchronized. |

## Root policy configuration

| Group | Previous path | Authoritative path | Scope |
| --- | --- | --- | --- |
| root-policy-configuration | `capability-maturity.toml` | `config/capability-maturity.toml` | Unchanged data, actual readers updated, root file removed; versioned authority names and explicit custom declarations paths preserved. |
| root-policy-configuration | `evidence-revision-origins.toml` | `config/evidence-revision-origins.toml` | Unchanged data, actual readers updated, root file removed; versioned authority names and explicit custom declarations paths preserved. |

## CI data location

| Group | Previous path | Authoritative path | Scope |
| --- | --- | --- | --- |
| ci-data-location | `ci/pr-gate-main.json` | `tools/ci/pr-gate-main.json` | Unchanged data, actual readers synchronized, root file removed; historical contract and evidence preserved. |
| ci-data-location | `ci/flagos_cuda_reference.lock.json` | `tools/ci/flagos_cuda_reference.lock.json` | Unchanged data, actual readers synchronized, root file removed; historical contract and evidence preserved. |
| ci-data-location | `ci/domestic_single_card_attestation.template.json` | `tools/ci/domestic_single_card_attestation.template.json` | Unchanged data, actual readers synchronized, root file removed; historical contract and evidence preserved. |

## Dependency policy location

| Group | Previous path | Authoritative path | Scope |
| --- | --- | --- | --- |
| dependency-policy-location | `dependency-policy.toml` | `config/dependency-policy.toml` | Unchanged data, actual defaults/cache/filter updated, root file removed; frozen contracts, diagnostics, historical records and explicit custom paths preserved. |
