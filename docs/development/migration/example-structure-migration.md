# Example structure migration

This batch builds on retirement commit `471a2b938be1b6dd46078e433b307cbfd5bb7239`.
The complete #1084 source tree audited for this batch is
`11fd9fa27ec31e74f03277010762e8d25d17c105` (open draft). Its base-to-head
inventory contains 4,464 changed paths without API truncation.

## Ownership and compatibility

The 63 root Python examples move into the existing capability categories. Training
owns the hybrid quick start, Transformer and reusable-program training benchmark;
Circuits owns construction/compilation examples; Ecosystem owns the CUDA-Q
interchange example. Local numerical examples live in Simulation, calibration
examples in Twin, and resource adapters in Compute. These are repository examples,
not a new installed product API. No old example forwarding files are retained.

Existing callers, module commands, relative imports, notebooks and contributor
links use the new paths. Five existing source-root bootstraps use the new file
depth. A separate subprocess verifies the quick start outside the repository and
asserts that it imports this source tree, so an older editable installation cannot
mask an incorrect bootstrap.

Frozen capability/parity manifests and raw evidence stay unchanged. Their
historical identifiers resolve through the existing relocation ledger. The
documentation renderer uses that same mapping to link to current executable
examples. It does not change capability levels, numerical results, API signatures
or evidence hashes. The package manifest already prunes examples from both product
artifacts; no installed FlagQuantum module is moved in this batch.

## #1084 integration boundary

None of these 63 source scripts is changed by the audited #1084 diff, and none of
the target paths is occupied in its complete tree. The full consumer audit reads 4,376 changed source, configuration, fixture and
documentation texts, with every Git blob SHA verified. Only seven files reference
these old examples: `.github/workflows/ci.yml`, root `README.md`,
`docs/development/TESTING.md`, `examples/README.md`, and the Compiler, Core and
Simulation READMEs. The workflow reference is the existing CUDA-Q smoke module;
there are no newly introduced production or test import consumers of the moved
examples. The current-tree reference updates cover the corresponding files.
At integration preserve incoming content and apply the same path/module updates;
a prepared patch against the audited incoming blobs is retained with the working
audit artifacts. A later #1084 head must repeat this audit rather than assuming
path nonintersection proves independence. This batch does not remove the three retained
#1084 frontend integration files or relocate its changed implementation families.

## Stacked PR validation

This batch can target the open retirement branch while its main merge remains a
separate human action. CI previously matched main and the QEC integration family
only. The branch selector now also includes the existing retirement branch and
`refactor/structure-*`, so stacked architecture PRs receive the ordinary PR
checks rather than an empty check list. The event types, fork/host routing,
permissions, capacity reservations and test selections are unchanged. Confirm
that the submitted PR actually starts CI; a matching configuration alone is not
remote validation.

## Validation recorded in the source preview

- A tracked-source export was transformed before updating the active branch.
- All 63 normalized algorithm ASTs match; the normalization permits imports,
  documentation, physical source paths and repository-root depth only.
- The affected example tests: 385 passed, 6 skipped; existing dependency/hardware
  exclusions remain unchanged. No tolerance or expected numerical value changed.
- The additional outside-repository quick start passes and checks source identity.
- Historical quick-start document link tests pass; all Markdown links resolve.
- Relocation ownership and retired-import guards report no errors.
- Generated claims were rendered using the exported snapshot and its real source
  repository's read-only Git history. A plain source export cannot verify revision
  provenance without that history; missing history is not treated as success.

This preview is preparation, not remote CI acceptance. The submitted branch must
pass its own required checks. The complete architecture migration remains open
for the #1084-dependent implementation, distribution and policy closures.

## Source mapping

| Previous path | Current path |
| --- | --- |
| `examples/absorptive_photon_counting.py` | `examples/simulation/absorptive_photon_counting.py` |
| `examples/ascend_910c_training.py` | `examples/training/ascend_910c_training.py` |
| `examples/benchmark_static_program.py` | `examples/training/benchmark_static_program.py` |
| `examples/boson_interference.py` | `examples/simulation/boson_interference.py` |
| `examples/carried_operand_execution.py` | `examples/simulation/carried_operand_execution.py` |
| `examples/compiler_optimize.py` | `examples/circuits/compiler/compiler_optimize.py` |
| `examples/compiler_synthesis.py` | `examples/circuits/compiler/compiler_synthesis.py` |
| `examples/constrained_basis_reference.py` | `examples/simulation/constrained_basis_reference.py` |
| `examples/cpu_statevector.py` | `examples/simulation/cpu_statevector.py` |
| `examples/cudaq_bell_export.py` | `examples/ecosystem/cudaq_bell_export.py` |
| `examples/density_matrix_execution.py` | `examples/simulation/density_matrix_execution.py` |
| `examples/driven_three_level.py` | `examples/simulation/driven_three_level.py` |
| `examples/driven_three_level_relaxation.py` | `examples/simulation/driven_three_level_relaxation.py` |
| `examples/dynamic_density_matrix.py` | `examples/simulation/dynamic_density_matrix.py` |
| `examples/dynamic_disjunctive_condition.py` | `examples/circuits/dynamic_disjunctive_condition.py` |
| `examples/gate_coverage_batch_b.py` | `examples/circuits/gate_coverage_batch_b.py` |
| `examples/hybrid_jax_torch_training.py` | `examples/training/hybrid_jax_torch_training.py` |
| `examples/hybrid_model_acceptance.py` | `examples/training/hybrid_model_acceptance.py` |
| `examples/ising_braid_execution.py` | `examples/simulation/ising_braid_execution.py` |
| `examples/ising_braid_source.py` | `examples/circuits/ising_braid_source.py` |
| `examples/ising_fusion_measurement.py` | `examples/simulation/ising_fusion_measurement.py` |
| `examples/kaiwu_matrix_boundary.py` | `examples/compute/kaiwu_matrix_boundary.py` |
| `examples/lindblad_evolution.py` | `examples/simulation/lindblad_evolution.py` |
| `examples/lindblad_output_names.py` | `examples/simulation/lindblad_output_names.py` |
| `examples/matrix_expectation_dynamics.py` | `examples/simulation/matrix_expectation_dynamics.py` |
| `examples/mixed_dimension_dynamics.py` | `examples/simulation/mixed_dimension_dynamics.py` |
| `examples/neutral_atom_pair_reference.py` | `examples/simulation/neutral_atom_pair_reference.py` |
| `examples/noisy_simulation_v1.py` | `examples/simulation/noisy_simulation_v1.py` |
| `examples/quantum_transformer.py` | `examples/training/quantum_transformer.py` |
| `examples/quick_start.py` | `examples/training/quick_start.py` |
| `examples/routed_artifact_preflight.py` | `examples/circuits/compiler/routed_artifact_preflight.py` |
| `examples/rydberg_native_dynamics.py` | `examples/simulation/rydberg_native_dynamics.py` |
| `examples/rydberg_program_geometry.py` | `examples/circuits/rydberg_program_geometry.py` |
| `examples/save_experiment.py` | `examples/training/save_experiment.py` |
| `examples/shared_dynamics_workflow.py` | `examples/simulation/shared_dynamics_workflow.py` |
| `examples/spin_mode_convergence.py` | `examples/simulation/spin_mode_convergence.py` |
| `examples/spin_mode_thermal_bath.py` | `examples/simulation/spin_mode_thermal_bath.py` |
| `examples/stabilizer_sampling.py` | `examples/simulation/stabilizer_sampling.py` |
| `examples/superoperator_channel_algebra.py` | `examples/simulation/superoperator_channel_algebra.py` |
| `examples/synthesis_offline_deployment.py` | `examples/synthesis/synthesis_offline_deployment.py` |
| `examples/target_aware_compilation.py` | `examples/circuits/compiler/target_aware_compilation.py` |
| `examples/terminal_instrument_reference.py` | `examples/simulation/terminal_instrument_reference.py` |
| `examples/terminal_instrument_sampling.py` | `examples/simulation/terminal_instrument_sampling.py` |
| `examples/train_boson_coupling.py` | `examples/training/train_boson_coupling.py` |
| `examples/train_parameterized_circuit_then_deploy.py` | `examples/training/train_parameterized_circuit_then_deploy.py` |
| `examples/train_piecewise_drive.py` | `examples/training/train_piecewise_drive.py` |
| `examples/train_rydberg_controls.py` | `examples/training/train_rydberg_controls.py` |
| `examples/triton_statevector_local_1q.py` | `examples/simulation/triton_statevector_local_1q.py` |
| `examples/twin_calibration_history.py` | `examples/twin/twin_calibration_history.py` |
| `examples/twin_candidate_fit.py` | `examples/twin/twin_candidate_fit.py` |
| `examples/twin_circuit_support.py` | `examples/twin/twin_circuit_support.py` |
| `examples/twin_connected_region.py` | `examples/twin/twin_connected_region.py` |
| `examples/twin_evidence_freshness.py` | `examples/twin/twin_evidence_freshness.py` |
| `examples/twin_evolution_cycle.py` | `examples/twin/twin_evolution_cycle.py` |
| `examples/twin_evolution_history.py` | `examples/twin/twin_evolution_history.py` |
| `examples/twin_parameter_estimation.py` | `examples/twin/twin_parameter_estimation.py` |
| `examples/twin_region_model.py` | `examples/twin/twin_region_model.py` |
| `examples/twin_region_model_persistence.py` | `examples/twin/twin_region_model_persistence.py` |
| `examples/twin_region_release_assessment.py` | `examples/twin/twin_region_release_assessment.py` |
| `examples/twin_validation_history.py` | `examples/twin/twin_validation_history.py` |
| `examples/visualize_rydberg_transport.py` | `examples/simulation/visualize_rydberg_transport.py` |
| `examples/vqe_switch_sv_mps_tn.py` | `examples/training/vqe_switch_sv_mps_tn.py` |
| `examples/weighted_routing.py` | `examples/circuits/compiler/weighted_routing.py` |

## Incoming head recheck

The subsequent #1084 increment through
`dcd7a8cf37436416492595a5c1061a624cf562fe` contains seven commits and eight
Cirq implementation, test and audit-document files. All eight blobs were
SHA-verified and contain no old example, matrix-free or physical numerical
module references. This supplements the complete 4,376 changed-text audit at
`11fd9fa27ec31e74f03277010762e8d25d17c105`; it does not merge that PR or
qualify its still-open construction and packaging changes. Recheck again if
the incoming head changes before final integration.

The next verified incoming increment, from `dcd7a8cf` through
`7bbffeed70da51252e6b9f36fd05b1d85509cd6f`, adds six commits and thirteen
format-factory, test and audit-document files. All thirteen blobs are
SHA-verified; none intersects the migration sources or targets or contains an
old operational path reference. The actual retired-import guard also passes
on these files, including relative and parent-package imports.
