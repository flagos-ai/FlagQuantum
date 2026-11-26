# Simulation examples

These source examples demonstrate supported FlagQuantum workflows. They are not
production implementation modules and are not included as a public import API.
Run them from the repository root with the documented environment and optional
dependencies required by each example. Remote or hardware examples require
their stated credentials and resources; local tests do not submit paid jobs.

| Example | Module command |
| --- | --- |
| [absorptive_photon_counting.py](absorptive_photon_counting.py) | `python -m examples.simulation.absorptive_photon_counting` |
| [boson_interference.py](boson_interference.py) | `python -m examples.simulation.boson_interference` |
| [carried_operand_execution.py](carried_operand_execution.py) | `python -m examples.simulation.carried_operand_execution` |
| [constrained_basis_reference.py](constrained_basis_reference.py) | `python -m examples.simulation.constrained_basis_reference` |
| [cpu_statevector.py](cpu_statevector.py) | `python -m examples.simulation.cpu_statevector` |
| [density_matrix_execution.py](density_matrix_execution.py) | `python -m examples.simulation.density_matrix_execution` |
| [driven_three_level.py](driven_three_level.py) | `python -m examples.simulation.driven_three_level` |
| [driven_three_level_relaxation.py](driven_three_level_relaxation.py) | `python -m examples.simulation.driven_three_level_relaxation` |
| [dynamic_density_matrix.py](dynamic_density_matrix.py) | `python -m examples.simulation.dynamic_density_matrix` |
| [ising_braid_execution.py](ising_braid_execution.py) | `python -m examples.simulation.ising_braid_execution` |
| [ising_fusion_measurement.py](ising_fusion_measurement.py) | `python -m examples.simulation.ising_fusion_measurement` |
| [lindblad_evolution.py](lindblad_evolution.py) | `python -m examples.simulation.lindblad_evolution` |
| [lindblad_output_names.py](lindblad_output_names.py) | `python -m examples.simulation.lindblad_output_names` |
| [matrix_expectation_dynamics.py](matrix_expectation_dynamics.py) | `python -m examples.simulation.matrix_expectation_dynamics` |
| [mixed_dimension_dynamics.py](mixed_dimension_dynamics.py) | `python -m examples.simulation.mixed_dimension_dynamics` |
| [neutral_atom_pair_reference.py](neutral_atom_pair_reference.py) | `python -m examples.simulation.neutral_atom_pair_reference` |
| [noisy_simulation_v1.py](noisy_simulation_v1.py) | `python -m examples.simulation.noisy_simulation_v1` |
| [rydberg_native_dynamics.py](rydberg_native_dynamics.py) | `python -m examples.simulation.rydberg_native_dynamics` |
| [shared_dynamics_workflow.py](shared_dynamics_workflow.py) | `python -m examples.simulation.shared_dynamics_workflow` |
| [spin_mode_convergence.py](spin_mode_convergence.py) | `python -m examples.simulation.spin_mode_convergence` |
| [spin_mode_thermal_bath.py](spin_mode_thermal_bath.py) | `python -m examples.simulation.spin_mode_thermal_bath` |
| [stabilizer_sampling.py](stabilizer_sampling.py) | `python -m examples.simulation.stabilizer_sampling` |
| [superoperator_channel_algebra.py](superoperator_channel_algebra.py) | `python -m examples.simulation.superoperator_channel_algebra` |
| [terminal_instrument_reference.py](terminal_instrument_reference.py) | `python -m examples.simulation.terminal_instrument_reference` |
| [terminal_instrument_sampling.py](terminal_instrument_sampling.py) | `python -m examples.simulation.terminal_instrument_sampling` |
| [triton_statevector_local_1q.py](triton_statevector_local_1q.py) | `python -m examples.simulation.triton_statevector_local_1q` |
| [visualize_rydberg_transport.py](visualize_rydberg_transport.py) | `python -m examples.simulation.visualize_rydberg_transport` |

For a change, keep the example on public FlagQuantum entry points and run its
existing integration test. Preserve the original numerical and gradient checks.
