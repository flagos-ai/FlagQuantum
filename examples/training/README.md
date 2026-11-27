# Training examples

These source examples demonstrate supported FlagQuantum workflows. They are not
production implementation modules and are not included as a public import API.
Run them from the repository root with the documented environment and optional
dependencies required by each example. Remote or hardware examples require
their stated credentials and resources; local tests do not submit paid jobs.

| Example | Module command |
| --- | --- |
| [ascend_910c_training.py](ascend_910c_training.py) | `python -m examples.training.ascend_910c_training` |
| [benchmark_static_program.py](benchmark_static_program.py) | `python -m examples.training.benchmark_static_program` |
| [hybrid_jax_torch_training.py](hybrid_jax_torch_training.py) | `python -m examples.training.hybrid_jax_torch_training` |
| [hybrid_model_acceptance.py](hybrid_model_acceptance.py) | `python -m examples.training.hybrid_model_acceptance` |
| [quantum_transformer.py](quantum_transformer.py) | `python -m examples.training.quantum_transformer` |
| [quick_start.py](quick_start.py) | `python -m examples.training.quick_start` |
| [save_experiment.py](save_experiment.py) | `python -m examples.training.save_experiment` |
| [train_boson_coupling.py](train_boson_coupling.py) | `python -m examples.training.train_boson_coupling` |
| [train_parameterized_circuit_then_deploy.py](train_parameterized_circuit_then_deploy.py) | `python -m examples.training.train_parameterized_circuit_then_deploy` |
| [train_piecewise_drive.py](train_piecewise_drive.py) | `python -m examples.training.train_piecewise_drive` |
| [train_rydberg_controls.py](train_rydberg_controls.py) | `python -m examples.training.train_rydberg_controls` |
| [vqe_switch_sv_mps_tn.py](vqe_switch_sv_mps_tn.py) | `python -m examples.training.vqe_switch_sv_mps_tn` |

For a change, keep the example on public FlagQuantum entry points and run its
existing integration test. Preserve the original numerical and gradient checks.
