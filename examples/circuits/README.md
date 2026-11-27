# Circuits examples

These source examples demonstrate supported FlagQuantum workflows. They are not
production implementation modules and are not included as a public import API.
Run them from the repository root with the documented environment and optional
dependencies required by each example. Remote or hardware examples require
their stated credentials and resources; local tests do not submit paid jobs.

| Example | Module command |
| --- | --- |
| [compiler_optimize.py](compiler/compiler_optimize.py) | `python -m examples.circuits.compiler.compiler_optimize` |
| [compiler_synthesis.py](compiler/compiler_synthesis.py) | `python -m examples.circuits.compiler.compiler_synthesis` |
| [dynamic_disjunctive_condition.py](dynamic_disjunctive_condition.py) | `python -m examples.circuits.dynamic_disjunctive_condition` |
| [gate_coverage_batch_b.py](gate_coverage_batch_b.py) | `python -m examples.circuits.gate_coverage_batch_b` |
| [ising_braid_source.py](ising_braid_source.py) | `python -m examples.circuits.ising_braid_source` |
| [routed_artifact_preflight.py](compiler/routed_artifact_preflight.py) | `python -m examples.circuits.compiler.routed_artifact_preflight` |
| [rydberg_program_geometry.py](rydberg_program_geometry.py) | `python -m examples.circuits.rydberg_program_geometry` |
| [target_aware_compilation.py](compiler/target_aware_compilation.py) | `python -m examples.circuits.compiler.target_aware_compilation` |
| [weighted_routing.py](compiler/weighted_routing.py) | `python -m examples.circuits.compiler.weighted_routing` |

For a change, keep the example on public FlagQuantum entry points and run its
existing integration test. Preserve the original numerical and gradient checks.
