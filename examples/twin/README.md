# Twin examples

These source examples demonstrate supported FlagQuantum workflows. They are not
production implementation modules and are not included as a public import API.
Run them from the repository root with the documented environment and optional
dependencies required by each example. Remote or hardware examples require
their stated credentials and resources; local tests do not submit paid jobs.

| Example | Module command |
| --- | --- |
| [twin_calibration_history.py](twin_calibration_history.py) | `python -m examples.twin.twin_calibration_history` |
| [twin_candidate_fit.py](twin_candidate_fit.py) | `python -m examples.twin.twin_candidate_fit` |
| [twin_circuit_support.py](twin_circuit_support.py) | `python -m examples.twin.twin_circuit_support` |
| [twin_connected_region.py](twin_connected_region.py) | `python -m examples.twin.twin_connected_region` |
| [twin_evidence_freshness.py](twin_evidence_freshness.py) | `python -m examples.twin.twin_evidence_freshness` |
| [twin_evolution_cycle.py](twin_evolution_cycle.py) | `python -m examples.twin.twin_evolution_cycle` |
| [twin_evolution_history.py](twin_evolution_history.py) | `python -m examples.twin.twin_evolution_history` |
| [twin_parameter_estimation.py](twin_parameter_estimation.py) | `python -m examples.twin.twin_parameter_estimation` |
| [twin_region_model.py](twin_region_model.py) | `python -m examples.twin.twin_region_model` |
| [twin_region_model_persistence.py](twin_region_model_persistence.py) | `python -m examples.twin.twin_region_model_persistence` |
| [twin_region_release_assessment.py](twin_region_release_assessment.py) | `python -m examples.twin.twin_region_release_assessment` |
| [twin_validation_history.py](twin_validation_history.py) | `python -m examples.twin.twin_validation_history` |

For a change, keep the example on public FlagQuantum entry points and run its
existing integration test. Preserve the original numerical and gradient checks.
