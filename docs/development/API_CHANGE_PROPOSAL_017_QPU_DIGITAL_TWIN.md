# API Change Proposal 017: QPU digital twin entry points

## Status

**Approved before the first public alpha.** On 2026-09-09, the API owner explicitly
authorized integrating the existing digital-twin research into `flagquantum/twin`
under vNext domain boundaries. The repository has not been publicly released;
the previous research module names and compatibility entry points are not retained.

## Decisions

- Introduce the candidate public namespace `flagquantum.twin` without new root exports.
- Initial entry points are `QPUDigitalTwin`, `TwinSnapshot`, `TwinPrediction`, and
  `TwinValidationReport`. Hardware validation adds `TwinExperiment` and
  `TwinHardwareReport`, without duplicating task polling or introducing a run manager.
- Twin owns calibration-conditioned device models, frozen predictions, and
  comparisons with hardware observations.
- Noise owns noise semantics, Simulation owns numerical execution, and Remote
  owns vendor calibration retrieval and task control.
- Q-ATLAS naming, phase numbers, candidate state machines, and experimental data
  workflows do not enter the formal public namespace.

## Maturity boundary

The first vertical slice supports a single circuit, exact density-matrix
prediction, and distribution comparisons based on measurement counts. This is
a calibration-driven device emulation and validation interface. It establishes
neither pulse-level equivalence nor general predictive validity across devices
or calibration periods.

## Acceptance

- Quafu calibration can be frozen into a twin snapshot through the existing
  Remote converter.
- The snapshot binds the device profile, noise model, physical mapping, and
  acquisition time.
- Predictions reuse the existing Noise and Simulation execution paths.
- Hardware-count comparisons produce serializable, identity-bound reports.
- Freeze the submitted circuit before transmission; the submission receipt and
  result must carry the same submission digest. Submission identity is distinct
  from execution identity. The public Quafu task interface does not provide a
  pre-submission receipt for the final circuit. Local QSteed or QuarkCircuit
  transpilation produces a candidate and cannot replace a provider receipt.
  Hardware reports therefore use retrospective diagnostic validation and
  separately record whether the executed circuit returned by the provider
  matches the submitted circuit.
- Simulation, Noise, and Remote must not import `twin`.
