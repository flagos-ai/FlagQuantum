# Reviewer audit checklist

Use this checklist for a full audit. Record each item as verified, failed, not
applicable, or blocked; attach the supporting file, equation, or dataset.

## Significance and novelty

- Is the main advance stated in one falsifiable sentence?
- Are the closest competing methods compared under matched assumptions?
- Does the paper demonstrate physics or methodology beyond API integration?
- Is the scope broad enough for the intended claim but narrow enough to prove?
- Are novelty and priority checked against current primary literature?

## Theory

- Are assumptions, domains, initial conditions, and conventions explicit?
- Does every important equation connect to an implemented quantity or test?
- Are channel ordering and complete-positivity constraints specified?
- Are analytic limits and at least one independent calculation checked?
- Is identifiability discussed through sensitivities, controls, or rank?
- Are fitted parameters separated from calibration-only inputs?

## Hardware and numerics

- Are logical and compiled physical circuits both archived?
- Are qubits, gates, durations, calibrations, timestamps, and shots recorded?
- Are compiler fusion, routing, idle periods, and bit ordering verified?
- Are all seeds, optimizer settings, stopping rules, and failures retained?
- Are density-matrix, sampling, and hardware curves labeled unambiguously?
- Does at least one held-out workload exercise every claimed gate mechanism?

## Statistics

- Is the independent unit a circuit, batch, epoch, mapping, or device—not a
  shot unless the claim is strictly conditional on one task?
- Are shared-shot covariances preserved?
- Are selection, tuning, and evaluation data separated?
- Are uncertainty intervals defined and calibrated empirically?
- Are effect sizes reported with batch-level intervals?
- Are exclusions and multiple comparisons handled prospectively?

## Predictive digital twins

- Was the model frozen before held-out execution?
- Was any target energy or count used to choose model parameters?
- Are calibration age, circuit depth, entangling count, and mapping varied?
- Are residual bias and predictive-interval coverage reported?
- Is failure against drift, crosstalk, leakage, or coherent error exposed?
- Does model-guided selection outperform strong operational baselines?

## Presentation and reproducibility

- Do title, abstract, figures, conclusion, and popular summary make the same
  bounded claim?
- Can every headline number be regenerated from archived raw data?
- Are figures legible at final size and distinguishable without color?
- Are units, sample sizes, intervals, and provenance in captions?
- Does the supplement contain derivations and task-level provenance?
- Does a clean LaTeX build finish without unresolved citations or references?
- Are code, data, environment, license, DOI, declarations, and author metadata
  ready for the current journal submission system?

## Stop conditions

Do not strengthen the claim when any of the following holds:

- the result is retrospective but described as predictive;
- a single hardware epoch supports a cross-time claim;
- an unentangled circuit is used to validate entangling-gate noise;
- the model was tuned on the same points used for evaluation;
- shots are treated as independent hardware replicas;
- a mechanism is not identifiable from the measured observables;
- uncertainty omits a component central to the claim;
- selected successful jobs are shown without the complete task record.
