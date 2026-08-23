# Repository PRX Quantum style anchor

## Source

Inspect this paper before making full-manuscript structural judgments:

`../../../paper/prx_digital_twin/reference_prxq_paper/PRXQuantum.4.020331.pdf`

Zhiyan Ding and Lin Lin, “Even Shorter Quantum Circuit for Phase Estimation on
Early Fault-Tolerant Quantum Computers with Applications to Ground-State
Energy Estimation,” *PRX Quantum* **4**, 020331 (2023), DOI
10.1103/PRXQuantum.4.020331.

This is a 30-page theory-and-numerics article. Use it to calibrate rigor and
argument architecture, not to force an experimental digital-twin paper into a
theorem paper's length or section count.

## Transferable patterns

- Open with explicit performance metrics and a concrete limitation of prior
  methods.
- State a short numbered list of properties the proposed method must satisfy.
- Introduce the main idea intuitively before presenting the complete method.
- Define symbols immediately and specify assumptions and domains.
- Connect each principal claim to a named theorem, algorithm, figure, or
  numerical experiment.
- Separate maximal resource requirements from total cost; analogously, keep
  shots per task, total shots, circuit depth, and independent hardware batches
  distinct in the FlagQuantum work.
- Explain why the method can fail and which assumptions prevent failure.
- Use numerical studies to test scaling and regimes already motivated by the
  theory rather than as decorative demonstrations.
- Keep the main line readable while moving long estimates and proofs to
  appendices.
- Write captions that identify variables, regimes, comparisons, and the
  physical meaning of each panel without relying on the surrounding prose.

## Adaptation to the FlagQuantum--Quafu manuscript

Use the corresponding argument spine:

1. Define predictive accuracy, hardware cost, calibration age, and the
   independent unit of replication.
2. State the requirements for a calibration-derived digital twin.
3. Present the calibration-to-compiled-circuit-to-observable map intuitively.
4. Give the mathematical channel and uncertainty model.
5. Establish physical constraints and identifiability limits.
6. Treat the Baihua 20-point VQE result as a retrospective pilot.
7. Use frozen, later-epoch, entangling workloads for the decisive prospective
   test.
8. Move detailed channel derivations, assignment covariance, task provenance,
   and sensitivity analysis to the supplement.

## Do not imitate blindly

- Do not add theorem labels unless a result is genuinely proved.
- Do not inflate length with derivations that do not support a measured or
  predictive claim.
- Do not imitate exact typography, sentence construction, figure geometry, or
  notation.
- Do not infer current journal policy from a 2023 article; verify official APS
  instructions separately.
- Do not use this theory paper as the only comparison for an experimental
  manuscript; inspect recent experimental PRX Quantum articles when available.
