# Component tensor report notebook

## Source and ownership

Complete `dev_tools/tensor-report-card.ipynb` from Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` has Git blob
`49d08b1f9edad52672322b73315b31c32c51415a`. All 21 original cells remain
unchanged in the external reference checkout. The native notebook preserves
all 21 cell kinds and all six report views. Copyright 2024 Google LLC,
Apache-2.0, applies to the adapted report workflow.

`examples/algorithms/component_tensor_report.py` supplies importable process
tasks and an explicit three-graph catalog. Existing Compiler graph traversal
selects primitive CNOT calls. Existing Simulation COPY/XOR suppliers build
factorized nodes. Existing development report and deadline owners measure
stages and own processes. The notebook composes these owners without introducing
an alternative component base class, numerical kernel or resource estimator.
This is a single-device CPU developer inspection workflow.

## Complete cell mapping

| Original cells | Native behavior |
| --- | --- |
| 0 | Explain storage, width heuristic, deadline and concrete provider scope |
| 1 | Import existing executor and importable native task |
| 2 | Select explicit native examples and print observed/model counts separately |
| 3 | Submit, drain, print remaining work, retain timeout records and build frame |
| 4–5 | Number of selected examples |
| 6–7 | Number with completed graph selection |
| 8–9 | Number with positive width |
| 10–11 | Width less than or equal to 25 |
| 12–13 | Width greater than 25, sorted by width |
| 14–15 | Missing width, including errors and timeouts |
| 16–17 | Slowest graph-selection durations |
| 18 | Qualify original timing observations as historical |
| 19–20 | Slowest tensor construction and width planning |

Run `examples/notebooks/component_tensor_report.ipynb` from the repository root
with the optional locked notebook and tensor planner dependencies installed.
Expected results are three rows named `cnot_chain_1`, `cnot_chain_2` and
`cnot_chain_3`, all passing the width heuristic. Measured CX counts and independent
repetition-model estimates are 1, 2 and 3. Timing values vary between runs.
Existing register-bound operations continue to use `circuit.append(op)` through
their accepted physical binding owners. This report adds no circuit API.

## Explicit adaptations and limits

- The native catalog contains three executable graphs, not the entire original
  example inventory. Native and source whole-library coverage and performance
  are not equated. Extending the catalog requires actual Compiler and Simulation
  providers for each graph.
- These examples already contain primitive calls. Their first timing column
  measures Compiler graph selection. Original `flat_dur` measures recursive
  composite flattening. Their labels remain comparable report fields, but their
  durations measure the stated different work.
- The original prose says minimum intermediate size and width below 25. The
  executable code filters `width <= 25`. The native prose describes the largest
  intermediate in the selected contraction tree and preserves that inclusive
  filter. Width is a planner heuristic, with no full simulation capacity claim.
- All-timeout or all-early-error original reports fail with `KeyError` at a
  missing timing column. Native reindexing supplies the complete column schema
  so all six views remain usable. The original failure is executed as a
  counterexample. Partial errors retain their original completed fields.
- The original explicitly skips an unpickleable lambda-bearing component.
  The complete source execution test exercises that skip. The native catalog
  has importable tasks and concrete serializable programs, with no such entry.
- The original labels every missing process payload `Timeout`, including child
  crashes. Native output retains that vocabulary. Executor failure/cleanup
  semantics reuse the separately accepted complete process-owner tests.
- Native processes are context managed. The eight-second deadline includes
  startup/import time. The original stage measurements and historical runtime
  observations do not establish current performance.
- The catalog has no quantum measurements or auxiliary registers. There are
  no measurement branches to enumerate or work bits to clear. Binding events
  are empty. Real process cleanup is checked separately from quantum ownership.
- Tensor node count, observed primitive CX count, repetition-model estimates,
  stage durations and contraction width are distinct quantities.

## Verification

The concentrated regression passed 398 cases with zero skips and 15 existing
fork deprecation warnings. Strict typing passed for both changed Python files.
Black, Ruff, Integration scope, architecture, IR, API and diff checks passed.
The receipt is `g2-tensor-notebook-regression-20261121.log` in the external
reference environment. The focused test executes all original cells on shared deterministic
records, including zero width, 25, 26, partial failures and timeouts. It compares
all six tables and their ordering. Real default-spawn workers and a real Jupyter
kernel exercise the native workflow. The selected graphs have exact dense
operator equality and entangled complex-state physical-binding checks, empty
workspace events and separate observed/model resource checks. Existing accepted
factorized CNOT source comparisons and tensor-report/executor tests are reused.
