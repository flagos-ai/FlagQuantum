# What has been checked

The original course checks were run on September 9, 2026. This record concerns the learning materials,
not event capacity or platform certification.

## Local execution

- Python 3.12.14 and PyTorch 2.13.0.
- Notebook tools: nbclient 0.10.4, nbformat 5.11.1, ipykernel 6.31.0, and matplotlib 3.11.1.
- All six notebooks passed format checks and ran with their default settings in separate Jupyter kernels.
- Notebooks 01, 03, and 06 produced the probability, training-loss, and hardware-comparison charts.
- The CPU Bell result was approximately `[0.49999997, 0, 0, 0.49999997]`, with a maximum absolute state error of zero against the reference tensor.
- The 80-step training run reduced squared error from `0.23878333` to about `0.00002105`.
- The recorded Quafu counts sum to 1024. The task ID and replay label were preserved; the total variation distance was `217/1024`.

The English revision was also checked: all six notebooks ran successfully with their default settings,
local links resolved, and no Chinese text remained in the workshop materials.

## What these checks do not cover

The local machine had no CUDA GPU, so notebook 02 skipped that branch.
The submission switches in notebooks 04 and 05 remained off. No participant credentials were used and no new cloud or hardware jobs were created.

The event login flow, concurrent users, GPU images, and current QSteed/Quafu configuration must still be rehearsed using [the instructor guide](INSTRUCTOR.md).

To repeat the local numerical checks from the repository root:

```bash
python workshops/flagos2026/scripts/verify_local.py
```

The checked-in notebooks have no saved outputs. Personal results belong in the ignored `outputs/` directory.

## Step-by-step teaching revision

All six notebooks were rebuilt so participants can write and inspect the core calculations in their cells.
The revised notebooks passed execution in separate local kernels: circuit construction, one manual gradient update followed by training, the locally written cloud-job program, and count normalization all ran successfully.
No notebook imports the workshop bell, training, or comparison helpers. Notebook 05 shows the public hardware call directly instead of launching a wrapper script.
GPU execution and remote submissions remained disabled or skipped during this local validation.

## Expanded lab collection

The collection now contains 17 independent notebooks. The default local paths of all 17 passed execution in separate Jupyter kernels.
New checks include hybrid-model fitting, VQE against exact diagonalization, shot-count totals, an analytic amplitude-damping curve,
MPS against a dense reference, compilation equivalence, analytic and parameter-shift gradients, and a small Max-Cut objective.
The compiler-extension lab also passes the SDK conformance checks. Charts for the graph optimization and MPS approximation experiments were visually inspected.

Validation caught and corrected result-access assumptions: counts are read from `result.counts[0]`, and remote task identity from `result.provenance["task_id"]`.
The exact density-matrix lesson reads the matrix diagonal explicitly because the current generic Pauli-output path does not support that result.
The extension lesson uses the scoped SDK registry; named `fq.compile` discovery requires an installed entry point.

The enabled result-handling branches of labs 05 and 13 were additionally checked with mocked remote responses using the current ExecutionResult contract.
These were offline interface tests, not new QPU executions. GPU branches, hardware calibration access, QPU submissions, and Jiuding submissions still need event rehearsal.
The repository default smoke/unit regression passed: 1329 passed, 14 skipped, 1330 deselected.

After following SETUP.md, repeat the notebook checks with:

```bash
python workshops/flagos2026/scripts/verify_notebooks.py
# Or check one edited lab:
python workshops/flagos2026/scripts/verify_notebooks.py --notebook 08_vqe.ipynb
```

The command starts a fresh kernel for each notebook, applies a per-cell timeout, and saves executed copies under `outputs/validation/`.
It rejects enabled remote-action switches in the supplied labs. This is an accidental-submission check, not a sandbox for arbitrary notebook code.

## Featured experiments refinement

The collection now contains 18 labs. In this revision, labs 07 and 13 were expanded and lab 18 was added.
All three passed execution in separate local Jupyter kernels with FlagQuantum 0.2.0 and PyTorch 2.13.0.
The other 15 retain the execution evidence recorded above; they were not rerun in this revision.

- Lab 07 checks gradients and parameter changes in both the classical encoder and quantum layer. Its 20 held-out interpolation points had an MSE of approximately `1.22e-8` after 150 updates. The lesson explains the model's parameter ambiguity and limits of this fitting example.
- Lab 18 checks the same two-qubit objective across statevector, MPS, and tensor-network modes with fallback disabled. Initial gradients differed by at most `1.20e-7`; all three trained energies were within `2.15e-7` of the exact ground energy `-sqrt(2)` after 160 updates. These are CPU numerical checks, not scalability or performance evidence.
- Lab 13 preserves the trained angle, circuit hash, ideal expectation, and measurement source. The ideal expectation was approximately `0.256696` for a training target of `0.25`. Its default local sample had 668 zero outcomes and 356 one outcomes, totaling 1024 shots. The report separates the learning gap from the measured deviation from the frozen circuit.

The three main result charts were visually inspected. All 18 source notebooks have empty outputs and execution counts;
Python cell syntax, English text, and local links were checked. The enabled hardware-result branch of lab 13 also passed an offline mocked-response check using the current result contract, including task ID and measurement-source preservation.
No new QPU or Jiuding task was submitted.

The README now highlights the three featured experiments, and FIELD_NOTES.md provides prediction and interpretation prompts.
The instructor guide includes a first-time-reader pilot procedure. That human usability pilot and live event-resource rehearsal remain pending.
