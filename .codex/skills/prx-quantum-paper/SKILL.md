---
name: prx-quantum-paper
description: Develop, audit, and revise rigorous quantum-information and quantum-computing manuscripts toward PRX Quantum submission quality. Use for LaTeX paper drafting, theory derivations, hardware experiments, numerical validation, noise-model or digital-twin claims, statistical analysis, APS-style figures, supplemental material, reproducibility packages, cover letters, and reviewer-response preparation. Also use when deciding whether evidence supports words such as predictive, accurate, quantum advantage, scalable, or general.
---

# PRX Quantum Paper

Build a defensible scientific argument, not merely a polished document. Treat
publication level as an evidence threshold: never promise acceptance or hide
missing experiments behind stronger prose.

## Start with an evidence audit

1. Read the manuscript, supplement, figures, analysis code, and primary data.
2. Verify current PRX Quantum and APS author instructions from official sources
   before enforcing format or submission requirements; journal rules change.
3. Write the central claim in one sentence and classify it as theoretical,
   numerical, retrospective experimental, prospective predictive, or
   operational.
4. Map every clause of the claim to a figure, table, theorem, dataset, or
   prespecified test.
5. Mark unsupported clauses explicitly. Weaken the claim or propose the
   smallest decisive experiment; do not manufacture support.
6. Read [references/reviewer-audit.md](references/reviewer-audit.md) when doing
   a full manuscript audit or planning additional experiments.
7. When working in this repository, read
   [references/style-anchor.md](references/style-anchor.md) and inspect the
   archived PRX Quantum paper it identifies before making structural, equation,
   or figure-density judgments.

## Establish the contribution

Require all three elements:

- A clear gap relative to the closest literature.
- A technically nontrivial method, result, or physical insight.
- Evidence that distinguishes the contribution from an engineering demo or a
  fit to one selected curve.

Use primary papers and official documentation for technical claims. Search the
current literature when novelty, priority, device specifications, or journal
policy matters. Separate demonstrated results from intended future work in the
abstract, introduction, captions, and conclusion.

## Build the theory-to-observable chain

Derive in this order whenever applicable:

1. State assumptions and the physical regime.
2. Define the ideal state, Hamiltonian, circuit, or dynamical generator.
3. Define the noisy channel with explicit composition order and targets.
4. Establish physical constraints such as trace preservation, complete
   positivity, normalization, symmetry, or limiting behavior.
5. Propagate the model to probabilities and measured observables.
6. Derive the estimator, uncertainty, and comparison metric.
7. Identify parameters that the experiment can and cannot distinguish.
8. State a falsifiable prediction or failure boundary.

Check dimensions, index ordering, endianness, sign conventions, basis changes,
gate fusion, channel ordering, and special limits. For a calibration-derived
noise model, distinguish externally measured parameters from parameters fitted
to the reported target data. Never describe nonidentifiable mechanisms as
validated merely because their combined output matches an energy curve.

Keep the main text focused on physical insight and decisive equations. Put
Kraus derivations, covariance algebra, compiler identities, calibration tables,
task identifiers, and analytic cross-checks in the supplement.

## Protect experimental validity

For predictive claims, freeze the model, code, circuits, endpoints, and
exclusion rules before acquiring held-out hardware results. Label a same-epoch
reconstruction or post hoc fit as retrospective.

Use the correct unit of replication. Shots estimate conditional measurement
noise; repeated shots from one task are not independent devices, calibration
epochs, or hardware batches. Aggregate evidence across prespecified circuits,
qubit mappings, days, and calibration epochs as required by the claim.

Archive:

- timestamps and calibration snapshots;
- logical and physical/transpiled circuits;
- physical-qubit mappings and compiler settings;
- shot counts per task and total hardware cost;
- all task identifiers, failures, retries, and exclusion reasons;
- raw counts before normalization or mitigation;
- immutable model/configuration identity and software versions.

Compare against strong, matched baselines. Apply the same information budget,
held-out split, circuit set, and scoring rule to every model. Use ablations to
isolate physical-gate replay, relaxation, dephasing, coherent error, correlated
readout, crosstalk, leakage, and drift only when the data can test them.

## Treat uncertainty as part of the claim

Report effect sizes and intervals, not only point errors. Distinguish:

- finite-shot uncertainty;
- calibration-estimation uncertainty;
- optimizer or seed variability;
- between-task and between-batch variability;
- temporal drift;
- model misspecification.

Preserve covariance between observables estimated from the same shots. Do not
pool shots to inflate the apparent number of independent replicates. For a
predictive model, assess residual bias, interval coverage, calibration with
age/depth, and out-of-distribution failure—not MAE alone. State whether an
error bar is a confidence interval, prediction interval, standard deviation,
or standard error.

## Write the manuscript around one argument

Use this narrative order unless the science demands otherwise:

1. Problem and gap.
2. Precise contribution and claim boundary.
3. Physical or mathematical framework.
4. Experimental protocol and preregistered comparisons.
5. Primary result.
6. Mechanistic ablations and failure analysis.
7. Generalization or operational utility.
8. Limitations and falsifiable next test.

Make the title and abstract no stronger than the evidence. Define acronyms and
symbols once. Prefer precise verbs such as predicts, reconstructs, estimates,
or correlates over vague verbs such as captures or enables. Remove promotional
language, repeated conclusions, and plans presented as completed results.

## Produce publication-grade figures

Follow current APS technical requirements and the visual conventions of recent
PRX Quantum research articles without imitating a specific paper. Ensure:

- legibility at final one- or two-column size;
- vector output for line art and embedded fonts;
- accessible colors plus redundant line styles or marker shapes;
- panel labels, units, uncertainty definitions, sample sizes, and model/data
  provenance;
- no overlap, decorative clutter, screenshots, or oversized legends;
- captions that state what was measured, what was simulated, and what the
  intervals represent.

Inspect the rendered PDF rather than trusting plotting source alone.

Use an archived reference paper as a quality and genre anchor, not as a
template. Extract its argument structure, definition discipline, theorem-to-
evidence rhythm, caption completeness, and appendix boundary. Do not copy its
phrasing, notation, layouts, or topic-specific organization. Match the target
manuscript's experimental or theoretical genre.

## Validate the submission package

Compile from a clean state and check for undefined references, missing fonts,
cropped panels, inconsistent numbers, stale generated files, and inaccessible
paths. Cross-check every abstract number against the source artifact and every
table against the analysis output. Ensure the supplement is independently
readable and the data/code statement names the actual reproducibility assets.

End each revision with two lists:

- Completed and verified improvements.
- Scientific blockers that still prevent the strongest intended claim.

Do not call a manuscript submission-ready while author metadata, permissions,
data availability, required declarations, decisive held-out evidence, or
current journal-format checks remain unresolved.
