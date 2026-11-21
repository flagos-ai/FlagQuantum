# Notebook user-experience delivery plan

## Outcome and independent measures

A reader should express the quantum problem in FlagQuantum, understand what the
compiler supplies, execute the program and interpret its result. Migration
acceptance alone does not establish that teaching outcome.

Retain the 212 accepted upstream workflows, 216 original-source replay rows and
six approved alternative paths with their existing semantics and evidence.
Do not relabel alternative paths as original passes. The deferred original
calibration workflow remains outside the 212 approved denominator.

Track three independent measures:

- Migration: 212/212 approved workflows accepted in the historical snapshot;
  current revisions require their applicable regression tests.
- Teaching: six new learning-path Notebooks passed their initial numerical checks;
  43/212 migration Notebooks have been reviewed and replayed. See the
  [per-case editorial ledger](NOTEBOOK_UX_REVIEW.md). Remaining cases are pending.
- Capability gaps: inspect actual code and missing public behavior. A hidden
  helper alone is a presentation issue, not proof that synth lacks a feature.

## Delivery sequence

| Batch | Deliverable | Acceptance | Status |
| --- | --- | --- | --- |
| UX-01 | Static screen of all 212 accepted workflows; separate navigation | Reproducible report tied to existing inventory; no evidence rewrite | Screen complete; editorial review active |
| UX-02 | Six learning-path Notebooks | All cells execute; independent amplitudes, cleanup, topology and optimizer checks | Initial independent checks passed |
| UX-03 | Review delegated construction by shared helper family | Expose useful public program at the point of use; preserve upstream workflow and numerical scope | First seven delegated workflows exposed; remaining review active |
| UX-04 | Review direct Circuit uses and capability gaps | Identify reference versus application code; reuse existing high-level APIs first | Seven initially visible Circuit workflows reviewed and replayed; hidden kernels still require inspection |
| UX-05 | Review remaining visible-qfunc workflows and five-category navigation | Explain problem, quantum semantics, compilation, result and limits; changed workflows pass tests | Functions 31/31 reviewed; case navigation names corrected |
| UX-06 | Final evidence and handoff | All 212 editorial reviews resolved; applicable corpus regression and default checks recorded | Pending |

Work proceeds in reviewable local batches. Remote CI is reserved for integration;
there is no need for one remote run per Notebook. Normal repository API-change
rules still apply to any discovered capability extension.

## Editorial standard

Each user-facing Notebook must explain the problem and why a quantum program is
being built, identify classical versus quantum values, and expose the central
program or an appropriate documented public algorithm call. A private workflow
wrapper with no explanation of its quantum construction is insufficient.

Show the route from program to synthesis constraints, execution, named results
and interpretation. Explain the specific work performed by FlagQuantum (for
example arithmetic construction, inverse cleanup or connectivity routing),
without claiming that simple gate syntax itself is automatic optimization.

Keep loading, plotting and reusable numerical algorithms in helpers when that
improves readability. References, exhaustive assertions and evidence internals
belong in tests or a clearly separated appendix. Do not inline whole algorithm
implementations just to increase the visible-qfunc count.

Declare simulation method, approximation, original-scale substitutions and
unsupported behavior. An exact probability is not a finite-shot measurement;
an optimization objective is not proof of a global optimum. Each lesson ends
with a small modification and a meaningful expected effect.

## Initial evidence-backed findings

The initial static screen identified 129 workflows without literal
qfunc definitions, 76 with visible qfunc definitions and seven with direct
Circuit constructors. These mutually exclusive buckets are review signals,
not quality scores or capability counts. The [current generated screen](NOTEBOOK_UX_AUDIT.md)
reflects subsequent source edits.

- Image-edge detection hides its central quantum composition behind
  `workflow.run()`. Its `quantum.py` also explicitly implements amplitude
  preparation with Circuit gates. Investigate current public preparation
  support and original-size budgets before calling this a missing capability.
- The small VQE workflow already uses actual synth executions and the local
  optimizer. It needs explanations of parameter binding and expectation
  evaluation more than another algorithm implementation.
- Hardware-aware MCX already exposes constraints, topology and reports. Retain
  this useful content while separating teaching from exhaustive evidence.

## Validation policy

Execute changed Notebook cells in order with the existing source-aware executor.
Use independent mathematical/Circuit references where quantum construction
changes. Keep resource limits and original-case numerical checks unchanged.
Run the catalog checks after navigation changes, then repository default checks
at each integration checkpoint. Record raw failures and distinguish environmental
rechecks from a passing complete run.
