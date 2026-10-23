# QEC-RESTRUCTURE-005: detector error model responsibilities

Status: implemented for review under the approved package plan.
Owner: Core; evidence-path changes precede implementation in a separate
Integration commit. Classification: `single_device_fast_path`.

## Review boundary

The branch integrates freshly fetched main `d658df0df` and the validated 002–004
prerequisites. Audit PR #735 has merged; #737, #739 and #740 remain separate PRs.
Before implementation, their exact heads passed full CI (38/38 each) and the
required single/two-GPU checks. The final #737 head is `5993c7f1b`, including all
four umask-independent test fixtures and automatic full CI for QEC stacked bases.

This PR targets frozen prerequisite `refactor/qec-restructure-005-base` at
`21366790263e1aecb754b309bad08a8041be7e04`. Its review diff contains batch 005;
Core, Compiler and Runtime implementations match main. It neither modifies nor
merges #731 or any other PR. After prerequisites land, retarget/replay this batch
onto current main and rerun required CI before merging.

## Ownership and dependencies

The [DEM README](../../flagquantum/qec/dem/README.md) states the eight owners:
`model`, `text`, `alternatives`, `matrices`, `construction`, `circuit`, `chunks`
and `graph`. The former `dem.py` becomes the package initializer atomically;
other former flat modules remain explicit aliases. There is one implementation
of every moved definition, with no forwarding algorithm, replacement class,
registry, protocol, dependency or policy exception.

The model imports code-owned matrices and declarative memory records directly.
Its memory factory imports construction only when called. A fresh-process test
first completes the historical root bootstrap, then rejects compiler, runtime,
execution and construction imports while reimporting the model and building
text/code-capacity models. This establishes the new model boundary; the existing
eager root facade deliberately retains its historical import behavior.

## Compatibility and extension impact

The frozen 001 baseline still checks all 108 ordered root exports, signatures,
defaults, fields, errors and registries. Both rotated surface classes retain
their independent identities. Neither registry implementation moves in 005.

Pre-move 005 fixtures retain 211 pickle globals, 244 definition/method metadata
entries, eight historical namespaces and exact numerical/serialization results.
They were captured before implementation, verified on Python 3.10–3.12 and never
refreshed to accommodate relocation. They cover independent/exclusive faults,
rates, matrices, seeded samples, error IDs, merge rules, memory/code-capacity
construction, text/IR realization, chunk seams/stitching/closure and refusals.
The 001–004 fixtures also remain unchanged.

The existing narrow metadata helper retains original class/function/method
`__module__` and qualified names. Legacy imports and canonical owners reference
the same objects. Old pickles load and round-trip without new reducers or
library-managed safe-global lists. Existing caller-owned allowlists keep their
historical identifiers. Imported helpers remain available through explicit aliases
to their actual defining modules; no duplicate helper types are created.

Physical source filenames change. Documentation links and physical contract
evidence follow the owner; capability status, schemas, counts, thresholds and
Stable Core snapshots do not change. Introspection tools should use a method's
code filename or the physical owner, rather than treating retained `__module__`
as a source filename. A test that substitutes the private construction global
`_mechanisms` now imports its actual owner; all its refusal assertions remain.
Public plugin imports and registration are unchanged. Rebinding private globals
through a compatibility facade is outside that public extension contract.

Core owns all compatibility aliases. No removal release is approved. Removing
them requires a separately approved deprecation and serialized-artifact plan.

## Runnable user example

No optional SDK, credential or GPU is needed:

```bash
PYTHONPATH=. python examples/qec/dem_model_and_chunks.py
```

Expected output:

```text
Detectors, observables, errors: 2 1 2
Detector rates: [0.3125, 0.25]
Observable rates: [0.125]
Graph edges: 2
Closed chunk: True
Historical pickle class: flagquantum.qec.dem
Same class: True
```

The [complete example](../../examples/qec/dem_model_and_chunks.py) uses existing
root imports only and is registered in the examples index. The PR description
also embeds its runnable Python and expected output.

## Validation

- Before migration on the integrated main: 1,811 QEC tests pass, with one existing
  no-applicable-gate-fault skip.
- After migration: 1,813 QEC tests pass with the same existing skip. All 39
  combined compatibility scenarios pass on Python 3.10.21, 3.11.16 and 3.12.14.
- All 71 original top-level class/function AST bodies match the frozen base after
  excluding import placement. Mathematical expressions and control flow match.
- Strict mypy checks all 820 package source files. Black 24.1 and Ruff 0.2 check
  the changed Python files; no suppression is added.
- All 184 contract/maturity/documentation tests pass, including all 35
  alignment-checker mutations; their existing-file proof names the new model file.
- The 499-check alignment gate, complete vocabulary ledgers, parity renderer,
  architecture boundaries, Stable Core API snapshot, documentation source of
  truth, public doctests and example registration are checked without relaxing
  their policies.

Full QEC, supported-Python compatibility and exact-head CI results are recorded
in the PR. Installation artifact and optional-extra matrix closure remains the
separate final batch; local CPU results are not GPU scalability evidence.
