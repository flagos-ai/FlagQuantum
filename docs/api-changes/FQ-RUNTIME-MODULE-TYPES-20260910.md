# Runtime Module object typing

The user authorized continuing the Runtime Module typing remediation. This
change records the concrete existing objects rather than changing execution.

All three repairs are implemented. The user explicitly approved the protected
ExecutionResult.plan change on 2026-09-10. The implementation and the exact
candidate-contract annotation now include MPSProductionPlan. The API checker
previously rejected the unapproved annotation; the historical API baseline
remains unchanged.

## Changes and compatibility

- `Module.parameter_groups` changes its return annotation from
  `Mapping[str, torch.Tensor]` to
  `Mapping[str, torch.Tensor] | torch.nn.ParameterDict`. Named modules still
  return their registered ParameterDict; empty and flat modules still return an
  empty dictionary. Identity, mutation, registration, and checkpoints remain
  unchanged. Static clients may narrow to ParameterDict when using its methods.
- `ExecutionResult.plan` changes from
  `ExecutionPlan | RuntimePlanContract | None` to
  `ExecutionPlan | RuntimePlanContract | MPSProductionPlan | None`. This describes
  the plan already returned by `Module.execute_production_mps`; it does not add a
  backend or modify plan selection, summaries, evidence, or serialization.
- Internal `CompiledInstruction` becomes an `Instruction` subclass. Its required
  constructor arguments and dynamic parameter mapping remain unchanged. It is
  built exclusively from validated template IR, so it deliberately does not run
  the base initializer that would materialize currently unbound parameter slots.
  Core does not import Runtime. No new instruction protocol or adapter is added.

No call-site migration, new export, default change, or mapping copy is required.
The approved `ExecutionResult.plan` entry in
`contracts/execution-result-v1-candidate.json` changes with its annotation;
the historical API baseline remains unchanged. The verification scope is the
single-device fast path;
CPU tests cannot establish multi-GPU or multi-node scalability.

## Verification requirements

Verify compiled instruction substitutability, required constructor arguments,
unbound construction, live slot rebinding, repeated Module gradients, parameter
registration and identity, and preservation of MPS plans through result tensor
conversion and detachment. Run targeted Module and MPS tests, CPU smoke/unit
regression, full-package strict mypy, and the API and architecture checks.

## Verification before approval

Before approval, the two implemented repairs reduced full-package strict mypy
from 28 errors to 26 errors in nine files (448 source files checked). The MPS
result-plan mismatch remained visible pending approval; 25 errors was then only
an intermediate count with the proposed annotation.

Focused validation passed 76 tests. CPU smoke/unit selection passed 1,938 tests,
with 13 skipped, 1,802 deselected, and one PyTorch complex-module warning.
The new test file passes a standalone strict check. API, architecture, formatting,
lint, and ownership checks pass. No GPU or remote execution was performed.


## Verification after approval

Full-package strict mypy now reports 25 errors in eight files across 448 source
files. The Runtime Module plan mismatch is removed; remaining diagnostics are
24 untyped Triton decorators and one PyTorch LinAlgError export annotation.
Focused MPS and result validation passed 23 tests in 3.95s. API, architecture,
Ruff, and Black checks passed. Logs use /private/tmp/fq-approved-mps-plan-*.txt.
This annotation-only change did not rerun the preceding 1,938-test CPU selection.
