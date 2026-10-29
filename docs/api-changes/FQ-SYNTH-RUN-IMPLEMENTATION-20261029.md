# Synthesis Program run admission implementation authorization

The human user explicitly requested "合入963，下一步" after being shown PR #963
as the concrete fq.run(Program) contract requiring approval before implementation.
That instruction authorizes implementing the merged scope on a review branch.
Proposal merge: 8011f20300d7bf2f4b66ce6b15409fc8176d1224. This records the actual
conversation direction, not an invented GitHub review or independent developer
trial. Release approval and the two trial records remain pending.

The historical proposal/JSON remain unchanged. The new
`synthesis-program-run-v1-candidate.json` is the implementation admission record.
Classification: single_device_fast_path; maintained domain extension, unreleased
implementation candidate. The implementation PR remains reviewable and is not
automatically merged or released.

## Approved implementation boundary

- Move the existing private Services synthesis module to
  flagquantum.services.synthesis. Re-export its seven approved symbols from a
  thin flagquantum.synth package. One canonical Program class, no wrapper,
  subclass, registry, serializer or new executor. Remove the unreleased private
  module and migrate live callers; no permanent compatibility adapter.
- Add SynthesisProgram to the root run input annotation. Lazy root dispatch
  recognizes the canonical Program, rejects non-None overrides before plan
  access, and continues with the existing ExecutionPlan path. User subclasses
  are outside the initial factory-created profile. Runtime and Compiler remain
  unchanged and never import Services or the synth facade.
- Existing non-Program behavior, including existing Core circuit coercion,
  remains unchanged. An unrelated execution property is never a new admission
  hook; do not interpret old circuit-to-IR support as checked Program admission.
- Program.probabilities, validation, maps, read budgets, gradients, all current
  preparation semantics and final constraints stay on their existing authorities.
  Result remains ExecutionResult with ExecutionPlan; no schema/field changes.
- Exact current aggregate run signatures in the execution-plan and
  observable-output candidate contracts receive only this authorized input-type
  addition. The historical baseline and previous Ising/presence approval records
  stay intact. Extend the checker to validate the new narrow admission exactly,
  including canonical identity, exports, signatures and rejection policy. Do not
  regenerate snapshots or introduce an arbitrary accepted-type exception.

The historical public-api disposition file classifies original root symbols;
new synth domain symbols are registered in their dedicated candidate contract
rather than misclassified as historical root migrations. Root __all__ stays
unchanged. The new namespace is not a general Qmod compatibility or hardware
capability claim, and manual Program construction/field layout is not a promise.

## Verification and release boundary

Production fq.run(program) must match explicit-plan identity, full phases,
resources and names for logical/line profiles and both schedules. No planning,
compilation or synthesis during execution. All eight override families must fail
before plan resolution. Preserve normal Circuit lazy imports, old plan replay,
Ising/Lindblad behavior, gradients, cleanup and bounded probability reading.
Negative checker tests must reject approval, signature, namespace and type-owner
tampering rather than accepting arbitrary manifest edits.

Replace proposal-only non-admission guards with actual production acceptance
tests; retain the historical proposal as evidence. Independent users must test
the implementation branch and record task/time/blocker/help/output evidence
before release/activation review. Automated acceptance is not those records.
The candidate explicitly keeps release_authorized=false and trial records empty
until real evidence and the normal review process provide them.
