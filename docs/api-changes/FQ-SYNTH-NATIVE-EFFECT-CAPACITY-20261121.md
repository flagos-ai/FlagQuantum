# Bounded native static-composition capacity proposal

Status: approved for implementation by the user on November 21, 2026.
Approval: the user replied “Approved” (translated from Chinese) to the explicit request to approve both this
proposal and the companion capacity proposal. The approval covers the bounded
changes and acceptance requirements below; it does not declare acceptance or
production release readiness.

## Observed requirement

The accepted CFD QSVT workflow retains two original systems: an eight-row
24-to-20-term Pauli block and a sixteen-row, 45-entry, six-band block. The
inverse sequences retain their respective 102 and 256 real phases. Existing
native operations express both sequences, but public capture refuses each at
`construction.python.budget: program exceeds 4096 effect operations`.

The corrected sixteen-row block independently passes its full 256-column
unitary and clean-work tests. Its actual offsets are -8, -7, -1, 0, 7 and 8;
a discarded three-band probe produced nonfinite angles and was correctly
rejected at binding. Neither failed probe counts as accepted native QSVT.

Bounded zero-pair/one-pair diagnostic captures contain 480/614 top-level
operations for the small sequence and 428/498 for the large sequence, including
declarations and return. Linear extrapolation gives 7180 and 9318 at original
phase counts. These are diagnostics, not successful original-size captures.
Raw source, logs and historical accepted packed-Circuit results remain intact.

## Proposed admission

Admit up to 16384 expanded effect operations for ordinary root native qfunc
program composition. Preserve the existing source AST-node, static-loop,
classical-binding, preparation, register and lifetime limits. Preserve final
instruction, qubit, workspace and memory ceilings and their failure categories.
No public export, signature, result field or serialized schema changes.

The existing QPE callback and QPE power-callback contracts explicitly retain a
4096-effect maximum. Those paths must continue to enforce that limit, including
ordinary callers containing QPE; the ordinary-root extension does not authorize
updating their candidate contracts or weakening their refusal fixtures. Qmod
capacity and compact-evolution admission also retain their current boundaries.
Implement admission through the authoritative frontend checks rather than
changing an unscoped shared constant for every capture path.

This is separate from the full-image preparation proposal. It admits neither
larger state vectors nor symbolic runtime angles nor arbitrary Python execution.

## Acceptance required after approval

1. Capture, verify, lower and execute both original nozzle sequences at 102/256
   phases, retaining all matrix entries, Pauli trim, right-hand sides and phase
   convention. No shortened sequence or dense response/SELECT replacement.
2. Compare complete complex states to the independent sequential Circuit/matrix
   reference with absolute error at most 1e-10, without global-phase alignment.
   Check non-success branches, ordering, clean work and zero-block inverse;
   preserve the original solution L2 bound of 0.1.
3. Use the native signal in adjoint and repeated composition after selectors
   become nonzero. Preserve conditional/global-phase interference and alias,
   lifetime and Output escape refusals.
4. Test the exact ordinary 4096/4097/16384/16385 effect boundaries, nested helper
   expansion and loops. Retain QPE, QPE power, compact-evolution and Qmod refusal
   boundaries. Repeated calls must not accumulate cross-program mutable state.
5. Record capture, verification, lowering, planning, execution, named readout
   and peak memory using existing observers. Respect explicit user resource
   budgets; report a refused budget rather than substituting a packed matrix.
6. Replay the original fourteen-cell Notebook, original/alternative source
   regressions and applicable compiler/default checks. Update protected admission
   records only after approval; retain prior raw failures and resource baselines.

Approval requested: the ordinary-root capacity increase and source-workflow
implementation above, with specialized profiles retained.

## Approved internal boundary

Compiler register verification, expansion and selection accept a private explicit
effect ceiling. Existing callers default to 4096, preserving the combined node
budget. Native ordinary capture requests 16384 effects and retains a separate
4096-node structural/expression limit. Source AST, loop and helper call limits
remain unchanged. Services forwards this ceiling only for already checked Python
programs; Qmod and direct register callers retain the default. Capturing any QPE
call restores the existing whole-program 4096 admission before the graph leaves
the frontend. No source flag or public serialized field is added.

Conformance covers exact effect boundaries, default refusal, repeated capture,
and QPE before/after larger regions. Domain implementation lands separately from
the Services integration.


## Implementation outcome on November 21, 2026

Compiler admission and exact closed-fragment normalization are implemented in
`62694e9a8`, with the Python-only Services selection in `0f8e252c4`. Specialized
QPE/Qmod/direct-register limits remain unchanged. Nested helper traversal retains
its separate bounded visit guard; a larger effect budget does not authorize
unbounded helper expansion. The 2152-test Compiler/QPE run and ten additional
nested/public-flow checks pass.

The original full inverses execute native streams in `36ee861ad`: 102 phases
use thirteen total/four work qubits and 40447 logical instructions; 256 phases
use twelve/three and 87012. Both original solution errors remain below 0.1,
complete complex-state errors below 1e-10 and work probabilities below 1e-24.
The fourteen-cell Notebook and named readout also execute. No packed response
matrix, shortened phase sequence or larger operation cap enters acceptance.
Full regression outcomes and intrusive diagnostic spans remain separately
recorded in the delivery ledger; these counts do not certify hardware efficiency.
