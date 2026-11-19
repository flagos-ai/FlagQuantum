# External initial-state execution approval

On 2026-11-12 the human API owner explicitly approved the concrete proposal
reviewed in local commit `c420795e2e6d668a2039cbc986945449f7da345f`.
The approval followed the request to add the fq.run initial_state keyword, admit
the versioned initial-state plan extension and enforce the associated FQX save
constraint. The user's response translates to "Approved".

The [approved specification](../../contracts/execution-initial-state-proposal.json)
and [candidate contract](../../contracts/execution-initial-state-v1-candidate.json)
define the scope. The name initial_state denotes per-execution quantum data;
program preparation remains an ordered program operation. No root export,
fq.plan argument, constructor deprecation or universal input specification is
approved. The existing public baseline and historical approvals remain intact;
the checker may admit only this reviewed signature addition.

The first implementation is exact, unbatched CPU PyTorch statevector execution
from an ordinary static Circuit, with full-state or probability outputs and
first-order input/gate gradients. Its effective ExecutionPlan admits compatible
new inputs without compiling again. The extension contains no amplitudes or
amplitude digest. Existing zero-input plans retain their semantics.

Saving a marked plan without a result must fail before writing because the
existing program-only archive route would discard its input requirement.
Saving with a result retains the plan requirement and outputs; loading does not
execute or restore the original input or autograd graph.

This authorizes implementation and the exact candidate/checker update. It does
not authorize expanding to other engines, dynamic/QEC external inputs, remote
execution, released capability promotion or claims of bounded process memory.
Replacement/conformance, artifact tests and no-input regression evidence remain
acceptance work, not facts established by approval.
