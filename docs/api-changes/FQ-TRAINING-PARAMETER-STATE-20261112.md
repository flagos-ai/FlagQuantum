# Training parameter state approval

On 2026-11-12 the human user explicitly approved the exact
[v3 proposal](../development/TRAINING_PARAMETER_STATE_PROPOSAL.md) and its
replacement fake. This authorizes the qualified checkpoint v3 writer and strict
v1/v2/v3 reader behind the existing save/load interfaces.

V3 adds only `parameter_training`, containing exact registered parameter flags
and ordered optimizer parameter names. It uses the approved domain-separated
captured-builder or qualified symbolic-Circuit workload identity. Restore keeps
Parameter identities, restores flags, clears stale gradients, validates before
mutation and rolls back failed restoration. Unsupported hooks and parameter
storage must be refused. Existing v1/v2 algorithms and caller-owned flags remain
unchanged; old readers reject v3. Unqualified frozen sources refuse save.

Public signatures, exports, Module state_dict, restore metadata, initialization,
backend defaults and optimizer arithmetic remain unchanged. Historical snapshots
must not be regenerated. This approval admits no additional Compiler import,
Simulation dependency, distributed or accelerator promotion, performance claim,
shared checkpoint aliases, or gradient accumulation recovery.

The contract and fake precede the Runtime implementation in the review series.
Implementation requires production conformance, trajectory/rollback tests,
historical compatibility and the applicable repository checks.
