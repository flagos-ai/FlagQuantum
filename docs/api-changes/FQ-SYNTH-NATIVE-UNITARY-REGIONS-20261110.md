# Native checked unitary regions

Standing authorization covers migrating the retained examples to native Python
without Qmod loading or gate-index plumbing. Extend the existing pre-freeze
`within`, `apply` and `adjoint` markers; no public names or signatures change.

Compiler owns one inverse-effect implementation, extracted from the existing
Qmod adapter. Ecosystem retains source-located rejection and binding checks.
Native gate regions accept bounded static statements, mutable gate helpers and
nested regions on existing initialized storage. Preparation and action must
contain checked pure effects only and preserve storage ownership. Allocation,
Output publication, fresh arithmetic locals mixed with a unitary preparation,
measurement and host callbacks are refused. Existing arithmetic-local
within/apply and Const-only inverse calls keep their existing contracts.

An adjoint captures a checked forward effect sequence, reverses order and
inverts each effect, including constant global phase, QFT and matrices. A
unitary within/apply emits U, V and U-dagger with fresh threaded effect tokens.
Literal powers in phase expressions use the existing shared sparse Boolean-polynomial
lowering, avoiding a needless arithmetic result register. Existing helper alias,
Const, static-loop and operation/resource limits apply.
This does not admit arbitrary array-controlled gates or general mutable
callbacks. State-vector acceptance compares absolute complex amplitudes;
phase alignment is forbidden.

Acceptance covers retained QCE part I, the parameterized credit-card kernel
with two nonzero data pairs, and the entire randomized-benchmark sequence.
Independent Circuit references, coherent inverse-only checks, nested regions,
SX global phase, alias/Output/ownership refusal and resource budgets accompany
existing arithmetic-scope and Qmod regressions. Local implementation is
authorized; release claims and remote CI remain deferred.
