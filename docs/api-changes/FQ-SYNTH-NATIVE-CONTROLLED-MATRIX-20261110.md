# Checked native coherent control of static matrices

The authorized native migration of the five-category example library admits
`control(bit, lambda: unitary(matrix, target))` and its one-bit complement
`control(~bit, lambda: unitary(matrix, target))`. This closes a concrete blocker
in the retained harmonic oscillator program. Existing public names and
signatures do not change; no new physical opcode or numerical kernel is added.

The selector is one initialized writable qubit, either a scalar register or a
static QArray element. The target retains the existing static matrix, whole
register, bounded view and explicit scalar-list rules. Selector and target
physical locations must be disjoint. The selector plus target is at most five
qubits, matching the existing matrix verifier and decomposition budget. There
is no arbitrary computed predicate, multi-bit truth test, conditional allocation,
implicit branch, host matrix evaluation or automatic bound increase.

Capture builds an identity/active-unitary block matrix and emits the existing
`qreg.unitary` effect with checked indexed locations. The complete matrix retains
conditional global phases, including controlled negative identity. Existing
finite-unitarity, initialization, mutable borrowing, duplicate-location and
resource checks remain authoritative. Plain mutable helpers and root
within/adjoint reuse the existing pure-unitary effect contract. Const selectors
are initially refused because that contract conservatively treats every matrix
operand as writable and invalidates dependent cleanup proofs; this admission
does not weaken that ownership rule.

Acceptance requires independent complete-unitary columns for positive and
negative controls, complex/global phases, nonzero targets, entanglement,
indexed bit order and inverse composition. Alias, Const, uninitialized storage,
width, unsupported conditions, nonunitary matrices and actual resource budgets
must refuse. The harmonic oscillator must retain its original matrix entries,
widths, all source branches and exact source preparation completion, and match
an independent Circuit. Fresh 216-original replay must preserve pinned source
identities, outcomes and resource measurements. No release or optimality claim.
