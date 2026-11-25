# Checked native matrix phase estimation

The authorized original-example migration admits `qpe(callback, phase_result)`
for a zero-argument static lambda containing exactly one existing `unitary`
call. The initial scope is an unconditional program root and a whole writable
QNum counter with one to eight bits, all fractional, signed or unsigned. It must
have been initialized by allocate and unused since. Its numeric metadata stays
unchanged; there is no conversion to an array or relaxed numeric-helper borrow.

The callback retains existing static immutable matrix/literal and bounded target
view rules. Target storage must be initialized, writable and disjoint from the
counter. No host callback executes. A callback with allocation, drop, arbitrary
helper, nested control/QPE, multiple statements or keyword/unpacked arguments
is refused. General callback composition, flexible powers, nested or inverse
QPE, measurement, reset and dynamic execution remain separate work.

Compiler emits existing Hadamards, qreg.unitary with control_index and bounded
power 1..128, and qreg.inverse_qft. Existing matrix verification, initialization,
alias checks and actual resource budgets remain authoritative. No new physical
opcode, matrix decomposition, numerical kernel or ecosystem dependency is added.
Host invocation raises TypeError. Acceptance compares complete complex states
including retained environment and absolute phase against independent matrix
powers and dense inverse Fourier, with nonzero states, signed numeric labels,
bit order, all refusal boundaries and actual costs. Both matrix-QPE originals
must remain byte-identical; fresh216replay must retain all prior outcomes and
resources. This admits implementation, not release or optimality claims.
