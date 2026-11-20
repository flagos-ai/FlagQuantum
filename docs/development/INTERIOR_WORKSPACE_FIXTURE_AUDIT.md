# Interior workspace fixture

The Q-0493 implementation and Q-0494 test match their frozen blobs at
Qualtran v0.7.0 commit `bcb17bb25360ed68ce3bc084645db3e929799389`.
The original test passed unchanged.

The native fixture composes the existing register builder, atomic SWAP
and workspace binder. The call sequence is Allocate, Swap, Swap, Free.
The independent source workspace reader binds the source allocation
and release to the same caller-owned slots. Widths 1, 2, 3 and 10 match
the source gate operands, ordered outputs and both lifetime events. The
symbolic shape retains two `n`-bit logical words and a `3n`-qubit peak,
including `n` work wires. The source model counts `2n` Clifford calls.
The native physical stream has `2n` SWAP instructions, zero T-family
instructions and zero measurements.

For widths 1 through 3, `circuit.append(op)` maps a coherent
superposition of all x values on the y=0 subspace to x=0 and y=old x,
preserving the full source complex vector. Every workspace bit is zero
at release.

The original classical callback rejects nonzero y at Free. In the
native physical stream the same input remains in the middle word,
because the binder tracks the lifetime and does not inspect the
quantum state at release. That input is outside the fixture's clean
ancilla contract. The difference is retained by a separate test. The complete-source follow-up
below proves matching numerical protocols and classifies this physical input
as outside the source clean-Free precondition. Q-0494 is covered by
the direct call, gate, ownership and resource comparisons.

The workspace regression passed 183 tests. The shared source workspace
reader's 144 serial shaped tests completed before the combined consumer
run timed out. The cross-pool consumer suite passed all 24 tests in an
isolated rerun. Strict typing, formatting, team scope, architecture, IR
and public API checks passed.

## Complete source-file closure

Q-0493 is accepted after complete transport, nominal type, projected-tensor and
classical protocol closure. The source has one class, its width field, one
signature and one builder method. These map to the existing native example's
signature and interior Allocate/Swap/Swap/Free Program. Actual original libraries
round-trip bytes and object identity at depths zero, one and two for widths one,
two and three. Each resolved root and all four direct callee signatures now match
the original canonical types. Exact emitted SWAP operands, work ownership,
returned aliases and original independent resource costs are also checked.

A genuine one-bit nominal type mismatch was reproduced from the unmodified batch
base. The native example used QAny(1) for its root and word-swap declarations,
where original Signature.build declares QBit. Both declarations now reuse the
example's existing `interior_workspace_signature`. Allocation still retains the
source QAny work descriptor, and loose call compatibility remains with the
existing register owner. No new compatibility rule or protected admission is
introduced. The same example and its callers continue to use ordinary append.

The example also now stores the complete symbolic root decomposition. It keeps
Allocate, two Swap calls and Free, with the original width and ownership edges.
The symbolic Swap child remains a typed declaration without a physical body,
matching the original Swap decomposition refusal. Physical binding refuses that
unresolved child. Direct native construction retains the supplied symbol and its
assumptions. Original foreign argument reconstruction loses SymPy assumptions;
its round-tripped object is unequal to the original assumed-width object. Native
wire transport preserves the exact bytes and the same decoded nominal symbol.
That source limitation is asserted at all three depths without reference edits.

The former open nonzero-y item compared source classical checking with native
physical lifetime tracking. Every small classical x/y pair at widths one through
three now runs through the native selected-call state using existing Allocate,
Swap and Free numerical owners. For y=0, both return x=0 and y=old x. For nonzero
y, both reject at clean Free. The native state retains its completed prefix and
the caller inputs are unchanged. No new classical action is invented.

The complete native projected tensor at widths one and two now matches every
entry of the unchanged source tensor, including its zero columns for nonzero-y
inputs. Native suppliers obtain the Swap matrix from the existing bound SWAP
body and unitary evaluator. Allocation and Free use explicit zero-state tensors
through the existing register tensor owner. This is the pure tensor protocol.
It preserves the original partial-isometry action without changing physical
lifetime semantics. Existing append checks retain full complex amplitudes and
zero work on the valid y=0 subspace. The existing outside-subspace physical
residual test remains explicit and makes no clean-work claim for nonzero y.

Original zero-width decomposition omits x/y and fails with missing arguments;
negative-width depth-two serialization fails with negative dimensions. These
source-invalid cases remain recorded separately. The physical stream remains
2n SWAP, zero T-family operations and measurements. Its independent original
model is 2n Clifford and 3n peak qubits. No physical gate or workspace is added
by the nominal-signature or symbolic-declaration correction.


The complete follow-up passes 147 concentrated cases, 17 local integration cases
and 36 Cirq 1.6.1 cases. The unchanged original interior test passes. The example
and new test pass strict mypy, Black, Ruff, scope, architecture, IR and protected
API checks. Four frozen source files and unchanged shared execution owners are
verified in `g2-interior-foreign-evidence.json`. The real one-bit declaration
mismatch was reproduced from the prior committed example. No full-repository
certification or physical cleanup outside the valid subspace is inferred.
