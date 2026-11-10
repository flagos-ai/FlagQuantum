# Native reversible regions with clean predicate scratch

The authorized native-corpus migration extends existing adjoint and within/apply
composition to compiler-generated, directly computed predicate scratch. No public
export, signature, register type or IR opcode is added. This is a Compiler-owned
change on the local single-device construction path.

A closed predicate region consists of one local declaration, direct compute into
that local, existing pure unitary effects, matching uncompute and release. It must
not publish, return, retain or escape the temporary. The register verifier remains
authoritative for initialization, roles, disjointness, unchanged compute inputs,
reversible cleanup and actual width/work/operation/depth/memory budgets. Native
register bindings, widths and ownership must remain unchanged across the region.
User allocation, arbitrary local lifetimes, Output calls and drop remain rejected.
Do not add allocation, compute or release opcodes to the global PURE_EFFECTS set.

Invert closed regions as atomic units: retain declaration/compute, invert the
unitary action, then retain uncompute/release. Reverse the order of adjacent
regions and pure effects. Re-emission must freshen local values and labels and
rebuild the effect chain, so a within/apply inverse never redeclares the forward
SSA value. Scalar gates, rotations, unitary matrices and controlled QFT retain the
existing authoritative inverse rules. This closed-unitary contract is distinct
from the broader already-verified Const arithmetic-callee lifetime expansion.

Also admit bounded composite comparison predicates for direct QFT through the
same predicate-capture path already used by direct Pauli-axis rotations. This
must preserve the eight-input-bit bound and fail before execution for overlap,
uninitialized inputs, unsupported expressions and ambiguous callbacks. Equality
still uses the existing captured Python expression semantics; no public Python
comparison-operator typing contract is changed.

The first user path is the original Fourier LCU example: prepare the original
four-entry controller distribution, coherently select QFT/inverse QFT, and
unprepare the controller with within/apply. The native model must preserve the
identity branch and full controller/data environment, without substituting a
handwritten Fourier gate sequence or a source-specific compiler intrinsic.

Acceptance requires complete complex-state comparisons against an independent
Fourier block matrix for the unchanged source, native default and two nonzero
input models. Test nested adjoint/within regions, cleanup, relative/global phase,
SSA freshness, disjoint predicates and targets, user allocation and Output escape
rejection, Const protection and actual resource budgets. Run focused Compiler
regressions, broader Compiler tests justified by shared inverse-rule changes,
fresh 216-source replay and six approved revisions. Record production hash
changes explicitly; preserve original resources and archived evidence. Notebook
workflow and resource-optimization claims remain separate.
