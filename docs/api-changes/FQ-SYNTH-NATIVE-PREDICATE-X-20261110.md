# Native bounded predicate control of X

The authorized native-example migration admits comparison and Boolean-composite
comparison predicates in `control(predicate, lambda: X(target))`. This extends
the unreleased captured synthesis profile with no public export, signature,
register type or IR opcode. Existing direct scalar/array all-ones control keeps
its location-based path and semantics.

Reuse the existing bounded predicate graph, direct compute, indexed
`qreg.control_x`, uncompute and release. Comparison inputs are bounded by eight
quantum bits; target is one initialized writable scalar location, disjoint from
all predicate input registers. Reject uncertain aliases conservatively. Native
mutable helpers may use the same path on checked borrowed raw registers. Const
conditions remain read-only; Const targets remain rejected by register checking.
Unsupported expressions, host callbacks and runtime indices still fail closed.

Generated scratch must remain closed and clean. Reuse the preceding clean
predicate-region contract for adjoint and within/apply, with fresh local SSA
values on inverse emission. Do not weaken user allocation, Output escape,
ownership, initialization, role or actual resource-budget checks. This does not
admit controlling an arbitrary mutable helper, nesting arbitrary controls or
native in-place integer increments.

The first candidate paths are the original hypercube quantum walk and qfold.
Keep their original coin operators, conditional shifts, logical bit order and
absolute phases. Existing preparation, Pauli-axis rotation and unitary-region
building blocks must be verified in these complete programs before either is
counted as directly unlocked. The two-dimensional walk additionally needs
controlled in-place integer increments and is not a direct unlock of this change.

Acceptance includes independent complete Circuit states for both unchanged
originals and native examples, multiple step/angle variants, all predicate
branches with nonzero target XOR, complex coherence, nested inverse composition,
clean work, aliases and escaping Output refusal, Const/init/bounds and actual
width/workspace/operation/depth/memory refusal. Rerun focused and broader Compiler
regressions, all216frozen originals and six approved revisions. Keep source and
native resource differences explicit; Notebook workflows remain separate.
