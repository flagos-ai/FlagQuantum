# Constant-angle phase gate for one original Qmod case

Follow #1001 at f79a806ba1b456b5d3d79eb636fbad2b107e0bd7. Admit the pinned
standard_gates/PHASE.qmod unchanged and raise original-path coverage from six
to seven only after original-size execution and full-state verification.

Compiler extends the existing private qreg.gate record: gate="phase" requires
exactly gate/index/angle attributes, with a finite Python float angle in radians.
Other gates keep exactly gate/index; no change to existing producers. Preserve
initialized storage, numeric index, Const refusal, linear effects, mutable-target
invalidation, call expansion, budgets and source locations. Lower directly to
existing Core Instruction("phase", wires, params={"theta": angle}); reuse its Runtime
and numerical kernels. PHASE(theta)=diag(1,exp(i*theta)), including its global
phase difference from RZ. No custom gate, matrix, executor, public API or IR type.

Ecosystem admits uppercase PHASE(nonnegative_integer_literal,qbit), with the
literal <=2**53 so conversion to the internal float is exact. This is a bounded
first source profile for the original PHASE(1,q), not general CReal support.
Decimal/negative/symbolic/computed/quantum angles and numeric/array targets remain
refused. Lowercase phase(expression,pi) remains its distinct predicate operation.
PHASE becomes a reserved intrinsic and cannot be shadowed by a helper.
Do not broaden within/apply actions or Const gate permissions implicitly.

Acceptance: unchanged source and license pins, original file/text execution,
coherent relative and global phases, reference entanglement, helper composition,
Compiler/native producer replacement, malformed attributes/angle/index/storage
and role refusals, unchanged old gate records and budgets, actual resources.
Original zero input alone cannot validate the phase convention. Track seven
paths/six blobs separately from seven algorithm families; update whole-corpus
admission separately from execution. Trials remain 0/2; no release claim.
