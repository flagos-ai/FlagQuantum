# Native owned hidden storage and explicit drop

The authorized native example migration admits root bare annotated quantum
locals and `drop(target)`. Locals use the existing QBit, QNum numeric metadata
and QArray[QBit] annotation rules. They require definite initialization, a
bounded resolved width, and one explicit drop before root return. A declaration
is not allocation. Fresh preparation and allocate retain existing checks.

Drop removes access, not quantum information: it emits existing qreg.drop on
qreg.hidden storage, retaining the complete environment in the circuit. It
performs no reset, uncompute, measurement, physical reuse or zero assertion.
Named probabilities marginalize the retained environment. Dirty hidden storage
counts toward physical width and memory budgets, separately from clean work.
Free and proven-zero release are not added or aliased to drop.

The initial scope is root unconditional declarations and drops only, with whole
owned hidden registers. Signature ports, Const, views, borrowed aliases, helper
ownership, conditional/loop/region storage changes, missing drops, duplicate
drops, and every post-drop read/write/call are refused. Existing arithmetic
within locals keep their separate cleanup semantics. No new IR opcode or
numerical kernel is required. Host invocation raises TypeError.

Acceptance covers entanglement and reduced density, exact full environment
state including global phase, hidden scalar/numeric/array preparation, mutable
borrowing, all refusal boundaries and actual physical/memory budgets. Source
examples require independent Circuit comparisons and honest source precision
handling before native coverage increases. Existing original-corpus outcomes
must be replayed. This is implementation admission, not release authorization.
