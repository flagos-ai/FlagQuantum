# Bounded native phase predicate syntax

The standing authorization to complete all native quantum counterparts covers
this additive captured-Python syntax admission. No public export, callable
signature, Stable Core schema, register effect or Runtime route changes.
Implementation is authorized for the unreleased synthesis profile; release is
not. The integration-owned contract lands before Compiler implementation.

For `control(predicate, lambda: phase(value, coefficient))`, support Qmod-familiar
Boolean combinations `&`, `|`, `^` and `~` over scalar quantum bits and integer
comparisons. `~` complements a predicate; it does not introduce signed-word
bitwise arithmetic. Integer comparisons may include exact `+`, `-`, `*`, bounded
literal powers, unsigned `&`, `|`, `^`, and `%` by a positive power-of-two literal.
Subtraction in a predicate uses its mathematical integer value, including
negative differences, without wrapping an inferred unsigned temporary.

Compiler's existing `expand_phase` owns predicate evaluation, truth-table bounds,
Boolean phase coefficients and physical-location identity. The native adapter
adds AST spelling only, then emits the existing checked phase effects. New
predicate forms, including subtraction comparisons, must take this same route
at a root and in a borrowed helper. Existing scalar control paths and previously
supported arithmetic comparison paths retain their behavior. Arithmetic phase
values keep their existing grammar: admitting bitwise/modulo conditions does not
admit those operators as unguarded phase values or general numeric computation.

At most eight distinct input bits may participate in a phase condition. Existing
node/depth, word precision, angle, operation, qubit, workspace and memory ceilings
remain. Every operand is validated, even when a Boolean constant or algebraic
cancellation makes its contribution zero. Uninitialized/Const effects, aliases
and shared source locations use the existing verifier. No predicate is evaluated
by running a qfunc body or an arbitrary Python callable.

Python `and`, `or`, `not`, chained comparisons, general integer inversion,
non-power-of-two modulo, signed/scaled word reads, wider predicate enumeration,
non-phase controlled mutable helpers and generic quantum callback ports remain
outside this admission. A Boolean inversion inside integer arithmetic, such as
`(~bit) == 1`, is not part of the syntax; use `~(bit == 0)` in the predicate tree.
The control signature remains `QNum | QArray[QBit]`; no implicit Python boolean
truth conversion or new Boolean public type is introduced.

The direct vertical paths are the frozen Grover-operator library example,
the >=4 MaxCut Grover example, and the negative-difference oracle workshop.
Their native versions specialize each concrete oracle as a typed phase helper,
compose uniform preparation and the absolute-sign reflection, and preserve the
original iteration counts. This adds no generic phase_oracle/Grover API and
makes no arbitrary nonzero-marker XOR contract claim.

Acceptance requires exact original/native/independent Circuit full-state checks,
including global phase, declared bit order and hidden-work probability. Cover
negative differences, the >=4 marking boundary, products modulo four, root/helper
agreement, Boolean complements and nested combinations, coherent phase values,
shared views and spectators. Refuse nine-bit predicates, illegal arithmetic and
canceled illegal reads before allocation/execution, and exercise actual resource
ceilings. Simplify the already accepted SAT example to the familiar predicate
spelling only after equivalent full-state checks pass. Preserve frozen source
hashes/resource rows, replay all216 originals and six approved revisions, update
immutable evidence and the delivery ledger, and keep Notebook coverage separate.
