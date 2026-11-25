# UP-06 checked predicate phase oracles

This private Qmod profile changes no public exports or serialized owned IR.
A predicate qperm has one Const whole array or flat fixed unsigned record,
one distinct writable single-bit qbit/qnum marker, and optional exact scalar
integer ports. Its admitted body is exactly marker ^= expression. The marker
cannot appear in the expression or escape, and cannot alias the borrowed input.
Negative intermediate differences remain mathematical integers for comparison;
unsigned subtraction must not wrap before comparison. Only bounded integer
arithmetic and power-of-two unsigned modulo are added to the existing phase
predicate evaluator. Input, monomial, degree and expansion caps are unchanged.

phase_oracle accepts a named predicate or a two-argument static lambda forwarding
exactly its input and marker, with literal integer specialization. It multiplies
marked amplitudes by -1 and leaves the input word unchanged. A direct checked
predicate call XORs any initialized marker, including nonzero/coherent states.
Both forms share existing conditional phase lowering; the XOR form is H-Z-H on
the disjoint marker, without a new physical template or owned operation.

Flat record layouts are checked nominally and fields retain their declared
numeric significance. Existing record views are reused. Closed Grover callback
binding permits a same-named formal which shadows its own actual register;
other quantum captures, extra aliases, Output escape and mutation remain refused.
grover_search composes explicit uniform preparation and bounded Grover rounds.
General callback ports, mutable record aliasing, nested records and arbitrary
predicate side effects are outside this profile. Callback structure and alias
checks precede zero-iteration erasure; Compiler remains authoritative for actual
initialization, resource budgets and physical costs.

Accept unchanged grover_operator.qmod and oracles_workshop.qmod, plus grover.qmod
at r=0/1/2, against independent public Circuits and full complex128 vectors without
phase alignment. Check all sixteen workshop inputs, including negative x-y;
grover.qmod marks cut size >=4 (not the previous <=4 example). Direct XOR tests
must include both marker basis states and coherent markers, clean scratch and
bit order. Reject capture/alias/Output escape, malformed predicates and actual
resource excess. Recheck all 216 originals and preserve previous source/license
and AO/PF/resource evidence. Update the ledger; no PR/full CI before all-original
acceptance.
