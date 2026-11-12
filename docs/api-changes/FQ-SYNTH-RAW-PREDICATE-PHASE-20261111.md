# Raw integer predicate phase routing

The continuous native216 acceptance audit exposed checked raw integer phase
predicates being routed to the exact numeric front end solely because their
syntax contains subtraction. This introduces scratch for root predicates and
rejects scalar array elements in native Grover MaxCut. An in-memory routing
probe passes all 24 predicate-phase tests and 34 selected catalog checks,
including native and frozen-source Grover MaxCut comparisons. The probe is
exploratory, not production acceptance.

## Contract

For existing `control(predicate, lambda: phase(expression, coefficient))`,
actual signed/fractional input metadata determines whether the exact numeric
route is required. Raw unsigned integer arithmetic with subtraction may use
the existing exact polynomial/factored phase route. The presence of subtraction
alone must not force a numerical result register. Whole, borrowed and root
scalar reads retain their existing bit order and integer meaning.

This supersedes only the root raw conditional-phase routing restriction in
FQ-SYNTH-WHOLE-BORROW-PHASE-20261111. It does not change general root arithmetic,
fixed-point comparison support, assignment, numeric representation or APIs.
Negative differences in comparisons use mathematical integers; bitwise/modulo
operands must remain nonnegative over the complete bounded domain. Unsupported
negative operands must fail even if a condition later cancels. Python bools
must not be coerced into integer literals.

Keep actual numeric-format checks, eight-input predicate bound, phase-degree,
term, product, source, storage and operation limits. Static views must retain
their established checked lowering. Cancellation must preserve initialization,
Const and alias witnesses. Enclosing coherent controls count toward phase
input limits; any computed fallback remains checked and budgeted. No incomplete
candidate may be accepted and no original coefficients or iterations change.

## Acceptance

- Make all 24 existing native predicate-phase tests pass without weakening
  assertions, including negative differences and negative bitwise refusal.
- Run frozen and native Grover MaxCut at r=0/1/2 and compare every complex
  amplitude against the independent graph Circuit, including absolute phase.
- Add root/whole/partial consistency tests, signed/fixed-point fallback,
  uninitialized/canceled predicates, bool rejection and budget boundaries.
- Preserve whole-register borrow, controlled degree, QPE, within/apply and
  adjoint regressions. Re-run the complete original216+approved6 locally and
  preserve immutable source/resource baselines.
- Freeze production evidence and update the ledger. Final independent native216
  acceptance, Notebook workflow counts and release remain separate gates.
