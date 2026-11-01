# Native one-bit predicate phase admission

Date: 2026-11-01. Under the standing sixth-plan instruction, continue after
#984 (merge aeb818233f8bc2a54d17f34af35e098deabc2c1d; 57 successful checks,
three configured skips). This Ecosystem slice admits a source statement through
the existing private register phase contract. No public signature, export,
serializer, executor or Compiler contract changes.

## Source semantics

Admit `phase(expression, pi);` in the existing single-main, scalar-output Qmod
subset. The expression uses already admitted unsigned `+`, `*`, literal `**2`
and integer constants, must reference a quantum register and have an inferred
result width of exactly one bit. Its effect is `exp(i*pi*f) = (-1)**f`.
The coefficient must be the unshadowed symbolic token `pi`; no Python evaluation
or Classiq SDK is involved. A declared output named `phase` cannot be called as
the intrinsic, and an output named `pi` cannot supply its coefficient.

Reuse `_Expression` to infer and emit a verified arithmetic callee, and emit
`qreg.phase` borrowing its initialized operands. Reuse Compiler's linear effect
checking, alias handling, exact compute/mark/inverse cleanup, workspace reuse,
finite selection, budgets, target guard and existing Program/run/named reader.
Do not infer a smaller quantum domain from a particular prepared basis state.
Quantum-dependent expressions that are constant, such as `x*0+1`, retain their
overall phase. Pure classical/fixed phase remains outside this slice.

Source statement phase and the Python expression `output_mode="phase"` are
different entry modes: Qmod keeps the default value mode and returns its live
registers after ordered phase actions. The existing rejection of Qmod with the
Python-only output_mode flag remains. Default logical execution is supported;
the existing exact-phase target guard refuses the public basis/line conversions
that lack exact phase witnesses. No silent projective conversion is permitted.

Reject other arities, coefficients, shadowed intrinsic/coefficient names,
classical-only or multibit results, uninitialized borrows and unsupported source
syntax before planning. No comparisons, arbitrary angles, Input/Const roles,
helpers, arrays, scope/drop/free syntax or whole-library admission is added.

## Native reference and evidence

Classiq's [phase statement reference](https://docs.classiq.io/qmod-reference/language-reference/statements/phase)
defines expression first and coefficient second. The pinned library snapshot
2bfd835e73b6882fcc515a725098b04766c238e7 also uses that order in
`algorithms/search_and_optimization/QAOA/qaoa_max_cut.qmod` (Git blob verified
against the existing audit manifest). That original requires wider expressions,
real coefficients, arrays and helper calls and is **not** admitted here.
All examples in this slice are synthetic. Original fixtures, provenance,
licenses, hashes and the admitted-original count of 3/3 remain unchanged.

## Acceptance

Prove source validation and source locations, owned-IR handoff, basis inputs,
complete complex superpositions including global signs, interference, repeated
phase calls separated by mutations, subsequent arithmetic and zero workspace
under both schedules. Exercise text and Path entry, unchanged named outputs,
plan identity and resource/target refusals before planning. Memory failures use
the existing planning boundary and must precede execution.
Record actual default resources and a runnable public example without physical
indices or bitstring decoding. Public API, architecture and existing source
regressions must remain unchanged. Trials remain 0/2 and release_authorized=false.
