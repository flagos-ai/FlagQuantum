# Bounded sparse native predicate phases

Status: contract and actual-blocker probes; implementation pending.

## Observed shared blocker

On the validated QLBM integration head, unchanged whitebox and graph-coloring
predicate expressions in native qfunc both refuse at the eight-input phase
predicate budget. Their original programs use twelve and ten logical data bits.
Independent classical enumeration finds respectively six marked assignments of
4096 and sixteen of1024. Primitive admission alone does not complete either
program: whitebox retains ten Grover iterations, graph coloring one, and both
require exact uniform preparation and reflection at their original widths.

## Semantic boundary

Add a separately bounded sparse predicate phase path behind existing native
capture and register effects. Preserve the existing eight-input polynomial and
Boolean value limits. Do not globally widen ANF expansion or introduce an
application-specific truth table, dense oracle matrix or runtime Python callback.
Use the original expression as the source of truth. Bound static syntax,
truth-domain enumeration, integer arithmetic, cube cover size, emitted operations
and clean workspace before creating a circuit. Refuse unsupported syntax,
overwide inputs, nonfinite/static-invalid values or budget exhaustion explicitly.

For at most twelve distinct initialized raw unsigned input bits, evaluate an
admitted pure predicate exactly over its bounded domain. Map a verified exact
partition of satisfying assignments into disjoint signed cubes. Preserve read
witnesses even for zero coefficients and algebraic cancellation. Reuse the
existing checked physical locations, controlled phases, helper alias ownership
and clean-work lifecycle. Where a cube exceeds the existing direct-control
profile, compose bounded checked factors and uncompute them; do not weaken the
indexed-effect verifier. Keep constant global phase explicit. The implementation
must preserve arbitrary finite phase coefficients, coherent inputs, negative
controls and helper aliases, including the all-true/all-false predicate cases.

Resource feasibility and candidate budget completion remain distinct. Existing
synthesis must continue to refuse incomplete evaluation rather than silently
returning a best-so-far result. No optimality, asymptotic advantage or hardware
resource parity claim follows from this bounded path.

## Acceptance

- Independent exhaustive truth checks for comparisons, modulo by supported
  positive literal powers of two, bounded arithmetic and Boolean composition.
  Invalid shifts, divisors, negative unsupported values and oversized domains
  refuse before unbounded expansion.
- Full complex coherent amplitudes for signed cubes and nontrivial coefficients;
  test absolute global phase without phase alignment and every work amplitude.
- At least two different expressions share the same implementation. Retain
  original graph-coloring widths/one iteration and whitebox widths/ten iterations.
  Compare unchanged source, native qfunc and independent Circuit complete logical
  states, using declared layouts instead of assuming physical field order.
- Check satisfying/unsatisfying assignment boundaries, helper actual/formal
  aliases, target/control overlaps, uninitialized and Const reads, and actual
  qubit/work/gate/depth/memory refusal. Preserve all existing eight-bit tests.
- Fresh full216 source replay and six approved revisions; preserve raw resource
  baselines and record each resource delta. Independent local example acceptance,
  catalog/inventory and immutable ledger evidence precede any new coverage credit.
- User examples contain native FlagQuantum code and Circuit comparisons only.
  Original Qmod, license and source hashes stay test-side. No integration PR or
  remote full CI before all216 native quantum programs pass.

## Delivery split

1. Compiler contract, bounded pure predicate evaluation/partition and conformance.
2. Compiler capture/factor lowering integration and lifecycle/budget tests.
3. Two native programs, independent Circuit references and complete acceptance.
4. Local full-corpus replay, resource deltas, catalog/inventory and delivery ledger.

Notebook and broader Python workflow completion remain separate from quantum
program acceptance. The validated starting point is189/216 native quantum
programs,24/213 Notebook workflows and14/19 revision-nine merged scopes.
