# Native scalar Const admission through Compiler-owned register calls

Date: 2026-11-01. Continue after #988 (57 successful checks, three configured
skips; merge 9ed21b991da11ede194cea9e2eadc0005c6cc891). Reuse its verified
Const/fresh-Output contract. This slice changes only Ecosystem production code;
public signatures, exports, snapshots, reports and execution remain unchanged.

## Source contract

Helpers admit `const name:qbit` and `const name:qnum<N>` for literal unsigned
integer widths. Output scalars keep their existing syntax and inference. Source
parameter order may interleave Const and Output; preserve positional matching
then emit the Compiler ABI with all Const actuals before fresh Output actuals.
Keep exact scalar kind and resolved width matching. Const actuals must have a
known width and be initialized; they may alias other Const actuals. Output
actuals remain fresh and disjoint from every argument. Const is basis-value
preservation, allowing phase/entanglement, not classical const or a promise of
an unchanged reduced state.

Main continues to accept Output scalars only and owns its preparation. Reject
unqualified mutable parameters, consuming Input, input const, output const,
bare-width Const qnum, arrays, views and classical parameters. Helpers cannot
capture caller variables. The existing expression and one-bit/pi phase grammar
remains unchanged. Per the closed Compiler proof, Const can be arithmetic/phase
input or nested Const, but never a gate target, initialization/compute destination
or nested Output, including cancelling gate pairs. Callers may mutate values
after a helper returns. No arbitrary phase, local scope, free/drop or within/apply
admission is added.

## Replacement and authority

Remove Ecosystem's Output-only effect substitution and role/alias checker.
Translate every source helper into the existing owned register callee with
Const entry arguments and fresh Output declarations. Emit one qreg.call at the
source call site. Qualified arithmetic definitions and register helpers are
deduplicated in the existing handoff; all definitions, including unused helpers,
reach the same Compiler verifier before any Program is returned. Output width
inference can construct records before validation but no unchecked records may
cross the adapter boundary.

Compiler alone owns initialized/fresh/basis-preserving roles, aliases, complete
returns, effect order, graph/depth/expansion limits, substitution, scratch cleanup,
phase target checks and selection. Ecosystem retains lexical names, scalar
kinds, widths, source coordinates and version context. Closed adapter syntax
records remain private and are discarded at the owned-IR boundary. Reuse the
same Services Program, planning/memory guard, Runtime and named reader.

Error class and source locations are retained. Role/alias/expansion violations
now use the existing Compiler diagnostic categories rather than a second native
role namespace. Existing byte/token/expression/declaration bounds remain;
Compiler owns the 4096-node/visit and 32-call-level bounds. No new dependency,
contract type, dispatcher, serializer, numerical implementation or public option.

## Acceptance

Prove a helper call stays owned qreg.call until Compiler lowering, including
Output-only replacement without changing the execution consumer. Test source
parameter reordering, inferred Output/exact Const widths, all sixteen two-bit
inputs, coherent states and external correlations, repeated Const aliases,
phase/global signs/interference, caller mutation and workspace cleanup under
both schedules. Exercise text/Path, invalid unused definitions, uninitialized
and empty borrows, forbidden Const writes, role combinations, cycles/bounds,
target/memory/resource refusals and no execution during synthesis. Give a complete
public example with no gate design, physical indices or bit decoding. Preserve
source/license hashes and 3/3 originals; trials remain 0/2 and
release_authorized=false. This is not full native roles or all-216 compatibility.
