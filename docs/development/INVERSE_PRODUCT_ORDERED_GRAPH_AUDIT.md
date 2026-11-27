# Ordered inverse and product graph closure

Acceptance review: 2026-11-28. Reference sources remain unchanged at Qualtran
v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`. Receipts retain the batch
start date `20261127` in their filenames.

## Interface, ownership and complete source responsibility

The existing nine Kaliski constructor/physical owners cover all 48 definitions
of `mod_division.py`. The five product/window owners cover all 45 definitions
of `mod_multiplication.py`. Their classical actions, resource models, notebooks,
source primitive comparisons and ordinary `circuit.append(op)` consumers were
already delivered. The missing responsibility was the source-defined ordered
symbolic graph; a signature-only Program does not preserve that graph's child
constructor expressions.

`formats/inverse_modular_library.py` now owns the algorithm and public wrapper
libraries. `_inverse_iteration_library.py` separately owns comparison, exchange,
arithmetic and iteration recipes. `product_modular_library.py` owns word doubling
and constant-product recipes. The existing `_ModularGraph` pairs constructor
metadata with the existing typed builder/exporter; it now retains word-sized
allocation/release, generic adjoint declarations and reference-safe inlining.
No second IR, stable export, opcode, schema or execution backend is introduced.

Words and history are MSB first. The inverse iteration owns a/b until their exact
source Free calls. The algorithm retains two 2n-bit histories. The public
forward wrapper returns them as a single 4n-bit `junk` output; the inverse takes
that same history as LEFT input. History is not clean private work. The public
wrapper reproduces the source's inlined algorithm and all allocation/release
roles. Reverse construction uses the existing exact adjoint and inline engines,
with explicit source decisions for each leaf and refusal for unknown leaves.
Constructor references are rebased as structured data; symbols are not embedded
in function names or replaced with numeric samples.

## Every newly resolved source profile

| Source-defined profile | Direct source calls | Native responsibility |
| --- | ---: | --- |
| Symbolic-width step3 | 6 | Half comparison, three-control predicate, two copies and exact cleanup |
| Symbolic-width step4 | 2 | Both word swaps |
| Symbolic-width step5 | 6 | Negative-control predicate, complements, two additions and cleanup |
| Symbolic-width iteration | 10 | Six layers plus two flag allocations/releases |
| Fixed-width/symbolic-modulus iteration | 10 | Same literal parameterized calls |
| Fixed-width/symbolic-modulus implementation, n=4 | 22 | Initial loads, eight iterations, final and zero-sentinel corrections |
| Fixed-width/symbolic-modulus public forward, n=4 | 35 | Inlined implementation, private lifetimes and retained history |
| Fixed-width/symbolic-modulus public inverse, n=4 | 35 | Reversed inlining, consumed matching history and restored word |
| Fixed-width/symbolic-modulus ModDbl | 15 | Signed widened correction and exact source bit rotation |
| Symbolic-width CModMulK | 5 | Allocation, two scale calls, conditional swap and release |
| Fixed-width/symbolic-modulus CModMulK | 5 | Literal reciprocal-expression branch |
| Fixed-width/symbolic-scale CModMulK | 5 | Same branch, without mathematical substitution |

These are logical call counts, not primitive gate counts or analytical model
annotations. The libraries deliberately supply no fabricated `counts` model.
All original constructor arguments, referenced children, operands/results,
array indices and directions compare against actual unchanged source libraries.
The source public inverse's add-from order is reproduced by the existing native
inliner and adjoint scheduler, rather than by loosening graph comparison.

Concrete native libraries feed the existing constructor registry with Qualtran,
Cirq and Kickmix imports blocked. Their whole CircuitIR, feedback, output aliases
and physical workspace events equal the accepted direct owners. Unresolved
parameters remain declaration/refusal cases for physical execution. Existing
complex amplitudes, coherent spectators, exhaustive small histories, all-history
certificates and clean work are reused by this exact execution-path replacement.
No new abstract leaf is treated as an executable identity.

## Literal source differences and domain witnesses

The constant multiplier uses negative integer modular inverse in its concrete
branch. Its symbolic-expression branch instead records
`Mod(Pow(k, -1), modulus)`, without that minus sign. The native ordered library
preserves this actual source expression and both original scale operands. It
is not presented as a corrected mathematical inverse. Reconstruct the native
operation from bound constructor values when selecting its concrete execution
path; simple substitution inside the symbolic source graph is not that path.
The existing numeric exact adjoint remains a reversal of the source calls,
including the swap and both scale calls, rather than another multiplier circuit.

Fresh original-only witnesses separately retain:

- At n=3,p=7, the original AddK(k=8) graph reaches XorK(k=8); its three-bit
  load fails with `Too-large classical value`. Native numeric refusal is kept.
- At step5 n=3 with r=s=7 and active subtraction/addition, the raw callback
  returns s=14 while its decomposed finite-width circuit returns s=6.
- At step6 n=3,p=5 with an odd active v=1, the raw callback drops the low bit
  to zero while the decomposed cyclic shift returns v=4. This is outside the
  reachable iteration promise, not a newly admitted input domain.
- At public n=4,p=9,x=3, the nonunit callback asserts and the decomposition
  attempts to free a nonzero register 2. Units and the documented zero sentinel
  retain the accepted clean-work/complex-state domain.
- Symbolic-width split/window cases and symbolic-modulus window implementation
  retain their original body refusals. SingleWindow with symbolic modulus has
  the original `NoneType` iteration failure; no table values are fabricated.

These witnesses do not patch the reference or overwrite its classical protocol
with the decomposed map. They also do not expand the source numerical domain.

## Physical erasure versus original models

Fresh source checks follow SingleWindow's actual QROAM adjoint wrapper to its
QROAMCleanAdjoint leaf for (n,w,p)=(3,1,3),(4,2,5),(5,5,13). All three original
Bloq decompositions fail. The direct SDK decomposition also fails, and default
SDK one-step decomposition returns None. Thus the original leaf has an analytical
model but no physical recipe at either protocol. Its original Toffoli estimates
are 0,0,8 respectively.

Native paper-based measured erasure is already implemented and separately
certified. Existing window/product suites verify source root connections,
forward lookup primitives, complex channels with retained logical outputs,
private work zero and specialized inverse restoration. Small lookup instruments
exhaust raw histories; larger product histories retain their separately labeled
selected trajectories and compositional evidence. No absent original erasure
gate stream is claimed to match the native stream. Global generic SDK, transport
and resolver qualification remains owned by its separate open parents.

## Independent resource accounts and acceptance

Reuse the [inverse](MODULAR_INVERSE_FILE_AUDIT.md) and
[product](MODULAR_MULTIPLICATION_FILE_AUDIT.md) actual primitive tables and
independent source models. At n=3,p=5, the existing public inverse emits 5,533
instructions, 1,292 T-family gates and 176 measurements, retaining 12 logical
history bits. Source peak is 38. Its default source T estimate is 1,040;
configuring the same source model for seven-T exact swaps gives 1,292. The
84 swap and 176 And model categories remain distinct from emitted primitives.
The new native artifact paths reproduce the whole accepted circuits, so they
claim no gate-count or storage improvement. Window erasure model substitution
remains explicitly different from an unmodified original model query.

The concentrated batch passes 385 cases and 32 integration cases. It reruns all
nine inverse layers, whole-file models/values, window channels, exact adjoints,
normal append replacement and source constructor families. Final graph/type/
format/architecture checks and the independent SDK lane are recorded under
`g2-inverse-product-graphs`. Reuse the unchanged complete original inverse
55-pass/1-skip/1-xfail and product source-suite receipts at matching frozen blobs.
Source-inherited failures and already recorded generic workflow limitations are
not reclassified as native implementation gaps or silently certified here.

Under the approved faithful-migration closure rule, Q-0573 and Q-0576 are
eligible after verified publication and final local checks. Every own-source
definition and direct algorithm dependency has a native owner, a defined graph,
or an explicit reproduced source refusal. General inherited SDK/serialization
intake remains separately tracked by Q-0866/Q-0876; this acceptance grants no
unqualified coverage of arbitrary wrappers. No additional parent IDs, G3 work,
CI wait or integration-PR merge is introduced.
