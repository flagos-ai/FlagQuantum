# Chebyshev constructors and exact dependency closure

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
The source, test and notebook blobs asserted by the reference fixture are
`327d0ace4b411c9d8290d59d496e717b0c9e3915`,
`2bc29ea862924b8e213d94b9f3f15ff1c53414b9` and
`30fa0a9688a8704c7ed9cc32619ca7ed96c93ada`. The fixture also requires
Qualtran 0.7.0 and the actual Kickmix 0.1 Simulator.

## Interface and ownership

The complete `block_encoding/chebyshev_polynomial.py` file contains ordinary
and scaled polynomial constructors. Both retain ordered MSB-first opaque
system/ancilla/resource THRU words, omitting zero-width words. These logical
words belong to the caller. Child operations, reflections and scalar carriers
own their private clean work and measurement feedback.

Compiler preparation owns the literal reflection/B/reflection/B-adjoint pairs
and the odd final reflection/B pair. Exact inverse ownership reverses the
actual calls. Order zero imports no unused child workspace. Positive control
uses explicit already-specialized child and reflection Programs; generic
inverse remains distinct from guessing a self-adjoint block. The shared call
sequence also replaces the existing Algorithms callback stage loop without
changing its public binding signature or callback ownership.

Scaled composition retains the single source linear-combination property call.
Ecosystem captures that actual property, including the original NumPy
coefficient transform, normalization and precision, and immediately replaces
its children with native directional Programs. Compiler does not infer alpha
from an operator body or retain an external SDK object. Default LCU SELECT
reuses its existing packed/indexed owner with an explicit positive child
callback for polynomial terms. Generic scaled controls expose QBit even when
the specialized property child exposes an opaque one-bit control word.

Formats owns scoped constructor recipes, exact adjoints and ordinary/positive
child or coefficient ownership. Negative and grouped controls reuse existing
predicate computation and exact erasure. The existing Unitary owner reports
its authoritative exact pair to both Phase and Chebyshev constructors; Phase's
duplicate directional reconstruction is removed. Unknown ordinary, positive
or coefficient owners refuse rather than being inferred from signatures.

The existing scalar recipe is shared with the reflection owner. An uncontrolled
nontrivial zero reflection retains its actual registerless GlobalPhase call;
the carrier is allocated, restored and freed below that call boundary. No new
Core opcode or public parameter, control or serialization contract is added.
Proposal 119 remains unapproved and unused.

## Verification scope

Ordinary original even/odd child fixtures cover orders 0 through 5 in both
directions, full complex logical operators, source call dependencies and named
operands, zero work and erasure certificates. Non-Hermitian T children retain
the literal alternating directions instead of an assumed matrix power.
Measurement-bearing positive child fixtures cover orders 0, 1, 2 and 5.
Constructor tests cover original examples with explicit source contexts,
generic inverses, both scalar control polarities and grouped 10 controls.
Small raw histories are additionally enumerated with complex input amplitudes.
Literal native definition matching reuses separately verified dependency
closures while the constructor root's own call graph is checked anew.

Both original high-order scaled examples compare complete sparse complex
outputs for a superposition of system inputs, with clean logical ancilla and
resource inputs. Zero/one representative raw histories, all-history erasure
certificates and exact native inverse roundtrips check private work zero.
This is not an exhaustive dense operator over arbitrary logical scratch inputs
or an enumeration of exponentially many large-example measurement histories.

The original scaled-even direct Cirq recursive route fails at
Controlled(SingleQubitCompare), whose non-THRU registers have no such control
decomposition. The unchanged source's CompositeBloq.flatten followed by
to_cirq_circuit uses its typed LCU specialization and succeeds: 1703 typed
instances, 1060 Cirq operations and 1024 nonzero complex output amplitudes.
That original executable route supplies the scaled-even numerical reference;
the direct-route failure remains separately recorded. The sparse test evaluator
now accepts literal identity instructions instead of rejecting the explicit
identity leaves in that stream.

## Resources and source boundaries

Ten directional resource rows in
`g2-chebyshev-constructor-resources-20261125.json` separate actual primitive
inventories and private/live peaks from original QEC/logical-peak estimates.
For the ordinary even example, execution emits 88 T-family, 196 CX, 56 H,
8 measurements, 12 CZ, 8 reset, 8 phase, 104 X and explicit identity/atomic
controlled-H leaves, with two private/eight live wires. Its source model
reports eight CSwap, eight And, 192 Clifford, eight rotations, eight
measurements and peak eight. Model CSwap/rotation counts are not emitted
primitive instructions.

The scaled-even example emits 426 T-family, 997 CX, 291 H and 61 measurements,
with 14 private/37 live wires. Its source model separately reports 26 CSwap,
61 And, 740 Clifford, 20 rotations, 61 measurements and peak 37. The scaled-odd
example emits 178 T-family, 462 CX, 130 H and 34 measurements, with 14
private/32 live wires, versus six CSwap, 34 And, 348 Clifford, 30 rotations,
34 measurements and peak 32. Allocation release is checked separately from
the complex-amplitude proof that private wires return to zero.

For odd order three with a nonzero signal register, the actual source executes
three reflections while its direct model declares two. Both routes are kept;
neither reference counts nor source code are repaired to claim agreement.
GlobalPhase remains source-atomic with zero analytical gate cost, while its
native scalar carrier has explicit X/phase instructions and private storage.

All four original producers are probed in both directions at depths 0/1/2.
Depth zero works with explicit native ownership for omitted LCU metadata.
Deeper enum, object-array and ragged-array serialization failures remain
recorded in `g2-chebyshev-wire-boundaries-20261125.json`. Symbolic declarations
retain the existing concrete-execution admission boundary. Arbitrary child
constructors still require their own exact native owners.

Only ChebyshevPolynomial occurs in the frozen 303-constructor inventory;
ScaledChebyshevPolynomial contributes no additional inventory identifier.
The batch regression passes 439 cases. A further 18 focused control cases
cover both orders of applying adjoint and control, including six additional
inverse-before-control cases. Another 145 local integration cases pass in a
separate scope. Unchanged non-slow source tests report 25 passes, two original
count-mode skips and three original serialization xfails; four slow tensor
tests and the notebook are separately deselected. The native full-small and
sparse-large numerical checks above retain their explicit scope rather than
claiming those deselected source tests ran. Fourteen implementation/test files
pass strict typing. Black, Ruff, team scopes, language, architecture, IR and
public API checks pass; dependency findings match the existing 720-line baseline.
