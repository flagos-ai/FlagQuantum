# Relative swap constructor and exact inverse dependency disposition

Reference: complete `swap_network/cswap_approx.py` and its unchanged tests at
Qualtran v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`. Source/test blobs
are `362798e240461e6fa6269af8820e9436e5645b4a` and
`665e1cb122308cb86d7494728ae85d540275806a`. Fixtures require actual Qualtran
0.7.0 and Kickmix 0.1 Simulator. No reference source is modified.

## Ownership and interface

Compiler formats owns width metadata and scoped exact inverse recipes. Existing
Compiler primitives owns the source S/H/T/CX phase segments and typed exchange
graph. Compiler bits owns the fanout tree. No Algorithms import, duplicated
exchange algorithm, backend selection or quantum numerical execution is added.
Control is a QBit and x/y are opaque MSB-first THRU words. All remain owned by
the caller; neither primitive allocates work or introduces measurement/reset.

Both shared typed owners now support an explicit exact adjoint. The graph
adjointer reverses the original calls and Split/Join views, maps S to S-dagger
and T to T-dagger, and preserves self-inverse H/CX. Relative swap additionally
uses the exact reversed fanout child. A forward fanout is logically self-inverse
but its forward tree is not the generic source Adjoint call order. The same
distinction applies to the full relative swap. No measured stream is reversed.

## Source and execution scope

All three original factories resolve forward and inverse at depths zero and
one. Concrete four-bit and 64-bit examples have executable roots and complete
ordered recursive call/operand comparison. The symbolic factory preserves its
logical declaration. Both concrete and symbolic resolution work with source
imports disabled. Source symbol names and typed ports are retained.

Widths one, two and three compare the complete complex operators, including
source relative signs and phases; global phase is not factored out. Width eight
checks the complete unmodified source forward primitive stream, its exact
measurement-free inverse, and complex action on occupied words entangled with
an additional spectator. Independent recursive comparison checks the actual
source generic Adjoint structure in both directions. The primitive observer
retains S-dagger explicitly. Small full operators and zero private leases
complement one another. There are no measurement histories beyond the empty
history for these primitives.

Fanout inverse widths one, two, three, five and eight compare original recursive
calls and primitive operand dependencies on a nontrivial physical word ordering.
Ordinary `circuit.append(op)` and JSON roundtrips use the actual relative-swap
instructions. Previous complete source-file controls, inverse, value, notebook
and model evidence is reused in the concentrated regression.

## Separate resources and retained boundaries

For word widths four, 64 and eight, both directions emit respectively 16, 256
and 32 T-family gates, 23, 383 and 47 CX, and nine, 129 and 17 total wires.
Their separate source models have the same T totals, 87, 1407 and 175 Clifford
counts, and matching logical peaks. No private work, measurement or reset is
introduced; all resource rows return zero live private leases.

Eight-target fanout emits 15 CX in either direction on nine wires. The original
forward QEC model charges one atomic Clifford, while its generic Adjoint model
charges 15 Clifford after expansion. This source asymmetry is retained rather
than replacing a model count with the measured primitive inventory.

Symbolic physical execution, malformed/nonpositive width admission, arbitrary
controlled constructor dispatch and the original controlled ArbitraryClifford
resource-query defect remain separately recorded. A 64-bit structural/resource
comparison does not imply exhaustive 129-wire numerical execution. No GPU or
performance claim is made and no protected API changes.

Acceptance receipts use `g2-relative-swap-constructor-*-20261124` in the external
reference ledger. They include original blobs, source outcomes, concentrated
regression, local integration, strict checks and separate primitive/model rows.

Final acceptance: 520 concentrated cases pass without skips, including all 20
new constructor and direct inverse cases; local integration passes 145. The
complete unchanged original test file, including its notebook, has 24 passes,
three original skips and two original xfails. The initial sandbox denial of a
local notebook-kernel port is recorded separately; the unchanged notebook
passes with the locked kernel environment outside that sandbox restriction.
Strict typing covers all four changed Python files. Black, Ruff, Compiler scope,
language, architecture, IR and API checks pass. Dependency output is exactly
the existing 720 findings, with no additional finding from this batch.


## Complete source acceptance (2026-11-28)

The approved faithful-migration closure rule supersedes the historical open
parent status for Q-0707. Complete source and direct fanout review covers all
CSwapApprox properties, nested G/G-inverse segments, physical decomposition,
classical permutation, drawing adapters, analytical categories, three factories
and DocSpec. Existing typed and flat owners agree at comparable recursive calls,
ordered operands and primitive dependencies. The exact generic Adjoint retains
the reversed fanout sequence instead of substituting a forward equivalent.

Existing complete small complex operators, absolute relative signs, eight-bit
occupied-word/spectator action, source-free constructor, append and JSON evidence
are reused. There is no private workspace or nonempty measurement history for
the bare primitive. Extra controlled wrappers use their separately recorded
predicate history and clean-work certificates. Unknown wrapper constructors do
not acquire implicit recipes from this acceptance.

The source generic-controlled ArbitraryClifford aggregate-cost failure remains
an upstream boundary. Symbolic widths retain logical/resource declarations and
require resolution for execution. A 64-bit call/primitive inventory establishes
structure and counts, not exhaustive 129-wire numerical simulation. Native
controlled CH/CCX/phase leaves remain unsynthesized resource categories.


The complete three-file closure passes 335 concentrated source/primitive/
complex-execution cases and 51 independent Cirq 1.6 constructor cases. All 54
frozen source/dependency/asset blobs are unchanged. Algorithms scope, language,
links, architecture, IR, protected API baseline and whitespace pass. Existing
strict Python, 32 local integration and real Kickmix Simulator evidence from
the accepted preceding batches is reused: no Python code changes here. Original
source tests/notebook outcomes and separately recorded defects remain intact.
