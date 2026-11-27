# Swap-network constructor dependency disposition

Reference: complete `swap_network/multiplexed_cswap.py` and
`swap_network/swap_with_zero.py` with their unchanged original test files at
Qualtran v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389`.
The four source/test blobs match the pinned inventory. Fixtures require actual
Qualtran 0.7.0 and Kickmix 0.1 Simulator. No reference file is modified.

## Ownership and interface

Compiler formats retains constructor metadata, direction and typed registers.
The existing Compiler lookup owner composes bounded unary traversal with exact
CSwap for MultiplexedCSwap. The existing SwapWithZero owner composes the source
exchange order using relative CSwapApprox. Neither network is replaced with
an optimized alternative. Compiler has no Algorithms import, backend selection
or quantum numerical execution.

Named control registers, scalar bounded selectors, row-major opaque target
arrays and multiplexed output remain caller-owned THRU ports. Words are
MSB-first. Unary traversal owns clean predicate work and measured erasure;
exact CSwap and relative CSwapApprox do not allocate private work. The existing
typed joins retain native bounded output descriptors where the source Cirq
importer uses generic QAny storage. This nominal source boundary is retained.

MultiplexedCSwap's generic Adjoint now uses the shared exact typed graph
adjointer with explicit And compute/erase pairs, conjunction compute/erase,
self-inverse X/CX and exact CSwap. Stored calls reverse in source order;
measured primitive streams are never reversed. Constructor recipes retain
the complete owned helper closure and delegate unknown wrappers only to an
explicit caller. SwapWithZero's source uncompute flag instead reverses its
relative exchange stage sequence and keeps the phase of each source child.

## Source and execution evidence

All four original concrete factories resolve at depths zero and one, forward
and inverse, to executable roots. Complete ordered recursive calls, operands
and child definitions match the unchanged source. This includes the original
136-wire SwapWithZero example; it is structurally compared and resource-counted,
not represented as an exhaustive 136-wire numerical simulation.

Small multiplexed complete complex operators cover zero through three controls,
including a packed three-bit control word, and a two-axis selector. Forward and
inverse graphs compare to source tensors. The shared certificate covers every
applicable raw erasure history and every basis column verifies private-work
cleanup. Relative networks compare complete complex operators, retaining the
source phase, in both directions for singleton, non-power-of-two and multi-axis
domains, including a trailing zero-width axis. No measurement or private lease
is introduced by that network. Ordinary append and IR JSON roundtrips use the
actual emitted multiplexed instructions.

Both constructors compile without importing source code. Symbolic word widths
retain typed logical declarations in both directions; physical execution still
requires concrete sizes. Existing complete-file primitive, feedback, source
slot ordering, input-grid and resource evidence is reused in the concentrated
regression, including the exact seven-T CSwap and relative exchange recipes.

## Retained boundaries

Uncontrolled singleton-first-axis MultiplexedCSwap fails upstream with IndexError;
the existing native owner refuses it explicitly. SwapWithZero with a leading
zero-width axis followed by a live axis also raises upstream IndexError; the
native owner retains its explicit refusal. Malformed vectors and unsupported
typed register layouts fail before physical binding. Object/symbolic dimension
array serialization and source nominal Join behavior remain separate from
the concrete constructor path; neither is silently rewritten in the reference.
These boundaries do not increment complete generic constructor counts or close
the original file parent records.

## Separate primitive and model resources

Both directions of the original multiplexed example emit 82 T-family, 101 CX
and three measurements with 17 physical peak wires. Its source model retains
ten CSwap, three And, eight Clifford, three measurements and 17 logical peak
wires. An atomic source CSwap count is not a primitive T count; the actual
source-derived exact decomposition uses seven T-family gates per swapped bit.

The original SwapWithZero examples emit respectively 384, eight and 88
T-family gates, with 573, 11 and 121 CX and 136, seven and 28 peak wires.
Each has zero measurements and zero private work. Their separate source model
T counts are 384, eight and 88, Clifford counts 2109, 43 and 473, with matching
logical peak models. All eight forward/inverse resource rows have zero live
private leases. Counts and physical leases complement numerical cleanup;
lease release alone is not a complete quantum-state proof.

Receipts use `g2-swap-constructor-*-20261124` in the external reference ledger.
They include full original source outcomes, concentrated regression, local
integration and strict checks. Larger-example exhaustive simulation, arbitrary
symbolic physical synthesis, GPU execution and performance remain unclaimed.

Final acceptance: 215 concentrated cases pass without skips, including all 31
new constructor-path cases. Local integration passes 145 cases. The complete
two unchanged original test files pass 40 cases with two original skips and
no deselection. Strict typing covers all three changed Python files; Black,
Ruff, Compiler scope, language, architecture, IR and API checks pass. Dependency
output matches exactly the existing 720 findings; no new finding is introduced.


## SwapWithZero complete source acceptance (2026-11-28)

The approved faithful-migration closure rule supersedes historical open status
for Q-0712. Complete source review covers scalar/array/iterable conversion,
axis-count validation, recursive inner-axis-first stage traversal, selectors,
target shape/signature, CSwapApprox leaf, sequence reversal, composite body,
product-minus-one call model, diagrams, classical in-place callback, adjoint,
string, all three examples and DocSpec. Existing Compiler routing, relative-swap
and Simulation callback owners remain separate and are reused.

Selector words are MSB-first; source stage indices are little-endian offsets
converted explicitly. Targets are row-major and caller-owned. Reversing stage
order preserves the source child phase rather than replacing it with an exact
Fredkin. Small complete complex maps, padded selectors, multi-axis arrays,
trailing zero-bit axes, true inverse composition, callback strided views and
keyword insertion order remain checked. Bare networks allocate no private work
and have only the empty measurement history.

Leading zero-width axes followed by live exchanges still fail upstream with
IndexError and are refused explicitly before native emission. Unresolved axes
and widths do not imply executable storage. The 136-wire original example has
complete structural/primitive/resource evidence; no exponential-size numerical
simulation is claimed. The source model, actual instructions and live/physical
workspace categories remain separate. Q-0709 and Q-0707 full-source acceptance
is recorded in their respective linked source/constructor audits.


The complete three-file closure passes 335 concentrated source/primitive/
complex-execution cases and 51 independent Cirq 1.6 constructor cases. All 54
frozen source/dependency/asset blobs are unchanged. Algorithms scope, language,
links, architecture, IR, protected API baseline and whitespace pass. Existing
strict Python, 32 local integration and real Kickmix Simulator evidence from
the accepted preceding batches is reused: no Python code changes here. Original
source tests/notebook outcomes and separately recorded defects remain intact.
