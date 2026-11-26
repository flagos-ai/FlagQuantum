# Complete Hubbard SELECT stored graph and constructors

## Ownership and execution

The immutable reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
The source intake covers the complete `select_hubbard.py`, all five original
factories and sixty concrete, singleton, symbolic and control contexts.

Compiler owns shared bounded traversal, typed unary Pauli calls, directional
And lifetimes and exact measured adjoints. `majorana_component.py` owns the
Pauli Majorana accumulator. `selected_pauli_component.py` owns selected Pauli
execution with one or two positive controls. Compiler's
`lookup/hubbard_majorana_component.py`, `lookup/hubbard_spin_up_component.py`
and `control/hubbard_select_component.py` now own their shared typed source
compositions. Algorithms retains numerical operations, validation and resource
models, delegating its existing stored builders to those same owners. Layout,
swap, binding, allocation, lowering and numerical execution retain their
existing owners. Production construction has no upstream SDK dependency.

Coordinates are MSB-first bounded unsigned scalar words. Majorana targets
follow `(x * dimension + y) * 2 + spin`. Spin-up targets follow
`dimension**2 + y * dimension + x`. Selection and target ports remain live.
Accumulator, conjunction and traversal predicates are private clean work.
For non-power-of-two extents, the logical clean-work contract requires populated
selection indices. Padded addresses retain the original traversal behavior.

The ordinary workflow remains `circuit.append(op)`. Existing standalone Hubbard
operations and their accepted source-free append evidence are reused. Stored
Programs additionally pass normal `bind_register_program`, Program copying,
ordinary append and lowered CircuitIR JSON round trips. No public export or
protected contract changes are introduced by this batch.

## Complete successful graphs

Shared Majorana and Hubbard Majorana forward/inverse graphs retain exact source
call insertion indices, signatures, named operands and recursive dependencies.
The Hubbard wrapper inlines its child in connection visitation order, matching
`add_from`. Its inverse reverses that inlined graph. Reversing a separately
inlined inverse changes source insertion indices and is deliberately avoided.

Spin-up retains source control Join/Split, atomic Split2/Join2 signatures and
inlined selected-Z traversal in y/x order. The native layout partition owner
supplies execution beneath those atomic source boundaries. Both directions
retain exact source graphs.

Positive-controlled SELECT retains its eleven source calls, including both
Majoranas, four word swaps, two atomic bit swaps, S-dagger on the control,
control/U CZ and final spin-up. The specialized source adjoint and every nested
successful child are compared recursively in both directions at dimensions
2, 3, 4 and 5. CY retains the original shaped `q[2]` signature, while CZ retains
`q1/q2`. Original primitives remain direct native instructions.

The physical nested unary refactor matches the published pre-change stream
exactly in 72 contexts, including both traversal directions, nested bounded
axes, controls, early folding and zero-selection callbacks.

## Acceptance and independent resource accounts

Focused complete graphs, all columns of small complex Majorana operators and
all their raw measurement histories are checked. Hubbard Majorana tests use
arbitrary full complex logical inputs, raw history samples and an all-history
predicate-erasure certificate. Spin-up tests exhaust all dimension-two raw
histories with complex logical states. SELECT tests compare complex sparse
states with the independently accepted flat emitter, use an all-history
predicate-erasure certificate and sample three complete raw histories in both
directions. Full SELECT's eighteen measurements are certified structurally;
their 262,144 raw histories are not exhaustively simulated. Private work is
checked at output. Existing larger generator, model, append and source binding
evidence is reused through the concentrated regression.

`g2-hubbard-select-stored-resources-20261122.json` contains 56 separate literal
primitive and original QEC/width model rows. Reference source/test/notebook
blobs and the Qualtran 0.7.0 / Kickmix 0.1 Simulator environment are verified.
At dimension two, positive-controlled SELECT literally uses 199 CX, 34 X,
66 H, 60 T, 54 T-dagger, 18 S, 8 CY, 39 CZ, 18 measurements, 18 resets and
one S-dagger on 21 wires. Its independent original model contains 6 CSwap,
18 And, 27 Clifford and 18 measurements, width 21. These accounts remain
separate because the original declared Majorana model omits Pauli costs.

## Remaining source and native boundaries

- The former uncontrolled stored SELECT gap is closed by the existing private
  scalar carrier, as detailed below. Proposal 119's new Core opcode remains
  pending and is not used.
- Original zero-controlled Majorana raises `KeyError`. Original SELECT rejects
  direct zero control. Original spin-up accepts zero control but executes the
  positive-control body. The source constructor retains that positive spin-up
  body for zero, while the existing Algorithms stored validator still refuses
  it. Majorana zero remains a declaration without executable body. Existing
  standalone corrected negative controls remain separate.
- Singleton coordinate and symbolic stored decomposition failures remain
  recorded. Shared stored constructors currently admit concrete bounded axes
  of extent greater than one, Pauli leaves and at most two selected-Pauli
  controls. Existing general external binding evidence covers broader inputs.
- Original Cirq 1.7 CY classification failures and declared model omissions are
  retained. Their earlier diagnostic evidence is reused.
- General prepared-state reflection still has the independently reproduced
  dirty-junk disposal counterexample. Pending Proposals 118, 119, 124, 126 and
  127 remain separate.
- This batch closes successful stored dependencies within the complete source
  group. Parent Q-0353/Q-0354 and Q-0594 remain open for their recorded boundaries.

## Complete constructor and uncontrolled scalar closure

All three source constructors resolve through
`formats/legacy_hubbard_selection_factories.py`. Their original MSB-first
control/coordinate/target signatures have one Compiler owner in
`layout/hubbard_signatures.py`. Both existing Algorithms callers and named
constructors use the shared typed composition; no SDK objects, numerical
kernels or independent resource formulas move into Compiler.

Uncontrolled SELECT keeps the literal eleven source calls, including the
zero-port GlobalPhase with coefficient `-i` and the separate Z on U. The
accepted `scalar_phase_program` supplies that atom's exact execution with one
explicit private carrier, which is restored and freed. It allocates no new
Core opcode and does not implement pending Proposal 119. Its two X and two
phase primitives are reported separately from the original zero-cost scalar
model. Controlled SELECT retains its original S-dagger/CZ pair. Explicit
source adjoints use inverse child programs and `+i`, not a reversed measurement
stream or an identity placeholder.

The constructor tests cover all five original factories, symbolic declarations,
depths zero through two, both X/Y choices and successful control modes. Exact
ordered recursive calls, named operands, output aliases and physical path
replacement are compared against the unchanged source. Generic Majorana and
spin-up adjoint constructor wires resolve through scoped exact recipes;
unrelated same-signature children cannot acquire an inverse from that context.
Source-free construction and physical binding use named primitives only.

The published `ad0cde787` baseline proves 56 complete existing Program records
identical after ownership moves, including their helpers and exact inverses.
Full complex SELECT branch checks now cover uncontrolled execution in both
directions. All 32,768 uncontrolled dimension-two histories are certified by
predicate-erasure composition, with three full histories numerically sampled
per direction; they are not claimed to be exhaustively simulated. Normal
append executes the 19-wire uncontrolled operator in both local strategies,
including absolute complex phase and zero private work. Its fifteen
measurements execute thirty times across two shots. Existing Majorana/spin-up
exhaustive small-history and larger literal generator evidence is reused.

| Dimension-two SELECT | Actual named primitives | Independent original model |
| --- | --- | --- |
| Uncontrolled | 177 CX, 38 X, 57 H, 54 T, 48 T-dagger, 15 S, 8 CY, 35 CZ, 1 Z, 2 phase, 15 measurements and resets; live peak 19 | 6 CSwap, 15 And, 23 Clifford, 15 measurements; peak 19 |
| Positive controlled | 199 CX, 34 X, 66 H, 60 T, 54 T-dagger, 18 S, 8 CY, 39 CZ, 1 S-dagger, 18 measurements and resets; live peak 21 | 6 CSwap, 18 And, 27 Clifford, 18 measurements; peak 21 |

Twelve new dimension-two/three resource rows retain actual allocation lifetimes
and original model accounts separately. Carrier allocation events are not
additional simultaneous live work. The model does not acquire the omitted
Pauli primitive costs merely because their execution is now reconstructible.

Remaining complete-file boundaries stay explicit: original SELECT specialized
adjoint protobuf export fails on `SpecializeOnCtrlBit.ONE`; it is not repaired
with an invented enum codec. Direct zero-controlled SELECT is unsupported,
Majorana zero loses its control, spin-up zero executes positive control, and
singleton/symbolic decompositions retain their earlier failures. Non-square
lattices and nonbinary control metadata are refused. Original Cirq 1.7 CY
classification failures remain failures, with accepted Cirq 1.6 evidence
reused. No complete parent is counted twice, and these constructors do not
close the remaining generic catalogue or external-capture scope.

### Verification profiles

The primary frozen-source environment uses Cirq 1.7.0: all 208 concentrated
constructor, stored graph, Majorana, spin-up, SELECT and PREPARE checks pass.
Local integration passes 145 cases. Strict typing covers ten Python files;
format, lint, both owner scopes, language/links, architecture, IR and protected
API checks pass. No production reference source is changed.

The separate Cirq 1.6.1 run yields 166 passes and 42 literal call-port failures.
Those failures are retained, not skipped or reported as accepted. Its source
`Y.controlled()` is a ControlledGate converted to `CYGate(ctrl, target)`;
Cirq 1.7.0 instead converts CYPowGate to `CirqGateAsBloq(q[2])`. The native
stored graph retains the published 1.7 port representation. An independent
probe verifies that the complete dimension-two Majorana Program equals the
published baseline in each environment and that both source CY operators have
zero full-complex matrix discrepancy from the native primitive. Thus the
alternate profile's literal signature coverage remains a registered boundary;
the primary comparison is not weakened by renaming or flattening its operands.
Existing 1.6 original count-suite acceptance remains distinct from these graph
checks. Compiler never selects a representation by importing an SDK version.

### External control constructor closure

The scoped SELECT mapping now supplies negative, grouped, shaped, packed-word
and nested control constructors through existing shared control and exact
inverse owners. The [external-control audit](HUBBARD_EXTERNAL_CONTROL_CONSTRUCTORS.md)
records recursive source calls, complete complex branch and work proofs, plus
twenty-eight separate primitive/model rows. Both Cirq source port profiles use
the accepted exact CY views; the historical 1.6 failures above are superseded
for the covered graph consumers without changing source gates or operands.

Source direct specialized-adjoint enum encoding and deeper NumPy scalar codec
failures remain explicit. Nested control of an already-controlled SELECT hits
an original concatenate-axis TypeError. Its executable specialized child is
compared recursively, while native outer composition is independently verified;
the failing source root and its two source peak queries are not counted as
executable equivalence. Generic Majorana/spin-up controls and wider external
capture remain outside this SELECT constructor closure.
