# Complete controlled Hubbard SELECT stored graph

## Ownership and execution

The immutable reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
The source intake covers the complete `select_hubbard.py`, all five original
factories and sixty concrete, singleton, symbolic and control contexts.

Compiler owns shared bounded traversal, typed unary Pauli calls, directional
And lifetimes and exact measured adjoints. `majorana_component.py` owns the
Pauli Majorana accumulator. `selected_pauli_component.py` owns selected Pauli
execution with one or two positive controls. Algorithms owns the Hubbard
coordinate conventions, target partitions and SELECT composition. Layout,
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

- The uncontrolled stored SELECT requires the registerless `-i` phase contract
  in pending Proposal 119. Its existing standalone execution evidence is kept.
- Original zero-controlled Majorana raises `KeyError`. Original SELECT rejects
  direct zero control. Original spin-up accepts zero control but executes the
  positive-control body. Stored native constructors refuse these routes
  explicitly. Existing standalone corrected negative controls remain separate.
- Singleton coordinate and symbolic stored decomposition failures remain
  recorded. Shared stored constructors currently admit concrete bounded axes
  of extent greater than one, Pauli leaves and at most two selected-Pauli
  controls. Existing general external binding evidence covers broader inputs.
- Original Cirq 1.7 CY classification failures and declared model omissions are
  retained. Their earlier diagnostic evidence is reused.
- General prepared-state reflection still has the independently reproduced
  dirty-junk disposal counterexample. Pending Proposals 116–119 remain separate.
- This batch closes successful stored dependencies within the complete source
  group. Parent Q-0353/Q-0354 and Q-0594 remain open for their recorded boundaries.
