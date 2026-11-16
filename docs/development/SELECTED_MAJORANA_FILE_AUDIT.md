# Selected Majorana source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `selected_majorana_fermion.py` | `656373799b28c6838bc3528733c26f0fba0adacf` | Q-0594 |
| `selected_majorana_fermion_test.py` | `d0b964420757f48f6f600d3d8e555855c1b6843c` | Q-0595 |

## Flat bounded entry

`SelectedMajoranaFermion.make_bounded_on` binds explicit storage and can
be passed to `circuit.append(op)`. Its flat address is big-endian and
may have more bits than the minimum for the target count. It accepts
X, Y and Z leaves, a controlled singleton with a nonempty selector,
and odd target counts. The control is either absent or one positive
qubit. Work starts with the parity accumulator and continues with
the traversal predicates. Only predicates need feedback slots.

The same compiler owner still serves the orbital/spin and named nested
axes of the Hubbard path. The bounded and deferred entries preserve their
existing padding contracts. Use populated addresses for the Jordan–Wigner
selection action. The bounded entry retains source pruning when there
are unused bit patterns.

## Source correspondence

The frozen suite passed all nine tests. The native audit maps the original
physical port order from both the explicit and inferred source bindings.
It checks the source diagram symbols and the full decomposed circuit of
the three-target diagram case. The six flat width and extent pairs
`(2,3)`, `(2,4)`, `(3,5)`, `(3,8)`, `(4,9)`, and `(5,9)` match
gate dependencies, operands, feedback and independent resources for X
and Y. Added Z and controlled singleton cases also match.

Each tested address has a coherent superposition of target words. Both
activation states and the complete output amplitudes are checked. The
three-, four-, and five-target cases check every raw measurement history.
The eight- and nine-target cases check all-zero, all-one, and alternating
histories. The output comparison requires the accumulator and every
reserved work wire to return to zero. Earlier bounded tests cover nested
axes and the six-orbital padding witness.

Run `python -m examples.algorithms.majorana_select` to print the primitive
compilation of the original three-target X diagram and its resource model.

| Three-target controlled X | Counts |
| --- | --- |
| Emitted primitives | 8 T/T†, 2 measurements, 2 resets |
| Declared source model | 2 And, 14 Cliffords, 2 measurements, 9 peak live wires |

The extra selection bit in the `(5,9)` case reserves a work wire that does
not become live. The source model counts the occupied peak separately.

## Remaining source-file boundaries

Q-0595 is accepted. Q-0594 retains arbitrary foreign gate objects,
symbolic iteration extents and multiple controls on the bounded entry.
The deferred named entry supports its own multiple-control contract.
A zero-width selector is rejected by the frozen source and by the
native bounded binder. The uncontrolled singleton is outside the
supported bounded traversal.
