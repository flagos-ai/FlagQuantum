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

## Complete external-object and source-adjoint binding

The complete locked implementation, original test file and direct unary
iteration implementation were read. The existing Ecosystem register emitter
captures the actual source object in both directions, and the existing native
`SelectBlockEncoding` consumes restored JSON snapshots. No parallel emitter,
compiler path or public API is introduced. All external objects stop at capture.
The native consumer receives owned named layouts and circuit snapshots, with
control and selector words preceding the row-major target word. Caller-owned
workspace covers the maximum of both source directions and feedback slots cover
the maximum measured storage. Captured directions retain their independent
source decompositions, including the asymmetric controlled-S adjoint.

The additional source objects use H, S and Rz(0.3) target gates and zero, one or
two control bits in a single named control word. Each direction checks the
complete complex output vector for every small measurement history. The S cases
with zero or one control also compare every logical operator column, including
inactive control sectors. The output comparisons include branch normalization
and require every reserved work bit, including the accumulator, to be zero.
A separate two-axis row/column object with two control bits checks the flattened
row-major Jordan-Wigner target index, all histories and both source directions.
Each consumer stream equals its captured source primitive stream and survives
JSON serialization. Existing flat X/Y/Z, padded-address, original test-suite,
diagram and independent resource evidence above is reused.

| Two-row fixture | Captured primitive inventory | Independent source model |
| --- | --- | --- |
| One-control S forward | 4 T/T†, 10 CX, 2 controlled phases, 1 measurement, 6 active wires | 3 And, 11 Cliffords, 3 measurements, 7 peak wires |
| One-control S adjoint | 12 T/T†, 22 CX, 3 measurements, 7 active wires | 3 And, 11 Cliffords, 3 measurements, 7 peak wires |
| Two-control H, either direction | 8 T/T†, 15 CX, 1 CCX, 2 controlled H, 2 measurements, 8 active wires | 1 Toffoli, 2 And, 15 Cliffords, 4 rotations, 2 measurements, 8 peak wires |

The captured inventory preserves admitted controlled gates. The model further
expands those leaves according to the locked source policy, which explains why
the S forward capture and model have different And/measurement counts. Reserved
storage uses the larger directional requirement. These inventories do not claim
that a controlled-H primitive is a single Clifford or that a model rotation is
an emitted native Rz.

## Source boundaries retained by this binding

Symbolic address widths fail capture until an immutable constructor is explicitly
resolved. A resolved width object binds to the same consumer. Symbolic iteration
extents fail in the source `target_registers` property. The source uncontrolled
singleton raises `IndexError`; this remains outside the bounded native route.

The source's separate-control-register case has a further clean-work defect.
Its initial accumulator flip reads only `self.control_regs[0]`, while unary
iteration combines all control registers. With `first=1, second=0`, selection
zero and a zero target, every measured branch leaves the accumulator nonzero.
The test retains that raw source result and makes no clean-work acceptance claim
for that layout. The single two-bit control-word cases above pass the complete
clean-work checks. Q-0594 remains open for these source-domain boundaries and
generic arbitrary-gate cases beyond the verified capture admission set.
