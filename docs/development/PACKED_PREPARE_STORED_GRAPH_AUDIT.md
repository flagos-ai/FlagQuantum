# Packed preparation stored dependency closure

The current shared packing graph lives in
`compiler/registers/layout/packed_oracle_component.py`; the existing Algorithms
entry delegates to it. See `RESOLVER_PACKED_ORACLE_AUDIT.md` for the subsequent
constructor integration and AutoPartition primitive-child repair. The original
acceptance observations below retain their historical scope.

## Ownership and interface

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
This batch completes the successful stored call paths of
`state_preparation/black_box_prepare.py` and its concrete Hubbard preparation
dependency. Existing complete source/test and external-binding evidence remains
in `BLACK_BOX_PREPARE_FILE_AUDIT.md` and the Hubbard preparation tests.

| Responsibility | Native owner | Contract |
| --- | --- | --- |
| Typed word replay | `compiler/registers/component_wire_graph.py` | Original earliest insertion, explicit nested-circuit boundaries, retained nominal single-bit keys |
| Exact immediate adjoints | `compiler/registers/component_adjoint.py` | Literal child matches and canonical Split/Join/Cast inversion |
| Atomic unit increment binding | `compiler/registers/arithmetic/bounded_increment_component.py` | MSB-first unsigned word, ordered control array, unchanged padding, private clean work |
| Hubbard preparation | `algorithms/primitives/hubbard_prepare_component.py` | Original selection signature and through-junk temp, private conjunction work |
| Packed preparation | `algorithms/primitives/packed_prepare_component.py` | One AutoPartition call around supplied domain-owned child, explicit exact inverse |

Selection and junk are packed MSB first, with shaped child elements in row-major
order. Logical junk remains caller-owned. The preparation child owns its private
clean work and feedback. The packed wrapper uses the shared AutoPartition owner.
It adds no primitive gates, measurement or workspace. A zero selection width
retains the original empty root, including when junk is nonempty. Symbolic child
and partition bodies remain declarations. Existing users continue to append the
bound operation through `circuit.append(op)`.

## Complete source comparisons

The complete Hubbard source-success graphs for dimensions 2, 3, 4, 5, 8 and 9
are checked recursively in both directions. Original insertion indices, named
operands, typed views, component dependencies and directional conjunctions agree.
The source generator embeds the coordinate fanout in an explicit Circuit.
Its boundary starts a new earliest insertion group. Retaining nominal keys also
preserves the original bounded-one-bit-to-bit casts before gathering words.
The exact inverse reverses original calls and casts.

Packed roots cover both original Hubbard factories, all five original dense,
sparse and symbolic alias factories, and the actual Ising alias preparation.
Their complete recursive graphs agree in both directions. Original zero-group
and dummy-oracle behavior is retained. The inverse reverses the partition order
as well as the child body. Supplying only an inverse child within a forward
partition sequence would lose the original order.

Physical packed, copied and unpacked Programs produce identical CircuitIR for
Hubbard, dense alias, sparse alias and actual Ising, in both directions. Lowered
JSON and ordinary append preserve these instructions. All small increment
control polarities, padding and full complex operators are checked. An arbitrary
complex Hubbard input spans every selection and logical-junk basis state.
Every raw small measurement history is compared against the original generator,
with exact inverse recovery and private work zero. Wider accepted flat/source
execution and component evidence is reused.

## Literal execution and independent models

`g2-packed-stored-resources-20261122.json` records 24 separate literal/model rows.
Literal physical costs and independent model estimates occupy separate columns.

| Case | Literal execution | Original model |
| --- | --- | --- |
| Hubbard dimension 2, forward | 66 CX, 23 H, 26 T, 21 T-dagger, 19 X, 3 S, 3 measurements, 3 CZ, 3 resets, 1 Ry, 1 CRy, 1 CH; 12 wires | 5 CSwap, 2 And, 11 Clifford, 4 rotations, 2 measurements; width 12 |
| Hubbard dimension 3, forward | 262 CX, 101 H, 82 T, 73 T-dagger, 79 X, 23 S, 23 measurements, 23 CZ, 23 resets, 4 Rz, 1 Ry, 1 CRy, 1 CH; 20 wires | 9 CSwap, 17 And, 116 Clifford, 8 rotations, 17 measurements; width 18 |

The original ModAddK exposes an exact classical permutation and a five-Add model,
but refuses a primitive decomposition. Its native descendants reuse the existing
bounded increment emitter. Full logical permutation, complex phase and cleanup
are independently verified. Component comparisons stop at that original atomic
boundary. Previously accepted original primitive recipes for all comparable
children remain applicable.

## Remaining boundaries

- The original singleton Hubbard stored decomposition fails with unavailable
  coordinate soquets. Native stored construction retains an explicit refusal.
  The separate flat execution evidence qualifies its singleton path.
- The original packed object serialization failure remains separately recorded.
- General prepared-state reflection still has the recorded original dirty-junk
  disposal counterexample and remains open.
- Registerless scalar paths retain pending Proposal 119. Proposals 116–119 remain
  unimplemented.
- Two pre-existing width-eight ordering operand failures remain recorded in the
  preceding alias batch. This batch runs the affected preparation closure and
  retains those two explicit exclusions.

The earlier resource probe requested a nonexistent `prepare_hubbard.ipynb`.
The actual `hubbard_model.ipynb` is included in the final locked-blob verification.
An initial new binding test grouped the two shaped control elements together.
It was corrected to the existing element-wise physical input contract before
acceptance. The original dummy oracle also requires a bodyless child. Shared AutoPartition
now retains that declaration after validating its typed Program.
The first symbolic packed-inverse probe exposed a concrete-partition
construction attempt. It now retains the same symbolic declarations as the
original forward AutoPartition owner.

## Batch acceptance

The affected closure passes 310 concentrated checks. Shared comparison, addition,
and adjoint consumers pass 144 checks with two named pre-existing width-eight
ordering exclusions. Local integration passes 29 checks. Strict typing, Black,
Ruff, ownership, architecture, multi-level IR, public API baseline and whitespace
checks cover this batch. The five actual source/test/notebook blobs match the
frozen upstream Git tree. Qualtran 0.7.0 and Kickmix 0.1 Simulator are verified.
The final regression includes the independently owned increment test module.
