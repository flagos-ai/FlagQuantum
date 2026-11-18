# Kikuchi guiding-state source audit

The complete reference file is frozen at Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. Its source blob is
`0a96fbb1e60e21ece4d6c33f504a7187fae986b1` and its unchanged test blob is
`c215aad2516a4c61cc745c8909ef90f55fe74e5b`.

## Ownership and interfaces

`algorithms/kikuchi_guiding_state.py` owns the three declarations and five
example factories. Source `SimpleGuidingState` becomes `ScopeGuidingState`.
Sparse coefficient preparation remains in Algorithms primitives. Sorting,
duplicate predicates, measured And ladders and register rebindings retain their
existing Compiler owners. Core metadata and executable admission are unchanged.

All packed words are big-endian. `S`, `T`, `ancilla`, `q` and `flag` retain the
source through direction, including the discrepancy with the source docstring
that describes `ancilla` as RIGHT. The sorting record and three flags are logical
outputs. Probabilistic uncomputation retains the word after its Hadamards and
marks its zero predicate coherently. Only predicate workspace must enter and
leave zero. Caller-owned classical slots hold its And-erasure results.

The root program accepts actual typed bodies for the six source roles. It uses
shared split/join and component-call construction for scope tensor products,
sorting, duplicates, coherent zero marking and final OR flag. An identity child
fixture checks admission only and supplies no physical preparation evidence.

## Reused dependency closure

- Weighted sparse coefficients use the existing sparse rotation preparation and
  permutation implementation. Concrete and symbolic resource paths preserve
  their distinct frozen-source counts, including symbolic width roundtripping.
- Sorting uses the existing source signature and resource model. Its bounded
  permutation-register dtype is retained.
- Duplicate detection reuses the existing source QEC query.
- Coherent probabilistic uncomputation emits H followed by the existing zero
  controlled X ladder. The user appends the bound operation directly.
- Success-flag combination reuses the two-negative-control X source model.

## Evidence and acceptance

All five factories match signatures, sides, precision, amplitude bounds and all
seven QEC categories against the untouched reference, including symbolic count.
The root's six direct source roles and their operands are inspected. Frozen
source and test hashes are checked.

For widths 1 through 4, the complete probabilistic-uncompute primitive streams
match the original generic packed-word controlled-X decomposition per physical
wire, including parameters, measurement slots and feedback. Every one of the
1, 2, 4 and 8 measurement branches is checked with complex superpositions,
nonzero arbitrary flags and an entangled spectator. Unnormalized branch
amplitudes equal the coherent ideal divided by the branch factor, with all
predicate workspace exactly zero. Invalid bindings fail before append.

The batch regression passes 961 tests with three visible warnings. Both changed
Python files pass strict mypy, Black and Ruff. Team scope, architecture, IR and
API checks pass. The unchanged non-notebook source suite reports 14 passed,
9 original skips, 7 original xfails and two notebook cases deselected, with two
original unknown-notebook-marker warnings. The source count-consistency xfails
remain visible, alongside its five serialization xfails.

## Emitted primitives and model estimates

| Operation | Emitted primitive evidence | Frozen QEC estimate |
| --- | --- | --- |
| ProbabilisticUncompute(n), n=1..4 | 4(n-1) T/Tdg and n-1 measurements/resets, plus H/CX/feedback | n-1 And and n-1 measurement, Clifford 4/4/18/24 |
| Concrete scope preparation | Dependency evidence reused, root has no source physical decomposition | 63 Toffoli, 24 And, 204 Clifford, 24 measurement |
| Concrete full guiding state | Physical root blocked below | 132 Toffoli, 192 CSwap, 352 And, 4173 Clifford, 352 measurement |

Model rows are logical categories. They are independent of the number of emitted
primitive instructions.

## Remaining boundaries

Q-0611 remains active. The frozen scope root declares a sparse preparation child
but provides no composite decomposition and omits that child's phase-gradient
port. Full source sorting likewise has no physical decomposition. Its bounded
`pi` register is not admitted by protected Core executable descriptors, so the
native root program fails there. Downstream root wiring is present but cannot
be execution-certified beyond that admission boundary. Root complex amplitudes
and a full preparation/erasure experiment remain unverified. No substitute
sorter, reset of logical outputs, SDK fallback or repaired reference is used.

Q-0610 and Q-0613, the two notebooks, remain the next whole-file batch. The
original test parent Q-0612 remains active until those notebook cases are mapped.
