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
`pi` array is now admitted by the existing Core contract, as the current typed
fixture proves. Complete child helper ownership was still missing until the
context closure below. Downstream whole preparation cannot be execution-certified
while the original source leaves remain bodyless. Root complex amplitudes
and a full preparation/erasure experiment remain unverified. No substitute
sorter, reset of logical outputs, SDK fallback or repaired reference is used.

Q-0610 is already accepted by the complete notebook receipt 525190c3e. Q-0613,
the separate tutorial, remains open for four native symbolic/bounded graph
views. The original test parent Q-0612 remains active until that tutorial
case and its original protocol outcomes are completely mapped.


## Complete measured-child Program context (2026-11-27)

The existing GuidingState.program now also accepts complete ProgramRecord
children in each of its six source roles. Existing FunctionRecord callers
produce entirely identical Programs at ell 2 and 4 relative to a4fcc9857.
The same shared ComponentProgramBuilder imports and isolates complete child
helpers; it owns no execution backend. Call order, packed MSB-first scope
words, row-major bounded permutation words, flags and input/output directions
remain the actual source ones. Missing helper definitions fail validation
without mutating any supplied module.

ProbabilisticUncompute.program composes the existing OnEach Hadamard and
measured multi-control X owners. Source-sized words and flags remain THRU.
Only private predicate work is erased; the Hadamard-transformed logical word
and all three guiding-state flags are returned live. The serializable
ProbabilisticUncompute source constructor reuses this Program through the
Ecosystem factory. Libraries at depths 0/1/2 roundtrip and resolve with SDK
modules blocked. Zero and symbolic widths remain logical declarations; no
finite body or new instance serializer is invented.

The full example root and ell-two root retain the source's actual bit
predecessors and outputs after normalizing only zero-gate Split/Join/Partition
views. Preparation, sorting and duplicate detection stay source declarations.
Whole physical binding still refuses these three missing bodies. An identity
fixture only proves unchanged interface behavior; it never substitutes for
physical state preparation. Current bounded pi admission is verified without
changing Core, the existing API contract or serialized vocabulary.

The complete known uncompute/OR/X suffix has an independent original composite
tensor operator. At widths one through three, every logical column and every
raw measurement history agrees in absolute complex amplitudes, with private
work zero. SDK-blocked append and trajectory execution preserve the same map.
Logical words and arbitrary initial flags are included; they are not reset.
The complete new uncompute streams at widths one through four match the
previously source-validated flat consumer per wire, parameters and feedback.
A shared bit-lineage checker now serves both this root and the accepted mean
phase tests; computational stages and work lifetimes remain explicit.

| Suffix word width | Actual instructions | Actual T/Tdg | Measurements | Peak private bits | Physical wires | Source QEC model | Source peak model |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 31 | 4 | 1 | 1 | 5 | And=1, Clifford=7, measurement=1 | 5 |
| 2 | 55 | 8 | 2 | 1 | 6 | And=2, Clifford=7, measurement=2 | 6 |
| 3 | 77 | 12 | 3 | 2 | 8 | And=3, Clifford=21, measurement=3 | 8 |
| 4 | 99 | 16 | 4 | 3 | 10 | And=4, Clifford=27, measurement=4 | 10 |

These measured suffix inventories are separate from source QEC estimates and
from the full preparation's model-only resources. Width four has stream and
model evidence; full suffix complex-column/history checks cover widths one
through three. No full guiding-state primitive inventory, complex map or
preparation success bound is inferred from this suffix.

Concentrated regression passes 91 cases without skips per actual Cirq 1.7.0 /
Qiskit 2.5.0 and Cirq 1.6.1 / Qiskit 2.0.0 profile. Qualtran 0.7.0 and a real
Kickmix 0.1 Simulator are asserted. Earlier complete source/test/notebook and
seven-category model receipts are reused. The exact constructor catalogue
moves to 258/303; two instance-bearing roots and their original physical and
codec boundaries stay open. Q-0611, Q-0612 and Q-0613 are not closed by these
measured-child context checks.


The final local gate also passes 145 integration cases, strict mypy for all five
changed Python files, Black/Ruff, both owning-team scopes, English/local links,
architecture, multi-level IR and the protected API snapshot. Eight locked
source/direct-dependency blobs match the CSV. Constructor discovery invokes no
source constructor or child callback and finds no provider import failures.
