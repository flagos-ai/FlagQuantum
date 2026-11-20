# Classical binary-field feedback dependency closure

## Source and scope

Proposal 110 was explicitly approved on 2026-11-20. This closes the concrete
classical binary-field and polynomial storage dependency of the complete
`controlled.py` and `classically_controlled.py` source files. Existing owner
maps remain authoritative. Parent acceptance still includes separately tracked
symbolic, foreign-interface and source-defect boundaries.

The reference is unchanged Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The three blobs for controlled.py,
data_types.py and classically_controlled.py match the original inventory.
Recursive controlled.py inventory has 49 definitions, including the two nested
callbacks omitted by the earlier count of 47. Existing metadata identity and
returned attachment evidence cover those callbacks.

## Interface and ownership

Core projects classical storage to the existing quantum field/poly validator.
The storage marker stays first, quantum width is zero, and canonical field
metadata, optional polynomial identity and concrete binary widths are retained.
The polynomial word stores highest-degree-first coefficients, each MSB-first.
Arrays are row-major. Same-width nominally different fields remain distinct.

Compiler reuses `classical_word_bits`, owned reads, binding and feedback graph
composition. Entry values are built-in unsigned stored integers. Field scalars
and polynomial objects belong to Simulation codecs. Each read returns its
identically typed linear successor and ordered read-only bit snapshots. Reads
emit zero primitives and allocate zero new slots. Nested calls forward the same
slots. Empty arrays retain no elements. Split/Join retain the existing
quantum-only operation restriction. No new production owner is introduced.

The normal entry remains `circuit.append(operation)`. Run
`python -m examples.algorithms.typed_classical_feedback` to verify signed,
GF(4) and binary polynomial controls with complex spectator states.

## Comparisons

Six families cover GF(2), GF(4), GF(8), constant GF(2) polynomials, and degree-one
polynomials over GF(2) and GF(4). All activation/input words up to four stored
bits are exhausted. Original tensor blocks match complete native complex
states, with an entangled spectator and the clean private work returned to zero.
Original child primitives and operands are preserved. Actual append execution
covers both extreme activation/input words for each family. A mixed shaped
field/poly fixture exhausts all 64 input combinations in row-major order.

Measurement-produced CBit controls are conjoined with the field word through
existing nested calls. Both outcomes, both extreme activations and every small
input word match the original numerical callback and complete unnormalized
complex branch state. The measured consumed input is kept distinct from clean
workspace. Scalar/shaped/empty repeated calls retain classical output slots.
Invalid stored words and malformed descriptor projections fail before binding.

The immutable integer compatibility fixture records exact pre-extension
Program repr, CircuitIR JSON/hash, output aliases and slots. Existing Proposal
108 CBit and Proposal 109 quantum polynomial fixtures remain passing. One former
polynomial test expected the newly approved classical descriptor to fail
identity admission. That expectation is updated explicitly under Proposal 110;
its quantum-width entry points continue to reject classical storage.

## Resource accounting

| Scenario | Actual emission | Clean work | Classical slots | Original estimate |
| --- | --- | --- | --- | --- |
| Repeated stored child, width w | 4 CX, 2 phase, 2 H, w classical initializers | 1 reusable bit, 4 allocation/free events | w, no read extras | Child 2 T and 6 Clifford, wrapper model absent |
| Field word plus measured child control | H, measurement, guarded X, w initializers | 0 | w entry slots plus 1 measured CBit | Numerical wrapper reference, no wrapper model |
| Runnable two-bit example | Guarded X and 2 classical initializers | 0 | 2 | Wrapper model absent |

The original classical wrapper has no physical decomposition. For decoded GF
activations its resource cache can fail earlier because a field scalar is
unhashable. Both original failure classes are retained by tests. Original
polynomial queries reach the absent decomposition. Child estimates and actual
physical instructions are recorded independently in `g2-classical-field-evidence.json`.

## Validation and retained boundaries

Acceptance passes 107 focused cases, 4260 concentrated field/control/native
Kickmix cases without skips, 17 local integration cases, and strict typing of all
eight changed Python files, Black/Ruff, classified scope, architecture, IR and
public API checks. Whole-repository success is not asserted. Existing generic
logical/partition baseline failures remain documented in the packed polynomial
audit. The dependency audit retains 687 whole-repository findings and zero in
this batch. Qualtran 0.7.0 and native Kickmix 0.1 Simulator are verified.

Nonbinary and unresolved symbolic physical descriptors, conditional measurement,
reset and classical writes retain their separate boundaries. Generic foreign
object contracts and original source contradictions remain explicitly tracked.
The classical-field intake records the pre-extension refusal and is historical
evidence. The approved contract and this acceptance supersede that refusal.
