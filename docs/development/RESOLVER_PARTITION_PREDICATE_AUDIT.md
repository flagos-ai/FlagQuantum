# Partition and grouped-predicate constructor dependency audit

## Complete source scope and ownership

The unchanged reference is Qualtran v0.7.0 revision
bcb17bb25360ed68ce3bc084645db3e929799389. This batch closes the constructors in
bookkeeping/partition.py and mcmt/ctrl_spec_and.py, including their direct
Partition, And, MultiAnd and specialized Adjoint dependencies. Six frozen
source/test/resolver blobs and the real Kickmix 0.1 Simulator are verified.

Core owns the approved Proposal 120 declaration boundary. Bodyless logical
functions preserve symbolic array shapes, including mixed dimensions and SymPy
numeric expressions. Dtype, side and name validation remains in Core. Finite
bodies, calls, element enumeration and physical binding require concrete shapes.
No implicit size substitution is performed. The approval was committed before
implementation.

Compiler formats translates the three Partition/Split2/Join2 constructors to
existing layout/component_partition.py and CtrlSpecAnd to existing
control/component_predicate.py. The existing stored-value codec owns control
encoding. The existing conjunction resolver retains ownership of unrelated
Adjoint children. A per-compilation recipe map supports exact predicate adjoints
and nested adjoints without inferring behavior from names. Users retain
circuit.append(op).

Ports use row-major arrays of MSB-first words. Partition and its inverse retain
physical views with zero instructions. Empty-width ports retain their logical
identity. CtrlSpecAnd passes the original control groups through and produces
RIGHT junk and target outputs. Its inverse consumes those exact LEFT outputs
and restores their clean state. Signed, unsigned, opaque, bounded,
ones-complement and Montgomery integer controls reuse existing encodings.

## Verification and resource observations

Named original libraries and decompositions are compared with the native
constructors. Tests compare gate order, operands, measurements, resets and
feedback. Full complex64/128 execution, coherent spectator inputs, all small
raw measurement histories and zero work are checked. Original source tensors
are compared at small sizes, with original flattened control predicates used
for larger cases. Existing complete partition and predicate evidence is reused.

| Operation | Actual primitive observations | Independent source model |
|---|---|---|
| Partition, Split2, Join2 | Zero gates, measurements and work allocation; ordered physical aliases preserved | Zero QEC gates; qubit count equals total storage width |
| Predicate with n control bits | n-1 And computations, 4(n-1) T/TDG | Original QEC categories and peak 2n-1 compared independently |
| Predicate inverse | n-1 measurements and resets, feedback preserved, consumed output image restored clean | Original inverse QEC categories compared independently |

The focused declaration/constructor coverage lane passes 68 cases with 100
percent statements and branches in both constructor owners. Concentrated
regression passes 2598 cases; local integration passes 145. The two unchanged
original test files pass 46 cases, with 12 original skips, three expected
failures and three notebook cases deselected. Seven Python files pass strict
checking, Black and Ruff. Core test scope, approved protected integration scope,
Compiler scope, language, architecture, IR and API checks pass. Dependency
findings remain identical to the existing 720-line baseline.

## Explicit remaining boundaries

Original CtrlSpecAnd decomposition serialization fails on numpy.uint8 And
arguments at depth one for two-bit controls and depth two for larger controls.
Original Shaped controls cannot be serialized, and HasLength codec limitations
remain. These reference defects are reproduced without modifying source or
normalizing its objects to manufacture a passing wire round trip. Gate and
state comparisons are independently executed through original decompositions.

The four complete constructor identifiers bring coverage to 64/303, leaving
239. Generic ControlledViaAnd/Controlled/Adjoint/Power, arbitrary-target
MultiControlPauli, partial OnEach and permissive non-signed sign conversions
remain open. Pending Proposals 116-119 and whole-repository/GPU/performance
acceptance remain separate. Batch receipts use
ftqc-reference-env/g2-partition-predicate-constructor-*.
