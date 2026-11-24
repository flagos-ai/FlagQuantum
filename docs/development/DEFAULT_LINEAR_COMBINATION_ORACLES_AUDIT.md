# Default LinearCombination native oracle closure

Current constructor/control disposition is recorded in [the repeated-control audit](REPEATED_LINEAR_COMBINATION_CONTROL_AUDIT.md). Earlier open-dependency statements below describe their historical acceptance boundary.

## Scope and ownership

The unchanged original `block_encoding/linear_combination.py` example contains
four signed quarter-weight Unitary blocks: T, H, X and Z. This batch replaces
its actual default PREPARE and SELECT objects with native directional Programs.
Ecosystem owns external object inspection and immediately converts register
metadata through the existing source register wire and native decoder. No SDK
object enters Compiler, and no private coefficient is inferred from a lossy
serialized root declaration.

`_default_linear_combination_oracles.py` supplies explicit callbacks to the
already accepted `capture_linear_combination_context`. PREPARE uses the existing
complete alias constructor resolver and packed oracle owner, including exact
unpreparation. SELECT uses typed row-major indexed calls and explicit positive
child owners. Existing Phase, Unitary and AutoPartition source roles retain
their actual boundaries. Fixed H/X/Z controls reuse registered CH/CNOT/CZ
definitions; original atomic controlled T executes through existing cphase
with angle pi/4, with no fabricated upstream deeper decomposition.

The new shared controlled AutoPartition owner transforms the existing data
partition template with explicit child and bookkeeping control adders. It never
partitions the control together with data. The temporary logical child
declaration is replaced in all calls and removed before verification/binding.
Its source child data names are explicit and can differ from specialized gate
port names. Unused packed slices remain caller-owned bystanders. Exact reverse
partitions reuse the existing AutoPartition inverse owner.

## Source order and inverse distinctions

The unchanged generic control visitor visits the block before the disconnected
Phase scalar, although the scalar was inserted first by the constructor. Both
the default forward and nested inverse comparisons preserve this control SSA
dependency. A second distinction matters: `Controlled.adjoint()` returns control
of the source child's adjoint. Its controlled Phase inverse still visits the
inverse block before negative ZPow. An explicit generic
`Adjoint(Controlled(Phase))` instead reverses the two already-controlled calls.
Both are supported by their actual native recipes and separate tests; equivalent
complex matrices do not justify substituting their call graphs.

## Numerical and execution evidence

`test_default_linear_combination_oracles.py` compares every recursively available
original call, named operand and output for packed PREPARE, SELECT, positive
SELECT and the full default root in both directions. All have executable native
bodies. Full complex operators cover every logical input basis column for the
small default, including arbitrary logical junk and the controlled root. The
shared erasure certificate proves every raw measured-And history has the same
complex operator with its positive normalization factor; full basis columns
separately verify final private work zero.

A nonuniform signed (1/8,-1/4,3/8,-1/4) coefficient table uses original
sub-bit precision 8 and confirms source mu=8. Its full logical state is too large
for the small dense proof, so existing independent sparse interpreters compare
complex amplitudes against the unchanged source Cirq decomposition, retaining
classical feedback. Zero and one raw histories and the exact forward/inverse
roundtrip restore a complex system superposition and all work. The same shared
certificate establishes the erasure identity for all histories; this is not a
claim that every large history was numerically enumerated.

After explicit source capture and shallow library decoding, source imports are
disabled. Existing native constructor resolution, physical binding and ordinary
`circuit.append(op)` reproduce the same complete primitive stream. The actual
Kickmix 0.1 Simulator and Qualtran 0.7.0 are asserted in the reference tests.

The external `g2-default-lcu-oracles-resources-20261125.json` has twelve
directional inventories. Actual primitive counts/private lifetime peaks and
unchanged original QEC/logical-peak estimates occupy separate fields. Models
are neither substituted for measured gates nor asserted to be identical to
expanded primitive inventories. Private lease release and numerical zero-work
proofs are separate obligations.

## Remaining complete-file boundaries

This closes the original default native oracle dependency and its positive
control/inverse directions, not every arbitrary child constructor. Other child
types and already-controlled indexed oracles require explicit native owners.
The existing context interface retains arbitrary supplied PREPARE/SELECT owners;
it does not guess them from names, wire signatures or resource models.

Repeated control of an already-controlled LinearCombination and its source
specialization recipe remain to be closed before full constructor catalogue
acceptance. Original private-payload omission, deeper object/ragged tuple wire
failures and specialized inverse enum serialization failures remain explicit;
none is repaired in the reference. Genuine executed registerless phases
elsewhere still require pending Proposal 119. Pending 116/121 and unrelated
repository/GPU/performance gates are unchanged. The catalogue remains 114/303
until remaining complete-constructor acceptance is proved.
