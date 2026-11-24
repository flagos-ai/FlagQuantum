# Typed linear-combination root and dependency disposition

Unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, source/test blobs
`a3d8aa7fafa7a6f3fcaf095826751209427cd255` and
`bed60bc7ade281fca1e5b13e341e8b4dd07aabd9`, Apache-2.0, Google LLC.

## Shared root ownership

Compiler preparation owns `linear_combination_component.py`, receiving exact
native PREPARE and SELECT pairs and resolved widths. The existing Algorithms
signature entry delegates to this owner without changing its contract.
Coefficient normalization, alias preprocessing and child selection retain their
existing Algorithms/Ecosystem boundaries. No SDK or numerical execution is
introduced into Compiler.

The root explicitly partitions ancilla as selection then largest child ancilla,
and resource as largest child resource then PREPARE junk. Child system packing
is system, ancilla, resource. PREPARE precedes packing/SELECT and its exact
inverse precedes final unpacking and reassembly. Only SELECT receives positive
control. Zero-width optional portions retain the original omissions. Symbolic
layouts remain declarations. All packed words, including logical junk, belong
to the caller; child-private work must return clean.

Exact inverse pairs reverse the complete root graph. Self-inverse mathematical
actions do not authorize reusing their forward call order: the wide independent
SELECT fixture uses an explicit reversed OnEach child, preserving the source
bit-call order even though all its X gates commute.

## Evidence and resource boundaries

The supplied independent native oracle witnesses cover ordinary/positive-control
roots and generic exact inverses, every recursive comparable call and operand,
complete small complex operators, both controlled raw measurement branches,
their branch weights and private-work zero. Separate child-ancilla-only,
child-resource-only and combined cases verify both directions and full complex
operators. Source-free native construction, binding and `circuit.append(op)`
work with Qualtran and Cirq imports blocked. Existing coefficient, catalogue,
custom oracle, structured/repeated control and packed-block regressions retain
their original accepted scope; they do not prove missing constructor payloads.

Verification: 1026 concentrated existing/root cases plus seven additional
focused partition/source-free cases pass, with 145 local integration passes.
The unchanged non-slow source run yields 45 passes, one source mode skip and
one original serialization xfail; fifteen slow cases are deselected, with
their existing native catalogue/dense witnesses rerun in the concentrated
scope. Four files pass strict typing, format, ownership, language, architecture,
IR/API and whitespace checks. The dependency findings remain byte-for-byte
equivalent to the established 720-line baseline; no new finding is accepted.

`g2-linear-combination-component-resources-20261125.json` records ten actual
directional rows separately from source QEC and logical-peak models. The small
uncontrolled witness emits two H, S, S-dagger, two X, one CX and one phase,
without private work; the source reports seven Clifford, one rotation and peak
three. Positive control emits four T-family, one atomic CCX, six CX, five H,
two S, S-dagger, two X, one phase and one measurement/conditional CZ/reset,
with one private wire; source reports one Toffoli, one And, seven Clifford,
one rotation, one measurement and peak five. The atomic Toffoli and phase
boundaries are not silently expanded into a different counting vocabulary.
All leases return empty; complex-history checks independently establish zero
work. Pool capacity is not live peak.

## Full constructor remains open

This delivery accepts the shared typed root and its tested exact native
dependencies. It does **not** close the entire LinearCombination constructor or
increment the frozen 303-entry catalogue. The original depth-zero producer
omits private block, coefficient and optional-oracle fields, leaving only
precision and the control flag. Different operators can therefore have the
same surviving payload; inferring one would be incorrect. Original deeper
object/ragged array codecs and specialized inverse enum codecs also fail.

The original default signed SELECT additionally needs Phase scalar bodies
under pending Proposal 119. That protected behavior is not implemented here.
Full constructor closure requires explicit native role/child metadata and its
Ecosystem boundary, signed Phase admission when authorized, and remaining
source constructor/control/alias chain validation. These gaps stay owned by
open Q-0876, alongside other G2 constructor work; the older Algorithm evidence
does not discharge them. G2 and G3 status are unchanged.
