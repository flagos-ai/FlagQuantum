# Partial controlled source unitary boundary

The frozen reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

The And source explicitly declares `_has_unitary_` false when uncomputing.
Its controlled wrapper lacks a decomposition because the target is LEFT.
The wrapper nevertheless attempts Cirq unitary evaluation through a controlled
child before checking the sided signature. In the current SDK lane that attempt
recurses indefinitely. The old native importer inherited this recursion.

After all actual graph and SDK decomposition paths decline, the native importer
now honors a source leaf's explicit false unitary declaration before attempting
matrix fallback. This check unwraps control wrappers only to read the child's
protocol. It does not bypass declared native definitions, decompose a partial
operator into another circuit, or reject computed And's available unitary.
The error identifies the unavailable leaf and preserves the source limitation.
No public API, serialization vocabulary or physical admission changes.

Eight computed And cases cover all three control polarities, and their complete
16-column complex unitary agrees with the original SDK on reversed physical
wires. Both erasure-control polarities with one or two outer controls produce the
explicit refusal. Existing exact source boundary and register importer tests
remain applicable. This change emits no additional quantum primitive, workspace
or measurement, and supplies no independent resource estimate.

The actual controlled ECWindowAddR intake reaches the same controlled first-stage
erasure and now reports `C[And†]` in about 1.91 seconds. The source QROAM and
uncontrolled first-stage children import before that error. The whole ECC parent
remains open. Evidence is in `g2-partial-control-ecc-followup-20261121.json` and
`g2-partial-control-unitary-baseline-20261121.log` in the reference directory.
