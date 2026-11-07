# Source recursive lookup contract

G2-KLUT02 ports locked Kickmix `gen_lookup` in
`src/kickmix/gen/lookup/gen_lookup.cc`, commit
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

The register compiler owns the recursive highest-address split. The interface
uses big-endian address and output tuples and a tuple of big-endian table
rows. Row numbers retain their integer values. Table entries are bool constants,
`ClassicalFieldCoefficient` references to caller-defined classical inputs, or
integer quantum wire IDs. Quantum table wires may repeat across entries but
must not alias address/output/control/work. Output is XORed for X lookup or
phased for Z lookup. An optional quantum control is coherent.

Each recursive level leases one clean wire from `RegisterWorkspace` in caller
order; the two halves reuse the same suffix pool. Resets, negative-address X
pairs, low/high boundary CX, measured And erasure and result-slot reuse are
preserved. The result slot cannot alias a classical table input. No producer
is allowed to overwrite table/control/address values. Empty output returns no
operations, matching the source early return. Nonempty output needs at least
one equal-width row. The source uses a signed-int address shift; this port
accepts portable source widths zero through 30 and does not claim behavior
for the source's undefined larger shifts.

The source has a separate recursive schedule and does not call the unary
cursor. Do not replace that schedule merely to share traversal. Incomplete
rows and source measurement phases must be tested separately from ideal
lookup correctness. Keep source defects visible without changing reference
code or calling a modified schedule source-identical.

## Acceptance evidence

The source blob is `d527eeeddf06c10ffe69ab6408594c60fd248182`, already
verified against the locked upstream Git object. An independent C++ harness
links the unchanged native objects. All 480 complete streams match at address
widths zero through four, output widths one and two, X/Z, both quantum-control
modes and constant/classical/quantum/mixed table entries. Both full and partial
tables are covered. Only register declarations and informational tags are
excluded; all reset, HMR, phase, feedback and result-slot operands remain in
the committed fingerprints.

For complete tables, every small reset/HMR branch matches the independent
ideal lookup on the full unnormalized complex vector through two address
bits. Quantum control, arbitrary output, quantum table data and spectators
remain entangled; all work clears. Locked native Simulator tests independently
verify basis outputs, exact Fraction phases, work cleanup and preservation of
classical table inputs across 64 shots per mode. Larger numerical widths and
the original full upstream fuzz/performance suite remain separate open work.

The full two-bit controlled all-true single-column table emits 25 source
primitives: 6 X, 3 R, 3 CCX, 7 CX, 3 HMR and 3 CZ. Peak clean work is two
qubits, with one reused result slot; extra provided work is untouched.
The expanded Core IR contains 35 instructions: one classical initialization,
9 X, 6 H, 3 reset, 3 CCX, 7 CX, 3 measure and 3 CZ. These are syntactic
counts, not a T-resource model. In particular, an unsynthesized CCX must not
be counted as zero T cost merely because its literal opcode is not T.

Run `python -m examples.algorithms.recursive_lookup --pauli x` or `--pauli z`
for full-table coherent append examples. They print source counts and expanded
IR counts separately. Runtime examples need no upstream dependency.

## Preserved upstream defect: low-only incomplete subtree

For a subtree with rows only in its lower half, the source builds
`anc = control AND NOT address_msb`. It never performs the low/high boundary
CX, then erases with feedback controlled by `control AND address_msb`.
For HMR outcome one the mismatch adds Z on the parent control. With a true
parent this is an outcome-dependent global sign; with a quantum parent it
changes coherent relative phases. Thus missing rows are not generally a
correct zero-filled controlled lookup in the locked implementation.

A one-bit, one-row all-false table should ideally act as identity. The locked
native Simulator independently gives phase half turns
`control * erasure_outcome (mod 2)` on basis inputs. Exact small branch tests
show that control superposition becomes orthogonal to the ideal result for
outcome one, while work still clears. The corresponding all-true case exposes
the same defect after the intended conditional output flip. This source
behavior is preserved and explicitly diagnosed, not repaired in the reference
or claimed to be ideal partial-table correctness. Callers requiring ideal
coherent lookup should supply a complete table, including explicit zero rows.
No upstream issue or contribution is submitted by this work.

Local acceptance: 135 new tests; 464 focused regressions passed, no skips.
Strict mypy (two production/example files), Black, Ruff, compiler scope,
architecture and diff checks passed. Lookup stream parity and full-table
correctness are accepted; ideal partial-table coherence remains a known
upstream limitation. Unlookup is the next separate module.
