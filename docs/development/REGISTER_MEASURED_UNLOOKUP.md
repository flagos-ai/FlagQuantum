# Measured lookup erasure contract

G2-KLUT03 ports unchanged locked Kickmix `gen_unlookup` from
`src/kickmix/gen/lookup/gen_lookup.cc`, commit
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.

The compiler owns the source composition: initialize a full classical phase
table, HMR each output in low-bit-first order, accumulate measured table parity
inside saved conditions, unfold low address bits into unary, run full-table Z
lookup on the high address bits, then erase unary back to binary. It reuses
shared conditional XOR execution, binary/unary conversion and recursive lookup.
No source branch, feedback or operand schedule is replaced by an alternative.

All external quantum words and table rows are big-endian. Table entries are
bool or caller-defined ClassicalFieldCoefficient values; quantum table data is
rejected as in the source. For n address bits the caller supplies 2**n distinct
phase slots plus one distinct reusable HMR slot, disjoint from table inputs.
Phase slots are bound in increasing row-number order, not bit significance.
Classical allocations whose slots were recycled by the source allocator may
be listed as an ordered prefix in reused_phase_bits: emit the source allocator's initial zero store
before the explicit whole-table zeroing pass. Fresh slots have no extra store.
This supports both fresh fragments and a second invocation in the same native
allocator namespace without hiding the difference in source primitive counts.

For h=n//2, the unary buffer has 2**h wires. It takes the h low address wires,
then cleared output wires in source order, then caller clean work until full.
The remaining clean-work suffix is leased by recursive lookup on n-h high
address bits. Thus extra clean quantum work is
max(0, 2**h-h-output_width)+(n-h). All consumed work and output wires must return
to zero. Address and any entangled spectators are preserved. Correct erasure
requires output initially equal to the selected table row (missing rows mean
zero); this is not arbitrary quantum-data reset with coherent preservation.

Empty output returns no operations. Portable source address widths are 0..30.
Core/KMX artifacts use explicit caller-owned ports; no stable API or public
schema changes. Full source streams, coherent branches, work, resources,
recycled-slot behavior and the append example are separate acceptance checks.

## Source and execution evidence

The locked source blob is `d527eeeddf06c10ffe69ab6408594c60fd248182`.
The independent C++ harness links unchanged native objects. All 783 complete
streams match at address widths 0..6, output widths 0..3, constant/classical/mixed
tables and empty/full/partial row sets. They include 63 full compute/erase
chains and fresh versus recycled classical allocation. Only register
metadata/informational tags are excluded. No phase, reset, initialization,
feedback, source operand or redundant store is normalized away. Fingerprints
are retained in `tests/fixtures/kickmix_measured_unlookup.json`.

For address widths 0..2, every reset/HMR branch is projected for all four
classical-input pairs, two quantum layouts, full/partial tables and all table
modes. Full unnormalized complex vectors agree, preserving arbitrary entangled
address/spectator amplitudes while clearing output and quantum work. Widths 3
and 4 additionally enumerate every branch of selected mixed-table inputs,
including the extra clean unary-buffer wire at width 4. Full compute/erase
round trips enumerate all branches through width 2. The earlier incomplete
forward-lookup phase defect is not hidden: partial erasure tests prepare the
required ideal table data directly; end-to-end source compute/erase tests
use full tables.

Locked native Simulator independently verifies outputs, address/classical input
preservation, zero work and exact Fraction phase across 64 shots for 72 mode
combinations through six address bits. These larger basis/phase checks are
not dense arbitrary-state evidence. Dependency replacement fakes prove that
binary encoding, full Z lookup and measured binary restoration can be swapped
without modifying this consumer, while preserving buffer/phase/work bindings.

## Resource bases

For two address bits, two output bits and a full classical table, erasure uses
one clean quantum work wire, four phase slots and one reused measurement slot.
Source primitives total 37: 4 BIT_STORE0, 4 HMR, 2 PUSH_CONDITION, 8 BIT_INVERT,
2 POP_CONDITION, 2 R, 2 SWAP, 4 X, 3 CX, 4 CZ, 1 Z and 1 conditional NEG.
Binding produces 53 Core instructions:19 classical_xor, 6 H, 4 measure, 10 X,
2 reset, 2 swap, 3 CX, 4 CZ and3 Z. Two caller table slots plus owned slots and
one saved mask give eight bound classical slots. Classical phase work is not
required to finish zero; the source releases dirty classical values and clears
them when recycled. Quantum outputs/work do finish zero.

Tree-shape counts close independently through eight address bits. With h=n//2,
H=n-h, B=2**h and N=2**H-1, source counts are
CCX=max(0, 2**H-2), CSWAP=B-h-1, R=N+B-h and HMR=output_width+N+B-h.
A full all-classical width-one table has 2**n initialization stores and parity
updates. Additional clean quantum work is max(0,B-h-output_width)+H.
These are logical source counts and emitted-IR counts, not a synthesized T
measurement or an upstream analytical T model. The locked function declares
no separate T model; arbitrary CCX/CSWAP synthesis choices must be specified
before converting either count to T resources.

Run `python -m examples.algorithms.measured_unlookup` for a coherent
`circuit.append(op)` compute/erase example with an entangled spectator. It prints
source erasure, source whole-chain and expanded whole-chain counts separately.
No upstream package is required to execute the example.

Local acceptance: 247 new tests; 864 focused regressions passed with four
optional Qiskit/Aer skips. Strict mypy (two production/example files), Black,
Ruff, compiler ownership scope, architecture and diff checks passed. Full
original native fuzz/performance suites, generic external condition scopes
and typed/reverse conditional-assignment merges remain separate open items;
this batch does not declare G2 or the entire source file accepted.
