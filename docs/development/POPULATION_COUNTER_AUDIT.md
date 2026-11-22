# Packed population counter source audit

## Interface and ownership before implementation

The frozen group is Kickmix `util/pop_counter.h`, `.test.cc` and `.perf.cc`
at `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0. Core owns a
private immutable PopulationCounter with width 64/128/256/512, six packed
striped accumulators, 32 paired accumulators, uint32 shared count and uint8 step.
Bit k selects the k-th lane, ordered from the least significant bit. Each
uint64 segment contains two uint32 counters. Carry wraps independently at each
uint64 boundary. masked_increments returns a successor, compute_total reads one
lane and clear returns the zero state. Caller-provided masks and prior snapshots
remain immutable. SIMD object layout is adapted into Python integer storage.
Source SIMD definitions retain separate parents. This file maps only the complete
packed operations consumed by this counter to accepted scalar bit/word owners.
There are zero emitted quantum primitives, measurements and work qubits.
No quantum resource model is declared by this source.

All source and SIMD dependency blobs are verified before executing the unchanged
original tests. Performance goals and observed Python/C++ timings will remain
separate from counter value and full accumulator-state comparisons.

## Complete source disposition

All 12 unchanged original tests pass across the four portable packed widths.
The source compiler on this host exposes the portable implementations. SSE/AVX
hardware definitions retain their separate architecture-specific parents.
The generator verifies all 14 frozen source/SIMD blobs before compilation.
Its 960 full-state observations compare six striped words, 32 paired words,
step, shared count, every lane total and the next caller-owned RNG word.
All-off, all-on, independent random masks, mask aliases, clear, uint8 step
rollover, near-overflow paired lanes and shared_count boundaries are covered.
Repeated calls cross 31/32/33 and 63/64/65 boundaries and extend to 10,000 updates.

| Complete source definition | Native mapping |
| --- | --- |
| BATCH_SIZE and aggregate initialization | width and immutable zero state for 64/128/256/512 positions |
| bit_totals, paired_totals, shared_count and step | Immutable tuple/int fields with the original packed capacities and ownership |
| clear | clear constructs a zero successor and preserves the prior snapshot |
| compute_total | compute_total retains uint32 paired extraction, five-bit scaling, all six striped weights and shared count |
| masked_increments | Advance step before transfer, add independently within uint64 lanes, clear transferred striped bits and propagate the mask through six XOR/carry layers |
| All three original test bodies at four widths | Unchanged execution, full source-state corpus and an independent per-position count oracle |
| Complete benchmark body | Original evolving XOR/u64-add mask and counter state compared through 10,000 updates at each width |

Native indices and masks fail closed outside storage capacity. The source's raw
finite-capacity overflow and cross-uint32 carry within a uint64 are preserved.
No arbitrary-precision count claim is inferred from the uint64 return type.
SIMD clear, broadcast, shifts, bitwise operations, per-uint64 addition and lane
reads consumed here map to immutable integer storage and accepted word carry.
The remaining complete SIMD interface definitions stay in their own source scope.

## Performance and resources

| Portable width | Original goal ns/update | Observed original ns/update | Observed native ns/update |
| --- | --- | --- | --- |
| 64 | 1.9 | 1.44 | 4,178 |
| 128 | 2.1 | 2.70 | 4,987 |
| 256 | 2.3 | 6.27 | 6,712 |
| 512 | 1.9 | 9.29 | 10,220 |

Original observations use the best of four measured complete benchmark loops
at 10,000 updates. Native observations include the Python state/update workload
at the same iteration count. These are local observations and retain their
separate source goals. K-0320 remains open because native performance misses
those goals. All four final states match the original source. Receipts are
`ftqc-reference-env/g2-population-counter-source-timings-20261122.log` and
`g2-population-counter-performance-20261122.json`. There are zero emitted quantum
primitives, measurements or work qubits, with no declared source quantum model.

## Batch verification

All 145 concentrated tests and 92 local register integration tests pass without
skips. The new counter file has 100% measured statement and branch coverage
(49 statements and 20 branches). Strict typing covers all four changed/direct
dependency Python files. Black, Ruff, team scope, architecture, multi-level IR
and public API checks pass. All 960 original states regenerate byte-for-byte.
Dependency policy retains 720 existing findings and zero touched-path findings.
Prior whole-default-suite and upstream defect boundaries remain separate.
