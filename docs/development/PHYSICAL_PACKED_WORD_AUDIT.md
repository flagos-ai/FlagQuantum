# Packed physical scalar words

## Complete modules and ownership

This batch implements the complete `b64_polyfill.h`, `b128_polyfill.h`,
`b128_sse.h`, `b256_polyfill.h` and `b512_polyfill.h` source interfaces.
Their immutable parents are K-0279, K-0273, K-0274, K-0276 and K-0278.
The lock is `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.
Thirteen original source and direct-dependency Git blobs are verified against
the frozen inventory before source compilation. All four portable headers and
the SSE header were read completely, alongside the AVX declarations and
architecture/test/benchmark selection headers.

Simulation `_packed_word` operates on the existing unsigned Python integer
representation. Low scalar lanes occupy low integer bits. It introduces no
second packed value class, state container or execution loop. Existing
`physical_basis_rng.packed_words` now uses shared packing. Existing
`_packed_chacha` uses shared packing and uint32 broadcast for state and buffer
reconstruction. Its scalar quarter-round and block algorithm remain their
existing authoritative implementations. Compiler continues to own physical
artifacts and lowering. Simulation owns their numerical application.

## Complete interface mapping

| Source behavior | Native behavior |
| --- | --- |
| `BIT_SIZE`, fixed width and scalar lane storage | Explicit width, bounded unsigned integer, `word_mask` |
| `from_u64s`, `u64`, `u32` | `pack_lanes`, `unpack_lanes` with explicit scalar width |
| `from_u32_broadcast` | `broadcast_u32` |
| `bit`, `set_bit` | Integer bit selection, `set_word_bit` |
| `non_zero`, equality and inequality | Integer Boolean and equality operations |
| XOR, AND, OR and compound assignment | Existing integer operators and value reassignment |
| Fixed-width inversion | Integer inversion masked with `word_mask` |
| Clear to zero and maximum | Assignment to zero or `word_mask` |
| Independent uint64 and uint32 addition | `add_lanes`, with scalar wrap and no inter-lane carry |
| Compound uint64 addition | Reassignment of the `add_lanes` result |
| Scalar left and right shifts | `shift_lanes` |
| Per-uint32 equality masks | `equal_u32` |
| Random construction and randomization | `random_word` consuming the supplied shared iterator |
| Stream diagnostics and the 64-bit `str` method | `format_word`, low-bit-first underscores and ones |

Original mutable value updates become explicit immutable integer reassignment.
Physical C++ ABI size, alignment and object addresses are outside this numerical
representation. Compiler aligned raw byte storage has its separate owner.
Native bit and scalar admission is checked before coercion. Invalid pointer or
out-of-range bit access is excluded from source numerical comparisons.

## Original execution and source-specific details

All 96 unchanged original tests pass on native ARM for portable widths
64/128/256/512. An actual x86 feature probe under Rosetta reports SSE2 available
and AVX/AVX2/AVX512F unavailable. Building the unchanged test source for x86_64
and executing it under Rosetta passes all 120 original tests, including SSE.
These are translated x86 functional results, with no hardware-throughput claim.

The fixture compiles unchanged source twice, for native ARM and x86_64.
It records inputs separately from constructor outputs, scalar getters, every
operator and mutation form, zero/max/partial-equality patterns, random values,
all valid shift counts on boundary words, every bit position, diagnostics,
selected aliases and subsequent caller RNG values. Its original source probes
use deterministic seeds. Fixture regeneration is byte-exact. The retained corpus contains 4771 ARM
and 5933 translated-x86 observations.

SSE random construction calls `_mm_set_epi64x(rng(), rng())`. In the actual
Clang execution, the first draw becomes the high lane. Portable aggregate
initialization places it in the low lane. `random_word(high_lane_first=True)`
retains that observed SSE route. Both random construction and re-randomization
verify the next caller draw. C++ argument evaluation order makes this observation
compiler-specific. Existing FlagQuantum seeded entries keep their canonical
portable low-lane-first stream, independently of host instruction selection.

SSE shift intrinsics narrow their size_t argument to 32 bits. A count of 2**32
therefore behaves as zero, while a count of 32 or 64 can zero the corresponding
scalar lanes. These cases are executed directly against the original SSE body.
Native `shift_lanes` follows this narrowing and zero rule. Portable comparisons
use counts strictly below their scalar width. Original portable shifts beyond
that range have undefined C++ behavior and supply no numerical oracle.

## Source defects and still-open files

K-0275 `b256_avx.h` is guarded by `__AVX__`, but its integer operations require
AVX2. Compiling the original equality/addition consumer with only `-mavx` fails.
It compiles with `-mavx2`.

K-0277 `b512_avx.h` is guarded by `__AVX512F__`, but its equality expansion uses
an intrinsic requiring AVX512DQ. The original consumer fails with only
`-mavx512f` and compiles with the additional `-mavx512dq`.
Neither AVX variant was executed on this host. Compile success supplies no
runtime evidence. These parents stay open. Source files are unmodified.

The complete architecture alias header K-0280 and macro test/benchmark headers
K-0281 through K-0283 retain open full-configuration disposition. Current ARM
and SSE selections are observed, including all original tests available in those
configurations. AVX and AVX512 runtime paths and their selection/gate behavior
remain explicit gaps. The full eleven-file group is therefore still open.

## Existing entry paths and resources

The actual existing RNG and ChaCha consumers are replaced between shared packing
and the previous local implementation without modifying the consumers.
Every width retains 65 consecutive ChaCha outputs and the subsequent caller
stream. The existing spin entry is run through both implementations. It retains
all packed quantum/classical values, every exact phase accumulator, sign,
measurement feedback and per-lane raw operation counters over three scans.
Both measured/reset wires are verified zero after each scan.

The existing sixteen frozen normal `circuit.append(op)` cases are reused for
ordered source operations, full complex five-qubit states, zero workspace and
all 512 exact basis phase/count lanes. Existing QROAM erasure and broadcast
measurement regressions cover their applicable measurement branches. The 3064
previously accepted original ChaCha lifecycle observations are rerun. The batch
passes 997 concentrated and 121 integration checks with no skipped cases,
strict typing on all five Python files, and 100% statement and branch coverage
of the new packed-word module. Format, scope, architecture, IR and protected
API checks pass. The exact set of 720 existing dependency findings is unchanged.

| Resource quantity | Result |
| --- | --- |
| Packed numerical helper circuit primitives | 0 |
| Packed numerical helper quantum workspace and measurements | 0 |
| Normal append ordered primitives and per-lane counters | Exact frozen source equality |
| Seeded spin execution state, phase and actual counters | Exact replacement equality |
| Analytical model added by this batch | None |

A five-sample local spin measurement at 2048 shots gave shared/previous medians
of 0.616/0.612 ms at width 64, 0.521/0.521 ms at 128, 0.472/0.467 ms at 256
and 0.706/0.715 ms at 512. Every repeated hash agrees. These observations concern
one host and tiny local runs. They establish no throughput improvement.

## Contributor path

Run `tests/unit/test_physical_packed_word.py`, `test_packed_chacha.py` and
`test_kmx_spin_command.py` in the established locked environment. Include physical
operation counters and existing broadcast measurement/QROAM erasure tests for
execution changes. Regenerate the fixture with
`tests/unit/fixtures/generate_packed_word.py`, the frozen CSV and source tree.
The `--x86` option requires an executable x86_64 environment and its compiler.
Run strict typing, formatting, scope, architecture, IR and protected API checks.
