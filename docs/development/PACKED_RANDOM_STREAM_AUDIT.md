# Complete packed random stream source audit

## Interface and ownership before implementation

The source group is Kickmix `util/chacha.h` and `chacha.test.cc` at frozen
revision `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Apache-2.0. The source
and SIMD dependencies are Git-blob verified in
`ftqc-reference-env/g2-chacha-complete-intake-20261122.json`. All eight unchanged
original ChaCha8 and ChaCha20 tests pass across four portable packed widths.

Simulation already owns the scalar quarter round, eight-round block and seeded
packed generator in physical_basis_rng. The complete implementation will move
the shared quarter/block algorithm into private _packed_chacha and generalize
the round count. PackedChaCha owns immutable state[16], buffer[16], buffer index,
width and rounds. Each packed integer contains independent little-endian uint32
lanes. Existing packed_words retains its mt19937_64 owner. The current spin
consumer keeps its eager eleven-word seed consumption and circuit entry path.

randomize consumes eight key and three nonce packed words, resets each lane's
counter to zero and resets only the buffer cursor. It retains the old buffer.
refill overwrites all sixteen cached words, increments each lane's uint32 block
counter without nonce carry, and sets the cursor to zero. next_word returns a
word and successor, refilling when cursor >= 16. Repeated randomize, explicit
refill and arbitrary positive even rounds require independent source evidence.
The immutable draws iterator owns its successor state. Independent iterators
start from the same snapshot. No caller-owned RNG is cloned or reset.

Core owns no new contract. There are zero emitted quantum gates, measurements
and work qubits for this classical random stream. The original class declares
no quantum resource model. Original architecture-specific SIMD definitions
retain their separate source scope.

## Complete definition disposition

| Frozen definition group | Authoritative native owner |
| --- | --- |
| rol32<K> | _rotate32 preserves portable-defined amounts 1 through 31 and the complete defined SSE template count families, independently per uint32 lane. Counts 0/32 preserve the lane; other outlying SSE counts clear it. Signed template arithmetic remains bounded. |
| chacha_quarter_round and reference aliases | _quarter retains the source add/XOR/rotate order, including shared-index aliases |
| Template state, buffer, cursor and round constraint | PackedChaCha immutable fields, zero construction, positive even round validation and original uint32 lane width |
| randomize | randomize consumes exactly eleven packed words and preserves old cache storage while resetting cursor and counter |
| refill | refill reuses the moved existing scalar block algorithm, with column/diagonal ordering, feedforward, per-lane counter wrap and buffer overwrite |
| next | next_word returns a word and owned successor, with source refill when cursor >= 16 |
| ChaCha8 and ChaCha20 aliases | rounds=8/20 share one algorithm and state owner |
| All eight original test scenarios | Original unchanged execution and native tests using the same frozen initial-state and expected-word arrays at all four portable widths |
| Existing seeded physical spin consumer | packed_chacha8 delegates to the shared owner with eager seed consumption; KMX/KMB spin hashes and integration tests pass |

The source corpus records 3,064 complete states and output sequences, including
all 256 reference-alias patterns at each packed width and every defined rotation
amount 1 through 31. Complete streams cover 2/8/12/20 rounds, widths
64/128/256/512, two seeds, zero/raw/seeded states, old cached buffer reads,
reinitialization, explicit double refill, UINT32_MAX counter rollover, SIZE_MAX
cursor refill and lengths 0/1/15/16/17/31/32/65. State, buffer, cursor, each output
and the next caller-owned mt19937_64 word match the original source. The corpus
regenerates byte-for-byte after checking all 13 frozen source/dependency blobs.
Four separate original compilations verify invalid round counts 0/1/3/-2.

## Explicit source and native boundaries

Original rol32<0> and rol32<32> instantiate uint32 shifts by 32 in the portable
SIMD implementation. UBSan reproduces both failures. The earlier native helper
refused both inputs. The complete SSE-defined domain is now implemented below;
portable UB witnesses remain separate and unchanged.
The actual round algorithm uses only 16/12/8/7 rotations. The original header
is unchanged. Evidence is retained in
`ftqc-reference-env/g2-packed-chacha-rotation-boundary-20261122.json` and
`g2-packed-chacha-original-round-refusals-20261122.json`.

The [2026-11-28 closure review](G2_FUNCTIONAL_MIGRATION_DISPOSITIONS.md) did not
accept K-0291 solely from those portable UB witnesses. The unchanged SSE header
successfully executes rol32<0> and rol32<32> in an x86_64/Rosetta UBSan probe;
both return the input word without sanitizer errors. The native helper rejected
both amounts, establishing a source-defined implementation gap. That gap is
resolved by the implementation and complete source-domain evidence below.

## Defined SSE rotation closure

The scalar lane owner now retains the original SSE shift semantics. Counts
1..31 perform the original rotation, 0/32 retain the input lane, and other
source-valid int template counts produce zero. The source expression 32-K
must fit signed int; the native owner rejects template-domain overflow instead
of treating it as a valid source result. Values retain uint32 storage. This
changes no round behavior, since quarter rounds use only 16/12/8/7.

The new frozen-header generator checks both original Git blobs and calls the
unaltered SSE implementation under UBSan. Forty-three template amounts include
every 0..32 count, negative/outlying count families and both signed-domain
limits. Each uses all 32 basis vectors and four additional words, with four
distinct lane patterns: 1,548 complete vectors and 6,192 scalar comparisons.
The headers' per-lane Boolean-linear shifts/OR and the complete basis coverage
establish the defined uint32 operation; no random sampling substitutes for
the count-family boundary checks. The corpus regenerates byte-exactly with
no sanitizer diagnostics. This is numerical conformance, not SSE acceleration.

The original 3,064 complete lifecycle observations remain unchanged, including
all alias patterns, state/cache/counter behavior and caller RNG consumption.
Current spin, lifecycle, workload and normal register-execution regressions
validate the actual consumer paths. Source UB and ISA-specific stream ordering
remain explicit; no cross-ISA timing or cryptographic certification is inferred.
K-0291 now has its complete original function responsibilities and defined
rotation contexts implemented. Its portable UB follow-up follows the already
approved G2 rule, rather than remaining a missing-source-path hold.

Immutable successors replace C++ mutation. Cached state survives failed seed
construction. Source RNG consumption remains caller-owned, so failed seed
attempts can still consume available caller words. The existing generator's
insufficient-seed StopIteration, positive-width rounding and upper seed-bit
truncation are preserved. Its nonpositive-width nonterminating path now has an
explicit width refusal. The physical spin entry already validates its four
supported widths before reaching this private stream adapter.
Native width 32 and positive-width padding are compatibility evidence for the
pre-existing Python adapter, with original packed template correspondence
limited to the four compiled portable source widths. Positive even round counts
are admitted, with execution evidence for 2/8/12/20. Performance and cryptographic
certification have no acceptance claim in this batch.

## Verification and resource account

All 250 concentrated tests and 121 local integration tests pass without skips.
The new state owner has 100% measured statement and branch coverage
(66 statements and 18 branches). Strict typing covers all four changed Python
files. Black, Ruff, owning team scope, architecture, multi-level IR and public
API checks pass. Whole-repository dependency policy retains 720 existing
findings and zero touched-path findings. Existing complete complex amplitudes,
measurement-branch feedback and work-zero register tests are reused in the
integration run. The spin hash suite also exercises the actual replaced seeded
execution path through both KMX and KMB inputs.

Actual emitted quantum primitives, measurements and work bits are zero for the
scalar RNG itself. The source has no quantum analytical model. Numerical stream
observations are separate from physical circuit primitive/resource evidence.
The previous default-suite boundary remains recorded in the integer batch audit.
