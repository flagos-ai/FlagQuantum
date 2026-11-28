# Complete striped modular random sampling intake

The frozen group is Kickmix util/mod_rand.cc, .h, .test.cc and .perf.cc at
9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453. Source hashes are verified before
implementation. FixedWidthInteger, word_ops and standard mt19937_64 reuse the
accepted scalar owners. The original physical initialization consumer and its
current Python Random-based native owner are identified separately.

Simulation fixed_width_sampling owns this scalar sampler. The interface takes
an immutable Core modulus and caller-owned uint64 draw iterator. It returns
the output lane words and final scratch snapshot as immutable tuples. Tuple
length is the declared modulus width. Bit k of each lane j is stored in
(output[k] >> j) & 1. Quantum register MSB ordering keeps its existing boundary.

The modulus must have positive declared width and its highest declared bit set.
Original span-size checks become construction invariants of the returned tuples.
Native immutable outputs have independent owned snapshots. The shared private
mutable-span kernel also represents caller-owned overlapping storage. The existing physical initialization consumer has its own seeded Python Random
contract. Exact simulator-engine binding belongs to its complete source intake.
Source early full-batch replacement, partial-batch extraction, retained last 64
accepted samples and final RNG position require exact observation. Original
performance goals remain independent from native observed timings.


## Complete source disposition

The sole source function and header declaration map to sample_striped_modulo in
Simulation. It reuses Core extract_bits and the existing mt19937_64 iterator.
The function's descending bit traversal, strict less-than masks, extraction
order, partial accumulation and early full-batch replacement match the original.
Both returned tuples retain their final source contents. The immutable entry
allocates disjoint spans and calls the same mutable-span kernel. The original out/buf
size failures have explicit unchanged-source observations. Native sizes are
construction invariants of the immutable result.

Both unchanged original tests pass. The fresh 744-case corpus has 724 successful
streams and 20 original refusals. It covers every modulus 1 through 255, declared
widths 1/2/3/4/7/8/17/63/64/65/125/128/129/256, six seeds, repeated calls and the
original 125-bit benchmark modulus. Every output stripe, final scratch stripe
and next RNG word matches. A 10,000-batch modulus 9 run checks 640,000 samples with
the original min/max frequency bounds. Existing PackedBasisState and its actual
instruction executor consume the generated stripes and recover their complete
input state and exact zero phase after a CX pair.

## Retained source defect and ownership boundary

An isolated original call with out and buf aliased, width 4, modulus 9, seed 0
returns 11 of 64 samples outside the promised range. The header lists size and
full-capacity requirements but omits disjointness. The native shared kernel
`_sample_striped_modulo_into` now preserves this literal numerical defect,
including partial overlap in both directions. It takes an immutable modulus,
caller-owned draw iterator, mutable word allocation and output/scratch offsets.
Each span has the declared width. It zeroes output before descending random
writes, then accumulates ascending output bits in place; an earlier output write
can therefore change a later scratch read. No staging copy hides this ordering.
The existing immutable entry delegates to this kernel with disjoint spans and
retains its signature and immutable snapshots. No new public export is added.

The source full-batch branch uses memcpy. Overlapping nonempty spans in that
branch have undefined C++ behavior, including identical pointers. The private
kernel explicitly refuses this branch rather than inventing source semantics.
The refusal occurs after the same draws and scratch writes, before the copy.
Non-overlapping full-batch copies retain the original result. Span admission is
validated before writes; caller allocation outside the spans is untouched.

K-0315 now has its complete source responsibility implemented. The numerical
alias defect and undefined-copy boundary remain documented under the approved
faithful-migration closure rule. These are not claims of uniform sampling for
overlapping buffers or of equivalence to undefined behavior.

The physical Random-based initialization belongs to the separate complete
simulator source scope. This batch supplies the source scalar draw-driven sampler and its
existing packed-executor consumer evidence. Whole seeded simulator interchange
is still tracked on its own parents.

## Resources and measured workload

Scalar sampling emits zero quantum gates, measurements or work qubits. It owns
two width-sized classical word buffers. Quantum execution resource accounting
stays with the existing physical instructions. The original benchmark goal is
1.6e-6 seconds per 125-bit/64-lane batch. The unchanged source benchmark observed
5.06e-6 seconds locally. Native Python observed 1.09e-3 seconds per batch over 128
stateful iterations. Final output, scratch and next draw match original C++.
The observations and source goal are separate. The
[2026-11-28 review](G2_FUNCTIONAL_MIGRATION_DISPOSITIONS.md) accepts K-0317's
complete independent-buffer workload migration, with performance still unmet.
K-0315's mutable-span implementation is verified below; its original overlap
numerical defect remains a separate follow-up.

Regeneration uses tests/unit/fixtures/generate_striped_modular_sampling.py with
`--source-root`, `--corpus` and `--output`. Seven frozen blobs are checked before building
the original implementation. Its output reproduces the corpus byte-for-byte.

## Batch verification

All 115 concentrated tests and 92 local register integration tests pass without
skips. Strict typing covers the three changed Python files. Black, Ruff, team
scope, architecture, multi-level IR and public API checks pass. The complete
744-record corpus regenerates byte-for-byte from seven verified frozen blobs.
Whole-repository dependency policy retains 720 findings and names zero touched
paths. The previous full default-suite limitation remains recorded in the
integer batch audit. Initial fixture setup attempted a zero-width hex parse
before the sampler refusal check. The native test now checks that parse within
the expected refusal, and the oracle constructs the empty modulus directly to
exercise the original sampler's own validation.


## Shared mutable-span closure (2026-11-28)

The additional generator `generate_striped_sampling_aliases.py` verifies the
same seven frozen blobs, then compiles the original implementation unchanged
with AddressSanitizer and UndefinedBehaviorSanitizer. Its wrapper only supplies
storage and detects a would-be overlapping memcpy before invoking undefined
behavior; it does not replace the sampling implementation. The 419 observations
cover widths 1/2/4/8/17, eight seeds, one and three repeated calls, identical
spans, offsets in both directions, independent spans, and boundary sentinels.
Every full backing allocation and next random word matches native execution.
The historical modulus-9 witness retains exactly 11 invalid lanes. Explicit
native zero-draw cases exercise refusal of all overlapping full-batch forms.

The existing 744 source observations and 640,000-sample distribution check run
through the replacement kernel, as do current spin/lifecycle/workload consumers.
Their independent-buffer output, scratch and draw position remain unchanged.
Classical scratch ownership remains width-sized spans in caller storage; the
immutable convenience path allocates two width-sized spans. Sampling emits zero
quantum primitives, measurements or work qubits and has no quantum cost model.
Previously recorded source timings, native timings and unmet goal stay distinct.

Current batch verification: 187 concentrated sampler/spin/lifecycle/workload
checks and 82 local integration checks pass without skips. Strict typing covers
all three changed Python files. Black, Ruff, Simulation scope (including both
fixture sources), architecture, IR/API, required checks, Cirq interoperability,
English, local links and whitespace pass. Fresh sanitizer-enabled fixture
regeneration reproduces all 419 records byte-for-byte. Existing unchanged-source
744-case and original two-test evidence is reused for the same numerical owner.
