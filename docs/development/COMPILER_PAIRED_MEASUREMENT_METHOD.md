# Balanced compiler implementation controls

This declaration corrects an asymmetry in local temporary-function controls. It
changes no compiler implementation, SDKBench fixture, adapter, acceptance rule,
resource budget or formal study protocol. It is Compiler-owned
`single_device_fast_path` engineering work, not a publication qualification.

## Why the earlier timing observations need qualification

Several local paired drivers scan loaded FlagQuantum modules and install scoped
function patches for one variant while directly calling the other variant.
Their measured timer starts after that setup, but allocation, garbage collection,
interpreter caches and lazy imports can still affect subsequent work. This is an
unbalanced observer configuration. Its presence does not prove the cause or size
of any timing difference. Preserve the original samples and finite semantic
acceptances; do not use the earlier timings alone to establish a causal speedup,
a slowdown, or a production admission decision.

This qualification applies to the local numerical-composition controls in
PR #1290, the private overhead proposals archived in PR #1291, and earlier local
one-sided patch experiments. It does not alter independently replayed SDKBench
campaign records or retroactively qualify any result. A full-source CI pass is
also not a performance qualification.

## Fixed confirmation before execution

The first balanced confirmation uses the deterministic KAK product helper from
FlagQuantum integration `dce3948b946906875a759362e8570b22a8214d5f` and the
unchanged clean SDKBench revision
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Both variants execute in that
same FlagQuantum checkout. The baseline uses the original `_matmul`; the proposal
specializes inner dimensions two and four, preserving the original complex-zero
sum start, product order, addition order and generic fallback. The complete
proposed source is retained, rather than inferred from a later working tree.
No LAPACK, BLAS, dependency, tolerance, routing or synthesis policy changes.

The already inspected nonce remains `01` repeated 32 times. Widths 6/8/12, five
application families and four topologies run through both ordinary restored and
target-artifact workflows: 120 pairs and 240 independent acceptances. Every pair
must retain identical normalized operation/layout hashes, evaluator metrics and
adapter audits, including identity endpoint placement. Width six uses complete
operators up to phase; larger widths retain three-probe scope. This is an
inspected regression control, not new prospective evidence.

Both variants install a fresh function definition parsed from their respective
source into the same cached list of existing module attributes. Clone the
original function as well as the proposal so neither variant alone is the
unmodified function object. Cache patch locations once before observations;
never scan module globals for just one timed variant. Use equal scoped patch
setup and teardown, and verify restoration. This balances the known observer
asymmetry; it does not eliminate every possible shared-host effect.

After paired acceptance, measure complete target-artifact adapter execution on
six-wire QFT/grid, six-wire repetition/ring and twelve-wire QFT/grid. Keep two
warmups and nine retained samples per variant, alternating order, and report all
samples and medians. Do not redraw cases, tune candidates, repeat a timing run to
obtain a favorable result, or replace the first completed confirmation. Retain
failures. Separate Python traced allocation observations cover setup, compilation
and independent evaluation; they are not process RSS or compiler-only peaks.

Use Python 3.12.14 and one numerical thread, with no overlapping local test suite.
Use a separate 600-second process-group supervisor with TERM/KILL cleanup; leave
SDKBench's stage timer intact. Shared macOS measurements cannot establish a
universal improvement, statistical significance, SDK ranking, accelerator
behavior or scalability. A faster helper alone is insufficient to establish an
end-to-end improvement. Inconsistent workflow observations leave adoption
unproven. Formal host-qualified benchmarking remains user-deferred; no server,
service, driver or kernel change or reboot is permitted.

## Existing user workflow

The supported compiler workflow remains unchanged:

```python
import flagquantum as fq
from flagquantum.compiler import optimize

source = fq.Circuit(3).h(0).h(1).h(2).to_ir()
assert optimize(source).instructions == source.instructions
```

Expected: independent gates remain unchanged. This declaration introduces no
production optimization or public contract. Reproduction requires the exact
recorded source, driver and environment; temporary paths must be adjusted in a
separate retained reproduction, without rewriting the original archive.

## Completed first confirmation

The declaration above was published at
`b864121b459ef2aa37cde8320c59d23228996348` before the balanced execution.
[The exact archive](evidence/compiler-balanced-synthesis-products.json) retains
both the original asymmetric control and the first completed balanced control,
with their reconstruction hashes. Proposed helper source, baseline source and
SDKBench pins match across the two controls. Source and case selection were not
changed. Do not attribute differences between the separate observations solely
to patch setup: other shared-host conditions were not controlled.

Both workflows complete all 120 pairs and 240 independent acceptances. Within
and across the controls, every case retains identical normalized operation/layout
hashes, evaluator metrics and adapter audits. Both variants use one cached
module-attribute list, parsed function definitions and equal patch lifetimes;
restoration is checked after each observation. No local test suite or artifact
transfer overlapped the balanced measurements. Garbage collection policy was
unchanged. The supervisor completed once with status zero.

| Inspected case | Balanced original / proposed median | Whole-observation Python traced peak, original / proposed |
| --- | ---: | ---: |
| 6-wire qft, grid | 96.141 / 96.301 ms | 35,366,014 / 35,364,182 bytes |
| 6-wire repetition, ring | 121.818 / 122.288 ms | 35,364,353 / 35,363,788 bytes |
| 12-wire qft, grid | 605.918 / 604.833 ms | 35,363,187 / 35,364,909 bytes |

These observations do not demonstrate a consistent end-to-end gain. Keep the
production product helper unchanged. The earlier asymmetric six-wire QFT timing
(91.901 / 105.424 ms) is retained; it must not be presented as proof that this
helper causes a 15% workflow slowdown. Neither confirmation qualifies performance
or closes native-CZ selector resource admission. It is also not a balanced
confirmation of the two separate proposals in PR #1291; their adoption remains
unproven. Future implementation comparisons must balance their observers and
retain their own first completed result.

The new declaration branch is based on integrated main
`006c3268e803bd8dc13258b913cccfa9b69bb718`. Its seeded sparse default and
runtime tiers pass 236 unit and 42 integration tests. The first default attempt
passed 235 tests and failed because the unchanged local-GPU workflow fixture was
omitted from sparse checkout; restoring that tracked input resolves the setup
failure. No test, workflow assertion or resource limit was changed. These sparse
checks do not certify the complete repository or qualify performance.
