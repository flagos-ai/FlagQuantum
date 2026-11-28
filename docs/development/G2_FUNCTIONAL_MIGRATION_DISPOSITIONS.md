# Reviewed G2 functional migration dispositions

The user approved the [G2 closure rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md)
on 2026-11-27. This review applies that existing rule to complete source files;
it does not change the 1,280-file inventory or waive missing implementations.
The previous ledger retained delivered files solely for source defects or unmet
timing goals. Migration acceptance and those follow-ups now have separate status.

## Complete files accepted on 2026-11-28

| Parent | Complete responsibility and reviewed evidence | Retained follow-up and owner |
|---|---|---|
| K-0248, array_perf.py | The sole registered workload and script entry are owned by Benchmarking. All 300 input/output elements match at one and three repetitions; timing-suite registration, reporting and discovery dependencies were already accepted. See [Python workloads](PYTHON_BENCHMARK_WORKFLOWS_AUDIT.md). | Benchmarking performance remains unmet: original 2.8 us goal versus recorded native 144.53 us. Native group naming is an explicit product adaptation. |
| K-0250, circuit_builder_perf.py | All five registered scalar/array/mixed CCX and 256-bit addition workloads use real Compiler construction and Simulation execution. Full KMX, phase, counts and all 254 clean adder wires match the original observations. Script and timing dependencies are mapped. See [Python workloads](PYTHON_BENCHMARK_WORKFLOWS_AUDIT.md). | Benchmarking performance remains unmet for all five goals. No cached gate stream replaces the measured construction path; no throughput equivalence is claimed. |
| K-0271, sim.perf.cc | Core transpose and all three stateful simulator workloads have authoritative owners. Four portable widths retain complete state, buffers, exact phases and lane counts over repeated calls. Original ARM-host and Rosetta/SSE benchmark observations remain separate. See [basis lifecycle](PHYSICAL_BASIS_LIFECYCLE_AUDIT.md). | Benchmarking keeps the original 43 ns–4.8 us sampling goals and slower native measurements. ISA selection/execution remains on the separate open SIMD parents; this decision supplies no acceleration or cross-ISA certificate. |
| K-0298, fixed_width_int.cc | Every definition group has an existing Core/Simulation owner, with 74,043 exact source observations and all original integer tests. Defined carry, hanging-word and sequential alias defects remain literal comparison targets. See [fixed-width integers](FIXED_WIDTH_INTEGER_AUDIT.md). | Core retains ASan heap-overflow witnesses for zero-width scalar assignment/carry and a bounded nontermination witness for self-alias mutating remainder. Native refuses those unsafe/nonreturning cases. It does not claim to reproduce undefined memory accesses. |
| K-0300, fixed_width_int.perf.cc | All four stateful workloads and complete values after 128 iterations match the frozen original. See [fixed-width integers](FIXED_WIDTH_INTEGER_AUDIT.md). | Benchmarking retains all four unmet original goals and measured native timings. Functional migration acceptance is not performance acceptance. |
| K-0312, mod_int.cc | All constructors, arithmetic, shifts, division, inverse, sampling and formatting routes have Core/Simulation owners. The 12,442-record source corpus retains successful values, comparisons and failure snapshots. See [modular integers](MODULAR_INTEGER_AUDIT.md). | Core retains UBSan signed-minimum multiplication overflow. Native refuses it. Immutable native failure snapshots and recorded mutable source post-failure state remain an explicit representation adaptation. No mathematically corrected source result is substituted. |
| K-0317, mod_rand.perf.cc | The complete 125-bit, 64-lane benchmark uses independent output/scratch buffers. Output, scratch and next RNG state match after 128 calls. Its scalar dependency is verified on that complete called domain. See [striped sampling](STRIPED_MODULAR_SAMPLING_AUDIT.md). | Benchmarking retains the unmet 1.6 us goal, source observation 5.06 us and native observation 1.09 ms. The separate K-0315 writable-alias contract is not accepted by this decision. |
| K-0324, main.perf.cc | All selection, argument, adaptive timing, reporting and actual physical-workload routes are mapped. The 414-observation corpus verifies defined command and clock behavior. See [benchmark support](BENCHMARK_SUPPORT_AUDIT.md). | Benchmarking retains the unreachable empty-result diagnostic and UBSan NaN-to-int goal conversion. Native rejects the undefined conversion; defined no-goal reports and empty-result behavior remain unchanged. |
| K-0330, cmul16_mod_21.kmx | The exact original asset, full canonical text and KMB bytes are retained. All 64 operator columns, a complex superposition and normal append execution match its actual source permutation. See [circuit assets](SMALL_CIRCUIT_ASSETS_AUDIT.md). | Verification retains the filename discrepancy: the actual valid-residue multiplier is four, not sixteen. Do not describe this literal asset as multiplication by sixteen or silently replace its bytes. |

## Current-state verification and resources

The review checked 100 frozen source/dependency Git blobs against the immutable
inventory and unchanged reference files. Existing complete audits, original
oracles and source sanitizer/timeout receipts were inspected and reused.
At the current implementation, 34 complete lifecycle/Python-workload tests,
five asset storage/operator/count tests and 69 scalar/benchmark-support tests
pass: 108 checks, with no numerical production code changed for this review.
The current local register integration and strict/API/architecture evidence
from the preceding implementation batch remains applicable.

Classical scalar, transpose and orchestration helpers emit no quantum gates,
measurements or work bits. Circuit workloads retain their actual source/bound
inventories in the linked audits. The 256-bit adder's emitted records and executed
lane counts are distinct; its workspace is 254 clean wires. The literal K-0330
asset emits 8 X, 21 CX and 41 CCX, with no measurements or work wires. No
independent quantum model is invented for these assets or classical helpers.
Original targets, source timing observations and native timings remain separate.

Reproduction at the repository root:

```bash
python -m pytest tests/unit/test_physical_basis_lifecycle.py \
  tests/unit/test_physical_basis_workloads.py \
  tests/unit/test_python_benchmark_workloads.py -q
python -m pytest tests/unit/test_small_circuit_assets.py \
  -k 'unmeasured or multiplier_name or storage or records_and_bound' -q
python -m pytest tests/unit/test_packed_chacha.py \
  tests/unit/test_fixed_width_integer.py tests/unit/test_modular_integer.py \
  tests/unit/test_striped_modular_sampling.py \
  tests/unit/test_benchmark_support.py -q
```

Use the existing locked environment and mandatory reference policy. Source,
current-state test and unchanged implementation receipts are stored beside the
workspace ledgers under `g2-closure-review-*`.

## Files deliberately kept open

This rule is applied individually, not as a blanket source-defect waiver.

* K-0291 could not close solely from portable shift UB. The unchanged SSE source
  executes rol32<0> and rol32<32> successfully under UBSan, returning the input
  word in both cases. The native scalar helper currently refuses both. The
  review's x86_64/Rosetta probe returned `12345678 12345678` without sanitizer
  errors. The 3,064-state stream corpus alone did not close that missing path.
  The subsequent [complete rotation implementation](PACKED_RANDOM_STREAM_AUDIT.md#defined-sse-rotation-closure)
  implements all defined count families and adds unchanged-source SSE vectors;
  its whole-header acceptance is recorded separately from these nine decisions.
* K-0315 retains the source's successful but invalid overlapping-buffer samples.
  The native independently owned tuple interface does not expose that writable
  alias. The correct independent-buffer path cannot by itself certify literal
  alias behavior; its complete functional/contract disposition remains open.
* K-0284 still rejects an fopen mode that succeeds on the current source host.
  Missing Windows certification does not excuse that native host functionality.
* SIMD implementation/selection gaps and the source-success nested ECC control
  contexts remain open. Neither an unavailable platform nor a reproduced
  unrelated source defect turns a missing native implementation into acceptance.

After publication, the workspace parent ledger, stage overview and PR evidence
must record these nine decisions together. All other parent decisions remain
unchanged. G2 continues; this review neither merges the PR nor starts G3.


### Subsequent implementation: K-0315 mutable sampling spans

The former genuine gap is closed by the shared mutable-span sampling kernel.
Both exact and partial output/scratch overlap preserve literal source writes,
including the original 11-invalid-lane witness. The existing immutable sampler
uses that same kernel. All 419 additional original-source backing-allocation and
next-draw observations match. Overlapping full-batch memcpy remains undefined
in C++ and is refused explicitly; this boundary is not a missing defined path.
See [the complete sampler audit](STRIPED_MODULAR_SAMPLING_AUDIT.md).


### Subsequent implementation: K-0284 host C file modes

The POSIX source-success extended-mode gap is implemented through host fopen
and an owned descriptor transfer. The complete 2,756-observation source corpus
retains actual mode admission, file effects, defaults and diagnostics. Windows
text translation and other unmeasured runtime certificates remain explicit
platform follow-ups under the approved closure rule; no Windows parity is
claimed. See [the command audit](COMMAND_ARGUMENTS_AUDIT.md).
