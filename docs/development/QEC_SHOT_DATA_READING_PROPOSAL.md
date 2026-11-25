# Native shot-file reading and bounded batch decoding

Status: approved for 28-path implementation on 2026-11-11 under proposal SHA-256
`dd34a168261cdc11b3e0f4da8dbf75c20f8129b3908b449c98b94f382994e225`.
Implementation remains a local candidate. Native build, installed profiles and
local regression passed; final-head publication and actual-main evidence remain
separate gates. No performance claim is added.

## Verified baseline and actual gap

Base: actual main `504dc460f34f745db67566597cd69a94e345c298`, tree
`f55b8a1f42ac8fab3c10466943729e5556f421f0`. PR #1120 delivers bounded measurement
file conversion. Its actual-main offline installation and public example passed;
main installed-wheel smoke passed. PR CI and GPU passed on their own reviewed
revision. At proposal preparation, actual-main container publication was incomplete.
The prerequisite subsequently passed in run 42850330277 before progress-document
publication; the original observation is retained in the proposal evidence.

Existing compiled samplers write six record formats. convert_file accepts six
inputs and five selected outputs. PyMatchingDecoder.decode_packed_batch already
accepts LSB-first CPU uint8 detector rows with zero unused high bits and returns
packed logical rows. No public shot-file reader or iterator was found in the
current package; binary-record tests concern SyndromeRound/PauliFrame value
validation, not file decoding. This proposal joins file reading to the existing
decoder; it does not implement another decoder or an execution planner.

Upstream Stim 1.16.0 source identity:

- Commit: `e2fc1eca7fd21684d433aa5f10f4504ea4860d07`.
- Cached source archive SHA-256:
  `9092a996429dcf616d8d4ca01fec886b7b310caf9425a39fcfe499005636bd52`.
- `src/stim/io/read_write.pybind.cc` SHA-256:
  `7344f7a7810e866c94ff08eb39f99950571a82a66af1ed603b029c239f64a6e7`.
- The existing 96-file source lock has been rehashed unchanged, without a build.

The upstream binding defaults its dimension arguments to None and requires at
least one explicit dimension. Generated stubs print zeros. Follow the actual
binding and record this discrepancy. Its read_shot_data_file accumulates a
full_buffer before returning NumPy arrays; calling it once for each requested
batch would not establish bounded streaming. Do not implement that fallback.

## Proposed public contract

Canonical namespace: `flagquantum.qec.execution.shot_data`. Preserve all existing
exports, including fq root exports. Two new functions, no new public result or
configuration class. QEC supplies the declared M/D/L dimensions; Simulation owns
the codec and file boundary. Return Torch tensors, never upstream SDK objects.

```python
def read_shot_data_file(
    *,
    path: str | os.PathLike[str],
    format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"],
    num_measurements: int | None = None,
    num_detectors: int | None = None,
    num_observables: int | None = None,
    separate_observables: bool = False,
    bit_packed: bool = False,
    bit_pack: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

def iter_shot_data_file(
    *,
    path: str | os.PathLike[str],
    format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"],
    num_measurements: int | None = None,
    num_detectors: int | None = None,
    num_observables: int | None = None,
    separate_observables: bool = False,
    bit_packed: bool = False,
    batch_size: int = 1024,
) -> Generator[
    torch.Tensor | tuple[torch.Tensor, torch.Tensor], None, None
]: ...
```

The read function preserves upstream's bit_pack compatibility keyword by
combining it with bit_packed using OR. It is not used in beginner examples and
is not duplicated on the new iterator. This is an explicitly scoped upstream
keyword mapping, not a new compatibility layer. All three dimensions are
nonnegative integers or None; bool is not a valid dimension. At least one must
be explicitly supplied. Reject unknown formats and non-bool flags before I/O.

Column order is M then D then L. Without separation, shape is (shots, M+D+L),
bool; packed shape is (shots, ceil((M+D+L)/8)), uint8. With separation, the first
part contains M+D columns and the second L columns. Pack each part separately,
with zero high padding bits; slicing a shared padding byte is incorrect. Do not
assume that num_observables describes unstored circuit observables.

The iterator returns positive-sized batches up to batch_size (1 through 65,536),
except that an empty file yields no batches. Preserve final short batches and
owned buffers. A PTB64 encoded block contains 64 shots; the native cursor may
retain one such block and divide it across arbitrary consumer batch sizes.
Test batch sizes 1, 17, 63, 64, 65, 1024 and 65,536 across block boundaries.
Normal exhaustion and generator.close release the cursor, snapshot and file.
Use contextlib.closing for deterministic cleanup on early break or decode
failure. No public resume/checkpoint, serialization or async lifecycle is added.

Zero stored columns are refused with CapabilityError before codec preparation
in this first package, including nonempty zero-width text. This is a declared
narrower domain than upstream; no full upstream reader parity claim is allowed.
Zero-shot files with positive declared width are supported. The zero-column
domain remains an explicit future row, not an omission from the denominator.

## Source reuse and backend boundary

Use MeasureRecordReader<width>::make and start_and_read_entire_record from the
already locked measure_record_reader.h/.inl and their current dependencies.
The CPython bridge owns a private reader cursor plus explicit checked close.
Do not vendor read_write.pybind.cc or add pybind11/NumPy to the embedded build;
that file is the semantic reference. No new upstream files, translation units,
compiler flags, dependency bounds or setup.py changes are included. Extra source
requirements stop implementation for review, rather than expanding the lock.

Eager reads use the existing embedded-vs-SDK profile selection. Embedded mode
must call the extension without the Stim Python package. Standard SDK mode may
use pinned Stim 1.16.0 only after the full eager allocation estimate passes.
An explicitly selected broken embedded extension must not fall back to the SDK.

The iterator requires the embedded profile. On an SDK-only build it fails
before opening or snapshotting the file with an actionable CapabilityError.
The SDK eager API is not a streaming implementation. Replacement/conformance
must exercise the same eager consumer with native and SDK implementations, plus
a controlled private fake with native iterator consumers unchanged. State the
SDK-only iterator limitation in the compatibility inventory.

Reuse the regular-file/snapshot boundary from #1120. Move _snapshot/_identity
helpers only as necessary into the current private file-validation owner and
keep existing call sites and tests compatible; do not add a generic file manager
or a duplicate snapshot contract. Preserve existing stricter #1120 validators.
Reader-specific rules must follow the pinned reader's positive-width domain;
for example, do not silently tighten all six input formats to the conversion
writer's canonical spelling. Test typed M/D/L dets input and noncanonical valid
upstream input explicitly.

## Memory, disk and refusal contract

- Snapshot input cap: 1 GiB, unchanged. Copy using 64 KiB chunks. Regular local
  files only; reject endpoint symlinks, devices, pipes and URL-style inputs.
  Check identity around snapshot copying and retain original input on failure.
- Full framing/count validation occurs on the stable snapshot before native
  preparation/first yielded batch. A malformed tail cannot cause some earlier
  decode counts to be presented as a completed dataset. I/O errors retain their
  original cause. No destination file is published by a read operation.
- Principal payload guard: 64 MiB for each operation. Let C=M+D+L, P=ceil(C/8),
  S=ceil(C/1024)*128, and Q=C for dense output. For packed output Q is P when
  combined, or ceil((M+D)/8)+ceil(L/8) when separated. Use the conservative bound
  `4 * rows * (P + Q) + 128 * S + 65536 <= 64 * 1024**2` before allocation.
  Eager rows is the prevalidated complete shot count; iterator rows is the
  smaller of batch_size and remaining shots. Check native integer overflow
  before multiplication. The bound covers packed staging, result buffers and
  copies, plus conservative 64-shot codec buffering and file-copy scratch.
- Freeze the estimator in contract tests with integer arithmetic and allocation
  spies before the first native build. Native allocation review must confirm it
  dominates actual payload arrays; if it does not, stop and present the precise
  amendment before executing a build or claiming a memory bound.
- The guard is not total RSS and excludes Torch/SDK import costs, filesystem
  cache, decoder graph/workspace and any tensors retained by callers. The library
  retains no prior yielded batches. Never accumulate the iterator into a list
  inside implementation or describe a small tensor maximum as a process limit.
- Reject nonzero input b8 padding as an explicit FlagQuantum boundary and test it
  separately from upstream conformance where upstream ignores those high bits.
  Output padding is always zero. Truncation and partial PTB64 blocks fail.

## User workflow

The accompanying examples/qec/shot_file_decoding.py is a proposed API example, not a run
result. It requires an embedded FlagQuantum installation and the existing
pymatching extra. It uses 64 fixed combined D0/L0 b8 records, reads eager and in
17-shot batches, and calls the existing packed decoder. Expected output:

```text
shots 64 batches 4 logical_failures 0
```

The deterministic one-edge model has unique predictions. This proves row/order,
bit packing and decoder composition, not a sampled logical-error estimate,
threshold or performance result. The already accepted #1120 public workflow is
`python examples/qec/measurement_file_conversion.py`; its preserved actual-main
output is shots 64, detectors 10 / observables 1, file_bytes 64 64. It was not
rerun to prepare this proposal.

## Implementation and acceptance sequence

1. Approve names, signature/default/domain mapping, exact paths and memory guard.
   Land the candidate contract/fake/conformance cases before implementation in
   the same reviewed capability package; do not regenerate old API snapshots.
2. Complete shared private snapshot reuse and native cursor, then the thin QEC
   functions. Keep the existing sampler, convert_file, decoder APIs and frozen
   upstream bytes unchanged. Register the new functions in real doctests.
3. Run focused fake/contract tests first. Independently generate exact records
   for six formats, combine/separate M/D/L columns and 7+2/8+1 boundaries. Compare
   eager and all iterator batchings, plus pinned SDK eager reference where its
   domain overlaps. Test input mutation, malformed late records, iterator close,
   decoder exceptions, allocation refusal and missing/broken dependencies.
4. Validate Ruff, Black and strict Mypy without ignore or weaker rules, then the
   seeded smoke/unit and integration tiers for the changed candidate. No passed
   old-head CI/GPU is reexecuted for this work.
5. Build once offline from the reviewed candidate using the current locked wheel
   closure; verify package/wheel/sdist identity and an installed embedded profile
   without the Stim Python package. Qualify existing core/Stim-only/PyMatching-
   only/both wheel profiles and sdist-both through the established gate, without
   modifying dependency or runner configuration.
6. Run the example on final head in its required installed profile and attach
   actual stdout/native-call evidence to a direct-main PR. Never present this
   proposed example as already executed. Follow only new-head automatic CI/GPU
   under the applicable delivery authorization; no recovery of old failed runs.
7. Merge only after applicable checks/reviews pass at the reviewed head and
   perform actual-main installed acceptance. Then update both planning ledgers.

Planned commands (not executed for this proposal):

```bash
python tools/check_team_scope.py --team integration --files <the 28 listed paths>
python -m pytest tests/qec/test_shot_data_reading.py -q
python -m pytest tests/team/simulation/test_shot_data_reading_conformance.py -q
python -m ruff check <changed Python paths>
python -m black --check <changed Python paths>
python -m mypy --strict <changed Python paths>
python tools/ci_tier.py pr-default
python tools/ci_tier.py pr-runtime
FQ_BUILD_EMBEDDED_STIM=1 MAX_JOBS=2 python -m build --sdist --wheel --no-isolation
python tools/verify_embedded_stabilizer_install.py
python examples/qec/shot_file_decoding.py
```

Before execution, resolve commands to verified existing interpreter/tool paths
and pin version/import provenance. Proposed one-attempt budgets: focused tests
900 s, conformance 900 s, each static tool 300 s, smoke/unit 2400 s, integration
3600 s, offline build 1200 s, each offline installed profile 300 s, installed
verifier 120 s and final-head example 120 s. Missing resources stop before the
affected command; no dependency installation/download is authorized here.
Any new failure/timeout preserves raw logs and stops; no automatic retry.
No Linux admission, research measurement, benchmark campaign or CI configuration
change is part of this package.

## Exact path scope

1. `flagquantum/qec/execution/shot_data.py`
2. `flagquantum/simulation/stabilizer/_shot_data.py`
3. `flagquantum/simulation/stabilizer/_measurement_file_validation.py`
4. `flagquantum/simulation/stabilizer/_record_conversion.py`
5. `flagquantum/simulation/stabilizer/csrc/embedded_sampler.cpp`
6. `flagquantum/simulation/stabilizer/csrc/STIM_PROVENANCE.md`
7. `flagquantum/qec/execution/README.md`
8. `flagquantum/simulation/stabilizer/README.md`
9. `contracts/qec-shot-data-reading-v1-candidate.json`
10. `docs/development/QEC_SHOT_DATA_READING_PROPOSAL.md`
11. `docs/reference/API.md`
12. `docs/reference/KNOWN_LIMITATIONS.md`
13. `docs/reference/RELEASE_NOTES.md`
14. `docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md`
15. `docs/roadmap/FTQC_SIMULATION_MASTER_PLAN.md`
16. `docs/guides/STIM_USER_MIGRATION.md`
17. `tests/qec/test_shot_data_reading.py`
18. `tests/team/simulation/test_shot_data_reading_conformance.py`
19. `tests/api_contract/test_public_docstring_examples.py`
20. `tests/unit/test_qec_installed_profiles.py`
21. `tools/qec_installed_smoke.py`
22. `tools/verify_embedded_stabilizer_install.py`
23. `examples/qec/shot_file_decoding.py`
24. `examples/README.md`
25. `capability-maturity.toml`
26. `contracts/cudaq-parity-matrix.toml`
27. `docs/generated/CAPABILITIES.md`
28. `docs/reference/CUDAQ_PARITY_MATRIX.md`

## Progress accounting and remaining gaps

This package would extend FTS-04/R03 record conformance and the FTS-07/R08
bounded-ingestion composition slice. Those requirements remain partial until
broader gates are met. FTS-02/R04 and M0-M5 do not advance on this evidence alone.
Full Stim/PyMatching parity remains open: arbitrary-data write_shot_data_file,
zero-column records, SDK-only streaming, other upstream APIs/domains and their
qualification remain explicit gaps. iter_shot_data_file is a FlagQuantum
extension and must not be counted as a newly covered upstream declaration.
No hardware, large-scale capacity, logical-error suppression, GPU decoding or
mixed non-Clifford capability is inferred.

## Local implementation qualification — 2026-11-11

The approved 28 paths and 96 unchanged upstream files remain the scope.
Ruff, Black and strict Mypy passed for 11 Python files. Focused source regression
passed 356 tests (10 skipped). Embedded installation without the Stim Python
package passed 52 tests (one SDK-reference skip) and the standalone verifier.
All five wheel/sdist installation profiles passed. Broad smoke/unit passed
19,382 tests (378 skipped) in 1,015.39 seconds; integration passed 6,910 tests
(310 skipped) in 2,229.12 seconds. No failed test or timeout occurred in these
approved broad runs. The owned temporary extension was removed by inode/hash.
Final Git/artifact binding, final-head user code and PR publication are the
remaining local publication gates; actual-main acceptance is not yet claimed.
