# Arbitrary shot-data file writing capability proposal

Status: proposed only; no repository implementation or API change authorized by this document.
Owner: Integration for contract admission; Simulation owns codecs, QEC owns the typed user entry.
Classification: `single_device_fast_path`. No simulation, sampling, research or scalability claim.
Date: 2026-11-11.

## User outcome and naming

A user already has CPU measurement records, detector events or logical flips.
They can save those records in the six familiar Stim formats, read them back
through the accepted #1126 entry, and reuse the existing packed decoder without
inventing a sampler or writing Stim text. The operation does not execute a circuit.
The exact name is `write_shot_data_file`, symmetric with the existing reader and
independent of the underlying codec implementation. No `stim` branding or new
root `fq.*` export is introduced.

Proposed canonical public entry, appended after the existing two module exports:

```python
def write_shot_data_file(
    *,
    data: torch.Tensor,
    path: str | os.PathLike[str],
    format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"],
    num_measurements: int | None = None,
    num_detectors: int | None = None,
    num_observables: int | None = None,
) -> None: ...
# flagquantum.qec.execution.shot_data
```

This is one additive namespace API. Existing reader/iterator exports, signatures,
packing aliases, defaults and exceptions remain intact. Approval of a new
candidate contract is required before implementation; old snapshots are not
regenerated. The naming review is explicit: writing means persisting supplied
records, whereas `sample_write` executes an already prepared sampler.

## Executable user journey (proposed; not run)

Use an embedded FlagQuantum build with the existing PyMatching extra. The Stim
Python package is absent in the required installed acceptance profile.

```python
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder
from flagquantum.qec.execution.shot_data import (
    iter_shot_data_file,
    read_shot_data_file,
    write_shot_data_file,
)

model = DetectorErrorModel(1, 1, (DemError(0.1, (0,), (0,)),))
decoder = PyMatchingDecoder.from_detector_error_model(model)
# A single combined byte packs D0 at bit 0 and L0 at bit 1.
data = torch.tensor([[0], [3]] * 32, dtype=torch.uint8)
with TemporaryDirectory() as directory:
    path = Path(directory) / "events.b8"
    write_shot_data_file(data=data, path=path, format="b8",
                         num_detectors=1, num_observables=1)
    assert path.read_bytes() == b"\x00\x03" * 32
    dets, logicals = read_shot_data_file(
        path=path, format="b8", num_detectors=1, num_observables=1,
        separate_observables=True, bit_packed=True)
    assert torch.equal(dets, logicals)
    shots = batches = failures = 0
    with closing(iter_shot_data_file(
        path=path, format="b8", num_detectors=1, num_observables=1,
        separate_observables=True, bit_packed=True, batch_size=17,
    )) as records:
        for dets, logicals in records:
            prediction = decoder.decode_packed_batch(dets)
            failures += int((prediction != logicals).any(dim=1).sum())
            shots += len(dets)
            batches += 1
    assert (shots, batches, failures) == (64, 4, 0)
    print("shots", shots, "batches", batches, "logical_failures", failures)
```

Expected stdout: `shots 64 batches 4 logical_failures 0`. The writer's input is a
combined bit stream; independently packed `(dets, observables)` reader outputs
must not simply be concatenated when the detector width is not byte-aligned.
The first version deliberately has no tuple-of-sections input or batching API.
A 7-detector + 2-logical fixture verifies the boundary explicitly.

## Authoritative upstream identity and actual semantics

Pinned Stim 1.16.0, commit `e2fc1eca7fd21684d433aa5f10f4504ea4860d07`.
Existing sdist SHA-256:
`9092a996429dcf616d8d4ca01fec886b7b310caf9425a39fcfe499005636bd52`.
Inspected `src/stim/io/read_write.pybind.cc` SHA-256:
`7344f7a7810e866c94ff08eb39f99950571a82a66af1ed603b029c239f64a6e7`.
The archived C++ binding uses `None` defaults despite the generated `.pyi`
describing zero defaults. Use the actual source binding and the existing reader's
optional-dimension convention, and test the distinction.

- At least one dimension is supplied; unspecified means zero. Dimensions are
  nonnegative plain Python integers, not bool. Total columns C must be positive.
- The actual upstream writer refuses nonzero M together with D or L. Preserve
  that rejection before allocation/backend preparation. Support M-only, D-only,
  L-only, and D+L. The reader's broader mixed M/D/L support is unchanged.
- `torch.bool` is unpacked `(S, C)`; `torch.uint8` is little-endian packed
  `(S, ceil(C/8))`. Dtype determines packing, as upstream does. No extra flag.
- CPU dense strided tensors only. Noncontiguous dense input is copied into one
  owned bounded buffer. No GPU transfer, float coercion, sparse/quantized/meta
  tensor, arbitrary NumPy object or autodiff semantics are implied.
- Empty shot tables with positive C produce empty files. Zero columns remain an
  explicit FlagQuantum refusal, consistent with the existing reader.
- Nonzero unused high bits in packed input refuse rather than silently truncate.
  This strict canonical-input boundary is documented, not claimed upstream parity.
- `ptb64` requires S divisible by 64 (including S=0). Reject before opening a file.
- Existing destinations, including symlinks, refuse; no overwrite switch. The
  upstream SDK's overwrite behavior is intentionally not exported.

## Reuse boundary and implementation sequence

Keep all 96 vendor files, the source lock, legal bytes, 37 translation units,
compiler flags and dependency declarations unchanged. The inspected pybind file
is reference evidence only; do not vendor pybind11 or copy its NumPy adapter.

1. Add the approved contract, an injectable fake and consumer conformance first.
   The fake verifies actual data/section values and failure cleanup, not just
   method names. Preserve all earlier frozen candidate contracts.
2. QEC's public function delegates to one Simulation-owned private writer module.
   Reuse dimension checks, conservative output-size calculation and completed-file
   validation from the existing authoritative owners; parameterize only necessary
   private helpers, retaining their reader/sampler behavior and old tests.
3. Before copying user data or preparing an engine, validate shape/dtype/device,
   dimensions, format, native integer products and memory/output budgets. Create
   exactly one bounded owned byte buffer and a `torch.frombuffer` view; `copy_`
   fills it without `.tolist()`, dense unpacking of packed input, ctypes/data_ptr
   exposure, a full tensor clone plus byte serialization, or a NumPy dependency
   on the embedded route. Validate the copied buffer, not a subsequently mutable
   caller tensor. Concurrent caller writes during the copy are unsupported.
4. Existing embedded-profile selection stays authoritative. A present but broken
   extension fails with its cause and cannot fall back to the SDK. Without the
   profile, the eager SDK fallback requires precisely Stim 1.16.0. The optional
   SDK adapter may use a zero-copy NumPy view over the owned buffer, because NumPy
   belongs to Stim's existing optional closure; do not add a production dependency.
5. Add one private CPython bridge accepting an owned contiguous buffer plus
   checked S/C/section metadata. Recheck lengths, padding, dimensions and budget
   before allocation or fopen. Use existing `write_table_data` in blocks of at
   most 64 shots: its non-PTB64 path calls `MeasureRecordWriter`, and its PTB64
   branch uses the existing transposed-table writer. Both are already linked.
   `MeasureRecordWriter::make(PTB64)` explicitly refuses, so it cannot implement
   all six formats alone. Pass section transitions as M/M or D/L exactly as the
   upstream binding does. No codec is rewritten in Python or new C++ encoder.
6. Write only a same-filesystem temporary destination, check flush/close and
   structural validity, and publish by a no-replace hard link. Free buffer/table,
   release Py_buffer and remove the owned temporary directory on every failure.
   The complete destination is visible only after success; parent disk space and
   output bounds are checked. Staging inherits the existing portable temp-file
   rules; no runner/filesystem policy changes are authorized.

## Budgets and first-version limits

Let S be shots, C total columns, and Q be C for bool input or ceil(C/8) for uint8.
Let A(x)=1024*ceil(x/1024), with A(0)=0. Propose a conservative preflight:

`B = S*Q + 2*A(C)*A(max(S,64))/8 + 2*A(C)/8 + 65536`.

B must be <= 64 MiB, all intermediate products and native indices must fit the
native integer range. The estimator covers the owned input copy, a padded table
and transposed table, two padded vectors and a 64 KiB buffer allowance, using a
maximum supported SIMD width of 1024. Assert that width bound in the bridge and
verify actual upstream allocations before accepting the estimator. The SDK
path can allocate tables spanning all S, so the shared admission deliberately
uses its larger bound even though the embedded writer reuses a 64-shot block.
Zero-shot output still obeys the column/workspace cap.

This is a principal-array payload cap, not a process-RSS guarantee. Caller-owned
input storage, allocator overhead, library imports, Python objects and retained
prior batches are not covered. Document that fact in API/limitations/example.
No full unpacking of packed input, all-output accumulation or Python per-bit
list is permitted. If source audit finds an unaccounted principal allocation,
stop for a revised budget proposal before building.

The conservative encoded-file upper bound remains <= 1 GiB, calculated by the
existing format-specific bound; reject insufficient parent free space before
staging. The writer is one synchronous bounded table call. Streaming chunks,
append/resume, zero-column files, device tensors, split-section inputs and
arbitrarily large tensors are later proposals, not hidden partial behavior.
No count/time budgets from the unrelated W2/W3/W6 research launch are consumed.

## Acceptance, all future commands (not executed)

Use verified existing tool/interpreter paths and offline dependency caches.
No installs or downloads. Preserve exact command, environment/import provenance,
exit code, stdout/stderr, timeout cleanup and final-head identities. First error
or timeout stops; no automatic retry, check recovery or threshold weakening.

1. Fake/contract cases: both dtypes; typed section rules; all six formats; stable
   copy after admission; shape/device/padding refusal before native preparation;
   exact 64 MiB and 1 GiB boundaries using counting fakes; overflow/zero rows;
   missing/wrong-version SDK; broken embedded refusal; target/symlink races,
   incomplete/erroring writer, flush/close/validation/publication cleanup.
2. Independent bytes: enumerate small bit tables in test code independently of
   both readers and writers; fixed all-zero, all-one and alternating fixtures;
   7+2 and 8+1 D/L boundaries; M-only and L-only DETS labels; cross-byte R8 runs
   at 254/255/256; 0/1/63/64/65/128 shots and PTB64 rejection as applicable.
   Compare each supported output byte-for-byte with pinned SDK output, not only
   a shared-implementation round trip. Then read through the existing reader and
   decode with the unchanged packed decoder consumer.
3. Replace fake/native/SDK behind the writer boundary with the same public
   consumer; all file bytes and error/publication semantics must conform. Core
   tests skip only real codec cases on genuine absence, retaining all fake and
   dependency-refusal cases. SDK-only output is tested independently.
4. Offline build once in macOS arm64, then five existing wheel/sdist profiles.
   Without Stim Python, spy on the real extension call while it performs encoding;
   verify all six codecs, data values and native cleanup, not only an import.
   No extension or new upstream file may be added to fix a build without review.
5. Run Ruff/Black/strict Mypy, named doctest entry, focused tests and required
   smoke/unit + integration. Preserve 95% stabilizer coverage policy and other
   existing floors. Add meaningful missing-branch tests rather than policy changes.
6. Verify source/wheel/sdist/install bytes and the 96-file lock; execute the
   documented example at final head. Only then commit/push and create a direct-main
   PR with actual stdout and hash-bound evidence under explicit implementation/
   publication approval. Follow only that new head's automatic CI/GPU. Merging
   the future feature PR requires explicit authorization; this proposal does not
   preauthorize it or recover #1127.

```bash
python tools/check_team_scope.py --team integration --files \
    contracts/qec-shot-data-writing-v1-candidate.json \
    docs/development/QEC_SHOT_DATA_WRITING_PROPOSAL.md \
    flagquantum/qec/execution/shot_data.py \
    flagquantum/simulation/stabilizer/_shot_data_write.py \
    flagquantum/simulation/stabilizer/_measurement_file_validation.py \
    flagquantum/simulation/stabilizer/csrc/embedded_sampler.cpp \
    flagquantum/simulation/stabilizer/csrc/STIM_PROVENANCE.md \
    tests/qec/test_shot_data_writing.py \
    tests/team/simulation/test_shot_data_writing_conformance.py \
    tests/api_contract/test_public_docstring_examples.py \
    tests/unit/test_qec_installed_profiles.py \
    tools/qec_installed_smoke.py \
    tools/verify_embedded_stabilizer_install.py \
    examples/qec/shot_file_encoding.py \
    examples/README.md \
    docs/reference/API.md \
    docs/reference/RELEASE_NOTES.md \
    docs/reference/KNOWN_LIMITATIONS.md \
    docs/guides/STIM_USER_MIGRATION.md \
    flagquantum/qec/execution/README.md \
    flagquantum/simulation/stabilizer/README.md \
    capability-maturity.toml \
    contracts/cudaq-parity-matrix.toml \
    docs/generated/CAPABILITIES.md \
    docs/reference/CUDAQ_PARITY_MATRIX.md \
    docs/roadmap/FTQC_SIMULATION_MASTER_PLAN.md \
    docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md
python -m pytest tests/qec/test_shot_data_writing.py -q -x
python -m pytest tests/team/simulation/test_shot_data_writing_conformance.py -q -x
python -m ruff check \
    flagquantum/qec/execution/shot_data.py \
    flagquantum/simulation/stabilizer/_shot_data_write.py \
    flagquantum/simulation/stabilizer/_measurement_file_validation.py \
    tests/qec/test_shot_data_writing.py \
    tests/team/simulation/test_shot_data_writing_conformance.py \
    tests/api_contract/test_public_docstring_examples.py \
    tests/unit/test_qec_installed_profiles.py \
    tools/qec_installed_smoke.py \
    tools/verify_embedded_stabilizer_install.py \
    examples/qec/shot_file_encoding.py
python -m black --check \
    flagquantum/qec/execution/shot_data.py \
    flagquantum/simulation/stabilizer/_shot_data_write.py \
    flagquantum/simulation/stabilizer/_measurement_file_validation.py \
    tests/qec/test_shot_data_writing.py \
    tests/team/simulation/test_shot_data_writing_conformance.py \
    tests/api_contract/test_public_docstring_examples.py \
    tests/unit/test_qec_installed_profiles.py \
    tools/qec_installed_smoke.py \
    tools/verify_embedded_stabilizer_install.py \
    examples/qec/shot_file_encoding.py
python -m mypy --strict --explicit-package-bases \
    flagquantum/qec/execution/shot_data.py \
    flagquantum/simulation/stabilizer/_shot_data_write.py \
    flagquantum/simulation/stabilizer/_measurement_file_validation.py \
    tests/qec/test_shot_data_writing.py \
    tests/team/simulation/test_shot_data_writing_conformance.py \
    tests/api_contract/test_public_docstring_examples.py \
    tests/unit/test_qec_installed_profiles.py \
    tools/qec_installed_smoke.py \
    tools/verify_embedded_stabilizer_install.py \
    examples/qec/shot_file_encoding.py
python -m pytest tests/api_contract/test_public_docstring_examples.py -q -x
python tools/public_api_snapshot.py
python tools/docs_source_of_truth.py --check
python tools/check_qec_installed_profiles.py
python tools/ci_tier.py pr-default
python tools/ci_tier.py pr-runtime
FQ_BUILD_EMBEDDED_STIM=1 MAX_JOBS=2 python -m build --sdist --wheel --no-isolation
python tools/verify_embedded_stabilizer_install.py
python examples/qec/shot_file_encoding.py
```

Per-phase outer budgets: static tools 300 s each; focused fake/reference/source
and installed tests 900 s each; smoke/unit 2400 s; integration 3600 s; one offline
build 1200 s; each existing installation profile 300 s; native verifier and final
example 120 s each. Check already available locked cache closures before any
build. Terminate and verify stale-child cleanup after timeout, then stop.
These are proposed limits. Only read-only source identity, proposal syntax,
path ownership and document checks have been performed; no feature test, SDK
experiment or build has run for this proposal.

## Scope, progress and stopping rules

Exactly the 27 repository paths below may change after approval. Six are new:
the writing contract, repository proposal, private writer module, two test files
and the example. All other paths exist; no CI, runner,
manifest, setup.py, old API snapshot or vendor byte may change. The human-approved
scope record takes precedence over accidental omissions in a test-discovered fix.

| # | Path | Change |
| --- | --- | --- |
| 1 | `contracts/qec-shot-data-writing-v1-candidate.json` | Additive contract, approved signature and refusal/budget semantics |
| 2 | `docs/development/QEC_SHOT_DATA_WRITING_PROPOSAL.md` | Approved proposal and evidence plan |
| 3 | `flagquantum/qec/execution/shot_data.py` | Append only the new public function/export |
| 4 | `flagquantum/simulation/stabilizer/_shot_data_write.py` | Single Simulation owner for admission, buffer, codec and publication |
| 5 | `flagquantum/simulation/stabilizer/_measurement_file_validation.py` | Reuse/extend section-aware private validation without changing old behavior |
| 6 | `flagquantum/simulation/stabilizer/csrc/embedded_sampler.cpp` | One buffer-based private native writer using locked upstream encoders |
| 7 | `flagquantum/simulation/stabilizer/csrc/STIM_PROVENANCE.md` | Record reuse and integration evidence; no upstream bytes change |
| 8 | `tests/qec/test_shot_data_writing.py` | Readable fake/contract/refusal/cleanup workflows |
| 9 | `tests/team/simulation/test_shot_data_writing_conformance.py` | Independent byte fixtures and fake/native/SDK replacement |
| 10 | `tests/api_contract/test_public_docstring_examples.py` | Execute actual public writer doctest; retain old entries |
| 11 | `tests/unit/test_qec_installed_profiles.py` | Extend installed-profile contract without removing old checks |
| 12 | `tools/qec_installed_smoke.py` | Add actual write/read workflow to existing profile smoke |
| 13 | `tools/verify_embedded_stabilizer_install.py` | Exercise actual extension writer with no SDK |
| 14 | `examples/qec/shot_file_encoding.py` | Shortest write/read/packed-decode user workflow |
| 15 | `examples/README.md` | Register the new example |
| 16 | `docs/reference/API.md` | Document additive API, input semantics and budgets |
| 17 | `docs/reference/RELEASE_NOTES.md` | Record approved additive delivery |
| 18 | `docs/reference/KNOWN_LIMITATIONS.md` | Explicit bounds, divergences and remaining gaps |
| 19 | `docs/guides/STIM_USER_MIGRATION.md` | Map familiar Stim writer to Torch-native entry |
| 20 | `flagquantum/qec/execution/README.md` | Document owning public boundary and golden path |
| 21 | `flagquantum/simulation/stabilizer/README.md` | Document codec ownership and source reuse |
| 22 | `capability-maturity.toml` | Extend existing capability; no maturity inflation |
| 23 | `contracts/cudaq-parity-matrix.toml` | Extend generated parity evidence without claiming full coverage |
| 24 | `docs/generated/CAPABILITIES.md` | Regenerate only from authoritative capability data |
| 25 | `docs/reference/CUDAQ_PARITY_MATRIX.md` | Regenerate only from authoritative parity data |
| 26 | `docs/roadmap/FTQC_SIMULATION_MASTER_PLAN.md` | Record actual delivery only when accepted |
| 27 | `docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md` | Record actual delivery only when accepted |

FTS-04/R03 and FTS-07/R08 gain this slice only after real acceptance and remain
partial. FTS-02/R04/M0-M5 do not advance. The existing #1127 ledger PR and its
checks remain separate. No research, performance, throughput or complete-parity
claim is made. A missing build/test resource, new failure, timeout, source-closure
or path expansion stops with raw evidence and a concrete recovery proposal.

## Implementation authorization

The user approved this 27-path capability and PR publication on 2026-11-11,
bound to proposal SHA-256
`b22e6af76b65abd4a93ff97e9825197f83fe815b509b226f33b65b2818eeba83`.
Implementation and acceptance results are recorded separately; approval does
not establish passing tests or authorize a merge.

## Implementation status

The public candidate, Simulation-owned writer, six-format fixed-byte
conformance, SDK replacement path, installed smoke entry and user example are
under implementation. Focused fake tests passed 29 cases; fixed-byte plus
public-docstring checks passed 10 cases after correcting an r8 fixture to match
the direct Stim 1.16.0 reference. Ruff passed on the implemented Python paths.
Embedded build, no-SDK installed verification, broad test tiers, final-head
packaging and PR publication remain pending. Initial failure logs are retained
in the task evidence directory.
