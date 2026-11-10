# Bounded measurement-file to detector-file conversion capability package

Status: API and 26-path implementation approved by the user on 2026-11-10.
Implementation acceptance is pending; this document preserves the reviewed proposal.
Proposal SHA-256: `e2ef1156625a8760425f936f1e72c824e5d3fa8200e231f4e589acaf9cf8bc5a`.
Prerequisite #1116 merged as `0bf869e29c3cad0f5a0cac6d30018d7cb996a7e2`.
Prepared 2026-11-10, Asia/Shanghai. Suggested title:
`feat(qec): stream measurement files into detector and logical records`.

## Verified gap and dependency

Current-main inventory was read at `01976509b09c886c02548d0292854fec9af3659c`;
the converter implementation was separately read at `130b6737701402ca600ec5b4ec58ef88027d8dd8`. The existing
`CompiledMeasurementsToDetectionEventsConverter` has `convert()` but no
`convert_file()`. Its explicit and terminal factories already preserve measurement
occurrence columns and lower through the established Simulation boundary.

Do not open a second detector-writer PR. Existing
[#1116](https://github.com/flagos-ai/FlagQuantum/pull/1116), head
`2e30db35378a3e48876c4b0e42c533b9634f137a`, already implements the six-format detector
`sample_write()` bundle. It is OPEN at this read. Its local build/test claims in
the PR description were read, not independently reexecuted here. A green check
list or CLEAN merge state does not establish actual-main delivery.

Implement this follow-up only after #1116 lands, and re-audit its actual merge
and latest-main source. It changes shared file-validation/native-bridge paths;
concurrent writing there is not allowed. If #1116 changes the approved contract,
re-review this proposal before implementation. #1107's GPU failure and incomplete
dedicated actual-main QEC installation receipt remain separate open records.

## Proposed public API

Add exactly one method on the existing handle. No new public class, root export,
circuit kind, file-reader API, result field or format selector is introduced.
Names, keyword order and defaults follow Stim's familiar file-conversion API:

```python
def convert_file(
    self,
    *,
    measurements_filepath: str | os.PathLike[str],
    measurements_format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"] = "01",
    sweep_bits_filepath: str | os.PathLike[str] | None = None,
    sweep_bits_format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"] = "01",
    detection_events_filepath: str | os.PathLike[str],
    detection_events_format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"] = "01",
    append_observables: bool = False,
    obs_out_filepath: str | os.PathLike[str] | None = None,
    obs_out_format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"] = "01",
) -> None:
    ...
```

This is a proposed signature, not an implemented API. Existing `convert()` and
all factory signatures, root export order, serialized contracts and documented
refusals remain unchanged. All existing terminal/explicit handle factories must
support file conversion on their already accepted source domain; do not expose
a method that works only for one factory without documenting and approving that
restriction. The same internal prepared source/reference owns dense and file
conversion. Do not lower the circuit differently for the file path.
The method belongs on the already established QEC conversion handle instead of
another root export or experimental circuit type. Record an additive next-release
note in the existing API/proposal documentation and approve the behavior contract
before adding the machine-readable candidate entry.

| Input/output | First delivery |
| --- | --- |
| Measurement input | `01`, `b8`, `r8`, `ptb64`, `hits`, `dets` with M occurrence indices |
| Detector/logical output | `01`, `b8`, `r8`, `hits`, `dets` |
| `ptb64` conversion output | Reject before staging/conversion, including selected observable output; pinned upstream also rejects it |
| Logical layout | Detectors only; appended logical bits; separate same-record logical file; appended plus separate together is allowed by the converter |
| Sweep records | `None` only; reject a non-None path before file I/O because the accepted native Circuit domain has no sweep-bit inputs |
| Paths | `str` and path-like inputs; normalize to strings at the optional SDK boundary |
| Zero records | Empty accepted input yields complete empty outputs without invoking the numerical conversion engine |

Do not copy detector sampler flag rules into the converter: upstream
`convert_file()` permits appended output and a separate observable file together.
There is no prepend flag on this converter API. Current `sample()` and
`sample_write()` signatures are unchanged.

## Source reuse and architectural ownership

Pinned Stim 1.16.0 source distribution:
`https://files.pythonhosted.org/packages/77/bd/c0f5d6241cdb806400a63e0c7febe22982ddf1c8761142bb0ea6b51d3f50/stim-1.16.0.tar.gz`.
SHA-256 `9092a996429dcf616d8d4ca01fec886b7b310caf9425a39fcfe499005636bd52`;
upstream commit `e2fc1eca7fd21684d433aa5f10f4504ea4860d07`.

All 96 current vendor blobs were statically compared with that cached archive
and are identical. Reuse existing `stream_measurements_to_detection_events_helper`,
the pinned record readers/writers, reference sample and noiseless circuit from
`measurements_to_detection_events.h/.inl`. Keep the existing 96-file / 37-unit
closure and Apache-2.0 bytes. No upstream pybind binding, NumPy ABI, new reader
library or new dependency is needed by the embedded path. Static closure reuse
is not a successful new build; if compilation needs another upstream file, stop
and request an exact source-list amendment.

Simulation owns numerical preparation, native file handles and SDK translation.
QEC owns the public conversion request and measurement-map semantics. Extend the
existing private backend boundary so one consumer can be replaced by a contract
fake, pinned SDK implementation and embedded implementation. Do not introduce a
second circuit IR, public backend selector, registry or result envelope. File
inspection extends the existing private validator; preserve its existing
measurement and detector writer defaults and failure tests.

## Input, memory and file bounds

Retain source limits (4096 qubits, 65536 explicit measurements, 100000 lowered
operations, 65536 parity rows, 100000 parity references) and the original 64 MiB
input-copy/output/principal-table guards where applicable. Retain T and
conditional-H rejection before preparation, the accepted single-slot conditional-X
path, no-Stim native selection, gauge-parity refusal and no approximation/SV fallback.

- Only regular files are accepted as input. Refuse pipes, devices, input symlinks,
  missing parents, output/input aliases and duplicate output locations before
  conversion. Concurrent mutation of caller input is unsupported and must be
  detected by recorded file metadata around snapshot creation.
- Copy the opened input into one private bounded snapshot, then inspect/count
  that snapshot with at most 64 KiB parser reads. Both SDK and native paths
  consume it. This prevents a validation/conversion second-open race on the
  caller's path. Input snapshot size is at most 1 GiB. Input parsing retains no
  all-shot table, row list, or unbounded token/line buffer.
- Validate exact row width/count, typed M indices, monotonic sparse indices,
  truncated records/groups, b8 padding and PTB64 complete groups. Compute output
  bounds from actual inspected shot count before any numerical conversion.
  Observable width zero requires an explicit format-aware bound, including
  newline-only sparse/text records; do not reuse a zero-byte sparse bound.
- Aggregate final output bound is at most 1 GiB, including both selected files.
  Private snapshot plus staged output bounds are at most 2 GiB. Group required
  free-space reservations by filesystem device before starting conversion.
  These are request/allocation guards, not a filesystem quota or a whole-process
  RSS guarantee; disk-space changes and I/O failure remain observable errors.
- Pinned upstream converts in batches of 1024 records. Validate the conservative
  padded principal-table estimate, including input, both output orientations,
  frame/scratch and same-record observable tables, against 64 MiB before native
  table allocation. Do not call `convert()` on a fully loaded file.
- Stage each output on its target filesystem, check format/size/row count, then
  publish with the existing no-replace link pattern. Never overwrite or truncate
  a caller path. Pair visibility is sequential, not atomic across two files.
  On failure roll back only links proven to be created by this call; report any
  cleanup error/residual path and preserve the original failure. No silent retry.
- There is no RNG in conversion. Invalid file requests do not enter conversion;
  failures cannot silently return a partial-success record or change dense API
  behavior. Preserve current converter reuse semantics after a contained I/O
  error; do not add a new public close/state API or silently poison unrelated
  prepared numerical state. Backend corruption must be raised explicitly.

## Exact 26-path implementation scope

Integration owns the approved API/contract delta; Core and Simulation implement
behind it. Submit the contract, fake and consumer-conformance work first within
the same capability PR, then implementations. No CI, dependency, runner, setup.py,
root-export or vendor-source changes are in scope.

| # | Path | Purpose |
| --- | --- | --- |
| 1 | `flagquantum/qec/execution/measurement_conversion.py` | Add only the approved handle method; retain convert signature, fields and factory behavior. |
| 2 | `flagquantum/simulation/stabilizer/_record_conversion.py` | Prepare one authoritative file-conversion backend for explicit and terminal sources; retain current dense consumers. |
| 3 | `flagquantum/simulation/stabilizer/_embedded_sampler.py` | Add the private embedded file-call adapter and preserve measurement/detector routes. |
| 4 | `flagquantum/simulation/stabilizer/_measurement_file_validation.py` | Reuse bounded format parsing; add snapshot inspection/counting and typed M/D/L validation without altering existing defaults. |
| 5 | `flagquantum/simulation/stabilizer/csrc/embedded_sampler.cpp` | Bind the existing upstream stream converter through the limited CPython API; no new numerical implementation. |
| 6 | `flagquantum/simulation/stabilizer/csrc/STIM_PROVENANCE.md` | Record the reused 96-file, 37-translation-unit closure and native call. |
| 7 | `flagquantum/qec/execution/README.md` | Document QEC conversion ownership and shortest contributor workflow. |
| 8 | `flagquantum/simulation/stabilizer/README.md` | Document numerical/file ownership, limits and rejection stage. |
| 9 | `contracts/qec-measurement-file-conversion-v1-candidate.json` | Record the new API and its bounded file contract; no duplicate circuit/result/identity type. |
| 10 | `docs/development/QEC_MEASUREMENT_FILE_CONVERSION_PROPOSAL.md` | Commit the approved proposal and explicit compatibility analysis. |
| 11 | `docs/reference/API.md` | Document actual supported method and example after implementation. |
| 12 | `docs/reference/KNOWN_LIMITATIONS.md` | Record output PTB64, sweep, allocation and publication boundaries. |
| 13 | `docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md` | Map this delivery to upstream entries and remaining parity gaps. |
| 14 | `docs/guides/STIM_USER_MIGRATION.md` | Show the familiar Stim method name with native Circuit and maps. |
| 15 | `tests/qec/test_measurement_file_conversion.py` | Scenario tests: independent fixed records, contract fake, producer replacement, formats, budgets and I/O failures. |
| 16 | `tests/api_contract/test_public_docstring_examples.py` | Add the converter class/method to real doctest execution; retain all existing entries. |
| 17 | `tools/qec_installed_smoke.py` | Exercise current four installation profiles using the new public workflow. |
| 18 | `tests/unit/test_qec_installed_profiles.py` | Verify installed-smoke failure isolation and actual exercised paths. |
| 19 | `tools/verify_embedded_stabilizer_install.py` | Prove native converter calls with neither Stim nor PyMatching installed. |
| 20 | `examples/qec/measurement_file_conversion.py` | Add a deterministic public file-to-detector example. |
| 21 | `examples/README.md` | Register the example in the existing index. |
| 22 | `docs/roadmap/FTQC_SIMULATION_MASTER_PLAN.md` | Update P0/FTS-04/R03/R08 evidence without closing scientific milestones. |
| 23 | `capability-maturity.toml` | Record bounded file conversion within the existing capability; no maturity promotion without evidence. |
| 24 | `contracts/cudaq-parity-matrix.toml` | Update the existing generated parity source entry; do not infer CUDA-Q or full Stim parity. |
| 25 | `docs/generated/CAPABILITIES.md` | Regenerate capabilities from their authoritative source. |
| 26 | `docs/reference/CUDAQ_PARITY_MATRIX.md` | Regenerate parity documentation from its authoritative source. |

The candidate contract is justified by the existing protected converter having
no file API: it records this one additive method and its I/O/refusal semantics.
It does not replace or regenerate prior API snapshots to conceal a change.
Generated capability/parity documents derive from the two authoritative source
files; retain scientific maturity levels unless new acceptance supports promotion.

## User example and independent reference

The current public workflow can already convert the same fixed records in memory:

```python
import torch
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.execution.context import MeasurementMap

source = (
    DynamicCircuit(1).measure(0, classical_bit=0).measure(0, classical_bit=0)
    .conditional("x", 0, classical_bit=0).measure(0, classical_bit=0)
)
converter = fq.Circuit.from_ir(source.to_ir()).compile_m2d_converter(
    detector_map=MeasurementMap(3, ((0, 1), (0, 2))),
    observable_map=MeasurementMap(3, ((1,),)),
)
records = torch.tensor([[False, True, False]] * 64)
detectors, observables = converter.convert(
    measurements=records, separate_observables=True
)
assert detectors.tolist() == [[True, False]] * 64
assert observables.tolist() == [[True]] * 64
```

This current-API snippet is source/syntax reviewed here, not newly executed
FlagQuantum acceptance. It needs current FlagQuantum plus its embedded converter
or the optional pinned SDK. The proposal removes the need to load all records
into this dense tensor when the user starts from a file.

**Proposed FlagQuantum usage after implementation; not run against FlagQuantum
in this proposal.** The Circuit/MeasurementMap/compile/convert entry points already
exist; only `convert_file()` below is new. The fixed file represents externally
captured records with an injected fault at the second measurement. Reusing slot 0
does not merge its three occurrence columns: m0=0, m1=1, m2=0. Feedback reads the
overwritten value m1=1. D0=m0 XOR m1=1, D1=m0 XOR m2=0, L0=m1=1.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.execution.context import MeasurementMap

source = (
    DynamicCircuit(1)
    .measure(0, classical_bit=0)
    .measure(0, classical_bit=0)
    .conditional("x", 0, classical_bit=0)
    .measure(0, classical_bit=0)
)
circuit = fq.Circuit.from_ir(source.to_ir())
converter = circuit.compile_m2d_converter(
    detector_map=MeasurementMap(3, ((0, 1), (0, 2))),
    observable_map=MeasurementMap(3, ((1,),)),
)
with TemporaryDirectory() as directory:
    root = Path(directory)
    records = root / "measurements.b8"
    # Fixed external records (m0, m1, m2) = (0, 1, 0), with feedback after m1.
    records.write_bytes(b"\x02" * 64)
    events = root / "events.b8"
    logical = root / "observables.b8"
    converter.convert_file(
        measurements_filepath=records,
        measurements_format="b8",
        detection_events_filepath=events,
        detection_events_format="b8",
        obs_out_filepath=logical,
        obs_out_format="b8",
    )
    assert events.read_bytes() == b"\x01" * 64
    assert logical.read_bytes() == b"\x01" * 64
    print("shots 64")
    print("detectors 10; observables 1")
    print("file_bytes", events.stat().st_size, logical.stat().st_size)
```

Expected output:

```text
shots 64
detectors 10; observables 1
file_bytes 64 64
```

Run after acceptance with `python -m examples.qec.measurement_file_conversion`.
The example needs the embedded wheel, CPU PyTorch and no hardware, credentials,
Stim Python package or PyMatching. Batch decoding remains a later package; do
not describe output files as decoded logical-error estimates.

Actual local evidence in this proposal is **upstream only**: 90 fixed-record
byte comparisons (6 inputs x 5 outputs x 3 layouts) passed, and 12 PTB64-output
calls raised the expected upstream error. Input records were 010; expected
detectors 10, logical 1, appended b8 byte 05. The earlier detector-writing
audit passed 54 independent fixed-byte cases, and confirmed no need to duplicate
#1116. These small references are not a performance study or a FlagQuantum test.

Original reference failures are retained: NumPy was absent for the initial
upstream reader harness, so fixed byte references were used without installing
it; converter bindings rejected Path objects despite the type stub, so the
revised adapter reference passed strings. No dependency was installed.

## Acceptance commands and stop rules

All commands below are **future acceptance**, not executed here. Use the agreed
locked Python/build/runtime environments; if a required wheel/tool is missing,
stop and list it rather than installing or downloading opportunistically.
Check the final base/head and all 26 owned paths before editing:

```bash
python tools/check_team_scope.py --team integration --files flagquantum/qec/execution/measurement_conversion.py flagquantum/simulation/stabilizer/_record_conversion.py flagquantum/simulation/stabilizer/_embedded_sampler.py flagquantum/simulation/stabilizer/_measurement_file_validation.py flagquantum/simulation/stabilizer/csrc/embedded_sampler.cpp flagquantum/simulation/stabilizer/csrc/STIM_PROVENANCE.md flagquantum/qec/execution/README.md flagquantum/simulation/stabilizer/README.md contracts/qec-measurement-file-conversion-v1-candidate.json docs/development/QEC_MEASUREMENT_FILE_CONVERSION_PROPOSAL.md docs/reference/API.md docs/reference/KNOWN_LIMITATIONS.md docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md docs/guides/STIM_USER_MIGRATION.md tests/qec/test_measurement_file_conversion.py tests/api_contract/test_public_docstring_examples.py tools/qec_installed_smoke.py tests/unit/test_qec_installed_profiles.py tools/verify_embedded_stabilizer_install.py examples/qec/measurement_file_conversion.py examples/README.md docs/roadmap/FTQC_SIMULATION_MASTER_PLAN.md capability-maturity.toml contracts/cudaq-parity-matrix.toml docs/generated/CAPABILITIES.md docs/reference/CUDAQ_PARITY_MATRIX.md
python -m ruff check flagquantum/qec/execution/measurement_conversion.py flagquantum/simulation/stabilizer/_record_conversion.py flagquantum/simulation/stabilizer/_embedded_sampler.py flagquantum/simulation/stabilizer/_measurement_file_validation.py tests/qec/test_measurement_file_conversion.py tests/api_contract/test_public_docstring_examples.py tools/qec_installed_smoke.py tests/unit/test_qec_installed_profiles.py tools/verify_embedded_stabilizer_install.py examples/qec/measurement_file_conversion.py
python -m black --check flagquantum/qec/execution/measurement_conversion.py flagquantum/simulation/stabilizer/_record_conversion.py flagquantum/simulation/stabilizer/_embedded_sampler.py flagquantum/simulation/stabilizer/_measurement_file_validation.py tests/qec/test_measurement_file_conversion.py tests/api_contract/test_public_docstring_examples.py tools/qec_installed_smoke.py tests/unit/test_qec_installed_profiles.py tools/verify_embedded_stabilizer_install.py examples/qec/measurement_file_conversion.py
python -m mypy --strict flagquantum/qec/execution/measurement_conversion.py flagquantum/simulation/stabilizer/_record_conversion.py flagquantum/simulation/stabilizer/_embedded_sampler.py flagquantum/simulation/stabilizer/_measurement_file_validation.py tests/qec/test_measurement_file_conversion.py tests/api_contract/test_public_docstring_examples.py tools/qec_installed_smoke.py tests/unit/test_qec_installed_profiles.py tools/verify_embedded_stabilizer_install.py examples/qec/measurement_file_conversion.py
python -m pytest tests/qec/test_measurement_file_conversion.py tests/qec/test_explicit_measurement_conversion.py tests/qec/test_terminal_measurement_conversion.py tests/qec/test_native_detector_execution.py tests/team/simulation/test_explicit_record_conversion_conformance.py tests/unit/test_qec_installed_profiles.py -m "unit or integration" -q
python -m pytest tests/api_contract/test_public_docstring_examples.py -q
python -m examples.qec.measurement_file_conversion
python tools/ci_tier.py pr-default
python tools/ci_tier.py pr-runtime
```

The new file tests carry `unit` and real `integration` markers using the existing
SDK split; prove collected node IDs for both profiles so optional dependencies
do not hide the contract/fake cases. Existing no-SDK/T/conditional-H tests remain.
Add same-consumer fake/SDK/native replacement and both terminal/explicit factory
coverage. Test mixed output formats and appended-plus-separate output, padding
boundaries, zero records/observables, malformed input, limits and failed publication.
Assert bounded read sizes and native/SDK call accounting; do not replace these
with timing assertions or implementation-name tests.

Reuse the already locked offline macOS arm64 build closure and verify version,
source hash and installed import provenance first. Example commands in that
prepared environment (each externally supervised by an explicit timeout):

```bash
PIP_NO_INDEX=1 FQ_BUILD_EMBEDDED_STIM=1 python -m build --sdist --wheel --no-isolation --outdir /private/tmp/fq-file-conversion-dist
python tools/verify_distribution_artifacts.py /private/tmp/fq-file-conversion-dist
python tools/check_qec_installed_profiles.py --dist-dir /private/tmp/fq-file-conversion-dist --report-dir /private/tmp/fq-file-conversion-installed
python tools/verify_embedded_stabilizer_install.py
```

Run the native verifier only from a fresh installed no-Stim/no-PyMatching profile,
outside the checkout with cleared source overlays. The existing installed-profile
harness performs four wheel profiles and sdist-both; store source/sdist/wheel
member equality, all vendor hashes, native call counters and final-head example
output. Building within the checkout is not installed-package acceptance.
Budget focused runs at 300 s each and one offline build at 1200 s; broader tier
and profile commands use their reviewed existing limits, no detached workers or
automatic continuations. On a new failure retain raw logs and stop; do not
automatically retry, broaden paths, weaken thresholds or restore old CI.

After separately approved implementation/publication, create one PR directly
against main with final-head executed user code. Follow its automatic checks;
merging/actual-main receipt must name the exact reviewed source and preserve any
GPU exception as a separate user decision. This proposal requests no CI recovery,
runner/config change, research probe, worker or workload matrix.

## Progress impact and next boundary

FTS-04/R03 gain a bounded records-to-detector vertical path and implementation
replacement evidence. R08 gains file-scale processing without all-shot tables.
All three remain partial until their broader scenario, statistical and scale
gates close. FTS-02, mixed Clifford/non-Clifford execution, domestic-device
qualification and M0-M5 are unchanged. No total parity percentage is inferred.

After this package, inspect bounded file-to-decoder batch ingestion and remaining
Stim record operations against the generated upstream inventory. Do not add
either as an unapproved extra to this method's scope.
