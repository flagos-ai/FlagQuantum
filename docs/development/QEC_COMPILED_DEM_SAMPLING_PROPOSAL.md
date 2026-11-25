# Native reusable detector-error-model sampling capability bundle

Status: ready for API/source-scope review, not approved or implemented.
Date: 2026-11-12 (Asia/Shanghai).
Baseline: actual merged main `39b16b446733c215a14c59332bcf69c350f62069` from PR #1152.
Proposed branch: `feat/qec-compiled-dem-sampling`, cut from then-current main after checking for overlapping deliveries.
Scope: 30 integration paths plus 2 unchanged upstream files, 32 repository paths in total.
Execution classification: `single_device_fast_path`; CPU-only local sampling. No scalability or performance claim.

## 1. User problem and coherent delivery

A user can already produce a DEM from a supported circuit, decode its detector events, and read/write shot data in six formats. The missing reusable native DEM sampler prevents efficiently recording/replaying individual error mechanisms and streaming the resulting detector/observable records through that same workflow.

Deliver one complete bundle: compile a sampler from the existing owned DEM, return detector/observable/error tables, replay error records, write the three corresponding streams in supported formats, decode bounded detector batches, validate refusal and cleanup, and prove installed execution without the Stim Python package. Do not split the three methods into independent PRs.

This is the FlagQuantum tensor-native workflow. Matching method names and calling conventions reduce the learning cost, but CPU Torch tensors, bounded resources, owned DEM restrictions and no-overwrite files are intentional differences from Stim's NumPy API. It does not establish complete Stim API parity or replace the separately reviewed full compatibility namespace.

## 2. Existing behavior and preservation

`flagquantum/qec/dem/model.py` owns `DetectorErrorModel`, `DemError`, `DemSample` and `dem_sampling(shots=..., seed=...)`. Preserve every existing export, signature, field, default, exception and RNG behavior. `DemSample` remains its existing two-field record; do not add an error-record field to it.

`DetectorErrorModel` canonicalizes `errors` in its constructor. New error columns correspond to `model.errors[i]` in this canonical order, not the insertion order of the tuple passed to the constructor and not the source order of an imported upstream text. No mechanisms are combined or discarded for the new sampler. Document this beside `recorded_errors_to_replay` and test distinct and repeated signatures with different probabilities.

Existing `to_stim_text()` refuses every stated `error_id`, including a label appearing on only one mechanism. Reuse that refusal before preparing either backend. In particular, mutually exclusive groups must never be treated as independent Bernoulli mechanisms. Existing `dem_sampling()` remains the supported owned route for such groups; the new method does not silently choose it as a fallback.

Existing model restrictions remain: at least one detector; each mechanism flips a detector or observable; supported probability and target validation; no direct arbitrary upstream text/REPEAT constructor is introduced. Zero observables and zero errors remain representable. This bundle changes no Circuit gate, Clifford, Pauli-feedback, T-gate or conditional-H admission rule.

## 3. Proposed public API and naming review

Add only the following public methods and returned handle. These are proposed interfaces, not currently executable FlagQuantum APIs.

```python
# Existing flagquantum.qec.DetectorErrorModel; lazy delegation at call time.
def compile_sampler(self, *, seed: int | None = None) -> CompiledDemSampler: ...

# flagquantum.qec.execution.dem_sampling
class CompiledDemSampler:
    def sample(
        self,
        shots: int,
        *,
        bit_packed: bool = False,
        return_errors: bool = False,
        recorded_errors_to_replay: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...

    def sample_write(
        self,
        shots: int,
        *,
        det_out_file: str | os.PathLike[str] | None,
        det_out_format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"] = "01",
        obs_out_file: str | os.PathLike[str] | None,
        obs_out_format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"] = "01",
        err_out_file: str | os.PathLike[str] | None = None,
        err_out_format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"] = "01",
        replay_err_in_file: str | os.PathLike[str] | None = None,
        replay_err_in_format: Literal["01", "b8", "r8", "ptb64", "hits", "dets"] = "01",
    ) -> None: ...
```

`compile_sampler` names reusable preparation, `sample` returns samples, and `sample_write` writes samples. They match the relevant Stim 1.16.0 names and keyword names. No `native`, `stim`, `new` or `v2` qualifier enters the public names. The API belongs on the existing owned DEM and in the QEC execution namespace because it prepares and executes that model; an experimental parallel record would duplicate model identity. The handle is exported from its owning module only; no `fq.*`, `qec.__all__` or existing module export order is changed. Its constructor refuses direct use with instructions to call `model.compile_sampler()`.

For `N` shots and dimensions `(D, O, E)`, unpacked outputs have CPU `torch.bool` shapes `(N,D)`, `(N,O)`, `(N,E)`; packed outputs have CPU `torch.uint8` shapes `(N,ceil(D/8))`, `(N,ceil(O/8))`, `(N,ceil(E/8))`. Each stream packs its own columns independently, little-endian within bytes. Output padding bits are zero. The tuple always has three entries; the third is `None` unless requested. Arrays own their storage and remain valid across later calls and handle destruction.

Replay accepts CPU dense strided bool `(N,E)` or uint8 `(N,ceil(E/8))` tensors independently of output packing. Validate/copy noncontiguous inputs once under the capacity guard, retain no caller storage, and ignore unused packed input padding bits as the upstream conversion does; emitted padding is zero. No CUDA, sparse, other dtype, broadcasting, resampling or implicit dtype conversion. Replay is allowed even when `return_errors=False`. Error bits, including bits for p=0/p=1 mechanisms, specify the mechanism events to apply; replay does not redraw or constrain them to the sampling probabilities.

`shots` must be a nonnegative built-in integer, excluding bool; zero returns empty correctly shaped tables or empty files without advancing RNG. Seed is `None` or a built-in integer in `[0, 2**64)`, excluding bool. Preserve existing API behavior elsewhere even if a neighboring old sampler requires positive shots.

RNG guarantees cover identical backend/build/SIMD, model and call sequence. No equality guarantee across Stim versions, platforms, SDK versus embedded dispatch or different shot partitioning. Internal processing uses fixed 1024-shot stripes; direct upstream one-call seeded byte identity is not promised for differently partitioned calls. Replay and p=0/p=1 witnesses provide exact cross-implementation oracles.

The handle belongs to its creating process and thread. It is not serializable. Validation failures before execution leave it usable and consume no RNG; backend execution or publication failures invalidate it, with no transparent fallback or implicit retry.

## 4. Concrete proposed user workflow

This code is an acceptance target for the implementation PR, not a claimed current API.

```python
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from flagquantum.qec import DemError, DetectorErrorModel, PyMatchingDecoder
from flagquantum.qec.execution.shot_data import (
    iter_shot_data_file, write_shot_data_file,
)

model = DetectorErrorModel(1, 1, (DemError(0.1, (0,), (0,)),))
sampler = model.compile_sampler(seed=7)  # proposed
sampler_data, logicals, error_bits = sampler.sample(
    64, bit_packed=True, return_errors=True,
)
assert error_bits is not None
replayed, replayed_logicals, unused = sampler.sample(
    64, bit_packed=True, recorded_errors_to_replay=error_bits,
)
assert torch.equal(sampler_data, replayed)
assert torch.equal(logicals, replayed_logicals) and unused is None

decoder = PyMatchingDecoder.from_detector_error_model(model)
with TemporaryDirectory() as directory:
    root = Path(directory)
    write_shot_data_file(
        data=error_bits, path=root / "input-errors.b8", format="b8",
        num_measurements=model.num_errors,
    )
    sampler.sample_write(
        64,
        det_out_file=root / "detectors.b8", det_out_format="b8",
        obs_out_file=root / "observables.b8", obs_out_format="b8",
        err_out_file=root / "replayed-errors.b8", err_out_format="b8",
        replay_err_in_file=root / "input-errors.b8", replay_err_in_format="b8",
    )
    shots = batches = failures = 0
    with closing(iter_shot_data_file(
        path=root / "detectors.b8", format="b8", num_detectors=1,
        bit_packed=True, batch_size=17,
    )) as records:
        for events in records:
            predictions = decoder.decode_packed_batch(events)
            failures += int((predictions != logicals[shots:shots + len(events)])
                            .any(dim=1).sum().item())
            shots += len(events)
            batches += 1
    assert (shots, batches, failures) == (64, 4, 0)
    print("shots", shots, "batches", batches, "logical_failures", failures)
```

Expected final line: `shots 64 batches 4 logical_failures 0`. Error-file columns are measurement-labelled (`M`) for the existing generic file reader/writer; this is a byte-format convention, not a change in domain meaning. Detector `dets` output uses `D`, observable output uses `L`.

For an existing native circuit, obtain the supported owned model using the current `circuit.detector_error_model(detector_map=..., observable_map=...)`, then use the same proposed `model.compile_sampler()`. No new circuit abstraction or decoder type is required.

## 5. Upstream source reuse and exact closure

Pinned Stim 1.16.0 commit: `e2fc1eca7fd21684d433aa5f10f4504ea4860d07`.

- PyPI sdist SHA-256: `9092a996429dcf616d8d4ca01fec886b7b310caf9425a39fcfe499005636bd52`.
- Tag archive SHA-256: `f8b2b2e77adf91027583ef17fc44facd776694df0f45c4d8b990580bc68fb410`.
- Existing 96 file hashes agree with both archives and the accepted main snapshot.
- Add `src/stim/simulators/dem_sampler.h`, SHA-256 `31e8ce743db8e0b787b7e908c49d3f323ce7fea3647e95994e97a23363f2bb32`.
- Add `src/stim/simulators/dem_sampler.inl`, SHA-256 `330ebeb41ee90cf78737dce00fc2bf9f8e755686cb447128ac3c9aa3db3f039a`.

Both additions are byte-identical in the sdist and tag archive. Recursive quoted-include review finds no other missing upstream input. Candidate closure is 98 files and the same 37 translation units. This is a static closure finding; compile/link acceptance has not run. Any newly required file stops implementation for a reviewed scope change, never an ad hoc copy.

Reuse `stim::DemSampler<width>` construction, resample, native error replay tables, seeded RNG and sample_write directly. Keep the upstream license and all 96 existing upstream bytes unchanged. The two new files retain their upstream copyright/license headers. No upstream source patch, pybind11, NumPy C API, extra dependency or compiler flag is introduced. Existing C++20 CPython limited-API bridge and macOS arm64 admission stay in place; do not widen setup.py platform admission.

Upstream `dem_sampler.pybind.cc/.h` and `detector_error_model.pybind.cc` are semantic references only, not build inputs. Upstream native tests are references only. The source audit records their hashes. Translate only the owned CPython bridge and tensor/file boundary; do not rewrite the sampling algorithm.

QEC records remain dependency-neutral and import execution lazily only when compilation is requested. QEC execution owns composition, validation and returned handle semantics; Simulation owns upstream interaction and numerical buffers. Optional Stim objects never cross into QEC consumers. An embedded-profile missing/broken extension fails closed; the ordinary optional-SDK path uses the existing vetted Stim 1.16.0 loader. A contract fake, SDK adapter and embedded adapter must be swappable without editing the consumer.

## 6. Memory, file and model bounds

Preparation guards apply before serialization or engine creation: at most 65,536 detectors, 65,536 observables, 65,536 mechanisms, 262,144 total targets, and 1 MiB of canonical UTF-8 DEM text. Estimate serialized length before constructing that text. Existing model construction restrictions still apply. Also reject a preparation whose native principal-table estimate exceeds the 64 MiB guard.

The native sampler always has an error table even when the caller omits returned errors. The constructor allocates detector, observable and error tables. Use a fixed 1024-shot native stripe, matching upstream's initial stripe size. Do not call upstream Python `sample(N)` with unbounded N: both adapters process at most 1024 shots per internal call. Native file writes use that same fixed native sampler. Input-file replay is prevalidated as an immutable bounded snapshot, then streamed without a full error array.

Let `P(x)=1024*ceil(x/1024)` (including `P(0)=0`), `A=(P(D)+P(O)+P(E))*P(1024)/8`, `B` be all returned result bytes including errors only when requested, and `I` be the owned logical replay input byte size (zero without array replay). The candidate preallocation guard is:

`4*A + 3*B + 3*I + 65,536 <= 67,108,864`.

The factors reserve old/current/native transpose/scratch tables and overlapping result/replay copies. Implementation must enumerate actual live buffers against this bound; unexpected extra storage stops for review instead of increasing the limit. Zero-copy views of mutable caller tensors are forbidden. Replay never allocates an `N*E` floating-point probability table. Model/parser heap, Python interpreter/dependency memory and retained caller results are outside this principal-data limit: this is not a 64 MiB process-RSS claim.

For `(D,O,E)=(1024,16,2048)` with returned error records, the pure arithmetic review found:

| Output | Replay | Maximum N under this formula |
| --- | --- | ---: |
| bool | none | 7,010 |
| packed | none | 56,084 |
| bool | bool | 4,215 |
| packed | packed | 33,720 |

Each next N exceeds the candidate bound. These are simulated allocation guards, not benchmark/capacity measurements.

File output uses the existing six formats and validation helpers. Enforce an aggregate 1 GiB upper bound over all selected final outputs and existing per-device free-space checks. PTB64 output requires a multiple of 64 shots before execution. Validate unused format arguments as well. `det_out_file` and `obs_out_file` are required keyword arguments but may be None; all output files may be None, matching upstream's discard/advance workflow. Replay input has a separate 1 GiB snapshot bound; do not describe final-output plus replay-snapshot disk usage as a total 1 GiB bound.

Destinations must be distinct regular local target names, absent including symlinks/dangling links, with existing parent directories. No replacement, file-like objects, network URI or bytes path. Validate input/output aliasing. Files are staged next to their destinations, fully validated, then linked without replacement. Multiple outputs become visible sequentially; do not promise cross-file atomic visibility. On failure, remove only published inodes created by this call and invalidate the handle. Surface cleanup failures with retained paths and original cause.

Replay files must contain exactly the requested logical shot count. Validate format/width/padding and trailing/incomplete records before numerical execution. For zero-width errors, handle fixed-width empty files with the explicit shots argument; do not infer shot counts from zero-byte records. SDK-specific overwrite behavior or tolerance for trailing replay records is not adopted silently; record these native-policy differences.

## 7. Contracts, acceptance order and stop conditions

API owner approval is required by `docs/development/PUBLIC_API_PROTECTION.md`, Additive stable API item 5, "API-owner approval". Scope/source approval must explicitly cover all 32 paths and the 96-to-98 source expansion. Integration owns the coordinated boundary change; Core owns the model facade and Simulation the numerical implementation. Deliver approved contract, fake and conformance first, then the domain implementations in the same PR. Do not begin a parallel track on an unproven boundary.

First prove the smallest workflow: independent model -> preparation -> recorded sampling -> deterministic replay -> file round trip -> decoder -> observed result, plus rejection and cleanup. Preserve historical API baselines and frozen fixtures. New contracts are added only after approval; no snapshot update merely to obtain a passing check. Add concise Google-style examples and register all new public workflow examples in the real doctest list.

Acceptance stages, each once with raw stdout/stderr, command, environment, digest, JUnit where relevant, and owned-child cleanup on timeout:

| Stage | Limit | Required evidence |
| --- | ---: | --- |
| Preflight | 120 s | Exact base, clean branch/locks, approved hashes, ownership, cached dependency/tool closure, compiler/platform and source archive identity |
| Contract/fake and independent reference | 600 s | Signatures; replacement; canonical mechanism order; replay bool/packed combinations; p=0/p=1; zero dimensions/shots; duplicate signatures; no-group-flattening; ownership/invalidation |
| Static policy/style | 300 s per command | Dependency policy, Ruff, Black, strict Mypy; no ignores or weakened rules |
| Existing-SDK differential | 600 s | Replay/deterministic exact outputs and six file formats, three independent fixed-seed marginal sanity cases (N=4096, p=1/8,1/4,1/2; absolute count errors at most 127,167,192 respectively); no performance claim |
| macOS arm64 offline build | 1,200 s wheel; 180 s sdist | Unmodified 98-file closure; same flags; no dependency fetch; exact artifact hashes |
| Installed native/no-SDK and SDK profiles | 900 s per profile | Real package install with no dependency resolution, genuine metadata, extension origin/hash and actual new native entry calls; 1,800 s total |
| Default smoke/unit | 2,400 s | Complete seeded default tier in a prepared real environment |
| Local integration | 3,600 s | Complete seeded integration tier; all existing file/conversion paths retained |
| API/docs/release | 300 s per command | Existing API snapshot; additive contract; executable docstrings/example index; generated capabilities/parity; original claim thresholds |
| Final-head example and publication | 120 s example; 180 s push; 60 s PR creation | Execute user code at final head; PR directly to main and example/results in description |

Maximum active local allowance before publication is 14,220 seconds (sum of the named maxima, with one 120-second preflight, one 600-second contract run, four 300-second static commands, one 600-second SDK run, 1,200+180 build seconds, 1,800 installed-profile seconds, 2,400 default seconds, 3,600 integration seconds, eight 300-second API/docs commands and a 120-second example). This is a ceiling, not an instruction to exhaust it. Do not repeat accepted stages absent a new code change or a separately approved recovery.

Cached build/test dependencies must be proven available before starting. Offline project-wheel installation is allowed after approval; installing/downloading dependencies is not. No build/install/implementation has been performed by this proposal. No extra Linux runner admission, GPU workload, research task, dependency or CI/runner configuration change. On publication, automatic CI/GPU may run normally. Merge is not authorized by approval to implement/publish this bundle; verify exact-head gates and obtain applicable merge authorization.

Any new build/test failure, timeout, unavailable resource, source/hash mismatch or scope expansion preserves raw logs and stops. No automatic retry, old-head CI rerun, threshold change, test weakening or source-list expansion.

## 8. Commands after approval only

`PY` must name the verified interpreter; run from the new branch root. `EVIDENCE` is an owned fresh directory outside the repository. These commands are acceptance targets and have not been run against a new implementation.

```bash
python tools/check_dependency_policy.py
python -m pytest tests/qec/test_compiled_dem_sampler.py tests/unit/test_compiled_dem_sampler_budget.py tests/unit/test_embedded_dem_sampler_source_lock.py -q -x
python -m pytest tests/team/simulation/test_compiled_dem_sampler_conformance.py tests/qec/test_dem_sampling.py tests/qec/test_dem_error_ids.py tests/qec/test_correlated_circuit_noise.py tests/qec/test_shot_data_reading.py tests/qec/test_shot_data_writing.py -q -x
python -m ruff check flagquantum/qec/dem/model.py flagquantum/qec/execution/dem_sampling.py flagquantum/simulation/stabilizer/_dem_sampling.py tests/qec/test_compiled_dem_sampler.py tests/team/simulation/test_compiled_dem_sampler_conformance.py tests/unit/test_compiled_dem_sampler_budget.py tests/unit/test_embedded_dem_sampler_source_lock.py examples/qec/compiled_dem_sampler.py
python -m black --check flagquantum/qec/dem/model.py flagquantum/qec/execution/dem_sampling.py flagquantum/simulation/stabilizer/_dem_sampling.py tests/qec/test_compiled_dem_sampler.py tests/team/simulation/test_compiled_dem_sampler_conformance.py tests/unit/test_compiled_dem_sampler_budget.py tests/unit/test_embedded_dem_sampler_source_lock.py examples/qec/compiled_dem_sampler.py
python -m mypy --follow-imports=silent --strict flagquantum/qec/execution/dem_sampling.py flagquantum/simulation/stabilizer/_dem_sampling.py tests/qec/test_compiled_dem_sampler.py tests/team/simulation/test_compiled_dem_sampler_conformance.py tests/unit/test_compiled_dem_sampler_budget.py tests/unit/test_embedded_dem_sampler_source_lock.py examples/qec/compiled_dem_sampler.py
FQ_BUILD_EMBEDDED_STIM=1 PIP_NO_INDEX=1 UV_OFFLINE=1 MAX_JOBS=2 python -m build --no-isolation --wheel --outdir "$EVIDENCE/dist"
PIP_NO_INDEX=1 UV_OFFLINE=1 python -m build --no-isolation --sdist --outdir "$EVIDENCE/dist"
python tools/ci_tier.py pr-default
python tools/ci_tier.py pr-runtime
python tools/public_api_snapshot.py
python tools/check_capability_maturity.py
python tools/docs_source_of_truth.py --check
python tools/parity_matrix.py --check
python tools/check_repository_hygiene.py
python -m pytest tests/api_contract/test_public_docstring_examples.py tests/unit/test_qec_installed_profiles.py -q -x
python -m examples.qec.compiled_dem_sampler
```

The implementation runner supplies the above timeouts and process-group cleanup; shell lines alone do not enforce budgets. Installed checks run outside the checkout in two fresh project environments with a locked existing dependency closure: no Stim SDK with embedded marker, and existing Stim 1.16.0 oracle/profile. Use `python -m pip install --no-deps --no-index <exact-wheel>` only with verified existing pip tooling; never let this resolve dependencies. Execute the extended `tools/verify_embedded_stabilizer_install.py` and `tools/qec_installed_smoke.py` as appropriate to that existing profile, then the new focused suites against the installed package. Save exact expanded commands and origin receipts. If the required tooling/cache is unavailable, stop before building.

Apply Ruff, Black and strict Mypy to every changed Python path, including the two installed-verification tools, existing installed-profile test and doctest list; the focused lines above identify the principal new implementation and tests, and do not exempt other changed Python files.

The complete path/ownership list is in `all-modification-paths.json`; its 32-path ownership check has passed under Integration. The final contract must record this proposal's hash and explicit approval, not a historical approval from another bundle. Do not change `setup.py`, dependency manifests, workflow YAML, historical source locks outside this extension, root API exports or existing frozen contracts.

## 9. Evidence already obtained for this proposal

- Both pinned archives rehashed; all 96 existing source files verified; two new header/template hashes and quoted-include closure verified. No compilation.
- Existing cached Stim 1.16.0/NumPy 2.3.4: 26 deterministic, packed/unpacked replay, six-format write and file-replay checks passed. This is upstream reference evidence, not execution of proposed FlagQuantum APIs.
- Pure arithmetic: 5,000 SIMD upper-bound comparisons and 6 exact guard-boundary cases passed, with no large shot table allocated.
- Proposed 32 paths passed the existing team-scope checker. No repository code/configuration has been modified.
- PR #1152 source main acceptance is retained separately: 345 passed/1 expected SDK-related skip; the byte-compatible file/batched-decode example passed using the existing native artifact. This is not a new wheel build/install.
- Actual merge-specific Main smoke run 43165099437 attempt 1 succeeded. Development-container publishing run 43165099441 was still in progress at the read-only check; no final conclusion is claimed for it.

## Release note target

Add reusable CPU sampling and error replay to owned independent detector error models using `model.compile_sampler()`, with packed results and six-format bounded file output. Existing `dem_sampling()` and correlated-model behavior remain unchanged. This release note is proposed and becomes delivered text only after accepted implementation.

## 10. Explicitly deferred

Full upstream text object/REPEAT/coordinate compatibility; error-model sampler methods outside this bundle; zero-detector models requiring changes to existing records; categorical/exclusive-group native sampling; every Stim NumPy constructor/protocol; Pauli/Tableau/FlipSimulator algebra; non-Clifford execution and conditional non-Pauli feedback; C++ API/ABI stability; platform admission beyond the existing embedded profile; benchmarking or release/scalability promotion. Keep each as a named gap, not an implied capability.

Approval requested: implement and publish this single bounded 32-path capability PR including the two locked upstream additions and the listed offline build/installed verification. It does not authorize merge, old CI recovery, dependency installation, runner changes or research.

## Appendix: exact modification paths

1. `flagquantum/qec/dem/model.py` (existing)
2. `flagquantum/qec/execution/dem_sampling.py` (new)
3. `flagquantum/simulation/stabilizer/_dem_sampling.py` (new)
4. `flagquantum/simulation/stabilizer/csrc/embedded_sampler.cpp` (existing)
5. `flagquantum/simulation/stabilizer/csrc/stim-sampler-source-lock.json` (existing)
6. `flagquantum/simulation/stabilizer/csrc/STIM_PROVENANCE.md` (existing)
7. `contracts/qec-dem-sampling-v1-candidate.json` (new)
8. `tests/qec/test_compiled_dem_sampler.py` (new)
9. `tests/team/simulation/test_compiled_dem_sampler_conformance.py` (new)
10. `tests/unit/test_embedded_dem_sampler_source_lock.py` (new)
11. `tests/unit/test_compiled_dem_sampler_budget.py` (new)
12. `tests/api_contract/test_public_docstring_examples.py` (existing)
13. `tools/qec_installed_smoke.py` (existing)
14. `examples/qec/compiled_dem_sampler.py` (new)
15. `examples/README.md` (existing)
16. `flagquantum/qec/dem/README.md` (existing)
17. `flagquantum/qec/execution/README.md` (existing)
18. `flagquantum/simulation/stabilizer/README.md` (existing)
19. `docs/reference/KNOWN_LIMITATIONS.md` (existing)
20. `capability-maturity.toml` (existing)
21. `docs/generated/CAPABILITIES.md` (existing)
22. `docs/development/QEC_COMPILED_DEM_SAMPLING_PROPOSAL.md` (new)
23. `docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md` (existing)
24. `docs/roadmap/FTQC_SIMULATION_MASTER_PLAN.md` (existing)
25. `docs/guides/STIM_USER_MIGRATION.md` (existing)
26. `docs/reference/API.md` (existing)
27. `contracts/cudaq-parity-matrix.toml` (existing)
28. `docs/reference/CUDAQ_PARITY_MATRIX.md` (existing)
29. `tools/verify_embedded_stabilizer_install.py` (existing)
30. `tests/unit/test_qec_installed_profiles.py` (existing)
31. `flagquantum/simulation/stabilizer/csrc/vendor/stim-1.16.0/src/stim/simulators/dem_sampler.h` (new, unchanged upstream bytes)
32. `flagquantum/simulation/stabilizer/csrc/vendor/stim-1.16.0/src/stim/simulators/dem_sampler.inl` (new, unchanged upstream bytes)

## Adoption record

The API owner explicitly approved the 32-path implementation and PR publication on 2026-11-12. The approved original proposal SHA-256 is `cfd7b071023e7658501ea2419bbc513d9efdeced495a548036a45a11631aca73`. This record does not authorize merge or change the original review evidence. Implementation acceptance remains pending.


## Implementation acceptance record (2026-11-12)

The approved implementation is now locally accepted on macOS arm64. Sections
1–9 above retain the reviewed proposal and its historical acceptance targets;
this dated record reports execution rather than rewriting that review evidence.
The original 32-path scope was explicitly expanded to 33 paths solely for
`tests/qec/test_restructure_compatibility.py`. Its historical fixture remains
unchanged; the new candidate contract registers the exact approved additive
signature. The 98 unchanged upstream files and 37 translation units are
preserved. No dependency installation or download, CI configuration change,
research task or merge was performed.

- Contract/fake/budget/source-lock: 30 passed; SDK differential and existing
  regressions: 199 passed. Strict Mypy, Ruff and Black passed for changed Python
  paths. The approved physical-owner doctest repair passed the complete two-test
  doctest file while preserving historical pickle identifiers and all entries.
- Offline wheel and sdist builds passed using the existing Torch 2.10.0 build
  environment. Runtime acceptance used existing Torch 2.14.1.
- Installed core: 79 passed, one expected SDK-only skip; installed SDK profile:
  80 passed. Package and extension origins were locked to each installation,
  using importlib test mode. The core installation has no Stim/PyMatching SDK.
  Real native calls in core were prepare_dem=57, sample_dem=32, write_dem=48;
  the SDK profile recorded 58, 33 and 48 respectively. A separate installed
  validator also checked independent bytes, replay and existing routes.
- Complete smoke/unit: 19,971 passed, 172 skipped; complete integration:
  7,459 passed, 279 skipped. Temporary native CPU extension
  copies were removed after inode and SHA-256 verification.

Wheel SHA-256: `44c8bb3a091bcd56c087739e8a52862f9c458855b0838d464c4c5f840249a167`.
Sdist SHA-256: `2ce0f055fb9632adaf4eec5fdfffca637229aea81cdb970c5fd4250de5bd51ab`.
Embedded extension SHA-256:
`f73de18a6d03080d39c0d3e828b7cf02eab0235f9aadca823f460fd7271781d0`.

Raw initial failures and rejected source-contaminated installed-test evidence
are retained separately from the accepted recovery results. Each recovery had
explicit API-owner authorization; no failed acceptance was silently retried.
Final-head user execution and PR identity are recorded in the PR description.
The approved compatibility recovery passed all 12 tests; its pre-existing
dataclass reflection line was typed through `getattr` without weakening the
frozen-state assertion. These results do not promote maturity, full parity,
performance, thresholds or scalability; merge remains separately authorized.
