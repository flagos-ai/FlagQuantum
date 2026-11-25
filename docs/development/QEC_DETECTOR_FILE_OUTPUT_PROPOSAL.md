# Native detector file output capability bundle

Status: API-owner approved implementation candidate; local acceptance passed on 2026-11-10; PR/main delivery pending. Prepared 2026-11-09,
Asia/Shanghai. Baseline main: c46d8580109cf6c11de588a39aab6a6260ffe9a5.
Primary ownership: QEC composition and Simulation numerical execution;
Integration owns the additive public contract and cross-domain acceptance.
Classification: single_device_fast_path. No research or performance claim.

## Verified gap and delivery state

PR1112 is merged at the baseline above. Its 231 CI jobs passed; exact-main wheel
build, installation and outside-checkout smoke passed run 42638677612, job
`129233581560`. This closes six measurement-file output formats, not detector-file
output. The existing CompiledDetectorSampler provides only sample(). Its private
stream and embedded bridge likewise have no detector-file writer. Read-only
comparison with the main Git tree verified eight relevant implementation/contract/
roadmap files and all 96 vendor files. The existing upstream closure already
contains sample_batch_detection_events_writing_results_to_disk and record writers.
No new upstream file or dependency is needed by the proposed design. Actual
compilation remains an acceptance gate, not a completed fact.

The older delivery-plan final paragraph still calls the measurement formats
pending. Implementation must replace that current-state wording with PR1112 and
its actual-main evidence while preserving historical failures. No recursive
progress-only PR is required. See READ_ONLY_GAP_AUDIT.json alongside this proposal.

## Additive public contract

Add only this method to the existing handle; no new root export or Circuit class:

```python
CompiledDetectorSampler.sample_write(
    self,
    shots: int,
    *,
    filepath: str | os.PathLike[str],
    format: Literal["01", "b8", "r8", "hits", "dets", "ptb64"] = "01",
    prepend_observables: bool = False,
    append_observables: bool = False,
    obs_out_filepath: str | os.PathLike[str] | None = None,
    obs_out_format: Literal["01", "b8", "r8", "hits", "dets", "ptb64"] = "01",
) -> None
```

Match the actual pinned Python binding's keyword order and defaults. The displayed
upstream textual @signature has a different ordering; inspect the binding itself.
Keep existing sample(), compile_detector_sampler(), DEM conversion, result fields,
exports, signatures, errors and refusal domains unchanged. An additive candidate
contract records the new method without rewriting the historical contract.

The main file contains D columns by default, O then D when prepended, or D then O
when appended. A separate observable file contains O columns from the SAME draws.
At most one of prepend_observables, append_observables and obs_out_filepath being
non-None may be selected. The upstream in-memory disk writer explicitly refuses
these combinations; reject them before RNG use regardless of internal batching.
Validate both format arguments even when the observable path is absent.

Dets format uses D and L indices in their respective domains, never measurement
M indices or flattened logical indices. Packed formats use LSB-first bits; the
combined layout packs across the detector/observable boundary. Preserve zero
padding, sparse empty shots and ptb64's 64-shot group layout. Require a positive
integer shot count, consistent with the existing file-writing entry; refuse bool.
Both selected ptb64 streams require a multiple of 64 before staging or sampling.
Existing in-memory detector sample(0) remains unchanged.

Invalid requests do not draw. Keep thread/process affinity, source snapshot,
noiseless-reference normalization and backend-failure poisoning. Unsupported T,
conditional H, multi-slot feedback and non-Pauli noise still fail before engine
preparation; preserve the accepted single-slot conditional-X route. There is no
fallback, approximation or new backend-selection policy.

## Source reuse and numerical boundary

SDK adapter: one pinned Stim 1.16.0 sample_write call. Embedded adapter: one private
bridge call to the existing upstream disk writer, with the handle's existing
circuit and RNG. Do not implement file output by sample(shots), per-shot Python
loops, or reconstructing dense detector/observable tables in Python. Preserve
same-call observable correlation and existing no-SDK installed execution.

All 96 vendor files remain byte-identical. Source lock SHA-256:
5bde304312555b474cea3286a8c81a5ebd239f24e93c0ed59c19acb0e2b8293b.
Upstream identity remains commit e2fc1eca7fd21684d433aa5f10f4504ea4860d07,
Stim 1.16.0. Reuse frame_simulator_util.h/.inl and measure_record_writer code.
Stop for scope review if a new file, translation unit, dependency or build setting
is required. Preserve license bytes, setup, workflow, runners and dependencies.

Reuse _measurement_file_validation.py for framing, byte bounds and publication
primitives. Extend its private validation only enough to represent typed D/L
sections and zero-width observable streams, preserving existing M-file behavior.
No new public reader, generic storage manager, codec registry or duplicate parser.

## Memory, files and lifecycle

Existing source bounds remain: 4096 qubits, 65536 explicit measurements, 100000
lowered operations, 65536 combined parity rows, 100000 parity references and the
existing deterministic-parity/analysis guards.

No Python full-shot table is permitted. Upstream chooses batches of at most
1024 shots and may use streaming scratch. Compute a padded principal-table bound
at that batch size using the current detector frame estimate, plus a padded
(D+O)-row concatenation table when requested; refuse over 64 MiB before writing.
Keep the existing in-memory sample limits unchanged. Bound parser working
storage to 1 MiB with 64 KiB reads and scalar state. These are structural guards,
not a process RSS guarantee or total scratch-space quota.

For S shots and a stream of N bits, use these conservative final byte bounds:
01 and r8: S*(N+1); b8: S*ceil(N/8); ptb64: (S/64)*N*8;
hits: S*max(1,N*(digits(max(0,N-1))+1)).
For dets, use S*(5 + sum(n*(digits(max(0,n-1))+2) for each D/L section)).
Zero-width 01/hits streams still need one newline per shot; binary zero-width
streams have zero payload. Empty dets shots retain the shot marker and newline.
Validate actual framing and sizes, including zero-width cases.

The SUM of both selected final-file bounds must not exceed 1 GiB. Group free-space
checks by filesystem so two files on the same device do not each claim the same
available bytes. These observations reserve no disk space. Staging may add I/O
and upstream anonymous temporary files; do not call this an aggregate disk quota.

Refuse existing targets, symlinks, identical/aliased targets and invalid parents
before drawing. Stage each file in its destination filesystem. Close and validate
ALL streams before publishing either. Publish with the existing no-replace link
behavior. If the second publication fails, remove only the first link still
identifiable as this call's own inode; never delete another actor's replacement.
Poison the handle on backend, validation or publication failure, retain the error,
and clean all owned staging. If cleanup itself fails, expose paths and both causes.
Two-file publication is sequential, NOT a cross-file atomic transaction; an
observer may briefly see the first complete file. No silent overwrite or false
all-or-nothing visibility promise. No durable-fsync guarantee is introduced.

## User workflow (proposed method; not executed or available yet)

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.execution.context import MeasurementMap

# Existing APIs: three occurrence columns survive reuse of classical slot 0.
source = (DynamicCircuit(1).x(0)
          .measure(0, classical_bit=0).reset(0)
          .measure(0, classical_bit=0).bit_flip(0, 1.0)
          .measure(0, classical_bit=0))
circuit = fq.Circuit.from_ir(source.to_ir())
sampler = circuit.compile_detector_sampler(
    detector_map=MeasurementMap(3, ((0, 1), (1, 2))),
    observable_map=MeasurementMap(3, ((2,),)), seed=7,
)
with TemporaryDirectory() as directory:
    output = Path(directory) / "events.b8"
    # Proposed method. Noiseless records are (1,0,0), actual records (1,0,1).
    sampler.sample_write(64, filepath=output, format="b8",
                         append_observables=True)
    assert output.read_bytes() == b"\x06" * 64  # D0=0, D1=1, L0=1
    print("detector_bytes", output.stat().st_size)
```

Expected output: detector_bytes 64. Dets combined witness is
b"shot D1 L0\n" per shot. Separate b8 outputs are b"\x02"*64 and b"\x01"*64.
Deterministic logical flips here validate encoding, not a logical-error-rate
estimate or fault-tolerance claim. Final-head example MUST actually run before
publication and be marked with its environment and evidence receipt.

## Exact proposed path scope (20 paths)

1. contracts/qec-detector-file-output-v1-candidate.json (new additive contract)
2. flagquantum/qec/execution/detector_sampling.py
3. flagquantum/simulation/stabilizer/_detector_sampling.py
4. flagquantum/simulation/stabilizer/_embedded_sampler.py
5. flagquantum/simulation/stabilizer/_measurement_file_validation.py
6. flagquantum/simulation/stabilizer/csrc/embedded_sampler.cpp
7. flagquantum/simulation/stabilizer/csrc/STIM_PROVENANCE.md
8. tests/qec/test_native_detector_execution.py
9. tests/team/simulation/test_native_detector_execution_conformance.py
10. tests/unit/test_qec_installed_profiles.py
11. tools/qec_installed_smoke.py
12. tools/verify_embedded_stabilizer_install.py
13. examples/qec/native_detector_sampling.py
14. docs/development/QEC_DETECTOR_FILE_OUTPUT_PROPOSAL.md (new)
15. docs/guides/STIM_USER_MIGRATION.md
16. docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md
17. docs/roadmap/FTQC_SIMULATION_MASTER_PLAN.md
18. flagquantum/qec/execution/README.md
19. flagquantum/simulation/stabilizer/README.md
20. tests/unit/test_compiled_measurement_sampler_contract.py

The last path protects existing measurement formats against private-validator
regressions. Existing CompiledDetectorSampler doctest registration is reused;
no additional docstring-runner edit is planned. Contract files and cross-domain
changes belong to Integration; numerical work remains Simulation-owned behind
the existing stream boundary. Scope-check every path before edits.

## Acceptance commands and pass conditions

Use the existing qualified Python/tool environment; install/download nothing
unless separately approved. Preserve originals and give every command an explicit
subprocess timeout, terminate its process group on timeout, and stop on first
failure. Commands below are future acceptance, not claimed executed results.

```text
python tools/check_team_scope.py --team integration --files <the 20 paths above>
python -m pytest tests/qec/test_native_detector_execution.py -q
python -m pytest tests/team/simulation/test_native_detector_execution_conformance.py tests/unit/test_compiled_measurement_sampler_contract.py tests/unit/test_qec_installed_profiles.py -q
python -m pytest tests/api_contract/test_public_docstring_examples.py -q
python -m ruff check <changed Python paths>
python -m black --check <changed Python paths>
python -m mypy --strict <changed Python paths>
python tools/ci_tier.py pr-default
python tools/ci_tier.py pr-runtime
FQ_BUILD_EMBEDDED_STIM=1 python -m build --wheel --no-isolation
```

Budgets: scope/static checks 120 seconds each; focused tests 300 seconds each;
conformance/profile tests 600 seconds; doctests 300 seconds; seeded default/runtime
tiers 1800 seconds each; offline macOS arm64 wheel build 900 seconds. Use phase
heartbeats for long commands; these are bounded software acceptance tests, not
scientific studies. Reuse passed unaffected evidence rather than repeating it.

In a disposable qualified prefix, install the new wheel from the local wheelhouse
only, retain exact wheel/runtime hashes, unset PYTHONPATH, leave the source tree,
and run each with a 120-second timeout:

```text
python <absolute-source-path>/tools/verify_embedded_stabilizer_install.py
python <absolute-source-path>/tools/qec_installed_smoke.py core
```

Require no Stim Python distribution or import spec, installed module and extension
provenance, actual new native writer-call evidence and deterministic byte witnesses.
Run the pinned SDK replacement/conformance profile separately using already
available SDK dependencies; do not substitute an overlay for installed-wheel
proof. If a qualified profile or build dependency is unavailable, report the exact
missing resource and stop rather than installing automatically.

Test all six formats and four output choices (detector only, prepend, append,
separate files), including mixed output formats, correlated observable bits,
9-bit boundaries, zero observables, sparse empty shots, multi-digit D/L indices,
r8 254/255/256 runs, ptb64 64/128 shots and 63/65 refusal. Independent fixed-fault
reference plus literal bytes must accompany SDK comparisons. Verify default-state
reference normalization (raw 1 can normalize to 0), occurrence versus classical
slot identity, inverted/latest-slot X feedback, T/conditional-H early refusal,
unchanged DEM and converter behavior. Exercise malformed engine files, fail-on-
second-publication, concurrent target creation, path aliases, cleanup failures,
thread/process ownership, unchanged RNG after preflight refusal and handle
poisoning after backend failure. Do not demand byte-identical random sequences
across batch schedules or architectures. Compare stochastic same-call correlation
with independent invariants, not brittle cross-seed stream equality.

After approval: branch from current main; stop and review if concurrent work
changes these boundaries. Bind runnable example and receipts to the final head,
create one direct-main PR, and follow automatic checks without restoring failed
CI. Merge only an accepted reviewed head under the standing delivery instruction,
then record exact-main installed-wheel smoke and update both progress ledgers.
No GPU recovery, runner/configuration change or research task is authorized here.

## Overall progress and follow-ups

PR1112 completes the measurement-file output slice. This proposed bundle advances
FTS-04/R03/R08 detector sampling and bounded output evidence; all three overall
requirements remain partial. It does not close FTS-02/R04/M2, mixed non-Clifford
execution, general fault-tolerant compilation, threshold/performance/capacity
claims or full Stim/PyMatching API parity. FTS-07 model/decoder evidence is reused,
not relabelled complete by adding file output.

Next distinct bundles: file measurement-to-detector conversion; public bounded
record reading and streaming decode composition; remaining pinned Pauli/tableau,
circuit-analysis and PyMatching operations from the inventory. Inspect current
implementations and independent acceptance before choosing each; no API-count
completion percentage is inferred from this slice.


## Local candidate acceptance — 2026-11-10

The approved 20-path candidate passed 229 focused tests with 3 skips, both public
doctest checks, Ruff, Black and strict Mypy for the 11 changed Python files.
Complete node accounting covers 18,992 smoke/unit passes with 295 runtime skips,
and 7,025 integration passes with 227 runtime skips. These totals reuse verified
terminal results and bounded continuation batches; they are not a claim that
one uninterrupted full run passed. There are no missing nodes or unrecovered
failures. The original missing-import failure, environment failures and both
integration timeout logs remain retained. The isolated Shor and Vlasov cases
passed without changing their source, assertions or numerical thresholds.

The offline macOS arm64 wheel built and installed successfully. Outside-checkout
core verification confirmed that neither Stim nor PyMatching was installed,
and observed actual embedded calls for all six detector-file formats and
same-shot observable output. The installed native format-layout selection
passed 24 tests. T and conditional H refused before engine preparation, while
the repeated-slot conditional-X path remained accepted. All 96 vendor files
match the unchanged source lock.

Wheel SHA-256:
`4b848afd613637df700dedec809e203ce0224066ab0044364738a79617355012`.
Embedded extension SHA-256:
`68593f5c95c3861536068f3b24f0f92be1674140d64bf86755a8a1d3514883dd`.
These are local candidate acceptance results, not PR CI, actual-main, GPU,
performance or complete upstream-parity evidence. Final-head user-example
receipts and subsequent PR/main delivery are recorded separately.
