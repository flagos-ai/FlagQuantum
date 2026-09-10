# Maintainability cleanup — 2026-09-09

This change fixes a data-validation bug and simplifies existing execution and
verification code. It adds no public API, dependency, or capability claim.

## Changes

- QEC binary records validate values before conversion. Integer zeros and ones,
  including booleans, are accepted; floats and strings are rejected. Previously,
  `(0.2, 1.8)` silently became `(0, 1)`. Regression tests cover syndrome records
  and Pauli-frame readout.
- The root execution facade identifies `ExecutionPlan` by its actual type rather
  than the presence of an `identity` attribute.
- `run_native` opens its configuration and optional operator-backend contexts
  once, then calls the internal execution body. Recursive re-entry and the
  `_runtime_config_active` / `_operator_backend_active` flags were removed.
  Tests check cleanup after successful execution and exceptions.
- CI paths follow the current Runtime, Ecosystem, and Simulation layout and the
  existing local typing targets. Package-policy tests now check referenced paths
  and imported modules rather than step titles.
- Tests pinning AGENTS.md prose and command ordering were removed. Executable
  tier-policy tests remain in `tests/unit/test_ci_tier_policy.py`.

## Remaining work

The generic QEC names `Correction` and `PauliFrame` still encode repetition-code
restrictions. Their migration is deferred because streaming-feedback work is
changing their callers concurrently. Rename the code-specific records and update
the private candidate contract together after that work is integrated; do not
introduce aliases just to retain the misleading names.

Executing the repaired CI typing commands exposed 11 errors in untouched files:

- `runtime/module.py`: one `Any` return in the checkpoint path;
- `deployment/cloud.py`: five mapping/optional-value typing errors;
- `runtime/execution_plan_contract.py`: four imported-base/return typing errors;
- `ecosystem/contracts.py`: one imported exception-base typing error.

The foundation and strict runtime-contract typing commands pass. The remaining
commands still fail and have not been weakened or suppressed. This cleanup does
not establish that the full CI or release pipeline passes.

No cloud jobs, QPU tasks, or accelerator benchmarks were run. Concurrent QEC and
dynamic-feedback changes are outside this cleanup except for binary validation.

## Verification

- Focused regression suite: 99 passed.
- Smoke/unit suite: 1337 passed, 14 skipped.
- Local integration suite: 320 passed, 37 skipped. Local socket access was
  enabled for the single-rank distributed scenarios.
- Architecture, dependency policy, public API snapshot, and lint/format checks
  on the edited execution and test files passed.
- The broad CPU distributed/evidence selection did not pass: 90 failed, 356
  passed, 28 skipped. A focused reproduction confirms that the JAX execution
  tests invoke an unavailable optional JAX installation.
- Excluding `test_jax_distributed_plan.py` and `test_hybrid_jax.py` still produced
  42 failures, 305 passes, and 22 skips. These results were later found to mix
  source trees: the parent used vNext, but the environment's editable install
  pointed worker processes at `FlagQuantum-qpu-digital-twin`. The import and
  missing-field failures from this run are not valid evidence of vNext defects.

## Retest with JAX and consistent worker imports

JAX and jaxlib 0.10.2 were installed in the test environment. CPU JIT and
automatic differentiation passed. Installing JAX alone produced 404 passes,
42 failures, and 28 skips because worker imports still resolved to the other
checkout. A standalone script reproduced the wrong import; setting
`PYTHONPATH` to the vNext repository root corrected it without modifying the
shared environment's editable installation.

With that environment inherited by all workers, the selection
`distributed_cpu or benchmark_contract or release_gate` completed with
**445 passed, 1 failed, 28 skipped** in 140.33 seconds.

The remaining failure is
`test_abrupt_rank_loss_terminates_elastic_job_with_diagnostics`. The job exits
with failure within the deadline and reports the injected SIGKILL, the failed
rank, `ChildFailedError`, and SIGTERM of its peer. The test nevertheless requires
either the application's `peer_process_loss_detected` event or the literal
`ProcessGroup` in launcher output; neither appears in this macOS/PyTorch run.
This diagnostic-acceptance failure remains unresolved. No test assertion was
weakened for this retest, and the complete tier is not reported as passing.

These runs used Python 3.12.14 and PyTorch 2.13.0 against a working tree with
concurrent changes. They do not certify an immutable release commit or the
supported-version installation matrix.

## Repository-wide remediation in progress

The scope subsequently expanded to the repository-wide English and Python quality
requirements in `PYTHON_ENGINEERING_STANDARD.md`. This work is not complete.

- Added a repository language checker and CI/pre-commit enforcement. It checks
  tracked and nonignored untracked UTF-8 content, filenames, and decoded JSON or
  notebook strings. There are no grandfathered language exemptions.
- The text scan now reports zero files with Han-script content across 1,892
  checked paths, down from 88 affected files at the start of this round.
- Visual review covered 38 raster assets and PDF pages. Three MPS capacity PNGs
  contained Chinese labels; their English replacements were visually checked
  against the originals, including milestone values, timing, OOM history,
  truncation error, and the FP32-relative discarded-weight annotation.
- Translated architecture decisions, contract proposals, inventories, guides,
  and 22 SVGs while preserving approval histories and numerical evidence. XML
  comparison verified unchanged plot geometry in all 21 benchmark SVGs. The
  reference applicability diagram also received label-layout and contrast fixes.
- Annotated all five Drawer modules. Drawing now copies caller-owned wire orders
  and label options rather than mutating them. The existing dynamic device and
  Matplotlib option boundaries still use `Any`; this is not a claim of completely
  typed third-party inputs.
- Added a Drawer domain README and tests for caller input ownership. Five focused
  tests passed with Matplotlib available, including real Agg rendering.
- Annotated the IR iterator and declared the dataclass field contract used by the
  numerical serialization mixin. Twenty-two related tests passed.

Noise now validates restored correlated readout models through the same wire-count
check as direct construction. Its shared decoder produces explicit two-by-two
readout matrices, and Kraus validation avoids redundant CPU conversion. Sixty-five
noise tests passed, all four Noise modules pass strict checking, and the README's
round-trip example executed successfully.

Hamiltonian construction now normalizes wire iterables once. A regression test
first reproduced failure with a generator, then passed with the fix. Algorithm
callback binding now uses explicit partial application and a resolved energy
function. Twenty-six algorithm and CPU integration tests passed. Algorithms still
has full-audit typing errors; isolated module checking is not full-domain proof.

The earlier 11-error report covered only CI's selected typing targets. A full
strict check with Python 3.12 reported 1,058 diagnostics initially and 921 error diagnostics in the
latest run. Concurrent changes affect these totals; the difference cannot all be
attributed to this cleanup. Drawer passes strict checking, as do the selected
Core numerical/IR modules when checked with their exception definitions. Full
repository strict typing still fails and has not been hidden with suppressions.

The installed NumPy stubs require Python 3.12 syntax, so the repository's default
Python 3.10 mypy setting cannot parse this particular environment. The full audit
used `--python-version 3.12`; it does not establish Python 3.10 compatibility.

The shared virtual environment contains an editable import finder for another
FlagQuantum checkout. Even with `PYTHONPATH` pinned, that finder can synthesize
otherwise absent submodules from the other tree. Reliable parent-process checks
therefore use Python `-S` and explicitly append the virtual environment's
site-packages directory. `PYTHONPATH` remains pinned to vNext for worker imports.
The shared editable installation was not changed.

With this isolation, the smoke/unit run completed with **1,355 passed,
13 skipped, and 1,422 deselected**. A subsequent smoke/unit/integration selection
completed with **1,695 passed, 36 skipped, 3 failed, and 1,061 deselected**. All
three failures occurred at local socket binding under the sandbox; rerunning
those exact cases with local socket access produced **3 passed**. Together these
runs cover 1,698 passing selected tests, before the later Hamiltonian changes;
the 26 focused algorithm tests validate those later changes and overlap this
selection. The optional Matplotlib ownership test was
also exercised separately with its dependency available. These results apply to
a changing working tree, not an immutable release commit. The distributed
failure documented above remains unresolved, and full CI/release readiness has
not been established.

Earlier documentation checks passed architecture boundaries, the public API
migration baseline, documentation source-of-truth validation, and whitespace
checks. Historical hashes and checkbox counts were compared with HEAD across
changed Markdown files; no differences remain. All remaining architecture and
roadmap documents have since been translated. Historical hashes and acceptance
checkbox counts remain unchanged, and the final IR architecture translation
preserved all 41 original non-diagram fenced blocks exactly.

## Latest continuation

- Smoke/unit verification: **1,356 passed, 13 skipped, 1,432 deselected** in
  26.90 seconds, using isolated imports from this worktree.
- Deployment cloud handling now reuses validated target options, declares its
  metadata mapping, and separates scalar coefficients from their tensor form.
  Thirty-six focused deployment/provider/routing tests passed; `cloud.py` also
  passed strict checking with its dependencies.
- A refreshed full strict audit reported **915 errors in 160 files**, before
  the subsequent distributed identity and evidence-validation edits. This is
  still a failing audit, not a completed typing migration.
- Distributed identity construction no longer reuses one local variable for a
  mutable list and an immutable tuple. Transport evidence checks raise the same
  domain exception directly at list-validation boundaries, allowing static
  narrowing without casts or suppressed errors. Audit memory helpers explicitly
  exclude `None` before iteration.
- These three modules passed strict typing and Ruff. Thirty-six focused
  identity/transport tests passed, followed by 281 scalability-audit and
  benchmark-contract tests. These selections overlap; their counts are not
  reported as unique tests.
- After those edits, the full strict audit reports **906 errors in 157 files**
  across 416 checked source files. No typing settings were relaxed.

The three raster replacements used the built-in image generation tool with
text-localization prompts: translate every Chinese label to English, preserve
the original layout and all numerical evidence, and retain the MPS versus full
statevector limitation. The final assets replace these paths:

- `assets/readme/mps-distributed-capacity-complex64.png`
- `assets/readme/mps-distributed-capacity-complex64-131072.png`
- `assets/readme/mps-distributed-capacity-complex64-131072-fp32-relative.png`

No benchmark was rerun or promoted by these image edits. The earlier unresolved
distributed diagnostic test and full strict typing remain outstanding.

## Benchmark entry-point cleanup

Six benchmark modules now choose package or sibling imports from their execution
context, instead of catching every `ImportError`. Package dependency failures
therefore retain their original cause. Static checking resolves the same package
implementation through `TYPE_CHECKING`; the direct execution branch remains
covered by subprocess tests rather than being treated as unverified fallback.

The strong-, weak-, and training-scaling adapters previously imported their
report builder relatively before reaching the direct-script fallback. Their
report imports now follow the same context decision. Six regression cases invoke
the scripts with `python -S ... --help`, checking standalone argument parsing
without relying on an installed package. Existing CLI tests now have return and
fixture annotations and bounded subprocess timeouts.

Verification: 29 focused CLI/contract/report tests passed. Ruff, Black on the
seven edited Python files, architecture boundaries, dependency policy, and the
language scan passed (1,893 paths, zero Han-script text files).

The full strict audit after these source edits reports **895 errors in 151
files** across 416 checked source files. This is still a failing audit; no ignore
rules or weakened checks were added.

## Timing-sample validation

Strong-scaling reports and their shared bootstrap estimator now reject zero,
negative, NaN, and infinite durations with `ValueError` before aggregation or
resampling. The report validates even a baseline-only input, which never enters
the bootstrap estimator. Previously these inputs could contaminate derived
statistics or fail later during arithmetic.

Twenty regression cases first failed against the old implementation. They cover
both bootstrap inputs and single-/multi-artifact reports. After the fix, all
158 benchmark-contract tests passed. Existing valid seeded results and artifact
schemas remain unchanged; no raw benchmark evidence was edited.

The smoke/unit rerun completed with **1,359 passed, 13 skipped, and 1,458
deselected** in 23.78 seconds against the current working tree. Ruff, Black,
whitespace checks, and the repository language check also passed. This run does
not resolve the previously recorded distributed diagnostic failure or establish
full strict typing compliance.

## Correctness-metric validation

Strong-scaling aggregation now rejects nonfinite norms and observable values,
and validates every source's absolute tolerance as finite and nonnegative before
taking their maximum. Previously NaN differences bypassed the greater-than
comparison, and invalid individual tolerances could be hidden by aggregation.
The existing finite-value comparison and output schema are unchanged.

Twenty new invalid-input cases produced 12 failures and eight passes before the
fix. Afterwards all 178 benchmark-contract tests passed. Two additional boundary
cases verify that exact agreement at zero tolerance and a difference equal to
the allowed tolerance remain accepted; the resulting 46-test strong-scaling
report suite passed. Ruff and Black passed on the edited Python files.

## Typed estimates and explicit audit delegation

The bootstrap estimator now describes its returned fields with private
`TypedDict` records, including numeric speedup and confidence fields and the
resampling seed/count. Baseline estimates share the common record. The runtime
dictionary and serialized fields are unchanged. The estimator also documents
its input requirements, return values, and exceptions.

Audit policy delegation no longer goes through a generic string-based
`getattr` dispatcher. Its three existing wrappers import their concrete policy
functions lazily, retaining cycle-safe loading and function return types.

Verification: 68 report/CLI tests passed, followed by 323 scalability-audit and
benchmark-contract tests. Ruff, Black, architecture boundaries, the public API
baseline, and the English text check passed. The full strict audit now reports
**891 errors in 150 files**. The estimator's original diagnostic is resolved,
but an isolated check still follows its package into an existing registry
`Any` return error; it is not reported as a clean dependency-inclusive check.

## Distributed capacity and workload validation

Capacity topology checks now explicitly narrow world size and rank-list inputs.
Non-object rank entries raise `FlagOSCapacityProfileError` before dictionary
access; four regression cases cover null, integer, string, and list entries.
Validation remains at the existing profile acceptance boundary.

Workload evidence checks likewise narrow environment mappings, collective and
statevector lists, and required error strings directly. Existing domain errors
and acceptance conditions are preserved without casts or type suppressions.

Twenty-one focused tests passed, followed by 193 capacity and benchmark-contract
tests after the final edit. Ruff, architecture, English-language, and whitespace
checks passed. The full strict audit reports **877 errors in 148 files**; the
two edited evidence modules no longer contribute diagnostics to that audit.
These are CPU contract checks, not new accelerator evidence.

## Remove shadowed audit implementations

Removed 32 function definitions from the audit engine that were immediately
replaced by imports from `validation_helpers` and `statistics`. Runtime identity
checks before and after deletion confirmed that every name still resolves to
the same canonical implementation. MPS readiness and release policy now import
those helpers directly from their defining modules rather than through the
engine's incidental namespace.

Removed seven obsolete typing suppressions from the edited audit modules.
Normalized memory and communication helpers now declare their actual dictionary
return type. Release communication validation explicitly narrows its mapping
before accessing boundary evidence; its acceptance conditions remain unchanged.

The 323-test scalability-audit/benchmark-contract selection passed. Ruff and
Black passed for the edited audit code. The full strict audit reports **865
errors in 145 files**, with no relaxed typing settings.

Smoke/unit verification passed with **1,363 passed, 13 skipped, and 1,480
deselected** in 23.90 seconds. Architecture, public API baseline, and English
text checks passed. Full strict typing and the previously recorded distributed
diagnostic test remain unresolved.

## Explicit audit package exports

The audit package now imports scalability classification and MPS readiness
functions from their defining modules, instead of retrieving them through the
engine's dynamic attribute lookup. Existing exported names and function
identities are preserved. Payload attachment helpers likewise import their
evaluators directly and now describe their copy-and-attach behavior.

The 323-test scalability-audit/benchmark-contract selection, Ruff, and Black
passed. The full strict audit reports **860 errors in 142 files**, including
resolution of the indirect audit-call diagnostics in the JAX release adapter.
Architecture, public API baseline, and English-language checks passed.

The subsequent smoke/unit run passed with **1,363 passed, 13 skipped, and
1,480 deselected** in 23.70 seconds. All three changed package imports were
checked again for canonical function identity after the edit.

## Simulation helper imports

Tensor-network observables and entry points now import the greedy contraction
routine directly from path search. Local statevector execution imports its gate
matrix builder directly from the shared gate-matrix module. The selected
functions, numerical algorithms, and public interfaces are unchanged.

The focused numerical tests produced 150 passes, one skip, and three sandbox
failures at localhost socket binding. Those exact three single-rank CPU cases
passed when local socket access was available. Thus this selection has 153
passing cases and one skip; it is not remote or accelerator evidence.

Ruff, architecture boundaries, public API baseline, and English text checks
passed. The full strict audit reports **857 errors in 141 files**.

Black and whitespace checks also passed. The smoke/unit run completed with
**1,363 passed, 13 skipped, and 1,480 deselected** in 23.67 seconds. The full
typing migration and the earlier abrupt-rank-loss diagnostic failure remain
open.

## Gate-matrix typing and precision selection

The gate matrix registry now distinguishes fixed tensors from functions taking
and returning a tensor. Matrix-assembly helpers declare their tensor inputs.
Complex/real dtype selection uses explicit PyTorch dtype objects for the two
precision modes already validated by `RuntimeConfig`, replacing dynamic
attribute lookup.

The complex-conversion helper retains its existing broad PyTorch input boundary
but keeps the converted tensor in a separate, inferred tensor variable. Removed
a redundant union containing `Any`; this does not restrict or widen the actual
accepted inputs.

Eighty operator-schema, numerical-property, numerical-contract, and MPS tests
passed. Explicit checks covered both configured precision modes and conversion
from a scalar list. The full strict audit reports **849 errors in 138 files**;
the matrix module and shared gate-matrix builder no longer contribute errors
to that audit.

After formatting, smoke/unit verification passed with **1,363 passed, 13
skipped, and 1,480 deselected** in 23.54 seconds. Architecture, public API
baseline, Ruff, and English text checks passed. No numerical evidence or
capability claims were changed.

## TEBD local Hamiltonian assembly

Local TEBD terms now validate that their Pauli matrix is a fixed tensor before
device/dtype conversion. Accumulation no longer eagerly allocates `zeros_like`
for every dictionary lookup; the first term supplies the accumulator and later
terms are added without modifying either operand in place.

A numerical test independently constructs X, Z, and ZZ matrices and checks
aggregation of repeated and distinct terms on the same sites. All 52 TEBD/MPS
tests passed, including convergence against an exact ground-state energy and
the prohibition on dense state materialization. No speedup claim is made.

Ruff, Black, architecture boundaries, public API baseline, and English text
checks passed. The full strict audit reports **848 errors in 137 files**.

Smoke/unit verification passed with **1,363 passed, 13 skipped, and 1,481
deselected** in 23.82 seconds. Whitespace checks passed. Full strict typing and
the previously recorded distributed diagnostic failure remain open.

## Tensor-network model delegation

Removed twelve variadic contraction wrappers and their generic string-based
dispatcher from tensor-network models. The 27 consuming methods now import the
concrete path-search or contraction functions lazily, preserving cycle-safe
loading while allowing their parameters and results to be checked statically.
The existing public model methods and contraction strategies are unchanged.

Z-observable construction now verifies its fixed gate tensor before converting
device and dtype. This narrows the fixed/parameterized registry boundary without
casts. The initial path-search, stage, and tensor-network test selection passed
with 78 passes and one optional skip before this final guard was added.

After the guard, the same numerical selection again passed (78 passed, one
skipped), followed by **1,363 passed, 13 skipped, and 1,481 deselected** in the
23.25-second smoke/unit run. Ruff, Black, architecture, public API, and English
text checks passed. The full strict audit reports **815 errors in 136 files**;
the tensor-network model module no longer contributes diagnostics to it.

## Slicing budget and reduction state

Tensor-network slicing now resolves byte limits inside the branch where the
limit is present, declares variable-length slice labels explicitly, and declares
the result accumulator before either execution branch. Existing budget errors,
planning choices, and summation order are preserved. Execution of a prebuilt
slicing plan now checks compensation availability consistently with the other
compensated-reduction path.

The initial tensor-network/path/stage selection passed with 78 passes and one
optional skip. The full strict audit after those edits reported **808 errors in
136 files**; dry-run tensor representation remains a separate typing issue in
the contraction module.

After the final budget-branch cleanup, 12 focused slicing/budget tests passed
and one optional case skipped. The strict error count remained 808. Smoke/unit
verification passed with **1,363 passed, 13 skipped, and 1,481 deselected** in
23.13 seconds. Ruff, Black, architecture, public API, and English text checks
passed.

## Validate cached tensor-network programs

Local tensor-network construction now verifies that an occupied compiled-program
cache entry is a `CompiledTNProgram`. Previously a malformed entry was ignored
on CPU and could fail indirectly at binding on CUDA. The existing CPU rebuild
and CUDA template-reuse policy is unchanged for valid entries.

A regression test first demonstrated that an invalid cache entry was silently
accepted, then passed after the check. The tensor-network/path/stage selection
completed with 79 passes and one optional skip. Ruff, Black, architecture,
public API baseline, and English text checks passed. The full strict audit now
reports **807 errors in 135 files**; local tensor-network construction no longer
contributes an untyped return diagnostic.

Smoke/unit verification passed with **1,363 passed, 13 skipped, and 1,482
deselected** in 23.48 seconds. Whitespace checks passed. No GPU execution or
release-capability verification was performed in this round.

## Observable scalar protocol

The private real-scalar parser now returns `float | None`; the observable
multiplication method translates the unsupported case into `NotImplemented` at
the Python operator boundary. This preserves operand dispatch and exceptions
while removing the uninformative `float | object` return annotation.

Eight additional cases protect unsupported-operand handling and rejection of
nonfinite coefficients. All 14 observable/output contract tests passed,
including numerical expectations and autograd. The full strict audit reports
**805 errors in 135 files**. Optional-wire narrowing is still unresolved; its
existing type-specific behavior has not been changed to silence that error.

Ruff, architecture, public API baseline, and English text checks passed.

After formatting, smoke/unit verification passed with **1,371 passed, 13
skipped, and 1,482 deselected** in 23.69 seconds. Whitespace checks passed.

## Heisenberg observable scan

The rank-local Heisenberg scan validates fixed Pauli tensors and reuses converted
operators for each device/dtype pair within one call. This replaces repeated
per-site registry conversion with a call-local cache; no mutable global cache
or numerical fallback was introduced.

Two Bell-state tests cover complex64 and complex128, checking XX=1, YY=-1,
ZZ=1 through the energy and its coefficient gradients. All 42 focused MPS tests
passed. Ruff, formatting, architecture, public API baseline, and English text
checks passed. The full strict audit reports **804 errors in 134 files**.
No accelerator benchmark or quantified performance claim was added.

Smoke/unit verification passed with **1,373 passed, 13 skipped, and 1,482
deselected** in 23.39 seconds. The run emitted one existing PyTorch warning
about complex modules. No remote GPU or QPU jobs were submitted.

## Temporal decoder history typing

The temporal repetition decoder now explicitly types its previous-syndrome
tuple to match `SyndromeRound.bits`. Runtime validation still requires two
bits; correction selection and detection-event validation are unchanged.
This is a local QEC workflow annotation change, not a capability expansion.

All 82 QEC tests passed. The full strict audit reports **803 errors in 133
files**; the decoder module no longer reports an error. Architecture and public
API baseline checks passed. The smoke/unit run produced 1,371 passes, 13 skips,
and two repository-check failures because the system Git executable was
unavailable. Both failed tests passed when rerun with the bundled Git on PATH.
The English scan also passed with that environment: 1,893 paths and zero files
containing Han-script text. No tests or typing checks were suppressed.

## Repetition decoder reuse

Compiled-feedback evidence now reuses the decoder-owned syndrome-to-wire
lookup instead of maintaining a second copy of the four-entry table. The
selected runtime feedback decoder is annotated with `StreamingDecoder`, so
custom implementations are represented by the existing protocol rather than
an inferred union of the two built-in classes. Public signatures are unchanged.

All 82 QEC tests passed, including custom-decoder replacement and physical/frame
feedback scenarios. Smoke/unit verification passed with **1,373 passed, 13
skipped, and 1,482 deselected** in 23.32 seconds. Ruff, architecture, public API,
and English text checks passed. The full strict audit reports **801 errors in
133 files**. Final three-bit readout tuple narrowing remains unresolved; no
unchecked cast was introduced. This local workflow change makes no new
fault-tolerance or hardware capability claim.

## Repetition readout shape

The memory workflow explicitly unpacks its three data-wire readouts before
constructing the final tuple. This gives both raw and frame-adjusted records
the fixed three-element type required by the existing result model, without
casts or a new wrapper type. Ancilla readouts remain excluded. A truncated
internal sample raises `ValueError` during unpacking instead of reaching the
downstream frame or shot-record length check.

All 82 QEC tests passed, including the existing physical, offline, and runtime
Pauli-frame correction scenarios. Smoke/unit verification passed with **1,373
passed, 13 skipped, and 1,482 deselected** in 23.32 seconds. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The full strict
audit reports **799 errors in 132 files**, with no errors reported under
`flagquantum/qec`. This is not a claim that the full repository is type-clean.

## Capability comparison return values

Private capability comparison helpers now explicitly normalize equality and
ordered comparisons to `bool`, matching their existing return contracts.
JSON value validation, recursive coverage, requirement conflict detection,
and public schemas are unchanged. This is a Core predicate cleanup; it does
not change execution distribution or promote any target capability.

All 110 focused Core/Runtime capability tests passed. Smoke/unit verification
passed with **1,373 passed, 13 skipped, and 1,482 deselected** in 23.62 seconds.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The full strict audit reports **795 errors in 131 files**, with no
errors reported for `core/target_capabilities.py`. The dynamic JSON value
representation still uses `Any`; this change does not claim to fully type that
representation.

## Operator selection iterables

The private FlagGems operator parser now accepts `Iterable[str]`, matching the
existing public entry points. Iterator inputs are consumed through a generator
instead of an intermediate list; alias normalization and ordered deduplication
are unchanged. A scenario test supplies a one-shot iterator containing aliases,
duplicates, whitespace, and an empty name, then verifies the selected operators.

All 10 operator-backend tests passed. Smoke/unit verification passed with
**1,373 passed, 13 skipped, and 1,483 deselected** in 23.35 seconds. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The full strict audit reports **792 errors in 131 files**. No real FlagGems
accelerator validation or performance benchmark was performed.

The CPU probe's frozen implementation versus mutable protocol attributes remains
unresolved. Neither its protocol nor its immutability was changed merely to
remove that diagnostic.

## Execution option overlays

The resolver reads validated scalar fields directly instead of recursively
converting each overlay with `dataclasses.asdict`. This removes an unnecessary
intermediate dictionary and deep-copy traversal from local option resolution.
Overlay order, non-`None` selection, explicit false values, and field provenance
are preserved. No quantified speedup is claimed.

All 30 focused execution-option tests passed. Smoke/unit verification passed
with **1,373 passed, 13 skipped, and 1,483 deselected** in 23.58 seconds. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit remains at **792 errors in 131 files**. In particular, the
resolver's dynamically keyed final construction still has four type errors;
this cleanup does not suppress or resolve them.

## Statevector fusion locals

Single-device gate fusion now relies on dictionary insertion order instead of
maintaining duplicate wire-order lists. Instruction groups and compiled gate
groups have distinct local names, and the final CX pass no longer reuses
variables inferred for the earlier pass. Gate order, fusion eligibility,
dependency-reordering evidence, and numerical kernels are unchanged.

All 73 selected local numerical, native-circuit, and operator-profile tests
passed; 17 distributed cases were excluded from this focused local run.
Smoke/unit verification passed with **1,373 passed, 13 skipped, and 1,483
deselected** in 23.31 seconds. Ruff, architecture, public API baseline, English
text, and whitespace checks passed. The full strict audit reports **784 errors
in 131 files**. No accelerator or distributed scalability evidence was added.

## Fixed matrices in Pauli products

Dense Pauli-product construction and statevector expectation now verify that
registry entries are fixed tensors before converting device and dtype. Passing
a parameterized gate such as RX now raises a descriptive `ValueError` instead
of an incidental attribute error on a callable. Valid numerical paths and
public signatures are unchanged.

All seven Pauli tests passed, including new RX/RY/RZ rejection cases and
analytic Z-expectation gradients for statevector and density-matrix inputs.
Smoke/unit verification passed with **1,373 passed, 13 skipped, and 1,488
deselected** in 23.30 seconds. Ruff, architecture, public API baseline, English
text, and whitespace checks passed. Black's initial multi-file invocation
failed because multiprocessing could not reload the stdin launcher; individual
file checks then passed without edits. The full strict audit reports **782
errors in 130 files**, with no errors in `simulation/pauli.py`.

## Tensor-network observable construction

Four observable construction paths now share one fixed-matrix conversion helper
that validates the registry entry before device/dtype conversion. Hamiltonian
wire sets are explicitly typed, and missing per-wire factors use an explicit
identity key. Batched Hamiltonian products build the term's wire dictionary
once per term instead of rebuilding it for every wire. Existing unknown-name
handling, contractions, MPO compression, and gradient algorithms are unchanged.

Tensor-network verification passed with **66 passed and one optional skip**,
including Hamiltonian MPO values/gradients and batched observable comparisons.
Smoke/unit verification passed with **1,373 passed, 13 skipped, and 1,488
deselected** in 23.32 seconds. Ruff, architecture, public API baseline, English
text, and whitespace checks passed. The full strict audit reports **775 errors
in 129 files**, with no errors in `simulation/tensor_network/observables.py`.
These local construction changes add no distributed scalability claim.

## MPS site-kernel cache typing

The site-kernel cache, compiler adapter, and execution helper now carry Tensor
return types. Cache lookup narrows the retrieved entry directly while retaining
the existing cold/warm bookkeeping. The Dynamo configuration adapter verifies
its context-manager result; unsupported imports and missing attributes retain
their existing null-context behavior. Cache policy, telemetry, synchronization,
and numerical fallback behavior are unchanged.

All 18 focused site-kernel/workspace tests passed. After the final context
validation change, smoke/unit verification passed with **1,373 passed, 13
skipped, and 1,488 deselected** in 23.81 seconds. Black and Ruff passed; this
round's architecture, public API baseline, English text, and whitespace checks
also passed. The full strict audit reports **769 errors in 128 files**, with no
errors in `simulation/mps/site_kernels.py`. CPU tests do not establish GPU
compilation performance or multi-node scalability.

## Static MPS compiled buckets

The static MPS path now types its Dynamo context adapter and validates the
returned context object. Its site kernel uses the existing `RealImagTuple`
alias, and the two-output CX callable has a distinct name from the single-output
site callable. Compilation limits, bucket keys, truncation, and numerical
execution are unchanged.

All 45 static/general MPS tests passed, including compiled eager-backend parity,
bucketed loss/gradient comparisons, and finite truncated gradients. Smoke/unit
verification passed with **1,373 passed, 13 skipped, and 1,488 deselected** in
23.41 seconds. Ruff, architecture, public API baseline, English text, and
whitespace checks passed. The full strict audit reports **765 errors in 127
files**, with no errors in `simulation/mps/static.py`. GPU compilation was not
benchmarked.

## Contraction layout and TOML compatibility

The real/imag contraction module uses one private alias for its existing
six-part canonical BMM layout, replacing duplicated tuple annotations and an
unparameterized tuple. Layout values and materialization decisions are unchanged.
Numerical conformance selects the standard-library TOML parser by Python version,
retaining `tomli` for Python 3.10 without a duplicate import definition.

Focused conformance/MPS/TN tests passed with **115 passed and two skips**.
Real/imag kernel tests passed with **six passed and seven hardware skips**.
Smoke/unit verification passed with **1,373 passed, 13 skipped, and 1,488
deselected** in 23.53 seconds. Ruff, architecture, public API baseline, English
text, and whitespace checks passed. The full strict audit reports **763 errors
in 125 files**. Validation used Python 3.12; the Python 3.10 branch and skipped
accelerator cases were not executed.

## Missing Kraus matrices

The batched noisy-statevector kernel now rejects a channel without Kraus
matrices using a `ValueError` that identifies the channel. Previously this
malformed internal instruction produced an incidental iteration error on
`None`. A regression test constructs that missing-matrix instruction and checks
the diagnostic. Valid trajectory evolution, RNG use, and event counts are
unchanged.

All 28 focused noisy-kernel and trajectory tests passed. Smoke/unit verification
passed with **1,374 passed, 13 skipped, and 1,488 deselected** in 23.32 seconds.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The full strict audit reports **762 errors in 124 files**, with no
errors in `simulation/statevector/noisy.py`. No hardware execution was performed.

## Low-rank projection normalization

The deterministic range projection uses `math.sqrt` for its positive scalar
normalization instead of generic exponentiation. This states the operation
directly and preserves a real scalar type for tensor division. Random seeds,
projection shapes, power iterations, and the approximate QR method are unchanged.

All 41 low-rank/general MPS tests passed, including deterministic reconstruction,
power-iteration error, and finite gradients. Smoke/unit verification passed with
**1,374 passed, 13 skipped, and 1,488 deselected** in 23.49 seconds. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The full strict audit reports **761 errors in 123 files**, with no errors in
`simulation/mps/low_rank.py`. No new approximation-quality or performance claim
is made.

## Batched rotation builders

The local rotation composer checks that RX/RY/RZ registry entries are callable
matrix builders. Empty rotation topology now raises its explicit topology
`ValueError` instead of indexing an empty matrix tuple. Existing nonempty gate
composition order and precision selection are unchanged.

All 13 focused statevector operation tests passed. New cases cover empty
topology and compare batched RX/RY/RZ values and gradients against the existing
closed-form expression. Smoke/unit verification passed with **1,376 passed, 13
skipped, and 1,488 deselected** in 23.23 seconds. Ruff, architecture, public API
baseline, English text, and whitespace checks passed. The full strict audit
reports **760 errors in 123 files**. The separate Triton loop's dynamically
typed output remains unresolved; no GPU validation is claimed.

## Immutable capability decision evaluations

Candidate matching now freezes completed evaluations once and supplies the same
tuple to decision identity calculation and the returned decision. Previously
both selected and rejected decisions received a mutable list despite the
existing tuple annotation on the frozen result model. Candidate order, scores,
fallback authorizations, and decision hash contents are unchanged.

All 55 focused matching/requirement tests passed. The matching suite also passed
all 46 tests after adding tuple assertions to successful and rejected decision
scenarios. Smoke/unit verification passed with **1,376 passed, 13 skipped, and
1,488 deselected** in 23.43 seconds. Ruff, architecture, public API baseline,
English text, and whitespace checks passed. The full strict audit reports
**758 errors in 122 files**, with no errors in
`runtime/target_capability_matching.py`. No capability was promoted.

## Extension registry mapping ownership

Removed the one-line `_frozen_map` wrapper, whose string-key annotation did not
describe the registry's tuple keys. Both owners now use `MappingProxyType`
directly. Extension configuration reuses its already-copied and validated local
dictionary; the registry still copies caller-owned entries before wrapping.
Shallow immutability, credential rejection, and task-local registration are
unchanged. No additional registry or generic type abstraction was introduced.

All 18 SDK/protocol tests passed. Smoke/unit verification passed with **1,376
passed, 13 skipped, and 1,488 deselected** in 23.40 seconds. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The full strict
audit reports **757 errors in 121 files**, with no errors in
`ecosystem/extensions/sdk.py`.

## Operator probe loss reduction

The probe loss starts from the first tensor loss and accumulates subsequent
terms in the existing order. Empty outputs now raise a descriptive `ValueError`
instead of returning Python integer zero to a caller expecting a differentiable
tensor. No output stacking or detached scalar conversion was introduced.

All nine focused operator-profile tests passed, including empty tuple/list
rejection and analytic gradients for mixed real/complex outputs. Smoke/unit
verification passed with **1,376 passed, 13 skipped, and 1,491 deselected** in
23.33 seconds. Ruff, architecture, public API baseline, English text, and
whitespace checks passed. The full strict audit reports **756 errors in 121
files**. Two upstream untyped `backward` call diagnostics in this module remain.

## Remote compute worker entrypoint

The shared worker entrypoint now declares its `None` return and writes result
JSON with explicit UTF-8 encoding. Run identity, GPU visibility checks, finite
JSON validation, and atomic replacement remain unchanged.

All 52 local Jiuding/workspace adapter tests passed. Smoke/unit verification
passed with **1,376 passed, 13 skipped, and 1,491 deselected** in 23.33 seconds.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The full strict audit reports **754 errors in 120 files**, with no
errors in `remote/compute/_worker.py`. These were local tests; no remote compute
or QPU tasks were submitted.

## Trajectory checkpoint aggregation

Checkpoint merging explicitly types accumulated failures and reuses the
completed-ID set for duplicate detection and stale-failure filtering. Ordering,
execution identity checks, Welford aggregation, and resumable checkpoint content
are unchanged. This is a pure aggregation cleanup, not a transport change.

All 24 trajectory runtime tests passed. Smoke/unit verification passed with
**1,376 passed, 13 skipped, and 1,491 deselected** in 23.44 seconds. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The full strict audit reports **753 errors in 119 files**, with no errors in
`runtime/trajectories/distributed.py`. No distributed performance or hardware
claim follows from these local tests.

## QCIS adapter result validation

QCIS export now checks that a circuit-like object's `to_ir()` returns
`CircuitIR` before invoking lowering validation. Invalid adapters receive a
direct `TypeError`; valid circuit conversion and QCIS gate decomposition are
unchanged. Tests cover adapters returning `None` and an IR-shaped dictionary.

All 32 focused deployment/operator-schema tests passed. Smoke/unit verification
passed with **1,376 passed, 13 skipped, and 1,493 deselected** in 23.17 seconds.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The full strict audit reports **752 errors in 118 files**, with no
errors in `compiler/qcis.py`. Tests used local deployment primitives only.

## Dynamic condition metadata typing

The bounded hybrid lowering path explicitly identifies gate metadata as a
string-keyed object mapping. A single condition and a disjunction of clauses
have different tuple depths; inference previously treated both branches as the
single-condition shape. Serialized metadata, predicates, bindings, and emitted
instructions are unchanged.

All 53 dynamic-session/lowering tests passed. Smoke/unit verification passed
with **1,376 passed, 13 skipped, and 1,493 deselected** in 23.40 seconds. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The full strict audit reports **751 errors in 118 files**. Parameter-binding
mapping variance in this module remains unresolved.

## Resident executor result names

The Jiuding resident executor uses distinct names for batch response dictionaries
and typed measurement results instead of reusing `result`/`results` across
unrelated branches. Its bounded exception-chain list is explicitly typed.
Request handling, result schemas, measurements, device checks, and error-chain
content are unchanged.

All 52 local workspace/Jiuding tests passed. Smoke/unit verification passed with
**1,376 passed, 13 skipped, and 1,493 deselected** in 23.41 seconds. Black, Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The full strict audit reports **740 errors in 117 files**, with no errors in
`remote/compute/_workspace_executor.py`. No resident service was started and no
remote tasks were submitted.

## Workspace result decoding

The resident-response decoder now declares Tensor, count-map, and ExecutionResult
return types while retaining lazy runtime imports. Count keys preserve their
existing string-or-integer representation. Tensor decoding checks that dtype is
a supported string before dictionary lookup, so list/dictionary dtype values
receive the intended unsupported-dtype diagnostic rather than a hashing error.

All 57 local adapter tests passed, including five invalid-dtype cases.
Smoke/unit verification passed with **1,381 passed, 13 skipped, and 1,493
deselected** in 23.42 seconds. Ruff, architecture, public API baseline, English
text, and whitespace checks passed. The full strict audit reports **734 errors
in 116 files**, with no errors in `remote/compute/_workspace_results.py`.
Unstructured wire payloads still use `Any` at the decoding boundary; this is
not a claim of complete schema validation. No remote tasks were submitted.

## Calibration readout decoding

Quafu calibration conversion reuses the existing Noise-owned readout decoder.
This removes duplicate nested-tuple conversion and constructs the declared 2x2
shape through the same validation used for deserialized noise models. Physical
wire mapping, calibration units, fidelity conversion, and readout probabilities
are unchanged. No public parser or new cross-domain contract was introduced.

All 12 twin/noise-decoding tests passed. Smoke/unit verification passed with
**1,381 passed, 13 skipped, and 1,493 deselected** in 23.50 seconds. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The full strict audit reports **733 errors in 115 files**, with no errors in
`remote/qpu/calibration.py`. Calibration tests used local fixtures, not live QPU
access.

## Jiuding transport annotations

The client identifies SSH channels as binary subprocesses and captures their
validated input/output streams before the nested frame reader runs. HTTP
headers and authentication results are string mappings; JSON payload/cache
values remain explicitly dynamic. The redirect handler declares its existing
no-redirect return behavior. Framing, deadlines, token handling, redaction,
retries, and subprocess lifecycle are unchanged.

All 57 local Jiuding/workspace tests passed. Smoke/unit verification passed with
**1,381 passed, 13 skipped, and 1,493 deselected** in 23.45 seconds. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The full strict audit reports **724 errors in 115 files**. The client still has
untyped methods and dynamically decoded payloads; no remote execution was used
to validate this round.

## Jiuding discovery rows

Pagination declares its iterator of JSON object rows and rejects non-object
entries before workspace discovery dereferences them. The workspace-record
lookup retains that object type. Page limits, ordering, workspace selection,
and authentication are unchanged. Four regression cases cover null, numeric,
string, and list entries.

All 61 local adapter tests passed. Smoke/unit verification passed with **1,385
passed, 13 skipped, and 1,493 deselected** in 23.41 seconds. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The full strict
audit reports **720 errors in 115 files**. No live discovery or remote task was
performed.

## Jiuding measurement entrypoint types

The resident measurement adapter declares the existing OutputRequest inputs
and CircuitIR output. Single-program client methods declare ExecutionResult
returns. Type-only imports preserve lazy loading; program adaptation continues
through the existing IR converter. Call defaults and runtime validation are
unchanged.

All 61 local adapter tests passed. Smoke/unit verification passed with **1,385
passed, 13 skipped, and 1,493 deselected** in 23.90 seconds. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The full strict
audit reports **713 errors in 115 files**. Batch methods and JSON response
schemas still require work; no new remote workflow or execution claim is made.

## Jiuding batch and request types

Batch execution declares its accepted list/tuple inputs, existing output
requests, and tuple of ExecutionResult values. Internal execution uses
MeasurementNode sequences and typed JSON object responses. Batch bounds,
ordering, result-count checks, framing, and execution behavior are unchanged.

All 61 local adapter tests passed. Smoke/unit verification passed with **1,385
passed, 13 skipped, and 1,493 deselected** in 23.47 seconds. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The full strict
audit reports **708 errors in 115 files**. Dynamic JSON values remain at the
transport boundary; remote hardware was not used.

## Workspace cache copy

Workspace context returns a deep copy of its cached JSON-derived dictionary
instead of serializing and reparsing it. This retains nested ownership isolation
and the dictionary return type without an unnecessary JSON round trip. A
regression test mutates the returned name and nested storage list, then verifies
that the next read is unchanged.

All 62 local adapter tests passed. Smoke/unit verification passed with **1,386
passed, 13 skipped, and 1,493 deselected** in 23.48 seconds. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The full strict
audit reports **706 errors in 115 files**. No live workspace discovery was used.

## Jiuding lifecycle and job iterator

The job iterator and context-manager methods now declare their existing return
types; the module-level run adapter uses the same OutputRequest/ExecutionResult
annotations as the client. The internal execution method performs one IR
replacement instead of duplicating the call across empty/nonempty measurement
branches.

The first smoke run had three architecture-related failures because added
annotations brought the client to 1,252 lines, above its 1,250-line ceiling.
Removing the duplicate branch restored the limit without changing the policy.
All 62 local adapter tests then passed, and final smoke/unit verification passed
with **1,386 passed, 13 skipped, and 1,493 deselected** in 23.58 seconds.
Architecture and whitespace checks passed after the fix; Ruff, public API, and
English checks passed in this round. The strict audit reports **699 errors in
115 files**. No remote tasks were submitted.

## Jiuding JSON mapping annotations

Workspace, image, submission, and status methods now identify JSON objects as
string-keyed dictionaries. Receipt parameters carry the same boundary type,
while HTTP headers use string values. This documents the existing transport
representation; it does not validate individual JSON fields or replace the
remaining dynamic payload values with a schema.

All 62 local adapter tests passed. Smoke/unit verification passed with **1,386
passed, 13 skipped, and 1,493 deselected** in 23.42 seconds. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The full strict
audit reports **688 errors in 115 files**. The arbitrary script-result method
still needs typing and boundary review. No external calls were made.

## Submitted-job result files

Shared receipt writing and result reading now live in `_job_results.py`, keeping
file handling separate from the already-large Jiuding client. Receipt writes
retain atomic replacement and explicitly use UTF-8. Result reads check the JSON
object envelope, run identity, and presence of `value`. Arbitrary script values
return as `object`; no unchecked type cast is required.

All 65 local adapter tests passed, including malformed result-file cases after
platform success. Smoke/unit verification passed with **1,389 passed, 13 skipped,
and 1,493 deselected** in 23.35 seconds. Ruff, architecture, public API baseline,
English text, and whitespace checks passed. The full strict audit reports **686
errors in 114 files** across 417 source files, with no errors in the Jiuding
client or new file helper. Dynamic JSON schemas still need deeper review; a
clean module-level type audit is not full boundary certification.

## Digital-twin submission receipts

TwinExperiment now checks that a provider submission returns a
ProviderTaskHandle before examining its backend and program identity. Invalid
provider responses raise an explicit TypeError instead of failing during
attribute access. Three regression cases cover empty, mapping, and string
responses; existing receipt identity checks remain unchanged.

All 16 focused twin tests passed. Smoke/unit verification passed with **1,389
passed, 13 skipped, and 1,496 deselected** in 23.39 seconds. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The API check
initially lacked Torch because its isolated launcher omitted site-packages;
rerunning with the repository's established isolated dependency setup passed.
The full strict audit reports **685 errors in 114 files** across 417 source
files. JSON normalization in TwinExperiment still has an unresolved Any return;
the round does not claim complete typing or live QPU validation.

## Device-profile channel lowering

The private channel-lowering helper now accepts DeviceNoiseProfile directly.
Its callers already check that a profile exists, so passing the profile avoids
reopening an optional field on the broader mutable NoiseModel. This removes
three optional-value type errors without adding redundant validation, changing
the public lowering signature, or altering channel timing and provenance.

All 60 noise tests passed, including idle-channel placement, thermal relaxation,
and calibration provenance. Ruff, architecture, public API baseline, English
text, and whitespace checks passed. The full strict audit reports **682 errors
in 113 files** across 417 source files, with no errors in compiler/noise.py.
This round used the focused numerical tests; the previous smoke/unit result
remains the latest broad run. No accelerator or QPU execution was requested.

## Hybrid optimizer construction

The private optimizer factory now declares its Optimizer return type. The
closure-based step dispatch checks for the actual LBFGS instance, replacing a
method-string check and a non-None assertion. Public stage selection and
optimizer settings remain unchanged.

All 14 hybrid optimization, algorithms-package, and CPU vertical-slice tests
passed. Ruff, architecture, dependency policy, public API baseline, English
text, and whitespace checks passed. The strict audit remains at **682 errors
in 113 files**: typing the factory removes its missing-return annotation but
exposes the installed PyTorch LBFGS.step method's missing annotations. No cast
or suppression was added to hide that dependency boundary.

## Dynamic readout rule names

Dynamic noise validation now uses a distinct readout_rule variable for readout
rules, instead of reusing the gate-noise rule variable with a different type.
This makes the two validation domains explicit and removes misleading optional
wire and missing-attribute errors. Validation behavior is unchanged.

All 13 dynamic-noise and feedback tests passed. Ruff, architecture, public API
baseline, English text, and whitespace checks passed. The full strict audit
reports **679 errors in 113 files** across 417 source files. The execution-option
resolver was also reviewed; its dynamic mapping construction remains unresolved
and was not hidden behind casts or duplicate field declarations.

## Dynamic feedback plan narrowing

The trajectory loop now declares its observation list as
list[DynamicFeedbackObservation] and explicitly requires a feedback plan when
entering the feedback-point branch. This makes the relationship between an
optional plan and its trigger lookup visible to both readers and type checking.
Controller behavior and recorded observations are unchanged.

All 28 dynamic-circuit, noise, and feedback tests passed. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The full strict
audit reports **674 errors in 113 files** across 417 source files. Optional wire
typing for feedback actions remains unresolved; their existing constructor
validation was preserved instead of adding duplicate checks merely for typing.

## MPS layer lifecycle records

The forward executor now names its layer lifecycle dictionary lifecycle_record,
separately from an optional boundary-gate truncation record. This removes an
ambiguous variable reuse without changing diagnostics, callbacks, or tensor
lifetimes. The callback still receives a dictionary copy.

All 18 local lifetime, production-contract, and memory-report contract tests
passed. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The strict audit reports **671 errors in 113 files** across 417
source files. The compiled-layer cache's heterogeneous tuple representation
still needs review; this round does not certify distributed numerical execution
or actual GPU memory behavior.

## MPS parameter bucket allocation

Parameter synchronization now allocates owner lists only when encountering a
new dtype/device group. Previously, setdefault eagerly created and discarded a
full set of owner lists for every parameter already in a group. Per-round
unpacking indices also have a distinct name from the mutable grouping lists.
Bucket contents, ownership, buffers, and communication remain unchanged.

All 20 local MPS training and checkpoint tests passed, including parameter
bucket ownership and payload checks. Ruff, architecture, public API baseline,
English text, and whitespace checks passed. The strict audit reports **670
errors in 113 files** across 417 source files. Remaining optional gradient and
optimizer-state types require further work; no distributed performance claim
is based on this local allocation cleanup.

## Compiled MPS bucket names and broad regression

Two-site compiled layers now use two_site_buckets keyed by shape_pair, distinct
from the one-site tensor-shape buckets. This removes incompatible variable
reuse while preserving batching order and factorization behavior.

All seven focused compiled-layer numerical and lifetime tests passed. The broad
smoke/unit suite passed with **1,389 passed, 13 skipped, and 1,496 deselected**
in 23.32 seconds. Ruff, architecture, public API baseline, English text, and
whitespace checks passed. The strict audit reports **668 errors in 113 files**
across 417 source files. The heterogeneous compiled cache remains unresolved;
neither type ignores nor a widened Any annotation were used to conceal it.

## MPS communicator warmup declarations

Neighbor warmup now declares its P2POp list and retained tensor buffers, and
uses the existing time import directly instead of dynamic imports. Operations,
buffer lifetime, synchronization, and warmup caching are unchanged.

All 14 local packed-send, metadata-transport, and production-contract tests
passed. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The full strict audit reports **665 errors in 113 files** across
417 source files. These tests do not establish live communicator warmup or
multi-node transport performance; no remote jobs were submitted.

## MPS gradient piece planning

Gradient bucket planning now declares each pending piece as an
(index, start, stop) integer tuple and reads a parameter's element count once
before splitting it. Bucket boundaries, ordering, and owner grouping remain
unchanged, including zero-element parameters.

All 17 reverse-contract tests passed. Ruff, architecture, public API baseline,
English text, and whitespace checks passed. The full strict audit reports
**664 errors in 113 files** across 417 source files. This is a local planning
cleanup, not a distributed gradient or scalability certification.

## Local rotation-region annotations

The local statevector path now declares its collected rotation-region angles
as a list of tensors. This annotation preserves the existing batching and
fallback behavior. Optional parameter narrowing and duplicated angle collection
remain open work; they were not bypassed with casts.

All 15 local statevector numerical and performance-report contract tests
passed. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The strict audit reports **663 errors in 113 files** across 417
source files. CUDA fusion execution was not exercised in this round.

## Shared rotation-region collection

Both local rotation-fusion paths now use one private angle collector. It
preserves batch and region ordering, retains autograd, and returns None when a
gate has no parameter tensor so the existing unfused path remains available.
The helper replaces duplicate tuple collection and optional-value indexing;
it adds no public abstraction or type cast.

All 17 focused tests passed, including new batch-value, gradient, and missing
parameter cases. An initial Ruff import-order failure was corrected; final
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The strict audit reports **660 errors in 113 files** across 417 source
files. CUDA fusion still needs accelerator validation; these are local tests.

## Tensor-network DAG input handling

Replicated pair contractions now name and validate their left and right tensors
directly. Sharded contractions retain their input mapping and use a distinct
shard_input variable through redistribution. This removes tuple/dictionary and
optional-tensor variable reuse while preserving missing-input errors and
contraction behavior.

All 54 local distributed-DAG tests passed. Ruff, architecture, public API
baseline, English text, and whitespace checks passed. The strict audit reports
**653 errors in 113 files** across 417 source files. Other execution branches
still have optional-input and mode typing issues. These local tests do not
certify multi-node transport or accelerator performance.

## Owner-executed DAG initialization

The owner-executed DAG path now names its required initial tensor separately
from tensors looked up or received during contraction. This removes the last
two optional-tensor assignment errors in that path without changing ownership,
missing-value checks, or send/receive behavior.

All 54 local distributed-DAG tests passed again. Ruff, architecture, public API
baseline, English text, and whitespace checks passed. The strict audit reports
**651 errors in 113 files** across 417 source files. The module still has a
contraction-mode typing mismatch; distributed communication remains unverified
by this local round.

## Sharded contraction mode propagation

The operation preparation return annotation now preserves the same two Literal
mode values already declared by the authoritative mode selector and consumed
by the contraction kernel. No runtime conversion, additional mode, or unchecked
cast was introduced.

All 54 local distributed-DAG tests passed. Ruff, architecture, public API
baseline, English text, and whitespace checks passed. The strict audit reports
**650 errors in 112 files** across 417 source files, with no remaining errors
in distributed_execution.py. This module-level type result does not establish
live multi-node correctness or full repository readiness.

## Partial-mesh owner order and reverse records

Partial-mesh redistribution now declares the mapping from global slice ranges
to canonical owner ranks. Iteration uses its insertion order: owners are added
in ascending rank order, so re-sorting for every destination was redundant.
Dynamic reverse selection also declares its existing reverse-record list.

All 55 local DAG and dynamic-checkpoint tests passed. Ruff, architecture, public
API baseline, English text, and whitespace checks passed. The strict audit
reports **648 errors in 112 files** across 417 source files. Optional reverse
results and checkpoint payload typing remain open work; no distributed run
was used as evidence in this round.

## Dynamic checkpoint manifest validation

Checkpoint loading now includes non-object JSON manifests in the existing
collective invalid-manifest decision. Previously, a JSON null, list, or string
could reach dictionary operations after that decision. Rank-file validation
also checks entry presence directly, preserving the integrity failure path.

All 58 local checkpoint and DAG tests passed, including three malformed
manifest cases that verify the collective invalid flag. After correcting a
test import-order issue, Ruff and the four checkpoint tests passed again.
Architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reports **646 errors in 111 files** across 417 source files.
The collective is mocked in these regressions; actual multi-rank recovery has
not been validated here, and deeper manifest schema validation remains open.

## Reverse preflight device and layout types

Dynamic reverse preflight now receives the device already resolved by its
caller, replacing two repeated optional-result lookups. The remesh helper
declares its layout/tensor return pair, and the initial seed layout has a
distinct name from subsequently consumed cotangent layouts.

All 58 local DAG and checkpoint tests passed after the final change. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reports **643 errors in 110 files** across 417 source files,
with no remaining errors in dynamic_reverse.py. No collective behavior or
checkpoint schema changed; multi-rank recovery remains outside this round's
validation.

## Partial-mesh transfer byte totals

Redistribution planning now keeps network and self-transfer byte counts as
integer variables shared by the result and identity payload. Self-transfer
bytes are the delivered total minus network bytes, avoiding a second filtered
traversal. The mesh group context manager also declares its unused exit inputs.

All 54 local DAG tests passed. Ruff, architecture, public API baseline, English
text, and whitespace checks passed. The strict audit reports **640 errors in
110 files** across 417 source files. Layout construction from a dynamic payload
remains unresolved; identity fields and byte-accounting semantics were retained.

## Rematerialization helper contracts and regression

The rematerialization provider's layout helper now declares its existing
partial-mesh layout result. Peak prediction accepts a sequence of contraction
records and a mapping of remaining use counts, making its read-only inputs
explicit. Its local copy still owns the countdown mutations.

All 58 focused DAG and sliced-task tests passed. Broad smoke/unit verification
passed with **1,391 passed, 13 skipped, and 1,499 deselected** in 23.31 seconds.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The strict audit reports **638 errors in 109 files** across 417 source
files, with no remaining errors in rematerialization.py. Repository-wide strict
typing and live distributed validation are still incomplete.

## Measurement value representations

Measurement dispatch now declares its tensor-or-counts value and sampling
statistics types. The optional density-matrix probability probe uses a
separate variable before selecting the existing marginal-probability fallback,
so an unavailable probe is not confused with a completed measurement value.

All 17 measurement and Pauli-plan tests passed. Ruff, architecture, public API
baseline, English text, and whitespace checks passed. The strict audit reports
**634 errors in 109 files** across 417 source files. Optional subset expectation
values still require review; public result fields remain unchanged.

## Pauli subset expectation storage

Pauli sampling and marginal-probability reconstruction no longer prepend an
unused None sentinel to their subset expectations. Mask enumeration still
starts at one, and both paths iterate the tensor list directly instead of
copying a slice for every outcome. Numerical accumulation order is preserved.

All 17 measurement and Pauli-plan tests passed. Ruff, architecture, public API
baseline, English text, and whitespace checks passed. The strict audit reports
**630 errors in 108 files** across 417 source files, with no remaining errors
in measurements.py. Dynamic backend return schemas still warrant validation;
type checking alone does not certify every backend implementation.

## JAX MPS evidence collection types

Backward evidence aggregation now declares boundary records grouped by edge
and integer byte-count vectors by phase. Accounting formulas, blocker strings,
and evidence classifications remain unchanged.

JAX distributed-plan tests completed with **99 passed and 6 skipped**. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reports **628 errors in 108 files** across 417 source files.
Skipped accelerator cases are not passing hardware evidence; this round did
not execute remote jobs or promote capacity claims.

## JAX gradient primitive ownership

MPS parameter-gradient execution now imports the compute-dtype context setter
directly from its Simulation owner. The runtime kernel still supplies its own
parameter proxy; it no longer serves as an accidental re-export for this
primitive. Imports remain lazy and dtype restoration is unchanged.

JAX distributed-plan tests completed with **99 passed and 6 skipped**. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reports **626 errors in 107 files** across 417 source files,
with no remaining errors in mps/gradients.py. Hardware-dependent skips remain
unverified; no production backward-readiness claim was changed.

## JAX statevector and tensor-network dtype imports

Six remaining dtype-context imports in statevector kernels, statevector
execution, and tensor-network gradients now reference the Simulation primitive
directly. Runtime parameter proxies remain in their owning kernel module.
Lazy loading and try/finally restoration behavior are unchanged.

JAX distributed-plan tests completed with **99 passed and 6 skipped**. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reports **620 errors in 105 files** across 417 source files.
Additional array-conversion imports and dependency annotations remain open;
the skipped tests do not constitute accelerator validation.

## JAX gate-matrix and basis-index imports

JAX array conversion now imports the Torch instruction-matrix implementation
from Simulation directly. Statevector kernels likewise import basis-index
generation from its Simulation owner instead of the array-conversion module.
The functions and numerical behavior remain the same.

JAX distributed-plan tests completed with **99 passed and 6 skipped**. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reports **618 errors in 104 files** across 417 source files,
with no remaining errors in array_conversions.py. JAX partition-spec typing
and accelerator execution remain unresolved validation work.

## Statevector communication and adjoint variables

Chunked statevector exchange now declares its P2POp collection. The accelerated
adjoint branch uses accelerated_matrix separately from the optional matrix
constructed by the general reversible path, removing a conflicting declaration.
Communication order and gate mathematics are unchanged.

All 23 local statevector reverse-contract tests passed. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The strict
audit reports **616 errors in 104 files** across 417 source files. CUDA adjoint
and actual multi-rank exchange were not exercised in this round.

## MPS reverse bucket and ownership names

Two-site reverse preparation now names its bucket collection and three-shape
key separately from the one-site branch. Parameter gradient owners are
explicitly a list of integer sets. Ordering, gradient reduction ownership,
prefetch behavior, and checkpoint accounting are unchanged.

All 40 local reverse-contract, training, and compiled-layer numerical tests
passed. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The strict audit reports **613 errors in 104 files** across 417
source files. Other reverse tensor/tuple reuse remains open; these local tests
do not certify accelerator or multi-node reverse execution.

## MPS batched reverse input names

Compiled two-site reverse preparation now uses left_inputs and right_inputs
for tuples of tensors, keeping before_left and before_right for individual
site tensors in later paths. This eliminates cascading tuple/tensor inference
errors without altering contractions, cloning, factorization, or saved tapes.

All 40 local reverse-contract, training, and compiled-layer numerical tests
passed. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The strict audit reports **587 errors in 104 files** across 417
source files. The reduction includes dependent diagnostics from the ambiguous
names; it is not evidence of 26 separate numerical bugs being fixed.

## MPS one-site input collection

The compiled one-site branch now names its input tensor tuple site_inputs,
reserving before for a single tensor in the ordinary execution path. Saved
input tensors, output indexing, and detach/clone behavior are unchanged.

All 40 local reverse-contract, training, and compiled-layer numerical tests
passed. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The strict audit reports **584 errors in 104 files** across 417
source files. Remaining reverse return-shape and diagnostic-mapping issues
need substantive review; no type suppression was added.

## MPS one-site output collection

The compiled one-site kernel correctly returns a tuple of output tensors. Its
caller now names that tuple site_outputs, reserving after for individual
tensors. No kernel signature, return shape, or numerical computation changed.

All 40 local reverse-contract, training, and compiled-layer numerical tests
passed. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The strict audit reports **581 errors in 104 files** across 417
source files. The remaining MPS reverse issues include optional halo tensors
and structured diagnostic metadata.

## MPS local halo ownership branch

When both sites have the same owner, gate and canonicalization preparation now
assign the local halo only on that owner rank. Other ranks do not consume this
variable and no longer assign an unused None. Remote receive and prefetch paths
are unchanged.

All 40 local reverse-contract, training, and compiled-layer numerical tests
passed. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The strict audit reports **579 errors in 104 files** across 417
source files. This control-flow cleanup does not replace a multi-rank transport
or canonicalization correctness run.

## MPS wire-sequence boundaries

Canonicalization planning now consumes the already-created sorted dirty-bond
snapshot. Objective adjoint selection passes the deduplicated wire collection
as a tuple to the existing sequence contract. Selected bonds and wires remain
unchanged.

All 40 local reverse-contract, training, and compiled-layer numerical tests
passed. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The strict audit reports **577 errors in 104 files** across 417
source files. Numerical and distributed certification remains broader than
these local contract checks.

## FlagOS RNG-state boundary

The FlagOS adapter now verifies that the vendor RNG getter returns a Tensor.
Invalid values produce an explicit TypeError at the device boundary, before
being treated as checkpointable RNG state. Three regressions cover None,
bytes, and a list using a mocked vendor API.

All 18 platform-runtime tests passed. Broad smoke/unit verification passed
with **1,394 passed, 13 skipped, and 1,499 deselected** in 23.41 seconds. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reports **576 errors in 104 files** across 417 source files.
No Torch-FL device or remote workload was used in these tests.

## Training context annotations

The statevector deadline context now declares its yielded None value and the
signal handler's optional Python frame. The MPS NVTX helper declares a context
manager, preserving the CPU null context and CUDA range behavior. Timer setup,
restoration, and profiling lifetime are unchanged.

All 33 training-state and MPS training tests passed. Ruff, architecture, public
API baseline, English text, and whitespace checks passed. The strict audit
reports **574 errors in 103 files** across 417 source files. Hardware profiling
and distributed failure recovery were not tested in this round.

## Sliced reverse helper types

Sliced DAG construction now declares its contraction-DAG return type. Forward
peak prediction takes that DAG explicitly, and slice batching declares its
assignment sequences and node-tuple result. Checkpoint policy and contraction
execution are unchanged.

All 58 local DAG and sliced-task tests passed. Ruff, architecture, public API
baseline, English text, and whitespace checks passed. The strict audit reports
**569 errors in 103 files** across 417 source files. The expectation-plan versus
contraction-plan input mismatch remains open; no cast was used to hide it.

## MPS observable scan callbacks

The observable scan step now declares its tensor carry and tensor pair return.
The compiled fixed-target callback and its factory declare the same tensor
input and pair output. Scan recurrence, compilation options, and selected
observables are unchanged.

All 13 static-MPS and Hamiltonian-identification tests passed. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The strict
audit reports **565 errors in 103 files** across 417 source files. The kernel
factory and observable-name tuple types remain open in brickwork.py; these
local tests do not establish GPU compilation performance.

## Brickwork kernel factory and exports

The compiled kernel factory now declares its one-site and two-site callable
signatures. An earlier __all__ assignment that was overwritten later in the
module was removed; the final exported names and order remain unchanged.

All 13 static-MPS and Hamiltonian-identification tests passed. Black, Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reports **563 errors in 102 files** across 417 source files,
with no remaining errors in brickwork.py. The prior observable-name tuple
diagnostic came from the overwritten export assignment, not numerical output.

## Finite MPS communication timeouts

P2P timeout parsing now rejects NaN and infinite values explicitly, alongside
nonpositive values. The CUDA overlap path resolves the timeout before creating
a stream or launching communication, avoiding a configuration error after
requests have already started.

All seven local timeout-diagnostic and packed-send tests passed. Five invalid
configuration cases also exercise early rejection through the overlap entry
without accessing CUDA or running the overlap callback. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The strict
audit remains at **563 errors in 102 files** across 417 source files. This is
a runtime validation fix, not a type-error reduction or live transport test.

## MPS boundary synchronization record names

Local rank simulation now names its boundary synchronization record
boundary_sync, separately from truncation records. This removes cascading
field/type diagnostics while preserving tensor exchange, split accounting,
and protocol evidence.

Thirteen selected native MPS tests passed in the sandbox. Two additional
single-rank tests initially failed because loopback socket binding was denied;
both passed when rerun with permission for local networking. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The strict
audit reports **530 errors in 102 files** across 417 source files. The reduction
includes dependent diagnostics from one ambiguous variable; it is not a claim
of 33 independent behavioral fixes or multi-node certification.

## Typed MPS execution statistics

MPS execution statistics now use private TypedDict contracts for required
communication counters and boundary records, with optional local simulation
evidence. Replicated, local sharded, and synchronized execution share this
contract. The synchronized path no longer declares its record tuples to be
integers, and the other paths no longer erase these field types with Any.
Returned dictionaries and public result fields are unchanged.

Thirteen local native MPS tests passed. The smoke/unit selection passed with
**1394 passed, 13 skipped, and 1504 deselected**. Architecture boundaries,
public API baseline, Ruff, and English text checks passed; the text scan
covered 1896 paths with no Han-script content. Black formatted the module.
The strict audit reported **529 errors in 102 files** across 418 source files,
with no diagnostics in MPS execution. The checked source set changed since
the preceding audit, so the aggregate difference is not solely attributable
to this patch. This round did not run live GPU or multi-node workloads.

## Reject non-object tensor-network plan caches

The persistent contraction-plan reader now treats a non-object JSON root as
a cache miss, consistently with malformed JSON and incompatible cache metadata.
Previously, null, arrays, strings, numbers, and booleans reached an unchecked
get call and raised AttributeError. Five regression cases reproduced that
failure before the fix and passed afterward. The file-lock context manager
also declares its Iterator[None] return type.

The tensor-network test module passed with **71 passed and 1 skipped**,
including the existing plan round trip. Ruff, architecture, public API baseline,
English text, and whitespace checks passed. The text scan covered 1898 paths
with no Han-script content. The strict audit reported **528 errors in 101 files**
across 418 source files; the plan-cache module has no strict diagnostics.
This does not certify every nested cache field or live distributed execution.
MPS reverse metadata still needs a coordinated contract cleanup across its
numerical producers, record transport, and tape construction.

## MPS compiled training callback types

The training executor now declares its optional compiled value-and-gradient
callback with its actual tensor signature. Observable-wire normalization uses
an arbitrary-length integer tuple, rather than inferring a single-element
tuple from the integer-input branch. Neither change alters runtime behavior
or dataclass constructor fields.

The MPS and owner-sharded parameter synchronization selections passed with
**40 passed**, including parameter updates and compiler-failure fallback.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The strict audit reported **526 errors in 100 files** across 418 source
files, with no diagnostics in compiled_training.py. These local tests do not
establish live GPU compiler performance or distributed scalability.

## Statevector topology edge accumulation

Topology planning now accumulates segment indices in lists and converts each
list to a tuple once when constructing its communication edge. Previously,
each segment copied the entire preceding tuple. The internal accumulator
also has distinct integer-byte and integer-list positions rather than a
heterogeneous list with an incorrect fixed-length tuple annotation.

Ten focused planning, topology, pair-exchange, and performance-estimate tests
passed. A regression case checks three separated communication segments on
one edge, preserving their order and summed transfer bytes. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The strict
audit reported **520 errors in 99 files** across 418 source files, with no
diagnostics in statevector planning. This removes repeated tuple copying;
no measured wall-clock acceleration or live transport claim is made.

## Statevector configuration and diagnostic branches

CX segment selection now handles an unset or empty environment value directly,
without replacing a string with None before testing the same condition again.
The existing selection test also covers the empty-string case. All-gather
diagnostics construct the optional error tuple in one expression, preserving
the returned messages while avoiding empty-tuple-only type inference.

The forward and distributed-statevector selections passed with **41 passed
and 1 deselected**; the separate single-rank subprocess smoke was not selected.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The strict audit reported **518 errors in 98 files** across 418 source
files. Statevector local_execution.py has no strict diagnostics; forward.py
still has other unresolved diagnostics. No live GPU or multi-node test ran.

## Sliced tensor-network checkpoint summary typing

The checkpoint memory summary now explicitly uses its existing dict[str, object]
return contract while it is being assembled. This permits the optional tuple
of checkpoint value IDs alongside integer byte counters without incorrectly
inferring an integer-only dictionary. Runtime fields and output are unchanged.

The contraction-DAG and sliced-task selections passed with **58 passed**.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The strict audit reported **516 errors in 98 files** across 418 source
files. The two remaining diagnostics in sliced_reverse.py concern expectation
plans passed to the contraction-DAG planner; they require a contract review
rather than an unchecked cast.

## Initial MPS shape-table decoding

Initial-state reconciliation now copies the reduced shape table to CPU once
and decodes its four dimensions explicitly. Previously each wire performed
its own CPU conversion. Local tensor shapes also retain their four-element
type after the existing rank check. Validation and clone behavior are preserved.

Thirty-nine initial-shape, reverse-contract, and training tests passed. New
local tests verify exact shapes, batch size, independent tensor clones, and
rejection of mismatched neighboring bonds with a mocked collective. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reported **514 errors in 98 files** across 418 source files.
The initialization result's dictionary variance diagnostic remains unresolved.
These tests do not exercise live collectives or measure GPU transfer latency.

## Initial training-state shape ownership

The initial-state contract now constructs typed local shape tuples before
building its JSON descriptors. Both generated and supplied tensor states use
those tuples directly. Shapes no longer need to be recovered from descriptors
that also contain initializer, dtype, and hash strings. Descriptor fields,
serialization order, and tensor digest construction are unchanged.

The training and reverse-contract selections passed with **37 passed**,
including reuse of one tensor digest across multiple sites. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The strict
audit reported **513 errors in 98 files** across 418 source files. Optional
optimizer-state diagnostics in the training engine remain unresolved.

## Paired L-BFGS previous-step state

The previous parameter vector and gradient now live in one optional tuple.
They are created together, unpacked together, and counted together for optimizer
memory accounting. This expresses the existing invariant directly instead of
maintaining two independently optional variables. Curvature arithmetic, clone
operations, and history updates are unchanged.

Thirty-seven training and reverse-contract tests passed. Black, Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reported **512 errors in 98 files** across 418 source files.
The selected tests do not certify the full distributed Adam-to-L-BFGS training
path; a live-process regression remains needed for that path. The training
engine still has optional start-step and parameter-gradient diagnostics.

## Single-rank Adam-to-L-BFGS regression

A real single-rank CPU Gloo test now runs one Adam step followed by three
L-BFGS steps on a two-wire MPS with a trainable RY gate. It compares every
reported loss and the final parameter with native PyTorch Adam and L-BFGS on
the analytic cos(theta) objective, using 1e-9 tolerances. The test also checks
optimizer-stage transitions and always destroys the initialized process group.

The first attempt could not initialize Gloo under sandbox loopback restrictions.
The bounded rerun with local-network permission passed: **1 passed**. Black,
Ruff, architecture, English text, and whitespace checks passed. No production
source changed in this round; the latest strict baseline remains 512 errors
in 98 files. This closes the single-rank optimizer regression gap recorded
above, while multi-rank and accelerator coverage remain outstanding.

## Validated L-BFGS transition step

The trainer converts the optional L-BFGS transition step once, immediately
after its existing mode-specific validation. The loop uses the resulting
integer, preserving the transition condition and leaving non-L-BFGS modes
unchanged. Five input cases cover a missing, negative, zero, or out-of-range
transition step and verify rejection before execution.

The training unit tests and real single-rank Gloo analytic optimizer comparison
passed together with **26 passed**, using local-network permission. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reported **511 errors in 98 files** across 418 source files.
The remaining training-engine diagnostic is optional parameter-gradient access
in diagnostic reporting; it has not been hidden with a cast or suppression.

## Explicit training gradient diagnostics

Gradient reporting now snapshots each owned gradient through a small internal
function. A missing gradient raises MPSTrainingError identifying its parameter,
instead of failing later with an AttributeError from None.detach(). Empty
ownership remains valid. No missing gradient is silently replaced or omitted.

The real single-rank Gloo optimizer regression now enables gradient reporting
and compares every recorded gradient with the analytic -sin(theta), using
1e-9 tolerance. Together with training unit tests, including missing-gradient
rejection, **27 tests passed** with local-network permission. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The strict
audit reported **510 errors in 97 files** across 418 source files, with no
remaining training_engine.py diagnostics. Multi-rank execution remains outside
this round's evidence.

## Statevector derivative collection

The adjoint loop now uses one tensor list for derivatives, including gates
without active parameters. It no longer switches that variable to an empty
tuple, and the unreachable None-gradient branch was removed. Both derivative
producers append tensors; accumulation order and numerical kernels are unchanged.

The first focused test command used a nonexistent filename and ran no tests.
The corrected ISSUE-042 reverse selection passed. Ruff, architecture, public
API baseline, English text, and whitespace checks passed. The strict audit
reported **509 errors in 97 files** across 418 source files. The separate
optional CX scratch-buffer diagnostic remains unresolved.

## Paired CX adjoint scratch buffers

The CX-segment ket and adjoint scratch tensors now share one optional tuple.
Both are allocated together, unpacked before use, and replaced together with
the previous ket and adjoint buffers after the segment. This preserves buffer
reuse while making the paired-allocation invariant explicit.

The reverse selection passed with **23 passed**. The broader smoke/unit run
passed with **1409 passed, 13 skipped, and 1511 deselected**. Black, Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reported **507 errors in 97 files** across 418 source files;
the CX scratch optionality diagnostic is gone, while other adjoint diagnostics
remain. Concurrent repository changes may affect aggregate audit totals.
The selected CPU tests do not execute the Triton CX accelerator path, which
still requires GPU validation.

## Communication layout candidate selection

The wire-layout planner now names the selected sharded-wire tuple before
building its membership set. This separates ordered cost comparison from set
membership and avoids contextual inference widening the candidate type to an
arbitrary iterable. Candidate generation, tie breaking, and cost are unchanged.

The ISSUE-041 forward selection passed. Ruff, architecture, public API baseline,
English text, and whitespace checks passed. The strict audit reported
**506 errors in 97 files** across 418 source files. This is a readability and
typing cleanup, not a new layout algorithm or measured speedup.

## Module tensor-network state naming

The local tensor-network branch now names its state tensor_network_state,
separately from the MPS backend state used by Hamiltonian evaluation. An unused
type-ignore comment on the Module class was also removed. Public signatures,
mode selection, and returned values are unchanged.

The Module and training API contract selections passed with **53 passed**.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The strict audit reported **504 errors in 97 files** across 418 source
files. Other Module typing diagnostics remain unresolved.

## Module program and runtime mapping contracts

Module construction now explicitly retains its existing Circuit-or-CircuitIR
program contract across builder branches. Execution metadata is declared as
the Mapping contract shared by local dictionaries and LiveRuntimeSummary,
preserving live distributed evidence instead of copying it into a dictionary.

The Module and training API contract selections passed with **53 passed**.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The strict audit reported **502 errors in 97 files** across 418 source
files. These declarations do not change public interfaces or execution behavior.

## Circuit reconstruction option types

The six retained Circuit constructor options now use a private TypedDict:
wire count, batch size, device, dtype, initial tensor, and runtime configuration.
Circuit parameter binding and Module builder compilation pass these options
directly as keyword arguments, avoiding an extra dictionary copy and retaining
field-specific types. The stored object remains a dictionary and the public
constructor and serialized fields are unchanged.

The Module, API-contract, and selected native-circuit tests produced **119
passed and 2 deselected** in the sandbox, with one additional test blocked by
loopback binding permissions. That single-rank tensor-network test passed when
rerun with local-network permission. Ruff, architecture, public API baseline,
English text, and whitespace checks passed. The strict audit reported
**493 errors in 97 files** across 418 source files. The aggregate includes
concurrent repository changes and is not a count of independent fixes here.

## Validated dynamic gate wires

Generated gate methods now build their resolved wire list during missing-wire
validation. A None entry still raises the same TypeError before parameter
processing; validated entries retain their original order and values. This
replaces the discarded list of missing indices and avoids forwarding a
possibly-None wire list to Circuit.gate.

The selected native-circuit tests passed. Ruff, architecture, public API
baseline, English text, and whitespace checks passed. The strict audit reported
**492 errors in 97 files** across 418 source files. Public method signatures
and missing-wire error messages are unchanged.

## PyTorch platform dependency attributes

CPU and CUDA adapters now explicitly declare optional_dependency as str | None,
matching the existing writable PlatformRuntime attribute contract. Their values
remain None, and device discovery and activation behavior are unchanged.

The platform-runtime selection passed with **18 passed**. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The strict
audit reported **490 errors in 97 files** across 418 source files. The CPU and
CUDA registry compatibility diagnostics are resolved; the FlagOS dependency
contract and frozen CPU probe compatibility still need separate review.
No accelerator hardware was exercised.

## FlagOS activation retry correctness

FlagOS activation now caches the imported vendor module only after confirming
that it registered torch.flagos. Previously a failed registration check left
the cache populated, allowing the next activate call to return without repeating
validation. Status reporting also requires device registration when it detects
an externally imported Torch-FL module.

The regression test failed before the fix and now verifies two rejected
activation attempts followed by successful recovery once registration exists.
All **19 platform-runtime tests passed**. Ruff, architecture, public API baseline,
English text, and whitespace checks passed. These tests use a synthetic vendor
module and do not validate a real Torch-FL installation. No full strict audit
was rerun in this behavioral-fix round; the preceding baseline is 490 errors.
The FlagOS optional-dependency type contract remains outstanding.

## FlagOS optional dependency contract

FlagOS now implements the existing str | None dependency attribute contract.
An unset dependency reports unavailable and raises the installation error on
activation, before calling Python's module discovery or import machinery.
The default Torch-FL dependency and successful activation path are unchanged.

All **20 platform-runtime tests passed**, including an unset-dependency case
that rejects any discovery/import attempt. Ruff, architecture, public API
baseline, English text, and whitespace checks passed. The strict audit reported
**489 errors in 96 files** across 418 source files. All three platform registry
compatibility errors are now resolved. Hardware validation and the frozen CPU
probe contract remain separate outstanding work.

## JAX MPS probe result values

The accelerator probe computes total communication bytes directly from its
uniform per-edge byte count and edge count. It no longer re-reads the same
integer from heterogeneous evidence dictionaries. Its local backward wrapper
also unpacks and returns the value/gradient pair explicitly. Evidence fields,
classification, and numerical computation are unchanged.

The JAX distributed-plan selection passed with **99 passed and 6 skipped** on
the local CPU environment. Ruff, architecture, public API baseline, English
text, and whitespace checks passed. The strict audit reported **487 errors in
96 files** across 418 source files. These tests do not certify a live JAX GPU
probe or multi-node scaling. CPU probe mutability remains under review; its
frozen observation was not weakened to satisfy typing.

## JAX tensor-network expectation node assembly

Expectation contraction now appends operator nodes directly to the combined
bra/ket list. This removes the separate operator list and a second list
concatenation, while retaining bra, ket, and operator ordering. Labels share
the established arbitrary-length tuple type throughout assembly.

The JAX distributed-plan selection passed with **99 passed and 6 skipped** on
CPU. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The strict audit reported **486 errors in 95 files** across
418 source files, with no diagnostics in the JAX tensor-network kernel module.
No measured performance or accelerator scalability claim is made.

## Sparse target validation and result names

Single-amplitude targets are now assembled after rejecting a missing bitstring,
so downstream backends receive a non-optional target tuple. The distributed
expectation result also has its own name rather than sharing an amplitude-result
variable. Public signatures, target ordering, and the missing-bitstring error
message are unchanged.

The initial target/adapter selection passed with 10 tests. Three added cases
then passed with the full 7-test target module, covering missing bitstrings and
the valid integer-zero target across statevector, MPS, and tensor network. This
is **13 distinct passing tests**. Ruff, architecture, public API baseline,
English text, and whitespace checks passed. The strict audit reported
**479 errors in 95 files** across 418 source files. No remote tasks were submitted.

## CircuitIR sparse statevector execution

Sparse statevector execution now obtains wire count through ensure_circuit_ir,
so an existing CircuitIR does not need a nonexistent to_ir method. The expanded
backend test reproduced this failure before the fix. Circuit and CircuitIR
single-amplitude inputs now pass across all three local backends. Statevector
execution also rejects non-tensor backend results explicitly before indexing.

The target and adapter selections passed with **18 passed**, including two
invalid-result cases. Ruff, architecture, public API baseline, English text,
and whitespace checks passed. The strict audit reported **477 errors in 95
files** across 418 source files. The output-target Literal mismatch remains
unresolved; no remote compute or hardware was used.

## Shared output-target validation

The cost planner's existing target-name check now lives in one private function
that returns OutputTarget. The execution entry point uses it before calling
the typed planner. Accepted names and unsupported-target errors remain the same;
no unchecked cast or second validation table was added.

Target execution, backend cost selection, and adapter tests passed with
**34 passed**, including empty, misspelled, and differently cased output names.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. Both target_execution.py and backend_selection.py have no strict
diagnostics. The repository audit reported **479 errors in 95 files** across
420 source files; concurrent additions changed the source set from the prior
418-file audit, so aggregate totals are not a patch-only comparison.

## JAX wire and blocker collection types

Kernel observable wires and owner-local pullback wires now explicitly use
arbitrary-length tuples. MPS resource evidence keeps blocker lists during
collection and produces deduplicated tuples at the output boundary, rather
than overwriting lists with tuples. Blocker ordering, fields, and numerical
operations are unchanged.

The JAX distributed-plan selection passed with **99 passed and 6 skipped**
on CPU. Ruff, architecture, public API baseline, English text, and whitespace
checks passed. The strict audit reported **472 errors in 93 files** across
421 source files; concurrent changes again altered the checked source set.
No accelerator or multi-node capability was inferred from this local run.

## MPS Pauli matrix reuse

Pauli-product evaluation now resolves and converts each used X, Y, or Z matrix
once, then shares it across that axis's wires. Previously each wire repeated
the same device/dtype conversion. Fixed-matrix validation rejects callable
entries before observable contraction; wire overlap checks and axis order
are unchanged.

The MPS and sparse-output selections passed with 40 tests. Three additional
invalid-matrix cases then passed in the 5-test sparse-output module, yielding
**43 distinct passing tests**. Ruff, architecture, public API baseline, English
text, and whitespace checks passed. The strict audit reported **469 errors in
93 files** across 421 source files. GPU transfer savings were not benchmarked.

## Missing MPS channel operators

MPS instruction dispatch now rejects a channel with no Kraus operators before
entering trajectory execution. The explicit ValueError identifies the missing
input, and a regression verifies that failed dispatch leaves state tensors
unchanged. Valid channel execution is unaffected.

The noise and MPS sparse-output selections passed with **66 passed**. Ruff,
architecture, public API baseline, English text, and whitespace checks passed.
The strict audit reported **468 errors in 93 files** across 421 source files.
This validation covers absent operators, not every malformed channel payload.

## MPS discarded-weight accounting

Two-site and bucketed updates now read discarded_weight as a float once and
reuse it for truncation records and the positive-error check. This avoids
comparing a heterogeneous metadata value directly with zero or repeatedly
converting the same field. QR/SVD calculations and recorded values are unchanged.

The MPS and compiled-layer numerical selections passed. Ruff, architecture,
public API baseline, English text, and whitespace checks passed. The strict
audit reported **466 errors in 93 files** across 421 source files. The broader
split-metadata schema remains untyped by field; this is a local consumer cleanup.

## Remove unused autograd override suppressions

The split real/imag parameter-shift bridge no longer suppresses override errors
on its forward and backward methods. The current strict checker already accepts
these overrides, so both comments were unused. Signatures and gradient behavior
are unchanged.

The split real/imag autograd and optimizer contract selections passed with
**10 passed**. Ruff, architecture, public API baseline, English text, and
whitespace checks passed. The strict audit reported **464 errors in 93 files**
across 421 source files. Parameter-binding type diagnostics remain; none were
suppressed or weakened in this round.

## Autograd bridge binding dictionaries

Forward and backward bridge bindings now explicitly use the existing
str-or-Parameter key contract with Tensor values. This resolves dictionary-key
variance at executor calls without converting keys, changing parameter order,
or weakening the callee's accepted binding types.

The autograd and optimizer contract selections passed with **10 passed**.
Ruff, architecture, public API baseline, English text, and whitespace checks
passed. The strict audit reported **462 errors in 93 files** across 421 source
files. PyTorch Function.apply typing and the conformance runner's binding
declarations remain separate outstanding issues.

## Compiled MPS cache output structure

The rank-local compiled layer cache now describes its actual alternatives:
one updated site tensor, or two updated site tensors with split metadata.
The producer and forward consumer share this private type alias. Consumption
checks tuple width before unpacking and raises MPSForwardLifetimeError for a
cache entry belonging to the wrong gate width. Tensor computation, metadata
contents, and last-use cache removal remain unchanged.

The focused numerical, cache-lifetime, and initial-shape selections passed
with **11 passed**, including two injected malformed-cache cases. Ruff, Black,
architecture, public API baseline, and English text checks passed. The strict
Python 3.12 audit reported **460 errors in 91 files** across 421 source files;
the two edited production modules have no remaining diagnostics in that audit.
This is local CPU validation, not multi-rank or accelerator certification.

## Reassigned execution fallback policies

StrictExecutionScope accepts a mutable enum-or-string policy field, but its
route and fallback checks previously relied on enum identity after construction.
Reassigning the string "forbid" bypassed portable-route enforcement; reassigning
"host_debug_only" caused host-route acceptance to fail with AttributeError.
Both checks now normalize the current policy through the existing authoritative
FallbackPolicy method before enforcing it. Public signatures and audit records
are unchanged.

Two regression cases failed before the fix and pass afterward. The scope and
runtime lifecycle selections passed with **14 passed**. Ruff, Black,
architecture, public API baseline, English text, and whitespace checks passed.
The strict Python 3.12 audit reported **459 errors in 91 files** across 421
source files. The existing context-manager return annotation diagnostic remains
unresolved; this round did not alter its signature.

## Precision summary uses its declared return type

Removed a redundant cast from the selective Double-Single expectation summary.
The underlying state summary already returns dict[str, Any]; the caller now
uses that contract directly. Summary fields and numerical operations are unchanged.

The precision selection passed with **5 passed**. Ruff, Black, architecture,
public API baseline, English text, and whitespace checks passed. The strict
Python 3.12 audit reported **458 errors in 91 files** across 421 source files.
The precision module still has a separate parameter-binding key diagnostic.

## Interoperability adapter return contracts

Removed four redundant casts from the Qiskit and PennyLane adapters. Their
conversion functions already declare the concrete import and export result
types; adapters now return those results directly. Signatures, dependency
loading, conversion behavior, and lossy-conversion policy remain unchanged.

The selected adapter and interoperability contract tests passed with
**22 passed, 2 skipped**. The optional Qiskit and PennyLane integration modules
were skipped because those dependencies are unavailable in this environment;
this round does not establish real-framework conversion certification.
Ruff, Black, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit reported **454 errors in 89 files**
across 421 source files, with no remaining diagnostics in either adapter module.

## Hybrid model circuit construction

The two internal model builders now use Circuit.gate with explicit opcodes,
wires, and angle names, so their return types follow the declared Circuit
interface without relying on dynamically installed gate attributes. They read
the declared Tensor.device directly. Gate order, parameter expressions, builder
identities, and public model interfaces are unchanged.

All **5 hybrid model tests passed**, including classifier training, checkpoint
continuation, deployment binding, and native/JAX energy and gradient parity.
Ruff, Black, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit reported **450 errors in 89 files**
across 421 source files. Optional quantum-parameter annotations and the dynamic
PyTorch module-call return remain unresolved in this module.

## JAX MPS backward runtime availability

The minimal accelerator backward branch now explicitly requires both loaded
JAX modules before invoking them. An incomplete runtime raises RuntimeError
at branch entry rather than failing later on a missing module attribute.
Existing CPU selection and accelerator eligibility rules remain unchanged.
The new injected-runtime test verifies this failure without claiming GPU execution.

The JAX distributed-plan selection passed with **100 passed, 6 skipped** on
the local CPU environment. Ruff, Black, architecture, public API baseline,
English text, and whitespace checks passed. The strict Python 3.12 audit
reported **435 errors in 89 files** across 421 source files. The backward
module's heterogeneous memory-record typing remains unresolved. No accelerator
or multi-node execution was performed in this round.

## JAX backward memory totals before serialization

Per-rank peak byte counts are now calculated as integers before constructing
the heterogeneous memory records. The detailed records and aggregate plan reuse
the same tuple, avoiding a dictionary lookup and conversion back from metadata.
The byte-count formula and serialized values are unchanged.

The JAX distributed-plan selection passed with **100 passed, 6 skipped**.
Ruff, Black, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit reported **434 errors in 88 files**
across 421 source files; the minimal MPS backward module now has no diagnostics.
Validation remains local CPU evidence only.

## Explicit interoperability reference-circuit builders

Qiskit and PennyLane conformance reference circuits now use the declared
Circuit.gate interface instead of dynamically installed gate attributes.
Before editing, all five circuits' semantic fingerprints and CPU statevectors
were captured. After editing, every fingerprint and statevector matched exactly
(zero absolute and relative tolerance). Gate and wire order are unchanged.

The selected interoperability contract tests passed with **22 passed**.
Ruff, Black, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit reported **429 errors in 86 files**
across 421 source files, with no diagnostics in either edited conformance module.
The local reference comparison does not certify Qiskit or PennyLane execution;
their optional dependencies remain unavailable in this environment.

## Benchmark runner loading boundary

Resolved benchmark runners now validate that the imported entrypoint is callable
and that execution returns an integer exit code. A small checked callable keeps
execution lazy and preserves nonzero exit codes and raised runner exceptions.
Malformed registrations produce a runner-specific TypeError instead of leaking
an unchecked dynamic value through the declared callable contract.

The runner CLI, public surface, and boundary selection passed with **25 passed**,
including malformed entrypoints, invalid return values, lazy invocation, and a
nonzero exit code. Ruff, Black formatting, architecture, public API baseline,
English text, and whitespace checks passed. The strict Python 3.12 audit
reported **428 errors in 85 files** across 421 source files; the registry module
has no remaining diagnostics. Shape-only tensor-network planning still requires
a coordinated node and return-type correction and was not modified this round.

## Broad local regression after runtime and interoperability cleanup

The selection `(smoke or unit) and not gpu and not distributed and not slow`
passed with **1435 passed, 13 skipped, 1532 deselected, 1 warning** in
26.04 seconds. The warning is PyTorch's existing complex-module warning in
the invalid-dtype-before-mutation test. Whitespace validation passed.
This verifies the selected local regression tier, not all collected tests,
optional framework integrations, or distributed/accelerator execution.

The latest strict audit remains **428 errors in 85 files** across 421 source
files. Its largest categories are argument types (113), missing attributes (87),
missing function annotations (69), untyped calls (38), and Any returns (34).
No production source was changed during this verification round.

## MPS fixed gate matrix lookup

Five MPS lookup sites now share a private helper that verifies a fixed gate
entry is a Tensor before moving it to the state's device and dtype. This
distinguishes fixed Z, identity, and SWAP matrices from parameterized gate
functions in the same registry. Existing Pauli-observable validation is retained.
Normal matrix values, contraction paths, and swap counters are unchanged.

The MPS and sparse-output selection passed with **47 passed**, including three
malformed-registry cases that verify state tensors and swap counters remain
unchanged after rejection. Ruff, Black formatting, architecture, public API
baseline, English text, and whitespace checks passed. The strict Python 3.12
audit reported **423 errors in 84 files** across 421 source files; MPSState
has no remaining diagnostics in that audit. No hardware performance claim is made.

## Hybrid model deployment parameter ownership

Both maintained hybrid models now use one private extraction helper to reject
a missing quantum parameter tensor with ValueError before deployment payload
construction. Valid exports retain the detached clone and existing schema.
Regression tests cover both models and verify that modifying the exported tensor
does not mutate the model's live parameters.

All **9 hybrid model tests passed**, including the four new deployment cases;
the four new cases were rerun after final test annotation cleanup. Ruff, Black
formatting, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit reported **421 errors in 84 files**
across 421 source files. The dynamic PyTorch call return in VariationalEnergyModel
remains the model module's outstanding diagnostic.

## Energy model quantum output contract

VariationalEnergyModel now validates the Tensor result of the standard PyTorch
module call before summing it. It retains Module.__call__ so forward hooks and
autograd continue to run. A hook returning a non-Tensor now raises an explicit
TypeError at the model boundary. Tests verify both hook-adjusted values with
unchanged gradients and rejection of a mapping returned by a hook.

All **11 hybrid model tests passed**. Ruff, Black formatting, architecture,
public API baseline, English text, and whitespace checks passed. The strict
Python 3.12 audit reported **420 errors in 83 files** across 421 source files;
the maintained models module now has no remaining diagnostics in that audit.

## MPS reverse split metadata contract

The reverse executor now declares split information using the existing optional
Mapping return contract shared by local gate implementations. Two-site paths
reject missing split information before copying it into cached or recorded
metadata, using MPSReverseContractError. Numerical factorization and gradients
are unchanged; the separate shape/broadcast metadata schema remains unresolved.

The reverse numerical, reverse contract, and training selections passed with
**47 passed**. Ruff, Black, architecture, public API baseline, English text,
and whitespace checks passed. The strict Python 3.12 audit reported **417 errors
in 83 files** across 421 source files. This round's tests are local unit and
numerical checks; they do not certify distributed execution or exercise every
transport-dependent failure branch.

## Module checkpoint path adaptation

Module checkpoint methods now convert their accepted string or PathLike input
to pathlib.Path before calling the existing checkpoint helpers. This matches
the helpers' declared path contract without narrowing the public interface.
Checkpoint contents and the underlying save/load implementation are unchanged.
A custom __fspath__ object is exercised through a complete save/restore round trip.

The checkpoint and module training API selections passed with **23 passed**.
Ruff, Black, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit reported **415 errors in 83 files**
across 421 source files. Other Module typing diagnostics remain unresolved.

## Circuit builder invocation boundary

The private Module builder invocation captures and checks the callable before
inspecting its signature and calling it. This removes the assumption that the
Circuit-or-builder field must always be callable at this internal entrypoint.
One- and two-argument builder dispatch and the separate fixed-Circuit path are
unchanged. No public signature or export was modified.

The Module and training API selections passed with **53 passed, 1 warning**;
the warning is the existing PyTorch complex-module notice. Ruff, Black,
architecture, public API baseline, English text, and whitespace checks passed.
The strict Python 3.12 audit reported **412 errors in 83 files** across 421
source files. JAX builder inspection and the other Module diagnostics remain.

## Finite precision-policy tolerances

PrecisionPolicy now rejects non-finite absolute and relative tolerances as well
as non-positive values. NaN and positive infinity previously passed the positivity
comparison. Ten parameterized cases cover both fields and zero, negative, NaN,
and positive/negative infinity. The already validated, immutable parameter dtype
now maps directly to torch.float32 or torch.float64 instead of dynamic getattr.

The checkpoint/precision and Module selections passed with **68 passed, 1 warning**;
the warning is the existing PyTorch complex-module notice. Ruff, Black formatting,
architecture, public API baseline, English text, and whitespace checks passed.
The strict Python 3.12 audit reported **411 errors in 82 files** across 421
source files; the training-state module has no remaining diagnostics in that audit.

## Quantum natural-gradient state normalization

QNG now checks that the state norm is finite and positive before normalizing.
Zero states and NaN/Inf states raise ValueError at the state-function boundary
instead of propagating invalid values into the metric and linear solve.
Normalization uses Tensor.div, preserving its declared Tensor return type and
the differentiable expression for valid states.

All **12 hybrid optimization tests passed**, including the three invalid-state
cases and existing QNG energy reduction, staged Adam/L-BFGS, and Rotosolve checks.
Ruff, Black, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit reported **410 errors in 82 files**
across 421 source files. This remains local exact-state optimization evidence.

## Finite optimization-stage configuration

OptimizationStage now rejects NaN and infinite learning rates and damping
values at construction. The previous sign checks admitted NaN and positive
infinity, allowing invalid configuration to reach optimizer updates or the
QNG linear solve. Existing finite-value positivity and zero-damping rules
remain unchanged.

All **18 hybrid optimization tests passed**, including six new non-finite
configuration cases. Ruff, Black, architecture, public API baseline, English
text, and whitespace checks passed. This round changes runtime validation;
it does not claim a reduction from the latest strict audit of 410 diagnostics.

## Optimization-stage integer counts

Iteration counts, optimizer history size, and optional block size are now
checked for integer values at stage construction. Boolean and floating-point
counts fail before reaching range(), optimizer state allocation, or metric
partitioning. Ordinary integer subclasses remain accepted, and block_size=None
retains its existing meaning.

All **34 hybrid optimization tests passed**, including 16 invalid-count cases.
Those 16 cases were rerun after the final integer-subclass-preserving adjustment.
Ruff, Black formatting, architecture, public API baseline, English text, and
whitespace checks passed. No new full strict-audit count is claimed this round.

## MPS canonicalization helper state types

Five private canonicalization helpers now declare RankOwnedMPSState instead
of Any. They reuse the existing state contract without a new wrapper or protocol;
the import is type-checking-only. The owner helper uses the existing integer
return directly. Public signatures and QR/transfer operations are unchanged.

The local canonicalization, reverse numerical, and MPS selections passed with
**46 passed**. Ruff, Black, architecture, public API baseline, English text,
and whitespace checks passed. The strict Python 3.12 audit reported **409 errors
in 81 files** across 421 source files; the canonicalization module has no
remaining diagnostics. No cross-rank transport was exercised this round.

## Lowered noisy MPS result type

The internal lowered-program scheduler now declares MPSMonteCarloResult, matching
the trajectory runtime it directly returns and the higher-level noisy MPS entrypoint.
This replaces an unnecessary Any return annotation without changing scheduling,
sampling, result contents, or public exports.

The trajectory runtime selection passed with **24 passed**, and the numerical
noise selection passed with **60 passed**. Ruff, Black, architecture, public API
baseline, English text, and whitespace checks passed. The strict Python 3.12
audit reported **408 errors in 80 files** across 421 source files; the noisy MPS
executor module has no remaining diagnostics in that audit.

## Local statevector benchmark reference type

The sequential benchmark reference now explicitly declares its local state as
torch.Tensor, matching Circuit.initial_state and the numerical matrix-application
routine. This prevents an Any inferred through the lazy root export from
propagating to the return value. No benchmark algorithm or result field changed.

Both local performance contract tests passed (**2 passed**), including the
four-wire CPU reference comparison. Ruff, Black, architecture, public API baseline,
English text, and whitespace checks passed. The strict Python 3.12 audit reported
**407 errors in 79 files** across 421 source files; the local benchmark module
has no remaining diagnostics. The small CPU check is not new performance evidence.

## Statevector shard tensor fields

StatevectorShardState now declares its amplitudes and global_indices as
torch.Tensor instead of Any, matching all current PyTorch construction paths.
The type-checking-only import preserves lazy planning imports. Field order,
runtime storage, summary values, and JAX's separate shard record are unchanged.

The statevector forward, reverse, and training unit selections passed with
**53 passed**. Ruff, Black, architecture, public API baseline, English text,
and whitespace checks passed. The strict Python 3.12 audit reported **406 errors
in 79 files** across 421 source files. The change also resolves the stored-index
slice return diagnostic; it does not certify real distributed transport.

## Local distributed simulator result fields

LocalDistributedStatevectorResult now names torch.Tensor for its reconstructed
state and DistributedStatevectorPlan for its plan. These are the types produced
by its existing construction path. Its single-process development semantics,
full-state reconstruction count, and restrictions on scalability claims remain
unchanged.

The statevector forward, reverse, and training selections passed with **53 passed**.
Ruff, Black, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit remains **406 errors in 79 files**
across 421 source files; this round improves internal field precision without
claiming a diagnostic-count reduction.

## JAX Module builder selection

The JAX branch now captures and verifies the callable builder before signature
inspection, then passes that same builder to kernel compilation. This makes the
internal Circuit-or-callable distinction explicit without changing public
signatures, parameter dispatch, or fallback policy. Execution-plan serialization
and its compiler import ownership remain separate unresolved work.

The Module and hybrid-model selections passed with **55 passed, 1 warning**;
the warning is the existing PyTorch complex-module notice. Ruff, Black,
architecture, public API baseline, English text, and whitespace checks passed.
The strict Python 3.12 audit reported **405 errors in 79 files** across 421
source files. Numerical JAX checks used the local CPU backend.

## MPS P2P operation and callback types

The three private batched P2P helpers now accept lists of torch.distributed.P2POp
instead of Any. The overlap callback is an explicit zero-argument callable whose
return is ignored. Communication ordering, streams, timeout handling, and buffer
ownership are unchanged; no new runtime validation or wrappers were introduced.

All **12 selected P2P diagnostic, descriptor-cache, and send-pool tests passed**.
Ruff, Black formatting, architecture, public API baseline, English text, and
whitespace checks passed. The strict Python 3.12 audit remains **405 errors in
79 files** across 421 source files. Tests use local transport doubles; no real
cross-rank or GPU communication was run.

## MPS overlap callback failure cleanup

The CUDA overlap helper now enters its request-wait phase even when independent
local work raises. Stream joining and wait-time accounting also run in a finally
block. When communication completes successfully, the original callback exception
propagates. Previously a callback failure skipped request waits and stream joining.

All **14 selected transport tests passed**. Two new stream/work-double cases
verify normal completion and callback failure, including waiting on both issued
work handles before joining the current stream. Ruff, Black formatting,
architecture, public API baseline, English text, and whitespace checks passed.
This validates control flow with local doubles, not real CUDA transport failure
recovery. No new full strict-audit count is claimed this round.

## Broad regression after numerical validation and transport cleanup

The selection `(smoke or unit) and not gpu and not distributed and not slow`
passed with **1455 passed, 13 skipped, 1559 deselected, 1 warning** in
26.12 seconds. The warning is the existing PyTorch complex-module notice.
Architecture, public API baseline, English text, and whitespace checks passed.
The fresh strict Python 3.12 audit reported **405 errors in 79 files** across
421 source files.

This is the selected local regression tier. It does not include all optimization
or integration modules, optional dependencies, real cross-rank execution, or
accelerator certification. No production source changed in this verification round.

## Four-dimensional reverse MPS initialization shapes

ReverseMPSInitialization now records each site shape as exactly four integers:
batch, left bond, physical dimension, and right bond. This matches both validated
input tensors and generated initial states. The existing shape dictionary is
reused directly; no extra copy is introduced for large site counts.

The initial-shape, reverse contract, and training selections passed with
**45 passed**. Ruff, Black formatting, architecture, public API baseline,
English text, and whitespace checks passed. The strict Python 3.12 audit reported
**404 errors in 78 files** across 421 source files; the MPS state module now has
no remaining diagnostics. Reverse transport metadata typing remains separate work.

## Reverse metadata schema validation before integer conversion

The fixed-tensor record decoder now checks its one-dimensional, 21-field shape,
exact schema version, and finite positive integer dimensions before conversion.
Malformed dimensions and fractional versions previously could be silently
truncated; malformed record sizes could fail through indexing errors.
Valid records retain the same wire format and decoded contents.

The reverse contract, numerical, and training selections passed with **62 passed**,
including 15 new malformed-record cases. The existing collective round-trip test
now also compares the complete decoded payload. Ruff, Black formatting,
architecture, public API baseline, English text, and whitespace checks passed.
No real cross-rank collective or new full strict audit was run in this round.

## Shared reverse-record dimension validation

The encoder now validates dimensions before converting them to the floating-point
wire format. It shares the decoder's positive-integer check through one private
helper, rejecting fractional, non-finite, boolean, string, and non-positive
dimensions instead of truncating or coercing them first. Valid tensor shapes
retain the existing 21-field representation.

The reverse contract, numerical, and training selections passed with **69 passed**;
all **39 reverse contract tests** were rerun after the final numeric type adjustment.
Ruff, Black formatting, architecture, public API baseline, English text, and
whitespace checks passed. The strict Python 3.12 audit reported **404 errors in
78 files** across 421 source files. No real cross-rank transport was run.

## Typed reverse-record metadata across transport and execution

The existing input-shape, output-shape, and split-information record now has one
private TypedDict shared by reverse transport and its execution consumers.
Read-only encoder inputs accept mappings; decoded records retain their existing
mutable dictionaries and tensor wire format. Owner-local static QR validation
now checks for missing metadata explicitly before comparing shapes. Output
shapes are unpacked into the four dimensions required by the global shape table.
No public API snapshot or numerical algorithm was changed.

The reverse contract, numerical, and training selections passed with **69 passed**
after the final edits. Ruff, Black formatting, architecture, public API baseline,
English text, and whitespace checks passed. The strict Python 3.12 audit reported
**390 errors in 77 files** across 421 source files, down from 404 errors;
`reverse.py` and `reverse_transport.py` have no remaining diagnostics.
This is local CPU verification, not real cross-rank or GPU transport evidence.
Full-repository strict typing and hardware certification remain incomplete.

## Reverse cache and objective pipeline narrowing

The reverse segment cache now branches directly on the cached value, preserving
hit counters and identity-based record remapping while making the non-empty
branch explicit to type checking. The objective pipeline validates each parsed
slot before retaining it, so subsequent numerical work receives supported terms
without optional values, casts, or type suppressions. Unsupported terms still
raise before pipeline execution.

A new local CPU test compares three pipelined objectives with sequential
execution, including all local tensor gradients and a partial final window.
It also exercises unsupported objective rejection. Collectives are single-rank
test substitutes; this test does not establish real distributed transport.
The focused reverse, training, and pipeline selection passed with **70 passed**.
Ruff, Black formatting, architecture, public API baseline, English text, and
whitespace checks passed. The strict Python 3.12 audit reported **385 errors in
75 files** across 421 source files; reverse planning and Z/ZZ objective modules
now have no remaining diagnostics. Full strict typing and accelerator/multinode
verification remain unfinished.

## Resolve fixed observable matrices once per local scan

The generic reverse observable path resolves and checks each local site's fixed
matrix once, then reuses it in the left scan, right scan, and adjoint calculation.
A parameterized gate entry now raises a descriptive ValueError before the local
scan rather than failing when a function is treated as a tensor. Matrix device
and dtype conversion retain their existing semantics.

Six new numerical cases compare X, Y, and Z expectations and scalar-target MSE
objectives with direct complex128 tensor contractions and PyTorch autograd. A
separate case checks parameterized-observable rejection without initializing a
process group. The focused reverse, training, and objective selection passed
with **77 passed**. The numerical cases use a single-rank broadcast substitute;
no actual cross-rank failure coordination or accelerator transport was tested.
Ruff, Black formatting, architecture, public API baseline, English text, and
whitespace checks passed. The strict Python 3.12 audit reported **382 errors in
74 files** across 421 source files, with no remaining diagnostics in
`reverse_observables.py`. Repository-wide remediation remains incomplete.

## Explicit gate construction for split-real/imaginary conformance

The two internal split-real/imaginary conformance circuit builders now use
`Circuit.gate()` with explicit gate names and wire arguments. Removing dynamic
attribute calls also prevents the initial H gate from propagating an untyped
circuit through the training builder. Gate ordering, angles, parameter names,
precision, and device remain unchanged. Twelve complete serialized IR records
across four depths and two training seeds matched the pre-edit baseline exactly.

CPU statevector, training, and forward/training conformance tests passed with
**45 passed, 2 GPU cases deselected**. Ruff, Black formatting, architecture,
public API baseline, English text, and whitespace checks passed. The strict
Python 3.12 audit reported **374 errors in 74 files** across 421 source files,
down from 382. Parameter-binding type issues remain in this module. No new
hardware, throughput, or capacity evidence was produced.

## Reject non-finite adaptive MPS trajectory tolerances

Adaptive MPS trajectory execution now rejects NaN and infinite standard-error
targets before invoking a trajectory executor. Previously NaN and positive
infinity reached execution, allowing invalid stopping criteria. Finite targets
retain existing convergence behavior and finite non-positive inputs retain their
existing error message. Three regression cases cover non-finite targets; the
NaN and positive-infinity cases demonstrably reached the executor before the fix.

Trajectory lifecycle and noise tests passed with **87 passed**, including
deterministic stopping, unmet-target reporting, and checkpoint/resume behavior.
Ruff, Black formatting, architecture, public API baseline, English text, and
whitespace checks passed. No full strict audit or hardware run was repeated in
this round; the last strict baseline remains 374 errors in 74 files under
Python 3.12. Repository-wide remediation remains incomplete.

## CPU regression after reverse, conformance, and trajectory cleanup

The broader selection `(smoke or unit) and not gpu and not distributed and not
slow` passed with **1488 passed, 13 skipped, 1559 deselected, and 1 warning** in
23.84 seconds. The warning is PyTorch's existing complex-module notice in the
invalid-dtype module test. This selection verifies the accumulated local changes
without claiming coverage of excluded distributed, accelerator, or slow paths.

Architecture, protected public API baseline, English text, and whitespace checks
passed. The language scan checked 1909 paths with zero Han-script text matches;
binary images still require visual review. No source changes were needed for
this regression round. The last strict audit remains 374 errors in 74 files
under Python 3.12; full strict typing, Python 3.10 validation, and hardware
certification remain outstanding.

## Checked integer bounds in constant-loop analysis

Constant-loop analysis now retains each integer bound only after checking it,
so constructing the iteration range no longer relies on an optional-value tuple
whose aggregate validation was opaque to static analysis. Boolean, non-integer,
unknown-bound, and zero-step behavior is unchanged. No pass ordering or loop
unrolling limit changed.

Compiler pass and differential execution tests passed with **23 passed**,
including seeded optimized/unoptimized value and VJP comparisons and the
negative-step profile. Ruff, Black formatting, architecture, public API baseline,
English text, and whitespace checks passed. The strict Python 3.12 audit
reported **371 errors in 73 files** across 421 source files; the hybrid analysis
module has no remaining diagnostics. Full-repository typing and hardware
verification remain incomplete.

## Read-only names in the private compiler pass protocol

The private HybridProgramPass protocol now describes its name as a read-only
property. The pipeline only reads names, and built-in passes are frozen
dataclasses; requiring a writable attribute incorrectly excluded those existing
implementations from the protocol. Concrete passes, pipeline behavior, recorded
names, and protected public APIs are unchanged.

Compiler pass and differential execution tests passed with **23 passed**. Ruff,
Black formatting, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit reported **370 errors in 72 files**
across 421 source files; the pass pipeline module has no remaining diagnostics.
No hardware tests were run, and repository-wide remediation remains incomplete.

## Deterministic scalar arithmetic for adaptive MPS bond growth

Adaptive bond planning now uses `math.ceil` directly on the Python growth
calculation instead of constructing a tensor in the global default dtype. The
old float32 conversion rounded a growth factor of 1.00000001 down to 1 for a
rank-one bond, preventing requested growth when min_increment was zero. The
regression test failed before the fix: float32 suggested rank 1 while float64
suggested rank 2. Both now suggest rank 2 with identical complete plans.

MPS execution and reverse numerical tests passed with **43 passed**. Ruff, Black
formatting, architecture, public API baseline, English text, and whitespace
checks passed. Public signatures and recorded benchmark evidence were not
changed. No performance claim or hardware validation is implied by removing
this scalar tensor allocation. No full strict audit was repeated; its last
baseline remains 370 errors in 72 files under Python 3.12.

## Normalize trajectory checkpoint paths at the persistence boundary

The MPS trajectory runtime now converts supported PathLike inputs to Path before
calling checkpoint readers and writers. Resume checks directly establish the
non-null path already required by preflight. Public path annotations, checkpoint
schemas, and distributed rank-template behavior remain unchanged.

The existing interrupted/resumed trajectory parity scenario now runs with both
Path and a custom __fspath__ object. It verifies partial counts of 3 and 5, final
completion of 8, and exact mean/variance equality with uninterrupted execution.
Trajectory and noise tests passed with **88 passed**. Ruff, Black formatting,
architecture, public API baseline, English text, and whitespace checks passed.
The strict Python 3.12 audit reported **368 errors in 72 files** across 421
source files; optional-seed typing remains in the trajectory runtime. No remote
job or hardware test was run.

## Fail local preflight on non-finite numerical comparisons

Local fast-path preflight now reports a failed check when the state comparison
produces NaN or infinity. Previously a NaN comparison passed because it was not
greater than the tolerance. Fault-injected backend tests reproduced that false
pass before the fix and now verify both failed reports and raise_on_error.
Finite numerical comparisons and report fields retain their existing behavior.

All **5 local fast-path integration tests passed**, including the normal
statevector/MPS/tensor-network checks. Ruff, Black formatting, architecture,
public API baseline, English text, and whitespace checks passed. No full strict
audit was repeated; the last baseline is 368 errors in 72 files under Python
3.12. Similar comparisons in distributed parity and tolerance-input validation
remain to be reviewed separately. No real distributed or accelerator run was
performed.

## Reject non-finite development/production parity results

Both distributed parity comparison branches now fail explicitly on non-finite
numerical errors. Fault-injected statevector, MPS, and tensor-network cases
showed that NaN results could pass; comparing two infinite states could also
produce NaN and pass. Six cases now verify failed reports and the requiring
wrapper's exception behavior. Finite tolerances and layout comparisons are
unchanged.

Distributed backend policy and local preflight tests passed with **34 passed**.
These tests use local CPU orchestration and do not demonstrate real cross-rank
transport or production scalability. Ruff, Black formatting, architecture,
public API baseline, English text, and whitespace checks passed. Tolerance-input
validation remains separate follow-up work. No full strict audit was repeated;
the last baseline remains 368 errors in 72 files under Python 3.12.

## Validate preflight tolerances before numerical work

Local fast-path and development/production parity checks now require finite,
non-negative absolute tolerances. NaN and infinity can no longer disable useful
error comparisons, and negative tolerances fail at entry. Zero remains valid for
exact comparisons. Ten new cases cover the invalid values and zero-tolerance
acceptance in both entry points; existing result and fault-injection tests remain
unchanged.

All **44 backend policy and local preflight tests passed**. Ruff, Black
formatting, architecture, public API baseline, English text, and whitespace
checks passed. Validation used local CPU orchestration only, with no remote or
hardware jobs. The last strict audit remains 368 errors in 72 files under Python
3.12; it was not repeated in this round. Repository remediation remains ongoing.

## Statevector trajectory validation and checkpoint cleanup

Batched statevector trajectories now reject non-finite standard-error targets,
matching the corrected MPS runtime. Checkpoint readers and writers receive Path
objects for supported PathLike inputs. Resume ownership validation builds the
owned-ID set once instead of rebuilding it for each completed trajectory,
removing avoidable repeated work without changing accepted IDs.

Noise and trajectory tests passed with **92 passed**. Added cases cover NaN and
infinite targets plus custom __fspath__ checkpoint paths. The resumed
statevector workload matches continuous execution exactly in trajectory IDs,
expectations, and variance. Ruff, Black formatting, architecture, public API
baseline, English text, and whitespace checks passed. The strict Python 3.12
audit reported **366 errors in 71 files** across 421 source files; the
statevector noisy-runtime module now has no remaining diagnostics. No remote
job, hardware validation, or performance benchmark was run.

## Explicit and typed default preflight circuit builders

The local fast-path and distributed parity default builders now return Circuit
explicitly and construct gates through the existing gate method. Type-check-only
imports preserve lazy Circuit loading. Gate order, wires, parameters, and default
precision are unchanged: both complete serialized IR records matched the
pre-edit baseline exactly.

All **44 backend policy and local preflight tests passed**. Ruff, Black formatting,
architecture, public API baseline, English text, and whitespace checks passed.
The strict Python 3.12 audit reported **364 errors in 69 files** across 421 source
files; both preflight modules now have no remaining diagnostics. These local
checks do not establish accelerator or actual multi-node execution readiness.
Repository-wide remediation remains incomplete.

## Own the initial sample in online trajectory statistics

TensorWelford now clones its first observation instead of retaining the caller's
buffer. Reusing that buffer previously rewrote the accumulated mean before the
next update, corrupting both mean and variance. The new regression test
reproduced the mutation before the fix and now compares the two-sample result
with direct population statistics. Later updates already create new tensors;
only initial-sample ownership changed, and clone preserves autograd connectivity.

Noise and trajectory tests passed with **93 passed**, including checkpoint and
merge scenarios. Ruff, Black formatting, architecture, public API baseline,
English text, and whitespace checks passed. No hardware or performance
validation was run. The last strict baseline remains 364 errors in 69 files
under Python 3.12; no full audit was repeated in this round.

## Validate restored trajectory sample counts before conversion

Statistics restoration now requires an Integral sample count and excludes
booleans before converting to Python int. Previously fractional counts were
truncated, and booleans or numeric strings were silently accepted. Invalid NaN
and infinite counts now receive the same descriptive ValueError. Existing
non-negative and moment-consistency checks remain in place.

Five malformed-count cases failed before the fix. After correction, noise and
trajectory tests passed with **98 passed**, including normal checkpoint resume
and statistics merging. Ruff, Black formatting, architecture, public API
baseline, English text, and whitespace checks passed. The strict Python 3.12
audit remains **364 errors in 69 files** across 421 source files; no new typing
diagnostics were introduced. No remote or accelerator work was performed.

## Keep dynamic feedback actions inside validated wire scope

Physical-X feedback and frame-X updates now execute inside the branch that
checks active feedback modes and their target wire. The wire check explicitly
excludes None before membership and range checks. No-action feedback continues
to record its decision without applying a gate or changing the frame. No public
feedback schema or permitted-action policy changed.

Dynamic feedback, noise, circuit, and observability tests passed with **34 passed**.
Ruff, Black formatting, architecture, public API baseline, English text, and
whitespace checks passed. The current strict Python 3.12 audit reports
**362 errors in 68 files** across 422 source files; dynamic execution has no
remaining diagnostics. The source count increased elsewhere in the shared
working tree and is not attributed to this one-file change. No QPU or remote
job was submitted; repository-wide remediation remains incomplete.

## Explicit dynamic conformance circuit construction

The reset, conditional-flip, and qubit-reuse conformance builders now use the
explicit gate method while retaining their DynamicCircuit variables for
measurement and reset calls. All three complete serialized IR records matched
the pre-edit baseline exactly; expected outcomes and provider feature contracts
were not changed.

Dynamic conformance, feedback, and circuit tests reported **23 passed, 4 skipped**.
Skipped integrations are not certified by this run. Ruff, Black formatting,
architecture, public API baseline, English text, and whitespace checks passed.
The strict Python 3.12 audit reported **359 errors in 67 files** across 422 source
files; dynamic conformance now has no remaining diagnostics. No external
provider or hardware job was submitted. Repository remediation remains ongoing.

## CPU regression after preflight and statistics fixes

The broader selection `(smoke or unit) and not gpu and not distributed and not
slow` passed with **1496 passed, 13 skipped, 1592 deselected, and 1 warning** in
24.12 seconds. The warning is the existing PyTorch complex-module notice in
the invalid-dtype test. Counts describe the current shared working tree; they
are not all attributed to this remediation session. No source edits were
needed during this verification round.

Architecture, public API baseline, English text, and whitespace checks passed.
The language scan checked 1912 paths with zero Han-script text matches, without
certifying raster images. Excluded distributed, GPU, and slow paths remain
outside this run. The latest strict baseline is 359 errors in 67 files under
Python 3.12. Full typing and hardware certification remain incomplete.

## Explicit entangling gates in the hardware-efficient ansatz

The hardware-efficient ansatz now constructs fixed CX entanglers through the
existing gate method. Linear, circular, and unentangled layouts retain their
parameter ordering and gate sequence. Nine complete serialized IR records
covering those layouts at one, two, and three wires matched the pre-edit
baseline exactly. User-selected rotation dispatch was left unchanged.

Algorithm and hybrid optimization tests passed with **45 passed**. Ruff, Black
formatting, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit reported **356 errors in 67 files**
across 422 source files. Other algorithm-module diagnostics remain. No remote
job or hardware benchmark was run.

## Explicit gate construction for QAOA

QAOA preparation, weighted ZZ cost, and X mixer layers now use the explicit
gate method. One-layer and three-layer weighted-edge cases matched their
pre-edit complete IR, objective values, gamma gradients, and beta gradients
exactly (zero absolute and relative comparison tolerances). Public parameters
and accepted edge representations were not changed.

Algorithm and hybrid optimization tests passed with **45 passed**. Ruff, Black
formatting, architecture, public API baseline, English text, and whitespace
checks passed. The strict Python 3.12 audit reported **353 errors in 67 files**
across 422 source files. Remaining algorithm typing issues and hardware
verification are not resolved by this change.

## Explicit Heisenberg HVA gate construction

Heisenberg HVA preparation and evolution now use the existing gate method.
Rotation axes are represented by their gate names instead of dynamically
resolved bound methods. Shared-parity, bond-resolved, and phase-resolved
parameter indexing remain unchanged, as do Neel and dimer-singlet preparation.
Six four-wire, two-layer combinations matched their pre-edit complete IR,
energies, and parameter gradients exactly with zero comparison tolerances.

Algorithm, hybrid optimization, and MLP optimizer tests passed with **47 passed**.
Ruff, Black formatting, architecture, public API baseline, English text, and
whitespace checks passed. The strict Python 3.12 audit reported **344 errors in
67 files** across 422 source files. Remaining typing and hardware certification
work is not complete. No remote jobs or benchmark claims were introduced.

## Type the Hermitian eigenvalue result in ground-energy diagnostics

Ground-energy calculation now names and types the eigenvalue tensor returned by
the Hermitian eigensolver before taking its minimum. The redundant real-part
access was removed because these eigenvalues are already real. Dense matrix
construction, the 12-wire bound, and the public return type are unchanged.

Algorithm and hybrid optimization tests passed with **45 passed** after the
final edit, including the exact dimer-singlet ground-energy check. Ruff, Black
formatting, architecture, public API baseline, English text, and whitespace
checks passed. The final strict Python 3.12 audit reports **343 errors in 67
files** across 422 source files. No type suppression, hardware run, or new
scientific benchmark claim was introduced.

## Use the public linear-algebra exception in MPS factorization

MPS SVD validation now raises torch.linalg.LinAlgError rather than reaching into
torch._C. The installed runtime confirms both names identify the same exception
class and that it subclasses RuntimeError, so redundant exception tuples were
simplified without changing caught failures. Fault-injection tests use the
public name as well. The models import was moved into the import block, removing
an unnecessary E402 suppression.

SVD fallback, reverse numerical, and MPS tests passed with **45 passed**. Fallback
tests use CPU tensors and CUDA-branch substitutes; real GPU recovery remains
unverified. Ruff, Black formatting, architecture, public API baseline, English
text, and whitespace checks passed. Strict Python 3.12 checking reports
**339 errors in 67 files** across 422 source files. PyTorch typing does not
explicitly export the public LinAlgError alias, so that diagnostic remains
visible; no cast or type-ignore was added. Other factorization typing issues
remain separate work.

## Preserve the three-factor SVD type through CPU recovery

CPU SVD recovery now unpacks the left vectors, singular values, and right vectors
explicitly before moving them to the original device. This preserves the
existing three-tensor record type without casts or a variable-length tuple.
The fallback test now checks the contents of all three returned factors as
well as recovery counters and shape.

SVD fallback, reverse numerical, and MPS tests passed with **45 passed**. Ruff,
Black formatting, architecture, public API baseline, English text, and whitespace
checks passed. Strict Python 3.12 checking reports **338 errors in 67 files**
across 422 source files. The public exception-alias diagnostic remains. Recovery
was tested with CPU fault injection, not real GPU driver failures.

## Explicit memory-field validation for factorization workspaces

The workspace snapshot adapter now captures and checks free, total, allocated,
and reserved byte values individually before conversion. This removes optional
values from the successful path while preserving rejection of incomplete
provider snapshots. The provider test covers each missing field as well as the
complete snapshot, using a test provider without allocating CUDA tensors.

Workspace pool, planning, and benchmark-contract tests passed with **19 passed**
after formatting. Ruff, Black formatting, architecture, public API baseline,
English text, and whitespace checks passed. The strict Python 3.12 audit reports
**334 errors in 67 files** across 422 source files. The CUDA event typing issue
in this module remains. These checks do not certify real GPU memory behavior.

## Reject missing Kraus matrices before MPS channel execution

The lowered noisy-MPS boundary now rejects a channel instruction whose Kraus
matrices are absent with a descriptive ValueError. Previously None reached the
channel kernel and failed as a non-iterable value. The regression test reproduced
that path and now verifies the initial state is unchanged when the malformed
channel is rejected. Valid channel execution is unaffected.

MPS and noise tests passed with **104 passed**. Ruff, Black formatting,
architecture, public API baseline, English text, and whitespace checks passed.
The current strict Python 3.12 audit reports **334 errors in 67 files** across
423 source files; the noisy-MPS simulation module has no remaining diagnostics.
The shared tree gained source files elsewhere, so aggregate counts do not
represent this change alone. No remote or hardware work was performed.

## Preserve online statistics after a failed update

TensorWelford now computes replacement moments before publishing the new count
and tensor references. Previously an incompatible-device sample raised during
arithmetic after incrementing the count. The regression test reproduced count
2 after one successful sample and one failed update. It now verifies unchanged
statistics after failure and correct mean/variance after a subsequent valid
sample. Initial-sample allocations also precede state assignment.

Noise and trajectory tests passed with **99 passed**. Ruff, Black formatting,
architecture, public API baseline, English text, and whitespace checks passed.
The failure test uses a meta-device tensor and CPU storage; it does not certify
asynchronous CUDA error recovery or thread safety. The latest full strict
baseline remains 334 errors in 67 files under Python 3.12; no full audit was
repeated for this change. Repository-wide remediation remains incomplete.

## Reject restored moments on different devices

Statistics restoration now requires mean and second-moment tensors to share a
device, in addition to the existing shape checks. A CPU/meta regression case
previously restored successfully despite being unusable for normal updates; it
now raises a descriptive ValueError at restoration. Different dtypes remain
allowed because tensor promotion can support mixed-precision moments; this
change does not impose an unnecessary equal-dtype restriction.

Noise and trajectory tests passed with **100 passed**. Ruff, Black formatting,
architecture, public API baseline, English text, and whitespace checks passed.
No accelerator execution was used. The latest strict baseline remains 334
errors in 67 files under Python 3.12; this round did not repeat the full audit.
Repository-wide remediation remains incomplete.

## Make reverse-adjoint state requirements explicit

Reverse-adjoint execution now checks the exchange workspace and reversible
state directly at their use sites. Boolean dispatch decisions no longer hide
these requirements from static analysis. The CX segment optimization also
requires a reversible state, preventing rematerialization policies from entering
an optimization that accesses that state.

Two regression cases cover full rematerialization and interval checkpoints with
CX optimization availability substituted on CPU. Both compare the observable
and parameter gradient against dense complex128 execution. Removing the new
CX state guard in an isolated in-memory module caused both cases to enter the
accelerator-only branch; the repository source was not reverted. These tests
validate dispatch eligibility, not actual Triton kernel or multi-node execution.

Forward, reverse, training recovery, and exchange-contract tests passed with
**59 passed**. Ruff, Black, architecture, and the public API baseline passed.
Strict Python 3.12 checking reports **323 errors in 67 files** across 424 source
files. This module dropped from 12 diagnostics to one untyped PyTorch jvp call;
no type suppressions or casts were introduced. The shared tree gained another
source file independently. Repository-wide remediation remains incomplete.

## Preserve complex input precision in dense expectation evaluation

Dense operator expectations now pass the complex input state's dtype to the
internal Circuit. Previously a complex128 input was converted to the default
complex64 for operator application, then promoted back when contracted with
the original bra. Seeded single-state and batched regressions exposed errors
of roughly 1e-8 despite complex128 result tensors. The four cases cover
complex64 and complex128 values plus gradients with respect to both the ket
and operator; the double-precision cases now pass at 1e-12 tolerance. Real
inputs retain the existing runtime-configured complex dtype behavior.

Pauli-string evaluation uses the existing typed X, Y, and Z matrix constants
instead of looking them up in the mixed fixed/parameterized gate registry.
This removes an unnecessary callable-or-tensor ambiguity without a cast or a
new matrix definition.

Native circuit, algorithm, and Pauli measurement regressions passed with
**86 passed, 3 deselected**. The initial full file selection also ran three
single-rank distributed tests, all blocked by sandbox PermissionError when
binding a localhost TCP port; those tests remain unverified here. Ruff, Black,
architecture, public API baseline, and English-text checks passed. Strict
Python 3.12 checking reports **322 errors in 67 files** across 424 source files.
Six local-statevector diagnostics about undeclared Circuit runtime metadata
remain; this change does not claim that the module or repository is complete.

## Declare Circuit-owned statevector execution statistics

Circuit now owns an explicitly initialized private statistics dictionary, with
a TypedDict describing the existing boolean flag and integer counters. Keys
are optional because execution populates the record in stages and a fresh
circuit has no execution statistics. This replaces an undeclared dynamic
attribute, without changing counter names, values, public methods, or the
Simulation-owned execution loop.

Native-circuit and Module regressions passed with **114 passed, 3 deselected**
and one existing PyTorch complex-module warning. The three localhost-port
checks remain excluded because the sandbox denied them in the preceding round.
A direct lifecycle check covered a fresh circuit, cached state reads, mutation
and re-execution, and copying: cached reads retain their statistics and a new
execution replaces the record without mutating the previous record. Ruff,
Black, architecture, and public API checks passed. Strict Python 3.12 checking
reports **316 errors in 66 files** across 424 source files; the local statevector
simulation module has no remaining diagnostics. This does not certify whole-
repository typing, GPU execution, or distributed execution.

## Complete checkpoint planning and rematerialization typing

The tensor-network checkpoint payload now has an internal TypedDict matching
its existing serialized fields. Checkpoint selection uses one set for membership
checks while preserving forward-DAG order in the serialized selection; this
removes repeated set construction and linear tuple membership checks.
Rematerialization checks each operand directly and validates both returned
tensors inside materialize_pair, preserving the existing failure messages
without returning optional values under a non-optional annotation.

The distributed-TN DAG suite passed with **54 passed**, covering checkpoint
budget selection, full-tape versus rematerialized cotangents, and sliced reverse
parameter gradients. A separate comparison against the pre-edit Git index
confirmed identical complete plan summaries and identity hashes for seven
budgets, including zero and full storage. Ruff, Black, architecture, and public
API baseline checks passed. Strict Python 3.12 checking reports **309 errors
in 65 files** across 425 source files; checkpointing.py has no remaining
diagnostics. Another source file appeared in the shared tree independently.
These are CPU planning and numerical checks, not accelerator or multi-node
scalability evidence. Repository-wide remediation remains incomplete.

## Construct multi-axis plans through their existing data classes

Multi-axis peak plans and layouts now use explicit data-class constructor
arguments. Identity payloads are derived from those records, excluding the
identity field, and the immutable records receive their computed identity via
replace. This removes heterogeneous dictionary unpacking and the layout's
filtered reconstruction dictionary without adding another record schema.
The temporary record remains local until its identity is assigned.

The distributed-TN DAG suite passed with **54 passed**. Independent comparisons
against the pre-edit Git index verified identical complete summaries and
identity hashes for four layouts (including reordered shard axes) and two
peak plans with world sizes 2 and 4. Ruff, Black, architecture, and public API
baseline checks passed. Strict Python 3.12 checking reports **302 errors in
64 files** across 425 source files; multi_axis_sharding.py has no remaining
diagnostics. These checks validate CPU planning and numerical contracts,
not actual multi-GPU or multi-node execution. Repository-wide remediation
remains incomplete.

## Complete joint tensor-network plan construction typing

Joint planning now annotates checkpoint prefixes as variable-length value-ID
tuples with saved-byte and saved-cost integers. Its result is constructed with
explicit arguments to the existing data class; the identity payload is derived
from that record before the immutable result receives its computed hash.
This removes heterogeneous dictionary unpacking without changing the selection
objective, serialized fields, or introducing another plan schema.

The distributed-TN DAG suite passed with **54 passed**. Eight independent
comparisons against the pre-edit Git index covered world sizes 2 and 4,
checkpoint budgets 0 and 4096, and both absent and explicit working-set policies.
All complete summaries and identity hashes matched exactly. Ruff, Black,
architecture, and public API baseline checks passed. Strict Python 3.12 checking
reports **294 errors in 63 files** across 425 source files; joint_planning.py has
no remaining diagnostics. CPU planning tests do not certify actual accelerator
or multi-node memory behavior. Repository-wide remediation remains incomplete.

## Reject non-finite working-set multipliers before joint planning

Working-set policy validation now explicitly rejects non-finite kernel workspace
and communication buffer multipliers. Previously NaN and positive infinity
passed validation and reached subsequent byte arithmetic; negative infinity
was rejected only by the non-negative check. Six regression cases now require
clear finite-value errors at the policy boundary. Existing finite negative
value errors and valid policy arithmetic are unchanged.

The focused distributed-TN DAG suite passed with **60 passed**. To check the
accumulated statevector and TN changes, the CPU smoke/unit selection
`(smoke or unit) and not gpu and not distributed and not slow` was rerun:
**1513 passed, 13 skipped, 1616 deselected, 1 warning in 27.47s**. The warning
is the existing PyTorch complex-module warning. Ruff, Black, architecture, and
public API baseline checks passed. The latest full strict Python 3.12 baseline
remains **294 errors in 63 files**; no full typing audit was repeated for these
validation guards. This run does not cover GPU, multi-node, slow tests, or all
integration tests, and repository-wide remediation remains incomplete.

## Validate explicit backend policies at execution entry points

The consuming and non-consuming execution-option resolvers now require an
explicit distributed_backend_policy to be a DistributedBackendPolicy instance.
Previously strings and ordinary dictionaries could be returned as if they were
resolved policies, deferring failure until a downstream attribute access or
record replacement. Invalid inputs now raise a descriptive TypeError at the
resolver boundary; valid policy instances and profile-based selection retain
the existing behavior.

Backend-policy and local-fast-path tests passed with **48 passed**, including
four invalid-object regression cases across both resolvers. Ruff, Black
formatting, architecture, and public API baseline checks passed. Strict Python
3.12 checking reports **290 errors in 62 files** across 425 source files;
runtime/execution.py has no remaining diagnostics. The separate distributed
backend-policy helper remains follow-up work. No remote execution, accelerator,
or multi-node validation was performed. Repository-wide remediation remains
incomplete.

## Share the consuming distributed policy resolver

The common distributed backend-policy resolver now performs the same explicit
policy type check as execution entry points. Runtime execution imports that
existing helper under its established private name, removing its duplicate
implementation. MPS and tensor-network executors already use the common helper,
so consuming entry points now share policy validation, backend overrides, and
source attribution. The non-consuming peek resolver remains separate because
it has different mutation and provenance behavior.

Backend-policy and local-fast-path tests passed with **49 passed**. The added
override regression verifies consumed option keys, preservation of unrelated
options, source attribution, and no mutation of the original policy. Existing
invalid-object tests now exercise the shared consuming helper through its
execution alias. Ruff and Black passed after import/whitespace formatting;
architecture and public API baseline checks passed. Strict Python 3.12 checking
remains **290 errors in 62 files** across 425 source files. This round reduces
duplicate implementation and closes the sibling validation gap rather than
claiming additional typing progress. No remote or hardware execution occurred.

## Reject non-finite MPS growth independently of truncation state

Adaptive bond planning now rejects non-finite growth factors before inspecting
truncation hotspots. Previously NaN and positive infinity could pass unchanged
when there were no hot bonds, or fail later in rank arithmetic when truncation
had occurred. Local refinement inherits the same validation through its existing
adaptive-plan call. Finite growth behavior and existing below-one validation
remain unchanged.

Six regression cases cover NaN and both infinities on an untruncated state and
a Bell circuit truncated to bond dimension one. Each case checks adaptive and
local-refinement entry points. MPS and reverse numerical tests passed with
**49 passed**. Ruff, Black, architecture, and public API baseline checks passed.
The latest strict baseline remains **290 errors in 62 files**; no full typing
audit was repeated. The MPS planning mixin's undeclared state dependencies are
still unresolved and are not concealed by casts or diagnostic suppressions.
No hardware or remote execution occurred. Repository remediation is incomplete.

## Narrow executable-plan integer fields at consumption

Plan execution now reads world_size and batch_size once, explicitly verifies
positive non-boolean integers, and reuses those values for mode selection and
execution arguments. The existing plan validator already enforces this rule;
the local guards make the invariant visible where the generic decision mapping
is consumed, without casts, coercing strings/floats, or trusting a separate
mirror field on the plan. Error codes and messages match the contract validator.

Execution-plan semantics, option semantics, and plan assembly tests passed with
**21 passed**. Ruff, architecture, and public API baseline checks passed; Black
formatting was applied. Strict Python 3.12 checking reports **287 errors in
61 files** across 425 source files, with no remaining diagnostics in
runtime/plan_execution.py. The generic decision mapping remains an upstream
typing limitation; this change does not claim a new runtime bug was discovered
in previously validated plans. No accelerator or remote execution occurred.

## Route scalar wire inputs through the existing wire validator

Optional output-wire normalization now recognizes int instances before applying
the existing exact-int validator. Booleans remain invalid: they are routed to
_wire and rejected instead of being treated as iterables. Scalar and sequence
forms now report the same non-negative-integer ValueError for boolean and
negative wire inputs; previously scalar booleans raised an incidental iterable
TypeError. Valid integer, iterable, and unspecified wire behavior is unchanged.

Observable-output, modern measurement, and Pauli measurement tests passed with
**40 passed**, including nine scalar/sequence consistency cases across
probabilities, samples, and counts. Ruff, Black formatting, architecture, and
public API baseline checks passed. Strict Python 3.12 checking reports
**288 errors in 61 files** across 426 source files; the observables module has
no remaining diagnostics. Other changes in the shared worktree increased the
aggregate count independently, so the total is not a measure of this edit alone.
No remote or hardware execution occurred. Repository remediation is incomplete.

## Validate decoded twin metadata at the JSON boundary

Twin experiment metadata normalization now treats the JSON decoder result as
an object and checks that it is a dictionary before returning it. The encoder
already receives a concrete dictionary, so this makes the existing return
invariant explicit without trusting the decoder's Any annotation or adding a
cast. JSON ordering, recursive copying, and non-finite rejection are unchanged.

Twin experiment and digital-twin tests passed with **16 passed** using local
fixtures and provider substitutes. A direct normalization check verified sorted
keys, tuple-to-list conversion, independence from the source's nested values,
and rejection of nested NaN and both infinities. Ruff, Black, architecture,
and public API baseline checks passed. Strict Python 3.12 checking reports
**287 errors in 60 files** across 426 source files; twin/experiment.py has no
remaining diagnostics. No real provider submission or hardware execution was
performed. Repository-wide remediation remains incomplete.

## Make local remote-target count keys explicit

The in-memory remote target now explicitly requests binary Circuit counts and
constructs string-keyed result counts for DeploymentResult. Circuit.counts
supports both integer and binary output modes through one union return type;
this adapter requires only binary strings. Explicit binary selection preserves
bit width and leading zeros, while count values remain unchanged. No public
signature, receipt, or metadata field changed.

Remote adapter characterization, remote expectation, open-source golden paths,
and Pauli measurement tests passed with **18 passed, 1 skipped**. The skipped
case requires optional Qiskit. Existing characterization checks cover concrete
binary outcomes and shot totals. Ruff, Black, architecture, and public API
baseline checks passed. Strict Python 3.12 checking reports **286 errors in
59 files** across 427 source files; testing/remote.py has no remaining
diagnostics. The shared tree gained another source file independently. These
are local provider substitutes; no external submission or hardware execution
was performed. Repository-wide remediation remains incomplete.

## Avoid tensor allocation during shape-only size inspection

The shape-only tensor record now reads dtype.itemsize directly instead of
allocating a scalar torch.empty tensor for every element_size call. The existing
Python-integer shape product is retained, including shapes too large for an
actual PyTorch tensor. This removes numerical storage allocation from size
inspection without changing the shape representation or contraction algorithm.

Four regression cases cover float32, float64, complex64, and complex128 with a
logical element count of 2**128. They reject any torch.empty call during size
inspection and compare element sizes against real scalar tensors created before
the allocation guard. All four cases failed against the prior implementation.
Path-search and tensor-network tests passed with **83 passed, 1 skipped**.
Ruff, Black, architecture, and public API baseline checks passed. Strict Python
3.12 checking remains **286 errors in 59 files** across 427 source files; the
shape-only versus real-tensor typing issues remain unresolved. No allocation
speedup or hardware capacity benchmark is claimed. Repository remediation is
incomplete.

## Preserve field types throughout execution-option resolution

Execution options now resolve through typed field getters in descending source
priority, then construct the existing ResolvedExecutionOptions record explicitly.
A local generic selector records provenance while distinguishing None from
explicit False and zero. The mutable global defaults dictionary and generic
resolved-value dictionary are removed. Program batch and precision checks now
read the typed resolved record, preserving their existing conflict messages.

Option, plan, and execution semantics tests passed with **35 passed**. A seeded
comparison against the pre-edit Git index checked 250 combinations of call,
policy, and program options with a fixed runtime configuration: resolved values,
ordered provenance records, and conflict exception types/messages all matched.
Ruff, architecture, and public API baseline checks passed; Black formatting was
applied. Strict Python 3.12 checking reports **282 errors in 58 files** across
427 source files; options_resolver.py has no remaining diagnostics. This round
adds no public option or dependency. No remote or hardware execution occurred.
Repository-wide remediation remains incomplete.

## Restore rejection of explicit option-subclass extensions

Follow-up review found a regression in the preceding typed resolver refactor:
an ExecutionOptions subclass with an additional non-None data-class field was
silently ignored. The former dictionary construction rejected it when building
ResolvedExecutionOptions. Input validation now rejects explicit unsupported
subclass fields directly, naming the input source and offending field. Extra
fields left as None remain unspecified. Ordinary ExecutionOptions instances
retain a direct validation path without additional field traversal.

Nine regression cases cover all three supplied option sources and additional
values None, zero, and two; the six explicit-value cases failed before the fix.
Option and plan semantics tests passed with **44 passed**. Ruff, architecture,
and public API baseline checks passed; Black formatting was applied. Strict
Python 3.12 checking reports **283 errors in 59 files** across 428 source files.
The options resolver remains free of diagnostics; the shared tree gained another
source file independently. No remote or hardware execution occurred. Repository
remediation remains incomplete.

## Recheck accumulated changes and align the README assertion

A broad CPU regression initially reported one failure and 1537 passes. The
failure was a stale literal assertion: README edits elsewhere in the shared
worktree renamed build_program/trained_program to circuit/trained_circuit.
The example still reconstructs the circuit from detached trained parameters.
The assertion now checks the same operation using the current example names;
README content and the behavioral training test were not changed.

Documentation alignment and the executable README example passed with
**5 passed**, including ten completed training steps, decreasing loss, and
agreement between the measured expectation and cos(theta). The repeated
`(smoke or unit) and not gpu and not distributed and not slow` selection passed
with **1538 passed, 13 skipped, 1650 deselected, 1 warning in 27.32s**. The warning
is the existing PyTorch complex-module warning. Recent implementation files
passed Ruff and Black checks; architecture, public API baseline, and English
text checks passed. The updated assertion was formatted with Black.
The latest full typing baseline remains **283 errors in 59 files**; this round
did not repeat it. GPU, distributed, slow, and full integration coverage remain
outside this run. Repository-wide remediation is incomplete.

## Align conformance binding dictionaries with the existing key domain

The P1 training and P2 precision conformance builders now explicitly declare
binding dictionaries with str-or-Parameter keys and Tensor values, matching the
existing parameter-binding interfaces. Previously inference narrowed the keys
to str, which is incompatible with the invariant Mapping key parameter at
those call sites. Contents, tensor creation, gradient flags, and numerical
execution are unchanged; no dictionary copies, casts, suppressions, or public
signature changes were introduced.

Split real/imag training, precision, and evidence tests passed with **20 passed**.
Ruff, Black, architecture, and public API baseline checks passed. Strict Python
3.12 checking reports **278 errors in 58 files** across 428 source files, down
five diagnostics. split_real_imag_precision.py has no remaining diagnostics;
the P1 module still has a separate normalized-binding boundary issue. Other
conformance binding paths remain follow-up work. No accelerator or remote
execution occurred. Repository-wide remediation is incomplete.

## Align P3 and P4 conformance binding construction

P3 conformance now declares its literal binding dictionary with the existing
str-or-Parameter key domain. The private P4 binding builder returns the same
key-domain type, retaining Tensor values. Normalized reference bindings remain
string-keyed at their existing boundary. Neither change alters dictionary
contents, precision, device placement, reference programs, or gradients.
No casts, ignored diagnostics, additional copies, or public signatures changed.

P3/P4 numerical, evidence, and contract tests passed with **29 passed**. Ruff,
Black, architecture, and public API baseline checks passed. Strict Python 3.12
checking reports **271 errors in 56 files** across 428 source files, seven fewer
diagnostics. Both edited conformance modules have no remaining diagnostics.
The P5 autograd/optimizer paths and normalized binding interfaces still need
work. Tests ran locally on CPU, not on a production accelerator or remote
cluster. Repository-wide remediation remains incomplete.

## Type P5 conformance bindings and use explicit workload gates

P5 autograd conformance now declares the existing str-or-Parameter binding key
domain and orders gradients by the validated circuit's parameter names. The
optimizer conformance builders use explicit gate calls and typed initial/FP32
binding dictionaries. The normalized CPU reference mappings remain unchanged.
VQE and QAOA circuit IR, observables, initial parameter values, and learning
rates matched the pre-edit Git index exactly in a separate comparison.

Full CPU autograd and optimizer trajectory conformance, Double-Single SGD, and
P5 contract tests passed with **10 passed in 58.62s**. Ruff passed after removing
an extra import separator; Black, architecture, and public API baseline checks
passed. Strict Python 3.12 checking reports **263 errors in 55 files** across
428 source files, eight fewer diagnostics. Autograd conformance is clean; the
optimizer conformance module retains one candidate-loss binding boundary issue.
These are CPU reference checks, not native accelerator or distributed training
evidence. No remote jobs were submitted. Repository remediation is incomplete.


## Reject tensor-network label and dimension mismatches before path search

The shared dimension collector now checks that every node has one label per
axis before pairing labels and dimensions. Previously zip silently truncated
mismatched inputs: missing labels could pass shape-only planning, while extra
labels could trigger a later KeyError. Invalid nodes now raise ValueError with
the node name and both counts. Valid contractions and public signatures are
unchanged. The check belongs to local contraction planning and adds no new
abstraction or execution mode.

Six regression cases cover missing and extra labels in greedy, beam, and
optimal shape-only search. All six failed before the fix. Focused path-search,
tensor-network, and distributed DAG tests now pass with **149 passed, 1 skipped
in 6.46s**. Ruff, Black, architecture, and public API baseline checks passed.
The latest full strict Python 3.12 baseline remains **263 errors in 55 files**;
this validation-only guard did not trigger another typing audit. Tests ran
locally and do not establish accelerator or multi-node scalability.
Repository-wide remediation remains incomplete.


## Preserve typed rank measurements in TN memory evidence

The memory evidence builder now constructs its existing frozen dataclass with
explicit fields and retains DistributedTNRankMemoryMeasurement objects in
rank_measurements. Previously it serialized the records to dictionaries before
placing them in the object, contradicting its declared type and breaking rank
attribute access. Serialization now occurs when forming the identity payload;
asdict and replace preserve the existing summary format and identity inputs.
No public signatures, schemas, claim classifications, or recorded evidence were
changed. This is internal evidence assembly for sharded_across_ranks plans.

The extended regression failed before the fix and now verifies typed records,
rank access, and the serialized measurement values. Three pre-edit/current
comparisons preserved complete summaries and hashes, covering both passing and
failed calibration evidence. Distributed TN DAG and benchmark contract tests
passed with **240 passed in 4.20s**. Ruff, Black, architecture, and public API
baseline checks passed. Strict Python 3.12 checking reports **257 errors in
54 files** across 428 source files, six fewer diagnostics; memory_evidence.py
is now clean. These local contract tests do not establish GPU or multi-node
scalability. Repository-wide remediation remains incomplete.


## Reject non-finite TN memory calibration tolerances

The evidence builder now rejects NaN and infinite underprediction tolerances
before evaluating rank measurements. Positive infinity previously removed the
prediction tolerance bound, while NaN produced failed evidence containing a
non-finite configuration value. Negative infinity already failed the minimum
bound check and now receives the explicit finite-value diagnostic. Finite
threshold behavior, evidence fields, identities, and capability claims are
unchanged; no new public API or abstraction was introduced.

Three regression cases failed before the change: NaN and positive infinity
were accepted, and negative infinity reported only the minimum bound error.
Distributed TN DAG and benchmark contract tests passed with **243 passed in
4.17s**. Ruff, Black, architecture, and public API baseline checks passed. The
latest strict Python 3.12 baseline remains **257 errors in 54 files**; this small
input guard did not repeat the full typing audit. Local contract tests do not
establish accelerator or multi-node scalability. Repository-wide remediation
remains incomplete.


## Make forward exchange workspace narrowing explicit

The pairwise and subgroup statevector forward exchange paths now check the
optional workspace directly at pipeline counter updates and prefetch sites.
The existing pipelined flag already required a workspace; the explicit checks
make that invariant visible to static analysis without casts, suppressions,
new abstractions, or public signature changes. Communication scheduling,
allocation, numerical kernels, and counter semantics are unchanged. This is
internal orchestration for sharded_across_ranks execution, not a new capability.

Forward, reverse, training, and exchange-overlap contract tests passed with
**59 passed in 3.14s**. Ruff, Black, architecture, and public API baseline checks
passed. Strict Python 3.12 checking reports **252 errors in 53 files** across
428 source files, five fewer diagnostics; forward.py is now clean. Local tests
do not exercise real NCCL overlap or establish accelerator and multi-node
scalability. Repository-wide remediation remains incomplete.


## Reuse validated execution-plan serialization

plan_to_dict now returns the normalized copy produced by validate_plan_payload
instead of discarding it and repeating the JSON round trip. The validator checks
the decoded object's dictionary shape before returning it, making the boundary
explicit without casts or suppressions. Public signatures, payload schemas,
identity inputs, and rejection rules remain unchanged. This removes redundant
work on the local execution-plan serialization path without a new abstraction.

Execution-plan semantic, candidate, and assembly checks passed with **22 passed
in 0.80s**. Nine additional pre-edit/current comparisons across statevector,
MPS, and tensor-network plans preserved full payloads and identities; mutating
returned nested dictionaries did not change subsequent exports. Ruff, Black,
architecture, and public API baseline checks passed. This change removes two
strict typing diagnostics. The shared tree now reports **251 errors in 54
files** across 428 source files: an additional core/_artifacts.py diagnostic
appeared independently since the previous audit. That file was not edited in
this round. The two planner re-export diagnostics in execution_plan_contract.py
remain. No accelerator or remote execution occurred. Repository-wide
remediation remains incomplete.


## Reject boolean checkpoint generation counters before restoration

The private MPS checkpoint loader now requires an exact integer for the saved
completed step, matching a non-negative generation counter. Previously bool
passed isinstance(..., int), allowing True and False to restore training state
as if they were valid counters. Valid checkpoint serialization, public APIs,
and numerical execution remain unchanged. Validation still precedes optimizer,
parameter, and RNG restoration; this does not claim transactional rollback for
all possible restoration failures.

Five malformed-counter cases cover booleans, a negative integer, a fractional
value, and a missing value, checking that parameters and CPU RNG state remain
unchanged after rejection. The two boolean cases failed before the fix; the
other three preserved existing rejection behavior. MPS training/checkpoint
checks passed with **31 passed in 0.84s**. Ruff, architecture, and public API
baseline checks passed; Black formatted the added test. The latest full strict
Python 3.12 baseline remains **251 errors in 54 files**; this input guard did not
repeat the audit. The optimizer-state typing boundary remains unresolved.
No GPU or remote job ran. Repository-wide remediation remains incomplete.


## Reject ambiguous MPS checkpoint parameter membership

The private checkpoint loader now rejects distinct saved keys that normalize to
the same parameter index, such as 0 and "0". Previously set construction hid the
duplicate and restoration wrote the same live parameter twice. Integer
conversion overflow now uses the existing MPSTrainingError invalid-index
category, rather than leaking OverflowError. Unique convertible keys retain
their existing behavior; this change does not narrow all accepted index types.
Valid checkpoint schema, numerical behavior, and public APIs are unchanged.

Four regression cases cover two duplicate aliases and positive/negative
infinity. All failed before the fix. Tests also verify parameter and CPU RNG
state preservation after rejection. MPS training/checkpoint checks passed with
**35 passed in 1.07s**. Ruff, Black, architecture, and public API baseline checks
passed. The latest full strict Python 3.12 baseline remains **251 errors in
54 files**; this validation change did not repeat that audit. Optimizer-state
typing remains unresolved, and these checks do not establish transactional
rollback for failures during restoration. No accelerator or remote jobs ran.
Repository-wide remediation remains incomplete.


## Normalize checkpoint optimizer state at the PyTorch boundary

The checkpoint loader now explicitly narrows the validated optimizer state at
restoration and passes a shallow dictionary to PyTorch's dict-typed
load_state_dict interface. General Mapping inputs remain accepted; tensor and
nested optimizer values are retained without deep copies. Existing early
validation still rejects missing or malformed optimizer state. No casts,
suppressions, public signatures, or serialized checkpoint fields changed.

Checkpoint test helpers now accept a typed optional optimizer. Two continuation
tests cover Adam and momentum SGD: after changing live state and learning rate,
restoration recovers the saved parameter and learning rate, and repeating the
next update matches uninterrupted execution. MPS training/checkpoint tests
passed with **37 passed in 1.34s**. Ruff, Black, architecture, and public API
baseline checks passed. Strict Python 3.12 checking reports **250 errors in
53 files** across 428 source files, one fewer diagnostic; MPS checkpointing.py
is now clean. This does not establish atomic rollback of arbitrary optimizer
load failures or distributed recovery on hardware. Repository-wide remediation
remains incomplete.


## Resolve TN calibration device names through Compute discovery

Sparse tensor-network memory preflight no longer reads a nonexistent name
attribute from torch.device. CUDA-backed inputs previously reached this lookup
even when no measured calibration was requested. The calibrated path now obtains
the matching available device record from the existing Compute runtime and uses
its model name; missing device records fail explicitly. Uncalibrated preflight
performs no device discovery. CPU calibration behavior is preserved. No vendor
API, public signature, serialized evidence, or capability claim was added.

Four mocked CUDA boundary cases reproduced the AttributeError before the fix.
Final tests cover the selected device index rather than the first inventory
entry, unavailable-device rejection, and avoiding discovery without calibration.
Together with tensor-network tests they passed with **75 passed, 1 skipped in
4.48s**. Ruff, Black formatting, architecture, and public API baseline checks
passed. Strict Python 3.12 checking reports **249 errors in 53 files** across
428 source files, one fewer diagnostic. The tests ran on CPU with a mocked CUDA
reference and discovery provider; they do not verify real GPU kernels or
multi-node behavior. Repository-wide remediation remains incomplete.


## Type the TN broadcast slot and budget rejection boundary

The sparse tensor-network executor now declares its broadcast slot as an
optional pair of the existing slicing plan and contraction steps. This matches
both rank-zero construction and the initially empty receiving slot, instead of
letting inference restrict the list to None. The budget rejection branch also
checks the optional budget directly before formatting it. Its preceding
budget_satisfied calculation already guaranteed that invariant. No runtime
payload, communication order, numeric behavior, public API, or new abstraction
changed; no casts or suppressions were introduced.

Tensor-network, calibration-device, and distributed DAG tests passed with
**138 passed, 1 skipped in 6.33s**. Ruff, Black, architecture, and public API
baseline checks passed. Strict Python 3.12 checking reports **247 errors in
53 files** across 428 source files, two fewer diagnostics. The execution module
still has the existing PyTorch autograd Function.apply typing boundary. These
local tests do not verify real inter-rank broadcast transport or accelerator
scalability. Repository-wide remediation remains incomplete.


## Narrow TN redistribution shard metadata before use

The single-axis redistribution planner now captures and explicitly checks both
layouts' shard axes and labels before integer normalization and block planning.
The layout constructors already require this metadata for sharded layouts;
the local check makes that invariant visible at the consumer. Existing integer
normalization, public signatures, serialization, and communication semantics are
retained. No casts, suppressions, or additional record types were introduced.

Distributed TN DAG tests passed with **63 passed in 2.98s**. Eight pre-edit/current
comparisons across same-axis and cross-axis redistribution with two or three
source/destination ranks preserved complete plans, including block ranges,
byte counts, and identities. Ruff, Black, architecture, and public API baseline
checks passed. Strict Python 3.12 checking reports **241 errors in 52 files**
across 428 source files, six fewer diagnostics; redistribution.py is now clean.
These are local planning and reconstruction checks, not validation of real
multi-node transport. Repository-wide remediation remains incomplete.


## Construct partial-mesh layouts with explicit typed fields

The partial-mesh layout planner now constructs its existing frozen dataclass
with named fields instead of unpacking a heterogeneous dictionary. Its identity
is computed from asdict without the identity field and assigned with replace.
The serialized layout, field order, partition/replication classification, and
hash inputs are unchanged. No new contract, record class, casts, or typing
suppressions were introduced.

Distributed TN DAG tests passed with **63 passed in 3.10s**. Six pre-edit/current
comparisons covered fully partitioned, mixed partitioned/replicated, entirely
replicated, and reordered meshes; complete summaries and hashes matched. Ruff,
architecture, and public API baseline checks passed; Black formatted the edited
module. Strict Python 3.12 checking reports **238 errors in 51 files** across
428 source files, three fewer diagnostics; partial_mesh.py is now clean. The
partial_mesh_partition_with_replication classification is retained and is not
promoted to capacity-scaling evidence. No remote or hardware execution occurred.
Repository-wide remediation remains incomplete.


## Enforce independent readout at the dynamic sampling boundary

The internal readout dispatcher now rejects CorrelatedReadoutError before
calling the independent-bit sampler, using the same message as the existing
whole-circuit validator. Whole-circuit validation remains in place. This keeps
the bounded dynamic profile explicit at the numerical boundary and narrows the
union to the supported error type without casts or suppressions. Public APIs
and supported independent-readout behavior remain unchanged.

The new direct-boundary regression failed before the fix because correlated
readout reached sampling. It now verifies rejection without changing input bits
or consuming generator state. Dynamic noise and feedback integration tests
passed with **14 passed in 0.78s**. Ruff, Black, architecture, and public API
baseline checks passed. Strict Python 3.12 checking reports **237 errors in
50 files** across 428 source files, one fewer diagnostic; dynamic/_noise.py is
now clean. These are local execution tests; no remote jobs or hardware runs
occurred. Repository-wide remediation remains incomplete.


## Make checkpoint trajectory seeds explicit at consumption sites

The MPS trajectory resume branch now narrows the optional seed directly, and
the progress writer explicitly enforces the same seed requirement before
constructing a checkpoint. Entry validation already rejects checkpointed runs
without a seed; these checks preserve that invariant at the two consumers.
Seed derivation, checkpoint schema, resumed trajectory ownership, and random
streams remain unchanged. No public signatures, casts, or suppressions changed.

Trajectory runtime tests passed with **36 passed in 1.30s**, including resumed
execution, rank-local checkpoint isolation, and deterministic random streams.
Ruff, Black, architecture, and public API baseline checks passed. Strict Python
3.12 checking reports **235 errors in 49 files** across 428 source files, two
fewer diagnostics; trajectories/mps.py is now clean. Tests ran locally and do
not establish hardware recovery or multi-node execution. Repository-wide
remediation remains incomplete.


## Narrow sharded pair labels and verify cumulative CPU changes

Sharded pair-mode selection now checks optional output and contracted-input
labels directly before integer normalization. Existing DAG/layout validation
already requires these labels; the consumer checks expose that invariant to
static analysis without casts or public contract changes. Retained-output and
contracted-partial-reduction semantics are unchanged.

After the final guard change, distributed TN DAG tests passed with **63 passed
in 2.77s**, and Ruff/Black passed. Architecture and public API baseline checks
also passed during this round. Strict Python 3.12 checking reports **234 errors
in 48 files** across 428 source files, one fewer diagnostic; sharded_kernels.py
is now clean.

A broader CPU selection was run to check cumulative runtime/checkpoint changes:
`(smoke or unit) and not gpu and not distributed and not slow` passed with
**1582 passed, 13 skipped, 1658 deselected, 1 warning in 27.55s**. It started before
the final contracted-input guard; the final guard was checked by the focused
run above. The warning is the existing PyTorch complex-module warning. This is
not full integration, accelerator, or multi-node coverage. Repository-wide
remediation remains incomplete.


## Access the dynamically registered Torch-FL device namespace explicitly

The Compute adapter now resolves torch.flagos through getattr after its existing
activation and registration checks. FlagOS is registered dynamically by
Torch-FL and is not a statically declared PyTorch attribute. The lookup remains
fixed to the registered namespace even when device_type is adapted for tests;
using device_type initially broke the existing privateuseone test and was
corrected. Lazy activation, optional-dependency failure behavior, and returned
vendor objects are unchanged. The existing dynamic vendor boundary remains
Any-typed; this is not a claim of full static coverage of Torch-FL APIs.

Platform runtime and provider-consistency tests passed with **23 passed in
0.42s**, including missing-dependency, registration retry, discovery, memory,
and lifecycle cases. Ruff, Black, architecture, and public API baseline checks
passed during the round. Strict Python 3.12 checking reports **233 errors in
47 files** across 428 source files, one fewer diagnostic. Tests use local
provider fakes and do not establish execution on FlagOS hardware.
Repository-wide remediation remains incomplete.


## Distinguish compiler local variables by domain meaning

The schedule validator now names each layer's instruction sequence
instruction_indices, avoiding reuse of the integer layer variable. Native gate
parsing separately names schema-derived parameter sets schema_parameters,
avoiding reuse for the explicit descriptor's parameter sequence. Only local
names changed: scheduling, validation, lowering, public APIs, and serialized
results are unchanged. No abstractions, casts, or suppressions were added.

Schedule and native-gate legalization integration tests passed with **15 passed
in 1.12s**. Ruff, Black, architecture, and public API baseline checks passed.
Strict Python 3.12 checking reports **230 errors in 45 files** across 428 source
files, three fewer diagnostics; both edited modules are now clean. Remaining
optimizer constructor diagnostics were inspected but their protected parameter
interfaces were not changed. Repository-wide remediation remains incomplete.


## Type OpenQASM parameter-name sequences independently of gate arity

The target-conformance parser now declares parameter_names as a variable-length
tuple of strings. Inference previously fixed it to the three U-gate parameters
before other gate descriptors assigned their existing parameter sequences.
Parsing, arity rejection, parameter order, and public APIs are unchanged; no
casts, suppressions, or runtime transformations were added.

Target-conformance integration tests passed with **8 passed in 1.17s**. Ruff,
Black, architecture, and public API baseline checks passed. Strict Python 3.12
checking reports **229 errors in 44 files** across 430 source files, one fewer
diagnostic; target_conformance.py is now clean. The shared source inventory
increased independently from 428 to 430 files. No remote or hardware execution
occurred. Repository-wide remediation remains incomplete.


## Freeze validated artifact profiles without recursive reprocessing

The private artifact profile validator now wraps its validated dictionary copy
in MappingProxyType directly. Profiles have exactly three fields, exact string
values, and must equal one of the four supported constant profiles, so generic
recursive normalization adds no validation or conversion for accepted inputs.
The original mapping is already copied by _strict_fields. Public schemas,
accepted profiles, hash inputs, and immutable return behavior remain unchanged.
No type suppression or new helper was introduced.

Artifact V2, reconciliation, and candidate tests passed with **33 passed in
0.79s**. All four supported profiles also matched the previous recursive result,
rejected mutation, and remained isolated from source dictionary changes. Ruff,
Black, architecture, and public API baseline checks passed. Strict Python 3.12
checking reports **228 errors in 43 files** across 430 source files, one fewer
diagnostic; core/_artifacts.py is now clean. Repository-wide remediation
remains incomplete.


## Preserve tensor return types in result detachment

The local detach helper now declares overloads for Tensor and None inputs.
This records the existing invariant that a tensor measurement never becomes
None, while optional top-level result values can remain absent. The helper
implementation and all public signatures are unchanged; no casts, suppressions,
or duplicated detachment operations were introduced.

Execution-result and Module tests passed with **50 passed, 1 warning in 2.04s**.
The warning is the existing PyTorch complex-module warning. Both copy modes
also passed direct checks for detached gradients, tensor storage ownership,
absent state/samples, and retained count records. Ruff, Black, architecture,
and public API baseline checks passed. Strict Python 3.12 checking reports
**227 errors in 42 files** across 430 source files, one fewer diagnostic;
runtime/result.py is now clean. Repository-wide remediation remains incomplete.


## Type hybrid custom-op parameter bindings at construction

The private hybrid statevector template binder now explicitly declares the
existing str-or-Parameter key domain with Tensor values. This matches the
shared parameter-binding interface without changing that protected interface,
copying an existing mapping, casting values, or suppressing diagnostics.
Parameter order, tensor identity, template substitution, and gradient paths
remain unchanged.

Torch compile and hybrid gradient integration tests passed with **9 passed in
2.38s**. Black formatted the expanded import; Ruff then passed. Architecture
and public API baseline checks passed. Strict Python 3.12 checking reports
**226 errors in 42 files** across 430 source files, one fewer diagnostic. The
module still needs work on autograd context annotations and the dynamically
typed custom-op return boundary. No accelerator or remote execution occurred.
Repository-wide remediation remains incomplete.


## Describe the hybrid autograd callback context precisely

The private hybrid operator callbacks now type their input tuple, output tensor,
and gradient return. A module-private Protocol describes only the existing
context members they use: circuit_ir_json, saved_tensors, and save_for_backward.
PyTorch still creates and owns the real context; no wrapper, replacement state,
new execution layer, cast, or suppression was introduced. Callback operations
and custom-op registration are unchanged.

Torch compile and hybrid gradient integration tests passed with **9 passed in
2.26s**. Ruff, Black, architecture, and public API baseline checks passed.
Strict Python 3.12 checking reports **223 errors in 42 files** across 431 source
files, three fewer diagnostics. The shared source inventory independently grew
by one file. The private custom-op invocation still has a dynamically typed
return boundary and the module is not fully clean. No accelerator or remote
execution occurred. Repository-wide remediation remains incomplete.


## Validate the hybrid custom-op result at its typed boundary

The hybrid statevector entry now receives the dynamically typed PyTorch
custom-op result as object, verifies Tensor, and returns the same instance.
This makes the existing operator schema explicit to static analysis without
casting, suppressing diagnostics, or copying/detaching the result. A return
outside the declared operator contract receives TypeError. Valid numerical and
autograd behavior, registration, and public signatures are unchanged.

Torch compile and hybrid gradient integration tests passed with **9 passed in
2.23s**, showing that the type check remains compatible with the covered compile
and gradient workflows. Ruff, Black, architecture, and public API baseline
checks passed. Strict Python 3.12 checking reports **222 errors in 41 files**
across 431 source files, one fewer diagnostic; hybrid_torch.py is now clean.
These local checks do not establish accelerator performance or hardware
scalability. Repository-wide remediation remains incomplete.


## Annotate the single-qubit tangent callback; retain the Triton test gap

The local selected_output callback in RX/RZ parameter-tangent construction now
explicitly accepts and returns Tensor. Only annotations changed; gate arithmetic,
Jacobian construction, and kernel launch behavior are untouched. This removes
one untyped-definition and two untyped-call diagnostics without suppressions.

Static compilation, Ruff, Black, architecture, and public API baseline checks
passed. Strict Python 3.12 checking reports **219 errors in 41 files** across
431 source files, three fewer diagnostics. Runtime verification was unavailable:
the single-qubit-loop test module was skipped because Triton is not installed
(**0 collected, 1 skipped**, pytest exit 5). This is not a passing numerical test.
A possible zero-depth tangent issue was noticed during inspection, but no
numerical change or unvalidated regression test was retained. That behavior
needs verification in a Triton-capable environment. Existing untyped Triton and
PyTorch autograd boundaries remain. Repository-wide remediation is incomplete.


## Reject invalid global truncation budgets in MPS planning

Adaptive bond planning now requires a supplied global_error_budget to be finite
and non-negative after its existing float normalization. Previously negative,
NaN, and infinite budgets were accepted and could produce meaningless budget
status or hotspot thresholds. Local refinement inherits the same validation
through its existing adaptive-plan call. None and finite non-negative budgets
retain their behavior; public signatures and numerical state are unchanged.

After correcting a test setup name, four regression cases reproduced acceptance
of invalid budgets before the implementation change. MPS tests now pass with
**49 passed in 1.08s**, including both planning entry points. Ruff, Black
formatting, architecture, and public API baseline checks passed. The latest full
strict Python 3.12 baseline remains **219 errors in 41 files**; this input guard
did not repeat that audit. The MPS planning mixin's state-attribute typing remains
unresolved. No accelerator or remote execution occurred. Repository-wide
remediation remains incomplete.


## Declare the MPS planning mixin's existing state fields

MPSPlanningMixin now declares the types of config, truncation_errors,
truncation_records, and orthogonality_center. These fields are already initialized
by MPSState with the declared types. The annotations make the mixin's data
requirements explicit; they create no default instances, shared mutable state,
new properties, or initialization behavior. Existing state properties and
numerical methods remain authoritative and unchanged.

MPS tests passed with **49 passed in 1.19s**. Ruff, Black, architecture, and public
API baseline checks passed. Strict Python 3.12 checking reports **208 errors in
41 files** across 431 source files, eleven fewer diagnostics. Planning still
needs a precise contract for its inherited state properties and numerical
methods; it is not fully typed. No accelerator or remote jobs ran.
Repository-wide remediation remains incomplete.


## Declare existing MPS diagnostic counters

The planning mixin now declares the six integer counters and SVD method string
used in its summary. Each declaration matches an existing MPSState initializer;
no counter defaults or mutable class state were added. Counter increments,
summary keys, numerical kernels, and public APIs remain unchanged.

MPS tests passed with **49 passed in 1.32s**. Ruff, Black, architecture, and public
API baseline checks passed. Strict Python 3.12 checking reports **201 errors in
41 files** across 431 source files, seven fewer diagnostics. The remaining MPS
planning diagnostics concern read-only state properties and numerical methods,
not undeclared diagnostic counters. No accelerator or remote execution occurred.
Repository-wide remediation remains incomplete.


## Reuse computed MPS bond-profile fields in summaries

MPS summary construction now reuses the existing bond profile for wire count,
batch size, bond dimensions, maximum bond, truncation step count, and maximum
truncation error. The profile was already computed at the start of the method;
re-reading state properties and scanning truncation errors repeated that work.
Summary keys, types, numerical diagnostics, and public APIs are unchanged.

MPS tests passed with **49 passed in 1.16s**. Four pre-edit/current comparisons
across complex64/complex128 and truncated/untruncated states preserved complete
summary output. Ruff, Black, architecture, and public API baseline checks passed.
Strict Python 3.12 checking reports **197 errors in 41 files** across 431 source
files, four fewer diagnostics. The remaining mixin property/method contract
still needs work. No hardware or remote jobs ran. Repository-wide remediation
remains incomplete.


## Reuse bond dimensions during MPS diagnostics and hotspot planning

Bond-profile construction now reads bond_dims once for its stored dimensions
and mean calculation. Adaptive planning reuses one dimension tuple across hot
bonds instead of rebuilding it for each length check and indexed lookup. When
there are no hotspots, it does not add a dimension read. Existing maximum-bond
fallback behavior and returned plans are preserved. For valid hotspot indices,
this removes the repeated whole-chain scans inside the hotspot loop.

The regression with two independent truncated bonds observed four bond_dims
reads before the change and one afterward, with an identical adaptive plan.
MPS tests passed with **50 passed in 1.12s**. Ruff, Black formatting, architecture,
and public API baseline checks passed. Strict Python 3.12 checking reports
**195 errors in 41 files** across 431 source files, two fewer diagnostics. No
wall-clock speedup or hardware scalability claim is made. Repository-wide
remediation remains incomplete.


## Avoid duplicate adaptive planning in MPS summaries

MPS summaries now reuse the hotspot list and suggested maximum bond already
carried by the local refinement result. Local refinement invokes adaptive
planning with the same defaults, so the summary's separate adaptive-plan call
was redundant. Summary keys and normal MPS diagnostic values are unchanged;
no new helper, public API, or cached mutable state was added.

Two regressions covering truncated and untruncated states observed two adaptive
planner calls before the fix and one after it, with unchanged summary output.
MPS tests passed with **52 passed in 1.03s**. Ruff, Black, architecture, and public
API baseline checks passed. The latest strict Python 3.12 baseline remains
**195 errors in 41 files**; this removal did not repeat the full audit. No
wall-clock performance or hardware scaling claim is made. Repository-wide
remediation remains incomplete.


## Isolate the optional-JAX import smoke test

The core import/autograd smoke test previously inspected the shared pytest
interpreter. Running it after the hybrid JAX tests failed because those tests
legitimately load JAX. The test now starts a fresh interpreter, preserves the
test environment's import paths, executes a real circuit forward/backward, and
checks that JAX is absent before and after core execution. A 30-second timeout
bounds the child process, and captured output preserves failure diagnostics.
The assertion is not skipped or weakened when another test imports JAX.
Production code and public APIs are unchanged.

The combined hybrid JAX, simulation boundary, and core import selection changed
from **24 passed, 1 failed** to **25 passed in 10.03s**. The broader CPU selection
`(smoke or unit) and not gpu and not distributed and not slow` passed with
**1582 passed, 13 skipped, 1676 deselected, 1 warning in 27.23s**. Two initial
repository checks failed because the system Git launcher was unavailable; the
verified rerun used the installed fallback Git on PATH. The remaining warning
is PyTorch's existing complex-module warning. Ruff, Black, and diff whitespace
checks passed. The language scan checked 1960 paths with zero Han-script text;
binary images still require visual review.

This test-only repair did not repeat the strict typing audit; its latest
baseline remains **195 errors in 41 files**. No accelerator or remote jobs ran.
Repository-wide remediation remains incomplete.


## Reject incompatible JAX parameter shapes before batching

The optional-batch adapter now checks that the parameter tensor ends with the
shape recorded at compilation before flattening its batch axes. Previously,
a kernel compiled for (2, 3) silently reshaped (3, 2) or (1, 6) inputs and
computed a result under the wrong parameter layout. Shorter incompatible
inputs failed later with an opaque JAX reshape error. They now raise ValueError
with the expected and actual shapes. Public signatures and valid input
behavior are unchanged; the path remains a rank-local replicated JAX kernel.

Six regressions failed before the fix, covering three incompatible shapes with
JIT enabled and disabled. Four additional cases verify two batch axes for both
scalar and matrix-shaped samples against analytic RX expectations and
gradients; they pass before and after the fix. The combined hybrid JAX,
simulation boundary, core import, and distributed-plan selection passed with
**135 passed, 6 skipped in 22.41s**. Ruff, Black, architecture, public API
baseline, and diff whitespace checks passed. The language scan checked 1960
paths with zero Han-script text; it does not certify binary images.

No strict typing audit was repeated for this behavioral fix; the latest
baseline remains **195 errors in 41 files**. No GPU, QPU, or remote jobs ran,
and CPU plan tests are not hardware scalability evidence. Repository-wide
remediation remains incomplete.


## Preserve zero-sized parameter dimensions in JAX batches

The optional-batch adapter now computes the flattened batch size with
math.prod rather than asking reshape to infer it with -1. Parameter-free
circuits can carry zero-sized parameter dimensions; reshaping those tensors
with an inferred axis raised ZeroDivisionError before circuit execution.
Explicit batch size preserves both the sample shape and empty batches without
a special execution path or a public API change.

Twelve regressions failed before the fix and pass afterward. They cover
parameter shapes (0,) and (2, 0), one- and two-dimensional batches, empty
batches, and JIT enabled/disabled. A fixed X circuit returns the expected
Z expectation for every sample, and backward preserves the empty gradient
shape. The combined hybrid JAX, simulation boundary, and isolated core import
selection passed with **47 passed in 11.00s**. Ruff, Black, public API baseline,
and architecture checks passed.

This remains rank-local JAX execution; no accelerator or remote jobs ran.
The strict typing audit was not repeated; the latest baseline remains
**195 errors in 41 files**. Repository-wide remediation remains incomplete.


## Reject unsupported double backward through the JAX bridge

The private PyTorch backward adapter now uses torch.autograd.function's
once_differentiable decorator. Its saved gradients come from JAX through
DLPack as detached tensors. Without this boundary, differentiating a composed
loss twice could follow the outer PyTorch loss while omitting the quantum
kernel's second derivative. The bridge now rejects double backward instead
of allowing that incomplete derivative path. First-order parameter/input
gradients and public signatures remain unchanged. The limitation is documented
outside the generated region of KNOWN_LIMITATIONS.md; no new capability or
higher-order support is claimed.

Four regressions failed before the change because double backward did not
raise. They cover JIT enabled/disabled and parameter-only/parameter-plus-input
execution, and also check first derivatives against an analytic squared-cosine
loss. The combined hybrid JAX, simulation boundary, isolated import, and Module
selection passed with **95 passed, 1 warning in 11.52s**. The warning is the
existing PyTorch complex-module warning. Ruff, Black, architecture, and public
API baseline checks passed. No remote or accelerator jobs ran. The strict
typing audit was not repeated; its latest baseline remains **195 errors in
41 files**. Repository-wide remediation remains incomplete.


## Validate discrete MPS planning controls before inspecting hotspots

Adaptive bond planning now requires an integer min_increment, and local
refinement requires an integer window_radius. Boolean, fractional, and
nonfinite values are rejected before hotspot-dependent work. Previously,
fractional values were silently truncated, booleans acted as counts, and
nonfinite values could either pass on states without hotspots or fail later
during integer conversion. Existing negative-integer errors and valid
nonnegative integer behavior remain unchanged. Redundant integer conversions
in the validated arithmetic were removed; no helper or public signature was
added.

Sixteen regressions failed before the fix and pass afterward, covering both
controls with and without truncation hotspots. The MPS suite passed with
**68 passed in 1.29s**, including existing zero-increment and zero-radius
coverage. Ruff, Black, architecture, and public API baseline checks passed.
No numerical kernels, remote jobs, or accelerator execution changed. The
strict typing audit was not repeated; its latest baseline remains
**195 errors in 41 files**. Repository-wide remediation remains incomplete.


## Bound the MPS instruction schedule cache

The private schedule cache now retains at most 256 circuit structures and
evicts the least recently used entry. Previously, each new circuit structure
remained in a process-global dictionary indefinitely. Cache keys still contain
structural metadata rather than parameter tensors, preserving dynamic-parameter
reuse. A short lock protects lookup/promotion and insertion/eviction; schedule
construction remains outside the lock. No numerical kernel or public API
changed, and no throughput improvement is claimed.

The regression filled the cache, reused an older program, and inserted another
structure. Before the fix it retained 257 entries; afterward it stays bounded,
retains the recently reused schedule, and reconstructs the evicted schedule
with identical grouping. Existing dynamic-parameter gradient reuse coverage
also passes. The final MPS run passed with **69 passed in 0.95s**. Ruff and
Black passed after the locking change; architecture and public API baseline
checks passed for the cache change. The strict typing audit was not repeated;
its latest baseline remains **195 errors in 41 files**. No remote or accelerator
jobs ran. Repository-wide remediation remains incomplete.


## Expose fq.run implementation types without eager imports

Local runtime execution, the Jiuding execution branch, and remote output
normalization now use explicit imports inside their existing execution
branches. This preserves deferred loading while allowing static checking of
return values and OutputRequest narrowing. The Jiuding branch explicitly
checks that target is not None, and the counts request factory uses an alias
to avoid colliding with the later counts result dictionary. No cast, ignore,
new public API, or eager optional-provider import was introduced.

A fresh strict Python 3.12 audit confirmed the starting baseline of 195 errors
in 41 files. After the changes it reports **192 errors in 41 files** across
431 source files. The remaining fq.run dynamic-return boundaries still need
work. Local measurements, output/result contracts, mocked Quafu and Jiuding
workflows, and the isolated no-JAX core smoke test passed with **92 passed in
1.77s**. Ruff and Black passed on the final source; public API baseline and
architecture checks passed for the import change. No remote jobs were
submitted. Repository-wide remediation remains incomplete.


## Use authoritative result and planner types in the root facade

Remote result construction now imports ExecutionResult and MeasurementResult
directly from runtime.result inside the remote branch, removing two dynamic
lookups through the broader runtime.contracts facade. fq.plan likewise calls
a locally imported planner function. Deferred imports and public signatures
are preserved. The remote counts dictionary explicitly uses the existing
measurement contract's string-or-integer key type; values remain normalized
to string keys exactly as before, with no additional copy or cast.

Strict Python 3.12 checking reports **189 errors in 40 files** across 431 source
files, down from 192 errors in 41 files. The root facade _api.py is now clean
under that audit; the repository is not. Remote expectation/counts, mocked
Jiuding, output/result/plan contracts, local measurements, and isolated core
import tests passed with **104 passed in 1.79s**. Ruff, Black, architecture, and
public API baseline checks passed for the import changes. The final dictionary
annotation was additionally checked by the full strict audit. No remote or
accelerator jobs ran. Repository-wide remediation remains incomplete.


## Remove redundant dynamic math imports from MPS transport

Five transport shape-size calculations now call math.prod directly using the
module's existing math import. This removes repeated dynamic __import__ calls
from packed payload and receive-buffer sizing without changing arithmetic,
integer conversions, shapes, protocol metadata, or allocation behavior. No
additional helper or numerical execution path was introduced.

Packed-buffer reuse, P2P diagnostics, static descriptors, and reverse-contract
tests passed with **53 passed in 0.91s**. Ruff, architecture, and public API
baseline checks passed; Black reformatted the shortened expressions. These
local tests do not establish distributed hardware scalability. The optional
Work handles in the legacy asynchronous transport path still need attention;
this cleanup does not claim to resolve those typing diagnostics. The latest
strict baseline remains **189 errors in 40 files** and was not rerun. No
remote or accelerator jobs ran. Repository-wide remediation remains incomplete.


## Diagnose missing asynchronous MPS transport requests

The legacy asynchronous send/receive helpers now reject absent PyTorch Work
handles with RuntimeError identifying the peer and operation. Shape and payload
receives are checked separately before waiting or interpreting their buffers.
Previously, these cases failed with an uninformative None.wait AttributeError.
Valid transfers retain their submission order, waits, shapes, and payloads;
this does not introduce request cancellation or change timeout behavior.

Four missing-request regressions failed before the fix. A successful mocked
round trip also verifies a noncontiguous source tensor and all four waits.
The transport-focused selection passed with **58 passed in 0.93s**. Strict
Python 3.12 checking reports **186 errors in 39 files** across 431 source files,
three fewer diagnostics and one fewer affected file. Ruff, architecture, and
public API baseline checks passed, and Black formatted both changed files.
Tests used local mocked requests, not a real multi-rank hardware transport;
no scalability or recovery guarantee is inferred. No remote jobs ran.
Repository-wide remediation remains incomplete.


## Recheck cumulative CPU behavior and include async regressions in CI selection

The cumulative CPU selection `(smoke or unit) and not gpu and not distributed
and not slow` passed with **1582 passed, 13 skipped, 1724 deselected, 1 warning
in 27.34s**. A separate MPS/JAX numerical selection passed with **111 passed in
11.46s**. The warning remains PyTorch's existing complex-module warning. The
language scan checked 1961 paths with zero Han-script text; binary images
still need visual review.

The selection audit found that the new asynchronous transport regression file
was unmarked, so the default marker selection omitted its five cases. The
module now has the unit marker. All **5 passed in 0.78s** under the explicit
unit/non-hardware selection after this fix. The broader count above is from
before the marker addition; no combined post-marker total is claimed.

The latest strict audit still has **186 errors in 39 files**. Its diagnostic
categories include 65 missing annotations, 36 untyped calls, 24 untyped
decorators, and 19 missing attributes. The largest affected file is Triton
statevector_gates.py (38 errors); MPS planning has 15 and TN path_search has
13. These counts describe remaining work, not an approved suppression list.
No remote or accelerator jobs ran. Repository-wide remediation remains
incomplete.

## MPS batch truncation preserves every sample's required rank

Pair splitting, bucketed pair splitting, and statevector conversion selected
the smallest cutoff-derived rank across samples. This discarded singular
values above the cutoff in samples requiring a larger subspace. A batch
containing a product state and a Bell state could consequently lose half of
the Bell state's probability weight during statevector conversion.

All three paths now select the largest per-sample rank after applying the
existing cutoff and maximum-bond rules. Independent bonds in a factorization
bucket still choose separate ranks. No public signatures or diagnostic
schemas changed. Heterogeneous batches can use more memory than the incorrect
minimum-rank implementation, while continuing to respect max_bond.

Nine regression cases failed against the corresponding original paths. They
cover batched entanglement preservation, sequential and bucketed splits,
maximum-bond limits, reconstruction, discarded-weight reporting, and the
existing projected-gradient behavior. The focused MPS suite passed with
**89 passed in 1.11s**. The final CPU selection passed with **1651 passed,
13 skipped, 1737 deselected, 1 warning in 28.00s**; the warning is the existing
PyTorch complex-module warning. Ruff, Black, and the public API baseline
passed. Architecture and language checks also passed during this round.

These are local CPU correctness checks, not accelerator or distributed
capacity evidence. Strict typing was not rerun; the last recorded baseline
remains 161 errors in 38 files. Repository-wide remediation remains incomplete.


## Reuse canonical residuals in MPS bond profiles

Bond-profile construction now computes left and right canonical residuals
once. When no orthogonality center is set, it uses their maximum for the
mixed residual, matching the existing mixed_canonical_residual definition.
Previously that method recomputed both full-chain diagnostics, including
Gram matrices and host scalar reads. States with a center still use their
existing centered diagnostic. No persistent cache, public signature, or
numerical formula changed.

The no-center regression failed before the fix because both diagnostics ran
twice; it now observes one call each with identical profile output. The
centered regression preserves the specialized mixed calculation. Both cases
carry the unit marker. The MPS suite passed with **71 passed in 1.12s**. Ruff,
public API baseline, and architecture checks passed; Black formatted the
tests. No wall-clock speedup or hardware scaling claim is made. The remaining
mixin property/method typing contract is unresolved; the latest strict
baseline remains **186 errors in 39 files** and was not rerun. No remote or
accelerator jobs ran. Repository-wide remediation remains incomplete.


## Avoid backward bookkeeping in scalar MPS diagnostics

The left, right, and mixed canonical-residual methods now run under
torch.no_grad. Each already returns a detached Python float, so constructing
reverse-mode graphs for the intermediate Gram matrices served no caller.
The change is limited to those diagnostics; state_norm and the differentiable
expectation paths retain their existing behavior and public signatures.

Three unit-marked cases inspect saved-tensor hooks while running each
diagnostic and then check a real parameter gradient against the analytic
RX/RY cosine derivative. Before the change, two cases saved backward tensors
and failed the no-bookkeeping assertion; one already passed for this state.
All three now pass, with unchanged residual values and a valid subsequent
backward pass. The complete MPS suite passed with **74 passed in 1.19s**.
Ruff, Black, public API baseline, and architecture checks passed. The fresh
strict Python 3.12 audit remains **186 errors in 39 files** across 431 source
files. No wall-clock speedup, hardware memory reduction, or scalability claim
is made. No remote or accelerator jobs ran. Repository-wide remediation
remains incomplete.


## Keep the complete MPS bond profile outside reverse-mode recording

The bond_profile diagnostic now runs under torch.no_grad. After the residual
methods were fixed, its state-norm contraction still saved backward tensors
even though the profile immediately converts the result to detached scalar
statistics. The change covers the entire diagnostic, including callers through
summary, while direct state_norm and expectation calls remain differentiable.
No result fields, public signatures, or numerical formulas changed.

The unit-marked saved-tensor regression now also covers bond_profile. That
case failed before this change and passes afterward, retaining the same
profile and an analytic subsequent parameter gradient. The complete MPS
suite passed with **75 passed in 1.04s**. Ruff, Black, architecture, and public
API baseline checks passed. The latest strict baseline remains **186 errors
in 39 files**; no new full typing audit was run. No hardware performance claim
is made, and no remote or accelerator jobs ran. Repository-wide remediation
remains incomplete.


## Make the MPS training environment dependency explicit

The distributed MPS training engine now imports os normally and reads
LOCAL_WORLD_SIZE through os.environ. This replaces the remaining direct
__import__ call found in package Python sources. Environment precedence,
fallback to world_size, integer conversion, and ownership planning are
unchanged; this is a readability cleanup, not a scalability improvement.

The training, production-contract, and dirty-bond-planner selection passed
with **50 passed in 1.29s**. Ruff, Black, architecture, and public API baseline
checks passed. The language scan checked 1963 paths with zero Han-script text;
it does not certify binary images. The latest strict baseline remains
**186 errors in 39 files** and was not rerun for this import-only change.
No remote or accelerator jobs ran. Repository-wide remediation remains
incomplete.


## Use a typed standard-library product for TN dimensions

The shared private dimension-product helper now accepts an iterable of
integer-convertible values and uses math.prod over a generator. This removes
the manual accumulator and its Any parameter while retaining per-element
integer conversion, single-pass iteration, empty-product behavior, and
arbitrary-precision Python integer arithmetic. Existing compatibility imports
continue to reference the same helper; no tensor allocation or public
signature changed.

Path-search, contraction-stage, and tensor-network tests passed with
**94 passed, 1 skipped in 4.65s**, including shape-only 128-axis size checks
and contraction result parity. Ruff, Black, architecture, and public API
baseline checks passed. The fresh strict Python 3.12 audit still reports
**186 errors in 39 files**, now across 432 source files in the shared tree.
The shape-only/real-tensor node typing boundary remains unresolved; no casts
or widened public node fields were introduced to hide it. No remote or
accelerator jobs ran. Repository-wide remediation remains incomplete.


## Avoid sorting every candidate for deterministic TN pair selection

Greedy contraction pair selection now uses min when it needs only the best
candidate. Previously it sorted the entire candidate list and returned the
first item. The existing score tuple and first-minimum tie behavior are
preserved. Randomized selection still sorts its candidate pool, preserving
seeded selection behavior. For k candidates the deterministic selection work
changes from a full O(k log k) sort to an O(k) scan; candidate generation is
unchanged and no wall-clock speedup is claimed.

Twelve pre-edit/current comparisons across all four objectives and deterministic
or two seeded randomized searches preserved complete contraction outputs and
step records. Path-search, stage, and TN tests passed with **94 passed, 1 skipped
in 5.21s**. Ruff, Black, architecture, and public API baseline checks passed.
The latest strict baseline remains **186 errors in 39 files** and was not
rerun for this equivalent selection change. No remote or accelerator jobs
ran. Repository-wide remediation remains incomplete.


## Generate fallback TN candidate pairs lazily

Greedy pair selection now defines its all-pairs fallback once as a typed
generator. This replaces two duplicate list comprehensions and avoids
materializing the separate quadratic list of pair indices. Quality/treewidth
search still substitutes its sorted connected-pair list when available.
Candidate order, scoring, randomized selection, and the scored candidate list
are unchanged; total search memory is not claimed to be constant.

Twelve stored pre-edit/current comparisons preserve all contraction values
and step records. Path-search, stage, and TN tests passed with **94 passed,
1 skipped in 4.73s**. Ruff, Black, public API baseline, and architecture checks
passed. The fresh strict audit reports **189 errors in 40 files** across 432
source files. Compared with the prior audit, the three additional diagnostics
are dict-key binding types in shared core/_artifacts.py, which this round did
not modify; no diagnostic was added in the changed path-search module. Those
remaining errors are not suppressed. No remote or accelerator jobs ran.
Repository-wide remediation remains incomplete.


## Describe the TN greedy score tuple precisely

The private candidate list now declares its score as six integers followed
by an output-label tuple, matching both existing objective branches. This
replaces tuple[Any, ...] without introducing a record class, casts, or changes
to scoring and tie-breaking. The type checker can now catch incompatible
score fields instead of allowing an arbitrary tuple at this boundary.

Path-search, stage, and TN tests passed with **94 passed, 1 skipped in 4.71s**.
Ruff, Black, architecture, and public API baseline checks passed. Strict
Python 3.12 checking remains **189 errors in 40 files** across 432 source
files; removing this Any did not reduce an existing diagnostic count.
The shared Core binding implementation is undergoing separate changes and
was inspected but not modified in this round. No remote or accelerator jobs
ran. Repository-wide remediation remains incomplete.


## Reject noninteger TN multistart search counts

The private multistart entry now validates repeats and random_pool_size as
integers before building cache keys or starting search. Previously, positive
fractional values were truncated and booleans acted as counts; nonfinite
values failed later during conversion. Invalid values now produce an explicit
ValueError naming the control. Existing positive-integer and nonpositive-
integer behavior is preserved, and redundant conversions in the validated
loop were removed.

Eight unit regressions failed before the fix and pass afterward. The TN
path/stage/execution selection passed with **102 passed, 1 skipped in 4.61s**.
Ruff, architecture, and public API baseline checks passed; Black formatted
the tests. The latest strict baseline remains **189 errors in 40 files** and
was not rerun for this validation change. No remote or accelerator jobs ran.
Repository-wide remediation remains incomplete.


## Validate beam width before TN search work

The private beam-search entry now rejects noninteger beam_width values before
searching, allocating intermediates, or slicing its candidate list. Fractional
widths and booleans previously acted as truncated integer widths; nonfinite
widths failed later during conversion. Valid integer behavior and the existing
nonpositive-width error remain unchanged. The validated width is used directly
for candidate slicing. This fixes the private entry and does not claim an
audit of every higher-level cache-normalization path.

Eight regressions failed before the fix and pass afterward, covering actual
contraction and shape-only search. The TN selection passed with **110 passed,
1 skipped in 4.72s**. Ruff, Black, architecture, and public API baseline checks
passed. The latest strict baseline remains **189 errors in 40 files** and was
not rerun for this validation change. No remote or accelerator jobs ran.
Repository-wide remediation remains incomplete.


## Validate beam profiles before cache lookup

Beam and beam_sliced profile keys now validate beam_width before converting
it into a cache key. Previously, width 1.5 or True could reuse an existing
width-1 profile, bypassing the search entry's validation. Cold-cache calls
rejected those same values. One private validator now serves both search
and profile-key construction; unrelated strategies retain their existing
unused-width behavior and valid cache identities remain unchanged.

Eight cases cover both profile strategies, both invalid values, and warm or
cold caches. Four warm-cache cases failed before the fix; all pass afterward.
The TN selection passed with **118 passed, 1 skipped in 5.19s**. Ruff, Black,
architecture, and public API baseline checks passed. The latest strict baseline
remains **189 errors in 40 files** and was not rerun for this validation fix.
No remote or accelerator jobs ran. Repository-wide remediation remains
incomplete.


## Module pass: tensor-network search and contraction planning

This pass resolves the shape-only/real-tensor boundary in path_search.py and
the two path-conversion helpers in contraction.py. Public TensorNetworkNode
continues to contain a real torch.Tensor. Private search intermediates use
_SearchNode, whose tensor field can also carry _DryRunTensor metadata. This
representation is necessary because shape-only products may exceed PyTorch's
size limits; widening the public node or pretending metadata is a Tensor
would hide that distinction. No new public contract or tensor wrapper was
introduced.

The three search entry points have overloads distinguishing real execution
returns from dry-run metadata. Numerical branches explicitly require real
tensors. Tree reconstruction and path conversion use the same internal node.
The cached stage path still receives original real nodes. The module README
now explains the boundary and its verification path. No casts, ignores, or
relaxed checker settings were used.

A concrete bug was fixed alongside the type structure: beam-search output
reordering used torch.empty even in a dry run. The allocation-rejection test
failed for beam search before the fix and passes afterward. All three search
strategies now pass output-reordering checks and a 96-axis, 2**96-element
shape-only case whose input tensors are small expanded views. This is a
planning test, not evidence that such a quantum state can be simulated.
Three additional cases verify real operand gradients against direct matrix
multiplication. Twelve stored comparisons preserve complete contraction
outputs and step records.

Final focused verification passed with **127 passed, 1 skipped in 4.67s**.
The broader CPU selection passed with **1633 passed, 13 skipped, 1734
deselected, 1 warning in 28.42s**, before the last three gradient cases were
added; those cases passed in the final focused run. The warning is the
existing PyTorch complex-module warning. Ruff, Black, architecture, public
API baseline, and diff whitespace checks passed. The language scan checked
1973 paths with zero Han-script text; binary images still require review.

The strict Python 3.12 audit reports **176 errors in 39 files** across 435
source files. There are no diagnostics in simulation/tensor_network. This
pass removes all 15 prior diagnostics in its search/contraction files. The
net repository change is 189 to 176 because two additional diagnostics arose
in the separately modified remote/compute/_program_submission.py. No remote
or accelerator jobs ran; real CUDA cached-stage behavior was not certified.
Repository-wide remediation remains incomplete.


## Module pass: MPS planning and numerical-state requirements

MPSPlanningMixin is now an abstract base declaring the seven read-only
properties and four numerical methods its existing implementation requires.
MPSState already implements all eleven. This makes the existing inheritance
boundary explicit without importing the concrete state back into its base,
adding an adapter, copying tensors, or duplicating algorithms. Existing data
fields keep their instance-owned declarations; properties are not mislabeled
as mutable attributes. Public MPSState signatures and planning results remain
unchanged.

An incomplete planning base can no longer be instantiated and then fail later
with a missing state method. A unit regression failed before the change because
base construction succeeded; it now rejects construction and verifies a real
Bell-state profile. The module README explains the ownership and implementation
requirements. No casts, ignores, or checker relaxations were introduced.

All 15 prior planning.py diagnostics are resolved. Strict Python 3.12 checking
reports **161 errors in 38 files** across 436 source files, down from 176.
The MPS planning module is clean; simulation/mps/factorization.py still has
a separate torch.linalg.LinAlgError export diagnostic, and Runtime's MPS
factorization still has an untyped CUDA Event call. This pass does not claim
that the entire MPS execution stack is fully remediated.

The local MPS suite passed with **76 passed in 1.12s**. Expanded MPS, JAX,
initialization, noisy-lowering, and objective-pipeline checks passed with
**129 passed in 13.33s**. The CPU selection passed with **1637 passed, 13
skipped, 1734 deselected, 1 warning in 30.80s**. The warning is the existing
PyTorch complex-module warning. Ruff, Black, architecture, public API baseline,
and diff whitespace checks passed. The language scan checked 1974 paths with
zero Han-script text; binary images still need visual review. No remote or
accelerator jobs ran. Repository-wide remediation remains incomplete.


## Preserve training gradients through the MPS CPU SVD fallback

The batched SVD fallback no longer detaches matrices before moving them to
CPU LAPACK. The detached copy severed the training graph after requested
and isolated CUDA drivers failed, even when the CPU decomposition succeeded.
The existing differentiable device copy is retained in both directions;
fallback ordering, reconstruction, result shapes, and counters are unchanged.

Two unit regressions use real CPU SVD with simulated CUDA-driver failures.
Both float64 and complex128 previously reconstructed the input but produced
results without a gradient graph. They now preserve reconstruction and yield
the analytic gradient 2*A for the squared reconstruction norm. The fallback
test module now has the unit marker so default CI includes it. Combined
fallback/MPS verification passed with **80 passed in 1.16s**. Ruff, Black,
architecture, public API baseline, and diff whitespace checks passed.

This verifies CPU decomposition and graph preservation under fault injection,
not real CUDA transfer or hardware fallback behavior. The separate
torch.linalg.LinAlgError typing-export issue remains unresolved; no exception
category was broadened or suppressed to hide it. The latest strict baseline
remains **161 errors in 38 files** and was not rerun for this gradient fix.
No remote or accelerator jobs ran. Repository-wide remediation remains
incomplete.


## Algorithms: parameter layout and typed backward calls

Staged optimization now creates contiguous private parameter copies. Previously,
cloning a transposed tensor preserved its strides: Rotosolve updated a temporary
reshape copy without changing the objective parameters, while LBFGS failed when
flattening the noncontiguous gradient. Shapes, values, dtype, and placement are
preserved, and caller-owned input tensors remain unchanged. The change uses the
existing allocation rather than introducing an additional copy or abstraction.

Four parameter-layout cases cover both optimizers with contiguous and transposed
inputs. The two transposed cases failed before the fix; all four now reach their
analytic minima and verify input isolation. Four backward calls in the algorithm
loops now use the equivalent typed torch.autograd.backward entry point. No public
signatures, result schemas, casts, ignores, or checker settings changed.

Algorithm and CPU vertical-path verification passed with **54 passed in 10.02s**.
The final CPU selection passed with **1660 passed, 13 skipped, 1737 deselected,
1 warning in 27.94s**. The warning is the existing PyTorch complex-module warning.
The shared-tree test total includes changes from other work; this round added
four cases. Ruff, Black, public API baseline, architecture, and dependency policy
checks passed.

Strict checking reports **157 errors in 38 files across 436 source files**, down
from 161. Algorithms account for five remaining errors: two untyped Jacobian
calls, one untyped LBFGS step, and two optimizer-factory constructor mismatches
at protected public interfaces. These remain explicit follow-up work. This
round verifies local CPU behavior only; no GPU, distributed, or remote jobs ran.
Repository-wide remediation remains incomplete.


## Runtime training: typed backward entry point

The canonical training loop now calls torch.autograd.backward(loss), matching
its typed algorithm counterpart without altering backward defaults, optimizer
ordering, loss snapshots, callbacks, or public interfaces. Existing training
and API-contract tests verify these behaviors; no implementation-mirroring test
was added for the equivalent call.

Focused verification passed with **53 passed, 1 warning in 2.22s**. The CPU
selection passed with **1661 passed, 13 skipped, 1737 deselected, 1 warning in
28.08s**. The warning is the existing complex-module warning, and the shared-tree
suite count includes other sessions' changes. Ruff, Black, public API baseline,
architecture, and diff whitespace checks passed.

Strict checking now reports **156 errors in 37 files across 436 source files**.
The training module has no remaining strict errors in this run; this does not
certify all Runtime modules or hardware behavior. No GPU or remote jobs ran.
Repository-wide remediation remains incomplete.


## Runtime probes reject nonfinite errors and mismatched shapes

The operator probe comparator now rejects shape mismatches before subtraction
and treats nonfinite numerical errors as failure. Previously, broadcasting
could hide an incorrect output shape, and NaN comparisons against tolerances
could mark invalid forward results or gradients as valid. Backward comparison
now reuses the same checked comparator instead of duplicating arithmetic.
Public evidence fields and tolerances are unchanged.

Five fault-injection cases cover NaN and infinite outputs, mismatched shapes,
and NaN and infinite gradients. Three failed against the original code: the
NaN output incorrectly set forward=True, while the shape mismatch and NaN
gradient incorrectly produced passing evidence. Existing infinity rejection
remains covered. The probes and numerical certification workload also use
three equivalent typed torch.autograd.backward calls.

Focused verification passed with **22 passed in 0.94s**. The CPU selection
passed with **1666 passed, 13 skipped, 1737 deselected, 1 warning in 28.39s**.
The warning is the existing PyTorch complex-module warning. Ruff, Black,
public API baseline, architecture, and diff whitespace checks passed. Strict
checking reports **153 errors in 35 files across 436 source files**; neither
operator_probes.py nor numerical_validation.py has remaining strict errors.

Fault injection verifies fail-closed evidence semantics on CPU, not actual
accelerator capability or distributed capacity. No GPU or remote jobs ran.
Repository-wide remediation remains incomplete.


## JAX MPS scan uses integer bond dimensions

The scan's two rank calculations now express powers of two as integer shifts.
The sole production caller supplies nonnegative layer indices from enumerate;
valid bond indices likewise yield nonnegative exponents. This preserves the
existing rank schedule and maximum-bond cap while avoiding exponentiation's
integer-or-floating return ambiguity. No public signatures or JAX execution
strategy changed, and no cast or ignore was introduced.

Existing hybrid JAX tests passed with **42 passed in 10.81s**, including the
six-wire, two-layer scan comparison against the per-gate path for both losses
and parameter gradients. The CPU selection passed with **1666 passed,
13 skipped, 1739 deselected, 1 warning in 28.11s**. The warning is the existing
PyTorch complex-module warning. Ruff, Black, public API baseline, architecture,
and diff whitespace checks passed.

The two strict errors in simulation/jax/mps/kernels.py are resolved. The latest
shared-tree strict run reports **152 errors in 35 files across 439 source
files**: this round removed two errors, while concurrent Remote work introduced
one CompletedProcess type-argument error. That independently edited work was
left intact. CPU JAX verification is not GPU or distributed scalability
validation. Repository-wide remediation remains incomplete.


## CPU Double-Single autograd bridge boundary

The private P5 context now declares exactly the configuration, saved tensors,
and save operation used by forward and backward. Backward returns Tensor-or-None
entries rather than an unbounded Any tuple. The protocol describes PyTorch's
supplied context; it does not create a replacement context or additional state.

The untyped, variadic PyTorch Function.apply entry is isolated as an
object-returning callable, and its output is checked before entering the typed
public Tensor return boundary. A fault-injection unit test verifies rejection
of a non-Tensor result. No cast, ignore, public signature change, or checker
relaxation was introduced. Existing parameter order, FP32 delivery, and
higher-order rejection behavior remain intact.

The focused bridge, optimizer-contract, and conformance selection passed with
**12 passed in 29.76s**, including the bounded float32 gradcheck. The CPU
selection passed with **1673 passed, 13 skipped, 1740 deselected, 1 warning in
29.10s**. The warning is the existing PyTorch complex-module warning; shared-tree
test growth also includes other sessions' work. Ruff, Black, public API baseline,
architecture, and diff whitespace checks passed.

Strict checking reports **150 errors in 34 files across 439 source files**.
The P5 bridge module's two strict errors are resolved. This remains a CPU
bridge with an FP32 delivered-gradient boundary; it does not establish
end-to-end Double-Single optimizer precision or accelerator capability.
No remote jobs ran. Repository-wide remediation remains incomplete.


## JAX-to-PyTorch autograd context and return boundary

The JAX bridge now describes its PyTorch-owned saved-gradient context with a
private protocol and validates the object returned by Function.apply before
returning a Tensor. The callable boundary accepts the exact two tensor inputs
used by this bridge. Saved parameter and input gradients no longer propagate
through an Any context. The unused forward-local JAX import and deletion were
removed; required JAX imports remain lazy in their owning operations.

Existing hybrid JAX coverage passed with **42 passed in 10.21s**, including
parameter/input gradients, optional batch axes, parameter-free circuits, MPS
and tensor-network paths, and rejection of incomplete higher-order gradients.
The CPU selection passed with **1673 passed, 13 skipped, 1740 deselected,
1 warning in 27.82s**. The warning is the existing PyTorch complex-module
warning. Ruff, Black, public API baseline, architecture, and diff whitespace
checks passed. Public signatures and gradient semantics are unchanged.

Strict checking reports **148 errors in 34 files across 439 source files**,
down by two. kernel.py still has one external typing error at jax.config.update;
it was not suppressed. This verifies local CPU JAX behavior, not distributed
JAX capacity or GPU execution. Repository-wide remediation remains incomplete.


## Tensor-network slice reduction autograd boundary

The slice-sum custom autograd function uses an opaque object context because
it saves no state. Its untyped PyTorch apply entry is isolated behind an exact
single-Tensor callable and a checked Tensor result. No collective ordering,
reduction operation, backward formula, or public signature changed.

Three local unit cases cover inference and differentiable reduction, preservation
of the caller's local contribution, the existing identity backward without a
second reduction, and rejection of a non-Tensor framework result. They inject a
remote contribution into a fake collective; they do not establish multi-rank
transport or scalability. The focused TN selection passed with **78 passed,
1 skipped in 5.06s**. The CPU selection passed with **1676 passed, 13 skipped,
1744 deselected, 1 warning in 31.35s**. The warning is the existing PyTorch
complex-module warning.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. Strict checking reports **146 errors in 33 files across 440 source
files**, resolving both errors in the slice execution module. Other sessions'
shared-tree changes account for source-count growth. No real GPU, multi-rank,
or remote jobs ran. Repository-wide remediation remains incomplete.


## Statevector reverse context and result boundary

The sharded reverse autograd context now explicitly describes saved parameter
slots, checkpoint policy, materialized IR, device, backward evidence, the owned
StatevectorShardState, and inter-node ket checkpoints. Backward declares
Tensor-or-None gradient entries. These annotations describe the existing
PyTorch-owned context; no replacement state or checkpoint representation was
introduced. The process-group boundary remains opaque.

The Function.apply result is checked before constructing the public gradient
result, and the result's backward method uses the equivalent typed
torch.autograd.backward entry point. Public signatures, communication order,
checkpoint lifecycle, evidence fields, and gradient formulas remain unchanged.

Existing reverse, training, and persistent-layout tests passed with **45 passed
in 1.98s**. The final CPU selection passed with **1676 passed, 13 skipped,
1744 deselected, 1 warning in 30.08s**. The warning is the existing PyTorch
complex-module warning. Ruff, Black, public API baseline, architecture, and
diff whitespace checks passed. Strict checking reports **144 errors in 32
files across 440 source files**, resolving both errors in reverse.py.

These checks verify local CPU semantics, not real multi-rank transport or
accelerator capacity. No GPU or remote jobs ran. Repository-wide remediation
remains incomplete.


## Hamiltonian identity offsets preserve state precision

Identity terms previously constructed their expectation constants in float32
for every input representation. A complex128 state consequently caused its
float64 Hamiltonian coefficient to be rounded to float32 before multiplication.
The four identity paths now allocate constants from the real component of the
input tensor, preserving device placement and real precision without a dtype
lookup table or additional adapter. Existing normalization semantics are unchanged.

Eight unit cases cover Circuit, MPS, statevector, and density-matrix inputs in
complex64 and complex128. All four complex128 cases failed before the fix.
The corrected paths preserve a coefficient of 1 + 2**-40 exactly in float64
and retain its analytic gradient; the complex64 cases remain float32.

The focused algorithm, hybrid optimization, and MPS selection passed with
**133 passed in 9.88s**. The CPU selection passed with **1684 passed, 13 skipped,
1744 deselected, 1 warning in 32.57s**. The warning is the existing PyTorch
complex-module warning. Ruff, Black, public API baseline, architecture, and
diff whitespace checks passed. Strict typing was not rerun for this numerical
fix; the last recorded baseline is 144 errors in 32 files. No GPU or remote
jobs ran. Repository-wide remediation remains incomplete.


## Pauli-term expectation accepts complex coefficients consistently

The coefficient conversion helper now promotes real computation precision to
its complex counterpart when the supplied coefficient is a Python complex or
complex Tensor. Previously, Python complex coefficients raised TypeError,
while complex Tensor coefficients were cast to real with a discarded-imaginary
warning. Multiplication now retains the complex coefficient, and the existing
final real projection still defines the returned energy. Real coefficient
conversion and public signatures are unchanged.

Eight tests compare Python and Tensor coefficients across Circuit, MPS,
statevector, and density-matrix term expectations against a dense real-part
reference. They preserve complex128 precision and verify coefficient gradients.
The original run had five failures: four Python-complex conversion errors and
one complex-to-real warning treated as an error; PyTorch emits that warning
once per process. All eight cases now pass without that warning.

The focused algorithm selection passed with **65 passed in 8.26s**. The CPU
selection passed with **1692 passed, 13 skipped, 1744 deselected, 1 warning in
29.07s**. The remaining warning is the existing PyTorch complex-module warning.
Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. This validates individual Pauli-term expectation paths, not arbitrary
non-Hermitian optimization or every specialized Hamiltonian fast path.
Strict typing was not rerun; the last baseline remains 144 errors in 32 files.
No GPU or remote jobs ran. Repository-wide remediation remains incomplete.


## MPS Hamiltonian coefficient aggregation has explicit types

The Z/ZZ fast-path accumulation dictionaries now describe real, complex, and
Tensor coefficients explicitly. This removes the final Any usage from
algorithms/core.py. Already-normalized operator tuples and integer wires are
used directly rather than copied or converted again. Public annotations and
numerical formulas are unchanged.

A scenario test combines duplicate Z terms with a ZZ term sharing a trainable
complex coefficient. The MPS fast path agrees with a dense-state term-by-term
sum in both energy and coefficient gradient at complex128 precision. This
extends the preceding individual-term verification to the specialized chain
aggregation path; it is not a claim about every Hamiltonian specialization.

The final focused selection passed with **96 passed in 1.21s**. The CPU
selection during this round passed with **1693 passed, 13 skipped, 1744
deselected, 1 warning in 28.59s**; the final removal of redundant conversions
was subsequently covered by the focused rerun. The warning is the existing
PyTorch complex-module warning. Ruff, Black, public API baseline, architecture,
and diff whitespace checks passed. Strict checking still reports **144 errors
in 32 files across 440 source files**. No GPU or remote jobs ran.
Repository-wide remediation remains incomplete.


## MPS Z/ZZ chain rejects out-of-range terms

The chain expectation sweep now validates Z wire indices and ZZ left-bond
indices before contraction. Previously, negative indices or terms extending
past the state were never visited by the sweep and could be silently omitted
from the returned energy. Such inputs now raise ValueError instead of returning
a partial Hamiltonian result. Valid formulas, precision, and gradients are
unchanged; no signature or serialized schema changed.

Four Hamiltonian-level regression cases failed against the original code
because no error was raised. Two boundary cases verify acceptance of the last
valid Z site and adjacent ZZ pair. The focused selection passed with **102
passed in 1.27s**. The CPU selection passed with **1704 passed, 13 skipped,
1744 deselected, 1 warning in 29.43s**. The warning is the existing PyTorch
complex-module warning. This round added six cases; additional shared-tree
suite growth belongs to other sessions.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. Strict checking was not rerun for this bounds fix; its latest baseline
remains 144 errors in 32 files. No GPU or remote jobs ran. Repository-wide
remediation remains incomplete.


## MPS product observables reject out-of-range wires

The shared product-operator contraction now checks wire bounds before creating
its environment. Previously, single-site Z and X/Y/Z product requests outside
the MPS were ignored by the site sweep, potentially returning the state norm
instead of an observable expectation. The common guard covers these callers
without duplicating validation or changing valid contraction formulas.

Eight unit cases cover negative and upper-bound wire indices through single-Z
and Pauli-product entry points. All eight failed before the fix because no
exception was raised. The focused MPS/observable selection passed with **99
passed in 1.13s**. The CPU selection passed with **1712 passed, 13 skipped,
1744 deselected, 1 warning in 30.74s**. The warning is the existing PyTorch
complex-module warning. Ruff, Black, public API baseline, architecture, and
diff whitespace checks passed.

Strict checking was not rerun for this bounds fix; the latest baseline remains
144 errors in 32 files. No GPU or remote jobs ran. Repository-wide remediation
remains incomplete.


## Density Pauli expectations contract the trace directly

Dense density-matrix expectation now evaluates trace(rho * P) as the direct
contraction sum(rho[i,j] * P[j,i]), rather than computing the entire matrix
product and discarding its off-diagonal entries. For dimension D, the trace
contraction requires quadratic rather than dense-matmul cubic arithmetic and
avoids the full product output. The dense Pauli operator itself is still
materialized; this change does not remove its quadratic storage requirement.

Four tests compare the complex Y tensor X observable against explicit matrix
products for batched and unbatched complex64/complex128 inputs. Forward values
and density gradients agree. The focused selection passed with **32 passed in
1.34s**. The CPU selection passed with **1717 passed, 13 skipped, 1744 deselected,
1 warning in 30.56s**. The warning is the existing PyTorch complex-module warning;
this round added four cases, with other shared-tree growth left intact.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. Strict checking was not rerun; its last baseline remains 144 errors
in 32 files. No measured speedup or accelerator capacity is claimed, and no
GPU or remote jobs ran. Repository-wide remediation remains incomplete.


## Dense Pauli matrices reject out-of-range operators

Dense Pauli-product construction now rejects wire indices outside its declared
qubit count before allocating the operator. Previously, those entries were
never visited during Kronecker construction and were silently replaced by
identity factors. Density-matrix term expectations could therefore return an
incorrect identity expectation. Valid operator construction is unchanged.

Two Hamiltonian-term regressions for negative and upper-bound indices failed
before the fix. The focused density/algorithm selection passed with **26 passed
in 1.39s**. The CPU selection passed with **1719 passed, 13 skipped, 1784
deselected, 1 warning in 28.83s**. The warning is the existing PyTorch
complex-module warning. Ruff, Black, public API baseline, architecture, and
diff whitespace checks passed.

Strict checking was not rerun; its last baseline remains 144 errors in 32
files. No GPU or remote jobs ran. Repository-wide remediation remains incomplete.


## Directed topology and physical-plan serialization typing

Directed routing now declares its adjacency sets and names the optional
misplaced logical wire separately from loop indices. The BFS, swap sequence,
and layout restoration algorithm are unchanged. Physical-plan identity
serialization keeps its instruction dictionaries in an explicitly typed local
list, which is also used for version-specific field removal. The same list is
placed in the payload; no extra copy, schema, or serialized field was added.
The invalid type-ignore comment was removed rather than broadened.

All four strict errors in directed_topology.py and physical_plan.py are resolved.
The complete shared-tree check reports **144 errors in 32 files across 442
source files**, down from the latest 148-error check. Existing directed topology,
physical-plan, topology legalization, and candidate-contract tests passed with
**55 passed in 0.88s**. The CPU selection passed with **1721 passed, 13 skipped,
1786 deselected, 1 warning in 30.02s**. Test-count growth includes concurrent
work; this round added no tests. The warning is the existing PyTorch
complex-module warning.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. Public signatures, routing semantics, and plan identity payload fields
were preserved. No GPU or remote jobs ran. Repository-wide remediation remains
incomplete.


## Directed topology validates integer identifiers before normalization

DirectedCouplingMap now validates wire counts and edge endpoints before sorting
and storing them. Previously, int conversion silently accepted booleans,
fractional values, and numeric strings, potentially constructing a different
physical topology than the caller specified. The constructor now follows the
same exact-integer convention as its existing wire-query validator. Valid
integer topology identities, edge ordering, and routing behavior are unchanged.

Six unit regressions failed before the fix because malformed values were
accepted. The focused directed-routing and physical-plan selection passed with
**52 passed in 1.15s**. The CPU selection passed with **1727 passed, 13 skipped,
1786 deselected, 1 warning in 30.28s**. The warning is the existing PyTorch
complex-module warning. Ruff, Black, public API baseline, architecture, and
diff whitespace checks passed.

Strict checking was not rerun; the latest baseline remains 144 errors in 32
files. No GPU or remote jobs ran. Repository-wide remediation remains incomplete.


## Undirected coupling configuration rejects lossy integer coercion

CouplingMap now checks integer types for wire counts, edge endpoints, path-cache
capacity, and grid dimensions before normalization. Fractional values, booleans,
and numeric strings no longer silently select a different topology or cache
limit. Existing valid-integer edge deduplication, self-edge handling, routing,
and cache behavior are preserved.

Fifteen validation cases cover five configuration fields and three invalid
representations. The original implementation accepted twelve cases; three
False dimensions were rejected only after conversion to zero, with a misleading
positivity error. All now report the integer requirement directly. Focused
validation, cache, routing, and benchmark-contract tests passed with **28 passed
in 1.10s**. The CPU selection passed with **1742 passed, 13 skipped, 1786
deselected, 1 warning in 30.14s**. The warning is the existing PyTorch
complex-module warning.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. Strict checking was not rerun; the last baseline remains 144 errors
in 32 files. No GPU or remote jobs ran. Repository-wide remediation remains
incomplete.


## Undirected topology queries validate wires before cache access

CouplingMap's shared wire-query validator now rejects booleans, fractional
values, and numeric strings instead of coercing them to another wire. This
covers neighbors, both edge endpoints, and both shortest-path endpoints.
Validation happens before cache lookup and diagnostic counters are updated.
Valid integer queries, path ordering, and public signatures are unchanged.

Fifteen new regression cases failed against the original implementation because
all malformed queries were accepted. They now pass and verify that rejected
queries leave the warmed cache diagnostics unchanged. The focused topology,
cache, routing, and benchmark-contract selection passed with **43 passed in
1.12s**. The current shared-tree CPU selection passed with **1763 passed,
13 skipped, 1786 deselected, 1 warning in 32.03s**; the warning is the existing
PyTorch complex-module warning. Concurrent work also contributes to the total
collected tests.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. The language scan checked 2009 paths with no Han-script text; binary
images still need separate visual review. Strict checking was not rerun; its
last measured baseline remains 144 errors in 32 files. No GPU or remote jobs
ran. Repository-wide remediation remains incomplete.


## Coupling self-edges cannot bypass wire bounds validation

CouplingMap now checks endpoint bounds before discarding self-edges. Previously,
an edge such as (-1, -1) or (n_wires, n_wires) was silently ignored, allowing
invalid device identifiers through construction. Valid self-edges retain their
existing behavior: they are omitted from connectivity. No public signature or
serialized schema changed.

Two out-of-range regressions failed before the fix; a third case preserves
valid self-edge behavior and adjacency. Focused topology, cache, routing, and
benchmark-contract tests passed with **46 passed in 1.17s**. The shared-tree CPU
selection passed with **1767 passed, 13 skipped, 1786 deselected, 1 warning in
31.08s**. The warning is the existing PyTorch complex-module warning. Other
sessions also contribute tests to the shared-tree count.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. Strict checking was not rerun; its last measured baseline remains 144
errors in 32 files. No GPU or remote jobs ran. Repository-wide remediation is
not complete.


## Topology factories validate wire counts before edge generation

The line and ring factories now enforce the constructor's integer wire-count
requirement before arithmetic or range creation. Fractional values and numeric
strings previously leaked incidental Python TypeErrors before reaching topology
validation; they now receive the same actionable ValueError as direct
construction. Boolean counts remain rejected. Valid topology generation and
public signatures are unchanged.

Six regression cases cover both factories with booleans, fractions, and strings.
Four failed before the fix; both boolean cases already passed. The focused
validation, cache, routing, and benchmark-contract selection passed with
**52 passed in 1.18s**. Ruff, Black, public API baseline, architecture, and diff
whitespace checks passed. This narrow validation change was checked with focused
tests; the full CPU suite was not rerun in this round.

A fresh strict mypy run with Python 3.12 checked **443 source files** and reported
**144 errors in 32 files**. The run did not report routing.py errors. This is a
current measurement, not a claim that repository-wide typing is complete. Logs
are in /private/tmp/fq-topology-factory-mypy.txt. No GPU or remote jobs ran.
Repository-wide remediation remains incomplete.


## Strict typing: TN plan inputs and hybrid optimizer boundaries

The internal TN ownership-DAG planner now declares both contraction and
expectation plans, matching existing callers and tested behavior. Both types
provide the path-search methods and tensor nodes consumed by the planner. This
removes two sliced-reverse argument errors without converting plans or changing
DAG schemas or numerical execution.

Hybrid optimization now isolates PyTorch's untyped single-tensor Jacobian API
behind a precise callable protocol and checks the returned object before tensor
operations. QNG retains vectorize=True and create_graph=False, its existing
normalization, real/imaginary derivatives, and timing boundaries. The LBFGS step
call has an explicit tensor-closure callable signature; its unused return stays
object-typed. No casts, type ignores, checker relaxations, or optimizer algorithm
changes were introduced. All three strict errors in optimization.py are removed.

Strict checking with Python 3.12 now reports **139 errors in 30 files, checking
443 source files**, down from **144 errors in 32 files**. The five removed errors
are the two TN input mismatches and three optimizer external-call errors.
Remaining errors comprise **102 in Triton kernels and 37 elsewhere**. This
remains an incomplete repository-wide remediation.

An initial concurrent check reported 420 errors and 27 test failures with source
locations inconsistent with the subsequently inspected files. Its focused run
collected 96 tests, while the rerun collected 105 before adding new tests. The
cause was not established; preserve the logs rather than treating that transient
run as a comparable baseline. Rechecking the current tree produced the 139-error
count and 105 passing focused tests. Logs: /private/tmp/fq-typing-batch-mypy.txt,
/private/tmp/fq-typing-batch-focused.txt, and
/private/tmp/fq-typing-batch-mypy-recheck.txt.

Three additional tests verify an analytic Jacobian, unchanged input gradients,
and rejection of non-tensor external returns. Final focused optimization, TN
DAG, and sliced-task verification passed with **108 passed in 11.85s**. Ruff,
Black, public API baseline, architecture, and diff whitespace checks passed.
The language scan checked 2011 paths with no Han-script text; binary images
still require visual review. No full CPU suite, GPU, or remote jobs ran in this
round. CPU TN checks do not establish multi-GPU or multi-node capacity.


## Strict typing: Jiuding program submission transport requirements

The private program-submission mixin now declares its required target-resolution
and script-submission methods as abstract methods. JiudingClient already
implements both with matching signatures and remains concrete. This makes the
transport dependency explicit and eliminates two missing-attribute errors
without adding fallback implementations or changing public submission defaults.

The managed SSH helper now accepts explicit binary input and timeout arguments
and returns CompletedProcess[bytes]. All existing callers use these options;
removing unrestricted keyword forwarding prevents text-mode options from
contradicting the binary stderr decoding and result transport. This removes the
missing generic argument error. Existing Any-typed client and record surfaces
are not claimed to be fully remediated.

Strict checking with Python 3.12 reports **136 errors in 28 files, checking 443
source files**, down from **139 errors in 30 files**. The three removed errors
are all in the two modified remote transport modules. The log is
/private/tmp/fq-remote-types-mypy.txt. No casts, type ignores, or checker
relaxations were added.

Two local subprocess tests verify binary input/output, argument quoting,
timeout forwarding, and failed response decoding. Combined with the Jiuding
client, program-submission, and user-path tests, **66 passed in 1.04s**. Ruff,
Black, public API baseline, architecture, and diff whitespace checks passed.
No full CPU suite, remote jobs, or GPU/QPU execution ran in this round.
Repository-wide remediation remains incomplete.


## Strict typing: PyTorch compute resource constructors and smoke backward

The CUDA compute adapter now describes the untyped PyTorch Stream constructor
with an exact keyword-call protocol and the Event constructor with a zero-arg
callable type. Device checks, stream priority, event device context, returned
vendor objects, and public method signatures remain unchanged. These declarations
match the installed PyTorch constructors and remove two untyped-call errors.
FlagGems smoke validation now calls torch.autograd.backward(objective), preserving
its requires-grad guard and scalar objective while removing one untyped method
call. No casts, type ignores, or checker relaxations were added.

Strict checking with Python 3.12 reports **133 errors in 26 files, checking 443
source files**, down from **136 errors in 28 files**. The three removed errors
are in compute/pytorch.py and compute/flaggems.py. The log is
/private/tmp/fq-compute-types-mypy.txt.

Three added cases verify CUDA constructor argument forwarding and rejection of
CPU devices before invoking a CUDA constructor. Platform consistency and operator
backend tests include complex CPU smoke forward/backward. Combined focused
verification passed with **36 passed in 0.66s**. Ruff, public API baseline,
architecture, and diff whitespace checks passed. Black initially requested two
formatting changes; those were applied and Black and Ruff passed again.

CUDA constructors were replaced by test doubles. These checks do not verify
real CUDA execution or FlagGems GPU performance. No full CPU suite or remote
GPU/QPU jobs ran. Repository-wide remediation remains incomplete.


## Strict typing and four-device CPU verification of JAX execution

JAX x64 configuration uses an explicit string/bool callable boundary. The
statevector shard_map executor now declares the PartitionSpec constructor for
its named-axis inputs. JAX remains lazily imported at runtime; the protocol's
return type is imported only during type checking. Configuration values,
partition specs, public signatures, and schemas remain unchanged. No casts,
type ignores, or checker relaxations were introduced.

Strict mypy checking with Python 3.12 reports **130 errors in 24 files, checking
443 source files**, down from **133 errors in 26 files**. The three removed
errors are in the two JAX executor modules. Final log:
/private/tmp/fq-jax-types-final-mypy.txt.

Running with JAX_PLATFORMS=cpu and four forced host devices exposed a pre-existing
shard_map index-shape bug: the local index block retains shape (1, local_size),
but the numerical initializer expects a one-dimensional index vector. The
pair-exchange parameter-gradient test failed with a broadcast from (1, 4) to
(1, 1). Restoring the former PartitionSpec construction in memory reproduced the
same failure without editing the working tree; the evidence is in
/private/tmp/fq-jax-types-before-sharding.txt.

The executor now reshapes each local index block to the planned local_size
before initializing amplitudes or applying gates. This restores the existing
numerical input contract without changing Simulation kernels. The original
failing value/gradient parity test then passed. Final hybrid JAX, distributed
planning/runtime, and no-JAX import tests passed with **148 passed, 1 skipped in
35.65s**. Ruff, Black, public API baseline, architecture, and diff whitespace
checks passed; an initial import-order lint issue was corrected.

The four devices are virtual CPU devices in one process. This verifies local
shard_map behavior and gradient parity, not real multi-GPU or multi-node
capacity. No remote GPU/QPU jobs ran. Repository-wide remediation is incomplete.


## Strict typing: Module gradient-hook and tensor-conversion boundaries

Module now stores debug gradient hooks as tuple[RemovableHandle, ...] rather
than Any. A private registration helper describes PyTorch's untyped hook API
and verifies its returned handle before storing it. Hook reuse and removal
remain governed by the existing correctness_debug policy. The parent Module
_apply call has an explicit tensor-transform/recurse callable signature; dtype
validation, precision updates, and cache invalidation are unchanged. No public
signature, cast, type ignore, or checker relaxation was introduced.

The module's strict errors decreased from six to four. A fresh repository check
reports **128 errors in 24 files, checking 443 source files**, down from **130
errors in 24 files**. An intermediate run reported 129 errors because
compiler/physical_plan.py referenced a helper before it appeared in the shared
working tree. This session did not modify that compiler file; a subsequent
read found the definition and the error disappeared on recheck. Both logs are
preserved: /private/tmp/fq-module-types-mypy.txt and
/private/tmp/fq-module-types-mypy-recheck.txt.

Two new tests verify that enabling debug checks reuses hook handles, disabling
them removes the nonfinite-gradient check, and malformed external handles are
rejected. Focused Module, training, and API-boundary tests passed with **71
passed, 1 warning in 2.60s**. The warning is the existing PyTorch complex-module
warning. Ruff, Black, public API baseline, architecture, and diff whitespace
checks passed. No full CPU suite or remote GPU/QPU jobs ran. The module's
remaining four errors concern compiled instruction/binding representation,
ParameterDict's mapping boundary, and the MPS execution-result plan type.
Repository-wide remediation remains incomplete.


## Strict typing: generic gate-matrix JVP boundary

The statevector reverse executor now declares the PyTorch JVP call for its
single scalar parameter and validates that the returned pair contains a tensor
derivative. Analytic rotation derivatives remain the fast path. Generic matrix
JVPs retain the same inputs, unit tangent, create_graph=False, and strict=False
settings. Malformed external returns fail before they can enter adjoint matrix
operations or mark parameter gradients ready.

Strict checking with Python 3.12 reports **127 errors in 23 files, checking 443
source files**, down from **128 errors in 24 files**. The removed error was the
untyped JVP call in reverse_adjoint.py. No casts, type ignores, or checker
relaxations were introduced. Log: /private/tmp/fq-jvp-types-mypy.txt.

Three additional cases force the generic path: a double-precision RY gradient
matches its analytic derivative, and two malformed JVP responses are rejected
without marking gradients ready. Focused reverse execution, training, and
adjoint numerical tests passed with **50 passed in 2.08s**. Black, public API
baseline, architecture, and diff whitespace checks passed. Ruff identified an
extra import separator; it was corrected and Ruff passed on recheck.

These are local CPU and single-rank checks, not GPU or multi-node capacity
verification. No full CPU suite or remote GPU/QPU jobs ran. Repository-wide
remediation remains incomplete.


## Strict typing: MPS workspace completion event construction

The factorization workspace pool now creates completion events through the
existing Compute PlatformRuntime interface, which owns CUDA event construction
and selects the requested device context. Event recording on the current device
stream, lease release, and waiting before buffer reuse retain their existing
order. An initial direct-constructor typing change exceeded the architecture
ceiling for torch.cuda references; routing through the existing platform removed
that violation. No public signature, cast, type ignore, or checker relaxation
was introduced.

Strict checking with Python 3.12 reports **126 errors in 22 files, checking 443
source files**, down from **127 errors in 23 files**. The removed error was the
Event call in runtime/executors/mps/factorization.py. Log:
/private/tmp/fq-mps-event-final-mypy.txt.

A new test checks that completion is recorded while the lease is still active,
and that acquiring the released buffer waits on the recorded event without
allocating another entry. Workspace pool tests passed with **4 passed in
0.65s**. Ruff, Black, public API baseline, architecture, and diff whitespace
checks passed. CUDA resources are test doubles; this is ordering-contract
verification, not real GPU concurrency or capacity evidence. No full CPU suite
or remote GPU/QPU jobs ran. Repository-wide remediation remains incomplete.


## Strict typing: optional RX/RZ loop return boundary

The statevector RX/RZ loop checks that the optional kernel returned a Tensor
before reshaping and restoring the wire layout. This narrows the lazy export's
Any result through a runtime check and removes the no-any-return error, without
forcing Triton imports during ordinary CPU execution. No casts, type ignores,
checker relaxations, or public signature changes were introduced.

Strict checking with Python 3.12 reports **125 errors in 21 files, checking 443
source files**, down from **126 errors in 22 files**. Log:
/private/tmp/fq-loop-types-mypy.txt.

Five new tests use a replacement kernel to check angle batching, restoration
of three wire positions, and rejection of two non-tensor returns. Broader native
circuit/backend verification reported **85 passed and 3 failed**. All three
failures occur while binding a local TCP listener, before distributed execution,
with sandbox PermissionError. A marker-based non-distributed selection still
included these tests because they do not carry the distributed marker; it
reported the same 85 passes and three environment failures. These tests are not
claimed as passing. Initial output: /private/tmp/fq-loop-types-focused.txt.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. The replacement kernel validates boundary layout only, not Triton
numerics or GPU performance. No remote GPU/QPU jobs ran. Repository-wide
remediation and environment-restricted distributed verification remain
incomplete.


## Strict typing: explicit existing planner forwarding imports

Runtime planner now explicitly aliases its existing schedule_layers and
lower_noise_model imports to their same names. Execution-plan reconstruction
already imports these names through planner; making the forwarding explicit
allows strict mypy to recognize those existing module attributes. The imported
function objects, call sites, __all__, signatures, and serialized plans are
unchanged. No direct Compiler dependency was added to execution_plan_contract,
and the existing architecture migration allowance was not expanded.

Strict checking with Python 3.12 reports **123 errors in 20 files, checking 443
source files**, down from **125 errors in 21 files**. Both removed errors were
implicit export diagnostics in execution_plan_contract.py. No casts, type
ignores, or checker relaxations were introduced. Log:
/private/tmp/fq-planner-export-types-mypy.txt.

Existing execution-plan contract, candidate, assembly, and CPU vertical-slice
tests passed with **25 passed in 1.98s**. Ruff, Black, public API baseline,
architecture, and diff whitespace checks passed. This is an explicit declaration
of an existing internal dependency, not a new plugin or public API surface.
No full CPU suite or remote GPU/QPU jobs ran. Repository-wide remediation
remains incomplete.


## Strict typing: complex BMM autograd wrappers

The ordinary and layout-aware complex BMM autograd functions now declare their
saved tensor context, layout metadata, forward arguments, and backward tuples.
Their apply entry points retain object returns until a Tensor check succeeds.
No Triton JIT kernel body, launch geometry, conjugation rule, public signature,
cast, type ignore, or checker configuration changed.

Strict checking with Python 3.12 reports **114 errors in 20 files, checking 443
source files**, down from **123 errors in 20 files**. All nine removed errors
were in the BMM autograd wrappers. The same file still has Triton JIT kernel
annotation/decorator errors, so it is not claimed as completely remediated.
Log: /private/tmp/fq-bmm-types-mypy.txt.

Four CPU cases execute the real custom autograd functions with replacement
launch functions implemented by torch.bmm. They compare forward results and
both input gradients for complex64/complex128, including noncontiguous tensors
and permutation restoration. The test loader replaces Triton decorators only
inside an isolated module; it does not validate Triton compilation or execution.
Combined with real/imaginary lowering tests, **10 passed, 7 skipped in 1.82s**.
Skipped accelerator-dependent tests remain unverified.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. No full CPU suite or remote GPU/QPU jobs ran. Repository-wide remediation
and real GPU kernel validation remain incomplete.


## Strict typing: MPS two-site autograd wrapper

The fused MPS two-site autograd wrapper now declares its saved-tensor context,
forward inputs, three backward tensors, and the apply return boundary. It
checks the external apply return before exposing a Tensor. Gate contraction
formulas, shared/batched gate behavior, Triton kernel bodies, launch parameters,
and public signatures are unchanged. No casts, type ignores, or checker
relaxations were introduced.

Strict checking with Python 3.12 reports **109 errors in 20 files, checking 444
source files**, down from **114 errors in 20 files**. All five removed errors
were in the two-site wrapper; six Triton JIT kernel annotation/decorator errors
remain in that file. The shared tree gained another source file independently
of this change. Log: /private/tmp/fq-two-site-types-mypy.txt.

Four new CPU cases execute the real backward implementation with an isolated
replacement forward launch. Direct einsum/autograd references verify gradients
for left tensor, gate, and right tensor, for shared and batched gates in
complex64/complex128. Together with the BMM wrapper tests, **8 passed in 1.18s**.
The broader CPU smoke/unit selection passed with **1816 passed, 13 skipped,
1794 deselected, 1 warning in 39.02s**. The warning is the existing PyTorch
complex-module warning. Full log: /private/tmp/fq-two-site-types-cpu.txt.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. The language scan checked 2030 paths with no Han-script text; binary
images still need separate visual review. Triton decorators and forward launch
are replaced in the CPU wrapper test; actual GPU kernels remain unverified.
No remote GPU/QPU jobs ran. Repository-wide remediation remains incomplete.


## Strict typing: statevector matrix and CX autograd wrappers

The single-qubit matrix and CX-sequence autograd functions now declare their
saved tensors, wire metadata, forward signatures, and backward return tuples.
Their apply results are checked before exposing tensors. Saved reverse CX
masks, conjugate/negative gradient resolution, kernel launch arguments, public
signatures, and Triton JIT bodies are unchanged. No casts, type ignores, or
checker relaxations were introduced.

Eight strict errors were removed from statevector_gates.py. The shared-tree
count changed from **109 errors in 20 files** to **107 errors in 21 files,
checking 444 source files**: eight removals in this session plus six new
union-attribute errors in runtime/compilation_evidence.py, which this session
did not edit. The six errors concern allocation fields read from a union of
physical-plan evidence versions. Rechecking confirmed the same count; do not
report a hypothetical 101-error baseline. Logs:
/private/tmp/fq-gate-types-mypy.txt and
/private/tmp/fq-gate-types-mypy-recheck.txt.

Six CPU cases execute the custom autograd wrappers with replacement launch
functions. Matrix tests cover shared/batched matrices and both complex
precisions; CX tests use noncommuting gates and a weighted loss to verify the
reverse gate order. Together with BMM and MPS wrapper tests, **14 passed in
1.54s**. These tests verify saved-context and gradient orchestration, not the
replaced CUDA forward/backward kernels themselves.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. No full CPU suite or remote GPU/QPU jobs ran. Repository-wide
remediation and real accelerator validation remain incomplete.


## Strict typing: allocation evidence narrows with its bundle version

The compilation handoff validator now reads allocated_plan from the bundle
inside the existing CompilationEvidenceBundleV3 branch. Previously it reused
an alias created before version narrowing, so static checking still treated
the plan as a union containing V1/V2 evidence without allocation fields. The
same object and comparisons are used at runtime; no evidence validation,
serialization, public signature, or version acceptance behavior changed.

All six errors introduced in compilation_evidence.py during the preceding
round are removed. Strict checking with Python 3.12 reports **101 errors in
20 files, checking 444 source files**, down from **107 errors in 21 files**.
No casts, type ignores, or checker relaxations were introduced. Log:
/private/tmp/fq-evidence-types-mypy.txt.

Existing physical resource allocation, directional topology, private compilation
contract, and artifact pipeline tests passed with **97 passed in 1.86s**.
Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. No full CPU suite or remote GPU/QPU jobs ran. Repository-wide
remediation remains incomplete.


## Strict typing: RX/RZ loop autograd and CPU tangent boundaries

The RX/RZ loop autograd function now declares its saved-tensor context, forward
inputs, and backward tensor tuple. Its apply return is validated as a Tensor.
The CPU tangent path now isolates the single-tensor Jacobian call and validates
its result before constructing complex derivatives. Existing vectorization,
parameter ordering, saved output, backward launch arguments, kernel bodies,
and public signatures are unchanged. No casts, type ignores, or checker
relaxations were introduced.

Strict checking with Python 3.12 reports **95 errors in 20 files, checking 444
source files**, down from **101 errors in 20 files**. All six removed errors
were in the loop wrapper and Jacobian calls. The file still has nine Triton JIT
kernel annotation/decorator errors. Log: /private/tmp/fq-rxrz-types-mypy.txt.

Four new CPU cases compare the real tangent implementation against central
finite differences, checking interleaved RX/RZ order at batch sizes one/two and
depths one/three. Two malformed Jacobian returns are explicitly rejected.
Combined loop and gate-boundary verification passed with **17 passed in 1.09s**.
The test loader replaces Triton decorators; the CPU tangent implementation and
PyTorch differentiation execute normally. CUDA forward/backward kernels are
not verified by these tests.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. No full CPU suite or remote GPU/QPU jobs ran. Repository-wide
remediation and real accelerator validation remain incomplete.


## Strict typing: two-qubit Pauli tangent Jacobians

The RXX/RYY/RZZ CPU tangent path now declares the single-tensor PyTorch Jacobian
interface and validates each real/imaginary component before combining them.
Both components still execute in order with vectorize=True. Parameter ordering,
reference rotation formulas, CUDA kernels, and public signatures are unchanged.
No casts, type ignores, or checker relaxations were introduced.

Strict checking with Python 3.12 reports **93 errors in 20 files, checking 444
source files**, down from **95 errors in 20 files**. Both removed errors were
untyped Jacobian calls. The file still contains three JIT kernel annotation and
decorator errors. Log: /private/tmp/fq-pauli-tangent-types-mypy.txt.

Four new CPU cases compare layer-major tangents with central finite differences
at batch sizes one/two and depths one/three. Two malformed external Jacobian
returns are rejected. Together with RX/RZ tangent checks, **12 passed in 1.09s**.
Triton decorators are replaced during isolated test loading; CPU differentiation
runs normally. These checks do not verify the real GPU tangent kernel.

Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. No full CPU suite or remote GPU/QPU jobs ran. Repository-wide
remediation and real accelerator validation remain incomplete.


## Strict typing: void return annotations for output-buffer kernels

Twenty-two Triton kernels across seven modules now explicitly return None.
Each selected function writes output buffers and has no value-returning return
statement. The two helper functions that return computed values were excluded.
Before writing each file, AST comparison verified that removing only the added
return annotations reproduces the original tree exactly. Kernel bodies,
parameter annotations, decorators, launch geometry, and public wrappers did not
change. The edit inventory is /private/tmp/fq-void-kernel-annotations.json.

Strict checking with Python 3.12 reports **71 errors in 20 files, checking 444
source files**, down from **93 errors in 20 files**. All 22 removed errors were
missing return annotations. Untyped JIT decorators and missing parameter
annotations remain; this does not claim that the Triton modules are strict-clean.
No casts, type ignores, or checker relaxations were introduced. Log:
/private/tmp/fq-void-kernel-types-mypy.txt.

Existing CPU autograd-wrapper and tangent verification passed with **26 passed
in 1.95s**. These tests replace Triton decorators during isolated loading and
cannot certify real JIT compilation or GPU execution of the annotated kernels.
Ruff, Black, public API baseline, architecture, and diff whitespace checks
passed. No full CPU suite or remote GPU/QPU jobs ran. Repository-wide
remediation and real accelerator validation remain incomplete.
## Strict typing: dynamic Triton kernel parameters

Twenty-four JIT functions across seven kernel modules now annotate 127 dynamic
parameters with `tl.tensor`. Existing `tl.constexpr` annotations are retained.
AST comparison before each write verified that removing only the new parameter
annotations reproduces the previous tree. Bodies, launch geometry, and public
wrappers did not change. The edit inventory is
/private/tmp/fq-kernel-parameter-annotations.json.

Inspection of [Triton 3.1.0 JIT parameter handling](https://github.com/triton-lang/triton/blob/v3.1.0/python/triton/runtime/jit.py)
indicates that this annotation does not force a scalar argument type or set the
constant/constexpr flags. This source-level inference is not a substitute for
compiling and executing the kernels on supported accelerators.

Strict checking with Python 3.12 reports **47 errors in 20 files, checking 444
source files**, down from **71 errors in 20 files**. All 24 removed errors were
missing parameter annotations. Remaining diagnostics comprise 24 untyped Triton
decorators, two missing helper return annotations, and 21 errors elsewhere.
No casts, type ignores, or checker relaxations were introduced. Log:
/private/tmp/fq-kernel-parameter-types-mypy.txt.

Existing CPU autograd-wrapper and tangent verification passed with **26 passed
in 1.28s**. These tests replace Triton decorators during isolated loading and
do not verify real JIT compilation or GPU execution. Ruff, Black, public API
baseline, architecture, and diff whitespace checks passed. No full CPU suite or
remote GPU/QPU jobs ran. Repository-wide remediation remains incomplete.


## Strict typing: JIT helper returns and experimental artifact views

The two value-returning Triton helpers now declare their return types.
`_insert_zero_bit` returns a tensor; `_flattened_offset` permits both an integer
for an empty packed layout and a tensor for populated layouts. Only annotations
changed in these helpers. Real JIT compilation and GPU execution remain untested.

The first strict check reported **74 errors in 21 files, checking 445 source
files**: the two helper errors were removed, while the newly present experimental
artifact-view module contributed 29 diagnostics. The preceding baseline was 47
errors across 444 source files. This intermediate increase was not attributed to
the helper edits. Log: /private/tmp/fq-helper-return-types-mypy.txt.

Experimental artifact views now store the explicit union of supported Core
versions instead of `object`. Existing constructor validation remains in place.
Version-specific identity and payload digest access use concrete type narrowing;
obsolete type-ignore comments were removed. No new abstraction, cast, dependency,
checker exclusion, or stable API snapshot update was introduced.

The final strict check reports **45 errors in 20 files, checking 445 source
files**. The experimental module's 29 diagnostics and the two helper diagnostics
are resolved. The remaining 45 comprise 24 untyped Triton decorator diagnostics
and 21 diagnostics elsewhere. Log: /private/tmp/fq-artifact-view-types-mypy.txt.

Artifact lifecycle and CPU autograd/tangent checks passed with **16 passed in
1.37s**. Ruff, Black, public API baseline, and architecture checks passed. CPU
kernel-boundary tests replace Triton decorators and do not establish accelerator
compatibility. No full CPU suite or remote GPU/QPU jobs ran. Repository-wide
remediation remains incomplete.


## Strict typing: compiled Circuit binding storage

Circuit now declares the private `_parameter_bindings` attribute populated by
Module builder compilation. Its type is the existing Runtime `BuilderBindings`
class, imported only under TYPE_CHECKING. This does not initialize the attribute
on ordinary circuits or add a runtime import; existing optional lookup behavior
and task-local tensor binding remain unchanged. No cast, ignore, public API
snapshot update, or new contract was introduced.

Strict checking with Python 3.12 reports **44 errors in 20 files, checking 445
source files**, down from 45. The resolved diagnostic was Module's assignment to
an undeclared Circuit attribute. Log:
/private/tmp/fq-builder-binding-types-mypy.txt.

Module verification passed with **46 passed, 1 warning in 3.95s**, including
compiled-builder reuse and rebinding tests. The warning is PyTorch's existing
complex-module warning. Ruff, Black, public API baseline, and architecture
checks passed. No full CPU suite or remote GPU/QPU jobs ran.

Inspection of the remaining errors confirmed that named-parameter mappings and
mixed-key parameter bindings have different static key contracts. Compiled
instructions and ordinary instructions also remain distinct concrete types.
These boundaries were not hidden with casts, copies, or checker exclusions in
this pass. Repository-wide strict checking and accelerator validation remain
incomplete.


## Review preparation: public typing corrections

Prepared `docs/api-changes/FQ-API-TYPING-20260910.md` for five diagnostics
requiring public type-contract changes: three Circuit methods, the strict scope
exit annotation, and the CPU probe metadata protocol. The proposal records exact
signatures, static compatibility impact, owners, alternatives, and verification
requirements. API-owner approval is pending; production code and API snapshots
were not changed in this pass. The proposed five-error reduction is not a test
result. The latest completed strict check remains 44 errors across 445 source
files; no new full strict or test run was performed for this documentation-only
pass. Broader remediation remains incomplete.


## Approved public typing corrections implemented

The user approved `docs/api-changes/FQ-API-TYPING-20260910.md` with "do" on
2026-09-10 after the compatibility explanation. Implemented only its five
corrections: Circuit noise/inspection annotations, Literal[False] for strict
scope exit, and read-only CPU probe metadata properties. Existing bodies,
defaults, frozen observation storage, and result schemas are retained.

Full-package strict mypy with Python 3.12 reports **39 errors in 17 files,
checking 445 source files**, down from **44 errors in 20 files**. All five
removed diagnostics correspond to the approved proposal. No package checker
configuration, ignore, cast, or API snapshot was added or relaxed.

Behavioral verification passed **88 tests in 3.42s**. After replacing dynamic
gate aliases in the new tests with the existing statically declared `gate`
method, final verification passed **7 tests in 0.90s** (five contract tests and
two additional existing planner tests). This covers 90 distinct test cases
across the runs. The new test file passes a separate strict static check,
including assignments of mutable and frozen probes to CPUCapabilityProbe.
Full-package diagnostics remain measured independently without import silencing.

Ruff, Black, architecture, and public API checks passed. The release notes and
proposal record the approved compatibility impact and verified results. Logs
use the prefix /private/tmp/fq-approved-types-. No full CPU suite or remote
GPU/QPU jobs ran. Remaining package typing and accelerator validation are not
claimed complete.
