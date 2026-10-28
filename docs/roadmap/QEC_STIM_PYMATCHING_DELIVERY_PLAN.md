# QEC Stim and PyMatching delivery plan

Status: phase one accepted; continuing compatibility/performance delivery,
updated 2026-10-26 (Asia/Shanghai). Core owns QEC composition and adapters;
Simulation owns numerical execution; Runtime owns execution policy and lifecycle.
This document orders work and records the delivery authorization below. It does
not approve new public APIs or declare proposed modules implemented.

## Scope and continuing delivery authorization

On 2026-10-24 the user clarified that the long-running task covers the **whole
Stim/PyMatching compatibility roadmap**, requested this expected code structure
in the plan, and authorized the repeated cycle: implement the next bounded PR,
monitor its CI, merge after all applicable checks succeed and repository rules
are satisfied, then continue to the next step. This authorization covers PRs
within this compatibility roadmap, including #792; it is not limited to #788.
Do not end the follow-up merely because one successor PR is delivered.

Every PR description must contain executable **user-facing Python code**,
installation requirements and observed/expected output. A CLI command alone is
insufficient. Show existing public APIs that a user can run; label proposed APIs
explicitly and never present them as implemented. Run the example and preserve
its environment and evidence. Documentation/design PRs show the existing user
workflow they support as well as the proposed change.

Before each merge, re-read the actual PR head, all applicable checks, complete
CI/GPU workflow conclusions, review requirements and branch rules. Merge the
verified head normally, without admin bypass, force updates or weaker checks.
Verify the real merge commit and its actual main workflows, then synchronize a
clean branch from current main. Preserve unrelated work and inspect other QEC
deliveries before implementing overlapping work. Diagnose genuine failures;
retry infrastructure faults only with evidence and a finite budget.

Stable API changes still require a concrete proposal and explicit API-owner
approval under `PUBLIC_API_PROTECTION.md`. This delivery authorization is not
approval of proposal 087 Decision C or an unspecified Stim bridge contract.
Prepare reviewable proposals and complete independent work before requesting
such a decision; do not request routine implementation or merge approval again.

## Continuing delivery progress — 2026-10-26

Phase-one compatibility acceptance and the subsequent private CPU preparation
changes are delivered. The long-running roadmap remains active: a merged small
slice does not mean the broader functional roadmap or performance work is done.
The following is the current progress ledger; the earlier acceptance records
and original incomplete studies below remain historical evidence.

| Scope | Delivered change | Current acceptance state |
| --- | --- | --- |
| First performance tranche: #825, #830, #840, #842, #853, #861, #867 | Bounded compatibility/preparation studies and private sampling, semantic-preparation and lowering improvements; see the [baseline](../development/QEC_COMPATIBILITY_PERFORMANCE_BASELINE.md), [GPU feasibility](../development/QEC_GPU_PARITY_FEASIBILITY.md) and [matched prepared-CPU control](../development/QEC_PREPARED_PARITY_CONTROL.md). | Completed. The original incomplete grids remain incomplete; they do not establish a general GPU benefit, capacity claim or complete distance matrix. |
| [#872](https://github.com/flagos-ai/FlagQuantum/pull/872) | Prepare CPU parity indices once per live sampler and share one byte conversion per draw; [implementation and bounded evidence](../development/QEC_PREPARED_CPU_PARITY.md). | Merged as `58c98a0cb282cb1857b4614d4fc6299dd8bec440`. Complete head CI `40207427072` and GPU `40207427031` succeeded. Merge Main smoke `40209372637` and development-container publishing `40209372875` succeeded. |
| [#877](https://github.com/flagos-ai/FlagQuantum/pull/877) | Reuse channel-factory calculations within one program construction, with fresh operator clones at every location; [implementation and bounded evidence](../development/QEC_NOISE_CHANNEL_TEMPLATES.md). | Merged as `329dd0203532a182a740892e4f884efbda41f5d2`. Complete head CI `40251868248` and GPU `40251868262` attempt 2 succeeded. Merge Main smoke `40257459145` succeeded. Container publishing `40257459136` was cancelled: CPU full succeeded, three other matrix jobs were cancelled; publishing is not called successful. |
| [#885](https://github.com/flagos-ai/FlagQuantum/pull/885) | Remove the repeated sorted-item list while encoding frozen hybrid attributes, preserving canonical JSON bytes and semantic hashes; [Compiler-owned implementation and limits](../development/HYBRID_CANONICAL_ENCODING.md). | Merged as `2f59b4ff6d6ac937e9c9202411cbe188aabe30a7`. Complete head CI `40261269948` (54 successful jobs) and GPU `40261269944` succeeded; all 60 checks succeeded or had their original skips. Merge-specific [Main smoke `40263032238`](https://github.com/flagos-ai/FlagQuantum/actions/runs/40263032238) completed successfully and [container publishing `40263032317`](https://github.com/flagos-ai/FlagQuantum/actions/runs/40263032317) completed successfully. Post-merge acceptance is closed. |

Main smoke is not complete main CI. Local default failures and incomplete runs
are retained in the linked slice records and are not relabelled as green because
remote CI succeeded. No old profile, paired study or incomplete grid was rerun
to update this ledger, and no new timing or memory-capacity result is claimed
for #885.

The central progress correction merged through [#889](https://github.com/flagos-ai/FlagQuantum/pull/889)
as `30ce56953ab63340975c1a40d7f2260fa1a76011`. Its complete head CI
`40269344562` and GPU `40269344489` attempt 2 succeeded; merge-specific Main
smoke `40271285539` succeeded. This closes the progress correction, not the
broader functional roadmap. Every later implementation/performance delivery
must synchronize this ledger and the functional-plan entry point; a progress-only
correction does not recursively require another progress PR.

The subsequent candidate review selected no further performance implementation.
Streaming JSON hashing would switch the current C encoder traversal to Python;
a separate finite typed-value encoding comparison did not meet its frozen
selection rule. Engine assembly exposed no safely removable duplicate across
its independent public calls. Preserve canonical bytes, SHA256, positioned
operator law, RNG scope and every stage's verification. These are closed
candidate decisions on the retained evidence, not claims that further gains are
impossible. Do not resample completed studies to search for a favorable result.

The approved nested-repeat bound conformance slice merged in
[#896](https://github.com/flagos-ai/FlagQuantum/pull/896) as
`b88bffb1cc45910800f50e9312c80162fde1aea3`. Its complete CI `40274844573`
(all 54 jobs), GPU `40274844586` (three successful jobs and three original
optional skips), all 60 checks and exact-merge Main smoke `40276610613`
succeeded. It changes no production grammar, numerical policy or dependency.

The API owner explicitly approved the complete FUNCTIONAL-005 / Decision B
contract on 2026-10-26. Its implementation merged through
[#904](https://github.com/flagos-ai/FlagQuantum/pull/904) as
`92f5269caa126cd4220e7612a5d2207c644e6fd3`, with reviewed head
`3ff5fca2de51b18ee1f2cb88e7fbd5fa96dee029`. The complete head CI
`40280575633` succeeded in all 54 jobs; GPU `40280575629` succeeded with
three successful jobs and three original optional skips. All 60 checks
succeeded or retained those original skips. Exact-merge Main smoke
`40283856972` succeeded. Development-container publishing `40283856979`
was cancelled with all four matrix jobs cancelled; its cause is not established,
and overall publishing is not called successful or retriggered.

The canonical sparse batch operation preserves complete existing records for
all five certified decoder types, prevalidates finite input before decoding,
and retains decoder-specific duplicate/refusal behavior. Local verification
passed 83 new real-SDK cases, 68 core cases, 328 related cases and 12 actual
source/wheel/sdist install profiles. The public-docstring runner plus new suite
passed 85 cases with SDKs and 70 in core. Its existing coverage assertion was
unchanged. The bounded local default remained incomplete: its new doctest
registration failure was fixed, while four native-fusion expectation failures
reproduced on exact unmodified baseline
`b88bffb1cc45910800f50e9312c80162fde1aea3`. Local default is not relabelled
green. Local artifact builds used existing Torch 2.13 and do not certify the
Torch 2.10 floor; complete reviewed-head CI is recorded separately. The existing
logical-only Tensor method is unchanged and no speed claim follows.

Decision C is authorized for design refinement only; its public experiment
implementation remains explicitly deferred. No overall completion percentage
is assigned to this evolving scope.

## Remaining compatibility and functional gaps

This inventory separates missing functionality from completed acceptance and
from owner decisions. A roadmap title alone does not approve an implementation.
The [functional plan](QEC_FUNCTIONAL_DEVELOPMENT_PLAN.md#progress) retains the
numbered sequence.

| Area | Actual state | Next permitted action and acceptance |
| --- | --- | --- |
| Phase-one native/external compatibility, A-I | Accepted through #822; three approved bridge functions and four actual optional-install profiles are delivered. | Maintain these contracts. Recheck affected paths when their implementation/dependencies change; do not repeat the original install or sampling studies merely to fill a progress row. |
| Nested bridge expansion limits | Approved behavior exists; explicit nested-product, exact-limit and aggregate-repeat regression coverage was delivered in #896. | Test symbolic 2-by-3 and 1000-by-1000 counts; refuse 1001-by-1001 and separate over-budget blocks before SDK analysis/compilation. Inspect each repeat body once; no sampled shots or expanded million-instruction allocation. This is conformance coverage, not a claimed production defect or capacity result. |
| FUNCTIONAL-001/002/003/004 | Acceptance/proposals, Clifford propagation and correlated Pauli faults delivered in #774/#776/#779/#789. | Preserve their independent mechanism, exclusivity and numerical evidence. No second code registry, Surface class, DEM owner or propagation engine. |
| Schedule/feedback Groups A-E | The finite approved matrix and mainline acceptance are already delivered in #837. | Do not reimplement it or promote it to generic Surface feedback, general frames or FTQC execution. |
| FUNCTIONAL-005 | Complete contract approved 2026-10-26; implemented and accepted through #904 as recorded above. | Preserve complete-record equality, input prevalidation/iterator consumption and existing duplicate/refusal behavior. Actual five-type SDK qualification and reviewed-head delivery gates passed; maintain those contracts when affected paths change. No speed claim. |
| FUNCTIONAL-006 | Logical-only CPU Tensor PyMatching batch decoding is approved and implemented. | Do not mistake it for full correction records. Further full-result optimization depends on 005 and must preserve edges/weights/convergence fields; no current implementation is selected. |
| FUNCTIONAL-007 | Compiled circuit sampling and fixed-shot example counts/Wilson intervals exist; a general public experiment result/entry does not. | Refine and approve Decision C before a new public experiment API. Preserve source sampling, fixed denominator, tail batches, all logicals and all-or-error failure handling. |
| FUNCTIONAL-008/009 | Threshold workflow and broader resource/accuracy tranche remain pending their dependencies/contracts. | No threshold fit, adaptive stopping or distance-scaled quality claim follows from the existing small fixed-round examples. |
| FUNCTIONAL-010/011/012 | Subsystem/Floquet contracts and end-to-end memories remain later candidates. | Review schedule/gauge/logical tracking contracts after dependencies; no new execution implementation is authorized now. |
| Executed-law precision boundary | Actual Kraus-law semantics and the strict 1e-12 comparison remain authoritative; the recorded nominal-decimal difference is not silently repaired. | A change to shared factory numerics, tolerances or public noise semantics needs a concrete separately approved proposal. No name/metadata substitution. |
| GPU, hardware and new private evidence publication | Existing bounded results remain scoped; no general advantage or capacity result is certified. | New experiments need the specific device/workload/access/budget authorization. New private payload disclosure needs its own anonymized scope review. Neither is needed for the selected conformance tests. |

### Decisions for the API owner

1. **Decision B / FUNCTIONAL-005 is approved.** The owner explicitly approved
   the complete [independent-shot contract](../development/QEC_FUNCTIONAL_005_BATCH_CONTRACT.md)
   on 2026-10-26 and authorized verification/delivery. Do not request this
   approval again. Preserve all five concrete types, complete records and
   decoder-specific validation/refusals. The implementation has no speed claim.
2. **Decision C / FUNCTIONAL-007 remains design-only.** Following the
   [delivered design review](../development/QEC_FUNCTIONAL_007_EXPERIMENT_DESIGN.md),
   the owner requested proceeding with the three recommended choices to prepare
   the [complete contract draft](../development/QEC_FUNCTIONAL_007_EXPERIMENT_CONTRACT.md).
   Required explicit batch size, default-only decoder selection and caller-owned
   evidence without persistence now guide that draft. The complete typed
   signature, result, strict source path and exception/support tables still need
   implementation approval. Neither B approval nor design delivery activates C.
3. Adaptive stopping/threshold analysis, subsystem/Floquet protocols, hardware
   studies and changes to noise numerics are **later separate decisions**. They
   are not prerequisites for the current approved conformance slice; do not
   bundle them into Decision B or ask for blanket approval.

The nested-repeat work and Decision B are delivered. Decision C remains a
separate design review, with implementation deferred. No overall completion
percentage is assigned to this evolving scope.

## Expected code structure after compatibility delivery

Keep the existing domain packages and authorities. The entry marked `candidate`
below describes a possible ownership location, not an approved import path or
an instruction to create an empty module. The Stim adapter's three functions
received explicit approval after design PR #800 merged.

```text
flagquantum/
  qec/
    codes/                       # Existing code families and code registry
      repetition.py
      surface/rotated.py         # The one existing RotatedSurfaceCode
    circuits/memory.py           # Memory declarations and measurement layouts
    noise.py                     # QEC noise settings; reuse shared NoiseModel
    dem/
      model.py                   # Owned detector-error-model semantics
      construction.py            # Fault-to-detector/logical model construction
      circuit_noise.py           # Circuit-channel mechanisms and alternatives
      text.py                    # Stim DEM text interoperability
      graph.py                   # Decoder graph derived from the owned DEM
    execution/
      context.py                 # Measurement-to-detector/observable maps
      sampling.py                # Compile once, bounded batches, handle lifetime
      signatures.py              # Fault placement and propagation composition
      experiments.py             # Candidate: common evaluation, only after review
    adapters/
      pymatching.py              # Owned graph in, owned logical predictions out
      stim.py                    # Approved: annotated experiment import/export
    decoders/
      registry.py                # The one existing decoder registry
      matching.py                # Existing native matching implementation
      belief_propagation.py      # Existing alternative decoder
      ...                        # Other existing decoder owners
    types.py                     # Existing shared records, not a second schema
  simulation/stabilizer/
    engine.py                    # Clifford numerical execution and Stim boundary
    _fault_signatures.py         # Simulation-owned fault propagation primitives

examples/qec/                    # Runnable user workflows, including both entrances
tests/qec/                       # Composition, interoperability and decoder evidence
tests/team/simulation/           # Numerical and replacement/conformance evidence
```

The flow is code/noise/experiment declaration -> circuit and measurement layout.
One branch executes through Simulation and produces bounded detector/observable
batches; another constructs the owned DEM/decoding graph and selects a decoder.
Evaluation compares predictions with the **same sampled observable bits**, then
reports counts, fixed-shot uncertainty and configuration. Raw circuit sampling
and a transformed matching graph have different roles: decoder conversion must
not replace the original noise distribution used to generate samples.

Stim has two boundaries: QEC owns experiment annotations and interoperability;
Simulation owns numerical Clifford execution. PyMatching remains an optional,
lazy decoder adapter. Vendor objects stay inside their adapter/engine boundary.
Keep core imports and registry inspection usable without either optional SDK,
and fail explicitly when an unavailable engine/decoder is selected.

Later internalization replaces a numerical implementation inside
`simulation/stabilizer/` or a matching implementation inside `qec/decoders/`.
It reuses code/layout, DEM, logical-prediction meaning and conformance tests;
external SDKs may remain optional independent references. No whole-library
vendoring, second QEC IR, second Surface class or speculative backend registry
is required by this plan.

## Route status and final acceptance

- A/B are delivered by the approved sampling design and merged #784.
- C's fixed-shot example statistics are delivered by merged #788; the proposed
  generic library experiment API remains a separate unapproved decision.
- D merged in [#792](https://github.com/flagos-ai/FlagQuantum/pull/792) as
  `3d39640d42cf9c698a7e16dab116aafb5af95288`. Its final head `47e9344d`
  passed 54 main CI jobs and the GPU workflow. The actual merge commit's
  Main smoke run `40077460827` succeeded; it is not complete main CI evidence.
- E's [annotated native/external Stim experiment bridge proposal](../development/QEC_STIM_EXPERIMENT_BRIDGE_PROPOSAL.md)
  merged in [#800](https://github.com/flagos-ai/FlagQuantum/pull/800) as
  `bfb1fd6f9d042a24515295e311daf15930af9cf2`, after 54 main CI jobs and
  the required GPU checks succeeded. One failed-job retry resolved a quay.io
  HTTP 503 without a new commit or changed checks. The API owner then explicitly
  approved the three functions and their complete proposal. Implementation
  [#806](https://github.com/flagos-ai/FlagQuantum/pull/806) merged as
  `f7e58d9536bb694761d9799b0972625f69d75a42`: its final head `75bc38a6`
  passed all 54 main CI jobs and the required GPU checks. Its actual merge
  Main smoke `40084803885` succeeded; this is not complete main CI evidence.
  Exact exported-law acceptance continues below. Neither delivery activates
  the separate sparse batch or generic experiment contracts.
- F/G/H reuse the already approved correlated-Pauli contract 088 and existing
  circuit-noise/sampling/DEM implementations. The capability reconciliation
  below maps their location, distribution and independent evidence to this
  route; no duplicate noise grammar or implementation is required.
  The bridge audit also found configured decimal rates differ from default
  executed Kraus probabilities through existing tensor precision. Preserve the
  failed 1e-12 configured-rate comparison and actual-law evidence; resolve the
  numerical contract rather than silently widening a scientific gate.
  The first public-API joint-law audit after #806 also exposed an independent
  text-transport defect: Stim's circuit pretty printer shortens numerical
  arguments, for example `0.004999999422580004` to `0.005`. Native export must
  serialize the actual SDK arguments with round-trip precision while retaining
  SDK target syntax and annotations. This correction changes no Kraus operator,
  native seeded stream or nominal-versus-executed precision policy. Its strict
  detector/logical/selected-joint comparisons retain the 1e-12 gate; the original
  transport failure and nominal-rate discrepancy remain distinct evidence.
- The executed-law text correction merged in [#813](https://github.com/flagos-ai/FlagQuantum/pull/813)
  as `fa586473e365aa30871e376e7aec7cff76538321`. Final head `a93ce86d`
  passed all 54 CI jobs (run `40088006267`) and Local GPU Gate
  `40088006059` (three successful jobs, three existing optional skips).
  Actual merge Main smoke `40089903875` succeeded; this is not full main CI.
- The [noise support inventory](../reference/QEC_STIM_NOISE_SUPPORT.md) reconciles
  contract 088 and the bridge with current placement, exclusivity and precision
  evidence. It changes no API or numerical policy. The configured-versus-executed
  precision disposition and capability reconciliation below are the current
  review slice; I's final CI/GPU and delivery gates remain separate.
- I closes the whole compatibility task, not merely an individual PR.

The noise inventory merged in #815 as
`e873723bd359832de3e718235373affd19544bab`, after final head `e1088bd3`
passed all 54 CI jobs (`40090617506`) and Local GPU Gate (`40090617508`).
Its actual merge Main smoke `40092131772` succeeded. Installation acceptance
now runs in the existing package job: four fresh wheel environments (core,
Stim, PyMatching, both) and a fresh sdist/both environment execute installed
QEC workflows outside the checkout. Each environment is released before the
next; reports retain the existing artifact manifest, package paths, versions,
actual optional-dependency availability and all installer/smoke logs. Native
memory, exported memory and original external text use the existing public
sampler/DEM/decoder APIs, including 16 + 16 + 3 tail batches. The step is
required by the existing package/CI-complete gate; failed installation is not
an optional skip. The source smoke tests alone do not certify installed use.
The actual CI reports and artifact verification are recorded below.

### Installed-user delivery and remaining final gate

[#819](https://github.com/flagos-ai/FlagQuantum/pull/819) merged as
`0370a686a9fc0ba5ef9239de501f20ea8a22c425`. Its final head
`241d0030e92a6f8270a2965abab328d73867b8e7` passed complete CI
[40097069997](https://github.com/flagos-ai/FlagQuantum/actions/runs/40097069997)
(all 54 jobs), and Local GPU Gate
[40097070014](https://github.com/flagos-ai/FlagQuantum/actions/runs/40097070014)
(three successes, three unchanged optional skips). All applicable checks,
reviews and branch rules were verified before the normal fixed-head merge.
Actual merge Main smoke
[40099436354](https://github.com/flagos-ai/FlagQuantum/actions/runs/40099436354)
succeeded; it is not a full main CI.

The CI-tested synthetic merge source was
`bb991ba2357de35904ac042366a3329df1d2731c`, whose Git parents were independently
verified as main `d744e0efc74fa789817230d757e4ec4deed50c26` and the exact final
PR head `241d0030e92a6f8270a2965abab328d73867b8e7`. The package job's actual
`qec-installed-bb991ba2357de35904ac042366a3329df1d2731c` artifact
(ID `12561238671`, ZIP SHA256
`ff1e4ef851e82e24b014b640906954b0618296283d9a4142b0f857c40267280f`)
contains five directly parseable complete JSON reports, isolated package paths,
SDK availability, versions and installer/stderr logs. All five `pip check`
results succeeded. The copied distribution manifest matched the separately
downloaded distribution manifest, wheel and sdist sizes and SHA256 values:

| Built artifact | Bytes | SHA256 |
| --- | --- | --- |
| `flagquantum-0.4.0-cp310-abi3-linux_x86_64.whl` | 3101900 | `5f3bc7d9a06b3493b1d6993a733d457fa27715210f9dde0d7c181200db55589d` |
| `flagquantum-0.4.0.tar.gz` | 2396786 | `a9bbad9634b4915529d3980962f2bc4ab35d60174e3e98b82b60872e294edc88` |

| Actual installed profile | Stim / PyMatching | Accepted user workflow |
| --- | --- | --- |
| Wheel core | absent / absent | Owned DEM, native decoder/extension, lazy imports and explicit unavailable-SDK refusal. |
| Wheel Stim | present / absent | Native Repetition Z and Surface X/Z plus original external circuit, bounded sampling and strict DEM conversion. |
| Wheel PyMatching | absent / present | Owned graph and native/PyMatching same-sample batch predictions; explicit unavailable-Stim refusal. |
| Wheel both | present / present | Both experiment entrances, all four sampling cells and same-sample logical evaluation. |
| Sdist both | present / present | Fresh native build and the same installed workflows outside the checkout. |

Each environment used Linux x86_64, Python 3.12.15 and Torch 2.10.0+cpu;
SDK-enabled profiles used Stim 1.16.0 and PyMatching 2.4.0. Each Stim-enabled
profile completed 35 shots per cell in batches 16 + 16 + 3. Matching-enabled
profiles observed zero failures per cell, which is small-sample execution
evidence, not a threshold or performance claim. Core's real missing-NumPy Torch
warning remains in `wheel-core-smoke.stderr.log`; NumPy was not added to core.
The old head's contaminated core JSON is retained as failed report-format
evidence and is not substituted for these five current-head reports.

### Capability reconciliation

| Exit concern | Delivered evidence and scope | Remaining boundary |
| --- | --- | --- |
| B/C: lifecycle and evaluation | #784/#788; `test_memory_sampler.py`, `test_fixed_shot_statistics.py`; one compiled stream, explicit close, tail batches, any-logical-bit failure and fixed-shot Wilson intervals. | At phase-one acceptance these contracts were unapproved. Current Decision B approval above supersedes the sparse batch boundary; C implementation remains deferred. |
| D: code families | #792; `test_surface_batch_evidence.py`; Repetition and Surface X/Z, d=3/5/7 forced faults and bounded d=9/13/17 examples. | Fixed three rounds and 35 shots establish no distance trend, threshold or GPU benefit. |
| E: two entrances | #806/#813; `test_stim_experiment_bridge.py`, `test_noise_law_bridge_evidence.py`; original-source sampling, reference normalization, shape, coordinates, repeat/control, strict refusal and cleanup/replacement boundaries. | No arbitrary reverse compiler or approximate DEM conversion. |
| F/G/H: noise and numerical authority | Approved contract 088 plus the [noise support inventory](../reference/QEC_STIM_NOISE_SUPPORT.md); gate placement, categorical Pauli alternatives, target order, Cartesian joint law and independent executed-law detector/logical/selected-joint comparisons. | Nominal decimal versus finite-precision executed-law comparisons still fail at `1e-12`; no factory/default/operator change is delivered. |
| I: installed users | #819's five actual isolated installed profiles, reports and independently checked built artifacts above. | Actual-main full CI/GPU passed at bd69; normal delivery and merge verification of this reconciliation remain gate 5. |

The numerical disposition applies the existing approved bridge contract: actual
positioned Kraus operators define execution, and source text must preserve that
law. The strict independent `1e-12` comparisons remain unchanged. For an exact
execution-paired DEM use native export followed by strict circuit-to-DEM
conversion; nominal analytical construction is not certified as bit-equivalent
to factory-quantized execution. This records a supported boundary without
converting the retained failed nominal comparison into a pass. See the noise
inventory for the explicit user path and factory limitation.

Full actual-main CI
[40099662374](https://github.com/flagos-ai/FlagQuantum/actions/runs/40099662374)
was dispatched at the #819 merge source. Main advanced during dispatch, so a
same-source pair was also started at
`bd69c1def50771135f2b7d482a603d3dd3796f1a`: CI
[40099910239](https://github.com/flagos-ai/FlagQuantum/actions/runs/40099910239)
and Local GPU Gate
[40099856383](https://github.com/flagos-ai/FlagQuantum/actions/runs/40099856383).
The #819 merge-source CI completed successfully. The same-source bd69 CI also
completed successfully with all 54 jobs, including CI complete; its GPU workflow
completed successfully with three successful jobs and three unchanged optional
skips. Actual bd69 Main smoke
[40099682578](https://github.com/flagos-ai/FlagQuantum/actions/runs/40099682578)
also succeeded. The full CI, rather than the smoke alone, supplies implementation
main acceptance. No queued/running result is counted as successful evidence.

The capability checks below are accepted at bd69 within the documented support
and precision boundaries. Final delivery gate 5 closes only when acceptance
record [#822](https://github.com/flagos-ai/FlagQuantum/pull/822) passes its complete
current-head CI/GPU and repository rules, merges normally, and its actual merge
and main checks are verified. The PR's delivery record retains those final
source/run identifiers and conclusions; this avoids an endless sequence of
commits that merely insert their own future merge SHA. Any subsequent numerical
or QEC change requires its own relevant verification before claiming acceptance.

The continuing task is complete only when all of the following are recorded:

1. Supported native-memory and external-Stim experiment entrances have runnable
   user examples and feed the same logical evaluation semantics. Preserve
   measurement/detector/observable parity and supported coordinates; publish
   explicit unsupported cases rather than promise arbitrary reverse compilation.
2. The supported noise-location/channel contract is approved and implemented;
   original circuit/DEM marginal, selected joint and logical distributions have
   independent evidence, including exclusive channel branches and boundaries.
3. Repetition and Surface cases, bounded repeated sampling, failure/cleanup,
   same-sample decoder comparisons and optional replacement boundaries pass
   their reviewed acceptance cases without threshold or performance overclaims.
4. Four real install profiles (core, Stim, PyMatching, both), plus built wheel
   and sdist smoke checks outside the checkout, prove dependency isolation and
   installed-user workflows. Fake missing-module tests alone do not satisfy this.
5. All delivery PRs are normally merged after their current-head CI/GPU and
   repository gates pass. Actual merge-commit/main checks, limitations, evidence
   and the final capability checklist are reconciled in this plan.

Do not stop at a green intermediate PR or start internal kernel replacement,
AI training or unrelated framework work as a substitute for these exit gates.
When all gates pass, report the complete route's results and stop the follow-up.
If a necessary API-owner decision or external resource is unavailable, finish
independent work, retain a concrete reviewable result and report the precise
remaining requirement without repeatedly retrying unchanged blockers.

## Current baseline and user workflow

The structural/retirement baseline is main commit
`d286e4172cb9af4d1553a79199ee71f919af8c7a` (PR #766).
[PR #772](https://github.com/flagos-ai/FlagQuantum/pull/772) merged on 2026-10-24
as `0cb4086270ddc47dc431ee9d5e76984198a2d997`, adding the approved Tensor-only
PyMatching batch method. Its final head was `c1f5a65`; the main CI and Local GPU
Gate workflows both succeeded. Future implementation PRs must start from main
containing this merge and inspect subsequent changes before implementation.

The workflow #772 supports is:

```bash
pip install -e '.[stim,pymatching]'
PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py
```

It circuit-samples 128 repetition-memory shots in bounded chunks and predicts
logical flips in batches: 8 detector columns, 1 observable, and an observed
logical failure count. The random count is version/architecture scoped.
[PR #784](https://github.com/flagos-ai/FlagQuantum/pull/784) merged as
`a5fc00898af53ab85037bec10c2db7579c68823b`, adding compiled sampler reuse and a
continuous seed-53 stream across chunks. Its final head `8c942053` passed all
54 main CI jobs and both local GPU gates. No performance claim follows from the
example.

The existing `PYTHONPATH=. python examples/qec/memory_layout_and_sampling.py`
workflow remains available with the Stim extra; it exercises measurement
layouts, sampling and feedback.

## Historical phase-one action: reconcile final acceptance

Work D and the expected structure are on main through #792; the explicitly
approved [bridge proposal](../development/QEC_STIM_EXPERIMENT_BRIDGE_PROPOSAL.md)
is implemented by #806. The executed-law correction is merged through #813, the noise inventory through
#815, and actual installed-user acceptance through #819. The current slice
reconciles the approved numerical authority, delivered evidence and remaining
final normal delivery gate. Actual-main full CI/GPU at bd69 succeeded.
This historical phase-one record changed no API, factory or scientific tolerance
and did not approve the then-proposed 005 or Decision C. The later explicit
Decision B approval and implementation are recorded above; C remains design-only.

[PR #788](https://github.com/flagos-ai/FlagQuantum/pull/788) merged as
`47db63d6dd243e5a2fc498440a0a660858eb06a2`. Its final head `9daf326b`
passed all 54 main CI jobs and the single/two-GPU gates. The merge commit's
Main smoke run `40066862426` succeeded; that smoke is not a complete main CI.
The Surface evidence slice started at main `4be2375eff47c112ce8d8e24f8206ed5592130d8`,
including subsequent correlated-fault and dynamic-preflight work. It changes
only the existing example, tests and documentation, with no new library API.

The #772 CI repair corrected Black formatting in two tests and registered the
new baseline example in `examples/README.md`. All 54 main CI jobs passed, as
did the single-GPU and required two-GPU jobs. No test assertions or frozen
fixtures were weakened to obtain this result.

The API owner approved the [reusable sampling proposal](../development/QEC_REUSABLE_SAMPLING_PROPOSAL.md)
on 2026-10-24. Work B's two additive QEC exports and lifecycle are now on main.
The first work C slice extends the existing runnable example using those public
APIs. It adds an illustrative JSON report and private example statistics helpers,
without adding a library result class, export or serialized-artifact contract.
The existing proposed library experiment API in
[proposal 087, Decision C](../development/API_CHANGE_PROPOSAL_087_QEC_SCALABLE_EXPERIMENTS.md#decision-c-fixed-shot-memory-experiment)
remains a separate unapproved decision. RepetitionMemoryResult retains all
per-shot reference records, while NoiseSweepPoint carries repetition-specific
noise-event counts; neither expresses this bounded general logical-bit summary
without changing a protected record.

## Ordered PRs

Letters are planning identifiers, not reserved GitHub numbers.

| Work | Deliverable and dependency | Acceptance and boundary |
| --- | --- | --- |
| A: reusable sampling design | Review existing Runtime/session ownership and propose the narrow lifecycle; #772 interface must be stable. See the [sampling proposal](../development/QEC_REUSABLE_SAMPLING_PROPOSAL.md). | Concrete repeated-batch workflow, RNG and release semantics; no SDK objects in public records; no speculative session platform. New public API requires explicit review. |
| B: compiled sampler reuse | After A approval and prerequisite integration, prepare one private Simulation sampler and compose QEC parity extraction in `execution/sampling.py`. | One lowering/compilation per handle; bounded batch storage; owner isolation; explicit close; immutable configuration snapshot. Existing one-shot APIs retain behavior. |
| C: fixed-shot evaluation | After #772, accumulate requested/completed/failed shots, vector logical failures, 95% intervals and experiment metadata. Reuse existing result concepts first. | Tail batches counted, zero-error upper bound, fixed denominator/stopping rule, all observables checked. Atomic batch failure stays an error; no silently discarded rows. Any new public result requires its own proposal. |
| D: Surface batch evidence | Extend C's workflow using existing rotated code, memory builder and measurement maps; B supplies scalable execution. | X/Z supported readouts; d=3/5/7 correctness and representative d=9/13/17 workloads; initial/final rounds, boundaries and forced faults; same-sample decoder comparisons. No second Surface class. |
| E: annotated Stim experiment bridge | After A/D boundaries stabilize, add QEC annotations to native export through the Simulation seam and accept external Stim experiments at an adapter boundary. | Measurement/detector/observable parity and coordinates preserved; raw DEM separate from decoder conversion; no promise of arbitrary Stim-to-native reverse compilation. |
| F: circuit-level Pauli noise contract | After D/E, define supported locations, single/two-qubit mixtures, preparation/readout faults and explicitly positioned idle faults. | Preserve exclusive branches, original distributions and mechanism identity; state unsupported cases; review any public/schema deltas before G/H. |
| G: consistent noise placement | Implement F in `execution/signatures.py` and `execution/sampling.py`; reuse Simulation Pauli-mixture support. | Forced-fault propagation, exact placement, mutually exclusive depolarizing branches; Y is not independent X/Z. Numerical kernels remain in Simulation. |
| H: matching DEM construction | Implement F using G's location vocabulary in `dem/construction.py`; apply decoder transforms only at the declared boundary. | Original circuit/DEM marginal, selected joint and logical distributions agree; graphlike decomposition never replaces the sampling law; unsupported correlations fail explicitly. |
| I: phase-one acceptance | After B/C/D/E/G/H, certify the two experiment entrances, installability, bounded memory and replacement boundary. | Four actual optional-dependency installation profiles, wheel/sdist checks outside checkout, fixed workload evidence and required gates. Fake SDK absence tests are not four installations. |

Cross-domain implementation changes follow an approved shared contract and land
by owner. G/H must not invent separate fault vocabularies. Existing flat imports
remain retired; frozen restructuring evidence is not regenerated to hide drift.

## Dependency order and exit gates

```text
#772 closes and merges
  +-- A lifecycle proposal -> B reuse -------------------------------+
  +-- C statistics -> D Surface -> E bridge -------------------------+-> I
                               +-> F noise -> G sampling -> H DEM --+
```

C's example statistics use the merged sampling API. The graph is a dependency
plan, not a request to spawn parallel agents. Avoid long-lived stacks of
unmerged implementation PRs.

Exit gates:

1. B: repeated sampling amortizes preparation with explicit ownership/RNG rules.
2. C/D: reproducible Repetition/Surface logical-failure comparisons, including
   statistical uncertainty, without threshold claims.
3. E/H: native and external experiments enter one evaluator; supported circuit
   noise and DEM remain consistent.
4. I: stop broadening phase one and start data/AI or bottleneck-driven kernels.

## Prepared acceptance cases for work C

These scenarios specify behavior, not a new library result schema:

- Fixed inputs: 100 completed shots with zero failures must report a positive
  upper confidence bound; 100 failures must report a lower bound below one.
- For a vector of observables, any differing bit counts as one failed shot;
  two differing bits in one shot do not count as two failed shots.
- Requested 35 shots with chunks of 16 must count 16 + 16 + 3. Invalid or failed
  batches must not silently decrease the denominator.
- Use fixed-shot Wilson 95% intervals as the initial candidate. Freeze the
  formula and validate against an independent analytical reference; do not use
  the same helper to compute expected values. Zero completed shots have no
  estimated rate or interval and must be represented explicitly, not as zero.
- Keep logical failure distinct from decoder execution failure. Under #772's
  atomic semantics a failed batch terminates evaluation with its failure
  context; row-level retry/discard is a separate policy change.
- Record seed schedule, batching, code, rounds, noise and actual library
  versions. Do not derive a per-round failure rate by dividing by rounds.

Any new public evaluation entry or result requires a concrete consumer/API
review. The first implementation should reuse the existing example and
statistics utilities where their semantics match.

## Work C: fixed-shot report semantics

The example retains its `run_baseline(shots=128, batch_size=32)` call and existing
report keys. The statistics change adds `requested_shots`; existing `shots`
means completed shots. Successful evaluations complete the entire request,
including a partial final batch. One shot fails if any logical observable
differs from the sampled flip; `observable_failures` additionally counts each
column. These counts are distinct: two wrong logical bits in a shot still mean
one failed shot.

`logical_error_rate` is `logical_failures / shots`, for one complete memory
experiment. `confidence_interval` is the two-sided fixed-shot Wilson 95% score
interval without continuity correction, following the
[NIST definition](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
The original report makes no uncertainty claim; this slice adds that explicit
interpretation. Zero observed failures have a positive upper bound. Zero
requested shots are refused, and zero completed shots have no estimated rate
or interval. An invalid or failed batch terminates with the original error and
logged batch/completed-shot context, releases the sampling context and returns
no partial success report. No row is retried, dropped or silently counted as
correct.

The report records code class/distance, rounds, the actual uniform Pauli noise
settings, initial seed, continuous-stream schedule, batch size, package/Python
versions and platform. It records `sampling="circuit"` and
`stopping_rule="fixed_shots"`. This is an example report for the existing
repetition workload, not an approved portable artifact schema or a generic
experiment service. It does not divide a failure rate by rounds, apply an
ordinary Wilson interval after adaptive stopping, or claim a threshold.

Verification uses score-test inversion by numerical bisection, an optional
independent [SciPy Wilson reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html),
controlled multi-observable batches, batch failure/cleanup and tail-count tests,
plus actual Stim/PyMatching runs with reproducible identical call sequences.
The statistics tests run without optional SDKs; the real-engine integration
tests retain their dependency guards.

```bash
python -m pytest tests/qec/test_fixed_shot_statistics.py tests/qec/test_stim_pymatching_baseline.py -q
PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py
```

## Work D: bounded Surface evidence

The existing baseline example now also accepts:

```bash
PYTHONPATH=. python examples/qec/stim_pymatching_baseline.py --code surface --distance 5 --basis x --shots 35 --batch-size 16
```

This uses the same code record, builder, compiled stream, PyMatching adapter and
fixed-shot statistics as the repetition example. The Z experiment reads the
declared row operator. The X experiment passes the certified product of X on
**all data qubits**, so preparation and readout both use the uniform X frame.
The code still declares only Z; requesting `readout_basis="x"` without a product
is not added here. Passing a minimal column product would rotate only that
column and would describe a different experiment. The report records the
chosen basis and full logical support to make this distinction inspectable.

Predeclared developer evidence budget: distances 3, 5, 7, 9, 13, 17 in both
frames, exactly three syndrome rounds, 35 shots in chunks 16 + 16 + 3 per cell,
seed 53, independent X/Z faults 0.005/0.007 per data location and syndrome
readout faults 0.009. Y probability is zero; no correlated mechanism is split
for matching. All twelve cells run once with the same call schedule, bounded
to 240 seconds in total. Retain every cell, including failures; do not change
seeds, append shots or relax tolerances after inspecting results. This is
small structural/capacity evidence, not a threshold, precision, GPU throughput
or advantage experiment. Three fixed rounds do not establish distance-scaled
memory performance. A zero count at 35 shots still has a 9.89% upper bound.

The declared twelve-cell run completed on macOS arm64, Python 3.12.14,
Torch 2.13.0, Stim 1.16.0 and PyMatching 2.4.0. Every cell completed all 35
shots and observed zero logical failures; every interval was
`[0.0, 0.09890099232440398]`. These counts are engine/architecture scoped.
This small all-zero sample does not establish a distance trend.

| Distance | Data qubits | Detectors | X failures/shots | Z failures/shots |
| --- | --- | --- | --- | --- |
| 3 | 9 | 24 | 0/35 | 0/35 |
| 5 | 25 | 72 | 0/35 | 0/35 |
| 7 | 49 | 144 | 0/35 | 0/35 |
| 9 | 81 | 240 | 0/35 | 0/35 |
| 13 | 169 | 504 | 0/35 | 0/35 |
| 17 | 289 | 864 | 0/35 | 0/35 |

Committed integration coverage compares raw measurement-map parities with the
compiled sampler at d=3/5/7 in both frames and one/three rounds, verifies
noiseless preparation/readout, and executes forced X/Z data faults at corner
and interior locations and syndrome faults on weight-two/four checks in first
and last rounds. Injected programs are intentionally non-identical by round;
their independent oracle uses the existing Simulation Clifford seam without
weakening the QEC sampler's identical-round placement guard. At d=3, native
matching, PyMatching per-row and PyMatching batch decode exactly the same
zero/single-mechanism syndrome rows. This pins single-fault correction; it
does not require arbitrary tied matchings to choose identical logical classes.

```bash
python -m pytest tests/qec/test_surface_batch_evidence.py -q
```

## Later internal implementations

- GPU DEM sampling consumes the owned mechanism semantics and bit batches.
  Training labels require pre-merge mechanism identity and geometry; logical
  outcomes alone do not reveal actual faults.
- An in-house Clifford/Pauli-frame kernel replaces the Simulation seam. It does
  not require QEC callers to construct vendor objects.
- An in-house matcher uses the existing registry and logical-prediction meaning;
  it must pass the same conformance evidence with an independent reference.
- AI pre-decoding follows three separate changes: labeled data, model training,
  and correction/global-decoder evaluation. Combined correction must preserve
  detector and logical effects, not only reduce event density.

Do not vendor whole libraries or remove independent optional references as an
end in itself. Start replacement work only with a measured bottleneck, hardware
coverage requirement or concrete maintenance benefit.

## Delivery rules and estimates

Each PR contains a runnable user example, expected behavior, failure cases and
appropriate evidence. It owns one coherent step; no second decoder registry,
QEC IR, service database, queue platform or speculative GPU scheduler is added.
Use Black and Ruff as configured by CI; Ruff formatting is not the formatting
gate. Preserve CPU and single-device usability and dependency isolation.

A/F are design reviews; B/G/H are implementation changes after their contracts
are approved. Persistent samplers, common batch protocols and new result types
are not implicitly approved by the roadmap. Publishing PRs and changing release
status follow the continuing delivery authorization above. Normal merges within
this roadmap are authorized after the stated gates; new public API decisions
remain separate.

Planning estimate: two QEC/PyTorch contributors, roughly 4-6 weeks for A-I,
including review time. The bridge and circuit-level noise are the largest
uncertainties. Unrelated repository CI repairs and new numerical kernels are
outside that estimate.

## Decision C contract preparation after design delivery

[#912](https://github.com/flagos-ai/FlagQuantum/pull/912) delivered the design
review as merge `b9b4f51963a4ab85e9253b3d070e798423082ee8`. Exact-head full CI
`40289425554` passed all 54 jobs; GPU `40289425526` attempt 3 passed three jobs
with three original optional skips. All 60 checks were 57 success / 3 original
skips. Its exact merge Main smoke `40300707460` succeeded; that is not full
main CI. Earlier GPU variance failures remain preserved; thresholds were not
changed. The owner specifically authorized cancelling one blocked Scheduled
hardware run to release the shared queue; no general cancellation permission
follows.

The owner then requested proceeding with the recommended three choices:
required explicit batch size, default-only registered decoder selection and
caller-owned evidence without initial persistence. The
[complete C/007 contract draft](../development/QEC_FUNCTIONAL_007_EXPERIMENT_CONTRACT.md)
now makes the signature, result, strict actual-law source path, support and
exception tables concrete for review. This is document preparation, not
approval of the complete contract or implementation. Older design-choice
wording above records the #912 publication snapshot. Decision B/005 remains
complete; C implementation, adaptive stopping, threshold fitting and persistence
remain deferred. No new experiment, performance study or hardware run was done.

## Approved C/007 implementation candidate

After the #924 contract artifact was delivered and its exact merge Main smoke
`40314835571` succeeded, the owner explicitly approved the complete
[C/007 contract](../development/QEC_FUNCTIONAL_007_EXPERIMENT_CONTRACT.md) for
implementation, verification and delivery on 2026-10-26 (Asia/Shanghai).
This supersedes earlier C design-only/deferred wording in historical progress
snapshots above. B/005 remains delivered. Adaptive stopping, threshold analysis,
nondefault decoder configuration and persistence remain separate decisions.

The candidate adds only `MemoryExperimentResult` and `run_memory_experiment` in
the canonical `qec.execution.experiments` module. It reuses original-source
export/strict actual-law DEM/one compiled sampler, five default registered exact
decoders through complete-record 005, exact final partial batches, all-logical
counts and one shared private Wilson formula. No root or parent re-export,
registry entry, dependency, channel law, RNG schedule or numerical tolerance is
changed. Zero-noise PyMatching isolated-detector refusal remains explicit.
Focused correctness and actual-SDK acceptance are in progress; exact-head full
CI/GPU, reviews/rules, normal pinned merge and exact merge/Main smoke acceptance
are still required. No performance, threshold, GPU or capacity claim is made.
