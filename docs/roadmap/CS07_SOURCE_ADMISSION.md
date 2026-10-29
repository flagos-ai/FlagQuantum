# CS07-A source admission and current gaps

Baseline: merged #964, `7bb7e39c330fd6ab19ab2d598c19f1d3f3b5f1aa`.
This review-sized prerequisite restores retained source provenance and proves a
shared arithmetic consumer seam. No production Qmod parser/adapter is added.

## Original corpus

Upstream: Classiq/classiq-library, commit
`2bfd835e73b6882fcc515a725098b04766c238e7`.
[Local original files and license](../../tests/fixtures/classiq_qmod/README.md)
retain source bytes, paths, Git blob SHA-1 and SHA-256 identities. Seven checks
cover three pairs of source hashes and the original MIT license checksum.

| Original | Declared semantics and independent reference | Missing shared capability | Parsed | Lowered | Executed | Verified |
| --- | --- | --- | --- | --- | --- | --- |
| X.qmod | Fresh output qbit, allocate, then X; final q=1 | Fresh output/register effect profile and native parser | No | No | No | No |
| synthesis_tutorial.qmod | Fresh x:qnum<3>, H on all x bits; fresh y=x*x+1; eight positive amplitudes 1/sqrt(8) | Ordered preparation, output initialization, native parser and whole-program composition | No | No | No | No |
| multiplication_2vars_example.qmod | Initialize quantum a=4, b=5; fresh res=a*b; final (4,5,20) | Multiple live outputs, quantum constant initialization, native parser and composition | No | No | No | No |

Counts in this PR: source-integrity admitted **3/3**; parsed **0/3**, lowered
**0/3**, executed **0/3**, full-state verified **0/3**. Human trial records:
**0/2**. These are different denominators. No claim over the earlier 216-file
inventory is made by restoring three fixtures. Unspecified numeric widths remain
an adapter inference obligation; the literals and declared width above are not
an upstream serialized-layout oracle.

## Reuse, gap and PR order

Reuse Compiler's existing HybridProgram records and arithmetic semantics;
its lifecycle analysis, retain/recompute emitter, candidate selection, resource
estimator and target/layout evidence; and Services' checked plan/named reader.
Reuse Runtime/Simulation for execution. The current pure expression verifier
cannot accept allocation or ordered register effects, and Services' one-return
Python workflow cannot represent these originals merely by changing imports.

The [shared boundary contract](../api-changes/FQ-SYNTH-CS07-20261029.md) orders:
CS07-B Compiler initialization/preparation/output profile; CS07-C Ecosystem
bounded parser and semantic adaptation; CS07-D Services/integration conformance
and separately counted original-file evidence. Each must have its own meaningful
acceptance. A later public external-source entry requires API review; these
private boundaries do not add one by stealth.

The recorded-graph replacement probe in this PR uses a two-bit x and an existing
nonzero XOR target. It checks reuse of the same verifier/emitter for a different
producer, full relative phase and temporary cleanup under both schedules. It is
synthetic and does not parse either a reduced or original Qmod file. CPU results
are recorded below. No frontend support is inferred from
an already written analytic reference.

## Run the existing user workflow and boundary checks

```bash
python -m examples.algorithms.synthesis_trial
python tests/fixtures/classiq_qmod/verify_sources.py
python -m pytest tests/api_contract/test_synthesis_qmod_source_contract.py -q
```

The first command is the implemented Python workflow: basis (3,2,11), uniform
16-key joint probabilities and the supported line profile. The second command
checks source integrity only. The third checks provenance and the existing
shared arithmetic seam; its result must not increment original-file counts.
All run offline without Classiq SDK/cloud credentials. For installation, use the
[repository quick start](../../README.md#install-and-train-your-first-quantum-model).

## Independent developer handoff

[Developer trial instructions and blank record](SYNTH_DEVELOPER_TRIALS.md) target
the merged Python entry. No tester is enrolled or messaged by this PR. The two
independent developers must supply actual code, outputs, elapsed work, help and
blockers. Record failures as well as passes; fix an observed UX issue and retest
with an unseen task rather than filling in a success template. Keep current
release authorization false until real evidence and review justify changing it.

## Validation record

- All seven provenance digest checks passed; restored originals/license/manifest
  and verifier match the retained assets byte for byte.
- The new suite passed 11 cases: pinned identities, the offline verifier, four
  source/license corruption refusals, four full-state replacement scenarios
  (two producers by two schedules), and refusal of an unadmitted gate effect.
  Numerical checks use 18 qubits, complex128, full-amplitude tolerance 1e-12
  and dirty-workspace probability at most 1e-12.
- Final focused regression: 490 passed in 39.78 seconds, covering CS01-CS06,
  plan/readout admission and the subprocess user example. Both source producers
  use the same real Compiler verifier/lowering and Runtime; no Qmod is parsed.
- The default smoke/unit run reached 4,348 passes and 121 skips, then stopped
  at the same three existing macOS native-CPU QFT fusion-count failures:
  observed 44, expected 28 (181.39 seconds, 4,363 deselected, --maxfail=3).
  No Simulation code or expected counts were changed. Full local default
  testing is not reported as green.
- The new test module passes strict mypy; both new/copied Python files pass
  Ruff/Black. API migration, architecture, dependency, team scope, vocabulary,
  repository language, six-document links and generated-document checks pass.
  Production Python, stable contracts, export lists and release flags are
  unchanged from the merged baseline.

Numerical seam evidence remains separate from original-source compatibility and
from independent human usability. Parsed/lowered/executed/verified original
counts and human records remain zero. The new PR needs its own CI and review;
the successful #964 checks are merge evidence, not this PR's CI result.
