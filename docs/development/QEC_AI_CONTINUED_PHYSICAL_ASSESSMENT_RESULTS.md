# Continued-checkpoint physical assessment outcome

The fixed study completed successfully and passed its post-run retained-byte audit. It does **not** establish native logical-quality benefit: every native validation threshold was 1.0, all twelve native paths failed on the same 5319 of 4194304 physical shots as global-only, and every registered 10% reduction gate is inconclusive. The separate competitive-with-Fast conjunction is also unsupported. No optimization or deployment gate advances.

The [prospective protocol](QEC_AI_CONTINUED_PHYSICAL_ASSESSMENT.md) was published before the new data. Producing source is `67706be8a4c65fb8645611f032d1c0157a990830`; the later integration `937e323573861328ea2a37924aeb349868082978` and subsequent report commits did not produce these data. The source archive SHA256 is `763f321d7452881b9fb9026783b291063969404c957ce82233905f160c19d36b`.

## Frozen execution and retained evidence

The workload remains d=9/R=9/X/XV/p=0.003. All six existing checkpoints (seeds 11/22/33, absolute steps 200/2000) were retained. There were zero new optimizer updates, loss/model changes, timing repetitions or test-driven choices. CPU calibration used 4096 independent-DEM shots per seed at seed+14100000, 128 calls of 32, shared between horizons. This is 12288 independent calibration shots, not twice that number. Each horizon observes the same 48738304 mechanism labels per seed; those labels are not independent physical shots. The disclosed independent approximation of disjoint noise alternatives remains limited to training/calibration.

Validation used one original-circuit sampler at seed 15000011, 32 calls of 4096 (131072 shared physical shots), with all sixty native threshold trials retained. Temperatures used the existing BCE/tie rule; thresholds used the fewest complete logical failures with ties choosing the larger threshold. Frozen choices preceded a single 64-shot physical probe at seed 17000011 and complete zero/one/one-hot checks. The same saved probe passed actual CPU/CUDA residual/frame/final-prediction checks on both authorized hosts, on distinct recorded GPU UUIDs. The peer did not sample another probe.

Only A800-172 collected the primary stream: seed 16000011, one sampler, exactly 1024 calls of 4096, all fifteen paths on the same 4194304 shots. The three seeds, raw/calibrated modes, horizons and host replay do not multiply that sample count. The matching pinned image used Python 3.12.13/Torch 2.13.0+cu130, FP32, no TF32/autocast, one Torch thread, eight CPUs, 16 GiB, no network, read-only source/whitelisted inputs and auto-removed task containers. No old primary or heldout arrays were mounted. The global 24-hour limit did not trigger; both task containers were removed after success.

The [public evidence directory](evidence/qec_ai_continued_physical_20261025) retains all eighteen scientific JSONs byte-for-byte in `raw-metadata.tar.gz`, unchanged stage logs, a separate post-run audit, and archive/file inventories. Eleven smaller original scientific JSONs are also exposed directly; six stage plans with no trailing newline and the complete 1.96 MB primary manifest remain only in the byte-preserving archive. Full binary score/physical/prediction shards, inspected container metadata and private controller files remain in the verified local archives `/Users/baai/Documents/liuwei/qec-ai-continued-physical-20261025` and the original private host task directories. Binary predictions and weights are not included in the repository. This publication permits inspection of every registered count and selection; reproducing every bit requires those retained binaries and authorized inputs.

## Calibration and validation choices

| Seed | Step | Raw BCE | Calibrated BCE | Temperature | Raw threshold | Calibrated threshold |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | 200 | 0.04673931575 | 0.00981532249 | 0.5 | 1.0 | 1.0 |
| 11 | 2000 | 0.00438647645 | 0.00408021699 | 0.5 | 1.0 | 1.0 |
| 22 | 200 | 0.08501663272 | 0.01246463826 | 0.25 | 1.0 | 1.0 |
| 22 | 2000 | 0.00591322744 | 0.00478675561 | 0.5 | 1.0 | 1.0 |
| 33 | 200 | 0.06785155072 | 0.01122149713 | 0.5 | 1.0 | 1.0 |
| 33 | 2000 | 0.00509336480 | 0.00438510570 | 0.5 | 1.0 | 1.0 |

These are saved calibration-distribution scores, not logical-error estimates or a proof of training sufficiency. Every native threshold chose the valid inactive 1.0 control; no threshold was lowered afterward. The report retains the original [training-response result](QEC_AI_TRAINING_SUFFICIENCY_RESULTS.md) and [label-identifiability result](QEC_AI_LABEL_IDENTIFIABILITY_RESULTS.md), without using their heldout data to select this test.

## Complete primary comparisons

The family contains all 26 predeclared comparisons, with family alpha 0.05, comparison alpha 0.05/26, and each marginal Clopper-Pearson miscoverage alpha/2. The ratio bound is outward U(candidate)/L(reference). The original Hoeffding summaries remain beside every exact result in the raw summary. Paired p-values are auxiliary; the Fast/global nominal p-value 0.0486725121 does not meet 0.05/26 and does not certify a gate. No method or alpha was changed after data.

Global-only, mask-only and every native path have 5319 failures (LER 0.0012681484); official pretrained Fast has 5147 (LER 0.0012271404). Every path sees the same physical samples. Fast is an external pretrained checkpoint, not an equal-training-budget architecture control.

| Candidate | Reference | Candidate failures | Reference failures | Wins | Losses | Exact ratio upper | Paired p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| native-11-step-200-raw | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-11-step-200-raw | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-11-step-200-calibrated | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-11-step-200-calibrated | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-11-step-2000-raw | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-11-step-2000-raw | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-11-step-2000-calibrated | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-11-step-2000-calibrated | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-22-step-200-raw | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-22-step-200-raw | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-22-step-200-calibrated | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-22-step-200-calibrated | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-22-step-2000-raw | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-22-step-2000-raw | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-22-step-2000-calibrated | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-22-step-2000-calibrated | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-33-step-200-raw | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-33-step-200-raw | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-33-step-200-calibrated | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-33-step-200-calibrated | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-33-step-2000-raw | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-33-step-2000-raw | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| native-33-step-2000-calibrated | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |
| native-33-step-2000-calibrated | fast | 5319 | 5147 | 3676 | 3848 | 1.1323643776 | 0.04867251212 |
| fast | global | 5147 | 5319 | 3848 | 3676 | 1.0602919318 | 0.04867251212 |
| mask-only | global | 5319 | 5319 | 0 | 0 | 1.0949100101 | 1 |

For each of the three required step-2000 calibrated seeds, the ratio upper bound versus global-only is 1.0949100101, exceeding the fixed 0.9 target. The all-three native quality conjunction is therefore **inconclusive**. Their upper bound versus Fast is 1.1323643776, exceeding the separate 1.05 competitive limit; that statement is unsupported too. Failure to certify competitiveness does not prove inferiority. Fast/global has 3848 wins and 3676 losses, ratio upper 1.0602919318, so its lower point failure count also does not establish the predeclared reduction.

All six descriptive step-2000/step-200 comparisons have 5319 failures on each side and zero wins/losses; they are not extra independent experiments. The inactive selected controls prevent interpreting score response or raw activation as a measured LER improvement. No causal claim about representation, architecture or the reason thresholds were inactive follows from this study.

## Inspect the retained result

This standard-library example reads public archived results; it performs no sampling, training or GPU work. Run from the repository root.

```python
import json
from pathlib import Path
import tarfile

base = Path('docs/development/evidence/qec_ai_continued_physical_20261025')
summary = json.loads((base / 'study/summary.json').read_text())
with tarfile.open(base / 'raw-metadata.tar.gz') as archive:
    manifest = json.load(archive.extractfile('primary/manifest.json'))
fast = next(row for row in summary['comparisons']
            if (row['candidate'], row['reference']) == ('fast', 'global'))
print(summary['independent_physical_shots'], len(manifest['shards']))
print(len(summary['comparisons']), summary['native_quality_gate'])
print(fast['counts']['baseline_failures'], fast['counts']['hybrid_failures'])
print(summary['competitive_with_fast'])
print(json.loads((base / 'audit.json').read_text())['all_checks_passed'])
```

Verified output:

```text
4194304 1024
26 inconclusive
5319 5147
False
True
```

## Verification and limits

The producing code passed 204 focused tests locally and in the matching CPU image, followed by real six-checkpoint zero-sample preflight on both hosts. The private controller passed eight orchestration refusal/timeout tests. The post-run audit passed four independent reduction/refusal fixtures before executing in a fresh CPU-only matching-image container with sampling and training entry points forbidden. It checked both complete archive transfers and every file hash; all six checkpoint/DEM/version/dependency identities; source/protocol and frozen-choice bindings; both real device receipts; all 384 calibration score files and paired stream hashes; saved BCE/Brier reductions and temperature rules; all 32 validation shards and sixty threshold counts/ties; all 1024 primary input/truth/fifteen-prediction shards; independent four-cell count reductions for all 26 registered and six descriptive comparisons; exact/legacy summaries, family allocation and conjunctions. Validation model inference was not repeated: its retained counts and tie decision were checked with the previously verified composition and selection source. No new sample or optimizer update was performed by collection/audit.

The final integrated implementation head passed full CPU/GPU CI, including CI complete. Two artifact-upload ECONNRESET failures were recovered by targeted job retries; tests and gates were not weakened. This report-only head requires its own complete CI. For this report, pr-default initially could not collect in the API snapshot because its Git index was absent. After indexing exactly the snapshot and reviewed report files, the retry reached its explicit 120-second limit around 8%; neither run is passing evidence. The source snapshot cannot certify Git history; full CI must do that.

This is one circuit/noise/precision point. The larger fixed budget improved precision but did not establish native benefit. Do not pool it with the old 65536-shot study, extend sampling to significance, select another seed/horizon, change the loss or promote optimization/export/domestic-device/realtime work. Any subsequent scientific intervention requires a separate prospective budget and fresh calibration/validation/test streams.
