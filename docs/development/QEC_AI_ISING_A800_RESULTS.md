# Prospective d9 A800 Ising checkpoint comparison outcome

The complete two-host chain ran successfully. Native correction remained inactive;
all paired quality decisions are inconclusive. Global-only PyMatching was fastest
in this software configuration. No model, performance or hardware release gate is
promoted by these results.

## Protocol and implementation

The [prospective protocol](QEC_AI_ISING_A800.md) was published before training or
CUDA data collection. Producing source was branch commit
`7f34b84da0c0ff8bab69566abadf0b90234c032d`. The subsequent Linux RSS reporting fix
was made after collection; it does not change training, predictions or timing.
The source-file and protocol SHA-256 values in each raw manifest identify the
exact files executed. Do not substitute the final PR head for the producing code.

Both hosts used d=9, R=9, X memory, XV rotation, p=0.003, and the pinned upstream
25-parameter noise model. Native CPU training used seeds 11/22/33, 200 Adam steps
of batch 128, and learning rate 0.001. Calibration and validation each used 4096
shots per seed. All choices froze before sampling the 65536-shot primary circuit
stream (seed 6001011). The identical stream on the second host is a replication,
not a second independent test cohort. Three training seeds also do not multiply
the physical sample size.

The initial d9 smoke aborted before training or primary sampling because Stim's
flattened DEM retained duplicate complete detector/logical signatures. The fix
reuses the authoritative `merge_duplicate_mechanisms()` independent-parity rule:
13492 raw mechanisms become 11899 canonical mechanisms. Their detector degrees
are 1:200, 2:3224, 3:1944, 4:6531. Hyperedges remain intact; detector-identical
mechanisms with different logical contributions remain separate. Training and
matching still use the disclosed independent approximation of disjoint Pauli
alternatives; test shots come from the original circuit.

Each host passed exact CPU/CUDA residual and logical-frame checks on 786 probes
for every frozen path/seed before primary sampling. Final logical predictions
were also equal on the 64 physical probes. Synthetic one-hot/all-one probes do
not necessarily admit a perfect matching, so only their correction bits are
compared, as declared before collection. No CUDA fallback was used.

## Environment and evidence

Both runs used an isolated single A800-SXM4-80GB, FP32 inference, TF32/autocast
disabled, deterministic algorithms, one PyTorch CPU thread, and image
`sha256:9465e1fca0a8c22af1b2ee657c8a6ee7e9f9f8e74d873b0add84b274da58b9e2`.
Python was 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0, driver 580.126.20,
Stim 1.15.0 and PyMatching 2.4.0. The complete optional
[dependency lock](evidence/qec_ai_ising_a800_20261025/requirements.lock) is retained.

Pinned NVIDIA/Ising-Decoding source is
`ea7c1bb3b823fc6a38472d0d95974b6d01932288`, with code/config digest
`ed9e6c56f263e0cd9fcc04a494e9cf6544c9f78cbb54c788fcce71b2e961b70e`.
Authorized Fast weights have SHA-256
`22db7ad25b584d7588493d6ef815381c8d1263cfb29f212391a591b285ad0a05`;
stored FP16 weights were promoted to FP32. No weights are committed.

Raw JSON records are archived byte-for-byte under
[A800-172](evidence/qec_ai_ising_a800_20261025/a800_172/manifest.json) and
[A800-171](evidence/qec_ai_ising_a800_20261025/a800_171/manifest.json).
Each directory includes choices, parity checks, results, four timing reports and
a post-run audit. The audit rechecked hashes, saved tensor shapes, 200-step
checkpoints and source inputs without new training, inference or samples. Circuit,
DEM, shots, choices, prediction arrays and native model tensor states agree
between hosts. Serialized checkpoint-file hashes need not match because their
container metadata can differ; tensor-state hashes do match.

Private raw binaries and checkpoints remain on each producing host under
`/tmp/fq-ai-ising-a800-20261024-236167e/full-172` or `full-171`.
Local JSON/log copies are under
`/Users/baai/Documents/liuwei/qec-ai-ising-a800-20261025`.

## Paired logical quality

Both hosts and all three training seeds produced the same counts:

| Path | Failures / 65536 | Wins vs global | Losses vs global | Gate |
| --- | ---: | ---: | ---: | --- |
| Global-only PyMatching | 83 | n/a | n/a | Control |
| Native raw | 83 | 0 | 0 | Inconclusive |
| Native calibrated | 83 | 0 | 0 | Inconclusive |
| Official Fast + PyMatching | 72 | 61 | 50 | Inconclusive |
| Terminal-mask-only control | 83 | 0 | 0 | Control |

Every native validation-selected threshold was 1.0, with zero changed detector
bits on test. Calibrated temperatures were 0.5/0.25/0.5 for seeds 11/22/33.
Direct native-calibrated versus Fast has 50 wins and 61 losses. The exact paired
p-value is 0.3425805825718972. All alpha=0.05/12 summaries retain an inconclusive
gate; the reported ratio upper bound is unavailable at this budget. The lower
Fast point count is not evidence that the planned 10% quality gate was met.

The native network has 81 trainable parameters and saw 25600 training shots per
seed; Fast contains 912772 stored parameters and has different pretraining.
This is a deployed-checkpoint comparison, not equal-budget architecture evidence.
The unchanged native predictions motivate label-equivalence and training-sufficiency
diagnosis before another intervention or a native CNN layout extension.

## Full software timings

Each value is whole-batch wall time from CPU binary detectors to CPU logical
predictions, including H2D, layout conversion, forward, correction construction,
D2H, validation, CPU matching and logical XOR. CUDA work is synchronized. Sampling,
training and construction are excluded. Each path ran in a separate bounded
process with only its active model/DEM on CUDA; the parent released CUDA tensors
before worker launch. Inputs use the separate timing seed 7100011, with 10
warmups and 200 measured repetitions per fixed batch. Quantiles describe repeated
execution of these fixed batches, not the input-population tail or a feedback
deadline.

| Host | Path | Batch | p50 (us) | p95 (us) | p99 (us) | Shots/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 172 | pymatching | 1 | 97.141 | 101.883 | 109.604 | 10223.3 |
| 172 | pymatching | 32 | 568.927 | 587.533 | 592.163 | 56086.0 |
| 172 | pymatching | 128 | 1912.997 | 1929.535 | 1937.023 | 66514.3 |
| 172 | flagquantum_raw | 1 | 697.438 | 717.321 | 729.277 | 1429.0 |
| 172 | flagquantum_raw | 32 | 1665.182 | 1691.135 | 1699.542 | 19493.6 |
| 172 | flagquantum_raw | 128 | 4285.472 | 4344.079 | 4379.036 | 29829.5 |
| 172 | flagquantum_calibrated | 1 | 780.642 | 802.029 | 816.651 | 1277.7 |
| 172 | flagquantum_calibrated | 32 | 1712.182 | 1772.791 | 1778.968 | 18599.5 |
| 172 | flagquantum_calibrated | 128 | 4345.395 | 4376.948 | 4385.241 | 29456.8 |
| 172 | ising | 1 | 2767.642 | 3111.207 | 3135.291 | 349.9 |
| 172 | ising | 32 | 5118.659 | 5148.465 | 5160.393 | 6249.0 |
| 172 | ising | 128 | 13387.898 | 13539.326 | 13567.299 | 9526.9 |
| 171 | pymatching | 1 | 104.203 | 109.433 | 114.565 | 9541.6 |
| 171 | pymatching | 32 | 580.667 | 597.813 | 606.286 | 55057.1 |
| 171 | pymatching | 128 | 1922.331 | 1942.000 | 1958.486 | 66235.1 |
| 171 | flagquantum_raw | 1 | 671.611 | 689.961 | 703.926 | 1484.8 |
| 171 | flagquantum_raw | 32 | 1639.639 | 1663.904 | 1673.750 | 19957.8 |
| 171 | flagquantum_raw | 128 | 4253.956 | 4547.546 | 4667.032 | 29799.9 |
| 171 | flagquantum_calibrated | 1 | 779.494 | 801.441 | 828.955 | 1278.4 |
| 171 | flagquantum_calibrated | 32 | 1698.508 | 1774.862 | 1788.994 | 18630.3 |
| 171 | flagquantum_calibrated | 128 | 4342.464 | 4373.087 | 4398.713 | 29460.4 |
| 171 | ising | 1 | 2904.016 | 3112.059 | 3126.916 | 347.4 |
| 171 | ising | 32 | 5129.858 | 5164.139 | 5179.110 | 6234.8 |
| 171 | ising | 128 | 13425.506 | 15066.112 | 15235.423 | 9420.6 |

Global-only is fastest, and the inactive native predecoder adds overhead. Native
inference being faster than this Fast implementation does not satisfy a benefit
gate: capacity/training differ and both neural implementations are unoptimized
PyTorch. No TensorRT, compilation, quantization or overlap was enabled; these
numbers must not be compared directly with published optimized deployment latency.

## Memory boundary and measurement defect

CUDA allocator peaks match across hosts:

| Path | Peak allocated bytes | Peak reserved bytes |
| --- | ---: | ---: |
| Global-only | 0 | 0 |
| Native raw / calibrated | 287430656 | 339738624 |
| Fast | 1343108096 | 2040528896 |

These include active model/DEM and inference buffers, and exclude the parent
training process. They are not total physical VRAM usage.

The raw `isolated_process_peak_rss_bytes` fields are unusable for path-specific
host-memory comparisons. All four paths inherited the parent's historical Linux
`ru_maxrss` high-water mark across exec: 5036945408 bytes on 172 and 5155254272
on 171. The original records remain unchanged. The post-run fix reads Linux
`/proc/self/status` `VmHWM` for future workers and records the counter name;
missing data fails closed. It does not recover this run's per-path RSS. No extra
experiment was appended to replace these values. RSS collection is outside the
elapsed timing region; quality, CUDA allocator peaks and timing values remain
separately reported evidence.

## Verification and next gate

Before the reporting fix, focused local tests with the actual pinned upstream
passed: 103 passed, one real-CUDA test skipped locally. The real-CUDA hyperedge
parity/timing test passed separately on both A800 hosts. After the RSS fix,
106 focused tests passed with the pinned upstream enabled; the one real-CUDA
test skipped on the local CPU host. Strict mypy, Ruff, Black, architecture boundaries, repository language,
public API migration baseline and team ownership checks passed. The bounded
local `pr-default` run timed out at 120 seconds without completing; it is not
reported as a pass. Evidence field-name and inline-origin checks passed, but the
revision-history gate cannot certify the local shallow clone; complete-history
validation remains a CI requirement. Full current-head CI remains a PR acceptance
requirement.

Do not advance optimization, export or domestic-accelerator gates. The next
scientific slice must declare a new fixed protocol before collection, diagnose
equivalent correction labels and learning sufficiency, retain global-only and
native controls, and keep the official Fast checkpoint as an external baseline.
Any common-layout native CNN extension needs independent geometry/parity tests
before its quality comparison. No seed selection or sample extension is licensed
by these inconclusive results.
