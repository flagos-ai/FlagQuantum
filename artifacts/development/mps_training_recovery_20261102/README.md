# Fixed-workload MPS multi-step training and recovery

This completes the bounded development characterization requested by issue #902.
It extends the existing MPS acceptance producer and capacity-series supervisor;
it adds no public API, checkpoint schema, precision abstraction or certified rank
count. The existing single-device flagship benchmark remains a separate fast path.

## Frozen workload and evidence

The [training protocol](../../../benchmarks/manifests/mps_training_recovery_v1.json)
fixes the circuit independently of deployment: 32 sites, three layers, two shared
initial angles 0.31/-0.17, zero product input, complex128, maximum bond four,
cutoff zero, no canonicalization, and eight Adam updates at learning rate 0.025.
The objective is negative mean on-site Z. The diagnostic target is -0.95; the
prospective stopping rule always executes all eight updates. The existing
projected-subspace gradient is explicitly approximate, with discarded-weight
budget 0.01. That budget is not a bound on derivative error.

- [CPU report](cpu/REPORT.md): worlds 1/2 on Apple M5, Torch 2.14.1; correctness
  and recovery evidence only.
- [CUDA report](cuda/REPORT.md): worlds 1/2/4/8 on one A800 host and world 16 on
  two eight-GPU hosts; Torch 2.13.0+cu130, CUDA runtime 13.0. Physical UUID/name
  inventories and NVIDIA topology output accompany every launch. The independent
  audit matches each rank's actual device UUID to its node/local-rank inventory.
- [Successful hardware run](https://github.com/flagos-ai/FlagQuantum/actions/runs/41410450971)
  and [original raw artifact](https://github.com/flagos-ai/FlagQuantum/actions/runs/41410450971/artifacts/13124823979).
  The existing sixteen-rank transport/training regression passed before the sweep.
- The [retained large capacity leg](../mps_capacity_series_20261023/README.md) is
  a separate October 23 campaign: the unchanged 131072-site, bond-768 workload
  exhausted worlds 1/2/4/8 and completed one update at world 16. Its snapshot and
  digest are bound in `validation/capacity-leg.json`. It was not rerun here and
  is not convergence evidence for that large state.

CPU measurements were produced by clean source
`9ad41ce1a24cda37e17eb6e54988ff64bedf618d`; CUDA measurements by clean source
`b4391cfe68ac518c4a37ac9ca081e948f00c4a63`. Numerical producer and protocol bytes
are identical across those revisions; the latter fixes hardware source transport.
Each campaign records its source/producer/manifest digests. Report rendering and
raw-value validation are separately pinned in `validation/report-producer.json`
to `9f3f1f2e8630bb0b32e3f7b139806f32401e0b57`.

## Training, approximation and recovery observations

All seven formal deployment points completed eight uninterrupted updates plus a
4+4 checkpoint/resume path: 112 optimizer updates in total. All recorded objective
curves move from approximately -0.86751736 to -0.97435602. These are the first and
last pre-update objectives, not a post-eighth-update reevaluation. The fixed
threshold is reached within the recorded trajectory, and the full budget still
executes. Discarded weight decreases from about 8.12e-4 to 1.82e-6; the largest
realized bond remains four. All objective, gradient, bond and discarded-weight
samples remain in the raw rank records and derived summaries.

At every pre-update parameter point, a separate native local MPS forward/VJP and
finite-difference diagnostic runs outside candidate timing. The maximum CUDA
sharded/local projected-gradient discrepancy is approximately 1.42e-14. Across
world sizes, maximum discrepancies are about 2.29e-14 for objective, 7.25e-15 for
gradient, 5.14e-16 for final parameters and 1.25e-15 for per-parameter Adam state,
all below the frozen absolute tolerance 2e-5.

The local projected VJP itself differs from finite differences of the truncated
forward map by as much as about 0.336. This is an existing approximation boundary,
not sharding error or proof of an exact truncated derivative. A nontruncating
small-circuit check independently verifies finite-difference agreement when bond
headroom removes that approximation. No universal convergence conclusion follows
from this one bounded structured objective.

For each world, parameters and owner optimizers are reconstructed from the
committed fourth-step checkpoint in a fresh training call, then continue to step
eight. Maximum absolute errors versus the uninterrupted path are zero for all
recorded objectives, owned gradients, final parameters, checkpoint parameters and
Adam moments/counters. This is numerical equality under the declared tolerance,
not a statement that checkpoint files are byte-identical. Every rank's committed
manifest, immutable-shard hash, owned indices and initial bond layout are audited.

Recovery preserves world size and ownership. Cross-world comparison separately
reorders Adam states by global parameter index and compares the full optimization
trajectory; it is not checkpoint resizing. Each update reevaluates the circuit
from its fixed product initial state. A continuously evolving full MPS is not
checkpointed, and host-loss recovery is not simulated.

Known full-state collection and object-gather APIs are guarded inside training.
Only compact scalar diagnostics and owner checkpoint receipts are collected
outside that guarded region. Site maps must partition all 32 sites, and optimizer
owners must partition both parameters; no full-state reconstruction is substituted
for sharded training.

## Measured cost and scaling limits

The CUDA steady critical-rank median is 1.074370 seconds at one rank, 0.698639
seconds at four ranks and 0.875109 seconds at sixteen ranks. Measured speedup is
therefore about 1.54x at four ranks (38.4% efficiency) and 1.23x at sixteen ranks
(7.7% efficiency). More ranks do not monotonically improve this small problem.
CPU/Gloo's two-rank control is also slightly slower than its one-rank control.
None of these observations is used to claim practical training of the large
capacity state or to promote a deployment topology.

All eight absolute step samples are retained. The statistic is the median of
updates 2–8 after taking each update's maximum recorded rank end-to-end time.
Those samples are consecutive training updates, not independent initializations.
The one-rank control uses the same distributed engine, not the fastest local API.
Whole-call barrier-bracketed wall time includes setup, checkpoint restore and I/O.
Write/commit/prune costs are separately recorded; restore/read cost is not isolated
from call setup. Checkpoint I/O follows the recorded step metrics. No Amdahl
fraction was fitted or estimated.

Raw records retain per-rank allocated/reserved snapshots, allocated peaks,
process RSS, forward/reverse/optimizer/diagnostic times and named communication
counters. Allocator peaks exclude unobserved device-context/provider allocations;
CPU allocator zeros do not establish zero process memory. Boundary, gradient and
optimizer payload counters do not cover every metadata exchange or wire-level
packet. Topology, interface/HCA configuration and NCCL logs remain observable;
they do not certify GPUDirect or absence of all host staging.

## Failures, retained material and reproduction

The first hardware attempt
[was cancelled before measurement](https://github.com/flagos-ai/FlagQuantum/actions/runs/41409076439)
after direct Git fetch repeatedly failed with low-throughput/connection timeouts.
Its workflow receipt and sanitized compressed log remain in `validation/`. The
successful run reuses the repository's CPU-producer/GPU-consumer source artifact
transport and verifies the exact commit before execution. A round-trip regression
checks bytes, dirty-tree recovery, wrong-SHA refusal and producer-bound retries.

The pre-freeze debugging archive separately retains the local sandbox launch
failure, the initially incorrect collection-guard class, the zero discarded-weight
budget refusal and a first bounded CPU check. These came from a dirty development
checkout, are explicitly not pinned scientific evidence, and do not replace the
formal campaigns. No final deployment failed or was omitted; negative timing and
approximation results remain visible.

All JSON and logs use the existing infrastructure sanitizer. Raw campaign/rank
JSON and logs are losslessly gzip compressed to keep the review readable; commit
manifests, launch/outcome receipts and derived summaries remain readable JSON.
Original and publication digests are retained. Native checkpoint binaries remain in the original CI artifact (30-day retention);
Git retains semantic parameter/optimizer values, committed manifests and shard
hash receipts, rather than adding a second checkpoint format. The pinned commands
regenerate native checkpoints through the existing compatibility contract.

In a clean checkout of the appropriate producing revision, with its recorded
Torch environment, use fresh output directories:

```bash
FQ_MPS_TRAINING_BACKEND=gloo OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
  -m benchmarks.internal.evidence.mps_training_acceptance \
  --recovery-protocol benchmarks/manifests/mps_training_recovery_v1.json \
  --output /tmp/mps-reproduce/world-2.json
```

Repeat with one process for the matched CPU baseline. The optional CUDA series is
Scheduled hardware's `run_mps_training_series=true` input on the producing branch;
select other manual campaign inputs false. It uses the existing reserved-pair
supervisor, exact-source staging, telemetry, bounded workers and cleanup.

From the publication checkout, whose report bytes are verified against the pinned
renderer above, regenerate and audit the retained measurements:

```bash
python -m benchmarks.build_mps_training_recovery_report \
  --input artifacts/development/mps_training_recovery_20261102/cpu \
  --output /tmp/mps-cpu-report --worlds 1,2
python -m benchmarks.build_mps_training_recovery_report \
  --input artifacts/development/mps_training_recovery_20261102/cuda \
  --output /tmp/mps-cuda-report
PYTHONPATH=. python artifacts/development/mps_training_recovery_20261102/validation/verify_sources.py
python -m pytest tests/benchmark_contract/test_mps_training_recovery.py -q
```

The audit checks current report bytes against the pinned renderer, resolves
numerical producer/manifest bytes from Git, verifies checkpoint
commit/owner receipts and physical UUID mapping, recomputes recovery and
cross-world/local errors, and regenerates both reports byte-for-byte.

## Issue acceptance assessment

| Acceptance condition | Evidence and result |
| --- | --- |
| Prospectively frozen feasible objective; separate capacity leg | Source-pinned manifest fixes circuit, parameters, approximation, Adam, eight steps and stopping rule; retained large capacity campaign remains distinct. |
| Consecutive updates, traces and matched local MPS | All seven deployment points complete 8+4+4 updates; every pre-update point has local MPS/VJP/FD diagnostics, with sharding and truncation-derivative errors separated. |
| Checkpoint recovery and rank ownership | Fourth-step commit restores to step eight, numerical recovery errors zero; parameter/Adam state, immutable shards, site maps and owner partitions verified without full-state collection. |
| Resources and actual strong scaling | All rank memory/phase/communication traces, topology, raw times and completion receipts retained; fixed-workload efficiency measured, negative scaling visible, no Amdahl estimate. |
| Distinct CPU, accelerator and multi-node scopes | CPU worlds 1/2 remain correctness evidence; real A800 worlds 1/2/4/8 and two-node 16 remain bounded development evidence under existing maturity/audit flags. |
| Reproducibility and failures | Reachable numerical/report revisions, producer/manifest/raw digests, native checkpoint commands, sanitized logs, failed launch records and deterministic report/audit commands retained. |

All six conditions are met within these explicit boundaries. Closing this issue
does not certify arbitrary large-state training, checkpoint resizing, exact
truncated gradients, host-loss recovery or a new release topology. Use a merge
commit so that every source pin remains reachable.
