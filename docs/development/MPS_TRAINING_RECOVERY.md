# Fixed-workload MPS training and checkpoint recovery

This extends the existing MPS acceptance producer and capacity-series supervisor.
It reuses the structured circuit builder, owner-sharded training engine, compact
runtime metrics and current immutable-shard/atomic-manifest checkpoint contract.
The single-device flagship benchmark remains a separate fast path. No public API,
checkpoint schema, capability maturity or certified rank count changes.

The prospective protocol is
[the frozen training manifest](../../benchmarks/manifests/mps_training_recovery_v1.json).
It fixes 32 sites, three layers, complex128, two shared initialized angles,
product initial state, maximum bond four, cutoff zero, explicit approximate
projected-subspace gradients with discarded-weight tolerance 0.01, and eight
Adam updates at learning rate 0.025. The objective is negative mean on-site Z.
The diagnostic target is -0.95; all eight updates execute even if it is reached.
Training geometry never depends on deployment rank count.

The existing 131072-site capacity series remains an independent single-step
capacity leg. It cannot establish convergence or a training speedup. The new
bounded problem fits one device so that identical-workload strong scaling is
measurable. A useful negative efficiency result is retained.

## CPU correctness and optional hardware

With the existing numerical benchmark dependencies, run a fresh output path:

```bash
FQ_MPS_TRAINING_BACKEND=gloo OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 \
  -m benchmarks.internal.evidence.mps_training_acceptance \
  --recovery-protocol benchmarks/manifests/mps_training_recovery_v1.json \
  --output /tmp/mps-training/world-2.json
python -m benchmarks.build_mps_training_recovery_report \
  --input /tmp/mps-training --output /tmp/mps-training-report --worlds 2
python -m pytest tests/benchmark_contract/test_mps_training_recovery.py -q
```

Expect eight consecutive uninterrupted updates and a 4+4 checkpoint/resume path.
The raw per-rank receipts compare objectives, owned gradients, final parameters,
Adam moments and step counters. The checkpoint restores the same world size and
ownership; this is not checkpoint resizing. Each update reevaluates the fixed
circuit from its product initial state, so no evolving full MPS is checkpointed.
The resume test reconstructs all parameter/optimizer objects from files in a new
training call, not from an in-memory optimizer. It does not simulate host loss.

For the optional reserved A800 pair, manually dispatch Scheduled hardware with
`run_mps_training_series=true` and the other campaign inputs false. Its existing
`tools/run_mps_capacity_series.py --training-series` supervisor stages an exact
clean commit, checks transport, records hardware/topology, serializes accelerator
reservations, bounds workers, retains diagnostics and measures worlds 1/2/4/8/16.
Sixteen ranks use two nodes. The one-rank measurement uses the same distributed
engine, not the fastest local API. CPU/Gloo timings are never mixed with CUDA/NCCL
hardware efficiency or interpreted as accelerator capacity evidence.

## Numerical and cost boundaries

At every pre-update parameter point, a separate local native MPS evaluation
records the matching projected gradient, the forward value and finite differences
of the truncated forward map. Sharding error is comparison with that local
approximate MPS; the local projected-VJP/finite-difference disagreement is a
different approximation error. A nontruncating small-circuit regression checks
that this disagreement vanishes when there is sufficient bond headroom. No
truncated derivative is silently described as exact.

Full-state collection and object-gather APIs are blocked inside training. Only
compact scalar traces and owner checkpoint receipts are gathered afterward.
Rank site maps and optimizer-owned indices must partition the same frozen sites
and parameters. Cross-world trajectory comparison tests the optimization process,
while same-world recovery tests the existing checkpoint compatibility contract.

Keep all eight step samples, including startup. For cost comparison, take each
step's maximum recorded rank end-to-end time, then the median of updates 2–8.
Report measured speedup T1/Tp and efficiency T1/(p*Tp). No Amdahl parallel fraction
is inferred. Whole-call barrier-bracketed wall time, checkpoint I/O and raw
forward/reverse/optimizer/diagnostic times remain separate. Runtime synchronizes
accelerator phases; per-step samples are not independent repeated initializations.

Per-rank allocated/reserved snapshots, allocated peaks, RSS, bond/discarded-weight
and boundary/gradient/optimizer communication metrics remain in raw summaries.
Communication counters cover their named runtime payloads; they do not claim all
metadata traffic or exact wire-level bytes. CPU allocator zeros mean unavailable
accelerator accounting, not zero process memory. Failed/OOM/timeout records and
missing deployment outcomes cannot be scored as fast completions. Published
reports remain development evidence under the current audit/maturity contracts.
