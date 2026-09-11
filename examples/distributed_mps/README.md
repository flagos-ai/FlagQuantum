# Distributed MPS capacity example

Evolve and differentiate one open-boundary MPS whose sites are owned by different
ranks. The example tests state capacity and a complete training step, rather
than throughput from independent replicas.

## Start with the bounded profile

From an installed checkout on a host with eight suitable GPUs:

```bash
timeout --signal=TERM --kill-after=30s 10m \
  torchrun --standalone --nproc-per-node=8 \
  examples/distributed_mps/variable_bond_capacity_8gpu.py \
  --profile smoke --output /tmp/fq-mps-smoke.json
```

Inspect all-rank completion, numerical checks, gradient and optimizer updates,
truncation, memory, and boundary communication before attempting capacity runs.

[Capacity runbook](RUNBOOK.md) contains the matched baseline commands, workload
sizes, historical measurements, and the scientific VQE probe. Keep workload
and precision fixed when comparing devices. A completed run is not by itself
a release-grade scalability claim.

[Capability catalog](../../docs/generated/CAPABILITIES.md) ·
[Evidence policy](../../benchmarks/results/README.md)
