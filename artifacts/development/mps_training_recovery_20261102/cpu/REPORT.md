# Fixed-workload MPS training and recovery

One frozen 32-site, 3-layer problem; 8 Adam updates. The 131072-site capacity leg is separate. These are development measurements, with no new certified rank count or capability promotion.

Costs use the maximum recorded rank time for each uninterrupted update; all 8 raw samples are retained, and the steady statistic is the median of updates 2–8. Whole-call barrier-bracketed wall time and checkpoint IO are separate in raw rank summaries. The one-rank control uses the same distributed engine, not the fastest local implementation. Local MPS diagnostics are outside these timers.

Efficiency is measured T1/(p*Tp) on the identical workload and backend, not an Amdahl estimate. No parallel fraction was measured or fitted. CPU/Gloo correctness is separate from CUDA/NCCL hardware evidence.

| Ranks | Nodes | Initial / last pre-update objective | Median s | Speedup | Efficiency | Local projected-VJP error | Local VJP vs FD | Recovery |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | -0.86751736 / -0.97435602 | 0.095611 | 1.0000 | 1.0000 | 1.779e-14 | 0.3357 | True |
| 2 | 1 | -0.86751736 / -0.97435602 | 0.097274 | 0.9829 | 0.4915 | 1.774e-14 | 0.3357 | True |

The gradient is the existing projected stop-subspace approximation. Its difference from finite differences of the truncated forward map is recorded separately from sharding error against the same local approximate MPS. Losses in the table are pre-update, not a post-final-update reevaluation.

Each rank retains forward/reverse/optimizer/diagnostic timings, allocated/reserved snapshots, allocated peaks, bond/discarded-weight traces and boundary/gradient/optimizer communication counters. Those counters are scoped runtime payload counts, not a packet-level total including every metadata exchange. CPU allocator zeros do not establish zero memory use; worker RSS is separate. Topology and physical inventories accompany accelerator launches.

Recovery reconstructs parameters and the owner optimizer from committed rank shards at the same world size. The circuit is reevaluated from its fixed product state on every update; a continuously evolving MPS is not checkpointed. No checkpoint rank resize, arbitrary entanglement training, convergence theorem or full-state gather is claimed. The full-state gather and object-gather APIs are guarded inside training; only compact scalar diagnostics are collected afterward.
