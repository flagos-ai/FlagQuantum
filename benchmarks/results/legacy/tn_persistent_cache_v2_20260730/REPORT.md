# Locked, versioned, single-broadcast TN plan cache

This stage hardens the persistent distributed TN cache for shared production
storage and removes one redundant collective.

## Changes

- The slicing plan and complete pair-step DAG are now broadcast together once.
- Readers take a shared `flock`; writers take an exclusive `flock`.
- Each writer uses a cache-key-specific temporary file and atomic rename.
- Truncated or malformed JSON is treated as a cache miss.
- Cache records include the FlagQuantum and exact PyTorch versions.
- Missing or incompatible version metadata causes a safe cache miss.

The cache remains opt-in and JSON-based.  No pickle payload is executed.

## Eight-A800 result

The workload and 16 GiB working-set policy are unchanged.

| Metric | Cold miss | Cold hit | Improvement |
|---|---:|---:|---:|
| Path/slice phase | 11.834 s | 1.466 s | 8.07x |
| Steady execution | 4.622 s | 4.631 s | unchanged |
| Cold end-to-end | 17.518 s | 7.170 s | 2.44x |
| Peak/rank | 16.0 GiB | 16.0 GiB | unchanged |

Compared with the previous two-broadcast cold miss (24.272 seconds), the
single-broadcast implementation reduces first-run end-to-end latency by about
28%.  Cache-hit latency improves modestly from 7.279 to 7.170 seconds because
the actual contraction remains the dominant 4.63-second component.

All amplitudes agree, no allocator OOM warning occurs, and no rank independently
searches for a path.

## Remaining boundary

The next meaningful cold-hit improvement requires reducing the approximately
one-second tensor-network construction and roughly 1.5 seconds of process/NCCL
initialization and preflight synchronization.  Further path-cache tuning cannot
remove the 4.63-second contraction itself.
