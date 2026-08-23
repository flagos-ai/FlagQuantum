# Persistent distributed TN plan cache

## Implementation

FlagQuantum now supports an explicit `plan_cache_path` for distributed sparse
TN execution.  Rank zero stores a JSON record containing:

- a SHA-256 topology and policy fingerprint;
- the bounded-memory slicing plan;
- the complete representative pair-contraction DAG.

The fingerprint covers node names, labels, shapes, element sizes, output
labels, explicit slice labels, and intermediate-memory policy.  A topology or
policy change produces a cache miss.  Cache writes use a temporary file and an
atomic rename.  Other ranks never read or write the file; rank zero validates
and broadcasts the restored plan.

The format is JSON rather than pickle, so it is inspectable and does not execute
code while loading.

## Eight-A800 cold-start result

The frozen workload remains the 36-qubit 4x9 grid, four cycles, one amplitude,
16 GiB working-set policy, 4x safety factor, and 128 slices.

| Metric | Cold miss | Cold hit | Improvement |
|---|---:|---:|---:|
| Path/slice phase | 16.837 s | 1.507 s | 11.17x |
| Steady execution | 4.682 s | 4.696 s | unchanged |
| Cold end-to-end | 24.272 s | 7.279 s | 3.34x |
| Peak/rank | 16.0 GiB | 16.0 GiB | unchanged |

The cache file is approximately 544 KiB.  A hit occurs in a new `torchrun`
process, demonstrating persistence rather than an in-process dictionary hit.
Amplitudes are identical and neither run emits an allocator OOM warning.

## Production boundary

The persistent cache closes the dominant repeated-planning latency for recurring
workloads.  A first-seen topology still pays path search, and cache-hit cold
latency is now mostly TN construction, small-object broadcast/process
synchronization, and the 4.7-second contraction.

The cache is opt-in through an explicit path.  Automatic cache lifecycle,
concurrent multi-job locking, eviction, and cross-version migration remain
deployment concerns and are not silently imposed on user storage.
