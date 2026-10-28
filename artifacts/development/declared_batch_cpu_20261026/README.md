# Public declared-batch CPU development comparison

Producer: `5ea7e5de4bbacd4dad507e6d7ba82cf7f6c40703`, clean tracked tree; protocol:
[PROTOCOL.md](PROTOCOL.md). Three fresh processes per shape with rotated route
order; eighteen processes, 162 route checks. All initial losses, parameter
gradients and 103-update parameter trajectories passed. Exact complex128/float64
on one CPU device; no approximation, sharding or scalability claim.

The best explicitly selected FlagQuantum JAX Module candidate beats the best
measured official TensorCircuit Torch-interface candidate in all six aggregate
cases (1.16–2.06x). This is an oracle selection among recorded candidates, not
automatic dispatch or an exhaustive ranking. Default FlagQuantum Torch loses
all six cases. TensorCircuit whole-JAX training is faster than the best explicit
FlagQuantum candidate in all six cases (1.13–3.98x). The ordinary default and
whole-training performance objectives remain unmet.

All times below are microseconds/update: median of each process's five-block
median, including quantum execution, classical MSE, backward and SGD update.
First-call setup/compilation is excluded from this hot table and retained below
and in results.json.

| Wires | Batch | FQ default | FQ best explicit JAX | TC best official Torch interface | TC best whole JAX |
|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 433.5 | 66.5 | 137.2 | 16.7 |
| 2 | 32 | 484.0 | 125.1 | 234.3 | 57.2 |
| 8 | 1 | 1402.9 | 129.8 | 257.2 | 72.3 |
| 8 | 32 | 2906.6 | 512.9 | 639.1 | 353.4 |
| 12 | 1 | 2181.8 | 193.3 | 391.5 | 170.9 |
| 12 | 32 | 14505.8 | 1268.9 | 1469.7 | 932.2 |

FlagQuantum's selected mode is statevector for widths 2 and 8, tensor network
for width 12. TensorCircuit's strongest recorded hybrid candidate uses the
dense-state diagonal reduction with NumPy conversion except width 12/batch 32,
where the exact Z-sum MPO with NumPy conversion wins. Both conversion choices
and both observable algorithms remain in raw output. Stronger implementations,
other contractors, lightcone routes, MPS, GPUs, wider/deeper circuits, varying
data, input gradients and matched-quality stopping are not measured here.
Whole-JAX candidates compile loss, derivative and SGD together; they are a
different interface category and expose a real end-to-end performance gap.

| Wires | Batch | FQ explicit setup + first F/B (s) | TC hybrid setup + first F/B (s) | TC whole JAX setup + first F/B (s) |
|---:|---:|---:|---:|---:|
| 2 | 1 | 0.131 | 0.217 | 0.149 |
| 2 | 32 | 0.315 | 0.249 | 0.144 |
| 8 | 1 | 0.498 | 1.222 | 1.009 |
| 8 | 32 | 0.747 | 1.004 | 0.716 |
| 12 | 1 | 2.107 | 3.035 | 2.656 |
| 12 | 32 | 1.836 | 2.264 | 1.493 |

These first-call costs exclude Python imports, are not time-to-quality and
include separate value/VJP compilation for vector bridges. Every candidate
advances identical fixed features through 103 steps; final parameters are
retained. Process peak RSS is cumulative through earlier candidates, so it
cannot establish per-route simulator memory advantages. Three timing processes
are development evidence without a statistical release certification.

## Implementation and verification

Explicit Circuit.bsz JAX batches share parameters and compute cotangent-weighted
VJPs, not full Jacobians. Numerical changes include single-wire fusion, exact CX
permutations, dense-output contraction with an intermediate-size bound, true
greedy contraction and one exact Z-sum MPO. Selected representation is preserved.
Module snapshots account for batched angle storage and preserve consumed values.

422 focused numerical, Module, public-contract, shape-refusal, checkpoint
and vocabulary/census tests passed after the naming correction. Six changed production files and the benchmark pass strict mypy.
The full strict package check reports four errors in three untouched files; the
full smoke/unit suite timed out at 600 seconds around 82%, having already shown
six evidence-provenance failures and two census/qubit-vocabulary failures
before the naming correction. It is
not a passing full-suite run. Required CI remains outstanding; PR #914 stays draft.
The census/vocabulary failures were introduced by the moved internal function
using n_wires; a follow-up renames that argument to n_qubits without changing
its numerical operations. The artifact remains pinned to the measured producer,
not relabeled as a measurement of the follow-up. No contract ledger was changed.
Architecture, dependency, public API, lint, formatting, privacy and hygiene
checks are recorded for this update; no frozen contract or validator was relaxed.

`results.json` preserves all numerical outputs. `publication-hashes.json` records
original and published digests. Original raw JSON and test logs remain outside
the source repository. An earlier interrupted campaign overlapped the smoke/unit
suite and was discarded from performance conclusions; its raw file and reason
remain archived separately.

Remaining architecture work: #928 reusable IR-derived execution, #930 compatible
ordinary-route accessibility, #936 reusable complete training-step compilation.
Explicit batches alone do not close parent #918.
