# Statevector runtime

Coordinate exact statevector execution and differentiation from one device to
amplitude shards distributed across ranks. Capacity expansion must come from
partitioning one state, with bounded local storage and communication buffers.

This package owns layout, transport, replay, checkpoints, gradient reduction,
and optimizer lifecycle. Gate and adjoint mathematics belong in Simulation;
public use stays centered on `fq.Circuit`, `fq.plan`, and `fq.run`.

## Change the owning stage

| Change | Entry point |
| --- | --- |
| Shard and communication plans | [planning.py](planning.py) |
| Forward execution | [forward_executor.py](forward_executor.py), [forward_sweep.py](forward_sweep.py), [forward.py](forward.py) |
| Qubit placement | [layout.py](layout.py) |
| Adjoint replay | [reverse.py](reverse.py), [reverse_adjoint.py](reverse_adjoint.py), [reverse_adjoint_sweep.py](reverse_adjoint_sweep.py), [reverse_adjoint_kernels.py](reverse_adjoint_kernels.py) |
| Checkpoints | [checkpointing.py](checkpointing.py) |
| Gradient reduction | [gradient_reduction.py](gradient_reduction.py) |
| Owner-sharded optimizer and recovery | [training.py](training.py) |

## Verify a change

From the repository root:

```bash
python -m pytest tests/integration/test_cpu_vertical_slice.py \
  tests/unit/test_statevector_forward.py \
  tests/unit/test_statevector_reverse.py \
  tests/unit/test_statevector_training.py -q
```

Cross-rank changes also need the relevant `torchrun` and hardware tier. Check
state and gradient parity, per-rank memory, actual communication, and restart
equivalence; a local reference run alone does not establish distributed support.

[Protocol and specialized-path details](IMPLEMENTATION.md) explain precision
experiments, noisy execution, and the boundary between transport and numerics.

## Cached observable diagonals

Runtime owns the observable weight cache and its limits, keys, LRU and rollback.
The numerical scalar uses `simulation.statevector.adjoint` directly, which
retains the existing native provider and exact Tensor route. The old imported
Runtime `fused_observable_expectation` binding is removed.

## Gate composition boundary

The forward sweep groups adjacent gates, respects layout and communication
boundaries, and counts admitted operations. Matrix multiplication for same-wire
fusion, native rotation tiles, rotations following RZZ segments, and cross-shard
fusion uses the existing
[`simulation.statevector.operations`](../../../simulation/statevector/operations.py)
owner directly. Runtime keeps no second composition loop or imported function
alias. Replacing that Simulation implementation therefore reaches its consumers
without changing the executor. Grouping, scratch ownership and native admission
remain here; no public API or numerical precision policy changes.

For a short replacement check, run:

```bash
python -m pytest tests/unit/test_runtime_matrix_composition.py -q
```

This checks full amplitudes through the existing zero-initialized, one-process
internal executor with Simulation composition replaced. The internal executor
does not load a Circuit's external initial amplitudes. Local conformance is not
evidence of distributed capacity or accelerator performance.
