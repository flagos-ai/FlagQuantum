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
| Forward execution | [forward_executor.py](forward_executor.py), [forward.py](forward.py) |
| Wire placement | [layout.py](layout.py) |
| Adjoint replay | [reverse.py](reverse.py), [reverse_adjoint.py](reverse_adjoint.py) |
| Checkpoints | [checkpointing.py](checkpointing.py) |
| Gradient reduction | [gradient_reduction.py](gradient_reduction.py) |
| Owner-sharded optimizer and recovery | [training.py](training.py) |

## Verify a change

From the repository root:

```bash
python -m pytest tests/integration/test_cpu_vertical_slice.py \
  tests/unit/test_issue041_statevector_forward.py \
  tests/unit/test_issue042_statevector_reverse.py \
  tests/unit/test_issue043_statevector_training.py -q
```

Cross-rank changes also need the relevant `torchrun` and hardware tier. Check
state and gradient parity, per-rank memory, actual communication, and restart
equivalence; a local reference run alone does not establish distributed support.

[Protocol and specialized-path details](IMPLEMENTATION.md) explain precision
experiments, noisy execution, and the boundary between transport and numerics.
