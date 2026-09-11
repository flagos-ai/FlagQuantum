# FlagOS statevector F3 training profile

F3 is a fail-closed development profile for the existing owner-sharded
statevector training runtime. It does not add Torch-FL or FlagCX as a
FlagQuantum dependency and does not infer the provider's internal transport.

Run the controller inside an environment where importing `torch_fl` registers
`torch.flagos`:

```bash
python tools/validate_flagos_statevector_training.py \
  --output artifacts/flagos_statevector_training_f3.json \
  --world-sizes 2,4,8 \
  --steps 3
```

The controller starts a fresh torchrun job for each world size. Every job must
run on one complete node through `backend="flagos"` and `device="flagos"`.
For both complex64 and complex128 it measures:

- one sharded reverse pass against a bounded CPU complex128 value/gradient
  reference;
- a three-step owner-sharded SGD trajectory;
- a three-step owner-sharded Adam trajectory;
- strict rank-local amplitude ownership, positive backward communication,
  parameter-gradient reduction, owner update followed by broadcast, and
  cross-rank parameter consistency.

The profile rejects missing world sizes, duplicate device placement, full-state
materialization, full-state backward replay, incorrect ownership semantics,
non-positive communication, and any value, gradient, parameter or rank
consistency error outside its dtype threshold. A failed worker may still write
its complete measurements; the controller keeps the overall result failed.

An accepted F3 profile remains development evidence only. It does not authorize
FlagCX identity, host-staging, performance, convergence, scalability,
production-support or release claims.
