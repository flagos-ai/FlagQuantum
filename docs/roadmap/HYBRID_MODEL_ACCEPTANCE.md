# Hybrid Model Acceptance

FlagQuantum maintains two small product workflows in
`examples/hybrid_model_acceptance.py`: `HybridQuantumClassifier` and
`VariationalEnergyModel`. They are ordinary PyTorch modules containing
`fq.Module`; runtime changes are expressed through `RuntimePolicy`, not model
rewrites.

The classifier covers batching, optimizer training, evaluation, full-model
checkpoint/resume, deployment parameter binding, single GPU, and native
statevector sharding. The energy model covers differentiable Hamiltonian
evaluation and native-PyTorch versus real JAX-kernel value/gradient parity.
JAX is selected explicitly and remains a quantum kernel; PyTorch owns model,
optimizer, checkpoint, and distributed lifecycle.

Acceptance reports contain separate `correctness`, `accuracy`, `performance`,
and `deployment` sections. Accuracy is never interpreted as a performance or
scalability claim, and these CI/development workflows keep
`scalability_claim_allowed=false`.

```bash
python examples/hybrid_model_acceptance.py --model classifier --steps 20
python examples/hybrid_model_acceptance.py --model energy --backend jax --steps 5
```

CI uses deterministic two-qubit variants. Scheduled GPU and distributed jobs
run the same model classes on accelerator and sharded policies; larger future
production workloads may change sizes and steps but not the product API.
