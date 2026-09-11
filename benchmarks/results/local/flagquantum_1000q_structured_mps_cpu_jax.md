# FlagQuantum 1000q Structured MPS Result

Date: 2026-07-08

Command:

```powershell
python examples/single_machine_quantum_ai/05_mps_1000q_dimer_training.py --steps 1000 --n-qubits 1000 --jax-cache-dir .fq_jax_cache
```

## Result

| Metric | Value |
|---|---:|
| Qubits | 1000 |
| Dimer pairs | 500 |
| MPS bond | 2 |
| Trainable parameters | 1500 |
| Training backend | PyTorch-facing JAX kernel |
| Distribution semantics | single_device_fast_path |
| Scalability claim allowed | false |
| Initial loss | 0.262918 |
| Final loss | 6.00411e-11 |
| First loss+grad seconds | 0.228844 |
| Steady loss+grad seconds | 0.00060505 |
| Steady grad norm | 2.13537e-09 |
| JAX cache enabled | true |

Training trace:

| Step | Loss |
|---:|---:|
| 200 | 1.210092932524276e-06 |
| 400 | 6.193686119537745e-10 |
| 600 | 1.7134449414868413e-10 |
| 800 | 7.465328355493739e-11 |
| 1000 | 6.007960101639398e-11 |

## Interpretation

This is a structured 1000-qubit low-bond MPS quantum AI training result. The
task uses dimerized local quantum units and a teacher-student local observable
loss, so the workload is meaningful for MPS and naturally unfavorable to dense
statevector simulators.

Recommended claim:

> FlagQuantum trains a structured 1000-qubit low-bond MPS quantum AI model
> through a PyTorch-facing JAX kernel, reaching near-zero teacher-student loss
> with sub-millisecond steady-state loss+gradient evaluation on CPU.

Do not claim:

> FlagQuantum trains arbitrary 1000-qubit quantum circuits.

Do not claim:

> This is distributed capacity scaling.

This result is explicitly `single_device_fast_path`; it is not evidence that a
single 1000-qubit workload was sharded across ranks.
