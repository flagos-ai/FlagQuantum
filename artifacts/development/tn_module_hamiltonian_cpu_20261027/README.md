# Tensor-network Module Hamiltonian energies on CPU

These measurements record the tensor-network `fq.Module` Hamiltonian route added
for issue #938. The route was a refusal before this change: the same circuit,
loss, `backward()` and optimizer workflow ran on statevector and MPS, and
`mode="tensor_network"` raised `CapabilityError` for any Hamiltonian, including a
single `Z` term.

The payload `summary.json` was produced by `benchmarks/tn_module_hamiltonian.py`
from source commit `a0efee36195bf48e88309909f7f1640afef92532` (base `origin/main`
`057424b5df734da9e14ca6e3400cdd93131d6cab`) with the measuring script committed.
Environment: Python 3.12.14, PyTorch 2.14.1, macOS 27.0 arm64, CPU, one process,
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, 6.6 s total. Every
number below is read from that payload; the raw per-repeat samples, medians and
median absolute deviations are in it.

Reproduce:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/tn_module_hamiltonian.py \
  --output artifacts/development/tn_module_hamiltonian_cpu_20261027/summary.json
python -m pytest tests/unit/test_module_tensor_network_hamiltonian.py -q
```

## Agreement with the exact route

The workload is a three-wire ansatz whose parameters are shared and used
nonlinearly, with the energy `-0.5 Z_w - 0.7 X_w` on every wire and a squared
outer loss against 0.4. `statevector` is the reference for the many-body case.

| Route | Energy | Energy error vs SV | Max parameter-gradient error vs SV |
| --- | --- | --- | --- |
| statevector | -1.3788423308910158 | — | — |
| mps | -1.378842330891016 | 2.22045e-16 | 1.66533e-16 |
| tensor_network | -1.3788423308910167 | 8.88178e-16 | 2.66454e-15 |

A closed-form single-qubit check keeps a shared mistake in both simulators from
passing as agreement. For `RY(0.7)` on `|0>` with `Z + 0.5 X`, the analytic value
is `cos(0.7) + 0.5 sin(0.7) = 1.0869510309033341` and the analytic derivative of
that expression is `-0.26179659359544677`. The tensor-network module returns
value error `2.22045e-16` and gradient error `5.55112e-17`.

A declared batch of four shared-parameter examples reproduces the single-example
energy exactly (maximum absolute error `0.0`).

## Training and checkpoint recovery

The tensor-network module trained for six Adam steps at `lr=0.05`, was
checkpointed after step 2 with its optimizer, and a second module restored the
checkpoint and continued. Losses before the checkpoint were
`3.1642800, 2.7085191`. The four steps after the checkpoint were
`2.2540171, 1.8187110, 1.4160304, 1.0551892` for the uninterrupted run and the
same four values for the resumed run: maximum loss error `0.0`, maximum
parameter error `0.0`, restored step `2`.

## Execution mode and contraction cost

Each Hamiltonian term is one bra-operator-ket contraction. For the same circuit
and term count, the tensor-network energy is slower than the statevector energy
at every measured width, because statevector holds one dense state while the
contraction rebuilds a network per term:

| Width | Terms | SV forward (s) | TN forward (s) | TN / SV | TN forward+backward (s) |
| --- | --- | --- | --- | --- | --- |
| 2 | 4 | 0.000439 | 0.002515 | 5.73 | 0.004130 |
| 4 | 8 | 0.000737 | 0.013474 | 18.28 | 0.019712 |
| 6 | 12 | 0.001115 | 0.037405 | 33.55 | 0.052993 |
| 8 | 16 | 0.001457 | 0.080484 | 55.24 | 0.109026 |

Median absolute deviations of the tensor-network forward medians were
`2.4e-05, 1.46e-04, 2.5e-04, 3.75e-04` s; the statevector ones were
`1.6e-05, 4.0e-06, 3.2e-05, 3.0e-05` s. These are single-process wall times on
one machine, not a cross-machine or accelerator comparison.

The same tensor-network state answers a `z_sum` observable through
`expectation_z`, and that route reports `observable_execution="dense_state"` with
`full_state_materialized=True` at every measured width, because all four widths
are below the 12-wire dense-observable threshold. The Hamiltonian route reports
`observable_execution="direct_expectation_contraction"` and
`full_state_materialized=False` at every width, and each module records one
contraction per term through `metrics["observable_count"]`. This is why the
Hamiltonian branch does not reuse the Z readout it would otherwise be tempting to
build the energy on.

## Scope and remaining gaps

- Gradient is first-order PyTorch autograd through the contraction. No adjoint,
  second-order, GPU, or distributed derivative was measured.
- The route contracts once per term. Terms are not fused, and a batched
  observable contraction that shares one network across terms was not
  implemented; the measured TN/SV ratio above is what that choice costs at these
  widths. The banded Z/ZZ fast path that MPS uses has no tensor-network
  counterpart yet.
- No observable contraction peak is reported for this route: estimating it would
  repeat the path search that answered the call, so the payload records `null`
  rather than the state-contraction peak.
- Widest measured circuit is 8 wires. No claim is made about widths where the
  dense statevector route is infeasible, which is the regime this route exists
  for.
- Capability maturity is unchanged. `tensor_network_training` stays
  `experimental`; this artifact evidences one CPU workflow, not a promotion.
