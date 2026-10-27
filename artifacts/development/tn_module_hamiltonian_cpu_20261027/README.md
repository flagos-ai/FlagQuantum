# Tensor-network Module Hamiltonian energies on CPU

These measurements record the tensor-network `fq.Module` Hamiltonian route added
for issue #938. The route was a refusal before this change: the same circuit,
loss, `backward()` and optimizer workflow ran on statevector and MPS, and
`mode="tensor_network"` raised `CapabilityError` for any Hamiltonian, including a
single `Z` term.

The payload `summary.json` was produced by `benchmarks/tn_module_hamiltonian.py`
from source commit `9d8c1221d8fe8d1819bc1aabb294aa717590adf5` (base `origin/main`
`057424b5df734da9e14ca6e3400cdd93131d6cab`) with the measuring script committed.
Environment: Python 3.12.14, PyTorch 2.14.1, macOS 27.0 arm64, CPU, one process,
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, 6.8 s total. Every
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
| tensor_network | -1.378842330891016 | 2.22045e-16 | 2.66454e-15 |

A closed-form single-qubit check keeps a shared mistake in both simulators from
passing as agreement. For `RY(0.7)` on `|0>` with `Z + 0.5 X`, the analytic value
is `cos(0.7) + 0.5 sin(0.7) = 1.0869510309033341` and the analytic derivative of
that expression is `-0.26179659359544677`. The tensor-network module returns
value error `2.22045e-16` and gradient error `5.55112e-17`.

A declared batch of four shared-parameter examples reproduces the single-example
energy to `2.22045e-16`.

## Training and checkpoint recovery

The tensor-network module trained for six Adam steps at `lr=0.05`, was
checkpointed after step 2 with its optimizer, and a second module restored the
checkpoint and continued. Losses before the checkpoint were
`3.1642800, 2.7085191`. The four steps after the checkpoint were
`2.2540171, 1.8187110, 1.4160304, 1.0551892` for the uninterrupted run and the
same four values for the resumed run: maximum loss error `0.0`, maximum
parameter error `0.0`, restored step `2`.

## Execution mode and contraction cost

The energy is one operator-network contraction regardless of the term count. For
the same circuit and observable, the tensor-network energy is still slower than
the statevector energy at every measured width: the bra-operator-ket contraction
builds and walks a network, while the statevector route multiplies one dense
state by cached sign weights. The gap grows with width, from `1.93x` to `4.16x`
over widths 2 to 8:

| Width | Terms | SV forward (s) | TN forward (s) | TN / SV | TN forward+backward (s) |
| --- | --- | --- | --- | --- | --- |
| 2 | 4 | 0.000427 | 0.000822 | 1.93 | 0.001372 |
| 4 | 8 | 0.000727 | 0.002050 | 2.82 | 0.003285 |
| 6 | 12 | 0.001056 | 0.003903 | 3.70 | 0.005675 |
| 8 | 16 | 0.001453 | 0.006043 | 4.16 | 0.008488 |

Median absolute deviations of the tensor-network forward medians were
`1.00e-05, 2.00e-05, 2.10e-05, 6.60e-05` s; the statevector ones were
`8.00e-06, 8.00e-06, 8.00e-06, 1.70e-05` s. These are single-process wall times
on one machine, not a cross-machine or accelerator comparison.

The same tensor-network state answers a `z_sum` observable through
`expectation_z`, and that route reports `observable_execution="dense_state"` with
`full_state_materialized=True` at every measured width, because all four widths
are below the 12-wire dense-observable threshold. The Hamiltonian route reports
`observable_execution="hamiltonian_mpo_contraction"` and
`full_state_materialized=False` at every width. This is why the Hamiltonian
branch does not reuse the Z readout it would otherwise be tempting to build the
energy on.

## Cost does not grow with the term count

The first revision of this artifact contracted one term at a time and recorded
the consequence honestly: at a fixed eight-wire circuit the energy cost rose from
`5.7x` to `55.2x` the statevector cost as the sum grew from 4 to 16 terms, because
each term repeated the same self-contraction of the shared state. That revision is
`a0efee36195bf48e88309909f7f1640afef92532` and its payload is in this
directory's history.

The sum is now contracted as one matrix-product operator. Holding the circuit
fixed at width 8 and varying only the term count:

| Terms | Operator nodes | Operator build (ms) | Operator contract (ms) | Module forward (ms) |
| --- | --- | --- | --- | --- |
| 1 | 117 | 0.137 | 4.667 | 5.901 |
| 2 | 117 | 0.131 | 4.855 | 6.200 |
| 4 | 117 | 0.141 | 4.778 | 6.357 |
| 16 | 117 | 0.164 | 4.820 | 6.018 |

The node count is the structural claim and it is exactly constant. The wall
clock agrees within noise: the `terms=4` forward row carries a median absolute
deviation of `3.70e-04` s, so its `6.357` should not be read as a trend.

## Scope and remaining gaps

- The sum is compiled into one operator network over the state's own ket and bra
  copies. There is no retained per-term route: a caller who wants one term at a
  time uses `HamiltonianTerm.expectation`, which is still answered by a single
  bra-operator-ket contraction.
- The operator build cost grows with the number of distinct local Pauli
  patterns, not with the term count. The measured row varies term count only, at
  a fixed circuit and a fixed operator family; a sum with more distinct patterns
  per wire was not measured here.
- No observable contraction peak is reported for this route: estimating it would
  repeat the path search that answered the call, so the payload records `null`
  rather than the state-contraction peak.
- Gradient is first-order PyTorch autograd through the contraction. No adjoint,
  second-order, GPU, or distributed derivative was measured.
- Widest measured circuit is 8 wires. No claim is made about widths where the
  dense statevector route is infeasible, which is the regime this route exists
  for. The tensor-network energy is still slower than statevector wherever a
  dense state fits.
- Capability maturity is unchanged. `tensor_network_training` stays
  `experimental`; this artifact evidences one CPU workflow, not a promotion.
