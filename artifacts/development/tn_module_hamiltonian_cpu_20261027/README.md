# Tensor-network Module Hamiltonian energies on CPU

These measurements record the tensor-network `fq.Module` Hamiltonian route added
for issue #938. The route was a refusal before this change: the same circuit,
loss, `backward()` and optimizer workflow ran on statevector and MPS, and
`mode="tensor_network"` raised `CapabilityError` for any Hamiltonian, including a
single `Z` term.

`run-1.json`, `run-2.json` and `run-3.json` are three fresh processes of
`benchmarks/tn_module_hamiltonian.py` from source commit
`a915e19ee63b293ce0d551134eff4dc7188113fa` (base `origin/main`
`057424b5df734da9e14ca6e3400cdd93131d6cab`), each with the measuring script
committed. Environment: Python 3.12.14, PyTorch 2.14.1, macOS 27.0 arm64, CPU,
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`, 31.65–32.62 s per process.
Cost rows below are the median of the three processes; the per-process values are
beside them so a reader can see the spread rather than take one sample. Every
number is read from those payloads, which also retain the raw per-repeat samples,
medians and median absolute deviations.

Reproduce:

```bash
for run in 1 2 3; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python benchmarks/tn_module_hamiltonian.py \
    --output artifacts/development/tn_module_hamiltonian_cpu_20261027/run-$run.json
done
python -m pytest tests/unit/test_module_tensor_network_hamiltonian.py -q
```

## Agreement with the exact route

The workload is a three-wire ansatz whose parameters are shared and used
nonlinearly, with the energy `-0.5 Z_q - 0.7 X_q` on every qubit and a squared
outer loss against 0.4. `statevector` is the reference for the many-body case.
All three processes measured the same values.

| Route | Energy | Energy error vs SV | Max parameter-gradient error vs SV |
| --- | --- | --- | --- |
| statevector | -1.3788423308910158 | — | — |
| mps | -1.378842330891016 | 2.22045e-16 | 1.66533e-16 |
| tensor_network | -1.378842330891016 | 2.22045e-16 | 2.22045e-15 |

A closed-form single-qubit check keeps a shared mistake in both simulators from
passing as agreement. For `RY(0.7)` on `|0>` with `Z + 0.5 X`, the analytic value
is `cos(0.7) + 0.5 sin(0.7) = 1.0869510309033341` and the analytic derivative of
that expression is `-0.26179659359544677`. The tensor-network module returns
value error `2.22045e-16` and gradient error `5.55112e-17`.

A declared batch of four shared-parameter examples reproduces the single-example
energy to `2.22045e-16` in all three processes.

## Training and checkpoint recovery

The tensor-network module trained for six Adam steps at `lr=0.05`, was
checkpointed after step 2 with its optimizer, and a second module restored the
checkpoint and continued. Losses before the checkpoint were
`3.1642800, 2.7085191`. The four steps after the checkpoint were
`2.2540171, 1.8187110, 1.4160304, 1.0551892` for the uninterrupted run and the
same four values for the resumed run: maximum loss error `0.0`, maximum
parameter error `0.0`, restored step `2`.

## Execution mode

The same tensor-network state answers a `z_sum` observable through
`expectation_z`, and that route reports `observable_execution="dense_state"` with
`full_state_materialized=True` at every measured width, because all measured
widths are below the 12-qubit dense-observable threshold. The Hamiltonian route
reports `observable_execution="hamiltonian_mpo_contraction"` and
`full_state_materialized=False` at every width. This is why the Hamiltonian
branch does not reuse the Z readout it would otherwise be tempting to build the
energy on.

## Cost against statevector

The energy is one operator-network contraction regardless of the term count, but
it is still slower than the statevector energy at every measured width: the
bra-operator-ket contraction builds and walks a network, while the statevector
route multiplies one dense state by cached sign weights.

| Width | Terms | SV forward (ms) | TN forward (ms) | TN / SV | TN forward+backward (ms) | Per-process TN / SV |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 | 0.435 | 0.850 | 1.95 | 1.403 | 1.95, 2.00, 1.94 |
| 4 | 8 | 0.740 | 2.177 | 2.95 | 3.320 | 2.98, 2.86, 2.95 |
| 6 | 12 | 1.094 | 3.832 | 3.49 | 5.682 | 3.46, 3.49, 3.89 |
| 8 | 16 | 1.446 | 6.051 | 4.17 | 8.533 | 4.18, 4.17, 3.97 |

The per-process column is the aggregate ratio of each process, so the spread is
visible: the widest is width 6, where one process measured `3.89` against `3.46`.
No crossover was measured. These are single-process wall times on one machine,
not a cross-machine or accelerator comparison.

## The contraction peak is linear in width, not exponential

The reason to select a contracted representation is capacity, so the peak of the
operator contraction is the number that decides. A dense state holds `2**width`
amplitudes. Holding depth and circuit family fixed and varying width:

| Width | Terms | Dense state elements | Operator contraction peak | Peak / dense |
| --- | --- | --- | --- | --- |
| 8 | 16 | 256 | 1152 | 4.5 |
| 10 | 20 | 1024 | 1664 | 1.625 |
| 12 | 24 | 4096 | 2176 | 0.5312 |
| 14 | 28 | 16384 | 2688 | 0.1641 |

The peak grows by `256` elements per qubit over this range, that is linearly,
while the dense state doubles. The two cross between width 10 and width 12. All
three processes recorded identical peaks, because the peak is a property of the
network and its ordering rather than of the timing.

This is the honest scope of the result: below the crossover the contracted route
is both slower and larger than a dense state, and it is justified there only by
representation consistency. Above it, the peak is the reason the route exists.
The measured range stops at width 14; no claim is made beyond it.

## Cost against term count

The sum is compiled into one matrix-product operator, so the operator network's
node count does not depend on how many terms it carries. Holding the circuit
fixed at width 8:

| Terms | Operator nodes | Operator build (ms) | Operator contract (ms) | Module forward (ms) | Operator peak |
| --- | --- | --- | --- | --- | --- |
| 1 | 117 | 0.141 | 4.588 | 5.913 | 64 |
| 2 | 117 | 0.133 | 4.807 | 5.960 | 128 |
| 4 | 117 | 0.138 | 4.689 | 5.985 | 512 |
| 16 | 117 | 0.159 | 4.714 | 6.049 | 1152 |

The node count and the contraction time are flat; **the peak is not**, and that
qualification matters. It grows from `64` to `1152` elements as the sum grows from
1 to 16 terms, because the operator's own core is wider. A longer sum therefore
costs no more time but does cost more of the byte budget the capacity argument
relies on.

## Scope and remaining gaps

- The sum is compiled into one operator network over the state's own ket and bra
  copies. There is no retained per-term route on the module path: a caller who
  wants one term at a time uses `HamiltonianTerm.expectation`, which is still
  answered by a single bra-operator-ket contraction.
- The operator build cost grows with the number of distinct local Pauli
  patterns, not with the term count. The measured row varies term count at a
  fixed circuit and a fixed operator family.
- Under a declared byte budget the contraction is sliced, and no peak is reported:
  the unsliced profile would describe a network that did not run. The reported
  peak is the peak of the ordering that answered the call, and it is not a process
  resident-memory measurement.
- Gradient is first-order PyTorch autograd through the contraction. No adjoint,
  second-order, GPU, or distributed derivative was measured.
- No claim is made about widths beyond 14, about accelerator execution, or about
  distributed contraction. The tensor-network energy is still slower than
  statevector wherever a dense state fits.
- Capability maturity is unchanged. `tensor_network_training` stays
  `experimental`; this artifact evidences one CPU workflow, not a promotion.
