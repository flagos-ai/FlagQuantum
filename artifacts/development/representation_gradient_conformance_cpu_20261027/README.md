# Representation conformance for state and gradient, on CPU

This extends the representation-agreement grid of #709 from one expectation
value to the state, several observables and the parameter gradient, for issue
#899. The payload `complex128.json` was produced by
`benchmarks/representation_agreement_sweep.py --gradients` from source commit
`b8cf16f5f66735d355674aa9e5b32d488f7693b8` (base `origin/main`
`914e662ec7be501ea45b6a5ceb6aaf4b9e53fc55`), with the producer committed.
Environment: Python 3.12.14, PyTorch 2.14.1, CPU, complex128 amplitudes,
80.7 s for 20 cells.

Reproduce:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python benchmarks/representation_agreement_sweep.py --gradients --dtype complex128 \
    --widths 4,8,12 --depths 2,4 --max-bond 4,16 \
    --structures nearest_neighbour,all_to_all --all-to-all-max-width 8 \
    --output artifacts/development/representation_gradient_conformance_cpu_20261027/complex128.json
python -m pytest tests/integration/test_representation_agreement_sweep.py -q
```

`statevector` is the exact reference throughout. `relative_l2_error` is
`‖g - g_ref‖ / ‖g_ref‖`; `cosine_similarity` is the direction agreement;
`fidelity_vs_statevector` is the normalized overlap `|<ψ_ref|ψ>|²`; and
`norm_drift` is `|‖ψ‖/‖ψ_ref‖ - 1|`, kept apart because a pure rescaling of a
correct state has fidelity 1 and should not read as a disagreement.

## Where nothing is discarded, all three paths agree to machine precision

16 of the 20 cells hold the state exactly at their cap. Across them:

| Quantity | mps | tensor_network |
| --- | --- | --- |
| max gradient relative L2 error | 2.751e-15 | 3.435e-15 |
| min gradient cosine similarity | 1.000000000000 | 1.000000000000 |
| min state fidelity | 1.000000000000000 | 1.000000000000000 |
| max norm drift | 4.88e-15 | 4.44e-16 |
| max `z:0` / `x:1` / `zz:0,1` difference | 1.68e-15 / 1.04e-14 / 3.73e-15 | 2.57e-16 / 6.66e-16 / 2.98e-16 |

The tensor-network path never truncates, and it is exact in every cell here,
truncated MPS cells included.

## Where the cap bites, the MPS gradient points away from the exact one

4 of the 20 cells have `mps_truncated: true`. In those:

| Structure | Width | Depth | Cap | Bond | Headroom | Fidelity | Norm drift | Gradient rel. error | Gradient cosine | TN rel. error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_neighbour | 8 | 4 | 4 | 4 | 8 | 0.993211 | 3.43e-03 | 0.7325 | 0.6833 | 1.37e-15 |
| nearest_neighbour | 12 | 4 | 4 | 4 | 8 | 0.984777 | 7.70e-03 | 0.8558 | 0.5176 | 3.43e-15 |
| all_to_all | 8 | 2 | 4 | 4 | 8 | 0.915980 | 6.00e-02 | 1.0282 | **-0.2332** | 1.42e-15 |
| all_to_all | 8 | 4 | 4 | 4 | 8 | 0.875059 | 9.90e-02 | 1.0832 | **-0.0589** | 1.89e-15 |

Two things follow, and the second is not a property of truncation.

**The gradient error is not a scaled version of the state error.** Fidelity
0.876 is a mostly-correct state, yet the gradient's relative L2 error is 1.08
and its cosine similarity is -0.059: the reported gradient points *against* the
exact one. An optimizer following it steps uphill. The state and the observable
are both far better behaved than the gradient at the same cell - `x:1` reads
0.669 against the exact 0.969 - so a reader who checked only fidelity would not
see this coming.

**It is not the derivative of the truncated computation either.** Central
finite differences of the *same* truncated map agree with the exact gradient to
about 1e-9 in the cells measured here, because a cap of 4 changes the energy by
roughly 1e-6 at these depths. So there are three candidate gradients - the
exact one, the derivative of the truncated map, and what autograd reports - and
in these cells the third matches neither of the first two.

Measured directly at width 8, depth 6, cap 4: autograd reports `1.060e-01`
absolute error against finite differences of its own forward pass, on a
gradient of norm `1.875`. The same circuit with the cap raised to 16 agrees to
`1.6e-09`. The full set of measured cells and the shape scan are in the
payload's cells and in `tests/integration/test_representation_agreement_sweep.py`.

This is reported here, not explained away: the local MPS backward pass does not
reproduce the derivative of the forward pass it differentiates when a bond cap
discards weight. It is tracked as its own defect rather than absorbed into this
artifact, because #899 asks for the distinction and this artifact is the
measurement that shows why the distinction matters.

## Scope and what was not measured

- **The sharded-versus-unsharded leg is not measured.** It needs two processes,
  and a multi-process `gloo` group on this host did not complete: the c10d store
  spent minutes failing to resolve its own IPv6 address and the run was killed.
  The harness for it exists (`tests/distributed/mps_reverse_runtime.py`,
  `execute_torch_distributed_mps_reverse`), so the leg belongs to a CI
  `distributed-cpu-shard` lane or an accelerator host, and this artifact says so
  instead of implying a placement result.
- One structure/width/depth/cap grid at complex128 only. complex64 was not run
  for gradients: at complex64 the untruncated MPS gradient already separates
  from the exact one by about 4e-7 through float32 accumulation, which would
  need its own bound rather than one borrowed from this run.
- `all_to_all` at width 12 is skipped and recorded: the tensor-network
  contraction of that cell exceeds the sweep's CPU budget. It is a gap in this
  grid, not a claim about that regime.
- Gradients are first-order PyTorch autograd. No adjoint, second-order, GPU or
  distributed derivative is measured.
- These are conformance measurements, not a performance or capability
  promotion. `tensor_network_training` and the MPS training capabilities are
  unchanged.
