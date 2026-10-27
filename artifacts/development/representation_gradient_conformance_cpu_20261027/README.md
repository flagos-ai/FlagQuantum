# Representation conformance for state and gradient, on CPU

This extends the representation-agreement grid of #709 from one expectation
value to the state, several observables and the parameter gradient, for issue
#899. The payload `complex128.json` was produced by
`benchmarks/representation_agreement_sweep.py --gradients` from source commit
`27995d5b98973c4bea9290041792077c6b122a9b` (base `origin/main`
`914e662ec7be501ea45b6a5ceb6aaf4b9e53fc55`), with the producer committed.
Environment: Python 3.12.14, PyTorch 2.14.1, CPU, complex128 amplitudes,
97.9 s for 20 cells.

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

## Where the cap bites, the MPS gradient separates from the exact one

4 of the 20 cells have `mps_truncated: true`. Each records the distance from the
exact gradient, the distance from the derivative of **its own** forward map
(central finite differences of the same truncated computation), and whether that
map is locally smooth at all:

| Structure | Width | Depth | Cap | Fidelity | Gradient rel. error | Cosine | Own-map rel. error | Smooth | Largest jump |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_neighbour | 8 | 4 | 4 | 0.993211 | 0.7325 | 0.6833 | 9.312e-02 | true | 1.782e-03 |
| nearest_neighbour | 12 | 4 | 4 | 0.984777 | 0.8558 | 0.5176 | 5.695e-01 | true | 1.776e-03 |
| all_to_all | 8 | 2 | 4 | 0.915980 | 1.0282 | **-0.2332** | 2.233e+05 | **false** | 5.073e-01 |
| all_to_all | 8 | 4 | 4 | 0.875059 | 1.0832 | **-0.0589** | 2.216e+04 | **false** | 2.623e+00 |

For reference, the 16 cells whose cap does not bite record an own-map relative
error of at most `3.873e-09` and a largest jump of `1.807e-03`.

The two structures are not the same phenomenon, and reporting one number for
both would hide that.

**On the chain the map is smooth and the backward pass is wrong.** The energy
moves by `1.8e-03` at most between scan points `0.01` apart, the same order as
an untruncated cell, so the truncated energy is differentiable there. Autograd
reproduces the derivative of that map only to `9.3%` and `57%` relative error at
widths 8 and 12, while the same check holds to `3.9e-09` when the cap does not
bite. There is no non-differentiability to appeal to: the reported gradient is
not the derivative of the computation that ran.

**On the all-to-all structure there is no derivative to reproduce.** The energy
jumps by `0.507` and `2.623` between scan points `0.01` apart: the SVD keeps a
different subspace either side of a point, so the truncated map is
discontinuous. A finite-difference comparison is meaningless there, which is why
its own-map relative error is order `1e4`–`1e5`, and the negative cosine
similarity is a property of the landscape rather than proof of a backward-pass
bug. What it does show is operational: at fidelity `0.876` a gradient exists in
the result object, reads as finite, and points against descent, with nothing in
the interface saying the surface has no derivative.

So the two conclusions are kept apart. The smooth-cell disagreement is a defect
in the MPS backward pass and is tracked as #941. The jumping-cell result is a
different, weaker statement: no representation-independent gradient is defined
there, and reporting one without a smoothness flag misleads.

### A correction

The first version of this artifact claimed that central finite differences of the
truncated map "agree with the exact gradient to about 1e-9 in these cells" and
that "a cap of 4 changes the energy by roughly 1e-6 at these depths". That held
for the chain, where it was measured, and was not true for the all-to-all cells,
where it was not. The grid had no smoothness record, so the claim looked uniform
across cells it had only been checked on in one family. The `mps_own_map`
section and its `smoothness` entry exist because of that: the difference check is
now read only where the map is smooth, and the artifact states which cells those
are.

## Scope and what was not measured

- **The sharded-versus-unsharded leg is not measured.** It needs two processes,
  and a multi-process `gloo` group on this host did not complete: the c10d store
  spent minutes failing to resolve its own IPv6 address and the run was killed.
  The harness for it exists (`tests/distributed/mps_reverse_runtime.py`,
  `execute_torch_distributed_mps_reverse`), so the leg belongs to a CI
  `distributed-cpu-shard` lane or an accelerator host, and this artifact says so
  instead of implying a placement result.
- The smoothness probe scans one parameter index over `[-0.05, 0.05]` in 7
  points and reports whether the observed bond changed. It is a local sample,
  not a proof of differentiability; a cell it calls smooth could still have a
  jump outside the scanned window.
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
- The `all_to_all` truncated cells do not establish a defect, only that the
  gradient there is not meaningful. Whether a different truncation policy would
  make that surface smooth was not measured.
