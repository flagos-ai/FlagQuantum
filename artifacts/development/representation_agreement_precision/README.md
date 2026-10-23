# Representation agreement in two arithmetic precisions

These CPU measurements complete issue #709's precision comparison. Both runs
used source commit `4446d1a7c4baa39f5db6e14861b67fd236ac873f`, PyTorch 2.14.1,
one CPU thread, seed 260720 and the same angles drawn in float32, then exactly
widened for complex128. Each JSON records the executed command, source revision,
environment, per-cell energies, bond headroom, spreads and explicit exclusions.
The earlier complex64 artifact is unchanged.

| Regime | Cells per precision | complex64 maximum relative spread | complex128 maximum relative spread |
| --- | --- | --- | --- |
| MPS cap did not truncate | 22 | 4.10343e-06 | 1.10583e-14 |
| MPS cap truncated | 6 | 0.225926 | 0.281990 |

The grid uses widths 4/8/12/16/20, depths 2/4, caps 4/16, a nearest-neighbour
ansatz and an all-to-all ansatz at widths 4/8. All three representations completed
every included cell in both precisions. All-to-all widths 12/16/20 are explicitly
excluded by the pre-existing contraction budget, not reported as measured.
The observable is the open-chain transverse-field Ising energy at h=0.7.

Higher precision reduces the spread in cells where no weight is discarded;
it does not restore agreement after bond truncation. The worst truncated cell
is all-to-all, width 8, depth 4, cap 4 in both runs. The two truncated states
also differ across precisions, so this grid does not establish a
precision-independent truncation bound. These are measured energy spreads for
one circuit family, not bounds for arbitrary states, gradients or GPU execution.
The static MPS path is not a TEBD time-evolution measurement.

Reproduce from the recorded source revision:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/representation_agreement_sweep.py --dtype complex64 --output /tmp/agreement64.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/representation_agreement_sweep.py --dtype complex128 --output /tmp/agreement128.json
python -m pytest tests/integration/test_representation_agreement_sweep.py -q
```

The CI subgrid checks untruncated energy agreement in both precisions, retaining
the existing complex64 threshold and a 1e-12 complex128 threshold. These files
remain development evidence with release and scalability claims disabled.
