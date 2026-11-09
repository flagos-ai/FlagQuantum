# Pauli evolution and one-qubit fold precision

A two-qubit qDRIFT program exposes accumulated error in Euler reconstruction.
Its ordered coefficients are `(-0.2, 0.7, -0.3)` for `(I, Y[0], X[0] Z[1])`.
With time `-0.81`, 2048 samples and seed zero, the independent inverse-CDF
reference differs from the original 7887-instruction sequence by `3.21e-14`.
The previous optimized 1444-instruction sequence differs by `4.18e-12`.
Stage-by-stage evaluation locates the increase in `collapse_one_qubit_runs`.
This is an absolute complex-amplitude comparison, including global phase.

The fold now reconstructs its proposed two-by-two matrix and declines when an
entry differs from the source product by more than eight float64 unit ulps.
This is a local numerical guard, not a formal program-wide error bound.
Uncertain folds retain the source instructions. The end-to-end regression
keeps absolute tolerance `1e-12` and checks positive/negative times, mixed-sign
identity terms, three seeds and sample counts 1, 17 and 2048.

The unchanged seeded microbenchmark was re-measured on local CPU using Python
3.12 and complex128. No accelerator, distributed, gradient or optimality claim
is made. No source population, exactness tolerance or archived evidence changed.
The optional Qiskit comparison was unavailable in this environment.

| Source run length | Prior mean optimized length | Guarded mean |
| --- | --- | --- |
| 2 | 1.3325 | 1.3325 |
| 3 | 1.5588 | 1.5600 |
| 4 | 1.7237 | 1.7300 |
| 5 | 1.8400 | 1.8438 |
| 6 | 1.8838 | 1.8887 |

On the identity-vocabulary population, target-legal gates under the strict
vocabulary rule increase from the prior local 685 to 737. The guarded shipped
rule uses 563, retaining the 174-gate gain from its identity exemption. The
ordinary seeded control remains 771 versus 763. Across the 1200 entangled
programs, maximum raw state difference is `1.46e-15`. Resource pin changes
reflect these measured costs of the precision fix; they are not relaxed gates
for the previous approximate fold. The three-gate cross-machine spread remains.

Pinned original Qmod resource rows remain historical baselines. Fresh replay
records observed resource differences separately; for example the qDRIFT guide
uses 235 operations and depth 207 versus its frozen 225 and 197.
