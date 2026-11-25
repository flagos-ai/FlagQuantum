# Elliptic-curve discrete logarithm

Recover the discrete logarithm of `(0, 2)` relative to `(0, 5)` on
`y² = x³ + 5x + 4 mod 7`. The two scalar registers each keep three fractional
bits. The point register keeps both three-bit coordinates, initialized to
`(4, 2)` in the original nonsingular orbit.

Open [walkthrough.ipynb](walkthrough.ipynb) or run the same workflow from the
repository root.

```bash
python -m examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.overview.main
```

The sampled path uses native `qfunc` point translations and inverse QFTs. It
runs locally, draws 2,048 joint scalar-and-point outcomes, and decodes the ten
highest-count eligible logarithms. An independent Circuit built from the curve
equation checks every complex amplitude.

The second path builds both circuits with reversible modular arithmetic. The
inverse uses six [Kaliski almost-inverse](https://eprint.iacr.org/2017/598.pdf)
rounds with retained branch decisions. The reciprocal is computed, used to
accumulate the slope, and uncomputed. Slope recovery from the output restores
the slope register. The formula assumes both denominators in the affine
translation are nonzero, as they are for this orbit. The notebook checks all
64 scalar branches, verifies clean workspace, and applies the emitted inverse
QFTs on the 12 data qubits for a complete amplitude comparison.

The source's arithmetic circuits are synthesis-only. This walkthrough reports
their native static instruction counts and depths, verifies the emitted
reversible prefix on the supported orbit, and runs the Fourier suffix on the
clean data subspace. The Kaliski schedule spends more work qubits and
instructions than the table-driven route.
