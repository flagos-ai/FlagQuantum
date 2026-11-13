# Fermionic one-body density measurement

## Scope and ownership

Algorithms owns the experimental concrete module
`flagquantum.algorithms.fermion.density`. It builds Hermitian observables and
assembles their measured values into an ordinary PyTorch tensor. It does not
execute circuits, own a solver or introduce a density-matrix class. Consumers
keep using `fq.run(..., outputs=[fq.expectation(...)])`. No Stable Core export,
signature, runtime, dependency or capability registration changes in this slice.

The profile is `single_device_fast_path`: exact local CPU measurements with
float64 expectations and complex128 density tensors. Float32/complex64 assembly
is accepted, without a precision-equivalence claim. Circuit-angle gradients
remain connected through stacking and complex construction. Observable
coefficients and orbital labels are static. No OpenFermion or Cirq is imported.

## Convention and boundary

`gamma[p,q] = <a†[p] a[q]>` uses physical mode order, independent of the encoded
qubit labels. Mode zero is the first occupation bit. The input register is full,
unreduced JW, prefix parity or Fenwick BK with explicit positive `n_modes`.
The caller must prepare the state in the selected encoding; no function can
infer that convention from an arbitrary circuit or unlabelled expectations.
Tapered/ancilla registers and automatic ansatz conversion are outside scope.

`one_body_density_observables(n_modes=..., encoding=...)` returns a flat tuple:
for p in ascending order, emit n[p], then for each q>p emit
`(a†p aq + a†q ap)/2` and `(a†p aq - a†q ap)/(2i)`.
There are exactly n_modes**2 real expectation values. The assembler
`one_body_density(values, n_modes=...)` consumes that order and fills the lower
triangle by complex conjugation. Its result has shape (n_modes, n_modes).
It validates the count and finite real scalar CPU tensors with one common
float32/float64 dtype; it never detaches gradients, clips eigenvalues,
normalizes particle number or projects noisy data onto physical constraints.
Hermiticity is imposed by the measurement design, not an independent noise test.

For `H = offset + sum(h[p,q] a†p aq)`, reconstruct energy using
`offset + (h * gamma).sum()` (no transpose or conjugation). For orbital modes
`b[i] = sum(U[i,p] a[p])`, the density is `U.conj() @ gamma @ U.T`.
Complex Hermitian h is allowed in that consumer. Non-Hermitian energy
coefficients are not made physical by dropping an imaginary residual.

Output and observable count are O(n**2). Encoded strings introduce additional
width-dependent storage and execution cost. No dense Fock matrix/state is
constructed by the density module; runtime statevector storage is still
exponential. Tests use independent occupation-bit references only up to four
modes. No shots, noise, QPU, accelerator or distributed claim is made.

## Acceptance and contributor path

Implement a two-mode coherent one-particle CPU example, reconstruct a complex
one-body Hamiltonian energy, and compare it to direct native measurement.
Vacuum, occupied determinants, complex coherence and a correlated fixed-number
state must match independent ladder action. Test all three encodings,
Hermiticity, trace N, spectral bounds, orbital changes, energy and parameter
gradients. Replace direct occupation-oracle measurements by encoded native
expectations without changing the tensor/energy consumer. Reject invalid
widths, encodings and malformed values with actionable errors. A subprocess
must run the example with external quantum/science SDK imports blocked.

The shortest intended contributor path is:

```bash
python -m examples.algorithms.fermion_one_body_density --encoding parity
python -m pytest tests/unit/fermion/test_density.py tests/integration/test_fermion_one_body_density.py -q
```

This document first records implementation scope; the example and acceptance
results are recorded after implementation. Capability registration remains a
separate Integration change after scientific acceptance.
