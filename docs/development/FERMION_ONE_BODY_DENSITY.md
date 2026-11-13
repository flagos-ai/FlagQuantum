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
It validates the count and finite real single-element CPU tensors with one common
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

The example reports particle number 1, energy 0.4520082847, angle gradient
-0.5016897975 and a direct-energy residual below 1e-12 for all three encodings.
Its preparation uses X/RY/CX/S gates without dense state loading. The test-only
occupation oracle and complex unitary completion are bounded to four modes;
they are not part of the normal execution path. The separate Integration
registration records this executed scope as experimental in
[`capability-maturity.toml`](../../capability-maturity.toml). The static
observable producer and differentiable measured-tensor assembler are distinct;
the entry's gradient field describes the latter in the existing execution
composition, not trainable observable coefficients.

## Local verification

On macOS CPU, the density unit/integration tests and selected example checks pass
47 cases; the existing encoding, fermion-algebra, dependency-isolation and public
API candidate checks pass another 71 cases:

```bash
python -m pytest tests/unit/fermion/test_density.py tests/integration/test_fermion_one_body_density.py tests/test_algorithm_examples.py -k 'density or covered_and_indexed' -q
python -m pytest tests/unit/fermion/test_encoding.py tests/unit/test_observables_fermion.py tests/integration/test_native_fermion_isolation.py tests/unit/test_public_api_candidate.py -q
```

The density tests include a planner refusal for an observable wider than its
circuit. They do not claim detection of mislabeled same-width states. Import
blocking covers OpenFermion, Cirq and external chemistry/science SDKs in a fresh
process for every encoding. Normal example execution refuses eigensolver use.
Strict module typing, Ruff, Black, architecture boundaries, protected API
baseline, generated-document consistency, repository language and local Markdown
links also pass. This records local scientific evidence; Linux CI and the full
default tier are recorded independently in the PR rather than inferred here.

## Merged implementation acceptance

Implementation [PR #1179](https://github.com/flagos-ai/FlagQuantum/pull/1179)
merged as `8f71c80fd0732feeb74d25a71b9ad9b21ab3ad9f` on November 13, 2026
at 14:08:53 UTC. Its final head
`184a072641faffd10eeeca52a9e974492fed6906` passed 239 CI checks, with three
conditional skips. The repair adds the actual density observable producer to
the executable-docstring entry list; both previously failing jobs then passed.
No assertion or production behavior was weakened.

The actual merged source archive passed 118 density, complete docstring,
encoding, fermion-algebra, dependency-isolation and public-API candidate tests.
The protected API baseline also passed. Exact-commit
[Main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/43439979724)
passed. Git fetch timeouts were followed by an exact-revision GitHub API archive
for that local acceptance, rather than treating the PR head as actual main.
These are bounded CPU scientific and contract results, not performance or
hardware qualification.

The earlier local full default tier reached approximately 56% at its 600-second
budget and was not a pass. Its three QFT phase-fusion count failures reproduced
on a separate clean baseline `b58fc8a01d9395972042abd1aaa507c9769b73da`.
Those local results remain distinct from the later successful Linux PR CI.
Optional local coverage instrumentation exited before test output, so no local
coverage percentage is claimed.
