# Native fermionic two-body density

The experimental workflow reconstructs a dense complex two-body reduced density
tensor from native exact CPU expectations. It uses the same full-register
Jordan–Wigner, prefix parity and Fenwick Bravyi–Kitaev encodings as
[one-body density](FERMION_ONE_BODY_DENSITY.md). Prepare the state in the chosen
encoding at the requested width. The producer constructs static observables;
callers execute them with `fq.run`. The assembler consumes ordered tensors,
preserves their circuit-parameter gradients and performs no execution.

## Ordering and contraction

In physical mode order, `Gamma[p,q,r,s] = <a†p a†q a_r a_s>`. Let
`A(p,q) = a_q a_p` for `p < q`. In lexicographic pair order, measure each
diagonal of `<A†(p,q) A(r,s)>`, then the real and imaginary parts against each
later pair. The matrix element at pairs `(p,q),(r,s)` is `Gamma[p,q,s,r]`.
Assembly fills conjugate partners, applies a minus sign for either pair swap,
and assigns zero when indices in either pair coincide. In particular:

- `Gamma[p,q,r,s] = conj(Gamma[s,r,q,p])`.
- `sum_{p,q} Gamma[p,q,q,p] = N(N-1)` for an N-particle state.
- `sum_q Gamma[p,q,q,s] = (N-1) gamma[p,s]` in a fixed-number state. This
  identity cannot recover the 1-RDM at N=1; the example measures it separately.
  Vacuum and single-particle states have zero 2-RDM.

For interleaved spin modes, the native molecular coefficient convention
multiplies `Gamma[2p+a,2r+b,2s+b,2q+a]` by `0.5 * (pq|rs)` for spins
`a,b` in {0,1}. Add `h[p,q] * gamma[2p+a,2q+a]` over equal spins and add
the energy offset once. Neither contraction conjugates the density. Coefficients
are fixed data; only the circuit angle is trainable in the example.

## Executable example

```bash
python -m examples.algorithms.fermion_two_body_density --encoding parity
```

The four-mode H2 FCIDUMP example prepares a fixed pair of determinants with
RY, CX and X gates. It reports two ordered particle pairs, energy
`-0.8766911918 Ha`, angle derivative `0.5945439485 Ha/rad`, and residual
below `1e-12 Ha` against the direct Hamiltonian expectation. The energy is
for that prepared state. JW and BK give the same results.

Run `python -m pytest tests/unit/fermion/test_density.py tests/integration/test_fermion_one_body_density.py -q`.
The tests apply annihilators independently in the occupation basis and verify
all tensor indices for zero through three particles across the three encodings.
They also compare the H2 energy and angle gradient with direct measurement and
centered finite differences, without a dense oracle in the example.

## Boundaries

At least two modes are required. For n modes, pass exactly
`binomial(n, 2)**2` finite single-element CPU float32 or float64 tensors of
one dtype; the output is complex64 or complex128 respectively, shaped
`(n, n, n, n)`. Both the number of observables and the dense output cost
O(n^4); the encoded Pauli strings and statevector execution have additional
costs. Scientific evidence is bounded to four modes in complex128; float32
coverage concerns assembly only.

The assembler enforces algebraic symmetries without clipping, normalization,
noise projection or an N-representability check. It cannot detect a mislabeled
same-width encoding or mode order from the expectation values. Sampled,
tapered, ancilla, accelerator and distributed workflows have no qualification
here. There is no OpenFermion or Cirq dependency, stable export, new RDM type
or solver. Capability registration is a separate Integration change.

## Merged implementation acceptance

Implementation [PR #1182](https://github.com/flagos-ai/FlagQuantum/pull/1182)
merged at `41a11c13b24624034ec9d2d613d1add4fc0f763f`. Its final head
`dbbb94012cd9edd79c5518486c3bcbecfa82a619` passed 239 checks with three
conditional skips. Linux CI on the repaired head validated the indexed H2 CLI
example and the numerical doctest independent of signed-zero formatting.
[Main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/43795387760)
passed on the merged revision. A focused local run on that revision passed
69 density, CLI, doctest and dependency-isolation tests. This evidence
supports the bounded CPU workflow; the capability remains experimental.
