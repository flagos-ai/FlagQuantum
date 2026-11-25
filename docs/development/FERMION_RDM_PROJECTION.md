# Fixed-particle density constraint projection

Use this optional postprocessing step when the particle number is known and
measured one- and two-body densities disagree with its linear identities.
`project_fixed_particle_densities` in
`flagquantum.algorithms.fermion.density_projection` accepts the same physical
mode order as the native [two-body workflow](FERMION_TWO_BODY_DENSITY.md).
It returns fresh tensors; retain the raw measurements for comparison. The
projection does not read a circuit, choose a particle sector or call a runtime.

For `gamma[p,s] = <a†p a_s>` and
`Gamma[p,q,r,s] = <a†p a†q a_r a_s>`, it enforces Hermiticity, pair
antisymmetry, `trace(gamma) = N`, and
`sum_q Gamma[p,q,q,s] = (N-1) gamma[p,s]`. The latter two imply
`sum_{p,q} Gamma[p,q,q,p] = N(N-1)`. The minimum-distance objective is the
sum of squared Frobenius distances in the one-mode matrix and the independent
pair matrix `P[(p,q),(r,s)] = Gamma[p,q,s,r]` for `p<q`, `r<s`.
Equivalently, the full four-index tensor carries weight 1/4 in that objective.
This is a fixed metric, without shot-variance weights.

The projection first enforces symmetry and trace in each space. The contraction
adjoint lifts the traceless residual back to pair space. On that subspace,
the contraction followed by its adjoint in reverse order, `C C*`, acts as
multiplication by `m-2` for `m` modes; the coupled correction divides by
`m-2+(N-1)^2`. No dense linear solver is used in the implementation. An
independently constructed pair-space constraint matrix and pseudoinverse
verify the least-norm solution in bounded tests.

Run the four-mode H2 composition:

```bash
python -m examples.algorithms.fermion_two_body_density --encoding parity --project
python -m pytest tests/unit/fermion/test_density_projection.py -q
```

The example verifies that exact inputs are fixed points and reports the
perturbed contraction residual (about 0.137), projected residual (below
1e-10), and energies before and after (about -1.0730413218 and -0.8944686996
Ha). The exact prepared-state energy is -0.8766911918 Ha. The demonstration
uses deterministic rescaling, not samples from a noise or hardware model.

The experimental scope is static dense CPU complex128 tensors, two through
eight modes, and explicit integer `2 <= N <= m`. The function rejects
non-finite data, autograd inputs, unsupported shapes and outputs outside an
absolute residual tolerance of 1e-10. Dense work and storage grow as O(m^4).
The result satisfies the declared linear identities; it has no positivity or
N-representability guarantee, and energy error need not improve. No dependency,
stable API, runtime, or automatic correction is added. The capability is
registered separately as experimental.

## Merged implementation acceptance

Implementation [#1185](https://github.com/flagos-ai/FlagQuantum/pull/1185)
merged at `7b45cad4f4bcb8db27dd407adac7f586c872036d`. Its final head
`a67d2047e409dc1a063f94ae004d7eb62622c44d` passed 239 checks with three
conditional skips. The merge tree matches the verified PR tree.
[Main smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/43807297709),
75 focused tests, and the protected API baseline passed on the merged revision.
This evidence supports the experimental static CPU scope described above.
