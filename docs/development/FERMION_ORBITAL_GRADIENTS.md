# Differentiable orbital rotations

## Stage contract

EXT08 supplies a fixed basis of real antisymmetric orbital generators and a
trainable real parameter vector. The generator-weighted sum is exponentiated to
an orthogonal orbital rotation. One-body and real chemist-convention two-body
integrals are transformed in that same basis, preserving autograd through the
parameters. The returned tensors remain differentiable and are not wrapped in
`MolecularProblem`, whose authoritative fields are immutable static snapshots.

For column orbitals `C_new = C U`, the transform is `h_new = U.T h U` and
`(pq|rs)_new = sum_abcd U_ap U_bq U_cr U_ds (ab|cd)`. Generator tensors and
integrals are static CPU inputs; only the real floating parameter vector can
carry gradients. Inputs must be finite and real, and generators must be
antisymmetric. No geometry, SCF, integral-engine, eigendecomposition or Givens
branching gradient is included.

A closed-shell determinant energy contracts the transformed tensors for an
explicit occupied-orbital list. Its scalar output remains in the graph and
provides the smallest end-to-end orbital-optimization objective.

## Boundaries

The implementation lives in `chemistry_domain`, independent of circuit lowering,
fermion encodings and execution. It adds no dependency, Core interface or stable
root export. It supports real spatial integrals and explicit closed-shell
occupations only. Complex unitary rotations, unrestricted/open-shell states,
active-space selection gradients and optimization policy remain out of scope.

## Acceptance

- Zero parameters reproduce the supplied integrals exactly and preserve a finite
  gradient graph.
- Independent index-wise transforms and all chemist permutation symmetries
  validate the tensor convention.
- The rotated second-quantized Hamiltonian retains its complete small Fock-space
  spectrum.
- Closed-shell determinant energy and its parameter gradient agree with an
  independently assembled Fock-space expectation and central finite differences.
- Invalid generator dimensions, complex/nonfinite data, non-antisymmetric
  generators and invalid parameter tensors fail at the boundary.
- A fixed H2 example computes a nonzero orbital gradient and compares it with a
  finite difference. Linux full-suite validation remains in CI.
