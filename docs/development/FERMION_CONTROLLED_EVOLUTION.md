# Controlled low-rank fermionic evolution

## Stage contract

EXT07 composes the existing `low_rank_evolution` builder with the existing
`Circuit.control(1, ctrl_qubits=2*n)` operation. No new public result, wrapper,
Core contract or production dependency is needed. This stage is independent of
EXT06 stabilizer reduction and uses the merged full-register evolution baseline.

The system occupies wires `0..2*n-1` in spin-interleaved, unreduced JW order.
One positive control is appended at wire `2*n`, the least significant state bit.
The zero branch is exactly the identity and the one branch is the existing
first- or second-order product formula, including `exp(-i*time*constant)`.
The constant becomes an observable relative phase and must never be discarded.
No global-phase alignment is allowed in acceptance comparisons.

Static integrals and factors keep their existing validation; only the scalar
time is differentiable. Negative and zero time remain supported. Truncated
factors retain their full contraction correction and separate discarded_bound;
this bound does not certify the product-formula error.

## Architecture and resources

The algorithm owns decomposition and scheduling. Core owns gate control and
wraps each existing one- or two-qubit matrix independently. Every controlled
operand is at most 8 by 8, independent of system size. Instruction count is
unchanged for these matrix-carrying gates and grows linearly with steps; gate
arity increases by one. This is a synthesis bound, not a runtime performance
or hardware routing claim. No full-system unitary is constructed in production.

This profile is `single_device_fast_path`, validated on CPU complex128. It does
not cover parity/BK/tapered registers, multiple or negative controls, gradients
through integrals/eigendecompositions, hardware decomposition or distributed
execution. General Circuit.control behavior outside this profile is unchanged.

## Acceptance

- Independent Fock-space exponentials and signed density-square components
  check both complete control branches and all particle sectors.
- Constant-only and superposed-control interference check the relative phase.
- Both product-formula orders converge with steps; signed/truncated factors,
  rank zero, complex orbital bases, zero and negative time remain covered.
- Time gradients match independent matrix-exponential derivatives and finite
  differences, with no gradient through static scientific inputs.
- Symbolic circuit synthesis checks bounded operand size and resource growth
  without constructing a full-system matrix.
- A fixed H2 example measures a control interferometer with fq.run, and the
  fresh-process workflow runs with external science imports blocked.
- Existing evolution tests continue to pass; full Linux testing belongs to CI.

The stage is delivered as one PR with contract, evidence and workflow commits.
Reverting it removes its examples, tests and documentation without changing
Core, runtime or existing evolution semantics.
