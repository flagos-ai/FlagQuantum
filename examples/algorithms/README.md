# Algorithm examples

Runnable demonstrations of the Phase 2 algorithm units in
`flagquantum.algorithms`, one per unit of that phase, plus a private constrained
arithmetic synthesis workflow trial.
[`docs/guides/ALGORITHMS.md`](../../docs/guides/ALGORITHMS.md) is the per-unit
reference they follow, and the place each unit's advantage premise is recorded in
full.

Run them from the repository root:

```bash
python -m examples.algorithms.pca
python -m examples.algorithms.kmedians
python -m examples.algorithms.quantum_kernel
python -m examples.algorithms.feature_selection
python -m examples.algorithms.qarm
python -m examples.algorithms.svd
python -m examples.algorithms.error_mitigation
python -m examples.algorithms.pec
python -m examples.algorithms.cdr
python -m examples.algorithms.readout_mitigation
python -m examples.algorithms.folding
python -m examples.algorithms.variational_solvers
python -m examples.algorithms.vqe_solvers
python -m examples.algorithms.spsa_optimizer
python -m examples.algorithms.nelder_mead_optimizer
python -m examples.algorithms.cobyla_optimizer
python -m examples.algorithms.trotter
python -m examples.algorithms.block_encoding
python -m examples.algorithms.linear_combination
python -m examples.algorithms.pauli_lcu_block_encoding
python -m examples.algorithms.sparse_select
python -m examples.algorithms.logical_resources
python -m examples.algorithms.arithmetic
python -m examples.synthesis.tutorials.getting_started.trial.main
python -m tools.synthesis_import_examples.source_trial.main
```

[`tests/test_algorithm_examples.py`](../../tests/test_algorithm_examples.py) runs
every script in this directory as a subprocess and asserts on what it printed, so
a script that stops working fails the test lane instead of going stale.

What they show:

- [`pca.py`](pca.py): the eigenvalue readout of a 2x2 data matrix's density
  matrix, by phase estimation over its exponential.
- [`kmedians.py`](kmedians.py): one k-medians assignment step, each point's
  nearest centroid found by a Grover-style minimum search, then the classical
  median update.
- [`quantum_kernel.py`](quantum_kernel.py): a kernel matrix estimated by swap
  test, and a classical kernel ridge classifier fitted on the estimated entries.
- [`feature_selection.py`](feature_selection.py): one feature-selection objective
  built and evaluated. It runs no circuit, because the unit has no quantum part.
- [`qarm.py`](qarm.py): the fraction of a database's items whose support meets a
  threshold, by amplitude estimation over a support register.
- [`svd.py`](svd.py): a matrix's singular values read off the phase of its
  Hermitian embedding's exponential, plus the boundary of a one-qubit counting
  register.
- [`error_mitigation.py`](error_mitigation.py): an observable continued to zero
  noise by polynomial least squares and by Richardson extrapolation over four
  scaled models, with the residual of an underfit, of a square fit, and of a
  non-polynomial family printed beside the estimate.
- [`pec.py`](pec.py): a declared Pauli channel inverted from its Pauli transfer
  matrix and inserted after the channel it inverts, with the cost that composes
  over locations printed beside the estimate, and the four channel families whose
  transfer matrix is not diagonal refused by name together with a channel whose
  transfer eigenvalue reaches zero.
- [`cdr.py`](cdr.py): the noise's effect on a target fitted as an affine
  relation over Clifford training circuits built by snapping each rotation to its
  nearest quarter turn, with the residual that exposes a premise which did not
  hold printed beside a two-point fit's residual reported as absent and a
  three-point fit's reported as a measurement.
- [`readout_mitigation.py`](readout_mitigation.py): a declared readout confusion
  inverted off a measured distribution, one block per readout rule, with the
  amplification the correction costs printed beside the corrected vector, the
  negative mass a corrected vector is allowed to carry summed and reported rather
  than clipped away, a correlated pair shown inverted as one block rather than as
  two marginals, and three refusals printed by measured quantity: a singular
  value against the floor, a block width against the ceiling, and a bit-string
  length against the plan's width.
- [`folding.py`](folding.py): one program folded by both strategies, with the
  requested factor printed beside the ratio each strategy actually realized, the
  instructions each spent to reach it, the amplitude gap that shows the fold is
  an identity rather than an approximation, and the pair's own location shown to
  decide which observables can see the noise the fold added. A second model names
  one instruction of the five, and both placements are folded against it at one
  request: the default rule's curve is printed as one number at every abscissa,
  because its pair lands on an instruction the model does not name, while the
  placed rule's curve moves — and the placed estimate is printed as the farther
  one from the noiseless value, so what a placement buys is shown to be reach
  rather than accuracy.
- [`variational_solvers.py`](variational_solvers.py): a weighted MaxCut
  instance solved by QAOA at three layer counts from one constant start, with the
  cost operator checked against the cut spectrum enumerated in Python, the
  gradient at the uniform superposition printed as exactly zero so the saddle is
  shown rather than asserted, and the maximum cut decoded from the optimized
  circuit's own probability distribution rather than read off the energy.
- [`vqe_solvers.py`](vqe_solvers.py): one two-qubit operator minimized by all four
  VQE entry points rather than by one, so the fixed-ansatz, the operator-growing,
  the scheduled and the deepening descents are read side by side against the
  operator's own ground energy. The ADAPT screen is printed as the exact
  derivative it is -- measured from `|00>` at `-1.0, 0.0, -0.0` over three pool
  directions -- and the same pool run from the stationary `|++>` state is shown
  selecting nothing and still reporting `converged=True`, which is the premise
  the unit cannot check rather than a defect. `converged` is then read against
  three runs that reach it for three different reasons, and the one reporting
  `False` is the one furthest from the ground energy, so the flag is an exit test
  and not an arrival test. Two refusals close it: an objective with no gradient
  and a pool element with none, each naming the entry point and the cause instead
  of the torch backward error that names neither.
- [`spsa_optimizer.py`](spsa_optimizer.py): a Pauli energy minimized from samples
  at two evaluations per step, with the parameter-shift gradient's own evaluation
  count measured beside it.

- [`nelder_mead_optimizer.py`](nelder_mead_optimizer.py): the same Pauli energy
  minimized with no gradient, no parameter-shift rule and no random draw, with
  SPSA run beside it at the *same evaluation budget* so the two optimizer units
  are compared on one objective rather than each being merely present, and the
  two-well quartic printed to measure what a converged flag does not mean.
- [`cobyla_optimizer.py`](cobyla_optimizer.py): that Pauli energy again, this time
  under a constraint that binds, with the unconstrained run beside it so the cost
  of the bound is a number, and one quadratic under one affine constraint solved
  twice at two prices so the second half of the unit's premise is measured rather
  than asserted: the cheaper price returns a point that violates the constraint
  and still scores better than the feasible optimum, which is why `feasible` is
  printed beside `value`.
- [`trotter.py`](trotter.py): a transverse-field Ising Hamiltonian turned into the
  circuit a product formula applies, with the defect at two step counts per order
  measured against `torch.matrix_exp`, the primitive's own emitted gates printed
  for six words, and the resulting circuit run through the static resource
  estimator and the gradient path.
- [`block_encoding.py`](block_encoding.py): a Hermitian matrix encoded into the
  flagged block of a unitary, with the block read back out of the circuit one
  basis state at a time and checked against both the matrix over `alpha` and the
  matrix itself, the walk step's eigenphases matched against the matrix's own
  spectrum from a different routine, a second implementation that holds no matrix
  read by the same consumer, and the ten construction and three register refusals
  printed by name.
- [`sparse_select.py`](sparse_select.py): runs sparse chemistry SELECT with dynamic sampling and checks the unitary gate decomposition against the dense reference.
- [`pauli_lcu_block_encoding.py`](pauli_lcu_block_encoding.py): verifies the target block and eigenphases of a gate-level qubitization walk.
- [`linear_combination.py`](linear_combination.py): a six-term Pauli sum block
  encoded from its coefficients alone, with `alpha` printed beside the spectral
  and Frobenius norms it deliberately is not, the block read back out of the
  circuit and checked against the signed sum and against the same sum with the
  signs dropped, the walk step's cosines matched on the ladder-zero subspace
  where the identity is claimed rather than on the whole register, and the
  encoding's own non-Hermiticity printed as the reason the adjoint reverses the
  select's terms instead of reusing them.
- [`logical_resources.py`](logical_resources.py): a Clifford+T program costed on a
  rotated surface code, with the compiler's own resource record printed beside the
  report's copy of it so a second T rule would show, the distance sweep tabulated
  against the model's closed form, the one-layer charge for a measurement record
  shown against the same program without one, and the two refused opcode families,
  a lowered channel, and six refused arguments each printed by name.
- [`arithmetic.py`](arithmetic.py): a reversible adder built at seven widths and
  then run on the statevector path, with the sum decoded back out of the runtime's
  own state vector, the two ancillas' behaviour off the contract measured over
  every input rather than described, the T-cost read from the compiler's own
  Toffoli rule, and the conversion bound shown to be a bound the caller can raise.

- [`synthesis_trial.py`](../synthesis/tutorials/getting_started/trial/main.py): the maintained synthesis candidate
  synthesizes `x*x+y` with named basis and uniform input preparation, executes
  its checked bound Program, and reads named UInt tuples without
  user-written gates, physical indices or bitstring decoding. It also executes
  a constrained line-topology example. Finite candidate selection and exact
  local CPU statevectors are supported. There is no optimality claim; independent
  developer trials and final release review remain pending.
