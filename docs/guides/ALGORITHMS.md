# Quantum algorithms (demonstration scale)

> This guide is the shared index for the algorithm units in
> `flagquantum.algorithms/`. Every capability entry in this programme links
> here, and the promise below applies to all of them.

`flagquantum.algorithms` composes circuits, observables, and optimization into
user-facing algorithm units. This guide records, for each unit, what it does,
where its construction comes from, and what advantage premise it rests on. The
advantage premise is the part that is easiest to lose: a unit that is correct,
tested, and reproducible still does not carry an advantage of its own.

## Index

| Unit | What it does | Citation | Advantage premise |
| --- | --- | --- | --- |
| `qubo.py` — QUBO ↔ Ising mapping | Available. Converts a quadratic unconstrained binary optimization problem into an Ising Hamiltonian for the existing variational workflows, and reads the QUBO form back. | Boros & Hammer 2002; Barahona 1982; Lucas 2014 | **None.** A polynomial classical transformation. |
| `primitives/qft.py` — quantum Fourier transform | Available as a shared primitive; phase estimation consumes it. | — | None. It is a subroutine. |
| `primitives/phase_estimation.py` — phase estimation | Available. Applies a controlled unitary's powers to a uniform counting register, then inverts the Fourier transform on it, turning the accumulated phase into a readable integer. | Kitaev 1995; Brassard et al. 2002 | **None.** It is a subroutine, and the cost of preparing the operator's eigenstate is not counted. |
| `primitives/state_preparation.py` — state preparation | Available. Prepares a uniform superposition, and prepares an arbitrary state from a classical amplitude vector with uniformly controlled rotations. | Möttönen et al. 2005 | **None, and the input is exponential.** The rotation angles come from a classical pass over all `2**n` amplitudes and a `2**n` by `2**n` linear solve, so the amplitudes must already be known. |
| `data_encoding.py` — encoding a classical feature vector | Available. Pads a feature vector to a power of two and normalises it into the amplitudes of a prepared state, and encodes a second vector as one rotation per wire in either a fresh circuit or a circuit the caller already holds. | Möttönen et al. 2005 (amplitude encoding; the preparation itself is the state-preparation primitive) | **None, and the input is exponential for the amplitude half while the angular half is not an advantage either.** Amplitude encoding hands the padded vector to the same uniformly controlled rotation ladder the state-preparation primitive uses, so its classical input is already ``2**n`` amplitudes; angular encoding is one cheap gate per wire and packs far less data per wire, so it is a feature map rather than a compression. Neither is a speedup claim. |
| `primitives/oracle.py` — oracle synthesis | Available. Synthesizes a phase or bit oracle from a classical predicate's truth table, on top of the multi-controlled X and comparator building blocks. | — | **None, and the cost is exponential.** Synthesis enumerates all `2**n` inputs classically. |
| `primitives/block_encoding.py` — block encoding and qubitization walk step | Available. Encodes a Hermitian matrix into the flagged block of a unitary at a subnormalisation the caller chooses, and appends one qubitization walk step whose eigenphases are the arccosines of the encoded eigenvalues. | Gilyén, Su, Low & Wiebe 2019 | **None, and the cost is a dense eigendecomposition.** The gate is one dense matrix on `n + 1` qubits built from `torch.linalg.eigh` of the input, so the cost is diagonalising the caller's matrix; the error of an approximate input is the caller's, because nothing here bounds it. |
| `primitives/linear_combination.py` — block encoding of a weighted Pauli sum | Available. Encodes a Hamiltonian given as a weighted sum of Pauli words, from its coefficients rather than from a matrix, at the sum of the coefficient magnitudes as the subnormalisation, and satisfies the same two protocols so one consumer reads both this and the spectral encoding. | Gilyén, Su, Low & Wiebe 2019 | **None, and the preparation is exponential in the index register.** The coefficient weights are prepared as `2**k` amplitudes by a dense classical precomputation, with one index wire per factor of two in the term count rather than one wire per term, and the select is one multi-controlled Pauli per term; there is no gate-efficient synthesis, no amplitude amplification, and no gate-count or error bound, so this is a construction rather than a cost claim. |
| `grover.py` — Grover search | Available. Amplifies the amplitude of the states a predicate marks, so a marked state is recovered from far fewer samples than uniform sampling needs. | Grover 1996 | **Query model.** The oracle's own cost is not counted; here it is a truth table, so no end-to-end advantage at demonstration scale. |
| `amplitude_estimation.py` — amplitude estimation | Available. Estimates the amplitude a marking operator selects, by phase estimation over the Grover operator. | Brassard et al. 2002 | **The state-preparation unitary is assumed free.** A real distribution needs QRAM, so this is not an end-to-end advantage. |
| `spsa.py` — simultaneous perturbation stochastic approximation | Available. Minimizes a scalar objective with no gradient, from two evaluations per step whatever the parameter count, on the recursion ``theta_{k+1} = theta_k - a_k g_hat_k`` with ``g_hat_k`` built from one random sign vector. | Spall 1992; Spall 1998 | **The premise is that no gradient is available, and the estimate is not a gradient.** It is biased for every finite perturbation and its expectation reaches the gradient only as the perturbation shrinks, so a single estimate is not a descent direction. An objective with an exact gradient is served more cheaply and exactly by autograd or parameter shift, and no end-to-end advantage follows. |
| `nelder_mead.py` — Nelder-Mead simplex search | Available. Minimizes a scalar objective with no gradient and no random draw, by reflecting, expanding, contracting, and shrinking a simplex of ``n + 1`` vertices, and stops when the objective values across the simplex and the vertex positions across the simplex both fall under the caller's thresholds. | Nelder & Mead 1965; Lagarias, Reeds, Wright & Wright 1998 | **The premise is that the objective is deterministic, and it is a local method.** Every step is a ranking of two objective values against each other, so a stochastic objective turns those decisions into noise and this unit is not a substitute for `spsa.py` beside it; `converged` reports a collapsed simplex rather than a global minimum, and the published counterexample's collinear start is refused rather than reproduced, so a local minimum is what remains reachable. No end-to-end advantage follows. |
| `trotter.py` — time evolution by a product formula | Available. Turns a weighted Pauli sum into the circuit a product formula applies, as a basis change and a CX ladder per term, so a time-evolution workload is an ordinary circuit; the primitive underneath it is the exact circuit for ``exp(-i * theta * P)`` of one Pauli word. | Trotter 1959; Suzuki 1990; Suzuki 1991; Lloyd 1996 | **The premise is commutativity, and it fails by exactly the amount nothing here bounds.** A product formula is exact only when the terms commute; otherwise the defect falls off with the step length at the composition's own rate and no bound is reported, because a bound needs a commutator norm that belongs to the caller. The term count is the caller's Hamiltonian's, so no end-to-end advantage follows. |
| `logical_resources.py` — logical-layer resource estimation | Available. Reads a Clifford+T program's operation counts, T family, and schedule depth out of the compiler's own static estimate and costs them on a rotated surface code at a distance the caller names: logical layers, physical qubits, surface-code cycles, and their product. | Fowler et al. 2012 | **None, and it is a count rather than a measurement.** Nothing runs, and no wall-clock time, memory, or allocation is read; the input has to already be Clifford+T, so a parametric rotation is refused rather than synthesised; no distillation factory, magic-state budget, routing overhead, placement, or device model is included; and no logical error rate is reported, because that number needs a device's threshold fit. |

| `arithmetic.py` — reversible integer addition | Available. Builds Cuccaro's in-place ripple-carry adder: two `n`-bit registers added into the second of them, on one working carry wire, with the carry out of the most significant position left on a wire of its own. | Cuccaro et al. 2004 | **None, and the cost is the compiler's rather than this unit's.** No seven-T expansion is written here, because the T-count is the compiler's own Toffoli identity reached through `convert_basis` and `estimate_resources`, and a second expansion would be a second source of truth for the one number the two must agree on; the construction is Cuccaro's and not the smaller Gidney-Ekera one, which uncomputes its ancillas by measurement and feedforward and needs a mid-circuit measurement path this repository has no costable circuit for; two wires must enter holding `|0>` and only those inputs are promised a sum; and no distillation factory, magic-state budget, routing, or layout is included, so a surface-code figure derived from it is a floor. |

Two rows carry `—` in the Citation column rather than a source, and that is a
statement rather than a placeholder: the Fourier transform and oracle synthesis
are standard constructions with no single paper to cite, and the one cited
component of the second, the bit-string comparator, carries the semi-verified
record under Sources. A source that exists but has not been confirmed is
recorded as exactly that, the way the comparator's is — so a `—` never means
"sought and not yet found".

The primitive rows are listed separately because the package admits a primitive
on one of two grounds: when more than one algorithm module needs it, or is
expected to need it and the expectation is later confirmed — the Fourier
transform shipped with one consumer, phase estimation, and was admitted on the
expectation of a second, which amplitude estimation's arrival confirmed — or
when it is a public unit callers use directly, which is how state preparation
was admitted, with no consumer inside `flagquantum/` at all, and how block
encoding was admitted after the singular-value unit held it privately and the
comment recording its own promotion condition named that ground. A primitive does
not by itself change what a caller can run.

The Phase 2 quantum machine learning units are indexed in the table under
[Quantum machine learning units (Phase 2)](#quantum-machine-learning-units-phase-2)
below, which is where each unit of that phase adds its row.

## Quantum machine learning units (Phase 2)

These rows are the Phase 2 quantum machine learning units. They are tabulated
apart from the index above so that a phase's own set can be read together, and
each unit of this phase appends its row here as it lands. The same promise
applies to them as to everything else in this guide.

**Each unit of this phase ships with a runnable demonstration.**
[`examples/algorithms/`](../../examples/algorithms/README.md) holds one script per
Phase 2 unit — `pca.py`, `kmedians.py`, `quantum_kernel.py`,
`feature_selection.py`, `qarm.py` and `svd.py` — and each runs from the
repository root as `python -m examples.algorithms.<unit>`. A script prints the
advantage premise its unit rests on beside the result it measured, and
`tests/test_algorithm_examples.py` runs all six and asserts on what they printed,
so a script that stops working fails a test rather than going stale. The scripts
import their unit from `flagquantum.algorithms.<unit>`, because none of these
units carries a root-level `fq.` name.

| Unit | What it does | Citation | Advantage premise |
| --- | --- | --- | --- |
| `pca.py` — quantum PCA | Available. Estimates the eigenvalues of a data matrix's density matrix from a purification of it, by phase-estimating `exp(-2 pi i rho)` and reading the counting register. | Lloyd et al. 2014 | **The premise is the input model, and it is not met.** The paper's subroutine consumes copies of `rho` and never forms it, in `O(1/eps**3)` of them; this unit forms `rho` classically, builds its exponential as a dense matrix, and takes the purification's `2**n` amplitudes from the caller. No end-to-end advantage follows. |
| `kmedians.py` — quantum k-medians | Available. Assigns each point to its nearest centroid with a Grover-style minimum search over a centroid index register, then moves each centroid to the classical coordinate-wise median of its cluster. | Aïmeur et al. 2007 | **The premise is the oracle model, and the oracle is not free.** The search's oracle is synthesized from the predicate's truth table at `O(2**n)` cost, and the distance table the predicate compares is computed classically, one point at a time, before any circuit is built. No end-to-end advantage follows. |
| `quantum_kernel.py` — quantum kernel estimation and kernel ridge classification | Available. Estimates a kernel matrix by swap test over an angle-encoded feature map — one sampled entry per pair of points, mirrored across the diagonal — and fits a **classical** kernel ridge classifier on the estimated entries. | Havlíček et al. 2019; Liu et al. 2021 | **The premise is the data-access model, and it is not met.** The kernel-matrix circuit's cost counts the swap tests and not the data access: the two feature states are assumed to be available, through a qRAM or an amplitude-encoding unitary, and each is built here gate by gate from the classical feature vector. No end-to-end advantage follows. |
| `feature_selection.py` — feature selection as a QUBO | Available. Builds the binary objective of a feature-selection instance — a subset's relevance and pairwise redundancy, scored with a penalty on the subset's size — and evaluates it at an assignment or maps it to an Ising Hamiltonian. | Ferrari Dacrema et al. 2022 | **This unit does not solve, and there is no solver here.** The repository has no annealer: the unit builds the objective and evaluates it, so which subset comes back, and at what cost, belongs to whatever solver the problem is handed to, and any advantage such a solver observes is the solver's. |
| `qarm.py` — frequent-item fractions by amplitude estimation | Available. Estimates the fraction of a database's items whose support meets a threshold: a uniform superposition over the items, a support register the circuit fills one controlled increment per transaction-item membership, and a mark at the threshold, read out by amplitude estimation. | Yu et al. 2016 | **The premise is coherent entry-wise database access, and it is not met.** The paper's speed-up counts calls to an oracle that returns one database entry per call, and it reaches the candidate itemset superpositions it prepares through a qRAM; here the transactions are iterated classically and the incidence matrix is read in Python to emit that loop. Its improvement is **quadratic and conditional**, stated for the case `M_f^(k) << M_c^(k)` — not exponential. No end-to-end advantage follows. |
| `svd.py` — singular values by phase estimation | Available. Estimates a matrix's singular values from the phase of its Hermitian embedding — exponentiated, phase-estimated, and read at the counting register's mode — and builds that embedding's block encoding out of the shared primitive. | Kerenidis & Prakash 2017; Rebentrost et al. 2018; Gilyén et al. 2019 | **The premise is the input model, and it is not met.** The algorithm's cost is counted in queries to a structure that returns the matrix's entries, and against that count the state the estimation is applied to is assumed to be preparable. Here the matrix, its embedding, the embedding's exponential and the input state are all formed classically, the input state from the very decomposition the readout estimates. No end-to-end advantage follows. |

## Advantage premises

- **The QUBO mapping carries no advantage premise.** It is a polynomial
  classical transformation between two encodings of the same objective. It
  produces no speedup of its own: any advantage a caller observes belongs to the
  solver that consumes the Hamiltonian, not to this mapping. The module says so
  in its own docstring, and the capability entry repeats it as its boundary.
- **The other primitives carry none either.** The quantum Fourier transform,
  phase estimation, state preparation, and oracle synthesis have all shipped as
  subroutines. They do not have an advantage premise to state: their cost is
  paid by whatever algorithm calls them.
- **State preparation specifically rests on the inverse of a speedup claim.** It
  is the one primitive in the index whose classical input is as large as its
  quantum output: see the section below.
- **Grover search and amplitude estimation are query/oracle-model results.**
  Their advantage premise is a cheap oracle — and, for amplitude estimation, a
  cheap state-preparation operator as well. A gate-level implementation pays
  both costs rather than assuming them away: writing an oracle for an arbitrary
  predicate costs on the order of `2**n` gates, and the state-preparation
  operator is not free. Neither unit yields an end-to-end advantage at
  demonstration scale.
- **Quantum PCA's premise is its input model, and the unit does not meet it.**
  The algorithm's contribution is that it never forms `rho`: it consumes copies
  of it, one use per run, in `O(1/eps**3)` of them, and that count is what a
  speedup would be measured against. This unit forms `rho` classically, builds
  its exponential as a dense matrix, and takes the purification's amplitudes
  from the caller, so the property the paper's cost counts is absent. See the
  section below.
- **Quantum k-medians rests on Grover's oracle model, and the oracle is not
  free.** The paper counts oracle calls, and the call it counts evaluates a
  distance in one step. This unit synthesizes its oracle from the predicate's
  truth table at `O(2**n)` cost, and it computes the whole distance table
  classically, one point at a time, before the circuit exists — the register is
  capped at three wires, which is also what keeps the distances out of it. No
  end-to-end advantage follows. See the section below.
- **Quantum kernel estimation rests on a data-access model it does not meet.**
  The kernel-matrix circuit's cost counts the swap tests: `O(eps**-2)` of them per
  entry, in the paper's own words, and therefore `O(m**2 / eps**2)` for an `m` by
  `m` matrix. It assumes the two feature states are already available, reached through
  a qRAM or an amplitude-encoding unitary whose cost the count does not include.
  Each feature state is built here gate by gate from the classical feature vector,
  so that cost is paid rather than assumed away. Two further statements are the
  literature's and are not softened here: the classical hardness of estimating
  these kernel entries is **a conjecture** in Havlíček et al., and the rigorous
  speed-up results for quantum kernel methods require a **fault-tolerant quantum
  computer** (Liu et al. 2021). The cited construction itself is written for
  noisy intermediate-scale devices and does not require fault tolerance. See the
  section below.
- **Feature selection carries no advantage premise of its own, and runs no
  solver either.** `feature_selection.py` builds a binary objective — a subset's
  relevance and pairwise redundancy plus a penalty on the subset's size — and
  evaluates it at an assignment, or maps it to an Ising Hamiltonian. The
  repository has no annealer, so which subset comes back, and at what cost,
  belongs to whatever solver the problem is handed to: this is a polynomial
  classical transformation, the position `qubo.py` occupies, and any advantage a
  caller observes belongs to the solver. The scores and the penalty weight are
  the caller's, and the unit computes no weight at which the target size binds.
  See the section below.
- **Frequent-item fractions rest on coherent database access, and that access is not
  exercised here.** `qarm.py` estimates the share of a database's items whose support
  meets a threshold, by amplitude estimation over a support register the circuit fills
  one controlled increment per transaction-item membership. The paper's count is a count
  of oracle calls that return one database entry each, and it reaches the candidate
  itemset superpositions it prepares through a qRAM; neither is present here, and the
  incidence matrix is read in Python to build that loop, so the access cost is paid
  rather than assumed away. **The improvement the paper claims is quadratic and
  conditional** — it is stated for the case `M_f^(k) << M_c^(k)`, and it is not
  exponential. See the section below.
- **Singular values rest on an input model this unit does not meet, and the
  dequantization is recorded beside it.** `svd.py` reads a matrix's singular values from
  the phase of its Hermitian embedding, and it builds that embedding's block encoding in
  private. The cited algorithm's cost is counted in queries to a structure that returns
  the matrix's entries, and against that count the state the estimation is applied to is
  assumed to be preparable; here the matrix is an ordinary tensor, its embedding and the
  embedding's exponential are dense classical objects, and the input state is built from
  the singular vectors that a classical `torch.linalg.svd` returns — the decomposition the
  readout itself estimates. **Tang's classical algorithm removes the exponential speed-up
  and is only polynomially slower** — its bound contains `eps**-12` — and that is not the
  same as a classical algorithm matching the quantum runtime. The block encoding is not
  free to read either: a readout that post-selects its ancilla costs `(||A|| / alpha)**2`
  at the most. See the section below.
- **The variational workflows the Hamiltonian feeds are heuristics.** The
  existing VQE and QAOA paths in [core.py](../../flagquantum/algorithms/core.py)
  are approximation heuristics. Their evidence and their boundary are the ones
  already recorded for local statevector execution; this guide adds no
  asymptotic advantage conclusion for them. **`variational.py` carries no
  advantage premise either, and it states two concessions rather than one.** The
  MaxCut solver is a heuristic over one ansatz and not an exact solver: a finite
  layer count is an approximation, so the energy it reports is an expectation
  and not a bound. And the uniform superposition — the state the first layer acts
  on — is an eigenstate of every cost and mixer term, so the objective's gradient
  at a zero start is **exactly zero** at every layer count, which makes the start
  a required argument instead of a default; a solver that defaulted to the origin
  would return the origin and call it an optimum. What the unit adds is not a
  speedup but an agreement: one checked edge list reaches both the circuit that
  rotates by it and the operator it is scored against, which the primitives
  beside it left to the caller. See the section below.
- **SPSA carries no advantage premise either, and what it offers is a cost.**
  `spsa.py` is not an algorithm with a speedup to state: it is an optimizer, and
  the quantity it improves is the number of objective evaluations per step, which
  is two regardless of the parameter count. **Its estimate is biased for every
  finite perturbation** and is not a gradient, so a caller with an exact gradient
  available should use autograd or parameter shift instead. The claim this unit
  supports is a cost claim on a stochastic objective, not an accuracy claim and
  not an end-to-end advantage. See the section below.
- **Nelder-Mead carries no advantage premise either, and its premise is the one
  SPSA cannot meet.** `nelder_mead.py` is the second optimizer here and it decides
  every step by comparing two objective values, so it needs the objective to be
  deterministic and it is defeated rather than merely degraded by a sampled one.
  It is also a local method whose `converged` flag reports that a simplex collapsed,
  not that the point it collapsed onto is a minimum. The claim it supports is a cost
  claim on a deterministic objective with no available derivative. See the section
  below.
- **The time-evolution unit carries no advantage premise, and the quantity that
  would bound it is not computed.** `trotter.py` is a compiler-facing
  construction: it turns a Pauli sum into the circuit that approximates
  `exp(-i t H)`, and nothing there is asymptotically better than exponentiating
  the dense generator at the sizes this repository simulates. The defect is real
  and stated — a product formula is exact only when the terms commute — but it is
  a number the caller measures rather than one the unit reports, because a bound
  on it needs a commutator norm the caller has and the unit does not. See the
  section below.
- **The error-mitigation units carry no advantage premise, because they are not
  advantage algorithms: what they rest on is an assumption about the noise.**
  `error_mitigation.py` measures one observable at several error strengths and
  continues the curve to zero, so it is unbiased exactly when that curve is a
  polynomial of degree at most the fitted order in the scale factor — an
  assumption nothing checks and nothing can check from the measurements alone.
  `pec.py` inverts the channel the noise model declares, at the location the
  model declares it, so it removes exactly the noise the model carries and
  leaves untouched every error the model does not: a miscalibrated gate, a
  leakage process, or a drift between the declaration and the run survives the
  combination. `cdr.py` fits the affine relation between what a training circuit
  should give and what the noise makes it give, so it is unbiased exactly when
  that relation holds over the region the training circuits span — again an
  assumption nothing checks, and one whose only diagnostic is the residual the
  fit left. All three units return a point value with no confidence interval, and
  none claims a variance reduction: `variance_amplification` and
  `sampling_overhead` are factors a **sampled** implementation of the same
  quantity would pay, computed from the weights rather than measured, and the
  paths here consume no shot at all. A mitigated value closer to the ideal is
  therefore evidence about the model before it is evidence about a device, and
  all three units say so where the number is. **None of the three is given a row
  in the index above or a source under Sources**: this programme has verified no
  citation record for any of the methods, and a `—` in that column already means
  "standard construction with no single paper to cite" rather than "sought and
  not yet found", so the absence is stated here instead of being spelled as
  something it is not. See the three sections below.
- **The folding unit carries no advantage premise either, and it rests on a
  different kind of assumption: one about the device rather than about a curve.**
  `folding.py` lengthens a program by an exact identity so that the noise its
  instructions carry runs more times, and the identity is exact for every unitary,
  so nothing is approximated and nothing is fitted. What it assumes is that
  repeating a gate repeats that gate's error and changes no other gate's — a
  statement about crosstalk, drift and calibration stability that no program
  carries and no measurement in the unit checks. **It also does not estimate the
  noise it amplifies**: what it returns is a length ratio, and the conversion from
  a length ratio to a noise strength is the caller's, taken from the same
  polynomial or Richardson assumption `error_mitigation.py` already states. It is
  the one mitigation unit that needs no noise model and no channel parameter, and
  the one whose curve is only as readable as its readout is sensitive, because its
  added gates land where the strategy puts them rather than where the observable
  is. Like the four beside it it has no row in the index above and no source under
  Sources. See the section below.
- **The adder carries no advantage premise either, and it is not a speed-up to
  state: it is a construction.** `arithmetic.py` builds Cuccaro's in-place
  ripple-carry addition, which is `2 n` Toffolis and `6 n + 1` `cx` for `8 n + 1`
  operations on `2 n + 2` wires, and it is *linear* in the width — the same
  asymptotics as every ripple-carry adder, and the same operation count whichever
  pair of addends is loaded, because the circuit is a permutation rather than a
  data-dependent schedule. What the unit is honest about is its price and its
  domain: **the T-count is not its own**, since `ccx` lowers through the
  compiler's fifteen-gate identity so the adder is priced at `14 n` T by
  [basis_translation.py](../../flagquantum/compiler/basis_translation.py) and
  [resource_estimation.py](../../flagquantum/compiler/resource_estimation.py)
  rather than by a seven-T expansion written beside it, and **two wires must
  enter holding `|0>`** — the working carry, which is restored, and the carry-out,
  which is left holding the top carry — so only inputs meeting that contract are
  promised a sum. Off that domain the circuit is still a permutation and never
  loses information, which is exactly why the map can be reversible without being
  addition there. The construction is Cuccaro's and not the smaller Gidney-Ekera
  one this repository's parity matrix names on the same row: Gidney-Ekera
  uncomputes its ancillas by measuring them and feeding the outcomes forward, so
  reaching it needs a mid-circuit measurement and a classical feedforward path
  that no costable circuit here provides. See the section below.

## A runnable example

The example builds the same three-variable problem that
[the QUBO tests](../../tests/unit/test_algorithms_qubo.py) exercise, converts
it, reads the QUBO form back, and evaluates a few assignments. Every import
comes from `flagquantum.algorithms.qubo`; nothing here needs a root-level name.

```python
from flagquantum.algorithms.qubo import (
    QuboProblem,
    ising_to_qubo,
    max_cut_qubo,
    qubo_energy,
    qubo_to_ising,
)

problem = QuboProblem(
    n_variables=3,
    linear={0: -1.0, 1: 0.5, 2: 2.0},
    quadratic={(0, 1): 3.0, (1, 2): -1.5},
)

hamiltonian = qubo_to_ising(problem)
print(len(hamiltonian.terms), [term.pauli for term in hamiltonian.terms])
# 6 ['Z', 'Z', 'Z', 'ZZ', 'ZZ', 'I']  -- one constant term, always emitted

recovered = ising_to_qubo(hamiltonian)
for assignment in ((0, 0, 0), (1, 0, 1), (1, 1, 1)):
    print(assignment, qubo_energy(problem, assignment), qubo_energy(recovered, assignment))
# (0, 0, 0) 0.0 0.0
# (1, 0, 1) 1.0 1.0
# (1, 1, 1) 3.0 3.0

graph = max_cut_qubo(((0, 1), (1, 2), (2, 0)), n_nodes=3)
best = min(
    qubo_energy(graph, (a, b, c)) for a in (0, 1) for b in (0, 1) for c in (0, 1)
)
print(best)
# -2.0  -- the negated maximum cut size of a triangle
```

The converted Hamiltonian is an ordinary `Hamiltonian`, so `vqe_loss`, `run_vqe`,
and `qaoa_loss` accept it without a new execution path. `qaoa_circuit` does not:
it takes a wire count and the cost edges as a list of ZZ pairs, with the QAOA
angles, and it is `qaoa_loss` that pairs such a circuit with a Hamiltonian. The
constant is carried as one identity term and recovered as `QuboProblem.offset`,
so the two forms of the same problem agree on every assignment.

## State preparation

`primitives/state_preparation.py` prepares a quantum state from a classical
amplitude vector. `uniform_state` is the Hadamard case; `arbitrary_state` builds
the state whose amplitudes are the normalised argument, and
`append_arbitrary_state` composes the same preparation onto a circuit that is
already carrying gates.

**The caveat is the point of the section.** Computing the rotation angles
requires a classical pass over all `2**n` amplitudes *and* a `2**n` by `2**n`
linear solve, so **the input is already exponential in size**. The unit shows
that a state can be prepared efficiently *given* its amplitudes. It does not
show that preparing a state is cheaper than the classical description of one,
and nothing here should be read as such a claim. The preparation is exact only
to the working precision of that solve.

```python
import torch

from flagquantum.algorithms.primitives import arbitrary_state, uniform_state

uniform = uniform_state(2).state().reshape(-1)
print([round(abs(complex(a)), 6) for a in uniform])
# [0.5, 0.5, 0.5, 0.5]  -- every amplitude has magnitude 1/sqrt(2**n)

amplitudes = torch.tensor([0.5, 0.5j, -0.5, 0.5], dtype=torch.complex64)
target = amplitudes / amplitudes.norm()
state = arbitrary_state(amplitudes, wires=[0, 1]).state().reshape(-1)
print(round(float(torch.abs(torch.vdot(target, state)) ** 2), 6))
# 1.0  -- fidelity with the target, to float32 precision

# The joint solve fixes the relative phases and leaves one global phase free, so
# the amplitudes agree with the target up to that single overall phase:
print([round(complex(a).real, 6) for a in state])
# [0.191342, 0.46194, -0.191342, 0.191342]
print([round(complex(a).imag, 6) for a in state])
# [-0.46194, 0.191342, 0.46194, -0.46194]
```

Wires are ordered most significant first, so `state[k]` is the amplitude of the
basis state whose bits read wire `0` to wire `n-1` from left to right.

## Oracle building blocks and synthesis

`primitives/oracle.py` holds the two pieces of reversible classical logic the
oracle units are composed from, and the truth-table synthesis that sits on top
of them. `append_multi_controlled_x` flips one target wire exactly on the operand
pattern that sets every control; one control is a `cx` and two are a `ccx`, and
three or more are built as an ancilla ladder, because the circuit layer has no
native gate above two controls.
`append_comparator` XORs one target wire with the truth value of `lhs > rhs`
for two equally wide bit strings read most significant first. On `n` bits the
comparator occupies `2n` operand wires, one target, `n + 1` prefix-equality
flags, and one scratch wire, so `3n + 3` in all. Neither unit makes an
advantage claim: both are reversible classical logic of `O(n)` Toffoli-style
cost, and the classical predicate they compute is the cost they pay.

**The ancilla precondition is load-bearing.** Above two controls the caller
must supply `len(controls) - 2` ancillas, and each one must be in `|0>` on
entry. Measured with a dirty ancilla, the target comes out wrong on a large
fraction of the operand patterns — 8 of 16 at three controls, 32 of 96 at four
— and nothing is raised. The ancilla's own value is left unchanged by the
ladder, so the fault cannot be seen from the ancilla either. A circuit builder
cannot read a wire's starting value, so meeting the precondition is the
caller's to do.

**The comparator exits clean.** Every wire it is given comes back to the value
it entered with, except the target, which is XORed with `[lhs > rhs]`. The
prefix-equality ladder and the scratch wire are uncomputed, so the comparator
composes into a larger circuit instead of leaving `n` wires holding
intermediate flags. Its internal three-control X is
`append_multi_controlled_x` with the scratch wire as the ladder's ancilla,
which is the same `|0>`-on-entry requirement one level down. **The comparator's
own flags and scratch wire carry that requirement too**: all `len(lhs) + 1`
equality wires and the scratch wire must enter in `|0>`, since a dirty
`equality[0]` makes the target come out wrong on a large fraction of the operand
patterns — 6 of 16 at two bits — and the ladder then restores the flag, so
nothing is raised.

```python
from flagquantum.algorithms.primitives import append_comparator, append_multi_controlled_x
from flagquantum.circuit import Circuit

# Three controls need an ancilla, and it must enter in |0>:
circuit = Circuit(5)
for wire in (0, 1, 2):
    circuit.gate("x", wire)
append_multi_controlled_x(circuit, [0, 1, 2], 3, ancillas=[4])
print(format(circuit.state().reshape(-1).abs().pow(2).argmax().item(), "05b"))
# 11110  -- the target (wire 3) flipped and the ancilla (wire 4) came back to |0>

# The comparator XORs its target with [lhs > rhs] and restores every other wire:
circuit = Circuit(9)
for wire in (0, 3):  # lhs = 10, rhs = 01
    circuit.gate("x", wire)
append_comparator(
    circuit, lhs=[0, 1], rhs=[2, 3], target=4, equality=[5, 6, 7], scratch=8
)
print(format(circuit.state().reshape(-1).abs().pow(2).argmax().item(), "09b"))
# 100110000  -- lhs, rhs and the comparison bit, with the flags and scratch back at 0
```

**Truth-table synthesis sits on top of those blocks.** `marked_states` lists the
inputs a predicate marks; `phase_oracle` and `bit_oracle` build a standalone
circuit for one, and `append_phase_oracle` / `append_bit_oracle` append the same
construction to a circuit the caller already has. Each marked input is mapped
onto the all-ones pattern with `x` gates, acted on by the multi-controlled X
above, and mapped back, so the phase oracle multiplies exactly the marked
amplitudes by `-1`, and the bit oracle carries the predicate's value onto an
output wire by XOR and restores every wire it allocated.

**The cost is exponential and no advantage follows.** Synthesis enumerates all
`2**n` inputs of the predicate classically, so a gate-level oracle written this
way is as expensive as the classical search it is meant to replace; whatever a
query-model algorithm saves on queries, it pays again here, which is why the
index row claims nothing. `phase_oracle` is also capped at three wires: its
multi-controlled Z is a multi-controlled X with `n - 1` controls, and a
standalone circuit of exactly `n` wires has no wire to spare for the `n - 3`
ladder ancillas a wider one needs, so above three wires the caller supplies the
register and the ancillas through the append form. `bit_oracle` has no such cap,
because it allocates its own ladder ancillas and restores them.

## Grover search

`grover.py` is the first algorithm unit in this index: it consumes the oracle
primitives rather than extending them. `grover_circuit` puts the evaluation
register into the uniform superposition with one Hadamard per wire, then applies
`optimal_iterations` rounds of the phase oracle followed by the diffusion
operator, the reflection about the uniform superposition. `run_grover` samples
that circuit and ranks the states it marks by how often the sample landed on
them, and `optimal_iterations` returns the
`floor(pi / (4 * asin(sqrt(m / 2**n))))` rounds that leave `m` marked states out
of `2**n` with the largest total amplitude.

**The oracle is where the cost sits.** Grover's result is a query-count result,
and the oracle it counts is the truth-table form from the section above, which
enumerates all `2**n` inputs classically. The query count improves; the oracle's
own construction does not, so the index row claims no end-to-end advantage, and
nothing here should be read as one.

**The register is the whole circuit, and that bounds it at three wires.** The
diffusion operator's multi-controlled Z is a multi-controlled X with `n - 1`
controls, and above two controls that is an ancilla ladder needing `n - 3`
ancillas in `|0>`. A circuit that *is* the evaluation register has no free wire
for one, so `grover_circuit` refuses `n_wires > 3` rather than allocating it: a
wire added to this circuit is part of the register the samples are read over, so
the "ancilla" would be sampled along with the answer. A wider search is built
from `append_phase_oracle` on a register and ancillas the caller lays out.

```python
from flagquantum.algorithms.grover import run_grover

result = run_grover(lambda value: value == 5, 3, shots=1024, seed=0)
print(result.candidates, result.iterations, round(result.success_probability, 4))
# (5,) 2 0.9395

# Counts are keyed by the big-endian bit string, one character per wire:
print(result.counts[format(5, "03b")])
# 962
```

## Amplitude estimation

`amplitude_estimation.py` is the second algorithm unit in the index, and the last
of the Phase 1 set. It consumes the phase estimation primitive and the Grover
iteration rather than extending either: `amplitude_estimation_circuit` prepares the
operator's own register and hands the controlled Grover operator
`Q = -A (I - 2|0><0|) A† S_chi` to `append_phase_estimation`, which emits the
counting register's Hadamards, the controlled powers and the inverse Fourier
transform itself. `run_amplitude_estimation(operator, n_counting_wires=...,
shots=..., seed=...)` samples that circuit and returns the maximum likelihood grid
point, and `amplitude_resolution` reports the widest step of the grid
`sin²(pi j / 2**(m+1))`.

**The readout is a phase, not an amplitude.** The two eigenphases of `Q` are
conjugate, so a counting outcome is a phase and the amplitude behind it is
`sin²(theta)`: reading the register's value as an amplitude is a wrong answer
rather than an error. `maximum_likelihood_estimate` inverts the two-term model
both eigenphases produce, and because sample keys carry every wire it marginalises
the evaluation register's bits away first; skipping that step loses their whole
share of the probability mass and returns a wrong estimate without raising.

**The estimate is a resolution, not a confidence interval.** An estimate is always
one of the grid amplitudes, so its error is at most about one grid step at the
counting width used. That is a property of the grid, not a coverage-calibrated
error bar: no interval is computed or reported, and
`AmplitudeEstimationResult.within` checks the resolution and says so.

**The premise is `A`, and it is not free.** The quadratic speedup over classical
sampling counts applications of the state-preparation unitary and its adjoint;
here the caller supplies that unitary and the count assumes it costs nothing, and
a real distribution would need QRAM to be loaded in. Nothing in this unit shows
that a Monte Carlo integral is estimated faster than classically, and the
capability entry repeats the boundary.

## Quantum PCA

`pca.py` is the first unit of Phase 2 and the third algorithm unit in this
guide's programme. `principal_components` takes the data matrix `A`, the keyword
`n_counting_wires`, and the sampler's `shots` and `seed`: it forms the density
matrix `rho = A A^T / tr(A A^T)` of a real `A` with at least two rows and two
columns, each a power of two, prepares the purification `vec(A) / ||A||_F` on the
data and purification registers with `append_arbitrary_state`, and hands
`exp(-2 pi i rho)` to `append_phase_estimation`. The counting register's marginal
is the eigenvalue distribution: `PcaResult` carries it, with the eigenvalue read out
at the distribution's mode, that value's measured share, and the resolution the
register achieves. The readout is the mode's counter value; whether that is the
largest eigenvalue is an outcome of the shares rather than a rule — see *Which
eigenvalue the mode reports* below.

**The phase is not the eigenvalue.** At `t = 2*pi` the exponential's eigenphase on
an eigenvector of `rho` with eigenvalue `lambda` is `exp(-2 pi i lambda)`, which is
the phase `phi = (-lambda) mod 1`, so a counter value `k` of `m` counting wires
reads `lambda = 1 - k / 2**m`. Reading `k / 2**m` is a wrong eigenvalue rather than
an error, which is why the inversion is part of the readout and has a test of its
own: with it removed, the two tests that compare the readout against
`torch.linalg.eigvalsh` of the density matrix fail.

**A share is not an eigenvalue.** The counter value nearest a small eigenvalue
carries the dominant peak's phase-estimation tail rather than that eigenvalue's
weight. Measured at six counting wires: the eigenvalue `0.0038` comes back on its
own counter value with `0.0331` of the sample, about nine times its weight.
`PcaResult.within` states the resolution contract as half a counter step, and no
confidence interval is computed or reported.

**Which eigenvalue the mode reports is not something the unit predicts.** The mode is
the counter value with the largest share, and the readout is that counter value's
eigenvalue, `1 - k / 2**m`. The unit does not claim that this is the largest
eigenvalue and does not predict which eigenvalue it will be: that depends on how each
eigenvalue's weight is spread across the counter values the register has, set against
the other eigenvalues' shares at theirs, and the unit computes neither. `within`
answers only what it says — whether a named eigenvalue lies within half a counter
step of the readout — so a caller who needs the largest eigenvalue specifically has
to check the readout, and cannot infer it from the counter width or the shape of the
peak.

Two runs illustrate two outcomes. They are examples, not a criterion. Both use
`rho = diag(spectrum)`, six counting wires, `shots=8000` and sampling `seed=1`, and
both have the same largest eigenvalue, `0.51015625`, at phase `31.35/64`, so the
largest eigenvalue and the trace are not what differs between them. In each, the
second largest eigenvalue also sits exactly on a counter value — a different one,
because moving that eigenvalue's weight moves its phase with it.

- Spectrum `[0.51015625, 0.296875, 0.096484375, 0.096484375]`. The mode is counter
  31, reading `0.515625`, with share `0.3385`, and `within` accepts `0.51015625`.
  The largest eigenvalue's two counter values, 31 and 32, carry `0.3385` and
  `0.0973`; the second largest eigenvalue sits at counter 45 with `0.2981`. The
  mode's counter value is one of the largest eigenvalue's, although the other one
  carries the smallest share of the three.
- Spectrum `[0.51015625, 0.390625, 0.049609375, 0.049609375]`. The mode is counter
  39, reading `0.390625`, with share `0.3889`, and `within` rejects `0.51015625`.
  Counter 39 is where the second largest eigenvalue sits; the largest eigenvalue's
  two counter values carry `0.3385` and `0.0973`.

Both readouts held for every sampling seed from 0 through 299 at each of 4096, 8000,
20000 and 50000 shots, and for seeds 0 through 4999 at 8000 and 20000. That is what
those runs did, and not a claim that either readout is independent of the sampler's
noise. Where the margin is thin enough it is not: the first example's leading counter
values are separated by only about `0.004` at 4096 shots, and six of the first 5000
sampling seeds at that width put the mode on the second largest eigenvalue, the
narrowest of the six by `0.000244`.

```python
import math

import torch

from flagquantum.algorithms.pca import principal_components

# A data matrix whose density matrix has eigenvalues 0.9962 and 0.0038.
data = torch.diag(torch.tensor([math.sqrt(0.9962), math.sqrt(0.0038)]))
rho = (data @ data.T) / torch.trace(data @ data.T)
print([round(float(value), 4) for value in torch.linalg.eigvalsh(rho)])
# [0.0038, 0.9962]

result = principal_components(data, n_counting_wires=6, shots=20000, seed=11)
print(round(result.dominant_eigenvalue, 4), round(result.dominant_probability, 4))
# 1.0 0.8196  -- the mode's readout; here it is the dominant eigenvalue, read off counter 0
print(round(result.distribution["111111"], 4))
# 0.0331  -- the counter value nearest 0.0038, and this share is the tail above
```

**The premise is the input model, and this unit does not meet it.** The paper's
subroutine consumes copies of `rho`, one use per run, in `O(1/eps**3)` of them, and
never forms it; that count is what its speedup would be measured against. This unit
forms `rho` classically from the whole matrix `A`, builds `exp(-2 pi i rho)` with
`torch.matrix_exp`, and passes the result to the circuit as a dense gate matrix, so
no part of the paper's input model survives here. The purification inherits a second
premise: `append_arbitrary_state` solves for its angles with a classical pass over
all `2**n` amplitudes and a `2**n` by `2**n` linear solve, so the caller must
already hold the entire amplitude vector. Nothing in this unit is faster, or
smaller, than diagonalising `rho` with `torch.linalg.eigvalsh` directly, and the
capability entry repeats the boundary.

## Quantum k-medians

`kmedians.py` is the second unit of Phase 2 and the fourth algorithm unit in
this guide's programme. `kmedians(points, centroids, shots=..., seed=...)` takes
the points, the centroid positions, and the sampler's `shots` and `seed`, and
performs one assignment step of k-medians: `KMediansResult` carries the centroid
each point was assigned to, the updated centroid positions, and the number of
Grover searches the run sampled. Each point's nearest centroid is found by
`grover.run_grover` over a register holding a centroid index, and the centroid
update is classical arithmetic. **The quantization is the assignment step and
only the assignment step**: the search is a Grover-style minimum search, not a
distance measured in superposition.

**The search is a minimum search over a moving threshold.** One point's
assignment is a loop, not a single circuit: it holds the index of the best
centroid found so far, marks the indices whose pair of distance and index is
strictly smaller than the held one's, and moves to one of the marked indices
when the sample found one. A round that finds none ends the loop, and the index
it holds is the assignment. Each moving round strictly decreases a pair drawn
from a finite set, so a point costs at most as many searches as there are
centroids, and a point whose nearest centroid is index 0 costs exactly one.

**Ties are broken by the index, inside the search rather than after it.** The
comparison is on the pair of distance and index, so a centroid at exactly the
threshold distance is marked only when its index is the smaller one. Measured on
a point at `(3, 3)` against centroids at `(0, 0)`, `(2, 0)` and `(0, 2)`, whose
distances to the last two are exactly equal at `3.1622776601683795`: the
assignment was index `1` for every sampling seed from 0 through 199 at 1024
shots. The same instance with the index dropped from the comparison returns
index `2` at seed 0 and both tied indices across those seeds, because which tied
centroid the sample favours is then the assignment — which is why the index is
part of the order and not a patch on the result.

**That is a statement about the comparison, not about every run.** The loop ends on
the first round whose sample finds none of the indices that round marked, so the
rule decides the assignment under one condition: no round of the point's loop
misses the indices it marked. Under it the assignment is the smallest pair of
distance and index in the table — the nearest centroid, and the lower-indexed of
two that are exactly as near as each other. A run with a round that misses ends
on an index it has not finished improving. Measured on a point at `(2.5,)`
against centroids at `(0,)`, `(1,)`, `(2,)` and `(3,)`, whose distances to the
last two are exactly equal at `0.5`, over seeds 0 through 199: 102 of the 200
runs had no round that missed a marked index at one shot and every one of those
102 ended on index `2`; at four shots the counts are 186 and 186; at eight, 198
and 198; and at 16, 64 and 1024 shots all 200 runs had no missed round and every
one ended on index `2`. The runs that did miss ended on index `1` or, at the
smallest widths, on index `0`, which is a centroid farther away than either of
the tied ones.

**The rule is stated for a pair.** `kmedians.py` makes no general claim about
three or more centroids at exactly the same distance: the marked sets it
compares are not a pair there, and the measured tables above are all two-way.
What the tests pin is one such instance, and it is an instance rather than a
rule: a point at `(1, 1)` with centroids at `(0, 3)`, `(3, 0)` and `(2, -1)`,
whose three distances are all exactly `2.23606797749979`, is assigned index `0`,
because no centroid beats the one the loop starts from and that first round
marks nothing.

**The distance table is classical, and that is the whole of what the unit gives
up.** `grover_circuit` builds its own register and refuses more than three
evaluation wires, so the search runs over the centroid index alone and the
distance from the point being assigned to each centroid is computed in double
precision outside the circuit — the predicate closes over that table, and every
round builds its register afresh. The circuit holds no state about the
points or the centroids: what it searches is a table the classical caller built,
which is precisely the cost the paper's oracle model assumes away.

**The median update is classical, and two of its cases have to be named.** Each
coordinate of a centroid is set to the median of that coordinate over the points
assigned to it. A cluster with an even number of points has a range of medians
rather than one, and this unit takes the lower of the two, which is what
`torch.median` returns. A cluster with no points has no median at all, and its
centroid keeps the position it was given. Both in one measured run: points
`(0,)`, `(1,)`, `(2,)` and `(10,)` against centroids `(0,)` and `(50,)` give the
assignment `(0, 0, 0, 0)`, the first centroid moving to `1.0` — the lower middle
of `0, 1, 2, 10` — and the second keeping `50.0`.

**The search is sampled, not read out, and the sample size is visible.** A round
ends when its sample found no marked index, so a sample that missed a marked
index ends a point's loop early, at an index that is not the nearest centroid.
Measured on two points at `(7, 0)` and `(4, 0)` against eight centroids at
`(0, 0)` through `(7, 0)`, whose nearest centroids are index 7 and index 4: the
assignment differed from the classical labelling for 43 of the 50 sampling seeds
tried at one shot, for 23 at two shots, for 8 at four, and for none at 16, 64 or
1024 shots. That is a measurement at those seeds and those shot counts, not a
guarantee: the tail a small sample leaves belongs to the sampler, and the module
computes and reports no error bound, confidence interval or repetition scheme.
The default is 1024 shots per search.

```python
import torch

from flagquantum.algorithms.kmedians import kmedians

points = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [5.2, 0.1], [0.2, 1.5], [4.9, -0.4]],
    dtype=torch.float64,
)
centroids = torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 2.0]], dtype=torch.float64)

result = kmedians(points, centroids, shots=1024, seed=7)
print(result.labels)
# (0, 0, 1, 1, 2, 1)  -- each point's nearest centroid
print(result.medians)
# ((0.0, 0.0), (5.0, 0.0), (0.2, 1.5))  -- the coordinate-wise medians of the three clusters
print(result.searches)
# 10  -- ten searches sampled: six points, and four rounds that moved
```

**The premise is the oracle model, and this unit does not meet it.** The paper
counts oracle calls, and in that model the distance function has to be computed
into a register — a cost the count does not include. Here the oracle is
synthesized from the predicate's truth table, so it costs `O(2**n)` — over the
at most eight register values the three-wire search can carry — and the distance
table the predicate compares is computed classically, one point at a time,
before any circuit is built. Neither cost is in the query count, and nothing
here reads a qRAM or runs an adiabatic evolution, so no conclusion that rests on
either applies. The unit is bounded at eight centroids by the three-wire
register, and the search is sampled rather than read out, so a small sample can
stop a point's search short. The capability entry repeats the boundary.

## Quantum kernel estimation and kernel ridge classification

`quantum_kernel.py` is the third unit of Phase 2, and the fifth algorithm unit
in this guide's programme. `quantum_kernel_matrix(data, shots=..., seed=...)`
takes a real matrix of feature vectors and returns a `KernelMatrixResult` whose
`matrix` holds one estimated entry per pair;
`kernel_ridge_classifier(training_data, labels, regularization=..., shots=...,
seed=...)` returns a `KernelRidgeClassifier` carrying the dual coefficients of a
ridge-regularised kernel regression, the training set they were fitted on, and
`decision_function` and `predict` for new rows.

**The kernel is the only quantum part.** An entry is the squared overlap of two
feature states, estimated with the swap test — the kernel-matrix circuit of
Havlíček, Córcoles, Temme, Harrow, Kandala, Chow and Gambetta, "Supervised
learning with quantum-enhanced feature spaces", *Nature* **567**, 209–212
(2019), DOI 10.1038/s41586-019-0980-2. A Hadamard on the ancilla, a native
controlled swap per wire pair, and a Hadamard again leave the ancilla found set
with probability `1/2 - 1/2 |<a|b>|^2`, so the module estimates the **squared**
overlap and reads `|<a|b>|^2 = 1 - 2 * share` back out of the sample.

**The readout is sign-blind by construction, and the tests check it as a
contract rather than assume it.** One state, measured against three partners
that differ only in a relative phase, has the overlaps `+1/sqrt(2)`,
`-1/sqrt(2)` and `+i/sqrt(2)` — all three of the same magnitude, and all three
the same experiment to this circuit, so all three must return the same
statistic. The magnitude assertion is what catches a sign-carrying
implementation, first and on its own: its reading of the first partner is
`0.707` against the `0.5` the magnitude is asserted at. The equality across the
three is a second net, for an implementation whose three readings all land
inside that tolerance but differ from each other. A test built on the
**identical-states pair alone** could not tell a sign-carrying implementation
from a correct one — both paths read `1` there — which is why the triple is the
check the unit carries.

**The feature map is this package's own angle encoding, and not the cited
paper's.** A Hadamard on every wire, a phase rotation carrying each feature on
its own wire, and an entangling phase rotation on every pair whose angle is the
product of the two features' complements to `pi`, with features in `[0, 2*pi]`:

    |Phi(x)> = prod_{j<k} exp(i (pi - x_j) (pi - x_k) Z_j Z_k)
               prod_k exp(i x_k Z_k) H^{tensor n} |0...0>

The map is chosen to be small, concrete and classically computable, which is
what makes the independent reference path in the tests possible: the tests sum
`<Phi(x)|Phi(z)>` over the `2**n` basis states with the encoding's own phase and
compare every entry against it. Because the map is this module's and not
Havlíček et al.'s, their hardness conjecture says nothing about this unit, and
nothing here should be read as though it did.

**One kernel entry costs `O(eps**-2)` shots, in the paper's own words, and an
`m` by `m` kernel matrix therefore costs `O(m**2 / eps**2)`.** The per-entry
figure is the paper's; the matrix figure is that one multiplied by the `m**2`
entries, and not a second figure the paper states. Nothing here converts either
into an accuracy: no error bound, no confidence interval, no shot-selection rule
and no repetition scheme is computed or reported. The estimate is
`1 - 2 * share` with `share` in `[0, 1]`, so an entry lies in `[-1, 1]` by
construction and one whose overlap is near zero can come back slightly
negative; the readout is reported as it comes and is not clamped.

**The classifier is classical, and its fit inherits the sample.**
`kernel_ridge_regression` solves `(K + lambda I) alpha = y` for the dual
coefficients and `predict` returns the sign of the resulting decision function,
a value at or above zero reading `+1`. The ridge weight is a required argument
with no default anywhere in the module: it is what keeps the training system
solvable when the kernel matrix is close to singular, and how closely the fit
should follow the training targets is the caller's choice rather than the
module's. A point whose decision value is small can be decided differently by a
different sample, and no margin, bound or accuracy estimate is computed.

**Wire layout and scale.** One kernel entry is one circuit: the ancilla on wire
0, the left feature state on the next `n` wires and the right one on the last
`n`. The unit is bounded at three features, and the matrix is symmetric by
construction — the swap test of `(i, j)` and of `(j, i)` is the same experiment,
so the entry is sampled once and mirrored rather than sampled twice. The
diagonal is sampled like every other entry and comes back exactly one. That is
the circuit's arithmetic: the overlap of a state with itself is one, so the
circuit never finds that ancilla set.

```python
import torch

from flagquantum.algorithms.quantum_kernel import (
    kernel_ridge_classifier,
    quantum_kernel_matrix,
)

train = torch.tensor(
    [[0.4, 0.5], [1.2, 1.7], [4.6, 1.1], [5.4, 1.9]], dtype=torch.float64
)
labels = (1, 1, -1, -1)

matrix = quantum_kernel_matrix(train, shots=4096, seed=5)
print([round(value, 3) for value in matrix.matrix[0]])
# [1.0, 0.419, 0.218, 0.306]  -- the first row; the exact values are 1.0, 0.4199, 0.2037, 0.3042
print([round(value, 3) for value in matrix.matrix[1]])
# [0.419, 1.0, 0.514, 0.138]  -- symmetric, and every diagonal entry is exactly 1

classifier = kernel_ridge_classifier(
    train, labels, regularization=1e-2, shots=4096, seed=5
)
print(classifier.predict(train, shots=4096, seed=6))
# (1, 1, -1, -1)  -- the training labels, recovered by the sign of the decision function
print([round(c, 3) for c in classifier.coefficients])
# [1.036, 1.609, -1.768, -1.095]  -- the dual coefficients of the sampled kernel
print(
    [
        round(value, 3)
        for value in classifier.decision_function(
            torch.tensor([[1.3, 1.8], [4.5, 1.2], [0.5, 0.6]]), shots=4096, seed=7
        )
    ]
)
# [1.451, -1.034, 0.769]  -- held-out decision values; the exact-kernel values are
# 1.394, -0.996 and 0.667, so every sign agrees
```

**The premise is the data-access model, and this unit does not meet it.** The
kernel-matrix circuit's cost counts the swap tests, not the data access: the two
feature states are assumed to be available, reached through a qRAM or an
amplitude-encoding unitary whose cost the count does not include. Here each
feature state is built gate by gate from the classical feature vector on every
run, so that cost is paid rather than assumed away, and no end-to-end advantage
follows. The classical hardness of estimating these entries is a conjecture in
Havlíček et al. and not a theorem, and the rigorous speed-up results for quantum
kernel methods require a fault-tolerant quantum computer (Liu et al. 2021) —
which the cited construction does not, being written for noisy
intermediate-scale devices. The capability entry repeats the boundary.

## Feature selection as a QUBO

`feature_selection.py` is the fourth unit of Phase 2, and the sixth algorithm unit
in this guide's programme. `feature_selection_qubo(relevance, n_selected=...,
penalty=..., redundancy=...)` takes one relevance score per feature, an optional
pairwise redundancy, a target number of features and a penalty weight, and returns
a `FeatureSelectionProblem`: the objective as a `QuboProblem`, with `energy` to
evaluate it at an assignment and `to_ising` to map it to a `Hamiltonian`.

**The objective is a subset's scores plus a penalty on its size.** With `r_i` the
relevance of feature `i` and `s_ij` the redundancy of the pair `(i, j)`, a subset
`S` is scored

    F(S) = - sum_{i in S} r_i + sum_{i < j, both in S} s_ij
           + penalty * (|S| - n_selected) ** 2

and the QUBO carries it with the squared size term expanded. The diagonal of that
square is where a binary variable's own square goes -- `x_i ** 2 = x_i` -- so the
penalty reaches the linear coefficient map as well as the quadratic one and the
offset.

**The unit does not solve, and the repository has no annealer.** What it does is
build the objective of one instance and evaluate that objective at an assignment:
`energy` is `qubo_energy` on the carried problem and `to_ising` is `qubo_to_ising`,
so neither the evaluation nor the mapping is new here. Which subset a solver
returns, and at what cost, belongs to the solver the problem is handed to, and any
advantage such a solver observes is the solver's. The module says this before it
describes the objective it builds, and the position is `qubo.py`'s: a polynomial
classical transformation with no advantage of its own.

**The penalty weight is the caller's decision.** There is no default for it
anywhere in the module, and no weight at which the target size starts to bind is
computed or predicted: a weight small enough against the scores can leave a subset
of another size cheapest, and how far the size term should outweigh them is the
caller's choice rather than the module's. The weight must be positive and finite,
because the term it multiplies exists to price a subset's size.

**The scores are the caller's data.** This unit defines no relevance measure and no
redundancy measure, and puts no interpretation on either. The diagonal of a
redundancy matrix is not read -- a pair is two features -- and the two triangles
must agree exactly, because `(i, j)` and `(j, i)` are one pair with one score.

```python
import torch

from flagquantum.algorithms.feature_selection import feature_selection_qubo

relevance = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6], dtype=torch.float64)

problem = feature_selection_qubo(relevance, n_selected=2, penalty=5.0)
print(problem.qubo.linear)
# {0: -16.0, 1: -15.9, 2: -15.8, 3: -15.7, 4: -15.6}
#   -- -relevance[i] + 5.0 * (1 - 2 * 2): the size penalty's linear term and the score
print(problem.qubo.offset)
# 20.0  -- penalty * n_selected ** 2
for assignment in ((1, 1, 0, 0, 0), (1, 1, 1, 1, 1)):
    print(assignment, round(problem.energy(assignment), 3))
# (1, 1, 0, 0, 0) -1.9   -- the two most relevant features, at the target size
# (1, 1, 1, 1, 1) 41.0   -- all five, where two were asked for
print([term.pauli for term in problem.to_ising().terms].count("ZZ"))
# 10  -- one quadratic term per pair of features

loose = feature_selection_qubo(relevance, n_selected=2, penalty=0.05)
for assignment in ((1, 1, 0, 0, 0), (1, 1, 1, 1, 1)):
    print(assignment, round(loose.energy(assignment), 3))
# (1, 1, 0, 0, 0) -1.9
# (1, 1, 1, 1, 1) -3.55  -- at a weight this small the size term does not bind
```

**The scale figure is the paper's own, and not a measurement here.** Feature
selection posed as a binary objective for an annealer is the route of Ferrari
Dacrema, Moroni, Nembrini, Ferro, Faggioli & Cremonesi, SIGIR 2022, DOI
10.1145/3477495.3531755, arXiv:2205.04346 -- **that paper's own largest problem
solved directly on the QPU had 124 features**, which is the paper's figure and not
something this unit does or measures. What is built here is this package's own
spelling of the objective, and this unit neither hands its problem to a solver nor
observes one.

## Frequent-item fractions by amplitude estimation

`qarm.py` is the fifth unit of Phase 2, and the seventh algorithm unit in this guide's
programme. Association rule mining asks which itemsets recur across a database's
transactions, and the amplitude-estimation route to it is the one of Yu, Gao, Wang and
Wen, *Physical Review A* **94**(4), 042311 (2016), DOI 10.1103/PhysRevA.94.042311,
arXiv:1605.07444v3. `frequent_itemset_operator(database, threshold=...,
n_support_wires=...)` builds the amplitude operator of one database and threshold, and
`run_frequent_itemset(database, threshold=..., n_counting_wires=..., shots=...,
seed=...)` hands it to `amplitude_estimation.run_amplitude_estimation` and returns that
function's result unchanged: the estimate it carries is the fraction of the items whose
support meets the threshold. **This unit asks a smaller question than the paper's.** The
paper mines the frequent itemsets; what is read out here is one number, the fraction of
the items that are frequent, and no rule-mining stage is part of the module.

**The circuit, register by register.** The item register carries one wire per item-index
bit and is put into a uniform superposition by a half-turn `ry` on each of them — the
state a Hadamard on every item wire prepares, chosen because `ry` has a native controlled
form and the Grover operator has to control the preparation. The support register is then
filled by the transaction loop: one controlled increment per transaction whose itemset
holds the item in superposition, so the register ends holding that item's own support.
The marking operator flips the phase of the subspace whose support register holds a value
at or above the threshold, so the marked subspace is exactly the frequent items, and its
amplitude is the frequent fraction. Amplitude estimation reads that amplitude off the
counting register, on the grid and at the resolution the amplitude estimation unit
documents.

**The transactions are iterated classically, and that is the unit's headline
limitation.** The paper's speed-up is measured in calls to an oracle that returns one
database entry per call, and it reaches the candidate itemset superpositions it prepares
through a qRAM; neither is present here. The loop that fills the support register is a pass
over the incidence matrix in Python, one controlled increment per transaction-item
membership, so the database is walked entry by entry outside the circuit and the access
cost is paid rather than assumed away. **No end-to-end advantage follows**: that loop is
exactly the part of the paper's assumption its speed-up is measured against.

**The comparison at the threshold is inclusive.** An item whose support equals the
threshold is frequent, and the marking operator is built over the support values at or
above it. The boundary is reachable and it changes the answer: the measured database
below has an item whose support is exactly 2, so a threshold of 2 counts it and the
fraction is one half, where a strict comparison would mark nothing and read zero. A
threshold below one, or above the transaction count, is refused rather than run: every
item meets it, or none does, and neither needs a circuit.

**The support register has to be wide enough, and a narrow one is refused.** The
increment is a permutation of the register's own values, so a support the register
cannot hold comes back as another value, the mark then sees a support that can be on the
wrong side of the threshold, and the readout can come back wrong with nothing raised.
Measured on the database below with the support register narrowed to a single wire, a
width the builder refuses, and with that refusal lifted for the measurement: the first
item's support is 2 and comes back as 0, the estimate reads 0.0 against an exact fraction
of 0.5, and nothing is raised. `frequent_itemset_operator` therefore refuses a register
that cannot hold the largest support any database of that transaction count could
produce, and the default is the fewest wires that can hold it.

**The item count is a power of two.** The item register is addressed by one wire per
item-index bit, and a uniform state over another count has no controlled preparation in
this package, which is what the Grover operator needs. The unit is bounded at eight items
and seven transactions.

**Measured.** Two transactions over two items, `[[1, 0], [1, 1]]`, at a threshold of 2:
the supports are 2 and 1, so the exact frequent fraction is one half.

```python
import torch

from flagquantum.algorithms.qarm import frequent_itemset_operator, run_frequent_itemset

database = torch.tensor([[1, 0], [1, 1]])

operator = frequent_itemset_operator(database, threshold=2)
print(operator.support, operator.n_support_wires, operator.n_wires)
# (2, 1) 2 5
#   -- the column sums, the register's width, and the evaluation register's

result = run_frequent_itemset(database, threshold=2, n_counting_wires=4, shots=8000, seed=17)
print(round(result.estimate, 6), round(result.resolution, 6))
# 0.5 0.097545  -- the readout, against an exact fraction of 0.5

# The support register has to hold the largest support any database of this transaction
# count could produce, so a narrower one is refused rather than left to wrap into a
# readout that can come back wrong.
try:
    frequent_itemset_operator(database, threshold=2, n_support_wires=1)
except ValueError as error:
    print(error)
# the support register needs at least 2 wires to hold the largest support 2 that 2
# transactions can produce, got 1; the increment is a permutation of the register's
# values, so a narrower register wraps a support into another value and the readout can
# come back wrong with nothing raised
```

A second database reaches fractions the two-transaction one cannot. Four transactions over
four items, `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]]`, has supports 3,
2, 2 and 1, so its frequent fraction is `3/4` at a threshold of 2 and `1/4` at a threshold
of 3. Measured at four counting wires, 8000 shots and seed 17, the readouts are `0.777785`
and `0.222215` — the grid values `sin²(11 pi / 32)` and `sin²(5 pi / 32)`, each within the
register's resolution `0.097545` of the exact fraction. Both are what a sampled grid
estimate gives, not a claim that the estimate is exact.

**The premise, and what this unit does not show.** The cited paper's improvement is
quadratic in the number of database queries, and it is stated conditionally, for the case
`M_f^(k) << M_c^(k)`; it is not exponential. An earlier arXiv listing of the same work is
the only place that wording appears, and nothing in this guide repeats it. Because the
paper's oracle and its qRAM are both absent here, no part of its query-count advantage
survives into this unit.

## Singular values by phase estimation

`svd.py` is the sixth unit of Phase 2, and the eighth algorithm unit in this guide's
programme. `estimate_singular_values(A, n_counting_wires=..., shots=..., seed=...)` forms
the Hermitian embedding `[[0, A], [A^T, 0]]` of a real square matrix, exponentiates it,
phase-estimates a counting register against that exponential, and returns a
`SingularValueResult`: the singular value read out at the register's mode, that value's
share of the sample, the register's marginal distribution, the step the register resolves,
and the subnormalisation the readout is scaled by. The counting width is the caller's and is
at least one, and one wire is degenerate rather than a second contract: the register then
holds two counter values, of which one is refused, so the only value it can return is
`alpha` itself. `A`'s singular values are the magnitudes
of the embedding's eigenvalues — the embedding carries `A` and `A^T` as its two
off-diagonal blocks, and its spectrum is `±sigma_i` — so what the readout inverts to is the
magnitude of one of the embedding's eigenvalues, which is one of `A`'s singular values.

**The phase is not the singular value.** The exponential's eigenphase on an eigenvector of
eigenvalue `mu` is `exp(-i pi mu / alpha)`, which is the phase `phi = (-mu / (2 alpha)) mod
1`, so a counter value `k` of `m` counting wires reads the singular value
`2 * alpha * (1 - k / 2**m)`. The inversion is part of the readout and not a cosmetic
detail: dropping the `2 * alpha` factor reports a phase, which is a wrong singular value
rather than an error, and the tests that compare the readout against `torch.linalg.svdvals`
fail with it removed. **The readout's range is `(0, alpha]` because a readout the register's
lower half produces is refused**, not because of the resolution: a lower-half counter value
reads a value above `alpha`, and such a value is refused rather than returned. At one counting
wire the register holds two counter values and one of them is the refused half, so the only
value a one-wire run can return is `alpha` itself, and a one-wire run whose mode falls in the
refused half raises. **Separately, the input state carries no weight on the embedding's
negative eigenvectors**, whose phases lie in the lower half, so nothing peaks at their counter
values. What the lower half carries at the example's six counting wires — the dominant
peak's tail, under a fiftieth of the sample against the mode's half — is therefore not a
peak of its own there. At one wire it need not be a tail at all: with a single upper-half
counter value, the dominant peak's own phase wraps into the refused half instead, which is
the raise above.

**The input state is where the singular vectors enter, and it is the premise.** The circuit
prepares the state whose overlap with the embedding's eigenvector of `+sigma_i` is
`sigma_i / ||A||_F` and whose overlap with the eigenvector of `-sigma_i` is zero, which is
the weighting the vectorised matrix carries. Its amplitudes are the classical `U` and `V`
factors scaled by the singular values, so a singular value decomposition is computed before
the circuit exists and the readout is an estimate of something the module already holds.
That is the algorithm's input model, paid rather than assumed.

**The block encoding, and what reading it costs.** The module also builds, in private, a
block encoding of the embedding: a preparation, a selection and the adjoint of the
preparation, whose composition extracts the embedding over `alpha`. `alpha` is the
subnormalisation — the embedding's Frobenius norm by default, and at least its spectral
norm, so `||A|| <= alpha` — and a block encoding is not free to read. A readout that
post-selects the ancilla succeeds with probability `||(A / alpha) |psi>||**2` on a
normalised input, whose greatest value over inputs is `(||A|| / alpha)**2`: where `alpha`
is much larger than `||A||` that probability is exponentially small. The block's column `j`
is the post-selected amplitude vector of the basis state `|j>`, so the number the tests
read out of the circuit is the same one. A subnormalisation below the embedding's spectral
norm is refused rather than encoded: the block would then be an operator of norm greater
than one, and no unitary has such an operator as one of its blocks, so there would be no
encoding to build.

**The mode is not a statement about the largest singular value.** The readout is the
counting register's mode, and the register's resolution spreads each eigenvalue's share
over neighbouring counter values. `within` answers only whether a named singular value lies
within half a counter step of the readout, and it is the whole of the accuracy contract: no
error bound, no confidence interval and no shot-selection rule is computed or reported
anywhere in this unit. The unit is bounded at four rows and four columns, because the
embedding is twice as wide as the matrix and every form of the phase unitary is a dense
gate on it.

**Measured.** The matrix `[[1, 2], [3, 4]]` has singular values `5.4649857...` and
`0.3659661...`. At six counting wires, 20000 shots and sampling seed 11:

```python
import torch

from flagquantum.algorithms.svd import estimate_singular_values

matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# Read in double precision, so the digits quoted below are the matrix's rather than the
# installed BLAS's: in float32 the largest lands 2e-7 from the sixth decimal's rounding
# boundary, closer than float32 resolves, and builds disagree about which side they round on.
exact = torch.linalg.svdvals(matrix.double())
largest = float(exact[0])
print([round(value, 6) for value in exact.tolist()])
# [5.464986, 0.365966]  -- the decomposition the readout is an estimate of

result = estimate_singular_values(matrix, n_counting_wires=6, shots=20000, seed=11)
print(round(result.dominant_singular_value, 6), round(result.resolution, 6))
# 5.567414 0.242061  -- the mode's readout, and the step it was resolved at
print(round(result.alpha, 6), round(result.dominant_share, 4))
# 7.745967 0.5306  -- the subnormalisation, and the share that landed on the mode
print(result.within(largest))
# True  -- the readout is within half a counter step of the largest singular value
print(
    [
        (key, round(share, 5))
        for key, share in sorted(result.distribution.items(), key=lambda item: -item[1])[
            :3
        ]
    ]
)
# [('101001', 0.53065), ('101010', 0.28805), ('101000', 0.04545)]
#   -- the mode's counter value, then the two around it: one peak, spread by the register
```

The readout is a grid value of the register's own resolution, and its distance from the
singular value it estimates is what `within` states — here `5.567414` against a step of
`0.242061`, so the value printed is not accurate to the digits shown. The mode's counter
value is the one nearest the embedding's largest eigenvalue's phase, and the two
neighbouring counter values carry the peak's own spread rather than separate peaks.
**That is what this run did and not a criterion:** nothing here says which spectra put the
mode where, and the unit does not predict which singular value the mode reports.

The same matrix at other widths, to show what the step is: four counting wires reads
`5.809475` at a resolution of `0.968246`, six reads `5.567414` at `0.242061`, and seven
reads `5.446383` at `0.121031`. Each of the three is within its own half step of
`5.4649857`, which is the contract; none of them is an exact reading.

**The premise, and the dequantization.** The cited algorithm's cost is counted in queries
to a structure that returns the matrix's entries, and against that count the state the
estimation is applied to is assumed to be preparable. Neither is present here: the matrix
is an ordinary tensor, its embedding and the embedding's exponential are dense classical
objects, and the input state is built from the singular vectors a classical
`torch.linalg.svd` returns. **No end-to-end advantage follows.** **Dequantization is
recorded rather than glossed over:** Tang's classical algorithm for the recommendation
problem removes the *exponential* speed-up and is **"only polynomially slower"** — its
bound contains `eps**-12`, which the author calls "a large slowdown in some exponents".
It is not a classical algorithm that matches the quantum runtime, and nothing in this guide
says that it is. The counter-evidence is recorded with it: the practical conditions the
dequantized algorithms need are Arrazola et al.'s, and Gharibian–Le Gall dequantize the
quantum singular value transformation for sparse matrices at constant precision; their
hardness result is for a different task, estimating a local Hamiltonian's ground-state
energy at inverse-polynomial precision given a state close to the ground state.

## Zero-noise extrapolation

`error_mitigation.py` is the first error-mitigation unit. It measures one observable at
several error strengths and continues the resulting curve to zero noise. Noise is scaled by
multiplying the single error-probability parameter a channel declares, so a scaled channel
is the same family at a different strength. `run_zne(circuit, hamiltonian,
noise_model=..., scale_factors=..., order=...)` returns a `ZneResult`: the extrapolated
estimate, the fit that produced it, the unmitigated measurement when the curve includes a
scale factor of one, the measurements themselves, and the assumptions and limitations the
estimate rests on. Two fits are offered — `polynomial_least_squares` and `richardson` — and
each method checks its own point count before anything is simulated.

**The estimate is `Tr(O rho)`, which is before measurement.** The unit extrapolates exact
state expectations rather than samples: no shot is consumed, the estimate is a point value,
and no confidence interval is computed or reported. That is why a noise model that declares
a readout rule is refused rather than measured without it. Classical readout confusion is
applied after measurement, so it is not in `rho`, and extrapolating a curve that omits it
would return a state-preparation estimate under the name of a measured one. The refusal is
by name rather than a silent omission, and the correction itself is a separate unit —
[readout-error mitigation](#readout-error-mitigation) below, which inverts a declared
confusion on a measured vector rather than on a state.
`variance_amplification` is the factor by which the fitted weights would amplify the
variance of a shot-based estimate of the same points — it is a property of the scale grid
and the fitted degree alone, and it is not a measured variance.

**The assumption is the method's, and nothing checks it.** The estimate is unbiased exactly
when the measured curve is a polynomial of degree at most `order` in the scale factor. The
ideal signal must therefore not depend on the scale factor, and no other process may depend
on it either. Neither condition is checkable from the measurements alone, and the unit says
so where the number is rather than in a footnote. What exposes a violated assumption is the
largest absolute residual the fit left. **A square fit's residual is zero by construction
and is therefore not evidence**, so the record reports `max_residual` as absent when no
degree of freedom is left rather than as an arithmetic zero, and the honest reading order is
residual first, estimate second.

**Scaling is the caller's claim, and two units of scale are offered.** The declared scaling
multiplies the one error-probability parameter each of `bit_flip`, `phase_flip`,
`depolarizing`, `two_qubit_depolarizing`, `amplitude_damping` and `phase_damping` declares.
It refuses `coherent_overrotation`, `reset_error` and `thermal_relaxation` by name:
multiplying an angle or two independent reset probabilities does not scale the noise, and
`thermal_relaxation`'s factory does not take its duration as one leading parameter. A
channel whose product leaves the unit interval is refused too, because the scale factor is
bounded by the channel it scales. A caller who needs one of the refused families supplies a
`scaling` callable and owns the claim that the scaled model differs from the original only
in noise strength; the result records which of the two claims it rests on, and the refusal
is all-or-nothing over the model rather than leaving it partly scaled.

**Not here:** shot-based execution and
readout-error mitigation. Probabilistic error cancellation and Clifford data regression are
units beside this one rather than modes of it — see the two sections below — because one
inverts a channel that is declared rather than scaling it, and the other fits the noise's
effect on near-Clifford circuits rather than scaling anything. Gate and circuit folding is a
third way to reach the same curve and is a unit of its own beside this one: it lengthens the
program by an exact identity instead of scaling a channel parameter, so it fits against the
length ratio it realized rather than against the factor requested, it needs no noise model at
all, and a call naming both it and a `scaling` callable is refused because the product of a
length ratio and a channel parameter names neither. See
[Gate and circuit folding](#gate-and-circuit-folding) below.

**Measured.** On `h(0); cx(0, 1)` with the observable `zz(0, 1)`, whose value is
analytically `1.0` and whose exactly simulated noiseless read is `0.9999999999999998` in
`complex128` and `0.9999999403953552` in the runtime's default single precision, and with
depolarizing noise at `p = 0.05` on the `cx`, the four scale factors `1, 3, 5, 7` give the
exact curve `0.871111109257`, `0.639999987284`, `0.444444444444`, `0.284444452922`:

```python
import torch

import flagquantum as fq

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm, run_zne
from flagquantum.noise import NoiseModel, depolarizing_channel

circuit = fq.Circuit(2).h(0).cx(0, 1)
observable = Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])
model = NoiseModel().add("cx", depolarizing_channel(0.05, dtype=torch.complex128))

underfit = run_zne(
    circuit, observable, noise_model=model, scale_factors=(1, 3, 5, 7), order=1,
    dtype=torch.complex128,
)
print(f"{underfit.estimate:.12f}", f"{underfit.fit.max_residual:.4e}")
# 0.951111100846 1.7778e-02

result = run_zne(
    circuit, observable, noise_model=model, scale_factors=(1, 3, 5, 7), order=2,
    dtype=torch.complex128,
)
print(
    f"{result.estimate:.12f}",
    f"{result.fit.max_residual:.4e}",
    result.fit.degrees_of_freedom,
)
# 1.000000003030 4.1723e-09 1

print(f"{result.unmitigated:.12f}", f"{result.variance_amplification:.6f}")
# 0.871111109257 2.940625

print([f"{weight:.6f}" for weight in result.fit.weights])
# ['1.537500', '-0.237500', '-0.637500', '0.337500']
```

The same four points fitted by Richardson at order 3 — its construction needs exactly
`order + 1` points, so the fourth point buys a square degree-three fit rather than a
redundant degree-two one — estimate `1.000000021110` at a variance amplification of
`11.390625`, with `max_residual` absent because the fit interpolates. The degree is chosen
by residual and not by ambition: degree three is 3.87x more costly here and an order of
magnitude less accurate, because the exact curve is degree two and the extra freedom
interpolates rounding rather than signal. Precision moves the floor, not the conclusion. In the
runtime's default single precision the same degree-two run reaches `1.3e-7` from the analytic
value `1.0` and `6.8e-8` from the exactly simulated noiseless read `0.9999999403953552`, at a
residual of `6.3e-8`; its unmitigated measurement `0.871110976` sits `1.288890243e-01` from
`1.0`, an improvement of `1.01e6`. The double-precision run above reaches `3.0e-9` from the
same analytic value at a residual of `4.2e-9`, and `3.0e-9` from its own noiseless read
`0.9999999999999998`; its unmitigated measurement `0.871111109` sits `1.288888907e-01` away,
an improvement of `4.25e7`.

The reference matters as much as the precision, which is why every distance above names what
it is measured from. The default-precision noiseless read is itself `6.0e-8` away from the
analytic `1.0`, so a distance taken against it is dominated by that offset and cannot be read
as the extrapolation's accuracy; the example takes its reference at the dtype its fits run at
for this reason.

**A wrong assumption stays visible.** Under a coherent over-rotation of `0.15` on the `cx`,
the declared scaling refuses the channel by name, and a caller-supplied callable that grows
the angle with the scale factor gives estimates `1.271993498172` at degree one and
`1.105716958201` at degree two — `2.7e-1` and then `1.1e-1` away from the noiseless value —
with residuals of `8.9e-2` and `2.9e-2`. The residual falls but stays three to five orders
above the `4.2e-9` floor the polynomial family reached at the same scale factors, so a
higher degree does not converge to the right answer and the residual says so rather than the
estimate merely looking close. No demonstration here is a performance, scaling, hardware or
fault-tolerance claim: the whole path is single-process CPU density-matrix work on a
two-wire circuit.

## Probabilistic error cancellation

`pec.py` is the second error-mitigation unit, and it inverts a declared channel rather than
scaling one. `run_pec(circuit, hamiltonian, noise_model=..., dtype=...)` returns a
`PecResult`: the mitigated estimate, the unmitigated read, one `PecLocation` per noise site,
the product of the locations' word counts, and the assumptions and limitations the estimate
rests on. `pauli_twirl_decomposition(channel)` is the single channel-level operation the unit
is built on, and it is public because the decomposition is the whole of the method: given a
`KrausChannel` it returns the Pauli words, their signed coefficients and the channel's Pauli
transfer spectrum.

**The decomposition is exact and the arithmetic is standard.** For an `n`-wire channel with
`N = 4**n` Pauli words, the Pauli transfer matrix is `R[i, j] = Tr(P_i E(P_j)) / 2**n` and a
Pauli channel is one whose `R` is diagonal, with the diagonal entries as its transfer
eigenvalues. Writing the inverse as a signed combination `sum_i c_i P_i` and applying it to
`P_j` forces `sum_i c_i (-1)**<P_i, P_j> = 1 / lambda_j`, and the sign matrix is its own
inverse up to `1 / N` — it is a Walsh transform over the Pauli group — so
`c_i = (1 / N) sum_j (-1)**<P_i, P_j> / lambda_j`. The reciprocal is inside the sum. Every
decomposition reports `gamma = sum_i |c_i|`, which is at least one and equals one only for a
channel that is itself a Pauli conjugation.

**Two boundaries are refused by name rather than approximated.** A channel whose transfer
matrix has an entry away from the diagonal is not a Pauli channel, and its inverse is not a
finite signed combination of Pauli words at all, so it is refused and the refusal quotes the
off-diagonal magnitude it measured. A channel whose transfer matrix is diagonal but whose
spectrum reaches zero has no inverse, so its weights diverge; the refusal names the vanishing
eigenvalue and the word it sits at. Both boundaries are checked at `1e-6`, which sits two
orders above the `complex64` floor the measurement is subject to and five below the smallest
refusal measured here.

**Measured.** The admitted families are the five this repository's noise module provides that
are Pauli channels: `bit_flip` and `phase_flip` at `p = 0.1` give `gamma = 1.250000004657` on
four words, `depolarizing` at `0.1` gives `1.230769234737`, `phase_damping` at `0.1` gives
`1.054092554262`, and `two_qubit_depolarizing` at `0.05` gives `1.105633804480` on sixteen.
Each reproduces its closed form — `bit_flip`'s weights are `(1 - p) / (1 - 2p)` on the
identity and `-p / (1 - 2p)` on the flip, `depolarizing`'s are `(1 + 3 / lambda) / 4` on the
identity with `lambda = 1 - 4p / 3` — to the precision the channel's operators were stored at.
The largest off-diagonal entry of all five is exactly zero at `complex128`. The refused
families and the magnitudes their refusals report:

| channel | parameter | reported off-diagonal | verdict |
| --- | --- | --- | --- |
| `amplitude_damping` | `0.1` | `1.000e-01` | sends `I` partly to `Z` |
| `coherent_overrotation` | `0.2` | `1.987e-01` | sends `Z` partly to `Y` |
| `reset_error` | `0.05` | `5.000e-02` | sends `I` partly to `Z` |
| `thermal_relaxation` | `0.05, 0.05, 1.0` | `1.000e+00` | sends `I` partly to `Z` |
| `bit_flip` | `0.5` | vanishing eigenvalue at `Y` | no inverse exists |

**The correction is applied after the channel, not in place of it.** Each location inserts
its word immediately after the channel instruction it inverts, so `E^-1` composed with `E` is
the identity and the program that is read is the declared noise removed. On `h(0); cx(0, 1)`
with `zz(0, 1)` and `bit_flip` at `p = 0.1` on the `cx`, the two-location composite reads
`0.639999995231628` unmitigated and `0.999999999999999` mitigated in `complex128`:

```python
import torch

import flagquantum as fq

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm, run_pec
from flagquantum.noise import NoiseModel, bit_flip_channel

circuit = fq.Circuit(2).h(0).cx(0, 1)
observable = Hamiltonian([HamiltonianTerm(1.0, "zz", (0, 1))])
model = NoiseModel().add("cx", bit_flip_channel(0.1, dtype=torch.complex128))

result = run_pec(circuit, observable, noise_model=model, dtype=torch.complex128)
print(f"{result.estimate:.15f}", f"{result.unmitigated:.15f}")
# 0.999999999999999 0.639999995231628

print(f"{result.gamma:.12f}", result.term_count, result.executions)
# 1.562500011642 16 17
```

The same channel at the runtime's default single precision gives `0.999999866503455` against
an unmitigated `0.639999806880951`, so the floor the mitigation leaves is the precision the
channel was declared in rather than the method's. The one-location read of the same program
is `0.799999997019768` unmitigated, `1.000000000000000` mitigated, at `gamma` `1.250000004657`
on four terms: the cost and the correction both compose per location, and the two-location
`gamma` is the square of the one-location one.

**The price is `gamma**2`, and this path does not pay it.** A sampled implementation of the
same combination spends `gamma**2` in shots, because every signed weight carries its sign into
the variance; this unit evaluates each term exactly instead, so it consumes no shots, reports
no confidence interval and claims no variance reduction. `sampling_overhead` is therefore
arithmetic about a sampled implementation rather than a measurement of this one, and `gamma`
is the number that belongs beside a mitigated estimate either way.

**What the estimate is, and what it is not.** A quasi-probability average is not a physical
density matrix and can be non-positive, which is why `run_pec` returns the value rather than a
state. The premise is the declared model and only the declared model: an error the noise model
does not carry — a miscalibrated gate, leakage, drift between the declaration and the run —
survives the inversion untouched, and a mitigated value close to the ideal is a statement
about the model before it is a statement about a device. A model that declares a readout rule
is refused, for the same reason ZNE refuses one: the estimate is `Tr(O rho)`, read before
measurement, so classical readout confusion is not in `rho` and measuring while leaving the
rule unused would report a state-preparation estimate under the name of a measured one. A
circuit that declares its own input state is refused too, because the unit re-executes the
program from its IR once per term and every program built from an IR begins at the all-zero
state. And a program whose combination would need more than `1024` exact programs is refused
with the count it needed, because four terms per single-qubit noise location and sixteen per
two-qubit one is a real ceiling rather than a large number.

**Not here:** readout-error mitigation, and Clifford data regression, which is a separate
unit rather than a mode of this one. Noise is inverted at the
locations the noise model declares and nowhere else; no gate-folding scale factor is offered,
because folding is a way to *measure* an error strength rather than a way to invert one. No
demonstration here is a performance, scaling, hardware or fault-tolerance claim: the whole
path is single-process CPU density-matrix work on a two-wire circuit, and the noise it inverts
is the one the caller declared.

## Clifford data regression

`cdr.py` is the third error-mitigation unit, and it neither scales a channel nor inverts one:
it fits the noise's effect on a set of circuits whose exact values are known. `run_cdr(circuit,
hamiltonian, noise_model=..., variants=..., dtype=...)` returns a `CdrResult`: the mitigated
estimate, the unmitigated read, the `CliffordFit` the estimate came from, and the training
points. `clifford_variants(circuit)` is the circuit-level operation the unit is built on, and
it is public because building the training set is the whole of the method: given a program it
returns one `CliffordVariant` per candidate, each with the rewritten `CircuitIR`, the name of
the site it moved and the `angle_shift` that says how far that site travelled.

**The training set is a rewrite, not a device run.** Each `phase`, `rx`, `ry`, `rz` and `u1`
rotation is snapped to its nearest quarter turn and replaced by the named Clifford word for
that turn: `rz k` is `s`, `z`, `sdg` for `k = 1, 2, 3` and nothing at all for `k = 0`; `rx k`
is `sx`, `x`, `sxdg`; `ry k` is `s sxdg sdg`, `y`, `s sx sdg`; `phase` and `u1` follow `rz`.
Residue zero is the **empty word** — the rotation is dropped rather than emitted as an
identity — and `ry`'s odd residues need three gates because the half-turn about `y` is not a
single named gate here. Over all twenty entries of the table the rewritten word reproduces the
rotation's unitary to `1.837e-16` in `complex128`, which is that precision's floor, up to a
single global phase that leaves the program's density matrix unchanged; `phase` and `u1` are
exact with no phase at all. Each variant is then accepted by
`flagquantum.simulation.stabilizer.require_clifford_program` before it is measured, so a
program the rewrite cannot reach is refused by the engine rather than snapped approximately:
`cphase`, `crx`, `cry`, `crz`, `rxx`, `ryy`, `rzz`, `u2` and `u3` have no named-Clifford word
at all, and `ccx`, `cswap` and every non-Clifford operation are outside the engine's
vocabulary.

**Clifford membership is enforced and never exploited.** The ideal value of each training
circuit is read from exact density simulation of that circuit, not from the stabilizer
representation. Every training circuit is a Clifford circuit and this path makes no use of
that fact: it costs exactly what simulating the target costs, and no simulation-cost, capacity
or scaling claim follows from it. What the rewrite buys is a training set whose ideal values
are knowable and whose circuits differ from the target one site at a time, which is the
premise the fit needs rather than a shortcut.

**The premise is one relation and the residual is its only diagnostic.** The method is
unbiased exactly when the ideal expectation is affine in the noisy one over the region the
training circuits span. Nothing checks that and nothing can check it from the measurements
alone. The largest absolute residual the fit left is what exposes a premise that did not hold,
and a fit with no degree of freedom left reports it as absent rather than as an arithmetic
zero: two points determine a line, so a two-point zero is an identity presented as a check.
The estimate carries no error bound and no confidence interval, and no improvement over the
unmitigated value is guaranteed — the residual is the warning, and it can be read only as a
relative magnitude.

**Measured.** On `h(0); ry(0, 0.7); ry(1, 0.4); cx(0, 1)` observed as `z(0) + x(1)`, with
`two_qubit_depolarizing` at `0.06` on the `cx` and `depolarizing` at `0.03` on the `h`, the
three training points read ideal `0.0`, `-1.0`, `1.0` against noisy `0.0`, `-0.898560`,
`0.936000`, and the fit reads slope `1.090028330`, intercept `-1.360e-02` and a residual of
`1.360355e-02` over one degree of freedom. The target's exact value `-0.254799345` is read
`-0.214372678` unmitigated and `-0.247275845` mitigated, so the correction closes `4.043e-02`
of `4.043e-02` down to `7.524e-03` — and the residual is what says the closure is incomplete
rather than complete. Under the `cx` noise alone the same three training circuits fall on a
line to `1.110e-16` and the mitigated value is on the exact one to `1.665e-16`, so the
residual separates a premise that held from one that did not rather than being decorative.
The two-point case is the arithmetic identity the design admits to:

```python
import torch

import flagquantum as fq

from flagquantum.algorithms import Hamiltonian, HamiltonianTerm, run_cdr
from flagquantum.algorithms.cdr import clifford_variants
from flagquantum.noise import NoiseModel, two_qubit_depolarizing_channel

circuit = fq.Circuit(2, dtype=torch.complex128).h(0).ry(0, 0.6).cx(0, 1)
observable = Hamiltonian([HamiltonianTerm(1.0, "z", (0,))])
model = NoiseModel().add(
    "cx", two_qubit_depolarizing_channel(0.05, dtype=torch.complex128)
)

result = run_cdr(circuit, observable, noise_model=model, dtype=torch.complex128)
print([variant.name for variant in clifford_variants(circuit)])
# ['nearest', 'ry-at-instruction-1']

print([(point.ideal, point.noisy) for point in result.fit.points])
# [(0.0, 0.0), (-0.9999999999999998, -0.9466666658719378)]

print(f"{result.fit.slope:.12f}", result.fit.max_residual, result.fit.degrees_of_freedom)
# 1.056338029056 None 0

print(f"{result.unmitigated:.15f}", f"{result.estimate:.15f}")
# -0.534528207698562 -0.564642473395035
```

The slope there is the reciprocal of the shrinkage the model applied to the second training
point — a reading of the declared model rather than of a device — and the mitigated error of
`2.220e-16` against a target whose exact value is `-0.564642473395035` is the least-squares
solution reproducing its own constraints, not an independent check.

**Refusals, each by name.** A program with no operation from `phase`, `rx`, `ry`, `rz` or
`u1` is refused, because there is nothing to snap and a training set built by snapping nothing
would be the target measured twice. A rotation that carries its own matrix is refused, because
it is whatever that matrix is rather than the declared rotation the quarter-turn grid is
defined on, and a rotation whose angle is an unbound parameter is refused, because it has no
nearest turn. A program that declares a measurement or an observable request is refused,
because this unit reads `Tr(O rho)` from the state and consumes no request, so returning an
estimate while a declared request stayed unused would report a measurement that never
happened. A noise model that declares a readout rule is refused, for the reason the other two
mitigation units refuse one. A model that names an operation the rewrite removes is refused
too: the model keys its channels on gate names, so every training point would be measured
under less noise than the target and the line would fit a difference the rewrite created; a
caller who needs such a model supplies the training circuits explicitly and owns that claim.
A training set whose noisy values do not separate is refused at a declared floor of `1e-9` of
the span rather than fitted on rounding, and the training set is capped at `64` circuits,
which is `128` exact simulations.

**Not here:** any error bound, confidence interval or guarantee of improvement over the
unmitigated value — and readout-error mitigation, which is a separate unit beside this one:
[readout-error mitigation](#readout-error-mitigation) below corrects a measured vector for a
declared confusion, while this unit fits a value that was never read out through one. Nothing here is cheaper than the target it
corrects, and no claim is made that the training circuits being Clifford saved this path any
simulation. No demonstration here is a performance, scaling, hardware or fault-tolerance
claim: the whole path is single-process CPU density-matrix work on two-wire circuits, and the
noise it fits is the one the caller declared.

## Readout-error mitigation

`readout_mitigation.py` is the fourth error-mitigation unit and the one the other three named
as absent. It corrects a **measured distribution** for a classical confusion the caller's
`NoiseModel` declares. `plan_readout_mitigation(noise_model, n_qubits=...)` returns a
`ReadoutMitigationPlan`: one `ReadoutBlock` per readout rule, each carrying the declared
confusion, its exact inverse, its smallest singular value, its condition number and its
amplification; `run_readout_mitigation(observed, plan, shots=...)` applies the plan to a
distribution and returns a `ReadoutMitigationResult`; `run_readout_mitigation_counts(counts,
plan)` does the same for a histogram the runtime keys with qubit `0` leftmost.

**The correction is the transpose inverse, and that transposition is the whole of the index
convention.** A calibration table is written with the **true** value as the row index; the
runtime's flat index space is a column vector whose qubit `0` is the most significant bit, and
`NoiseModel.apply_readout_probabilities` applies the declared matrix **transposed** to it. The
block's correcting operator is therefore `inv(P.T)`, not `inv(P)`. The two agree on a symmetric
confusion and differ on every asymmetric one, so the unit asserts the relation directly rather
than through agreement with a symmetric table: on `((0.9, 0.1), (0.2, 0.8))` the two inverses
differ by `0.09` in the maximum entry.

**A rule on several qubits is one block, not its marginals.** `ReadoutRule((0, 1), error)`
names a two-qubit confusion, and inverting it as two one-qubit rules would invert a different
operator: the correlated table this unit is measured on does not factorize, and its joint form
read as a matrix has a second singular value of `3.4e-01` rather than the zero a tensor product
would leave. A plain `ReadoutError` naming one qubit is one block of two rows; a
`CorrelatedReadoutError` on `k` qubits is one block of `2**k` rows. Blocks are disjoint because
a qubit named by two rules is refused — the forward path would confuse it twice and the
inverse of a doubled confusion is not the correction of either — and they are ordered by their
first qubit so two plans built from the same rules in a different order are the same plan.

**The amplification is the whole map's, and it composes as a product.** The number the unit
reports is `||inv(M)||_1` of the whole map, which is the factor by which a shot-based estimate
of anything read off the corrected vector has its standard error multiplied. For a map that is
the tensor product of disjoint blocks that norm is exactly the **product** of the blocks' own
`||inv(P_b)||_1`, not the worst block's — the two coincide only when there is one block.
`plan.sampling_overhead` is that product and `plan.summary()["limiting_block"]` names the
factor that dominates it. The standard error bound the result carries is the plan's
amplification over the square root of the shot count, `||inv(M)||_1 / sqrt(N)`, which follows
because `||M^-T w||_inf <= ||w||_inf ||M^-T||_inf` and `||M^-T||_inf = ||M^-1||_1`. The exact
per-component variance of the corrected weight is the full covariance `(diag(d) - d d^T)/N`
with `d = M p` the **device** distribution, not the `sum_i v_i**2 d_i (1 - d_i)` a naive
per-outcome reading gives — the `(v^T d)^2` term is what the naive form drops, and dropping it
was measured `40%` low (`0.0463` against `0.0643`) on the two-wire case the unit is tested on.

**The corrected vector is not obliged to be a distribution.** The inverse of a stochastic
matrix has negative entries whenever the confusion is invertible and is not a permutation, so
a corrected vector routinely carries negative mass; on the measured case `[0.9, 0.1]` under a
`0.2` symmetric confusion it reads `[1.166667, -0.166667]`, with the negative mass summed and
reported beside it. **Clipping that mass would return a different vector under the same name**,
so it is not clipped: `negative_mass` is the sum of the negative entries, `total_variation` is
the L1 distance to the observed vector, and `is_physical` states which side of the simplex the
result landed on rather than asserting that it must be inside. Nothing here refits, smooths or
projects onto the simplex, and no maximum-likelihood correction is offered.

**The premise is the model's and nothing here checks it.** The correction is exact exactly
when the declared confusion is the device's confusion. A readout error that is miscalibrated,
that drifts between calibration and use, or that is correlated in a way the model does not
carry survives the inversion untouched, and a corrected vector that lands on the ideal one is
a statement about **the model** before it is a statement about a device. The unit is also not
a free improvement: the amplification it reports is the price, it is paid in shots rather than
in gates, and a device confused on many qubits pays the product of its blocks' amplifications.

```python
import torch

from flagquantum.algorithms import plan_readout_mitigation, run_readout_mitigation
from flagquantum.noise import NoiseModel, ReadoutError, ReadoutRule

model = NoiseModel(
    readout_rules=[ReadoutRule((1,), ReadoutError(((0.92, 0.08), (0.08, 0.92))))]
)
plan = plan_readout_mitigation(model, n_qubits=2)
print(plan.covered_qubits, plan.uncovered_qubits)
# (1,) (0,)

observed = torch.tensor([0.08, 0.92, 0.0, 0.0], dtype=torch.float64)
result = run_readout_mitigation(observed, plan, shots=4000)
print([f"{value:.12f}" for value in result.probabilities.tolist()])
# ['0.000000000000', '1.000000000000', '0.000000000000', '0.000000000000']

print(f"{plan.sampling_overhead:.9f}", f"{result.standard_error_bound:.9f}")
# 1.190476190 0.018823081
```

Qubit `0` there carries no rule, so it is left exactly where it was and is recorded as
uncovered rather than given a guessed confusion; the error against the exact distribution is
`1.1e-16`, which is the float64 floor of inverting one `2 by 2` matrix rather than a value
fitted to an assertion.

**Coverage is computed, not trusted.** The plan derives which qubits its rules cover, checks
that they are disjoint and in range, and reports the rest as uncovered: an uncovered qubit is
left untouched because the correction is the identity there rather than because its confusion
was assumed to be zero. The plan's `identity` is a digest of the confusion, the width and
nothing else, so two plans built from the same rules in a different order are the same plan and
a plan whose confusion changed is a different one.

**Refusals, each by name.** A block whose smallest singular value reaches the declared floor —
`1e-6` by default, `SINGULAR_VALUE_FLOOR`, matched to the decomposition tolerance
`probabilistic error cancellation` uses — is refused rather than inverted, because the inverse's
magnitude grows as the reciprocal of that singular value and the correction would report
weights no shot count can support; the refusal prints the singular value it measured and the
floor it was given, and the floor is the caller's, so a tighter one refuses more. A qubit named
by two readout rules is refused rather than composed. A correlated block wider than
`MAX_CORRELATED_BLOCK_QUBITS` qubits is refused before its matrix is built, because inverting
it needs a `2048 by 2048` float64 matrix and its singular values. A correlated rule whose
declared width disagrees with the size of its matrix is refused by the noise model itself, so
the two cannot disagree. A histogram whose keys are not all the plan's width, whose total is
zero, or whose counts are negative or boolean is refused; an observed distribution whose rows
fall further from one than `NORMALIZATION_TOLERANCE` is refused, and one inside that tolerance
is corrected rather than rescaled silently.

**Not here:** any smoothing, refitting or maximum-likelihood projection that would restore
positivity; any tensored approximation for a block wider than the ceiling; any correction of a
confusion the model does not carry; and any assumption that the declared confusion is the
device's. This unit also carries no error bound beyond the amplification-over-`sqrt(N)` bound
above, which is a bound on the corrected weight's standard error and not a bound on the
correction's error against the truth. No demonstration here is a performance, scaling or
hardware claim: the whole path is single-process CPU linear algebra on vectors the caller
supplies, and the confusion it inverts is the one the caller declared.

## Gate and circuit folding

`folding.py` is the fifth error-mitigation unit and the only one that neither scales a channel
parameter, inverts a channel, nor fits a relation. It lengthens a program by an **exact
identity** so that the noise attached to its instructions runs more times, which is the
scaling a zero-noise extrapolation needs when the device's error is not a parameter the caller
can name. `fold_program(program, *, scale_factor, strategy="gate")` returns a `FoldingPlan`
carrying the instructions it added, the ratio it actually realized, and the folded program;
`run_zne(..., fold="gate" | "circuit")` folds each point and fits against the realized ratios
rather than the requested ones.

**The identity is `U (U† U)^m`, and it is exact for any unitary.** A fold appends `m` copies
of the body's adjoint followed by the body again, or distributes the same pairs one instruction
at a time; both are `U† U` interleaved with `U`, so the folded unitary equals the original one
for every unitary the body is made of, and the only deviation the fold contributes is the
dtype's own rounding. It is not an approximation of the program: what changes is how much noise
runs, not what is computed. The stronger statement is per instruction — each folded instruction
is its original followed by the original's inverse and then the original again — which is what
makes a **channel** in the body a refusal rather than something to fold, because a channel has
no unitary inverse to insert and a fold that dropped it would change the ideal signal.

**The two strategies are not interchangeable, because they reach different ratios.** `circuit`
repeats whole bodies, so its realized ratio is `1 + 2m` and a request between two odd integers
is met by overshooting. `gate` distributes the same pairs one instruction at a time, so it
reaches a requested ratio exactly whenever the arithmetic allows, and reaches a nearby one in
fewer instructions than the circuit strategy needs for its next odd ratio. The plan reports the
**realized** ratio, and that ratio is the abscissa the fit is given, because it is the ratio
the noise saw; a run that reports the requested factor instead would be reporting what was
intended rather than what ran.

```python
import torch

from flagquantum.algorithms import fold_program
from flagquantum.circuit import Circuit
from flagquantum.compiler.noise import lower_noise_model
from flagquantum.core.ir import CircuitIR, ensure_circuit_ir
from flagquantum.noise import NoiseModel, depolarizing_channel


def exact_state(program: Circuit) -> torch.Tensor:
    """The program's state read at complex128, so the gap is the fold's own."""
    ir = ensure_circuit_ir(program)
    exact = CircuitIR(n_wires=ir.n_wires, instructions=ir.instructions, dtype="complex128")
    return torch.as_tensor(Circuit.from_ir(exact).state()).reshape(-1)


body = Circuit(2).h(0).cx(0, 1).rz(0, 0.4).ry(1, 0.6).t(0)
for strategy in ("gate", "circuit"):
    plan = fold_program(body, scale_factor=1.4, strategy=strategy)
    gap = float((exact_state(body) - exact_state(plan.program)).abs().max())
    print(strategy, plan.scale_factor, plan.folded_instructions, plan.exact, f"{gap:.1e}")
# gate 1.4 7 True 1.6e-16
# circuit 3.0 15 False 1.6e-16

model = NoiseModel().add(
    ("h", "cx", "rz", "ry", "t"), depolarizing_channel(0.03, dtype=torch.complex128)
)
for factor in (1.0, 1.4, 2.6):
    plan = fold_program(body, scale_factor=factor, strategy="gate")
    lowered = lower_noise_model(plan.program, model)
    channels = sum(1 for i in lowered.instructions if i.metadata.get("is_channel"))
    print(factor, plan.folded_instructions, channels)
# 1.0 5 6
# 1.4 7 8
# 2.6 13 16
```

The first block of output says the gate strategy reached the requested `1.4` in seven
instructions and the circuit strategy could only reach `3.0`, in fifteen; the amplitude gap on
both rows is the complex128 rounding floor, so neither fold changed what is computed. The
second block says where the noise comes from: the same model lowers to six channel
applications on the unfolded body and to eight and sixteen as the fold repeats gates, because
the model is matched per instruction and a repeated gate carries its channel again. That is
the mechanism the method rests on and the reason it needs no noise model of its own.

**The noise has to be local to the instruction, and a fold cannot check that.** A fold assumes
that repeating a gate repeats its error and that repeating a gate does not change the error of
the gates beside it. Crosstalk between neighbouring gates, a drift over the run, or a
calibration that changed between the unfolded and folded points all violate the assumption, and
none of them is visible in the program: what the unit reports is a length ratio, and the
conversion from a length ratio to a noise strength is exactly the part it does not estimate.
A caller who reads the realized factor as a measured error strength has substituted the
device's behaviour for the model's, which is the misreading this boundary exists to prevent.

**Not here:** any estimate of the noise a fold amplifies; any channel-parameter scaling, which
is `error_mitigation.py`'s unit and is refused together with a fold because the two are
different units of scale; any fold of a program carrying a noise channel, a non-unitary
instruction, or nothing to fold; any strategy beyond the two named; and any request below
`1.0`, which would have to remove instructions and change the ideal signal. A run may not
request two factors that realize the same length either, because repeated abscissas make the
extrapolation's linear system singular. No demonstration here is a performance, scaling or
hardware claim: the whole path is single-process CPU statevector simulation on programs the
caller supplies, and the noise it amplifies is the one the caller's model declares.

## SPSA optimization

`spsa.py` is the first of this guide's two optimizer units, and like
`nelder_mead.py` in the section after this one it is not an algorithm: both
decide where to move a parameter vector rather than what to compute from one,
and they are the two halves of the same choice, one for a stochastic objective
and one for a deterministic one. `SPSAOptimizer(maxiter=..., stability=...,
parameter_gain=..., perturbation=..., parameter_gain_exponent=...,
perturbation_exponent=..., generator=...)` takes exactly one of `maxiter` and
`stability` and drives a scalar objective through `step(objective, parameters)`,
`step_and_cost(objective, parameters)`, or `estimate_gradient(objective,
parameters)` alone. `steps` and `evaluations` report what it has spent.

**Naming.** The published recursion is written with single letters — `a`, `c`,
`A`, `alpha`, `gamma` — and none of those is a name. This unit uses the domain
name of each quantity and records the correspondence: `parameter_gain` is `a`,
`perturbation` is `c`, `stability` is `A`, `parameter_gain_exponent` is `alpha`,
and `perturbation_exponent` is `gamma`. The defaults are Spall's: `0.602` and
`0.101`. `stability` defaults to a tenth of `maxiter`, and `parameter_gain` is
then derived so that the first step is `0.05` regardless of the run length —
measured `0.05` for `maxiter` in 10, 200, and 5000.

**The estimate is an estimate, not a gradient.** The two-point difference quotient
is centered on a perturbation of size `c_k`, so the estimator is biased for every
finite `c_k` and its expectation reaches the gradient only in the limit. Measured
against the analytic gradient `[0.7480963877584119, 0.3586780454497614]` of a
two-qubit Pauli energy: one draw has relative error `0.9999`, the mean of 64 draws
`0.1259`, and the mean of 512 draws `0.0284`. The mean converges as `1/sqrt(n)`;
a single draw does not converge to anything usable. **The reason to use this unit
is the cost, not the direction:** one `step` spends two objective evaluations
whether the objective has one parameter or 128, where a parameter-shift gradient
spends `2 n` and an autograd step spends one forward and one backward pass. A
caller whose objective has an exact gradient should use one.

**The cost is constant and it is measured.** Ten `step` calls spend twenty
objective evaluations at parameter counts 1, 2, 8, 32, and 128, and
`SPSAOptimizer.evaluations` agrees with the counted calls at every size.
`step_and_cost` spends three: two for the estimate and one for the cost it
reports, which is read at the pre-update parameters.

**The perturbation comes from a caller-owned `torch.Generator`.** The sign vector
is drawn with `torch.randint` on a generator the caller passes in, so a run
replays from the call site and on the parameters' own device. Without a generator
one is created for the parameter device and the draw is no longer reproducible
from outside; a generator whose device does not match the parameters is refused
rather than silently drawing elsewhere. This is the one deliberate departure from
the reference implementations, several of which draw from process-global `numpy`
state; the comparison below measures what that costs and what it buys.

**Convergence, and what its last digits are.** On `RY(0) RY(1) CX(0,1)` against
`-(Z_0 + Z_1)`, 120 steps from `[0.4, 0.4]` with `perturbation=0.25` reach
`-1.9991819605521022`, `-1.9990938026576754`, and `-1.9996860765168998` for seeds
5, 11, and 13 at 240 evaluations each. The spread across seeds is the estimator's
variance and not a solver's tolerance: the last digits belong to the draw. On a
4096-shot objective of the same energy, 200 steps reach a distance from the exact
minimum of `1.24e-05` (seed 17) and `1.55e-05` (seed 23).

**Against the reference implementation, the trajectory agrees and the state
ownership does not.** PennyLane 0.45.1's `qml.SPSAOptimizer(maxiter=120, c=0.25)`
on the same circuit and the same seeds over the same 120 steps reaches
`-1.9991819605521028`, `-1.999093802657676`, and `-1.9996860765168996` for seeds
5, 11, and 13 against this unit's `-1.9991819605521022`,
`-1.9990938026576754`, and `-1.9996860765168998` — the same trajectories to about
`1e-15`, at 240 objective evaluations each. The two draws agree because a seeded
`torch.Generator` and a seeded `numpy` global produce the same sign stream at
these seeds; that is a measured coincidence of the two libraries' generators and
nothing here depends on it. **What differs is who owns the state.** PennyLane
reads and writes process-global `numpy` random state, so two optimizers in one
process interleave unless the caller re-seeds around every call, while this unit
owns a generator per instance. One consequence of the difference is a trap worth
recording: PennyLane's `compute_grad` perturbs only the arguments that carry
`requires_grad`, so passing a plain `numpy` array as the initial parameters makes
it evaluate the objective twice per step and update nothing — measured, the
final energy was exactly the initial `-1.769414348676468` after 120 steps and 240
calls, with nothing raised. This unit takes a `torch.Tensor` and refuses anything
else at the first call rather than looping silently.

**What it refuses, and before the objective runs.** A constructor without a gain
sequence, a non-positive `perturbation` or `parameter_gain`, a non-finite
exponent, and a generator that cannot draw on the parameter device all raise
`ValidationError` at construction. A non-callable objective, non-floating or empty
or non-finite parameters, an objective returning anything other than one finite
scalar, and **an objective that writes into the tensor it is handed** raise before
any evaluation is spent. The last one matters because the estimate is a difference
of two evaluations and the reported cost is a third: an in-place objective makes
those three describe different parameters, which is a wrong number rather than an
error, so it is refused instead of silently cloned.

**There is no averaging, no constraint handling, and no checkpoint protocol.**
The recursion as published updates from one estimate. Variance reduction, a
per-coordinate perturbation scale, and resuming an optimizer from serialized
state are all absent, and each is a second algorithm with its own conditions
rather than a knob on this one.

## Nelder-Mead simplex search

`nelder_mead.py` is the second optimizer unit, and it is the deliberate opposite of
`spsa.py` in the section above. `NelderMeadOptimizer(maxiter=..., initial_step=...,
reflection=..., expansion=..., contraction=..., shrink=..., tolerance=..., atol=...)`
drives a scalar objective through `minimize(objective, parameters, simplex=None)`,
which returns a `NelderMeadResult` carrying the best vertex, the final `(n + 1, n)`
simplex ordered by value, the two spreads the run stopped on, the iteration and
evaluation counts, and a `converged` flag. `initial_simplex(parameters)` returns
the simplex a run would build from a starting point, so the starting point can be
inspected before it is spent on a run, and the caller can build their own and pass
it as `simplex`.

**The method reads only the order of objective values, and that is what it costs
and what it buys.** Each iteration ranks the vertices and replaces the worst one:
reflect it through the centroid of the rest, expand the reflection when it beat
every vertex, contract it when it did not, and shrink the whole simplex toward the
best vertex when even the contraction failed. Nothing differences the objective,
so there is no perturbation size to choose and no step size, learning rate, or
momentum either, and — unlike `spsa.py` — there is no random draw anywhere in
it. The consequence is measured rather than asserted: two runs of the same
objective from the same start return bit-identical parameters and a bit-identical
simplex, because every arithmetic decision is a comparison of two stored values.

**The premise is determinism, and a sampled objective defeats the method rather
than degrading it.** A ranking of two objective values is only as meaningful as
the values themselves, so an objective whose value moves between two evaluations
of the same point turns each decision into a coin toss and the trajectory into a
walk. That objective is what `spsa.py` beside this unit is for, and the comparison
below runs both units on one objective at one evaluation budget.
`NELDER_MEAD_ASSUMPTIONS` states the three conditions a run rests on — determinism
at the resolution the simplex resolves, local continuity with the region the
simplex reaches, and the caller's own ordering being what they mean by better —
and the result carries them, so a caller who knows their objective is sampled
finds that out from the result rather than from a paragraph here.

**`converged` does not mean the vertex is a minimum, and the gap is measured.**
The run stops when both measured spreads fall under the caller's thresholds: the
objective values across the simplex under `atol`, and the vertex positions across
the simplex under `tolerance`. Both are needed, because a flat objective collapses
the values without moving the vertices and a stalled simplex collapses the vertices
without agreeing on a value. Neither says anything about the objective's slope,
and a simplex can collapse onto a point that is not a minimum. The classic
two-well quartic ``(x**2 - 1)**2 + 0.1 x`` started at ``0.5`` reports
`converged=True` on a value ``0.199984362162`` above the other well's, and the
quoted run below is that measurement rather than a description of it. Tightening
the thresholds does not repair it: the simplex is inside one basin, and this unit
applies no restart, no multi-start, and no basin hopping. `NELDER_MEAD_LIMITATIONS`
records the consequence, which is that the published convergence theory for this
widely used variant is incomplete in more than one dimension.

**The published counterexample is refused rather than reproduced, and that is a
boundary rather than a fix.** McKinnon's function is the standard demonstration
that the method's convergence theory is incomplete, and its construction needs a
simplex that stays flat in one parameter direction. This unit refuses a starting
simplex that does not span all ``n`` parameter directions, naming the number of
directions it does span, so a degenerate simplex is refused instead of being run
on: what remains reachable through this entry point is a local minimum, and the
incompleteness of the convergence theory is carried on the result rather than
claimed away.

**The cost is a count of objective calls, and it is bracketed rather than merely
printed.** A run spends one evaluation per vertex it orders, so the count lies
between ``(n + 1) + iterations`` and ``(n + 1) + (n + 1) * iterations``: the lower
bound is a run whose every iteration evaluated one new vertex, and the upper bound
is one whose every iteration evaluated all ``n + 1``. `NelderMeadResult.evaluations`
reports the actual count, which is the number a caller comparing this unit against
a gradient method has to use; it is a count and not a latency, a device cost, or a
parallel schedule, because vertices are evaluated one at a time and no
vectorization of the simplex is attempted.

**What it refuses, and when.** A coefficient that is not what its name promises --
an `expansion` that does not exceed 1 or falls below `reflection`, a `contraction`
or `shrink` outside the open unit interval, a non-positive `maxiter` or
`initial_step`, a non-finite `tolerance` or `atol` -- raises `ValidationError` at
construction. A non-callable objective, non-floating or empty or non-finite
parameters, a simplex of the wrong shape or dtype or one whose first vertex is not
the starting point or one that does not span the parameter directions, and an
objective that writes into the tensor it is handed all raise before a vertex is
stored beside a value, so a refused run leaves no half-described simplex behind.
The in-place refusal is the same reasoning as `spsa.py`'s and it matters for the
same reason: this unit stores a vertex and its value together, and an objective
that rewrites its argument makes that stored pair describe different parameters,
which is a wrong answer rather than an exception.

**There is no bounds handling, no constraint handling, no restart, no checkpoint
protocol, and no parallel evaluation.** A constrained problem is only as well posed
as the caller's own transformation of it, and the four coefficients, the simplex
construction, both thresholds, and the budget are the caller's. Each of the absent
mechanisms is a second algorithm with its own conditions rather than a knob on this
one.

```python
import torch

from flagquantum.algorithms import NelderMeadOptimizer, SPSAOptimizer


def quadratic(parameters):
    return ((parameters - torch.tensor([1.5, -2.0], dtype=torch.float64)) ** 2).sum()


search = NelderMeadOptimizer(maxiter=400, atol=1e-14)
first = search.minimize(quadratic, torch.zeros(2, dtype=torch.float64))
print(first.value, first.iterations, first.evaluations, first.converged)
# 1.586728257135276e-17 74 143 True  -- the value, the replacements, and the calls
print([round(float(value), 12) for value in first.parameters])
# [1.500000003819, -2.000000001132]  -- the target, to the resolution it stopped at
print(f"{first.value_spread:.3e} {first.simplex_spread:.3e}")
# 1.791e-17 9.173e-09  -- the value spread and the vertex spread, against 1e-14 and 1e-8
second = search.minimize(quadratic, torch.zeros(2, dtype=torch.float64))
print(
    torch.equal(first.parameters, second.parameters),
    torch.equal(first.simplex, second.simplex),
)
# True True  -- no draw anywhere, so the run replays bit for bit


def rosenbrock(parameters):
    return (
        (1 - parameters[0]) ** 2 + 100 * (parameters[1] - parameters[0] ** 2) ** 2
    ).reshape(())


for start in ([1.2, 1.0], [-1.2, 1.0], [0.0, 0.0], [2.0, 2.0], [-2.0, 2.0]):
    run = NelderMeadOptimizer(maxiter=2000, atol=1e-14).minimize(
        rosenbrock, torch.tensor(start, dtype=torch.float64)
    )
    print(start, run.iterations, run.evaluations, f"{run.value:.3e}")
# [1.2, 1.0] 73 140 3.188e-18
# [-1.2, 1.0] 147 281 4.308e-18
# [0.0, 0.0] 96 179 4.540e-18
# [2.0, 2.0] 91 176 3.793e-18
# [-2.0, 2.0] 150 280 1.140e-17
#   -- every start reaches the valley, and the counts vary by a factor of two


def two_well(parameters):
    return ((parameters[0] ** 2 - 1.0) ** 2 + 0.1 * parameters[0]).reshape(())


shallow = NelderMeadOptimizer(maxiter=400).minimize(
    two_well, torch.tensor([0.5], dtype=torch.float64)
)
deep = NelderMeadOptimizer(maxiter=400).minimize(
    two_well, torch.tensor([-0.5], dtype=torch.float64)
)
print(f"{shallow.value:.12f}", shallow.converged, f"{deep.value:.12f}")
# 0.099366985524 True -0.100617376638  -- both wells report that they converged
print(f"{float(shallow.value - deep.value):.12f}")
# 0.199984362162  -- and that is how far the worse one is from the better one

budget = first.evaluations
for seed in (13, 5, 11):
    draw = SPSAOptimizer(
        maxiter=budget // 2,
        perturbation=0.25,
        generator=torch.Generator().manual_seed(seed),
    )
    parameters = torch.zeros(2, dtype=torch.float64)
    for _ in range(budget // 2):
        parameters = draw.step(quadratic, parameters)
    print(seed, draw.evaluations, f"{float(quadratic(parameters)):.6f}")
# 13 142 0.019539
# 5 142 0.005164
# 11 142 0.005407
#   -- SPSA at the same 142-call budget, on the objective Nelder-Mead solved

for label, build in (
    ("expansion=1.0", lambda: NelderMeadOptimizer(expansion=1.0)),
    ("shrink=1.0", lambda: NelderMeadOptimizer(shrink=1.0)),
):
    try:
        build()
    except Exception as error:
        print(label, type(error).__name__, error)
# expansion=1.0 ValidationError expansion must exceed 1, because an expansion that
# does not reach past the reflection is not an expansion
# shrink=1.0 ValidationError shrink must lie strictly between 0 and 1, because it
# moves every vertex toward the best one without passing it

collinear = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64
)
try:
    NelderMeadOptimizer().minimize(
        quadratic, torch.zeros(2, dtype=torch.float64), simplex=collinear
    )
except Exception as error:
    print("collinear", type(error).__name__, error)
# collinear ValidationError the vertices must span all 2 parameter directions, and
# they span only 1; a simplex flat in one direction cannot move in it


def in_place(parameters):
    parameters[0] = 0.0
    return (parameters**2).sum()


try:
    NelderMeadOptimizer().minimize(in_place, torch.zeros(2, dtype=torch.float64))
except Exception as error:
    print("in-place", type(error).__name__, error)
# in-place ValidationError the objective modified the tensor it was given;
# Nelder-Mead stores a vertex and its objective value together, so an in-place
# objective makes the stored pair describe different parameters
```

## Variational solvers over a checked cost operator

`variational.py` is the first unit in this guide that is a **solver** rather than a
primitive, a construction or an optimizer. `maxcut_hamiltonian(n_qubits, edges)`
builds the cost operator of a weighted MaxCut instance, and
`run_qaoa(n_qubits, edges, initial_parameters, *, steps=..., lr=..., optimizer_factory=...)`
fits QAOA angles to it. The two exist together because they have to agree: a QAOA
circuit is built from a list of weighted `ZZ` edges while an energy is an
expectation against some Hamiltonian, and nothing inside either one relates the
two, so a caller wiring `qaoa_circuit` to a hand-built operator can score one
graph's ansatz against another graph's operator and read a number about a problem
nobody posed. Here the edge list is checked once and reaches both sides through
the same tuple.

**The operator is `sum w_ij Z_i Z_j`, and its value on a basis state is the edge
total less twice the cut.** That is what makes the energy a cost rather than a cut
size: the minimum is `W - 2 * (largest weighted cut)`, so minimizing the operator
maximizes the cut, and the two are checked against each other here rather than
assumed. A negative weight inverts that one edge's preference and is allowed. A
self-loop is refused, because `Z` on one qubit squares to the identity and the
edge is then a constant shift rather than a cut. A repeated undirected pair is
refused rather than summed, because `(1, 0)` and `(0, 1)` are one cut edge:
summing would count it twice in the objective while the cost layer rotated twice
as well, and the two errors would reinforce instead of cancelling.

```python
import itertools

import torch

from flagquantum.algorithms import (
    maxcut_hamiltonian,
    qaoa_circuit,
    qaoa_loss,
    run_qaoa,
)
from flagquantum.circuit import Circuit

edges = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))
cost = maxcut_hamiltonian(4, edges)


def basis(bits):
    circuit = Circuit(len(bits))
    for wire, bit in enumerate(bits):
        if bit:
            circuit = circuit.x(wire)
    return circuit


enumerated = {
    bits: sum(1 for source, target in edges if bits[source] != bits[target])
    for bits in itertools.product((0, 1), repeat=4)
}
values = {bits: float(cost.expectation(basis(bits))) for bits in enumerated}
print(min(values.values()), max(enumerated.values()), min(values.values()) == 5 - 2 * max(enumerated.values()))
# -3.0 4 True

for layers in (1, 2, 3):
    result = run_qaoa(4, edges, torch.full((2 * layers,), 0.1))
    state = qaoa_circuit(4, edges, result.gammas, result.betas).state().reshape(-1)
    mass = (state.abs() ** 2).real
    print(layers, f"{float(result.energy):.9f}", f"{float(mass.max()):.6f}")
# 1 -1.473527074 0.162757
# 2 -1.914364219 0.237453
# 3 -2.972455502 0.494200

parameters = torch.zeros(6, requires_grad=True)
loss = qaoa_loss(4, edges, parameters[:3], parameters[3:], cost)
torch.autograd.backward(loss)
print(float(loss.detach()), bool(torch.equal(parameters.grad, torch.zeros(6))))
# 0.0 True
```

The first block prints the operator's minimum, the graph's largest cut, and the
identity between them: the square with one diagonal has five edges and a largest
cut of four, so `min = 5 - 2 * 4 = -3`, and the enumerated cut sizes and the
operator's values agree exactly. The second block is the same instance solved at
three layer counts from one constant start: one layer reaches `-1.473527074` and
puts `0.162757` of the mass on its most probable bitstring, two reach
`-1.914364219` at `0.237453`, and three reach `-2.972455502` at `0.494200` against
an optimum of `-3.0`. **That progression is a fact about this instance and this
start and not a law**: a finite layer count is an approximation, a larger layer
count is not guaranteed to be closer in general, and the reported energy is an
expectation rather than a bound. The third block is the saddle: the gradient at
a zero start is exactly zero at every parameter, so a zero start cannot move and
the objective at it is `0.0`.

**The start is the caller's, and the unit does not choose one.** The uniform
superposition is an eigenstate of every `ZZ` term in the cost operator and of
every `X` in the mixer, so the objective's gradient there is zero for every
parameter at every layer count — that is why `initial_parameters` is a required
argument rather than an optional one, and why a run started at zero returns zero
angles and reports the objective at them as its result. This is a genuine saddle
rather than a rounding artefact and it is asserted by strict equality here. The
consequence is that a caller who wants a nonzero start supplies one, and the unit
does not perturb it: a displacement chosen inside `run_qaoa` would be a claim
about the caller's instance rather than a property of QAOA.

**The fitted angles and the energy are all the result carries, and the trajectory
is a record rather than a descent.** `QAOAResult` holds the flat parameter
vector, the energy at it, and the objective's value at every step. Adam, the
default optimizer, takes a step of the learning rate's size regardless of how
flat the objective is, so `history` can rise; only the final value is the result.
Angles, the circuit and the objective are `float32`, because the circuit builder
casts every angle to `float32`, so a `float64` start is downcast rather than
refused and the energy is `float32`-accurate. The result carries no cut:
sampling the optimized circuit, decoding a bitstring, ranking sampled cuts and
comparing a sampled best against the optimum are all the caller's steps.

**The gradient-free route is the objective, not a second optimizer parameter.**
The update inside `run_qaoa` is autograd-based and `optimizer_factory` is a
`torch.optim` optimizer, so the two gradient-free units this package ships cannot
be supplied in that slot: an SPSA step is a rate-limited update from two objective
values rather than a `torch.optim` optimizer, and a simplex is a whole run
released at once rather than one either, so a single parameter could not carry
both shapes without a dispatch that hid which method ran.
`maxcut_objective(n_qubits, edges, *, device=...)` therefore returns the very
callable `run_qaoa` minimizes — the same checked edge list, the same gamma-then-beta
split — and it is readable with `float()` without a backward pass, so
[spsa.py](../../flagquantum/algorithms/spsa.py) and
[nelder_mead.py](../../flagquantum/algorithms/nelder_mead.py) can be handed it
directly:

```python
import torch

from flagquantum.algorithms import (
    NelderMeadOptimizer,
    SPSAOptimizer,
    maxcut_objective,
    run_qaoa,
)

edges = ((0, 1), (1, 2), (2, 0))
objective = maxcut_objective(3, edges)
start = torch.full((2,), 0.05)
print(round(float(objective(start)), 6), float(run_qaoa(3, edges, start, steps=300).energy))
# 0.059502 -0.9999997615814209  -- the start's cost, and where Adam lands
spsa = SPSAOptimizer(
    maxiter=400, perturbation=0.05, generator=torch.Generator().manual_seed(11)
).minimize(objective, start)
print(spsa.value, spsa.steps, spsa.evaluations, spsa.history[0])
# -0.9999911189079285 400 1201 0.05950213968753815
#   -- the cost at the last iterate, the updates, the calls, and the starting cost
simplex = NelderMeadOptimizer(maxiter=200).minimize(objective, start)
print(simplex.value, simplex.evaluations, simplex.converged)
# -1.0000003576278687 706 False
#   -- at the optimum to the float32 resolution, with the flag reading False
```

On that instance, whose optimum is `-1.0`, all three routes land at it: Adam at
`-0.9999997615814209`, SPSA at `-0.9999911189079285` in 1201 evaluations, and the
simplex at `-1.0000003576278687` in 706. The simplex reaches the optimum while
reporting `converged=False`, because that flag measures the two spreads collapsing
and not the energy arriving, and SPSA reports no flag at all; the route is
asserted on the energy rather than on the flag.

**Not here:** a cost Hamiltonian for a general QUBO or Ising instance; any
constraint, penalty or slack term; the constrained derivative-free handling COBYLA
gives CUDA-Q's VQE path; adaptive or multi-angle QAOA variants; a warm start from
a classical heuristic; a multi-start or basin-hopping wrapper; and any shots,
hardware or sampled path — including on the gradient-free route, which reaches
the same exact objective and therefore carries no shot noise either. The VQE
entry points themselves stay where they were, in
[core.py](../../flagquantum/algorithms/core.py); this module is the QAOA half of
the family and it neither wraps nor replaces them. No demonstration here is a
performance, scaling or hardware claim: the whole path is single-process CPU
statevector simulation on instances of three to five edges.

## Time evolution by a product formula

`trotter.py` is the first unit in this guide that exists to build a circuit rather
than to read one out. `trotter_circuit(hamiltonian, time, steps=..., order=...,
n_qubits=..., dtype=...)` returns an ordinary `Circuit` whose unitary approximates
`exp(-i * time * H)` for a weighted Pauli sum `H`, and
`pauli_exponential_circuit(theta, word, targets, n_qubits, ...)` returns the exact
circuit for `exp(-i * theta * P)` of one word — a basis change that maps `X` and `Y`
onto the `Z` axis, one `rz` at twice the angle, and a CX ladder down to the
word's first supported wire and back. Nothing here is a new instruction: the
result is gates the compiler, the router, and the gradient path already read.

**The premise is commutativity, and it is what a product formula trades.** A
product of single-word exponentials equals the exponential of the sum exactly when
the words commute; when they do not, the difference is the defect and it shrinks
with the step length at the composition's own rate. `order=1` is the Lie-Trotter
product, whose defect falls off as the step length, so halving the step halves it;
`order=2` is the symmetric product, whose leading defect cancels, so halving the
step quarters it. Both rates are measured below against `torch.matrix_exp` of the
dense generator rather than asserted, and a dense generator exists only at the sizes
this repository simulates: **the exact side of every comparison here is a classical
matrix exponential, so the numbers certify the composition and not a scaling.**

```python
import torch

from flagquantum.algorithms import (
    Hamiltonian,
    pauli_term,
    transverse_field_ising,
    trotter_circuit,
)
from flagquantum.algorithms.trotter import pauli_exponential_circuit
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.simulation.unitary import get_unitary

primitive = pauli_exponential_circuit(0.37, "XYZ", (0, 1, 2), 3)
print([instruction.name for instruction in primitive.to_ir().instructions])
# ['h', 'sdg', 'h', 'cx', 'cx', 'rz', 'cx', 'cx', 'h', 'h', 's']  -- the exact word

hamiltonian = transverse_field_ising(3, coupling=0.7, field=0.5)
exact = torch.matrix_exp(-1j * 0.4 * hamiltonian.matrix(dtype=torch.complex128))
for order, steps in ((1, 2), (2, 2), (2, 4)):
    circuit = trotter_circuit(
        hamiltonian, 0.4, steps=steps, order=order, dtype=torch.complex128
    )
    defect = float((get_unitary(circuit) - exact).abs().max())
    print(order, steps, f"{defect:.9f}")
# 1 2 0.050855770  -- first order, two steps
# 2 2 0.003442249  -- second order, the same two steps
# 2 4 0.000854699  -- second order, twice the steps: a quarter, not a half

diagonal = Hamiltonian([pauli_term(1.0, "Z", 0), pauli_term(0.5, "Z", 1)])
diagonal_exact = torch.matrix_exp(-1j * 0.4 * diagonal.matrix(dtype=torch.complex128))
commuting = trotter_circuit(diagonal, 0.4, steps=3, order=1, dtype=torch.complex128)
print(f"{float((get_unitary(commuting) - diagonal_exact).abs().max()):.12f}")
# 0.000000000000  -- commuting terms are reached exactly at the first order

estimate = estimate_resources(trotter_circuit(hamiltonian, 0.4, steps=1, order=2))
print(estimate.n_wires, estimate.used_wires, dict(sorted(estimate.operation_counts.items())))
# 3 3 {'cx': 8, 'h': 12, 'rz': 10}  -- gates a static pass can already count
```

The commuting pair is the control for the other three numbers: it reaches the
exact evolution at the first order because there is nothing for the product to
get wrong, which is what makes the defects above a statement about the
Hamiltonian's term structure rather than about the method being inexact.

**Two refusals are deliberate and neither is worked around.** A term that is a
multiple of the identity exponentiates to a global phase, and no gate in this
repository applies one — `Circuit` and `CircuitIR` carry no `global_phase` field —
so such a term is refused by the index that names it rather than dropped, because
dropping it leaves a circuit that is right about the observable and wrong about the
state. And a product formula's value depends on the order its terms are declared
in, so the declared order is preserved: the same two terms written in the other
order emit a different gate sequence and reach a different state, and sorting them
into a canonical order would be a second, silent parameterization.

**`exp_pauli` is not here.** CUDA-Q applies `exp(-i * theta * P)` as one opaque
instruction and decomposes it in its compiler; this module emits the decomposition,
which is what lets the static estimator count the `rz`s above, a router move them,
and autograd reach a tensor coefficient through them.

**Not here:** an error bound, which needs a commutator norm the caller has to
supply; higher-order Yoshida and Suzuki compositions, which spend more
exponentials than they buy until a caller measures otherwise; time-dependent
Hamiltonians; and the rest of the family the parity contract groups with this unit
— qubitization, QSVT, and double factorization. The block encoding underneath is
now the shared primitive documented in the next section; the SVD unit is its only
consumer today, and the walk step it admits is what the family above would be
built from.


## Block encoding and the qubitization walk step

`primitives/block_encoding.py` is the first primitive in this guide that is a
*contract* rather than a construction. A block encoding of a Hermitian `H` at a
subnormalisation `alpha` is a unitary `U` whose flagged block is `H / alpha`:
writing `|0>` for the flag register's all-zero state and `P = |0><0|` for the
projector onto it, `P U P = H / alpha`, and `U` is otherwise unconstrained. Two
protocols state that — `BlockEncoding`, which names `num_system`, `num_ancilla`,
`alpha` and `append_apply`, and `WalkEncoding`, which adds `append_walk_step` and
`append_adjoint_walk_step` — and `spectral_block_encoding(matrix, alpha=None)`
is the implementation that ships.

**The construction is the direct spectral reflection, and it is exact in the
sense that matters here.** With `A = H / alpha` and `V` the eigenbasis of `A`,

```text
U = [[ A,                      sqrt(I - A**2) ],
     [ sqrt(I - A**2),         -A            ]]
```

built as one dense matrix on `n + 1` qubits, with the flag left in `|0>` rather
than prepared, so `append_apply` emits exactly one gate. `A` and
`sqrt(I - A**2)` are functions of the same Hermitian matrix and therefore
commute, which is why `U ** 2 = I` holds identically rather than to within the
quality of an eigenbasis; that is the property a truncated series or a
mean-of-two-branches approximation does not have.

**The walk step is what makes this more than a repackaging.** `W = Z_ancilla U`
is a rotation whose eigenphases are the arccosines of the encoded eigenvalues:
one phase `+-arccos(w_j / alpha)` per encoded eigenvalue `w_j`, so the cosines of
`W`'s eigenphases are the encoded spectrum twice over. That is the identity every
spectral algorithm in this family is built from, and it is what `WalkEncoding`
names. It holds for the reflection above and **not** for `U` on its own, whose
eigenphases are only `0` and `pi`.

The block below prints the step's *definition* as well as its spectrum, `W` against
`Z_ancilla U` as matrices. A spectrum alone does not pin a walk step: `W` and the
encoding it was built from are both Hermitian unitaries on the same register, so a
construction that never applied the flip would still have a spectrum. The adjoint
is printed as `W^dagger W` against the identity for the same reason — this
construction's step is its own adjoint, so an adjoint's cosines are its own.

The block is read back out of the *circuit* rather than asserted, one basis state
at a time, and it is checked against both the matrix over `alpha` and the matrix,
because a construction that returned its argument would agree with the second and
not the first:

```python
import torch

from flagquantum.algorithms.primitives import (
    BlockEncoding,
    WalkEncoding,
    spectral_block_encoding,
)
from flagquantum.circuit import Circuit
from flagquantum.simulation.unitary import get_unitary

matrix = torch.tensor(
    [
        [1.0, 0.5, 0.0, 0.0],
        [0.5, -2.0, 0.25, 0.0],
        [0.0, 0.25, 0.75, 0.0],
        [0.0, 0.0, 0.0, 1.5],
    ],
    dtype=torch.float64,
)
encoding = spectral_block_encoding(matrix)
print(encoding.num_system, encoding.num_ancilla, f"{encoding.alpha:.6f}")
# 2 1 2.904738  -- two system qubits, one flag qubit, and the Frobenius norm as the default factor

columns = []
for basis in range(4):
    circuit = Circuit(3)
    for index in range(2):
        if (basis >> (1 - index)) & 1:
            circuit.gate("x", 1 + index)
    encoding.append_apply(circuit, ancilla=0, qubits=[1, 2])
    columns.append(circuit.state().reshape(-1)[:4])
block = torch.stack(columns, dim=1)
print([round(float(block[index, index]), 3) for index in range(4)])
# [0.344, -0.689, 0.258, 0.516]  -- the block's diagonal, read back one basis state at a time
print([round(float(matrix[index, index] / encoding.alpha), 3) for index in range(4)])
# [0.344, -0.689, 0.258, 0.516]  -- the same entries of the matrix over the factor
print(round(float((block - matrix).abs().max()), 3))
# 1.311  -- and not the matrix, which is what a construction returning its argument would print as 0.0
print(round(float((block - matrix.to(torch.complex128) / encoding.alpha).abs().max()), 6))
# 0.0  -- while the block is the matrix over alpha to the statevector's own precision

select = encoding.select
identity = torch.eye(8, dtype=torch.complex128)
print(
    bool((select - select.mH).abs().max() < 1e-12),
    bool((select @ select - identity).abs().max() < 1e-12),
)
# True True  -- Hermitian, and its own inverse rather than only unitary


def step(target, adjoint=False):
    circuit = Circuit(3)
    if adjoint:
        target.append_adjoint_walk_step(circuit, ancilla=0, qubits=[1, 2])
    else:
        target.append_walk_step(circuit, ancilla=0, qubits=[1, 2])
    return get_unitary(circuit).to(torch.complex128)


def step_cosines(target, adjoint=False):
    return torch.sort(
        torch.cos(torch.angle(torch.linalg.eigvals(step(target, adjoint))))
    ).values


encoded = torch.linalg.eigvalsh(matrix) / encoding.alpha
expected = torch.sort(torch.cat((encoded, encoded))).values
observed = step_cosines(encoding)
print([round(float(value), 3) for value in expected])
# [-0.724, -0.724, 0.264, 0.264, 0.374, 0.374, 0.516, 0.516]  -- eigvalsh of the matrix, over alpha, twice over
print([round(float(value), 3) for value in observed])
# [-0.724, -0.724, 0.264, 0.264, 0.374, 0.374, 0.516, 0.516]  -- eigvals of the circuit's unitary, cosine of each phase
print(bool((observed - expected).abs().max() < 1e-6))
# True  -- two different routines on two different objects agree, so this is the identity and not one computation written twice
print(bool((torch.abs(torch.abs(observed) - 1.0) > 1e-3).all()))
# True  -- and no phase is trivial, so the identity carries more than the endpoints

plain = Circuit(3)
encoding.append_apply(plain, ancilla=0, qubits=[1, 2])
plain_cosines = torch.sort(torch.cos(torch.angle(torch.linalg.eigvals(get_unitary(plain))))).values
print([round(float(value), 3) for value in plain_cosines])
# [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]  -- U without the flag flip: an involution, with cosines only on the endpoints
print(bool((plain_cosines - expected).abs().max() > 1.0))
# True  -- so the phase flip is the construction and not a convention about which flip to use
flip = torch.kron(
    torch.diag(torch.tensor([1.0, -1.0], dtype=torch.complex128)),
    torch.eye(4, dtype=torch.complex128),
)
print(f"{(step(encoding) - flip @ select).abs().max():.3e}")
# 2.600e-08  -- the step is the flag flip composed with the encoding, by definition: a spectrum cannot show this, because the step and its own encoding are both Hermitian unitaries on the register
print(f"{(step(encoding, adjoint=True) @ step(encoding) - identity).abs().max():.3e}")
# 4.997e-08  -- and the adjoint inverts it, which the adjoint's own cosines would not show, since this construction's step is its own adjoint
```

**The surface is a protocol because two implementations are read by one
consumer.** A consumer written against `BlockEncoding` and `WalkEncoding` names
three numbers and one method, and this guide's readback above is such a consumer:
it reads the spectral encoding, which holds a dense matrix, and it would read an
encoding that holds no matrix at all just as well. That is what
[ARCH-012](../architecture/decisions/ARCH_012_CUDAQ_PARITY_CONTROL_SEQUENCE.md)'s
replacement gate asks for, and it is the ground this primitive was admitted on:
`flagquantum/algorithms/svd.py` used to carry a private mean-of-two-branches
construction, the direct reflection above replaced it, and the walk identity is
the capability the private form did not have — its `Z * U` product missed the
identity by `2.216` in the largest entry.

**Two divergences from CUDA-Q are deliberate and are stated rather than
smoothed over.** The flagged block is `+H / alpha` here, where CUDA-Q's
qubitization folds the sign of `H` into the walk step and carries `-H / alpha` in
the block; the identity above is stated for the block this unit emits. And
FlagQuantum addresses qubits individually, so `append_apply` takes the flag
`ancilla` and the operator's `qubits` as separate arguments rather than the
combined register a CUDA-Q control set needs. Neither divergence changes what the
walk step computes.

**Not here, and each is an owned gap rather than an omission.** The gate is one
dense matrix on `n + 1` qubits built from `torch.linalg.eigh` of the input, so
the cost is diagonalising the caller's matrix and there is no gate-efficient
synthesis; no error bound is reported, because an approximate `H` is the caller's
error and nothing here estimates it. A Pauli linear-combination encoding — whose
preparation is a superposition over coefficients rather than the identity, and
which needs a multi-wire flag reflection — is the family's second member; it is
[its own section below](#block-encoding-a-weighted-pauli-sum), and it is a
construction rather than a gate-efficient synthesis either. Qubitization is
exposed as two protocol methods rather than as a public walk object holding a
moment count, and QSVT and double factorization, the two algorithms the walk step
exists to serve, are absent.

## Block encoding a weighted Pauli sum

`primitives/linear_combination.py` is the second implementation of the two
protocols above, and it is the one that takes the Hamiltonian the way a chemistry
or spin-model workload states it: as a weighted sum of Pauli words rather than as
a dense matrix. `LinearCombinationEncoding(hamiltonian)` is the constructor, and
the Hamiltonian is `flagquantum.algorithms.core.Hamiltonian`.

**The factor is stated from the data, not named after a norm.**
``alpha = sum_j |c_j|``, computed from the coefficients before any circuit exists.
It is at least the spectral norm of `H`, so the flagged block ``H / alpha`` has
norm at most one, which is what a block encoding requires. It is deliberately
**not** the Frobenius norm — the example prints both — and the two are not even
ordered against each other: the Frobenius norm of a Pauli product on `n` wires is
``2 ** (n / 2)``, so a one-wire sum can have a Frobenius norm above the sum of its
magnitudes. A caller who wants a larger factor scales the Hamiltonian, which
scales the block by the same factor; the factor is not a parameter.

**The coefficients are prepared as amplitudes, and the sign is a separate
diagonal.** The index register holds a uniform superposition over the terms
weighted by ``sqrt(|c_j| / alpha)``, and the
[state-preparation primitive](#state-preparation) carries no relative phase, so
the coefficients cannot be prepared by signing the amplitudes: a Hamiltonian with
negative coefficients needs the sign as its own diagonal, applied to the index
register under the select. The block below is checked against the *signed* sum
over `alpha` and against the same sum with the signs dropped, and the second
residual is what a construction that lost the sign would print.

**The walk identity is stated on the ladder-zero subspace, and that is the honest
form of it.** The flag register is ``k + max(k - 2, 0)`` wires: `k` index wires,
where ``k = max(1, ceil(log2 m))`` for `m` terms, plus the ladder ancillas the
multi-controlled gates are built from. The encoding is exactly block diagonal in
the ladder register — the ancillas enter and leave ``|0>`` — but it is **not** the
identity there, so the arccosine spectrum is a claim about the subspace the
ladder ancillas read zero on, which is the subspace the pair of flag registers
selects, and the block below restricts to it rather than diagonalising the whole
register. On that subspace every eigenvalue of ``H / alpha`` is attained, each
carrying the walk's own multiplicity.

**This construction is not its own adjoint, and that is a property rather than a
defect.** The spectral encoding above is Hermitian and is its own inverse, so its
adjoint is a no-op and its `W^dagger W` identity is uninformative. The
linear-combination encoding is neither: its terms share the ladder ancillas the
multi-controlled gates are built from, so the product of them is not its own
reverse, and ``append_adjoint_walk_step`` has to emit the select's terms in the
reverse order. The block prints the deviation on the public surface — the matrix
residual, not a spectrum, because the two orders have the same kind of spectrum —
and prints `W^dagger W` against the identity over the **whole** register rather
than on the flagged subspace, since that is where a skipped reversal shows.

```python
import torch

from flagquantum.algorithms.core import Hamiltonian, HamiltonianTerm
from flagquantum.algorithms.primitives import (
    BlockEncoding,
    LinearCombinationEncoding,
    WalkEncoding,
    spectral_block_encoding,
)
from flagquantum.circuit import Circuit
from flagquantum.simulation.unitary import get_unitary

terms = [
    (1.0, {0: "x"}),
    (0.5, {1: "z"}),
    (-1.5, {0: "z", 1: "x"}),
    (0.25, {0: "z", 1: "z"}),
    (0.75, {0: "x", 1: "x"}),
    (-0.5, {0: "x", 1: "z"}),
]


def build(pairs):
    return Hamiltonian(
        [HamiltonianTerm(c, dict(axes), sorted(axes)) for c, axes in pairs]
    )


encoding = LinearCombinationEncoding(build(terms))
matrix = build(terms).matrix()
print(encoding.num_system, encoding.num_ancilla, encoding.index_width)
# 2 4 3  -- two system qubits, a four-wire flag register, and three index wires for six terms
print(f"{encoding.alpha:.6f}", round(sum(abs(c) for c, _ in terms), 6))
# 4.500000 4.5  -- the factor is the sum of the coefficient magnitudes, read off the data before any circuit exists
print(round(float(torch.linalg.norm(matrix, 2)), 3), round(float(torch.linalg.norm(matrix)), 3))
# 2.784 4.183  -- it clears the spectral norm and is not the Frobenius norm, which is what "stated rather than named after a norm" means

columns = []
for basis in range(4):
    circuit = Circuit(6)
    for index in range(2):
        if (basis >> (1 - index)) & 1:
            circuit.gate("x", 4 + index)
    encoding.append_apply(circuit, ancilla=0, qubits=[4, 5])
    columns.append(circuit.state().reshape(-1)[:4])
block = torch.stack(columns, dim=1)
print([round(float(block[index, index]), 3) for index in range(4)])
# [0.167, -0.167, 0.056, -0.056]  -- the block's diagonal, read back out of the circuit one basis state at a time
print([round(float(matrix[index, index] / encoding.alpha), 3) for index in range(4)])
# [0.167, -0.167, 0.056, -0.056]  -- the same entries of the signed sum over the factor
print(f"{(block - matrix.to(torch.complex128) / encoding.alpha).abs().max():.2e}")
# 1.99e-07  -- the block is the signed sum over alpha, to the statevector's own precision
print(round(float((block - matrix.to(torch.complex128)).abs().max()), 3))
# 1.167  -- and not the sum itself, which is what a construction returning its argument would print as 0.0
unsigned = build([(abs(c), axes) for c, axes in terms]).matrix()
print(round(float((block - unsigned.to(torch.complex128) / encoding.alpha).abs().max()), 3))
# 0.667  -- and not the same sum with the signs dropped, which is what a preparation carrying no relative phase would give
print(round(float((matrix - unsigned).abs().max()), 3))
# 3.0  -- the two candidates differ by twice the negative coefficients, so the check above compares a residual rather than a value with itself


def step(target, adjoint=False):
    circuit = Circuit(6)
    if adjoint:
        target.append_adjoint_walk_step(circuit, ancilla=0, qubits=[4, 5])
    else:
        target.append_walk_step(circuit, ancilla=0, qubits=[4, 5])
    return get_unitary(circuit).to(torch.complex128)


identity = torch.eye(64, dtype=torch.complex128)
print(round(float((step(encoding) - step(encoding).mH).abs().max()), 3))
# 1.744  -- the encoding is not Hermitian, because the terms share the ladder ancilla the multi-controlled gates are built from
print(f"{(step(encoding, adjoint=True) @ step(encoding) - identity).abs().max():.2e}")
# 9.11e-07  -- the adjoint inverts the step, which needs the select's terms reversed: reusing them forward leaves 1.66 here
ladder_zero = [state for state in range(64) if state & (1 << 2) == 0]
restricted = step(encoding)[ladder_zero][:, ladder_zero]
observed = torch.sort(torch.cos(torch.angle(torch.linalg.eigvals(restricted)))).values
encoded = torch.linalg.eigvalsh(matrix) / encoding.alpha
expected = torch.sort(torch.cat((encoded, encoded))).values
print([round(float(value), 3) for value in expected])
# [-0.619, -0.619, -0.272, -0.272, 0.372, 0.372, 0.519, 0.519]  -- eigvalsh of the sum over alpha, each eigenvalue twice
distinct = expected.unique()
reached = sum(
    int(float((observed - value).abs().min()) < 1e-6) for value in distinct
)
print(reached, distinct.numel(), observed.numel())
# 4 4 32  -- every encoded eigenvalue is attained, in a subspace carrying thirty-two eigenvalues of its own
print(bool(float((torch.abs(expected) - 1.0).abs().min()) > 1e-3))
# True  -- and no encoded phase is trivial, so the identity carries more than the endpoints


def consume(target: BlockEncoding) -> str:
    kind = (isinstance(target, BlockEncoding), isinstance(target, WalkEncoding))
    return f"{target.num_system} {target.num_ancilla} {kind}"
print(consume(encoding), consume(spectral_block_encoding(matrix.real, encoding.alpha)))
# 2 4 (True, True) 2 1 (True, True)  -- one consumer, two implementations: the second holds a dense matrix this one never forms
```

**One consumer, two implementations, and that is the admission ground.** The
`consume` function above names only `num_system`, `num_ancilla`, and the two
protocols, and it reads the spectral encoding and this one without being told
which it holds. Neither implementation replaced a third consumer's code —
`flagquantum/algorithms/svd.py` was already migrated to the spectral encoding —
so what this second implementation demonstrates is the replacement gate's other
half: the boundary was already load-bearing, and a new implementation was added
behind it with no consumer edited.

**Two divergences from CUDA-Q are inherited from the encoding above and one is
its own.** The flagged block is `+H / alpha` rather than CUDA-Q's `-H / alpha`
with the sign folded into the walk step, and `append_apply` names the flag
`ancilla` and the operator's `qubits` separately rather than a combined register;
both are as in the spectral encoding. This construction's own divergence is the
subnormalisation: `sum_j |c_j|` is a tighter factor than a norm-based default
would give and is not the spectral norm, so `alpha` here is a number the caller
can read off the coefficients rather than one that needs a matrix.

**Not here.** There is no gate-efficient synthesis, so the `2 ** k` preparation
is a dense classical precomputation and the cost is exponential in the index
register's width rather than in the term count; no amplitude amplification, so
the success probability of the select is not amplified; no gate-count or error
bound; and the walk is exposed as protocol methods rather than as a public
qubitization object holding a moment count. Complex coefficients are refused
rather than supported, because the preparation primitive's amplitudes carry the
magnitudes and a complex phase would need a second diagonal this unit does not
emit. QSVT, double factorization, and the Bravyi-Kitaev, parity and ternary-tree
transforms remain absent.

## Logical-layer resource estimation

`logical_resources.py` answers the question a fault-tolerant plan is costed with,
over a program that has already been compiled: how many Clifford operations and how
many T operations does it apply, how many logical layers is that once the compiler
has scheduled it, and what does a distance-`d` rotated surface code spend to run
those layers.

**The gate-level tally is not re-derived, and that is the unit's first property
rather than an implementation note.** The operation counts, the T family, and the
schedule depth are `flagquantum.compiler.resource_estimation.estimate_resources`,
the compiler's own static record, nested inside the report rather than copied field
by field. A second implementation of what a T-depth means would be a second source
of truth for the number this whole unit exists to produce, so there is not one: the
report carries the compiler's `ResourceEstimate` object itself, which is why a
reader who wants per-wire depths reads them from the record that owns them.

```python
from flagquantum.algorithms import (
    estimate_logical_resources,
    surface_code_qubits_per_logical,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler.resource_estimation import estimate_resources

program = Circuit(3).h(0).cz(0, 1).t(0).cx(0, 2).t(1).x(2).tdg(0)
direct = estimate_resources(program)
report = estimate_logical_resources(program, distance=5)
print(direct.n_operations, direct.t_count, direct.t_depth, direct.depth)
# 7 3 2 5  -- the compiler's static record: operations, T count, T depth, schedule depth
print(report.estimate == direct)
# True  -- and the report nests that record rather than recomputing it
print(report.clifford_count, report.t_count, report.n_clifford_t)
# 4 3 7  -- four Clifford operations and three T operations, partitioned by opcode
print(report.code_distance, report.physical_qubits_per_logical, report.n_qubits)
# 5 49 3  -- distance 5, 2 d^2 - 1 = 49 physical qubits per patch, three logical qubits
print(report.logical_depth, report.surface_code_cycles)
# 5 25  -- five logical layers, each costing d = 5 surface-code cycles
print(report.physical_qubits, report.spacetime_volume)
# 147 3675  -- 49 x 3 patches, and their product with the 25 cycles
print([surface_code_qubits_per_logical(d) for d in (3, 5, 7, 9)])
# [17, 49, 97, 161]  -- the patch the code model spends, tabulated without a program
print([estimate_logical_resources(program, distance=d).spacetime_volume for d in (3, 5, 7, 9)])
# [765, 3675, 10185, 21735]  -- d squared times d, so the volume grows as d cubed
print(sorted(report.to_dict()["capability_evidence"]))
# ['limitations']  -- the report's statements travel under the maturity matrix's own field name
```

**Two refusal families, and the second is the one worth reading.** A logical
resource estimate is defined over Clifford+T programs, and a program outside that
set is refused by name rather than counted as something else. A *parametric
rotation* is refused because this unit reads opcodes and runs no legalization: a
T-count taken over a rotation the program still contains would be the count of a
circuit nobody will run, and it would be reported with the authority of a circuit
that had been compiled. An exact quarter turn does have a Clifford+T form now, in
the compiler's own angle synthesis, and this unit deliberately does not reach for
it — asking whether a float angle is a Clifford+T angle is a decision about the
caller's synthesis pipeline. Approximation to an arbitrary angle remains absent,
and the row says so. A *compound
operation whose T-count is its decomposition's* — a Toffoli, a controlled swap — is
refused for a reason that is easy to miss: **a Toffoli is not a Clifford gate.** The
schema has a Clifford set, and treating `ccx` as one Clifford operation would
understate the T-count of every program containing one, silently, in the direction
that flatters the result. The decompositions do not agree on what a Toffoli costs
either, so the unit declines to pick one. The decomposition family is checked before
the rotation family deliberately: a caller who wrote a Toffoli most likely believed
its T-count was well defined, and that reading is worth answering directly. Both
messages name every offending opcode, sorted.

The classification is total over the operator schema rather than best-effort: every
declared opcode is a Clifford, a T, a channel, or one of the two refused families,
and the tests assert that partition against the schema itself, so an opcode added
later cannot quietly fall outside it.

```python
from flagquantum.algorithms import estimate_logical_resources, surface_code_qubits_per_logical
from flagquantum.algorithms.core import transverse_field_ising
from flagquantum.algorithms.trotter import trotter_circuit
from flagquantum.circuit import Circuit

program = Circuit(3).h(0).cz(0, 1).t(0).cx(0, 2).t(1).x(2).tdg(0)
for label, candidate in (
    ("a rotation", Circuit(2).h(0).rz(0, 0.3).t(1)),
    ("a Toffoli", Circuit(3).h(0).ccx(0, 1, 2)),
    ("a Trotter step", trotter_circuit(transverse_field_ising(3), 0.4, steps=1, order=2)),
):
    try:
        estimate_logical_resources(candidate, distance=5)
        outcome = "NOT REFUSED"
    except Exception as exc:
        outcome = type(exc).__name__
    print(label, outcome)
# a rotation CapabilityError  -- an un-synthesised angle has no T-count to report
# a Toffoli CapabilityError  -- a Toffoli is not a Clifford gate, and its decomposition is a choice
# a Trotter step CapabilityError  -- the gate-level estimator reports zero Ts here, which is the report this unit refuses to produce
for label, call in (
    ("distance 1", lambda: surface_code_qubits_per_logical(1)),
    ("distance 4", lambda: surface_code_qubits_per_logical(4)),
    ("n_qubits=0", lambda: estimate_logical_resources(program, distance=5, n_qubits=0)),
):
    try:
        call()
        outcome = "NOT REFUSED"
    except Exception as exc:
        outcome = type(exc).__name__
    print(label, outcome)
# distance 1 ValueError  -- a distance-1 patch carries no redundancy and corrects nothing
# distance 4 ValueError  -- the model's patch and its logical operators are defined on odd distances
# n_qubits=0 ValueError  -- a footprint over no logical qubit is not a smaller footprint
```

**One code model, named, with its conventions travelling beside its numbers.** The
report states `rotated_surface_code_2d`: `d**2` data qubits and `d**2 - 1` measure
qubits, so `2 d**2 - 1` physical qubits per logical qubit. That is the model's own
convention rather than a measurement, the distance must be an odd integer of at
least 3, and `surface_code_qubits_per_logical` is exposed on its own because
tabulating several distances needs the patch size without a program to estimate.

**A measurement is charged one logical layer, and the charge is visible.** A
measurement is an IR-level record outside the instruction sequence the scheduler
walks, so it is not in the schedule depth; the report adds one logical layer per
measurement record rather than dropping it, and the section below shows the same
program with and without two records so the difference is on the page rather than in
the arithmetic.

**What the report refuses to state is the failure rate.** No physical error rate is
read and no logical error rate is reported, because a threshold fit's prefactor and
its threshold are a device's numbers rather than this unit's. So this unit says how
much hardware a logical program would occupy and for how long, and never how often
it would fail — and the volume it reports is a floor for a circuit of these layers,
not a compiled estimate, because there is no distillation factory, no magic-state
budget, no routing overhead, no placement, no scheduling, and no device model.

The distance sweep is where that boundary is easiest to misread, so the guide states
it plainly: the patch grows as `d` squared and the cycles grow as `d`, so the volume
grows as `d` cubed for a fixed program. What that volume *buys* in logical error is
the device's threshold fit and not this report's, so the cost of a distance is
printed here and the benefit never is.

## Reversible integer addition

`arithmetic.py` builds one construction and nothing else: an in-place ripple-carry
adder that writes the sum of two `n`-bit registers into the second of them, using
one working carry wire and leaving the carry out of the most significant position
on a wire of its own.

**What it is, in one line.** `adder_circuit(n_bits)` returns a
`flagquantum.circuit.Circuit` of `2 n + 2` wires applying `2 n` Toffolis and
`6 n + 1` `cx` — `8 n + 1` operations, linear in the width and independent of the
values added, so the cost of an addition is known before the addends are.
`adder_wires(n_bits)` returns the register map as an `AdderWires` record, so a
caller assembling a larger computation knows which wire holds which bit rather
than reconstructing the layout from the constructor's source.

```python
from flagquantum.algorithms import adder_circuit, adder_wires

circuit = adder_circuit(3)
names = [instruction.name for instruction in circuit.to_ir().instructions]
print(circuit.n_wires, len(names), names.count("ccx"), names.count("cx"))
# 8 25 6 19  -- 2 n + 2 wires, 8 n + 1 operations, 2 n Toffolis, 6 n + 1 cx
print(sorted(set(names)))
# ['ccx', 'cx']  -- every gate is a Toffoli or a controlled X, and no rotation
wires = adder_wires(3)
print(wires)
# AdderWires(a=(0, 1, 2), b=(3, 4, 5), carry=6, carry_out=7)  -- the register map
print(wires.n_bits, wires.n_wires)
# 3 8  -- the width and the wire count the map describes
print([len(adder_circuit(n).to_ir().instructions) for n in (1, 2, 3, 4, 5)])
# [9, 17, 25, 33, 41]  -- 8 n + 1, one operation per bit plus the final cx
print(
    [
        sum(1 for i in adder_circuit(n).to_ir().instructions if i.name == "ccx")
        for n in (1, 2, 3, 4, 5)
    ]
)
# [2, 4, 6, 8, 10]  -- 2 n Toffolis, and the whole ripple rides one ancilla
```

**The construction is Cuccaro's, and the guide says so because the alternative is
the one a reader is likely to have in mind.** A majority gate folds the running
carry into a third wire, so the wire that leaves a majority holding the carry out
of one position is the wire that feeds the carry into the next; an unmajority gate
walks the ripple back down and gives the third wire back clean. That is why one
ancilla suffices where a naive ripple would put a carry wire under every bit, and
why the two data wires are their original selves by the time the carry into a
position is available — which is exactly when the sum bit can be written into the
second register. **The Gidney-Ekera construction is smaller and is not what this
unit builds.** It uncomputes its ancillas by measuring them and feeding the
outcomes forward, which needs a mid-circuit measurement and a classical
feedforward path, and this repository has none in a circuit it can cost
statically. The absent construction is recorded as an owned gap rather than
implied away by a name.

**The circuit is shown running, and the sum is read rather than asserted.** The
script prepares a computational basis state holding the two addends, runs the
adder on the ordinary statevector path, and decodes the wire labels back into
integers, so the number printed as the sum is read out of a state vector the
runtime produced. The register map is what makes the decoding possible, which is
the second reason it is public.

```python
import torch

from flagquantum.algorithms import adder_circuit, adder_wires
from flagquantum.circuit import Circuit


def add(n_bits, a, b, *, carry=0, carry_out=0):
    """Run the adder on one input and read the result out of the state vector."""
    wires = adder_wires(n_bits)
    width = wires.n_wires
    label = 0
    for value in (a, b):
        for index in range(n_bits):
            label = (label << 1) | ((value >> (n_bits - 1 - index)) & 1)
    label = (((label << 1) | carry) << 1) | carry_out
    vector = torch.zeros(2**width, dtype=torch.complex128)
    vector[label] = 1.0
    register = Circuit(width, inputs=vector.reshape(1, -1), dtype=torch.complex128)
    for instruction in adder_circuit(n_bits).to_ir().instructions:
        register.gate(instruction.name, instruction.wires)
    state = register.state().reshape(-1).abs()
    reached = int(state.argmax())
    bits = [(reached >> (width - 1 - index)) & 1 for index in range(width)]

    def read(offset):
        value = 0
        for index in range(n_bits):
            value = (value << 1) | bits[offset + index]
        return value

    return (
        read(0),
        read(n_bits),
        bits[2 * n_bits],
        bits[2 * n_bits + 1],
        float(state[reached]),
    )


for a, b in ((1, 1), (7, 1), (7, 7), (3, 5)):
    print(a, b, add(3, a, b))
# 1 1 (1, 2, 0, 0, 1.0)  -- a comes back unchanged, b holds the sum, both ancillas are clean
# 7 1 (7, 0, 0, 1, 1.0)  -- 8 does not fit in three bits, and the top wire holds the carry
# 7 7 (7, 6, 0, 1, 1.0)  -- 14 keeps its top bit above the register, not inside it
# 3 5 (3, 0, 0, 1, 1.0)  -- and the working carry wire is zero again, as promised
print(add(3, 1, 1, carry=1))
# (1, 3, 1, 0, 1.0)  -- off the contract: a dirty carry wire changes the answer
print(add(3, 1, 1, carry_out=1))
# (1, 2, 0, 1, 1.0)  -- but a dirty carry-out wire only flips the wire it entered on
```

**Two wires must enter holding `|0>`, and the two are not one rule.** The working
carry wire is restored by the circuit, so a caller that meets the contract gets it
back clean. The carry-out wire is where the top bit of the exact integer sum
lands, so it is an output rather than scratch, and the adder writes it as an XOR
of whatever it entered holding. The difference between the two wires is measured
over every input rather than described: a dirty carry-out wire cannot change the
sum at all, because the adder only ever writes that wire at the end and as an XOR,
while a dirty carry wire is read by every majority gate in the forward sweep and
therefore produces a different ripple. Only ancilla-clean inputs are promised a
sum. **The map is still a permutation of the whole space**, so an off-contract
input is reversible rather than added — nothing is silently lost, and a
permutation is not an addition. That is a restriction the unit states rather than
one it hides, and it is pinned by a phrase in the runnable example's premise
paragraph.

**The T-count is not written here, and that is the unit's first property rather
than an implementation note.** One Toffoli is seven T through the compiler's own
identity in `flagquantum/compiler/basis_translation.py`, which expands `ccx` into
fifteen gates — two `h`, four `t`, three `tdg`, six `cx`. A seven-T expansion
written beside the adder would be a second source of truth for the one number the
two must agree on, so the circuit is handed to
`flagquantum.compiler.basis_conversion.convert_basis` and priced by
`flagquantum.compiler.resource_estimation.estimate_resources` and
`flagquantum.algorithms.logical_resources.estimate_logical_resources` instead.

```python
from flagquantum.algorithms import adder_circuit, estimate_logical_resources
from flagquantum.circuit import Circuit
from flagquantum.compiler.basis_conversion import BasisConversionError, convert_basis
from flagquantum.compiler.resource_estimation import estimate_resources

basis = ("cx", "h", "s", "sdg", "t", "tdg")
one = estimate_resources(convert_basis(Circuit(3).ccx(0, 1, 2), gates=basis).program)
print(one.n_operations, dict(one.operation_counts), one.t_count)
# 15 {'h': 2, 'cx': 6, 'tdg': 3, 't': 4} 7  -- one Toffoli is seven T through the compiler
lowered = {n: convert_basis(adder_circuit(n), gates=basis) for n in (1, 2, 3, 4)}
tally = {n: estimate_resources(lowered[n].program) for n in (1, 2, 3, 4)}
print([tally[n].t_count for n in (1, 2, 3, 4)])
# [14, 28, 42, 56]  -- 14 n T, because the adder holds 2 n Toffolis and nothing else
print([tally[n].n_operations for n in (1, 2, 3, 4)])
# [37, 73, 109, 145]  -- one operation becomes fifteen, so the count grows by 28 per bit
print([tally[n].depth for n in (1, 2, 3, 4)])
# [28, 55, 82, 109]  -- 27 n + 1 layers, the schedule of the identity rather than of the adder
report = estimate_logical_resources(lowered[4].program, distance=5)
print(
    report.t_count,
    report.physical_qubits,
    report.surface_code_cycles,
    report.spacetime_volume,
)
# 56 490 545 267050  -- and the patch count is a floor, not a compiled estimate
try:
    convert_basis(adder_circuit(10), gates=basis)
except BasisConversionError as exc:
    print(type(exc).__name__, exc)
# BasisConversionError named-basis decomposition exceeds max_added_operations
print(
    estimate_resources(
        convert_basis(
            adder_circuit(10), gates=basis, max_added_operations=1024
        ).program
    ).t_count
)
# 140  -- the same circuit lowers once the caller says how much expansion is acceptable
```

**What is absent beside it, named rather than left to be discovered.** Both addends
are quantum registers, so there is no classical addend and no `add_constant`; there
is no modular adder, no controlled adder, no comparison, no multiplier, and no
modular exponentiation. There is no carry-in parameter, so a caller chaining two
additions through the carry wire is doing something the constructor does not
describe. And the conversion's own default bound — `max_added_operations=256` — is
reached at ten bits, which is a bound on one conversion rather than a limit of the
construction, and the block above shows it yielding to the caller's own number. The
runnable example under `examples/algorithms/arithmetic.py` runs the same
construction and prints the same refusals, and
[`tests/test_algorithm_examples.py`](../../tests/test_algorithm_examples.py)
executes it.

## Sources

- Boros & Hammer, "Pseudo-Boolean optimization", *Discrete Applied Mathematics*
  **123**(1-3), 155-225 (2002), DOI 10.1016/S0166-218X(01)00341-9 — the
  substitution `x_i = (1 + s_i) / 2` and the pseudo-Boolean form of the
  objective.
- SPSA is attributed to J. C. Spall, "Multivariate stochastic approximation using
  a simultaneous perturbation gradient approximation", *IEEE Transactions on
  Automatic Control* **37**(3), 332-341 (1992), DOI 10.1109/9.119632 — the
  simultaneous-perturbation estimate and the two gain sequences this unit names
  `parameter_gain` and `perturbation`. The practical guidance — the default
  exponents `alpha = 0.602` and `gamma = 0.101`, the `A` in the step-size
  denominator, and the advice to choose `A` from the expected number of
  iterations rather than to tune it — is J. C. Spall, "An overview of the
  simultaneous perturbation method for efficient optimization", *Johns Hopkins
  APL Technical Digest* **19**(4), 482-492 (1998). The second is a technical
  digest rather than a peer-reviewed article and is cited for the practical
  defaults only, not for the convergence result, which is the 1992 paper's.
- Nelder-Mead simplex search is attributed to J. A. Nelder & R. Mead, "A Simplex
  Method for Function Minimization", *The Computer Journal* **7**(4), 308-313
  (1965), DOI 10.1093/comjnl/7.4.308 — the reflection, expansion, contraction and
  shrink coefficients this unit names, and the simplex of ``n + 1`` vertices it
  moves. The convergence record is a separate and later one, and it is the record
  this unit's boundary rests on rather than the 1965 paper: J. C. Lagarias, J. A.
  Reeds, M. H. Wright & P. E. Wright, "Convergence Properties of the Nelder--Mead
  Simplex Method in Low Dimensions", *SIAM Journal on Optimization* **9**(1),
  112-147 (1998), DOI 10.1137/S1052623496303470, which is also where the statement
  that essentially no theoretical results had been proved for this algorithm
  before it is written down. **What that record contains is exactly what this
  unit reports**: convergence to a minimizer in one dimension, limited results in
  two, and two counterexamples in two dimensions — a nonconvergence example and an
  example of convergence to a nonminimizer. The guide says the theory is
  incomplete in more than one dimension because that is what the record says, and
  the unit computes no rate, bound or interval of its own. The family of
  examples that converges to a nonstationary point is K. I. M. McKinnon,
  "Convergence of the Nelder--Mead Simplex Method to a Nonstationary Point",
  *SIAM Journal on Optimization* **9**(1), 148-158 (1998), DOI
  10.1137/S1052623496303482, and it is cited for the shape of the failure rather
  than for anything built here: no construction from it is implemented, which is
  why the collinear starting simplex its demonstration needs is refused by name
  instead of reproduced.

- The product formula is attributed to H. F. Trotter, "On the product of
  semi-groups of operators", *Proceedings of the American Mathematical Society*
  **10**(4), 545-551 (1959), DOI 10.1090/S0002-9939-1959-0108732-6 — the
  factorization a product of exponentials approximates. The general decomposition
  this unit's two orders are instances of is M. Suzuki, "Fractal decomposition of
  exponential operators with applications to many-body theories and Monte Carlo
  simulations", *Physics Letters A* **146**(6), 319-323 (1990), DOI
  10.1016/0375-9601(90)90962-N, and M. Suzuki, "General theory of fractal path
  integrals with applications to many-body theories and statistical physics",
  *Journal of Mathematical Physics* **32**(2), 400-407 (1991), DOI
  10.1063/1.529425 — the symmetric composition whose leading defect cancels, which
  is what the measured quartering above is the rate of. **The two orders here are
  the ones this unit builds and not a claim about the theory**: the higher orders
  those papers develop are deliberately absent, and the module records why.
- Quantum simulation by exponentiating a Hamiltonian's terms is recorded to S.
  Lloyd, "Universal Quantum Simulators", *Science* **273**(5278), 1073-1078 (1996),
  DOI 10.1126/science.273.5278.1073 — the local-decomposition construction this
  unit's circuit is the classical description of. It is cited for that
  construction and not for a resource result.
- Barahona, *J. Phys. A* **15**(10), 3241-3253 (1982),
  DOI 10.1088/0305-4470/15/10/028 — the ground state of a spin glass in a field
  is NP-hard, which is why the mapping is interesting to a solver at all.
- Lucas, "Ising formulations of many NP problems", *Frontiers in Physics* **2**,
  5 (2014), arXiv:1302.5843 — problem-level formulations, including MaxCut.
- Möttönen, Vartiainen, Bergholm & Salomaa, "Transformation of quantum states
  using uniformly controlled rotations", *Quantum Information and Computation*
  **5**, 467 (2005), arXiv:quant-ph/0407010 — the uniformly controlled rotation
  ladder and the analytical formula for its angles, which
  `primitives/state_preparation.py` uses.
- Phase estimation is attributed to A. Yu. Kitaev, "Quantum measurements and the
  Abelian Stabilizer Problem", arXiv:quant-ph/9511026 (1995), a preprint. The
  circuit form built by `primitives/phase_estimation.py` is the one recorded by
  Brassard, Høyer, Mosca & Tapp, "Quantum Amplitude Amplification and
  Estimation", *AMS Contemporary Mathematics* **305**, 53–74 (2002),
  DOI 10.1090/conm/305/05215, arXiv:quant-ph/0005055 — the controlled powers of
  the unitary and the inverse Fourier transform on the counting register.
- Grover search follows Lov K. Grover, "A fast quantum mechanical algorithm for
  database search", *Proc. 28th Annual ACM Symposium on Theory of Computing
  (STOC '96)*, pp. 212-219, 1996, DOI 10.1145/237814.237866,
  arXiv:quant-ph/9605043 — the amplitude-amplification iteration, the inversion
  about the average it is built from, and the `O(sqrt(N))` query count that
  `grover.py` reports.
- Quantum PCA follows Seth Lloyd, Masoud Mohseni & Patrick Rebentrost, "Quantum
  principal component analysis", *Nature Physics* **10**, 631-633 (2014),
  DOI 10.1038/nphys3029, arXiv:1307.0401 — the density-matrix exponential whose
  eigenphases carry the eigenvalues, the purification the subroutine is applied
  to, and the `O(1/eps**3)` copies of `rho` that the paper's input model counts
  and this unit's classical formation of `rho` replaces.
- Quantum k-medians follows Esma Aïmeur, Gilles Brassard & Sébastien Gambs,
  "Quantum clustering algorithms", *Proceedings of the 24th International
  Conference on Machine Learning (ICML 2007)*, pp. 1-8, DOI
  10.1145/1273496.1273497; journal version *Machine Learning* **90**(2), 261-287
  (2012), DOI 10.1007/s10994-012-5316-5 — the quantization of k-medians by a
  Grover-style minimum search over the centroids, which `kmedians.py`
  implements, and the k-medians median update, which stays classical here.
  **No arXiv identifier is attached to that paper**: this programme's citation
  check recorded that it has no arXiv version, so the record above carries a DOI
  and a journal version and nothing else.
- Quantum kernel estimation follows Vojtěch Havlíček, Antonio D. Córcoles,
  Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow & Jay M.
  Gambetta, "Supervised learning with quantum-enhanced feature spaces", *Nature*
  **567**, 209–212 (2019), DOI 10.1038/s41586-019-0980-2 — the kernel-matrix
  circuit, the swap test whose ancilla is found set with probability
  `1/2 - 1/2 |<a|b>|^2`, which `quantum_kernel.py` estimates its entries with,
  and the `O(eps**-2)` sampling cost per entry that the module's boundary
  repeats. **Two things this record does not say.** The paper's classical-hardness
  statement is a conjecture and not a theorem, and the paper is written for
  noisy intermediate-scale devices — it does not require fault tolerance, and no
  entry in this guide says that it does.
- The fault-tolerance requirement is recorded separately, to Yunchao Liu,
  Srinivasan Arunachalam & Kristan Temme, "A rigorous and robust quantum
  speed-up in supervised machine learning", *Nature Physics* **17**, 1013–1017
  (2021), DOI 10.1038/s41567-021-01287-z — the rigorous speed-up result for
  quantum kernel methods, which is proved for a fault-tolerant quantum computer.
  It is cited for that requirement and for nothing else here.
- The bit-string comparator follows D. S. Oliveira & R. V. Ramos, "Quantum bit
  string comparator: circuits and applications", *Quantum Computers and
  Computing* **7**(1), 17-26 (2007) — **this record is not index-confirmed.**
  Its venue is not indexed by Crossref, DBLP, or INSPIRE, so the volume and
  page numbers are reported by citing works rather than confirmed against an
  index. A fully verified adjacent record by the same group is D. S. Oliveira,
  P. B. M. de Sousa & R. V. Ramos, 2006 International Telecommunications
  Symposium, DOI 10.1109/ITS.2006.4433341.
- Multi-controlled X is standard reversible logic and is cited to no paper here,
  matching the record kept for it: the ancilla ladder it is built as is a
  textbook construction, and the index above carries no citation for it.
- Feature selection posed as a binary objective for an annealer is recorded to
  Ferrari Dacrema, Moroni, Nembrini, Ferro, Faggioli & Cremonesi, SIGIR 2022, DOI
  10.1145/3477495.3531755, arXiv:2205.04346 — the route `feature_selection.py`
  follows, in this package's own spelling of the objective. **The one figure from
  that record, written as the paper's own:** the largest problem that paper solved
  directly on the QPU had 124 features. It is not a measurement of this unit,
  which solves nothing.
- Frequent-item fractions by amplitude estimation follow Yu, Gao, Wang & Wen, *Physical
  Review A* **94**(4), 042311 (2016), DOI 10.1103/PhysRevA.94.042311,
  arXiv:1605.07444v3 — the amplitude-estimation route to a transactional database's
  frequent itemsets, of which `qarm.py` reads out the fraction of the items whose support
  meets a threshold. **This record carries no title**: the citation this programme
  verified fixes the authors, the venue, the DOI and the arXiv identifier, and no title
  was confirmed with it. **Two things it does not say.** Its improvement is quadratic in
  the number of database queries and it is conditional, stated for the case
  `M_f^(k) << M_c^(k)`; it is not exponential, and the exponential wording appears only in
  an earlier arXiv listing of the same work, which is that paper rather than a second one.
- Singular values by phase estimation follow Kerenidis & Prakash, "Quantum Recommendation
  Systems", ITCS 2017, LIPIcs Vol. 67, 49:1-49:21, DOI 10.4230/LIPIcs.ITCS.2017.49 — the
  singular-value-estimation subroutine a recommendation algorithm is built on, and the
  query model this unit's readout is not taken under. **The DOI is registered with
  DataCite and not with Crossref**: a Crossref lookup of it returns nothing, which is a
  fact about the registrar and not a reason to replace the identifier or to drop it.
- The singular value decomposition of non-sparse low-rank matrices is the subject of
  Rebentrost, Steffens, Marvian & Lloyd, *Physical Review A* **97**(1), 012327 (2018),
  DOI 10.1103/PhysRevA.97.012327, approached there by exponentiating the matrix. **The
  author list recorded here is the published one, and it has four authors.** The arXiv
  listing of the same work, arXiv:1607.05404, carries three: Marvian is on the journal
  version only, so a three-author spelling belongs to the preprint and a four-author
  spelling to the published paper.
- The block encoding is that of Gilyén, Su, Low & Wiebe, "Quantum singular value
  transformation and beyond: exponential improvements for quantum matrix arithmetics",
  STOC 2019, pp. 193-204, DOI 10.1145/3313276.3316366, **Definition 1** — the definition
  and the subnormalisation convention this unit's `alpha` follows. That definition's
  neighbourhood is not the source of the term: the string does not appear in Childs &
  Wiebe, and that work is not cited for it here.
- Dequantization is recorded to Tang, STOC 2019, DOI 10.1145/3313276.3316310 — the
  classical algorithm for the recommendation problem that removes the exponential
  speed-up, and which is **"only polynomially slower"**: its bound contains `eps**-12`,
  and the author calls it "a large slowdown in some exponents". **This entry carries no
  title.** It does not say that a classical algorithm matches the quantum runtime, and
  nothing in this guide says so either.
- The two records that qualify it are kept separately. The practical conditions the
  dequantized algorithms need are Arrazola et al., *Quantum* **4**, 307 (2020) — **this
  entry carries no title**. Gharibian–Le Gall, STOC 2022 / SICOMP **52**(4) — **this
  entry carries no title either**, and no DOI is attached to it here — dequantize the
  quantum singular value transformation for sparse matrices at constant precision; their
  hardness result is for a different task, estimating a local Hamiltonian's ground-state
  energy at inverse-polynomial precision given a state close to the ground state.

- The rotated surface code, its patch, and its logical operators are recorded to
  Fowler, Mariantoni, Martinis & Cleland, "Surface codes: Towards practical
  large-scale quantum computation", *Physical Review A* **86**(3), 032324 (2012),
  DOI 10.1103/PhysRevA.86.032324 — the `d**2` data and `d**2 - 1` measure qubit
  patch and the `d` cycles per logical operation that
  `flagquantum/algorithms/logical_resources.py` costs a program on. **The citation
  covers the code model and not the resource estimate**: the operation counts, the T
  family, and the schedule depth are this repository's own compiler record, and the
  estimator cites it there rather than here.

- The single-ancilla ripple-carry adder is recorded to Cuccaro, Draper, Kutin &
  Moulton, "A new quantum ripple-carry addition circuit", arXiv:quant-ph/0410184
  (2004) — the majority/unmajority pair, the single ancillary qubit, and the
  linear depth that `flagquantum/algorithms/arithmetic.py` builds, whose abstract
  states the two properties this unit's construction is named for: one ancillary
  qubit rather than linearly many, and lower depth and fewer gates than the
  ripple-carry adders before it. **The citation covers the construction and not
  the cost**: the Toffoli identity the T-count comes from is this repository's own
  compiler expansion in `flagquantum/compiler/basis_translation.py`, cited there
  rather than here. The Gidney-Ekera construction the parity matrix names on the
  same row is **not implemented and is not cited here**: it is an owned gap, and a
  bibliographic entry for a construction this repository does not build would
  read as a second implementation.

## Scope

Everything here is a demonstration-scale, teaching-oriented construction. No
unit in this guide makes a performance claim, a capacity claim, or a
quantum-advantage claim, and none of them certifies solver behavior,
convergence, or hardware behavior. The one exception is a cost claim and it is
stated as one: `spsa.py` spends a measured two objective evaluations per step at
every parameter count it was tested at, and `nelder_mead.py` reports the
objective calls a run spent beside the bracket those calls lie in. Both are
counts of a unit's own calls and not performance results, and neither is a
convergence result. The trajectories its section reports are runs of one
two-parameter objective, and they certify nothing about convergence in general —
the unit's own `capability-maturity.toml` entry states the boundary, and the
spread across seeds that the section reports is the estimator's variance rather
than a tolerance. A unit is admitted to the index with the
tests that exercise it and the boundary that limits it: the units classified in
`capability-maturity.toml` record both there, and the two rows without an entry
— the Fourier transform and phase estimation — carry their advantage premise in
the index table above and their tests in `tests/unit/test_algorithms_qft.py` and
`tests/unit/test_algorithms_phase_estimation.py`.
