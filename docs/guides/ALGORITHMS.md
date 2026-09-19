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
| `primitives/oracle.py` — oracle synthesis | Available. Synthesizes a phase or bit oracle from a classical predicate's truth table, on top of the multi-controlled X and comparator building blocks. | — | **None, and the cost is exponential.** Synthesis enumerates all `2**n` inputs classically. |
| `grover.py` — Grover search | Available. Amplifies the amplitude of the states a predicate marks, so a marked state is recovered from far fewer samples than uniform sampling needs. | Grover 1996 | **Query model.** The oracle's own cost is not counted; here it is a truth table, so no end-to-end advantage at demonstration scale. |
| `amplitude_estimation.py` — amplitude estimation | Available. Estimates the amplitude a marking operator selects, by phase estimation over the Grover operator. | Brassard et al. 2002 | **The state-preparation unitary is assumed free.** A real distribution needs QRAM, so this is not an end-to-end advantage. |

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
was admitted, with no consumer inside `flagquantum/` at all. A primitive does
not by itself change what a caller can run.

The Phase 2 quantum machine learning units are indexed in the table under
[Quantum machine learning units (Phase 2)](#quantum-machine-learning-units-phase-2)
below, which is where each unit of that phase adds its row.

## Quantum machine learning units (Phase 2)

These rows are the Phase 2 quantum machine learning units. They are tabulated
apart from the index above so that a phase's own set can be read together, and
each unit of this phase appends its row here as it lands. The same promise
applies to them as to everything else in this guide.

| Unit | What it does | Citation | Advantage premise |
| --- | --- | --- | --- |
| `pca.py` — quantum PCA | Available. Estimates the eigenvalues of a data matrix's density matrix from a purification of it, by phase-estimating `exp(-2 pi i rho)` and reading the counting register. | Lloyd et al. 2014 | **The premise is the input model, and it is not met.** The paper's subroutine consumes copies of `rho` and never forms it, in `O(1/eps**3)` of them; this unit forms `rho` classically, builds its exponential as a dense matrix, and takes the purification's `2**n` amplitudes from the caller. No end-to-end advantage follows. |
| `kmedians.py` — quantum k-medians | Available. Assigns each point to its nearest centroid with a Grover-style minimum search over a centroid index register, then moves each centroid to the classical coordinate-wise median of its cluster. | Aïmeur et al. 2007 | **The premise is the oracle model, and the oracle is not free.** The search's oracle is synthesized from the predicate's truth table at `O(2**n)` cost, and the distance table the predicate compares is computed classically, one point at a time, before any circuit is built. No end-to-end advantage follows. |

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
- **The variational workflows the Hamiltonian feeds are heuristics.** The
  existing VQE and QAOA paths in [core.py](../../flagquantum/algorithms/core.py)
  are approximation heuristics. Their evidence and their boundary are the ones
  already recorded for local statevector execution; this guide adds no
  asymptotic advantage conclusion for them.

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

**The distance table is classical, and that is the whole of what the unit gives
up.** `grover_circuit` builds its own register and refuses more than three
evaluation wires, so the search runs over the centroid index alone and the
distance from the point being assigned to each centroid is computed in double
precision outside the circuit — the predicate closes over that table, and the
register is rebuilt for the next point. The circuit holds no state about the
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
tried at one shot, for 23 at two shots, for 8 at four, and for none at 16 shots
and above. That is a measurement at those seeds and those shot counts, not a
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
counts oracle calls, and the call it counts evaluates a distance in one step;
its k-medians is a query-complexity result against that oracle. Here the oracle
is synthesized from the predicate's truth table, so it costs `O(2**n)` — over
the at most eight register values the three-wire search can carry — and the
distance table the predicate compares is computed classically, one point at a
time, before any circuit is built. The classical evaluation the search stands in
for is therefore already paid for, and nothing here reads a qRAM or runs an
adiabatic evolution, so no conclusion that rests on either applies. The unit is
bounded at eight centroids by the three-wire register, and the search is sampled
rather than read out, so a small sample can stop a point's search short. The
capability entry repeats the boundary.

## Sources

- Boros & Hammer, "Pseudo-Boolean optimization", *Discrete Applied Mathematics*
  **123**(1-3), 155-225 (2002), DOI 10.1016/S0166-218X(01)00341-9 — the
  substitution `x_i = (1 + s_i) / 2` and the pseudo-Boolean form of the
  objective.
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
  check recorded that it has no arXiv version, so the identifier once attached
  to it here was the wrong one and is not restored.
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

## Scope

Everything here is a demonstration-scale, teaching-oriented construction. No
unit in this guide makes a performance claim, a capacity claim, or a
quantum-advantage claim, and none of them certifies solver behavior,
convergence, or hardware behavior. A unit is admitted to the index with the
tests that exercise it and the boundary that limits it: the units classified in
`capability-maturity.toml` record both there, and the two rows without an entry
— the Fourier transform and phase estimation — carry their advantage premise in
the index table above and their tests in `tests/unit/test_algorithms_qft.py` and
`tests/unit/test_algorithms_phase_estimation.py`.
