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
| `primitives/qft.py` — quantum Fourier transform | Available as a shared primitive; no algorithm unit consumes it yet. | — | None. It is a subroutine. |
| `primitives/phase_estimation.py` — phase estimation | Not yet available. | — | — |
| `primitives/state_preparation.py` — state preparation | Not yet available. | — | — |
| `primitives/oracle.py` — oracle synthesis | Not yet available. | — | — |
| `grover.py` — Grover search | Not yet available. | — | — |
| `amplitude_estimation.py` — amplitude estimation | Not yet available. | — | — |

Rows marked "not yet available" are placeholders for later units of this
programme; they carry no content yet. The primitive row is listed separately
because a shared primitive is admitted here only when at least two algorithm
modules need it, and it does not by itself change what a caller can run.

## Advantage premises

- **The QUBO mapping carries no advantage premise.** It is a polynomial
  classical transformation between two encodings of the same objective. It
  produces no speedup of its own: any advantage a caller observes belongs to the
  solver that consumes the Hamiltonian, not to this mapping. The module says so
  in its own docstring, and the capability entry repeats it as its boundary.
- **The primitives this mapping will be joined by carry none either.** The
  quantum Fourier transform, phase estimation, oracle synthesis, and state
  preparation are subroutines. They do not have an advantage premise to state:
  their cost is paid by whatever algorithm calls them.
- **Grover search and amplitude estimation are query/oracle-model results.**
  Their advantage premise is a cheap oracle — and, for amplitude estimation, a
  cheap state-preparation operator as well. A gate-level implementation pays
  both costs rather than assuming them away: writing an oracle for an arbitrary
  predicate costs on the order of `2**n` gates, and the state-preparation
  operator is not free. Neither unit yields an end-to-end advantage at
  demonstration scale.
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

The converted Hamiltonian is an ordinary `Hamiltonian`, so it goes straight into
the existing workflows: `vqe_loss`, `qaoa_circuit`, and `run_vqe` accept it
without a new execution path. The constant is carried as one identity term and
recovered as `QuboProblem.offset`, so the two forms of the same problem agree on
every assignment.

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

## Scope

Everything here is a demonstration-scale, teaching-oriented construction. No
unit in this guide makes a performance claim, a capacity claim, or a
quantum-advantage claim, and none of them certifies solver behavior,
convergence, or hardware behavior. A unit is admitted to the index only with the
tests that exercise it and the boundary that limits it, both recorded in
`capability-maturity.toml`.
