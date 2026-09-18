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
| `primitives/oracle.py` — oracle building blocks | Available. Multi-controlled X and a reversible bit-string comparator, the reversible logic the oracle units are built from. Truth-table oracle synthesis on top of them is not yet available. | Oliveira & Ramos 2007 (record not index-confirmed) | **None.** Reversible classical logic, `O(n)` Toffoli-style. |
| `grover.py` — Grover search | Not yet available. | — | — |
| `amplitude_estimation.py` — amplitude estimation | Not yet available. | — | — |

Rows marked "not yet available" are placeholders for later units of this
programme; they carry no content yet. The primitive rows are listed separately
because the package admits a shared primitive only when more than one algorithm
module is expected to need it: the Fourier transform has one consumer today,
phase estimation, and gains a second when amplitude estimation lands. A
primitive does not by itself change what a caller can run.

## Advantage premises

- **The QUBO mapping carries no advantage premise.** It is a polynomial
  classical transformation between two encodings of the same objective. It
  produces no speedup of its own: any advantage a caller observes belongs to the
  solver that consumes the Hamiltonian, not to this mapping. The module says so
  in its own docstring, and the capability entry repeats it as its boundary.
- **The other primitives carry none either.** The quantum Fourier transform,
  phase estimation, and state preparation have shipped as subroutines, and
  oracle synthesis will join them as one more. They do not have an advantage
  premise to state: their cost is paid by whatever algorithm calls them.
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

## Oracle building blocks

`primitives/oracle.py` holds the two pieces of reversible classical logic the
oracle units are composed from. `append_multi_controlled_x` flips one target
wire exactly on the operand pattern that sets every control; one control is a
`cx` and two are a `ccx`, and three or more are built as an ancilla ladder,
because the circuit layer has no native gate above two controls.
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
which is the same `|0>`-on-entry requirement one level down.

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

Truth-table oracle synthesis on top of these blocks, and the Grover search that
consumes it, are not available yet; the index above reserves their rows.

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
convergence, or hardware behavior. A unit is admitted to the index only with the
tests that exercise it and the boundary that limits it, both recorded in
`capability-maturity.toml`.
