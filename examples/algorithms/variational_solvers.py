"""Fit QAOA angles to a weighted MaxCut graph, and read what the fit is worth.

`flagquantum.algorithms.variational` is reachable through the subpackage surface
only -- `from flagquantum.algorithms import run_qaoa` -- because the algorithms
package adds no root-level `fq.` name.

The premise the unit rests on, and cannot check: the cost layer the ansatz
applies and the operator the energy is measured against describe one graph. A
QAOA circuit is built from a list of weighted ``ZZ`` edges and an energy is an
expectation against some Hamiltonian, and nothing inside either one relates the
two, so a caller can build the ansatz for one graph and score it against another
and every number that comes back is then a number about a problem nobody posed.
`run_qaoa` and `maxcut_hamiltonian` exist so the edge list is checked once and
reaches both sides through the same tuple. What the unit therefore does not do is
check the premise for anyone else: a caller who assembles their own circuit and
their own operator through `qaoa_circuit` and `qaoa_loss` has the freedom this
entry point removes, and nothing warns them.

Two halves of the premise are the caller's rather than the unit's. The uniform
superposition the first layer acts on is an eigenstate of every cost and mixer
term, so the objective's gradient at a zero start is exactly zero at every layer
count: a zero start cannot move, and a solver whose default were the origin would
return the origin as an optimum. The start is therefore a required argument, and
the saddle is shown here rather than asserted. And a finite layer count is an
approximation: the energy is an expectation rather than a bound, so a value above
the operator's ground energy is the normal outcome and a larger layer count is
not guaranteed to be closer to it. This instance happens to improve with depth at
the start used here, which is a fact about this instance and this start.

The instance is a weighted MaxCut graph on four nodes with five unit-weight
edges -- a square with one diagonal -- because its optimum is reached exactly by
three layers at a small constant start, so the layer progression from two
distant values to the optimum is visible in one run. The brute-force spectrum is
enumerated here so the ground energy is checked against the cut count rather than
trusted, and the cut is decoded from the optimized circuit's own probability
distribution rather than from the energy, because the energy is a cost and not a
cut.

Run it with:

    python -m examples.algorithms.variational_solvers
"""

from __future__ import annotations

import itertools
from typing import Any

import torch

from flagquantum.algorithms import (
    QAOAResult,
    maxcut_hamiltonian,
    qaoa_circuit,
    qaoa_loss,
    run_qaoa,
)
from flagquantum.circuit import Circuit

#: A square with one diagonal: four nodes, five edges, largest cut 4 of 5.
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2))
N_QUBITS = 4
#: One constant start, one entry per angle, so the layer count is the only thing
#: that changes between the runs below.
START = 0.1
LAYERS = (1, 2, 3)
#: The two cuts that split the square and the diagonal's endpoints apart.
MAXIMUM_CUTS = ((0, 1, 0, 1), (1, 0, 1, 0))
LABEL_WIDTH = 30


def brutal_cut_spectrum() -> dict[tuple[int, ...], int]:
    """Return the cut size of every assignment, enumerated rather than trusted."""

    return {
        bits: sum(1 for source, target in EDGES if bits[source] != bits[target])
        for bits in itertools.product((0, 1), repeat=N_QUBITS)
    }


def basis_circuit(bits: tuple[int, ...]) -> Circuit:
    """Return the circuit preparing one computational basis state."""

    circuit = Circuit(len(bits))
    for wire, bit in enumerate(bits):
        if bit:
            circuit = circuit.x(wire)
    return circuit


def probabilities(result: QAOAResult) -> dict[tuple[int, ...], float]:
    """Return the optimized circuit's own distribution over bitstrings."""

    state = (
        qaoa_circuit(N_QUBITS, EDGES, result.gammas, result.betas).state().reshape(-1)
    )
    mass = (state.abs() ** 2).real
    return {
        tuple(
            (int(index) >> (N_QUBITS - 1 - wire)) & 1 for wire in range(N_QUBITS)
        ): float(mass[index])
        for index in range(mass.numel())
    }


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def refuse(description: str, action: Any) -> None:
    """Run ``action``, print the refusal it must raise, and fail if it returns."""

    try:
        action()
    except ValueError as error:
        report(description, f"{error}")
    else:
        raise SystemExit(
            f"{description} was accepted, and the example exists to refuse it"
        )


def main() -> None:
    cost = maxcut_hamiltonian(N_QUBITS, EDGES)
    ground = float(cost.ground_energy())
    spectrum = brutal_cut_spectrum()
    largest = max(spectrum.values())
    best = {bits for bits, cut in spectrum.items() if cut == largest}

    print("=" * 72)
    print("variational solvers -- flagquantum.algorithms.variational")
    print("=" * 72)
    report("task", "fit QAOA angles to a weighted MaxCut graph")
    report("premise", "the cost layer and the operator describe one graph,")
    print(f"  {'':<{LABEL_WIDTH}}  because the edges are checked once and reach both")
    print(f"  {'':<{LABEL_WIDTH}}  sides through the same tuple, and a caller who")
    print(f"  {'':<{LABEL_WIDTH}}  assembles them separately has a freedom this unit")
    print(f"  {'':<{LABEL_WIDTH}}  does not remove; the start is the caller's, because")
    print(f"  {'':<{LABEL_WIDTH}}  the uniform superposition is a stationary point and")
    print(f"  {'':<{LABEL_WIDTH}}  a zero start cannot move; and a finite layer count")
    print(
        f"  {'':<{LABEL_WIDTH}}  is an approximation, so the energy is not a bound on"
    )
    print(f"  {'':<{LABEL_WIDTH}}  the ground energy and a larger layer count is not")
    print(f"  {'':<{LABEL_WIDTH}}  guaranteed to be closer to it")
    print()

    print("inputs")
    report("graph", f"{N_QUBITS} nodes, {len(EDGES)} edges")
    report("edges", ", ".join(f"({a}, {b})" for a, b in EDGES))
    report("objective", f"{len(EDGES)} ZZ terms of weight 1.0")
    report("start", f"{START} at every angle, so only the layer count changes")
    report("optimizer", "Adam at lr 0.05 for 100 steps")
    print()

    print("the operator stands for the cut, so the two are checked against each other")
    report("largest cut (enumerated)", f"{largest} of {len(EDGES)} edges")
    report("cut at the optimum", str(sorted(best)))
    report("ground energy", f"{ground:.17g}")
    report("total less twice the cut", f"{len(EDGES) - 2 * largest:.17g}")
    values = {bits: float(cost.expectation(basis_circuit(bits))) for bits in spectrum}
    report("terms", str([(float(t.coefficient), t.wires) for t in cost.terms]))
    report(
        "argmin equals argmax",
        str({bits for bits, value in values.items() if value == ground} == best),
    )
    print(f"  {'':<{LABEL_WIDTH}}  -- on a basis state the operator's value is the")
    print(f"  {'':<{LABEL_WIDTH}}  edge total less twice the cut, so minimizing it")
    print(f"  {'':<{LABEL_WIDTH}}  maximizes the cut and the energy is a cost rather")
    print(f"  {'':<{LABEL_WIDTH}}  than a cut size")
    print()

    print("a zero start is a stationary point, so the start is the caller's")
    saddle = run_qaoa(N_QUBITS, EDGES, torch.zeros(2 * len(LAYERS)))
    parameters = torch.zeros(2 * len(LAYERS), requires_grad=True)
    loss = qaoa_loss(
        N_QUBITS,
        EDGES,
        parameters[: len(LAYERS)],
        parameters[len(LAYERS) :],
        cost,
    )
    torch.autograd.backward(loss)
    report("gradient at the zero start", str(parameters.grad.tolist()))
    report("energy at the zero start", f"{float(saddle.energy):.17g}")
    report("angles returned", str(saddle.parameters.tolist()))
    report("steps taken", f"{saddle.n_steps}")
    print(f"  {'':<{LABEL_WIDTH}}  -- the first layer's cost and mixer both act on")
    print(f"  {'':<{LABEL_WIDTH}}  the uniform superposition, which is an eigenstate")
    print(f"  {'':<{LABEL_WIDTH}}  of every ZZ and every X, so every gradient is")
    print(f"  {'':<{LABEL_WIDTH}}  exactly zero and the zero start cannot move; a")
    print(f"  {'':<{LABEL_WIDTH}}  solver defaulting to it would report an optimum")
    print(f"  {'':<{LABEL_WIDTH}}  that is only a saddle")
    print()

    print(f"one start of {START}, three layer counts")
    results = {
        layers: run_qaoa(N_QUBITS, EDGES, torch.full((2 * layers,), START))
        for layers in LAYERS
    }
    report("layers", "  energy        gap         top two cuts")
    for layers, result in results.items():
        distribution = probabilities(result)
        ordered = sorted(distribution, key=lambda bits: -distribution[bits])
        mass = sum(distribution[bits] for bits in ordered[:2])
        report(
            f"  {layers}",
            f"{float(result.energy):>12.9f}  "
            f"{float(result.energy) - ground:>9.3e}  {mass:>9.6f}",
        )
    print(f"  {'':<{LABEL_WIDTH}}  -- three layers reach the optimum at this start")
    print(f"  {'':<{LABEL_WIDTH}}  and one layer does not, which is a fact about")
    print(f"  {'':<{LABEL_WIDTH}}  this instance and this start rather than a law: a")
    print(f"  {'':<{LABEL_WIDTH}}  finite layer count is an approximation and the")
    print(f"  {'':<{LABEL_WIDTH}}  energy is not a bound on the ground energy")
    print()

    print("the cut is read from the circuit, not from the energy")
    distribution = probabilities(results[LAYERS[-1]])
    for bits in MAXIMUM_CUTS:
        report(f"  mass at {bits}", f"{distribution[bits]:.12f}")
    heavy = sorted(bits for bits, mass in distribution.items() if mass > 0.1)
    report("bitstrings above 0.1", str(heavy))
    report("those are the largest cuts", str(heavy == sorted(best)))
    print(f"  {'':<{LABEL_WIDTH}}  -- the two maximum cuts carry almost all of the")
    print(f"  {'':<{LABEL_WIDTH}}  mass, and the assignment is decoded from the")
    print(f"  {'':<{LABEL_WIDTH}}  state; the result carries angles and an energy")
    print(f"  {'':<{LABEL_WIDTH}}  and no cut, so this step is the caller's")
    print()

    print("what is refused")
    refuse(
        "one cut edge listed twice",
        lambda: maxcut_hamiltonian(N_QUBITS, ((0, 1), (1, 0))),
    )
    refuse(
        "an edge to itself",
        lambda: maxcut_hamiltonian(N_QUBITS, ((1, 1),)),
    )
    refuse(
        "no optimizer step at all",
        lambda: run_qaoa(N_QUBITS, EDGES, torch.full((2,), START), steps=0),
    )
    refuse(
        "an odd number of angles",
        lambda: run_qaoa(N_QUBITS, EDGES, torch.full((3,), START)),
    )
    refuse(
        "a start that is not flat",
        lambda: run_qaoa(N_QUBITS, EDGES, torch.full((2, 2), START)),
    )
    print()

    print("take away")
    print("  QAOA needs a cost operator, an ansatz, a start, and an optimizer, and")
    print("  the first two have to agree about the graph. What this unit adds to")
    print("  the primitives beside it is that agreement: one checked edge list")
    print("  reaches the circuit and the operator, and a solver over them returns")
    print("  the fitted angles rather than an angle sweep. The premise it assumes")
    print("  is the caller's and the two things it gives up are stated: the uniform")
    print("  superposition is a stationary point, so the start is required and a")
    print("  zero start cannot move, and a finite layer count is an approximation,")
    print("  so the energy is a cost to compare rather than a bound to trust. What")
    print("  it does not do is sample: the mass above is the exact distribution,")
    print("  and turning it into shots, ranking sampled cuts, and comparing a")
    print("  sampled best against the optimum are all steps this unit leaves to the")
    print("  caller.")


if __name__ == "__main__":
    main()
