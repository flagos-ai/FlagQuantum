"""State a feedback condition that fires on either of two readouts.

Run with ``python -m examples.dynamic_disjunctive_condition``.

A mid-circuit condition in CUDA-Q is a boolean function of measured bits, and a
conjunction covers only the functions that hold on every tested bit at once. A
program such as "flip the flag if either readout came back one" is not a
conjunction, so ``DynamicCircuit.conditional`` takes ``any_of``: one mapping per
clause, and the gate runs when any clause holds.

The three spellings are not interchangeable, so this example runs one state
through a conjunction and a disjunction and reports both answers next to the
analytic probability.
"""

import math

from flagquantum.dynamic import DynamicCircuit
from flagquantum.experimental.dynamic import run_dynamic


def program(*, any_of=None, conditions=None):
    """Rotate two readout qubits, measure them, and condition a third on them."""

    circuit = (
        DynamicCircuit(3)
        .ry(0, theta=0.7)
        .ry(1, theta=1.1)
        .measure(0, classical_bit=0)
        .measure(1, classical_bit=1)
    )
    if any_of is not None:
        return circuit.conditional("x", 2, any_of=any_of)
    return circuit.conditional("x", 2, conditions=conditions)


either = program(any_of=({0: 1}, {1: 1}))
both = program(conditions={0: 1, 1: 1})
print(
    "disjunction clauses:",
    either.to_ir().instructions[-1].metadata["condition_clauses"],
)
print("conjunction clause: ", both.to_ir().instructions[-1].metadata["conditions"])

# P(bit 0 = 1) = sin(0.35)**2 and P(bit 1 = 1) = sin(0.55)**2, independently,
# because the two rotations act on separate qubits. A disjunction fires on the
# union of those events and a conjunction on their intersection.
first = math.sin(0.35) ** 2
second = math.sin(0.55) ** 2
union = first + second - first * second
intersection = first * second

shots = 200_000
for label, circuit, expected in (
    ("either readout", either, union),
    ("both readouts", both, intersection),
):
    # The exact route and the sampled route evaluate the same clauses, so the
    # two are compared rather than one being trusted on its own. The diagonal is
    # indexed with qubit 0 most significant, so the flag qubit -- the last one --
    # is the least significant bit, and the odd indices are the states it is set in.
    diagonal = circuit.density_matrix()[0].diagonal()
    exact = float(diagonal[1::2].sum().real)
    result = run_dynamic(circuit, shots=shots, seed=3)
    sampled = int(result.final_samples[:, 2].sum()) / shots
    print(
        f"{label}: exact {exact:.6f}, sampled {sampled:.4f} of {shots} shots, "
        f"analytic {expected:.6f}, "
        f"within three standard errors "
        f"{abs(sampled - expected) < 3 * math.sqrt(expected * (1 - expected) / shots)}"
    )

# A single clause is what `conditions` already states, so `any_of` refuses it
# rather than picking one of the two spellings for the caller.
try:
    program(any_of=({0: 1},))
except ValueError as error:
    print("one clause:", error)

# A disjunction whose clauses are not distinct and minimal describes a branch no
# trajectory can reach, so the builder refuses it instead of charging the executor
# for the unreachable clause. The second clause here contains the first.
try:
    program(any_of=({0: 1}, {0: 1, 1: 0}))
except ValueError as error:
    print("redundant clause:", error)

# `measure` and `reset` project a qubit and write or discard a classical outcome
# rather than applying a matrix, so neither has a conditional form to lower.
try:
    DynamicCircuit(2).measure(0, classical_bit=0).conditional(
        "reset", 1, classical_bit=0
    )
except ValueError as error:
    print("conditional reset:", error)

# A disjunction is not a sequence of conditionals. Appending one conditioned gate
# per clause flips the flag twice on the branch where both readouts came back one,
# which is the exclusive or of the clauses rather than their union, so the two
# programs answer differently on exactly that branch.
one_clause_each = (
    DynamicCircuit(3)
    .ry(0, theta=0.7)
    .ry(1, theta=1.1)
    .measure(0, classical_bit=0)
    .measure(1, classical_bit=1)
    .conditional("x", 2, conditions={0: 1, 1: 0})
    .conditional("x", 2, conditions={0: 0, 1: 1})
)
exclusive = first * (1 - second) + (1 - first) * second
stated = (
    int(run_dynamic(one_clause_each, shots=shots, seed=3).final_samples[:, 2].sum())
    / shots
)
print(
    f"two conditioned gates: sampled {stated:.4f}, analytic {exclusive:.4f}, "
    f"against the disjunction's {union:.4f}"
)
