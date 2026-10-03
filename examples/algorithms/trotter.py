"""Turn a Hamiltonian into the circuit a product formula applies, and measure the gap.

`flagquantum.algorithms.trotter` is reachable through the subpackage surface only
-- `from flagquantum.algorithms.trotter import trotter_circuit` -- because the
algorithms package adds no root-level `fq.` name.

The premise the unit rests on, and does not check: the circuit a product formula
builds is an approximation of `exp(-i t H)`, and nothing here bounds how far the
two are apart. A bound needs a commutator norm, and a commutator norm belongs to
the caller, so what this script prints instead is the measured defect at two step
counts per order -- the defect's own rate, which is a property of the composition
rather than a statement about any particular Hamiltonian. The right-hand side of
every one of those comparisons is `torch.matrix_exp` of the dense Hamiltonian
matrix, which is available here because the Hamiltonian is two or three wires; a
caller with a hundred wires has no such reference and therefore no such number.

Two limits are stated rather than worked around. A term that is a multiple of the
identity exponentiates to a global phase, and no gate in this repository applies
one -- `Circuit` and `CircuitIR` carry no `global_phase` field -- so such a term is
refused by the term index that names it instead of being dropped, because dropping
it would leave a circuit that is right about the observable and wrong about the
state. And a product formula's value depends on the order its terms are declared
in, so the declared order is preserved: this script shows the same two terms in
both orders producing different circuits and a different state, which is what makes
the refusal to sort them load-bearing rather than stylistic.

What the unit does *not* do is add an `exp_pauli` instruction. CUDA-Q applies
`exp(-i theta P)` as one opaque operation whose decomposition its compiler owns;
here the decomposition itself is emitted -- a basis change, a CX ladder, and one
doubled `rz` -- so the result is an ordinary circuit that the compiler, the router,
and the gradient path already handle. The last block of this script runs that
circuit through the static resource estimator and differentiates a Hamiltonian
coefficient through it, which are the two things an opaque instruction would have
made somebody else's problem.

Sizes, and why: three wires, because that is the smallest register where a
non-diagonal word such as `XYZ` still has a ladder with more than one rung, and a
two-term transverse-field Ising Hamiltonian, because its field and coupling terms
do not commute and therefore the product formula has a defect at all. `time=0.4` is
short enough that the second-order defect at eight steps is below `1e-3` without
being at the floating-point floor. `docs/guides/ALGORITHMS.md` carries the
per-unit boundary.

Run it with:

    python -m examples.algorithms.trotter
"""

from __future__ import annotations

import argparse

import torch

from flagquantum.algorithms import (
    Hamiltonian,
    pauli_term,
    transverse_field_ising,
    trotter_circuit,
)
from flagquantum.algorithms.trotter import pauli_exponential_circuit
from flagquantum.circuit import Circuit
from flagquantum.compiler.resource_estimation import (
    ESTIMATE_BASIS,
    estimate_resources,
)
from flagquantum.simulation.pauli import exponential_pauli_operator
from flagquantum.simulation.unitary import get_unitary

DTYPE = torch.complex128
LABEL_WIDTH = 30
TIME = 0.4
COUPLING = 0.7
FIELD = 0.5
PRIMITIVE_ANGLE = 0.37
STEP_COUNTS = (2, 4, 8)


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def defect(order: int, steps: int) -> float:
    """Return the largest entrywise gap between the circuit and exact evolution."""

    hamiltonian = transverse_field_ising(3, coupling=COUPLING, field=FIELD)
    circuit = trotter_circuit(
        hamiltonian, TIME, steps=steps, order=order, dtype=DTYPE
    )
    exact = torch.matrix_exp(-1j * TIME * hamiltonian.matrix(dtype=DTYPE))
    return float((get_unitary(circuit) - exact).abs().max())


def primitive_residual(word: str, targets: tuple[int, ...], width: int) -> float:
    """Return how far the primitive is from the dense exponential of the same word.

    The two differ by a global phase the gate set cannot express, so the phase is
    divided out before the subtraction.
    """

    circuit = pauli_exponential_circuit(
        PRIMITIVE_ANGLE, word, targets, width, dtype=DTYPE
    )
    expected = exponential_pauli_operator(
        PRIMITIVE_ANGLE, word, targets, width, dtype=DTYPE, device="cpu"
    )
    product = get_unitary(circuit) @ expected.conj().T
    phase = product[0, 0]
    return float((product / phase - torch.eye(product.shape[0], dtype=DTYPE)).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Trotter product formula demo")
    parser.add_argument(
        "--time",
        type=float,
        default=TIME,
        help="the evolution time the product formula approximates",
    )
    args = parser.parse_args()

    hamiltonian = transverse_field_ising(3, coupling=COUPLING, field=FIELD)
    exact = torch.matrix_exp(-1j * args.time * hamiltonian.matrix(dtype=DTYPE))

    print("=" * 72)
    print("Trotter product formula -- flagquantum.algorithms.trotter")
    print("=" * 72)
    report("task", "a Pauli sum into the circuit that evolves under it")
    report("method", "a product of exact single-word exponentials")
    report("premise", "the product approximates exp(-i t H) and nothing here bounds")
    print(f"  {'':<{LABEL_WIDTH}}  how far apart they are: a bound needs a commutator")
    print(f"  {'':<{LABEL_WIDTH}}  norm, and a commutator norm belongs to the caller.")
    print(f"  {'':<{LABEL_WIDTH}}  The exact side below is torch.matrix_exp of the")
    print(f"  {'':<{LABEL_WIDTH}}  dense matrix, which exists only at this size")
    print()

    print("inputs")
    report("circuit", "3 wires")
    report("hamiltonian", f"transverse_field_ising, coupling={COUPLING}, field={FIELD}")
    report(
        "terms",
        [f"{t.coefficient} * {''.join(t.pauli)}" for t in hamiltonian.terms],
    )
    report("term count", f"{len(hamiltonian.terms)} in declared order")
    report("wires", hamiltonian.n_wires)
    report("time", args.time)
    report("dtype", DTYPE)
    print()

    print("the primitive: exp(-i * theta * P) as one basis change and one ladder")
    for word, targets, width in (
        ("Z", (0,), 1),
        ("X", (0,), 1),
        ("Y", (0,), 1),
        ("ZZ", (0, 1), 2),
        ("XZY", (2, 0, 1), 3),
        ("ZIZ", (0, 1, 2), 3),
    ):
        circuit = pauli_exponential_circuit(
            PRIMITIVE_ANGLE, word, targets, width, dtype=DTYPE
        )
        gates = [instruction.name for instruction in circuit.to_ir().instructions]
        report(
            f"{word} on {targets}",
            f"residual {primitive_residual(word, targets, width):.3e}, "
            f"gates {len(gates)} {gates}",
        )
    print(f"  {'':<{LABEL_WIDTH}}  the residual is measured after dividing out one")
    print(f"  {'':<{LABEL_WIDTH}}  global phase, which is what the gate set cannot")
    print(f"  {'':<{LABEL_WIDTH}}  express and why the numbers are at the floor")
    print()

    print("the defect against exact evolution, and its rate")
    for order in (1, 2):
        for steps in STEP_COUNTS:
            report(
                f"order {order}, {steps} steps",
                f"defect {defect(order, steps):.6e}",
            )
    for order in (1, 2):
        coarse = defect(order, STEP_COUNTS[0])
        fine = defect(order, STEP_COUNTS[1])
        report(
            f"order {order} rate",
            f"{coarse / fine:.3f} per doubling of the step count "
            f"(the defect falls off as step**{order})",
        )
    print(f"  {'':<{LABEL_WIDTH}}  halving the step halves the defect at the first")
    print(f"  {'':<{LABEL_WIDTH}}  order and quarters it at the second, which is the")
    print(f"  {'':<{LABEL_WIDTH}}  rate each composition's own leading defect predicts")
    print(f"  {'':<{LABEL_WIDTH}}  and not a bound on anything")
    print()

    print("a Hamiltonian whose terms commute is reached exactly")
    diagonal = Hamiltonian([pauli_term(1.0, "Z", 0), pauli_term(0.5, "Z", 1)])
    diagonal_exact = torch.matrix_exp(-1j * args.time * diagonal.matrix(dtype=DTYPE))
    diagonal_circuit = trotter_circuit(
        diagonal, args.time, steps=3, order=1, dtype=DTYPE
    )
    report(
        "two commuting z terms",
        f"defect {float((get_unitary(diagonal_circuit) - diagonal_exact).abs().max()):.3e}",
    )
    print(f"  {'':<{LABEL_WIDTH}}  one step would do; the product formula still has a")
    print(f"  {'':<{LABEL_WIDTH}}  defect only when the terms fail to commute")
    print()

    print("the declared term order is preserved, and the circuit is not the same")
    # Two terms on the *same* wire, which anticommute: on different wires they would
    # commute and both orders would reach the same state, so the comparison would be
    # silent about the ordering it exists to show.
    forward = Hamiltonian([pauli_term(1.0, "Z", 0), pauli_term(1.0, "X", 0)])
    backward = Hamiltonian([pauli_term(1.0, "X", 0), pauli_term(1.0, "Z", 0)])
    forward_circuit = trotter_circuit(forward, args.time, order=1, dtype=DTYPE)
    backward_circuit = trotter_circuit(backward, args.time, order=1, dtype=DTYPE)
    forward_ir = forward_circuit.to_ir()
    backward_ir = backward_circuit.to_ir()
    report("z then x gates", [i.name for i in forward_ir.instructions])
    report("x then z gates", [i.name for i in backward_ir.instructions])
    report(
        "same circuit",
        forward_ir.content_hash == backward_ir.content_hash,
    )
    report(
        "state distance",
        f"{float((forward_circuit.state() - backward_circuit.state()).abs().max()):.6f}",
    )
    report(
        "defect against exact",
        f"{float((forward_circuit.state()[0] - torch.matrix_exp(-1j * args.time * forward.matrix(dtype=DTYPE))[:, 0]).abs().max()):.6f}",
    )
    print(f"  {'':<{LABEL_WIDTH}}  the two orders are different circuits evolving to")
    print(f"  {'':<{LABEL_WIDTH}}  different states, so sorting the terms into a")
    print(f"  {'':<{LABEL_WIDTH}}  canonical order would be a second, silent input")
    print()

    print("the result is an ordinary circuit, not an opaque instruction")
    estimate = estimate_resources(forward_circuit)
    report("basis", ESTIMATE_BASIS)
    report(
        "operations",
        f"{estimate.n_operations} over {estimate.used_wires} of {estimate.n_wires} wires",
    )
    report("counts", dict(sorted(estimate.operation_counts.items())))
    report("depth", estimate.depth)
    report("t count", estimate.t_count)
    print(f"  {'':<{LABEL_WIDTH}}  CUDA-Q ships exp_pauli as one operation the")
    print(f"  {'':<{LABEL_WIDTH}}  compiler decomposes; emitting the decomposition here")
    print(f"  {'':<{LABEL_WIDTH}}  is what lets these counters see inside it")
    print()

    print("a tensor coefficient keeps its gradient")
    coefficient = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    trainable = Hamiltonian([pauli_term(coefficient, "X", 0)])
    gradient_circuit = trotter_circuit(
        trainable, 0.3, order=2, dtype=DTYPE
    )
    gradient_circuit.ry(0, theta=0.7)
    value = gradient_circuit.expectation_ps(z=[0]).sum()
    value.backward()
    report("tensor coefficient", "0.6, requires_grad=True")
    report("readout <Z>", f"{float(value.detach()):.12f}")
    report("d<Z>/d coefficient", f"{float(coefficient.grad):.12f}")
    print(f"  {'':<{LABEL_WIDTH}}  the angle reaches the rotation as a tensor, so the")
    print(f"  {'':<{LABEL_WIDTH}}  coefficient is differentiable like any other")
    print(f"  {'':<{LABEL_WIDTH}}  parameter -- no custom backward rule anywhere")
    print()

    print("terms and calls the unit refuses by name")
    cases = (
        (
            "a term that is a multiple of the identity",
            Hamiltonian([pauli_term(1.0, "II", (0, 1)), pauli_term(1.0, "Z", 0)]),
            0.4,
            {},
        ),
        (
            "a coefficient with an imaginary part",
            Hamiltonian([pauli_term(1.0 + 2.0j, "Z", 0)]),
            0.4,
            {},
        ),
        (
            "a term past the declared register",
            Hamiltonian([pauli_term(1.0, "Z", 2)]),
            0.4,
            {"n_qubits": 2},
        ),
        (
            "an order the module does not build",
            Hamiltonian([pauli_term(1.0, "Z", 0)]),
            0.4,
            {"order": 3},
        ),
        (
            "a step count that is not a positive integer",
            Hamiltonian([pauli_term(1.0, "Z", 0)]),
            0.4,
            {"steps": 0},
        ),
    )
    for label, candidate, time, kwargs in cases:
        try:
            trotter_circuit(candidate, time, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the refusal is the output here
            report(label, f"refused -- {type(exc).__name__}: {exc}")
        else:
            report(label, "NOT REFUSED")
    print(f"  {'':<{LABEL_WIDTH}}  every one of these is refused before a circuit is")
    print(f"  {'':<{LABEL_WIDTH}}  built, so a rejected call leaves nothing half-made")
    print()

    print("take away")
    print("  a Trotter circuit is a real circuit, and its accuracy is a number the")
    print("  caller measures rather than one this unit reports. The product formula")
    print("  converges at the rate its composition implies and no faster, and the")
    print("  identity term, the non-Hermitian coefficient, and the term past the")
    print("  register are all absent from the circuit that would otherwise be")
    print("  quietly wrong. What is beside it: computing a Krylov or Taylor")
    print("  expansion, chebyshev, qubitization, QSVT, and double factorization,")
    print("  which share the block encoding the SVD unit keeps private. What is not")
    print("  here: error bounds, time-dependent Hamiltonians, and any claim that the")
    print("  approximation is good enough for a caller's problem.")


if __name__ == "__main__":
    main()
