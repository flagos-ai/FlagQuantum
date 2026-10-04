"""Add two registers in place, then cost the Toffolis the compiler already knows.

The unit this script demonstrates,
:func:`flagquantum.algorithms.arithmetic.adder_circuit`, builds a circuit and
nothing else.  Two things are worth watching for while it runs.

The first is that the sum is shown rather than asserted.  The script prepares a
computational basis state holding two addends, runs the adder on the ordinary
statevector path, and decodes the wire labels back into integers, so the number
printed as the sum is read out of a state vector the runtime produced.  The same
section runs one input with a dirty carry wire to show what the adder's ancilla
contract is for: off that domain the circuit is still reversible and it is no
longer an addition.

The second is where the T-count comes from.  One Toffoli is seven T through the
compiler's own fifteen-gate identity in
:mod:`flagquantum.compiler.basis_translation`, so the script lowers the adder with
:func:`flagquantum.compiler.basis_conversion.convert_basis` and prices it with
:func:`flagquantum.compiler.resource_estimation.estimate_resources` and
:func:`flagquantum.algorithms.logical_resources.estimate_logical_resources`.  No
seven-T expansion is written into the arithmetic module, because a second Toffoli
cost would be a second source of truth for the one number the two must agree on.

Run it with:

    python -m examples.algorithms.arithmetic
"""

from __future__ import annotations

import argparse

import torch

import flagquantum as fq
from flagquantum.algorithms.arithmetic import (
    adder_circuit,
    adder_wires,
)
from flagquantum.algorithms.logical_resources import estimate_logical_resources
from flagquantum.compiler.basis_conversion import BasisConversionError, convert_basis
from flagquantum.compiler.resource_estimation import estimate_resources

LABEL_WIDTH = 40
WIDTHS = (1, 2, 3, 4, 5, 8, 9)
#: The Clifford+T basis the compiler's Toffoli identity lands in.  ``tdg`` is
#: named because the identity uses it three times.
CLIFFORD_T_BASIS = ("cx", "h", "s", "sdg", "t", "tdg")
#: The bound ``convert_basis`` applies by default to the operations it may add.
DEFAULT_ADDED_OPERATION_BOUND = 256


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def explain(*lines: str) -> None:
    """Print an indented paragraph under the last reported line."""

    for line in lines:
        print(f"  {'':<{LABEL_WIDTH}}  {line}")


def basis_label(n_bits: int, a: int, b: int, carry: int, carry_out: int) -> int:
    """Return the basis-state index of the two addends and the two ancillas.

    Wire 0 is the most significant bit of the first register, wire ``n_bits`` the
    most significant bit of the second, and the two ancillas sit above both.  The
    label is assembled in that order rather than in the order the registers are
    stored, because the index convention is the same one the state vector uses.
    """

    value = 0
    for index in range(n_bits):
        value = (value << 1) | ((a >> (n_bits - 1 - index)) & 1)
    for index in range(n_bits):
        value = (value << 1) | ((b >> (n_bits - 1 - index)) & 1)
    value = (value << 1) | carry
    return (value << 1) | carry_out


def decode(n_bits: int, width: int, label: int) -> tuple[int, int, int, int]:
    """Read a basis-state index back into the two addends and the two ancillas."""

    bits = [(label >> (width - 1 - index)) & 1 for index in range(width)]
    a = 0
    b = 0
    for index in range(n_bits):
        a = (a << 1) | bits[index]
    for index in range(n_bits):
        b = (b << 1) | bits[n_bits + index]
    return a, b, bits[2 * n_bits], bits[2 * n_bits + 1]


def run_adder(n_bits: int, a: int, b: int, *, carry: int = 0, carry_out: int = 0):
    """Return the adder's output on one input, by running it on a state vector.

    The circuit is rebuilt on a fresh register carrying the input state rather
    than executed in place, so the printout below is the runtime's own answer:
    ``Circuit.state`` is the statevector kernel's output, and the script reads the
    basis state of largest weight out of it.
    """

    wires = adder_wires(n_bits)
    width = wires.n_wires
    vector = torch.zeros(2**width, dtype=torch.complex128)
    vector[basis_label(n_bits, a, b, carry, carry_out)] = 1.0
    register = fq.Circuit(width, inputs=vector.reshape(1, -1), dtype=torch.complex128)
    for instruction in adder_circuit(n_bits).to_ir().instructions:
        register.gate(instruction.name, instruction.wires)
    state = register.state().reshape(-1).abs()
    reached = int(state.argmax())
    return (*decode(n_bits, width, reached), float(state[reached]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reversible addition demo")
    parser.add_argument(
        "--bits",
        type=int,
        default=3,
        help="the register width the sums and the register map are shown at",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=5,
        help="the surface-code distance the logical cost is quoted at",
    )
    args = parser.parse_args()
    n_bits = args.bits

    print("=" * 72)
    print("Reversible addition -- flagquantum.algorithms.arithmetic")
    print("=" * 72)
    report("task", "add two n-bit registers in place on one carry wire")
    report("method", "Cuccaro ripple -- a majority/unmajority pair per bit")
    report(
        "premise",
        "this module builds the circuit and never runs it, and the T-cost",
    )
    explain(
        "belongs to the compiler's own Toffoli rule rather than to this",
        "construction: convert_basis lowers the ccx gates and",
        "estimate_resources prices them, and no seven-T expansion is written",
        "here, because a second Toffoli cost would be a second source of truth",
        "for the one number the two must agree on. The adder is also not the",
        "Gidney-Ekera construction: that one uncomputes its ancillas by",
        "measuring them and feeding the outcomes forward, which needs a",
        "mid-circuit measurement and a classical feedforward path this",
        "repository does not have in a circuit it can cost statically. Only the",
        "inputs the ancilla contract names are promised a sum.",
    )
    print()

    print(f"the construction at {n_bits} bits")
    circuit = adder_circuit(n_bits)
    names = [instruction.name for instruction in circuit.to_ir().instructions]
    wires = adder_wires(n_bits)
    report("wires", circuit.n_wires)
    report("operations", len(names))
    report("ccx", names.count("ccx"))
    report("cx", names.count("cx"))
    report("opcodes present", sorted(set(names)))
    report("register a", f"wires {wires.a} (most significant first)")
    report("register b", f"wires {wires.b} (most significant first)")
    report("working carry wire", wires.carry)
    report("carry-out wire", wires.carry_out)
    explain(
        "one ancilla carries the whole ripple. MAJ folds the running carry into",
        "the third wire, so the wire that leaves MAJ holding the carry out of one",
        "position is the wire that feeds the carry into the next, and the reverse",
        "sweep walks it back down. The two data wires are their original selves by",
        "the time the carry into a position is available, which is exactly when",
        "the sum bit can be written into the second register.",
    )
    print()

    print("the same construction across widths")
    report("n_bits", "".join(f"{n:>7d}" for n in WIDTHS))
    report("wires", "".join(f"{2 * n + 2:>7d}" for n in WIDTHS))
    report("ccx", "".join(f"{2 * n:>7d}" for n in WIDTHS))
    report("cx", "".join(f"{6 * n + 1:>7d}" for n in WIDTHS))
    report("operations", "".join(f"{8 * n + 1:>7d}" for n in WIDTHS))
    report(
        "operations, measured",
        "".join(f"{len(adder_circuit(n).to_ir().instructions):>7d}" for n in WIDTHS),
    )
    report("working ancillas", "".join(f"{1:>7d}" for _ in WIDTHS))
    explain(
        "both counts are linear in n and neither depends on the values added, so",
        "the cost of an addition is known before the addends are. The naive ripple",
        "would put a carry wire under every bit; this one carries the whole ripple",
        "on a single ancilla and gives it back clean.",
    )
    print()

    print("the sum, read out of the runtime's own state vector")
    cases = (
        (0, 0),
        (1, 1),
        (2**n_bits - 1, 1),
        (2**n_bits - 1, 2**n_bits - 1),
        (3 % 2**n_bits, 5 % 2**n_bits),
    )
    for a, b in cases:
        left, right, carry, carry_out, weight = run_adder(n_bits, a, b)
        report(
            f"{a} + {b}",
            f"a={left} b={right} carry={carry} carry_out={carry_out} "
            f"weight={weight:.6f}",
        )
    explain(
        "the first addend comes back unchanged, the second holds the sum modulo",
        "2**n_bits, the working carry wire is clean again, and the wire above it",
        "holds the carry out of the most significant position -- which is the top",
        "bit of the exact integer sum and is otherwise discarded. Nothing here",
        "approximates: the map is a permutation and the weight printed is the",
        "single basis state the input reached.",
    )
    print()

    print("the ancilla contract, and what happens outside it")
    clean = run_adder(n_bits, 1, 1)
    carried = run_adder(n_bits, 1, 1, carry=1)
    report("a=1 b=1, ancillas clean", f"reached {clean[:4]}")
    report("a=1 b=1, carry wire dirty", f"reached {carried[:4]}")
    report(
        "a=1 b=1, carry-out wire dirty",
        f"reached {run_adder(n_bits, 1, 1, carry_out=1)[:4]}",
    )
    print()
    # The two ancillas do not behave alike off the contract, and the difference is
    # measured over every input rather than described, because "the ancillas must
    # be clean" would otherwise read as one rule covering both wires.
    broke_sum = 0
    changed_answer = 0
    for a in range(2**n_bits):
        for b in range(2**n_bits):
            left, right, carry, carry_out, _ = run_adder(n_bits, a, b, carry_out=1)
            expected_carry = (a + b) >> n_bits
            if (left, right, carry) != (a, (a + b) % 2**n_bits, 0) or carry_out != (
                expected_carry ^ 1
            ):
                broke_sum += 1
            left, right, carry, carry_out, _ = run_adder(n_bits, a, b, carry=1)
            plain = (a, (a + b) % 2**n_bits, 0, (a + b) >> n_bits)
            as_carry_in = (a, (a + b + 1) % 2**n_bits, 0, (a + b + 1) >> n_bits)
            if (left, right, carry, carry_out) not in (plain, as_carry_in):
                changed_answer += 1
    inputs = 2 ** (2 * n_bits)
    report(
        "a dirty carry-out wire, over every input",
        f"{broke_sum} of {inputs} do not sum, and the top wire is XORed",
    )
    report(
        "a dirty carry wire, over every input",
        f"{changed_answer} of {inputs} are neither the sum nor the sum-with-carry-in",
    )
    explain(
        "the two wires are not one rule. A dirty carry-out wire cannot change the",
        "sum, because the adder only ever writes to it at the end and as an XOR,",
        "so every input still sums and the top wire comes back carrying the true",
        "carry out XORed with whatever it entered holding. A dirty carry wire does",
        "change the answer: it is read by every MAJ in the forward sweep, so the",
        "ripple it produces is a different ripple, and the measured result is",
        "neither a + b nor a + b + carry. Only ancilla-clean inputs are promised a",
        "sum. The map is still a permutation of the whole space, which is why",
        "nothing is silently lost on the other inputs -- but a permutation is not",
        "an addition, and the row above is where the difference is visible.",
    )
    print()

    print("the T-cost is the compiler's Toffoli rule, applied to this circuit")
    one = estimate_resources(
        convert_basis(fq.Circuit(3).ccx(0, 1, 2), gates=CLIFFORD_T_BASIS).program
    )
    report("one ccx, operations", one.n_operations)
    report("one ccx, operation counts", dict(one.operation_counts))
    report("one ccx, t_count", one.t_count)
    print()
    report("n_bits", "".join(f"{n:>7d}" for n in WIDTHS))
    lowered = {
        n: convert_basis(adder_circuit(n), gates=CLIFFORD_T_BASIS) for n in WIDTHS
    }
    tallies = {n: estimate_resources(lowered[n].program) for n in WIDTHS}
    report("t_count", "".join(f"{tallies[n].t_count:>7d}" for n in WIDTHS))
    report("14 n", "".join(f"{14 * n:>7d}" for n in WIDTHS))
    report(
        "operations, lowered",
        "".join(f"{tallies[n].n_operations:>7d}" for n in WIDTHS),
    )
    report("depth, lowered", "".join(f"{tallies[n].depth:>7d}" for n in WIDTHS))
    explain(
        "one Toffoli is seven T and the adder contains 2 n of them, so the cost is",
        "14 n T with nothing else contributing. The fifteen-gate identity replaces",
        "one operation with fifteen, which is why the lowered operation count grows",
        "by 28 per bit rather than by 8.",
    )
    print()

    print("and the logical cost of one of those additions")
    report("distance", args.distance)
    for n in (2, 4):
        report(
            f"n_bits={n}",
            f"t_count={tallies[n].t_count} physical_qubits="
            f"{estimate_logical_resources(lowered[n].program, distance=args.distance).physical_qubits} "
            f"cycles="
            f"{estimate_logical_resources(lowered[n].program, distance=args.distance).surface_code_cycles}",
        )
    explain(
        "the physical figure is a floor rather than a compiled estimate: it is a",
        "patch count with no distillation factory, magic-state budget, routing",
        "overhead, placement, or device model, and no logical error rate is",
        "reported because a threshold fit's numbers belong to the device.",
    )
    print()

    print("the conversion bound is a bound the caller can raise")
    widest = max(WIDTHS)
    report("widest width at the default bound", widest)
    report("bound", DEFAULT_ADDED_OPERATION_BOUND)
    try:
        convert_basis(adder_circuit(widest + 1), gates=CLIFFORD_T_BASIS)
    except BasisConversionError as error:
        report(
            f"n_bits={widest + 1} at the default",
            f"refused -- {type(error).__name__}: {error}",
        )
    else:
        report(f"n_bits={widest + 1} at the default", "NOT REFUSED")
    raised = convert_basis(
        adder_circuit(widest + 1),
        gates=CLIFFORD_T_BASIS,
        max_added_operations=1024,
    )
    report(
        f"n_bits={widest + 1}, bound raised to 1024",
        f"t_count={estimate_resources(raised.program).t_count}",
    )
    explain(
        "the refusal names max_added_operations rather than the construction, and",
        "it is a property of the conversion rather than of the adder: the same",
        "circuit lowers once the caller says how much expansion is acceptable.",
    )
    print()

    print("widths the constructor refuses")
    for label, width in (
        ("zero bits", 0),
        ("a negative width", -1),
        ("a float width", 3.0),
        ("a width given as text", "3"),
        ("a width given as a truth value", True),
    ):
        for entry in (adder_circuit, adder_wires):
            try:
                entry(width)
            except Exception as exc:
                report(
                    f"{label}, {entry.__name__}",
                    f"refused -- {type(exc).__name__}: {exc}",
                )
            else:
                report(f"{label}, {entry.__name__}", "NOT REFUSED")
    print()

    print("take away")
    print("  a reversible adder is a construction and a cost, and both are")
    print("  stated here: 2 n Toffolis and 6 n + 1 cx on 2 n + 2 wires, exact")
    print("  on the domain the ancilla contract names, and priced by the")
    print("  compiler's own Toffoli rule rather than by a second expansion of")
    print("  it written beside the circuit. What is beside it: the modular,")
    print("  controlled, and constant-addend variants, a comparison, and a")
    print("  multiplier, none of which exist here, and the Gidney-Ekera")
    print("  construction, which is smaller because it uncomputes its ancillas")
    print("  by measurement and feedforward rather than unitarily.")


if __name__ == "__main__":
    main()
