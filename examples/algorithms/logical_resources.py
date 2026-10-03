"""Cost a Clifford+T program on a rotated surface code, without running it.

The unit this script demonstrates,
:func:`flagquantum.algorithms.logical_resources.estimate_logical_resources`,
answers the question a fault-tolerant plan is costed with: how many Clifford
operations and how many T operations a program applies, how many logical layers
that is once the compiler has scheduled it, and what a distance-``d`` rotated
surface code spends to run those layers.  Two things are worth watching for while
it runs.

The first is that the gate-level tally is not recomputed here.  The operation
counts, the T family, and the schedule depth are
:func:`flagquantum.compiler.resource_estimation.estimate_resources`, and the
script prints that record beside the logical report's copy of it, so a second
implementation of what a T-depth means would show as two different numbers
instead of as two agreeing ones.

The second is what the unit refuses.  A logical resource estimate is defined over
Clifford+T programs, so a parametric rotation -- whose Clifford+T form is angle
synthesis's answer -- is refused rather than counted, and a compound operation
whose T-count is its decomposition's is refused for the same reason.  A Toffoli
is not a Clifford gate: counting one as a single Clifford operation would
understate the T-count of every program containing one, which is the number this
whole report exists to produce.  The script's last section puts each refusal on
one line.

Run it with:

    python -m examples.algorithms.logical_resources
"""

from __future__ import annotations

import argparse
import dataclasses

import flagquantum as fq
from flagquantum.algorithms import (
    estimate_logical_resources,
    surface_code_qubits_per_logical,
)
from flagquantum.algorithms.core import transverse_field_ising
from flagquantum.algorithms.trotter import trotter_circuit
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.core.ir import MeasurementNode, ensure_circuit_ir

LABEL_WIDTH = 40
DISTANCES = (3, 5, 7, 9, 11)


def report(label: str, value: object) -> None:
    print(f"  {label:<{LABEL_WIDTH}}: {value}")


def explain(*lines: str) -> None:
    """Print an indented paragraph under the last reported line."""

    for line in lines:
        print(f"  {'':<{LABEL_WIDTH}}  {line}")


def measured(program: object, count: int = 1) -> object:
    """Return the same program carrying ``count`` measurement records.

    Measurements live on the IR rather than in the instruction sequence the
    scheduler walks, so they are attached here the way a lowered program carries
    them: as records beside the operations rather than as operations.
    """

    ir = ensure_circuit_ir(program)
    records = tuple(
        MeasurementNode("counts", (wire,), shots=None) for wire in range(count)
    )
    return dataclasses.replace(ir, measurements=records)


def program() -> object:
    """Return the three-wire Clifford+T program every section below costs."""

    # Four Clifford operations and three T operations, with the T gates at two
    # different depths so `t_count` and `t_depth` are not the same number: a
    # report that confused the two would agree with itself here and nowhere else.
    return fq.Circuit(3).h(0).cz(0, 1).t(0).cx(0, 2).t(1).x(2).tdg(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logical-layer resource estimation demo"
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=5,
        help="the rotated surface-code distance the footprint is costed at",
    )
    args = parser.parse_args()

    circuit = program()

    print("=" * 72)
    print("Logical resources -- flagquantum.algorithms.logical_resources")
    print("=" * 72)
    report("task", "cost a Clifford+T program on a surface code")
    report("method", "the compiler's static tally, plus one code model")
    report(
        "premise",
        "the report counts rather than measures: nothing runs, no wall-clock",
    )
    explain(
        "time, memory, or allocation is read, and the physical figure is a",
        "floor for a circuit of these layers rather than a compiled estimate.",
        "The failure rate is absent by construction, because a logical error",
        "rate needs a device's threshold fit and that number belongs to the",
        "device rather than to this unit.",
    )
    print()

    print("the program")
    report("n_qubits", circuit.num_qubits)
    report("operations", "h(0) cz(0,1) t(0) cx(0,2) t(1) x(2) tdg(0)")
    report("Clifford+T?", "yes -- every opcode is in one of the two families")
    print()

    print("the gate-level tally is the compiler's record, not a second count")
    direct = estimate_resources(circuit)
    logical = estimate_logical_resources(circuit, distance=args.distance)
    report("compiler n_operations", direct.n_operations)
    report("report estimate n_operations", logical.estimate.n_operations)
    report("compiler depth", direct.depth)
    report("report estimate depth", logical.estimate.depth)
    report("compiler t_count", direct.t_count)
    report("report t_count", logical.t_count)
    report("compiler t_depth", direct.t_depth)
    report("report t_depth", logical.t_depth)
    report("compiler per_wire_depth", direct.per_wire_depth)
    report("report per_wire_depth", logical.estimate.per_wire_depth)
    report("the two records are equal", logical.estimate == direct)
    explain(
        "the report nests the compiler's ResourceEstimate rather than copying",
        "its fields, so a reader who wants per-wire depths reads them from the",
        "record that owns them. A second T rule would print two different",
        "numbers on the two lines above rather than agreeing twice.",
    )
    print()

    print("the Cliffords and the Ts, partitioned by opcode family")
    report("operation counts", dict(logical.estimate.operation_counts))
    report("clifford_count", logical.clifford_count)
    report("t_count", logical.t_count)
    report("n_clifford_t", logical.n_clifford_t)
    report("t_depth", logical.t_depth)
    report("clifford + t == operations", logical.clifford_count + logical.t_count)
    explain(
        "the classification is total over the operator schema: every declared",
        "opcode is a Clifford, a T, a channel, or one of the two refused",
        "families, and a program outside those families is refused by name",
        "rather than counted as other.",
    )
    print()

    print("the footprint at one distance")
    report("code_distance", logical.code_distance)
    report("physical qubits per logical", logical.physical_qubits_per_logical)
    report("report n_qubits", logical.n_qubits)
    report("used_wires", logical.estimate.used_wires)
    report("n_measurements", logical.n_measurements)
    report("logical_depth", logical.logical_depth)
    report("surface_code_cycles", logical.surface_code_cycles)
    report("physical_qubits", logical.physical_qubits)
    report("spacetime_volume", logical.spacetime_volume)
    explain(
        "the logical depth is the compiler's schedule depth plus one layer per",
        "measurement record, one logical layer costs d surface-code cycles, and",
        "the volume is physical qubits times cycles. The declared register is",
        "charged, not only the wires an operation touches.",
    )
    print()

    print("the same program at several distances")
    report("distance", "  ".join(f"{d:>6d}" for d in DISTANCES))
    report(
        "qubits/logical",
        "  ".join(f"{surface_code_qubits_per_logical(d):>6d}" for d in DISTANCES),
    )
    report(
        "cycles",
        "  ".join(
            f"{estimate_logical_resources(circuit, distance=d).surface_code_cycles:>6d}"
            for d in DISTANCES
        ),
    )
    report(
        "physical qubits",
        "  ".join(
            f"{estimate_logical_resources(circuit, distance=d).physical_qubits:>6d}"
            for d in DISTANCES
        ),
    )
    report(
        "spacetime volume",
        "  ".join(
            f"{estimate_logical_resources(circuit, distance=d).spacetime_volume:>6d}"
            for d in DISTANCES
        ),
    )
    report("2 d^2 - 1", "  ".join(f"{2 * d * d - 1:>6d}" for d in DISTANCES))
    explain(
        "the patch grows as d squared and the cycles grow as d, so the volume",
        "grows as d cubed for a fixed program. What that volume buys in logical",
        "error is the device's threshold fit and not this report's, so the",
        "cost of a distance is printed here and the benefit never is.",
    )
    print()

    print("a measurement is charged one logical layer")
    bare = estimate_logical_resources(circuit, distance=args.distance)
    with_records = estimate_logical_resources(
        measured(circuit, 2), distance=args.distance
    )
    report("n_measurements, measured", with_records.n_measurements)
    report("logical_depth, none", bare.logical_depth)
    report("logical_depth, measured", with_records.logical_depth)
    report("cycles, measured", with_records.surface_code_cycles)
    report("physical qubits, measured", with_records.physical_qubits)
    report("tally, measured", (with_records.clifford_count, with_records.t_count))
    explain(
        "a measurement is an IR record the instruction schedule does not cover,",
        "so it is charged as one further logical layer rather than dropped. The",
        "operations and their counts are unchanged and so is the qubit count.",
    )
    print()

    print("the assumptions the arithmetic rests on, carried by the report")
    for assumption in logical.assumptions:
        report("assumption", assumption)
    print()

    print("the record a caller serializes")
    payload = logical.to_dict()
    report("kind", payload["kind"])
    report("basis", payload["basis"])
    report("surface_code_model", payload["surface_code_model"])
    report("capability_evidence keys", sorted(payload["capability_evidence"]))
    report("estimate kind", payload["estimate"]["kind"])
    explain(
        "the limitations text is written under the field name the capability",
        "maturity matrix requires at every level, so the text a registry entry",
        "carries and the text a caller reads here can be the same text.",
    )
    print()

    print("calls the unit refuses by name")
    trotter = trotter_circuit(
        transverse_field_ising(3, coupling=0.7, field=0.5), 0.4, steps=1, order=2
    )
    cases: list[tuple[str, object]] = [
        ("one parametric rotation", fq.Circuit(2).h(0).rz(0, 0.3).t(1)),
        ("several parametric rotations", fq.Circuit(2).h(0).rz(0, 0.3).rx(1, 0.4)),
        ("a Toffoli", fq.Circuit(3).h(0).ccx(0, 1, 2)),
        ("a controlled swap", fq.Circuit(3).h(0).cswap(0, 1, 2)),
        ("a Toffoli beside a rotation", fq.Circuit(3).ccx(0, 1, 2).rz(0, 0.3)),
        ("a Trotter step", trotter),
        (
            "a lowered noise channel",
            fq.Circuit(2).h(0).gate("depolarizing", (0,), probability=0.1),
        ),
    ]
    for label, candidate in cases:
        try:
            estimate_logical_resources(candidate, distance=args.distance)
        except Exception as exc:
            report(label, f"refused -- {type(exc).__name__}")
        else:
            report(label, "NOT REFUSED")
    explain(
        "the Trotter step is the one that shows why the refusal is not a",
        "formality: the gate-level estimator reads that circuit happily and",
        "reports a T-count of zero, because its rotations are angles rather than",
        "T gates. A logical cost taken over it would be the cost of a program",
        "nobody will run.",
    )
    print()

    print("arguments the code model refuses")
    for label, call in (
        ("distance below three", lambda: surface_code_qubits_per_logical(1)),
        ("an even distance", lambda: surface_code_qubits_per_logical(4)),
        (
            "a distance that is not an integer",
            lambda: surface_code_qubits_per_logical(5.0),
        ),
        (
            "a distance below three, at the entry point",
            lambda: estimate_logical_resources(circuit, distance=2),
        ),
        (
            "zero logical qubits",
            lambda: estimate_logical_resources(circuit, distance=5, n_qubits=0),
        ),
        (
            "a logical count that is not an integer",
            lambda: estimate_logical_resources(circuit, distance=5, n_qubits=1.0),
        ),
    ):
        try:
            call()
        except Exception as exc:
            report(label, f"refused -- {type(exc).__name__}: {exc}")
        else:
            report(label, "NOT REFUSED")
    print()

    print("a caller may charge fewer or more logical qubits than the register")
    fewer = estimate_logical_resources(circuit, distance=args.distance, n_qubits=1)
    more = estimate_logical_resources(circuit, distance=args.distance, n_qubits=8)
    report("register", circuit.num_qubits)
    report("charged for one", fewer.physical_qubits)
    report("charged for the register", logical.physical_qubits)
    report("charged for eight", more.physical_qubits)
    explain(
        "an ancilla register is charged for the logical count alone, and a code",
        "block laid out for a larger algorithm is a legal request. The ledger is",
        "the logical qubit count; nothing here infers it from the register.",
    )
    print()

    print("take away")
    print("  a logical resource estimate is a count over a program text and a")
    print("  footprint over a code model, and the honest version of both is the")
    print("  point: the tally here is the compiler's own record rather than a")
    print("  second implementation of it, and the code model is one named model")
    print("  whose conventions travel with its numbers. What is beside it: the")
    print("  angle synthesis that would let an arbitrary rotation reach this unit")
    print("  at all -- an exact quarter turn already has a Clifford+T form in the")
    print("  compiler, and this unit reads opcodes rather than reaching for it --")
    print("  and the distillation factory, magic-state budget, placement, and")
    print("  device model that would turn a")
    print("  patch count into a compiled estimate. What is not here, and never")
    print("  was: a logical error rate. That number needs a device's threshold")
    print("  fit, so this report says how much hardware a logical program would")
    print("  occupy and for how long, and never how often it would fail.")


if __name__ == "__main__":
    main()
