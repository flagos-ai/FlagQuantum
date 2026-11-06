"""Count emitted 8-bit SelectTHC instructions and optionally show an upstream model.

Run with `python -m examples.algorithms.select_thc_resources`. Add `--qualtran`
in an environment with Qualtran 0.7.0 to include its separate model estimate.
"""

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterator
from importlib.metadata import version
from math import tau

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectTHC
from flagquantum.compiler.registers.arithmetic import (
    controlled_add_or_subtract_ir,
    measurement_add_or_subtract_ir,
)
from flagquantum.compiler.registers.phase_gradient import phase_gradient_state_ir
from flagquantum.compiler.registers.qrom import controlled_qrom_ir
from flagquantum.compiler.registers.swap_network import swap_with_zero_pair_ir
from flagquantum.compiler.resource_estimation import count_instruction_resources
from flagquantum.core.classical_control import instruction_condition_clauses
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit

ANGLE_BITS = 8
NETWORKS = (((0, 1, 0.2),), ((0, 1, 0.5),))
ONE_BODY_NETWORKS = ((), ((0, 1, -0.4),))


def _angle_tables(first: bool) -> Iterator[tuple[bool, tuple[int, ...]]]:
    networks = (*NETWORKS, (), (), *ONE_BODY_NETWORKS) if first else NETWORKS
    for inverse in (False, True):
        words = tuple(
            (
                round(((-net[0][2] if inverse else net[0][2]) % tau) / tau * 256) % 256
                if net
                else 0
            )
            for net in networks
        )
        yield inverse, words


def _stage_layout(measured: bool) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Expected opcode spans for the two-row, single-slot example."""
    angle, gradient = tuple(range(1, 9)), tuple(range(9, 17))
    carries = tuple(range(17, 24))
    bits = tuple(range(16))
    swap_count = (
        len(swap_with_zero_pair_ir(0, angle, gradient)) if measured else ANGLE_BITS
    )
    adder = (
        measurement_add_or_subtract_ir(
            0, angle, gradient, carries, tuple(range(16, 23))
        )
        if measured
        else controlled_add_or_subtract_ir(0, angle, gradient, carries[0])
    )
    yield "Gradient prepare", tuple(op.name for op in phase_gradient_state_ir(gradient))
    for first in (True, False):
        yield "Spin swaps", ("cswap",) * 2
        for inverse, words in _angle_tables(first):
            width = 3 if first else 2
            load = controlled_qrom_ir(
                width,
                ANGLE_BITS,
                (*words, *([0] * ((1 << width) - len(words)))) if measured else words,
                blocked=True,
                traversal_bits=tuple(range(23, 23 + width - 1)) if measured else (),
                relative_phase_swap=measured,
            )
            erase = controlled_qrom_ir(
                3 if first else 2,
                ANGLE_BITS,
                words,
                blocked=True,
                inverse=True,
                measurement_bits=bits if measured else (),
                relative_phase_swap=measured,
            )
            yield "Rotation basis changes", ("cx",)
            yield "Lookup XOR", tuple(op.name for op in load.instructions[:-swap_count])
            yield "Lookup bank exchange", tuple(
                op.name for op in load.instructions[-swap_count:]
            )
            yield "Rotation basis changes", ("sdg", "h")
            yield "Add/subtract", tuple(op.name for op in adder)
            yield "Rotation basis changes", ("h", "s")
            yield "Erase", tuple(op.name for op in erase)
            yield "Rotation basis changes", ("cx",)
            if not inverse:
                yield "Reflection", ("cz",) if first else ("x", "h", "ccx", "h", "x")
        yield "Spin swaps", ("cswap",) * 2
    yield "Address/spin-label exchange", ("x", "cx", "cswap", "cswap", "cswap", "x")
    yield "Gradient cleanup", tuple(
        op.name for op in phase_gradient_state_ir(gradient, inverse=True)
    )


def _print_emitted_stages(ir: CircuitIR, *, measured: bool) -> dict[str, Counter[str]]:
    stages: dict[str, Counter[str]] = defaultdict(Counter)
    conditioned: Counter[str] = Counter()
    offset = 0
    erasure_ccx = []
    for stage, expected in _stage_layout(measured):
        span = ir.instructions[offset : offset + len(expected)]
        if tuple(op.name for op in span) != expected:
            raise RuntimeError(f"{stage} does not match emitted IR at offset {offset}")
        stages[stage].update(op.name for op in span)
        if stage == "Erase":
            erasure_ccx.append(sum(op.name == "ccx" for op in span))
        conditioned[stage] += sum(
            bool(instruction_condition_clauses(op)) for op in span
        )
        offset += len(span)
    if offset != len(ir.instructions):
        raise RuntimeError(f"Unaccounted emitted IR after offset {offset}")
    print(
        "  Stage                       T/T†   CCX  CSWAP  Meas.  Cond.  Phase   T-equiv."
    )
    for stage, counts in stages.items():
        t = counts["t"] + counts["tdg"]
        subtotal = t + 7 * (counts["ccx"] + counts["cswap"])
        print(
            f"  {stage:27} {counts['t']:>2}/{counts['tdg']:<2} "
            f"{counts['ccx']:>5} {counts['cswap']:>6} {counts['measure']:>6} "
            f"{conditioned[stage]:>6} {counts['phase']:>6} {subtotal:>10}"
        )

    if measured:
        print("  Erasure CCX by mu forward/inverse, nu forward/inverse:", erasure_ccx)
    return dict(stages)


def _print_qualtran_executable_counts(stages: dict[str, Counter[str]]) -> None:
    import cirq
    from qualtran import Bloq, QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.data_loading import QROAMClean
    from qualtran.cirq_interop import BloqAsCirqGate

    names = {
        cirq.X: "x",
        cirq.H: "h",
        cirq.S: "s",
        cirq.S**-1: "sdg",
        cirq.T: "t",
        cirq.T**-1: "tdg",
        cirq.CNOT: "cx",
        cirq.CZ: "cz",
        BloqAsCirqGate(CNOT()): "cx",
    }

    def count(bloq: Bloq) -> Counter[str]:
        gates = cirq.decompose(
            bloq.decompose_bloq().to_cirq_circuit(),
            keep=lambda op: (
                op.gate in names
                or cirq.is_measurement(op)
                or isinstance(op.gate, cirq.ResetChannel)
                or isinstance(op, cirq.ClassicallyControlledOperation)
            ),
        )
        counts: Counter[str] = Counter()
        for gate in gates:
            if isinstance(gate, cirq.ClassicallyControlledOperation):
                gate = gate.without_classical_controls()
            if cirq.is_measurement(gate):
                counts["measure"] += 1
            elif isinstance(gate.gate, cirq.ResetChannel):
                counts["reset"] += 1
            else:
                counts[names[gate.gate]] += 1
        return counts

    lookup: Counter[str] = Counter()
    for first in (True, False):
        width = 3 if first else 2
        for _, words in _angle_tables(first):
            lookup.update(
                count(
                    QROAMClean.build_from_data(
                        (*words, *([0] * ((1 << width) - len(words)))),
                        target_bitsizes=(ANGLE_BITS,),
                        num_controls=1,
                        log_block_sizes=(1,),
                    )
                )
            )
    dtype = QUInt(ANGLE_BITS)
    arithmetic = count(ControlledAddOrSubtract(dtype, dtype, add_when_ctrl_is_on=False))
    arithmetic = Counter({name: 4 * total for name, total in arithmetic.items()})
    for label, actual, upstream in (
        (
            "Forward QROAM",
            stages["Lookup XOR"] + stages["Lookup bank exchange"],
            lookup,
        ),
        ("Controlled add/subtract", stages["Add/subtract"], arithmetic),
    ):
        if actual != upstream:
            raise RuntimeError(
                f"{label} opcode count mismatch: "
                f"FlagQuantum excess {actual - upstream}; "
                f"Qualtran excess {upstream - actual}"
            )
        print(f"  {label}: {dict(upstream)}")
    print("  Executable counts match FlagQuantum for both stages.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualtran", action="store_true")
    args = parser.parse_args()

    logical_width = 13
    for measurement_erasure in (False, True):
        # Two-row lookup in both cases, with six additional carry qubits
        # for the measurement-assisted adder.
        workspace_width = 34 if measurement_erasure else 28
        width = logical_width + workspace_width
        circuit = DynamicCircuit(width) if measurement_erasure else fq.Circuit(width)
        circuit.append(
            SelectTHC.make_on(
                mu=(0, 1),
                nu=(2, 3),
                succ=4,
                nu_eq_mp1=5,
                plus_mn=6,
                plus_a=7,
                plus_b=8,
                sys_a=(9, 10),
                sys_b=(11, 12),
                networks=NETWORKS,
                one_body_networks=ONE_BODY_NETWORKS,
                workspace=tuple(range(logical_width, width)),
                measurement_erasure=measurement_erasure,
            )
        )
        report = count_instruction_resources(circuit, logical_qubits=logical_width)
        counts = Counter(report.fixed_counts) + Counter(report.conditional_counts)
        print(
            "FlagQuantum:", "measurement-assisted" if measurement_erasure else "unitary"
        )
        print("  Fixed emitted instructions:", report.fixed_counts)
        print("  Conditional emitted instructions:", report.conditional_counts)
        print("  Peak reserved workspace qubits:", report.workspace_qubits)
        print("  Emitted T/T†:", counts["t"], counts["tdg"])
        print(
            "  T-equivalent subtotal (7 per CCX/CSWAP, excluding phase synthesis):",
            counts["t"] + counts["tdg"] + 7 * (counts["ccx"] + counts["cswap"]),
        )
        stages = _print_emitted_stages(circuit.to_ir(), measured=measurement_erasure)
        if measurement_erasure and args.qualtran:
            if version("qualtran") != "0.7.0":
                parser.error("--qualtran requires Qualtran 0.7.0")
            print("Qualtran 0.7.0 executable component counts:")
            _print_qualtran_executable_counts(stages)

    if args.qualtran:
        from qualtran.bloqs.chemistry.thc.select_bloq import SelectTHC as CostModel
        from qualtran.bloqs.chemistry.thc.select_bloq import THCRotations
        from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

        model = CostModel(
            num_mu=2, num_spin_orb=4, num_bits_theta=8, keep_bitsize=1, kr1=2, kr2=2
        )
        cost = get_cost_value(model, QECGatesCost())
        print("Qualtran 0.7.0 call-graph estimate:", cost)
        print(
            "  T-equivalent subtotal (7 per CCX/CSWAP, excluding rotation synthesis):",
            cost.total_t_count(ts_per_toffoli=7, ts_per_cswap=7, ts_per_rotation=0),
        )
        print("  Modeled total qubits:", get_cost_value(model, QubitCount()))
        print("  Rotation workspace and measurements are not fully modeled.")
        print("  Assumes an existing phase-gradient state.")
        print("  The actual v0.7.0 call graph includes 3 label and 8 system CSWAPs.")
        modeled_toffolis = 0
        for bloq, copies in model.decompose_bloq().bloq_counts().items():
            if isinstance(bloq, THCRotations):
                total = get_cost_value(bloq, QECGatesCost()).toffoli
                arithmetic = bloq.num_spin_orb * (bloq.num_bits_theta - 2)
                lookup = total - arithmetic
                print(
                    f"  {'nu' if bloq.two_body_only else 'mu'} "
                    f"{'adjoint' if bloq.is_adjoint else 'forward'} ×{copies}: "
                    f"{arithmetic} arithmetic + {lookup} lookup Toffolis per call"
                )
                modeled_toffolis += copies * total
        print("  Reflection Toffolis:", cost.toffoli - modeled_toffolis)


if __name__ == "__main__":
    main()
