"""Count emitted 8-bit SelectTHC instructions and optionally show an upstream model.

Run with `python -m examples.algorithms.select_thc_resources`. Add `--qualtran`
in an environment with Qualtran 0.7.0 to include its separate model estimate.
"""

import argparse
from collections import Counter
from importlib.metadata import version

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectTHC
from flagquantum.compiler.resource_estimation import count_instruction_resources
from flagquantum.dynamic import DynamicCircuit


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
                networks=(((0, 1, 0.2),), ((0, 1, 0.5),)),
                one_body_networks=((), ((0, 1, -0.4),)),
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
        print("  Includes phase-gradient state preparation and cleanup.")

    if args.qualtran:
        if version("qualtran") != "0.7.0":
            parser.error("--qualtran requires Qualtran 0.7.0")
        from qualtran.bloqs.chemistry.thc.select_bloq import SelectTHC as CostModel
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
        print("  Assumes a phase-gradient state and charges address swaps to PREPARE.")


if __name__ == "__main__":
    main()
