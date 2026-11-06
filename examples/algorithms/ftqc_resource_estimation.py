"""Estimate fault-tolerant resources from algorithm dimensions.

Run from the repository root with python -m examples.algorithms.ftqc_resource_estimation.
Add --thc 2 4 8 1 for a dimension-only THC SELECT estimate.
Add --single-factorization 10 50 12 for the single-factorization block encoding.
"""

import argparse

from flagquantum.algorithms.primitives.multiplexers import SelectTHC
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.algorithms.primitives.single_factorization_encoding_model import (
    source_single_factorization_block_encoding_cost,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.beverland_physical import (
    estimate_beverland_physical_resources,
)
from flagquantum.qec.resources.estimation import (
    FTQCResourceReport,
    estimate_ftqc_resources,
)
from flagquantum.qec.resources.gidney_fowler import estimate_gidney_fowler_resources


def make_circuit() -> DynamicCircuit:
    op = SelectTHC.make_on(
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
        workspace=tuple(range(13, 39)),
        measurement_erasure=True,
    )
    circuit = DynamicCircuit(39)
    circuit.append(op)
    return circuit


def _report_single_factorization(
    spin_orbitals: int, auxiliaries: int, alias_bits: int
) -> None:
    model = source_single_factorization_block_encoding_cost(
        auxiliaries, spin_orbitals, alias_bits
    )
    report = estimate_ftqc_resources(model, error_budget=1e-3, physical_error=1e-4)
    assert report.surface_code is not None
    gates = report.logical.gates
    print("Single-factorization block encoding (8-bit inner rotation precision)")
    print("Model Toffoli gates", gates.toffoli)
    print("Model controlled swaps", gates.cswap)
    print("Model measurements", gates.measurement)
    print("Model T equivalents", gates.total_t_count())
    print("Model peak logical qubits", report.logical.peak_logical_qubits)
    print(
        "Fast data-block physical qubits",
        report.surface_code.data_block_physical_qubits,
    )


def _print_thc_estimate(report: FTQCResourceReport) -> None:
    gates = report.logical.gates
    print("Model Toffoli gates", gates.toffoli)
    print("Model controlled swaps", gates.cswap)
    print("Model measurements", gates.measurement)
    print("Model T equivalents", gates.total_t_count())
    print("Model peak logical qubits", report.logical.peak_logical_qubits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model_options = parser.add_mutually_exclusive_group()
    model_options.add_argument(
        "--thc",
        nargs=4,
        type=int,
        metavar=("NUM_MU", "SPIN_ORBITALS", "ANGLE_BITS", "KEEP_BITS"),
        help="estimate THC SELECT without angle tables or circuit construction",
    )
    model_options.add_argument(
        "--single-factorization",
        nargs=3,
        type=int,
        metavar=("SPIN_ORBITALS", "AUXILIARIES", "ALIAS_BITS"),
        help="estimate a single-factorization encoding from its dimensions",
    )
    parser.add_argument(
        "--thc-erasure-blocks",
        nargs=2,
        type=int,
        metavar=("KR1", "KR2"),
        help="set the THC SELECT model's erasure block sizes (default: 1 1)",
    )
    args = parser.parse_args()
    if args.thc_erasure_blocks is not None and args.thc is None:
        parser.error("--thc-erasure-blocks requires --thc")
    if args.thc is not None:
        kr1, kr2 = args.thc_erasure_blocks or (1, 1)
        try:
            model = select_thc_cost(*args.thc, kr1=kr1, kr2=kr2)
        except ValueError as exc:
            parser.error(str(exc))
        _print_thc_estimate(estimate_ftqc_resources(model))
        return
    if args.single_factorization is not None:
        _report_single_factorization(*args.single_factorization)
        return

    model = select_thc_cost(num_mu=2, num_spin_orb=4, num_bits_theta=8, keep_bitsize=1)
    planned = estimate_ftqc_resources(model, error_budget=1e-3, physical_error=1e-4)
    assert planned.surface_code is not None
    _print_thc_estimate(planned)
    print(
        "Fast data-block physical qubits",
        planned.surface_code.data_block_physical_qubits,
    )

    factory_scenario = estimate_gidney_fowler_resources(
        planned.logical, data_distance=7, physical_error=1e-4
    )
    print("Gidney-Fowler scenario (data distance 7, 1 microsecond per cycle)")
    print("Data and factory physical qubits", factory_scenario.physical_qubits)
    print("Error-correction cycles", factory_scenario.cycles)
    print("Duration in hours", factory_scenario.duration_hours)
    print("Modeled failure probability", factory_scenario.failure_probability)

    beverland_scenario = estimate_beverland_physical_resources(
        planned.logical, data_distance=25
    )
    print("Beverland architecture with 15-to-1 distillation")
    print("Data and factory physical qubits", beverland_scenario.physical_qubits)
    print("Error-correction cycles", beverland_scenario.cycles)
    print("Duration in hours", beverland_scenario.duration_hours)
    print("Modeled failure probability", beverland_scenario.failure_probability)

    observed = estimate_ftqc_resources(model, emitted_ir=make_circuit().to_ir())
    assert observed.emitted is not None
    print("Emitted T/T-dagger instructions", observed.emitted.t_count)
    print("Emitted measurements", observed.emitted.unconditional.get("measure", 0))
    print("Emitted declared wires", observed.emitted.declared_qubits)
    print("Emitted unconditional opcodes", observed.emitted.unconditional)
    print("Emitted conditional opcodes", observed.emitted.conditional)


if __name__ == "__main__":
    main()
