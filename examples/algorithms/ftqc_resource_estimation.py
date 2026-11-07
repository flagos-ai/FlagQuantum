"""Estimate fault-tolerant resources from algorithm dimensions.

Run from the repository root with python -m examples.algorithms.ftqc_resource_estimation.
Add --thc 2 4 8 1 for a dimension-only THC SELECT estimate.
Add --single-factorization 10 50 12 for the single-factorization block encoding.
Add --add 8 8 or --qrom 17 8 --qrom-controls 1 for register-level models.
Add --json to any dimension-only model to export a structured analytical report.
"""

import argparse
import json

from flagquantum.algorithms.primitives.multiplexers import SelectTHC
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.algorithms.primitives.single_factorization_encoding_model import (
    source_single_factorization_block_encoding_cost,
)
from flagquantum.compiler._component_trace.gate_budget import (
    t_and_ccz_budget,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler.registers.symbolic_resource_model import (
    SymbolicResourceFormula,
    source_add_resource_formula,
    source_qrom_resource_formula,
)
from flagquantum.compiler.resource_reporting import resource_model_report
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.beverland_physical import (
    estimate_beverland_physical_resources,
)
from flagquantum.qec.resources.estimation import (
    FTQCResourceReport,
    estimate_ftqc_resources,
    estimate_symbolic_ftqc_resources,
)
from flagquantum.qec.resources.gidney_fowler import estimate_gidney_fowler_resources


def make_select() -> SelectTHC:
    return SelectTHC.make_on(
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


def make_circuit(op: SelectTHC | None = None) -> DynamicCircuit:
    circuit = DynamicCircuit(39)
    circuit.append(make_select() if op is None else op)
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
    budget = t_and_ccz_budget(gates)
    print("Model Toffoli gates", gates.toffoli)
    print("Model controlled swaps", gates.cswap)
    print("Model measurements", gates.measurement)
    print("Model T states", budget["n_t"])
    print("Model CCZ states", budget["n_ccz"])
    print("Model T equivalents", gates.total_t_count())
    print("Model peak logical qubits", report.logical.peak_logical_qubits)


def _print_register_estimate(formula: SymbolicResourceFormula) -> None:
    report = estimate_symbolic_ftqc_resources(formula)
    print("Basis: declared component model; no circuit construction or execution")
    for name in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        print("Model", name, getattr(formula, name))
    print("Model T states", report.t_states)
    print("Model CCZ states", report.ccz_states)
    print("Model T equivalents", report.t_equivalents)
    print("Model peak logical qubits", formula.logical_qubits)


def _print_json_estimate(model: ComponentCost | SymbolicResourceFormula) -> None:
    """Serialize estimates through the shared compiler report entry."""
    print(json.dumps(resource_model_report(model), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model_options = parser.add_mutually_exclusive_group()
    model_options.add_argument(
        "--add",
        nargs=2,
        type=int,
        metavar=("LEFT_BITS", "RIGHT_BITS"),
        help="estimate unsigned Add from word widths, without emitting gates",
    )
    model_options.add_argument(
        "--qrom",
        nargs=2,
        type=int,
        metavar=("ROWS", "TARGET_BITS"),
        help="estimate a shape-only QROM without allocating its table",
    )
    parser.add_argument(
        "--qrom-controls",
        type=int,
        help="number of QROM controls (default: 0); requires --qrom",
    )
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a dimension-only model report; requires a model option",
    )
    args = parser.parse_args()
    if args.json and all(
        value is None
        for value in (args.add, args.qrom, args.thc, args.single_factorization)
    ):
        parser.error("--json requires --add, --qrom, --thc or --single-factorization")
    if args.thc_erasure_blocks is not None and args.thc is None:
        parser.error("--thc-erasure-blocks requires --thc")
    if args.qrom_controls is not None and args.qrom is None:
        parser.error("--qrom-controls requires --qrom")
    if args.add is not None:
        try:
            formula = source_add_resource_formula(*args.add)
        except ValueError as exc:
            parser.error(str(exc))
        if args.json:
            _print_json_estimate(formula)
        else:
            _print_register_estimate(formula)
        return
    if args.qrom is not None:
        rows, target_bits = args.qrom
        controls = args.qrom_controls if args.qrom_controls is not None else 0
        if rows < 2 or target_bits < 1 or controls < 0:
            parser.error(
                "QROM requires at least two rows, positive target bits and nonnegative controls"
            )
        formula = source_qrom_resource_formula(
            (rows,), (target_bits,), control_count=controls
        )
        if args.json:
            _print_json_estimate(formula)
        else:
            print(
                "QROM shape model: table contents and uniform-region optimizations are unspecified"
            )
            _print_register_estimate(formula)
        return
    if args.thc is not None:
        kr1, kr2 = args.thc_erasure_blocks or (1, 1)
        try:
            model = select_thc_cost(*args.thc, kr1=kr1, kr2=kr2)
        except ValueError as exc:
            parser.error(str(exc))
        if args.json:
            _print_json_estimate(model)
        else:
            _print_thc_estimate(estimate_ftqc_resources(model))
        return
    if args.single_factorization is not None:
        if args.json:
            spin_orbitals, auxiliaries, alias_bits = args.single_factorization
            _print_json_estimate(
                source_single_factorization_block_encoding_cost(
                    auxiliaries, spin_orbitals, alias_bits
                )
            )
        else:
            _report_single_factorization(*args.single_factorization)
        return

    op = make_select()
    model = op.source_resource_model(keep_bitsize=1)
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

    observed = estimate_ftqc_resources(model, emitted_ir=make_circuit(op).to_ir())
    assert observed.emitted is not None
    print("Emitted T/T-dagger instructions", observed.emitted.t_count)
    print("Emitted measurements", observed.emitted.unconditional.get("measure", 0))
    print("Emitted declared wires", observed.emitted.declared_qubits)
    print("Emitted unconditional opcodes", observed.emitted.unconditional)
    print("Emitted conditional opcodes", observed.emitted.conditional)


if __name__ == "__main__":
    main()
