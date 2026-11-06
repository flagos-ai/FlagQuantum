"""Estimate 8-bit THC SELECT before and after emitting a circuit.

Run from the repository root with python -m examples.algorithms.ftqc_resource_estimation.
"""

from flagquantum.algorithms.primitives.multiplexers import SelectTHC
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
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


def main() -> None:
    model = select_thc_cost(num_mu=2, num_spin_orb=4, num_bits_theta=8, keep_bitsize=1)
    planned = estimate_ftqc_resources(model, error_budget=1e-3, physical_error=1e-4)
    assert planned.surface_code is not None
    print("Model T equivalents", planned.logical.gates.total_t_count())
    print("Model peak logical qubits", planned.logical.peak_logical_qubits)
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

    observed = estimate_ftqc_resources(model, emitted_ir=make_circuit().to_ir())
    assert observed.emitted is not None
    print("Emitted T/T-dagger instructions", observed.emitted.t_count)
    print("Emitted measurements", observed.emitted.unconditional.get("measure", 0))
    print("Emitted declared wires", observed.emitted.declared_qubits)
    print("Emitted unconditional opcodes", observed.emitted.unconditional)
    print("Emitted conditional opcodes", observed.emitted.conditional)


if __name__ == "__main__":
    main()
