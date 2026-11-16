"""Independent classical cost table and explicit Circuit mixer layers."""

import flagquantum as fq
from examples.synthesis.applications.physical_systems.ising_model.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.cost_reference import cost_phase_circuit

BINDINGS: tuple[tuple[float, ...], ...] = (
    (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1),
    (
        -0.014598540145985401,
        0.021897810218978103,
        -0.029197080291970802,
        0.0364963503649635,
        -0.043795620437956206,
        0.051094890510948905,
        -0.058394160583941604,
        0.06569343065693431,
        -0.072992700729927,
        0.08029197080291971,
    ),
)


def cost(bits: tuple[int, ...]) -> float:
    spins = tuple(2 * b - 1 for b in bits)
    ring = sum(spins[i] * spins[(i + 1) % 6] for i in range(6))
    return float(10 * ring - 20 * sum(spins))


def reference_circuit(
    params: tuple[float, ...] = (
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.1,
    )
) -> fq.Circuit:
    return cost_phase_circuit(cost, 6, params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("v",))


if __name__ == "__main__":
    run_example()
