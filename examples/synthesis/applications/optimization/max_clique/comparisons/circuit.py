"""Independent classical cost table and explicit Circuit mixer layers."""

import flagquantum as fq
from examples.synthesis.applications.optimization.max_clique.main import build_program
from examples.synthesis.compare_states import compare_states
from examples.synthesis.cost_reference import cost_phase_circuit

BINDINGS: tuple[tuple[float, ...], ...] = (
    (0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
    (
        -0.014598540145985401,
        0.021897810218978103,
        -0.029197080291970802,
        0.0364963503649635,
        -0.043795620437956206,
        0.051094890510948905,
    ),
)


def cost(bits: tuple[int, ...]) -> float:
    forbidden = ((0, 5), (1, 4), (2, 4), (2, 6), (3, 6), (4, 6))
    violations = sum(bits[a] * bits[b] for a, b in forbidden)
    return float(sum(bits) - 4 * violations**2)


def reference_circuit(
    params: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
) -> fq.Circuit:
    return cost_phase_circuit(cost, 7, params)


def run_example() -> None:
    for params in BINDINGS:
        compare_states(build_program(params), reference_circuit(params), ("v",))


if __name__ == "__main__":
    run_example()
