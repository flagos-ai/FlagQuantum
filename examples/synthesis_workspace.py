"""Private Compiler trial: measured retain/recompute costs for one source program.

Run with ``python -m examples.synthesis_workspace``. Candidate selection is a
CS-06 task; this diagnostic enumerates the two deterministic CS-05 schedules.
"""

import math

import torch

import flagquantum as fq
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import (
    ArithmeticCandidate,
    Schedule,
    schedule_arithmetic,
)


def pressure(x: int, y: int) -> int:
    t = x * y
    a = t * x
    u = a * y
    return t + u


def candidate(width: int, strategy: Schedule) -> ArithmeticCandidate:
    graph = capture(pressure, {"x": width, "y": width})
    block = graph.body.blocks[0]
    result_width = block.operations[-1].operands[0].type.parameters[0]
    start = 2 * width + result_width
    inputs = {
        value.id: tuple(range(i * width, (i + 1) * width))
        for i, value in enumerate(block.arguments)
    }
    output = tuple(range(2 * width, start))
    probe = schedule_arithmetic(
        (graph,),
        inputs,
        output,
        entry=graph.name,
        n_qubits=128,
        workspace=tuple(range(start, 128)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )
    total = start + probe.peak_workspace
    return schedule_arithmetic(
        (graph,),
        inputs,
        output,
        entry=graph.name,
        n_qubits=total,
        workspace=tuple(range(start, total)),
        workspace_state="fresh_zero",
        scheduling=strategy,
    )


def main() -> None:
    print("bits strategy workspace qubits operations depth recomputations")
    for width in (1, 2, 3):
        for strategy in ("retain", "recompute"):
            product = candidate(width, strategy)
            resources = estimate_resources(product.circuit)
            print(
                width,
                strategy,
                product.peak_workspace,
                resources.n_qubits,
                resources.n_operations,
                resources.depth,
                product.recomputations,
            )
            if width == 1:
                # The developer-only block path has explicit physical ports.
                # Normal CS-04 named preparation/readout remains unchanged.
                circuit = fq.Circuit(resources.n_qubits, dtype=torch.complex128)
                circuit.h(0).h(1).compose(product.circuit)
                result = fq.run(
                    circuit,
                    options=fq.ExecutionOptions(
                        mode="statevector",
                        backend="torch",
                        device="cpu",
                        precision="complex128",
                        require_gradients=False,
                        allow_approximate=False,
                        allow_backend_fallback=False,
                    ),
                )
                state = result.statevector().reshape(-1)
                for x in range(2):
                    for y in range(2):
                        index = (
                            ((x << 1) | y) << 2 | pressure(x, y)
                        ) << product.peak_workspace
                        assert math.isclose(
                            float(state[index].real), 0.5, abs_tol=1e-12
                        )
                        assert abs(float(state[index].imag)) <= 1e-12
    print("exact coherent CPU execution: both one-bit schedules passed")
    print("Two/three-bit rows are compiler costs; no large dense state was executed.")


if __name__ == "__main__":
    main()
