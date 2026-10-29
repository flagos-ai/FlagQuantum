"""Reproduce static CS06-B candidate costs and the historical manual recipe.

Run ``python -m examples.synthesis_selection_costs``. This emits JSON containing
compiler costs and elapsed construction observations; it executes no statevector.
"""

import json
from time import perf_counter

import flagquantum as fq
from examples.synthesis_selection import addition, polynomial, pressure, product
from flagquantum.algorithms.arithmetic import adder_circuit
from flagquantum.compiler import estimate_resources
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.selection import select_arithmetic
from flagquantum.compiler._construction.target import Target, compile_target
from flagquantum.core.ir import CircuitIR

BASIS = ("x", "h", "s", "sdg", "t", "tdg", "ry", "rz", "cx", "u3", "swap")


def manual_polynomial() -> CircuitIR:
    """Rebuild the original two-bit partial-product/adder compute-copy-uncompute."""
    x, y, z = (0, 1), (2, 3), (4, 5, 6, 7)
    partial, accumulator = (8, 9, 10, 11), (12, 13, 14, 15)
    carry, carry_out = 16, 17
    compute = fq.Circuit(18, dtype="complex128")
    adder = adder_circuit(4)
    adder_map = [*partial, *accumulator, carry, carry_out]
    for shift, control in enumerate(reversed(x)):
        row = fq.Circuit(18, dtype="complex128")
        for offset, operand in enumerate(reversed(x)):
            destination = partial[3 - shift - offset]
            if control == operand:
                row.cx(control, destination)
            else:
                row.gate("ccx", (control, operand, destination))
        compute.compose(row).compose(adder, qubits=adder_map).compose(row.adjoint())
    load_y = fq.Circuit(18, dtype="complex128")
    for source, destination in zip(y, partial[2:], strict=True):
        load_y.cx(source, destination)
    compute.compose(load_y).compose(adder, qubits=adder_map).compose(load_y.adjoint())
    circuit = fq.Circuit(18, dtype="complex128").compose(compute)
    for source, destination in zip(accumulator, z, strict=True):
        circuit.cx(source, destination)
    circuit.compose(compute.adjoint())
    program = circuit.to_ir()
    assert isinstance(program, CircuitIR)
    return program


def cost_matrix() -> list[dict[str, object]]:
    """All 32 successful rows; larger pressure states are not simulated."""
    rows: list[dict[str, object]] = []
    for target_name, target in (
        ("logical", Target()),
        ("named_basis", Target(basis=BASIS)),
    ):
        for function in (addition, product, polynomial, pressure):
            for width in (1, 2):
                widths = {"x": width, "y": width + int(function is addition)}
                graph = capture(function, widths)
                report = select_arithmetic(
                    (graph,), entry=graph.name, input_names=("x", "y"), target=target
                )
                assert report.selected is not None
                for row in report.rows:
                    assert (
                        row.target is not None
                        and row.resources is not None
                        and row.arithmetic is not None
                    )
                    rows.append(
                        {
                            "function": function.__name__,
                            "inputs": widths,
                            "target": target_name,
                            "strategy": row.strategy,
                            "workspace": row.arithmetic.peak_workspace,
                            "qubits": row.resources.n_qubits,
                            "used_qubits": row.resources.used_qubits,
                            "operations": row.resources.n_operations,
                            "depth": row.resources.depth,
                            "recomputations": row.arithmetic.recomputations,
                            "construction_seconds": row.construction_seconds,
                            "target_seconds": row.target.elapsed_seconds,
                            "selected": row is report.selected,
                            "feasible": row.feasible,
                            "executable_hash": row.target.executable.content_hash,
                        }
                    )
        start = perf_counter()
        manual = manual_polynomial()
        construction = perf_counter() - start
        compiled = compile_target(manual, target, 100_000)
        resources = estimate_resources(compiled.executable)
        rows.append(
            {
                "function": "polynomial",
                "inputs": {"x": 2, "y": 2},
                "target": target_name,
                "strategy": "manual",
                "workspace": 10,
                "qubits": resources.n_qubits,
                "used_qubits": resources.used_qubits,
                "operations": resources.n_operations,
                "depth": resources.depth,
                "construction_seconds": construction,
                "target_seconds": compiled.elapsed_seconds,
                "executable_hash": compiled.executable.content_hash,
            }
        )
    return rows


if __name__ == "__main__":
    print(json.dumps(cost_matrix(), indent=2))
