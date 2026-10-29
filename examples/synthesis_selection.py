"""Private CS06-B expert workflow; the ordinary user facade is CS06-C work."""

import torch

import flagquantum as fq
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.selection import Constraints, select_arithmetic
from flagquantum.services._synthesis_execution import inspect_execution


def addition(x: int, y: int) -> int:
    return x + y


def product(x: int, y: int) -> int:
    return x * y


def polynomial(x: int, y: int) -> int:
    return x * x + y


def pressure(x: int, y: int) -> int:
    t = x * y
    a = t * x
    u = a * y
    return t + u


def main() -> None:
    graph = capture(pressure, {"x": 1, "y": 1})
    preparation = fq.Circuit(2, dtype=torch.complex128).h(0).h(1).s(0).to_ir()
    for limits in (CircuitLimits(max_qubits=9), CircuitLimits(max_operations=42)):
        report = select_arithmetic(
            (graph,),
            entry=graph.name,
            input_names=("x", "y"),
            preparation=preparation,
            constraints=Constraints(limits),
        )
        assert report.selected is not None and report.selected.target is not None
        row = report.selected
        assert row.target is not None
        for candidate in report.rows:
            assert candidate.resources is not None and candidate.arithmetic is not None
            print(
                candidate.strategy,
                "workspace",
                candidate.arithmetic.peak_workspace,
                "qubits",
                candidate.resources.n_qubits,
                "operations",
                candidate.resources.n_operations,
                "depth",
                candidate.resources.depth,
                "feasible",
                candidate.feasible,
            )
        print("selected:", row.strategy)
        plan = fq.plan(
            row.target.submitted,
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
        checked = inspect_execution(plan, limits=limits, registers=report.registers)
        result = fq.run(checked.plan)
        assert checked.verify_result(result) == row.resources
        state = result.statevector().reshape(-1)
        reference = torch.zeros_like(state)
        assert row.arithmetic is not None
        for x in range(2):
            for y in range(2):
                index = (
                    (((x << 1) | y) << 2) | pressure(x, y)
                ) << row.arithmetic.peak_workspace
                reference[index] = torch.tensor(1j if x else 1, dtype=state.dtype) / 2
        torch.testing.assert_close(state, reference, atol=1e-12, rtol=0)
    print("constraint-driven schedules and exact phase/clean workspace: passed")
    print("Static final costs, not hardware timing or an optimality claim.")


if __name__ == "__main__":
    main()
