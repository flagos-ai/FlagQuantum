"""Estimate a symbolic arithmetic call graph without constructing a circuit.

Run python -m examples.algorithms.arithmetic_resource_graph.
The n=3 subtraction declaration has 5 And, 23 Clifford and 5 measurements;
its terminal T estimate is 20. These are model counts, not emitted gates.
"""

from sympy import Integer, Symbol, sympify

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from flagquantum.compiler._component_trace.terminal_t_cost import (
    estimate_terminal_t_cost,
)
from flagquantum.compiler.registers.arithmetic_declarations import arithmetic_call_graph
from flagquantum.compiler.registers.constant_add_resource_model import (
    source_constant_add_resource_formula,
    source_controlled_constant_add_resource_formula,
)
from flagquantum.compiler.registers.out_of_place_add_resource_model import (
    source_out_of_place_add_resource_formula,
)


def main() -> None:
    n = Symbol("n", integer=True, positive=True)
    root = ArithmeticCall("controlled_subtraction", (n,))
    declarations = arithmetic_call_graph(root)
    graph = symbolic_call_graph(root, declarations)
    gates = query_qec_gate_cost(root, declared_calls=declarations)
    terminal_t = estimate_terminal_t_cost(graph.leaf_counts)
    print("Declared logical QEC categories:", gates)
    print("Declared terminal T estimate:", terminal_t)
    print(
        "n=3:",
        {
            "And": sympify(gates.and_bloq).subs(n, 3),
            "Clifford": sympify(gates.clifford).subs(n, 3),
            "measurements": sympify(gates.measurement).subs(n, 3),
            "terminal_T": sympify(terminal_t).subs(n, 3),
        },
    )
    for msb in (False, True):
        for adjoint in (False, True):
            print(
                "Out-of-place sum model:",
                {"include_msb": msb, "adjoint": adjoint},
                source_out_of_place_add_resource_formula(
                    n, include_msb=msb, adjoint=adjoint
                ),
            )
    k = Symbol("k", integer=True)
    for label, width, value in (
        ("Python width and constant", 3, 5),
        ("Python width, symbolic integer constant", 3, Integer(5)),
        ("symbolic width and constant", n, k),
    ):
        print(
            "Constant-add model:",
            label,
            source_constant_add_resource_formula(width, value),
        )
    for control_value in (0, 1):
        print(
            "Controlled constant-add model:",
            control_value,
            source_controlled_constant_add_resource_formula(
                3, 5, control_value=control_value
            ),
        )
    print("No circuit emitted; no measured primitive counts or work-qubit claim.")


if __name__ == "__main__":
    main()
