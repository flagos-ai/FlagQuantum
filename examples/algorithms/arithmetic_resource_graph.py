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
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.compiler.registers.arithmetic.constant_add_resource_model import (
    source_constant_add_resource_formula,
    source_controlled_constant_add_resource_formula,
)
from flagquantum.compiler.registers.arithmetic.grouped_constant_add_resource_model import (
    source_grouped_constant_add_resource_formula,
)
from flagquantum.compiler.registers.arithmetic.out_of_place_add_resource_model import (
    source_out_of_place_add_resource_formula,
)
from flagquantum.compiler.registers.bits.hamming_weight_resource_model import (
    source_hamming_weight_resource_formula,
)
from flagquantum.compiler.registers.bits.xor_resource_model import (
    source_grouped_xor_resource_formula,
    source_pattern_controlled_xor_resource_formula,
    source_xor_resource_formula,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister


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
    for width in (3, n, Symbol("z", integer=True)):
        print("Plain XOR model:", source_xor_resource_formula(width))
        print(
            "Grouped XOR model:",
            source_grouped_xor_resource_formula(width, (ControlRegister(2, (2,)),)),
        )
    for width in (3, Integer(3), n):
        print(
            "Hamming count model:",
            type(width).__name__,
            source_hamming_weight_resource_formula(width),
        )
        print(
            "Hamming erase model:",
            source_hamming_weight_resource_formula(width, adjoint=True),
        )
    controls = Symbol("m", integer=True, positive=True)
    for shape in ((controls,), ()):
        print(
            "Unknown activation, control shape:",
            shape,
            source_pattern_controlled_xor_resource_formula(
                3, controls, control_shapes=(shape,)
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
    registers = (ControlRegister(2, (1,)),)
    for width, value in ((3, 5), (3, Integer(5)), (Integer(3), 5), (n, k)):
        print(
            "Grouped control model:",
            type(width).__name__,
            type(value).__name__,
            source_grouped_constant_add_resource_formula(width, value, registers),
        )
    print("No circuit emitted; no measured primitive counts or work-qubit claim.")


if __name__ == "__main__":
    main()
