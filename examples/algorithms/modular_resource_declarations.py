"""Query model-only modular declarations without emitting a circuit.

Run python -m examples.algorithms.modular_resource_declarations.
"""

from flagquantum.compiler.registers.modular_declarations import (
    ModularDeclaration,
    modular_arithmetic_registers,
    source_modular_arithmetic_resource_formula,
)
from flagquantum.qec.resources.estimation import estimate_symbolic_ftqc_resources


def main() -> None:
    kinds: tuple[ModularDeclaration, ...] = (
        "multi_controlled_x",
        "controlled_subtraction",
        "less_than",
        "controlled_half",
    )
    for kind in kinds:
        formula = source_modular_arithmetic_resource_formula(kind, 3)
        report = estimate_symbolic_ftqc_resources(formula)
        print(kind, "registers", modular_arithmetic_registers(kind, 3))
        print("Declared categories and peak:", report.logical)
        print("Declared T equivalents:", report.t_equivalents)
    diagnostic = source_modular_arithmetic_resource_formula("multi_controlled_x", 1)
    print("Original one-control diagnostic (not physical):", diagnostic)
    print(
        "Model declarations only; the original placeholders have no executable decomposition"
    )


if __name__ == "__main__":
    main()
