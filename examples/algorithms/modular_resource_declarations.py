"""Query model-only modular declarations without emitting a circuit.

Run python -m examples.algorithms.modular_resource_declarations.
Add --symbolic and/or --legacy-profile for the original shape/profile queries.
"""

import argparse

from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.modular_declarations import (
    ModularDeclaration,
    modular_arithmetic_registers,
    source_modular_arithmetic_resource_formula,
)
from flagquantum.qec.resources.estimation import estimate_symbolic_ftqc_resources


def main(*, symbolic: bool = False, legacy_shims: bool = False) -> None:
    width: SymbolicInt = 3
    if symbolic:
        from sympy import Symbol

        width = Symbol("n", integer=True, positive=True)
    print("Logical QEC profile:", "legacy" if legacy_shims else "modern")
    kinds: tuple[ModularDeclaration, ...] = (
        "multi_controlled_x",
        "controlled_subtraction",
        "less_than",
        "controlled_half",
    )
    for kind in kinds:
        formula = source_modular_arithmetic_resource_formula(
            kind, width, legacy_shims=legacy_shims
        )
        report = estimate_symbolic_ftqc_resources(formula)
        print(kind, "registers", modular_arithmetic_registers(kind, width))
        print("Declared categories and peak:", report.logical)
        print("Declared T equivalents:", report.t_equivalents)
    diagnostic = source_modular_arithmetic_resource_formula("multi_controlled_x", 1)
    print("Original one-control diagnostic (not physical):", diagnostic)
    print(
        "Model declarations only; the original placeholders have no executable decomposition"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbolic", action="store_true")
    parser.add_argument("--legacy-profile", action="store_true")
    options = parser.parse_args()
    main(symbolic=options.symbolic, legacy_shims=options.legacy_profile)
