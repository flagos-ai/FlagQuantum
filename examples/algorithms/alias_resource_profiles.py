"""Query dense/sparse alias model profiles without tables or circuit emission.

Run python -m examples.algorithms.alias_resource_profiles.
Add --symbolic for the locked symbolic declarations (requires Sympy).
"""

import argparse

from flagquantum.algorithms.primitives.symbolic_alias_preparation import (
    AliasPreparationShape,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.qec.resources.estimation import estimate_symbolic_ftqc_resources


def main(*, symbolic: bool = False) -> None:
    rows: SymbolicInt = 5
    domain: SymbolicInt = 16
    if symbolic:
        from sympy import Symbol

        rows = Symbol("L", positive=True, integer=True)
        domain = Symbol("N", positive=True, integer=True)
    dense = AliasPreparationShape(rows, rows, 3)
    sparse = AliasPreparationShape(domain, 4, 3, support_size=rows)
    for label, operation in (("dense", dense), ("sparse", sparse)):
        for legacy in (False, True):
            report = estimate_symbolic_ftqc_resources(
                operation.resource_model(legacy_shims=legacy)
            )
            print(label, "legacy" if legacy else "modern", "categories", report.logical)
            print("Model T equivalents", report.t_equivalents)
    print("No circuit emission; formulas retain locked source declarations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbolic", action="store_true")
    main(symbolic=parser.parse_args().symbolic)
