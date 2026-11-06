"""Reproduce the concrete and symbolic Hubbard notebook resource queries."""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's hubbard_model.ipynb and select_hubbard.ipynb.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/hubbard_model/qubitization

from argparse import ArgumentParser
from dataclasses import asdict

from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.algorithms.primitives.hubbard_qubitization_resource_model import (
    source_hubbard_majorana_formula,
    source_hubbard_spin_up_formula,
)
from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.qec.resources.estimation import (
    estimate_ftqc_resources,
    estimate_symbolic_ftqc_resources,
)


def notebook_models() -> tuple[tuple[int, ComponentCost], ...]:
    """Return the concrete top-level models used in the locked notebooks."""
    return (
        (4, SelectHubbard(4).source_resource_model()),
        (4, PrepareHubbard(4, t=1.0, u=4.0).source_resource_model()),
        (11, SelectHubbard(11, controlled=True).source_resource_model()),
    )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symbolic",
        action="store_true",
        help="Also show symbolic controlled leaf models",
    )
    args = parser.parse_args()
    for side, model in notebook_models():
        print(f"{side}×{side} {model.name}")
        graph = modeled_call_graph(model, max_depth=1)
        for child, count in graph.leaf_counts.items():
            cost = estimate_ftqc_resources(child).logical.gates
            categories = {name: value for name, value in asdict(cost).items() if value}
            print(f"  {count} × {child.name}: {categories}")
        result = estimate_ftqc_resources(model).logical
        print("  Total logical gates:", asdict(result.gates))
        print("  Peak logical qubits:", result.peak_logical_qubits)
    if args.symbolic:
        from sympy import Symbol

        side = Symbol("n")
        for name, formula in (
            (
                "Controlled Majorana X",
                source_hubbard_majorana_formula(side, controlled=True),
            ),
            (
                "Controlled spin-up Z",
                source_hubbard_spin_up_formula(side, controlled=True),
            ),
        ):
            report = estimate_symbolic_ftqc_resources(formula)
            print(name)
            print("  Logical resources:", asdict(report.logical))
            print("  T equivalents:", report.t_equivalents)


if __name__ == "__main__":
    main()
