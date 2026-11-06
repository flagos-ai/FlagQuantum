"""Run the locked double-factorization notebook's resource queries.

Run with python -m examples.algorithms.double_factorization_resources.
"""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's double-factorization notebook.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/df/double_factorization.ipynb

from dataclasses import asdict

from flagquantum.algorithms.primitives.double_factorization_encoding_model import (
    source_double_factorization_block_encoding_cost,
    source_double_factorization_one_body_cost,
)
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.qec.resources.estimation import estimate_ftqc_resources


def notebook_models() -> tuple[ComponentCost, ComponentCost]:
    """Return the one-body and full block with the original notebook parameters."""
    auxiliaries, spin_orbitals = 50, 10
    eigenvalues = auxiliaries * (spin_orbitals // 2)
    one_body = source_double_factorization_one_body_cost(
        spin_orbitals, eigenvalues, alias_bits=12, basis_rotation_bits=7
    )
    block = source_double_factorization_block_encoding_cost(
        auxiliaries,
        spin_orbitals,
        eigenvalues,
        alias_bits=12,
        basis_rotation_bits=7,
        inner_rotation_bits=7,
    )
    return one_body, block


def main() -> None:
    for model in notebook_models():
        print(model.name)
        print(
            "  Ordered stages:",
            " -> ".join(call.component.name for call in model.calls),
        )
        graph = modeled_call_graph(model, max_depth=1)
        print(
            "  Direct calls:",
            [(node.name, count) for node, count in graph.leaf_counts.items()],
        )
        report = estimate_ftqc_resources(model)
        print("  Logical gates:", asdict(report.logical.gates))
        print("  Peak logical qubits:", report.logical.peak_logical_qubits)


if __name__ == "__main__":
    main()
