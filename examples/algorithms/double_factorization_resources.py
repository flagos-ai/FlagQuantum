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
    source_double_factorization_one_body_signature,
)
from flagquantum.algorithms.primitives.double_factorization_encoding_model import (
    source_double_factorization_block_signature as double_factorization_block_signature,
)
from flagquantum.algorithms.primitives.double_factorization_prepare_model import (
    source_double_factorization_indexed_data_cost,
    source_double_factorization_indexed_data_signature,
    source_double_factorization_inner_prepare_cost,
    source_double_factorization_inner_prepare_signature,
    source_double_factorization_outer_prepare_cost,
    source_double_factorization_outer_prepare_signature,
)
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.register_score import (
    RegisterScore,
    RegisterStep,
    layout_register_score,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

_AUXILIARIES = 50
_SPIN_ORBITALS = 10
_EIGENVALUES = _AUXILIARIES * (_SPIN_ORBITALS // 2)


def notebook_models() -> tuple[ComponentCost, ComponentCost]:
    """Return the one-body and full block with the original notebook parameters."""
    one_body = source_double_factorization_one_body_cost(
        _SPIN_ORBITALS, _EIGENVALUES, alias_bits=12, basis_rotation_bits=7
    )
    block = source_double_factorization_block_encoding_cost(
        _AUXILIARIES,
        _SPIN_ORBITALS,
        _EIGENVALUES,
        alias_bits=12,
        basis_rotation_bits=7,
        inner_rotation_bits=7,
    )
    return one_body, block


def prepare_examples() -> tuple[tuple[str, Signature, ComponentCost], ...]:
    """Return the source PREPARE examples with their display names."""
    return (
        (
            "In-Prep",
            source_double_factorization_inner_prepare_signature(
                _SPIN_ORBITALS, _EIGENVALUES, 7
            ),
            source_double_factorization_inner_prepare_cost(
                _SPIN_ORBITALS, _EIGENVALUES, 7, 12
            ),
        ),
        (
            "OuterPrep",
            source_double_factorization_outer_prepare_signature(_AUXILIARIES),
            source_double_factorization_outer_prepare_cost(_AUXILIARIES, 12),
        ),
        (
            "In_l-data_l",
            source_double_factorization_indexed_data_signature(
                _AUXILIARIES, _SPIN_ORBITALS, _EIGENVALUES, 18
            ),
            source_double_factorization_indexed_data_cost(
                _AUXILIARIES, _SPIN_ORBITALS, _EIGENVALUES, 18
            ),
        ),
    )


def notebook_signatures() -> tuple[Signature, Signature]:
    """Declare the input and output words of both source notebook examples."""
    one_body = source_double_factorization_one_body_signature(
        _SPIN_ORBITALS, _EIGENVALUES, 7, 8
    )
    return one_body, double_factorization_block_signature(
        _AUXILIARIES, _SPIN_ORBITALS, _EIGENVALUES, 7, 7
    )


def notebook_block_score() -> RegisterScore:
    """Layout the original seven-stage teaching example."""
    _, signature = notebook_signatures()
    _, model = notebook_models()
    return double_factorization_block_score(model, signature)


def double_factorization_block_score(
    model: ComponentCost, signature: Signature
) -> RegisterScore:
    """Layout the source's seven calls and their live register operands."""
    widths = {
        reg.name + "".join(f"[{i}]" for i in indices): int(reg.bitsize)
        for reg in signature
        for indices in reg.all_idxs()
    }
    outer = ("l", "ctrl[0]")
    data = ("l", "ctrl[1]", "xi", "rot", "offset")
    body = (
        "ctrl[0]",
        "ctrl[1]",
        "ctrl[3]",
        "p",
        "rot_aa",
        "spin",
        "xi",
        "offset",
        "rot",
        "rotations",
        "sys[0]",
        "sys[1]",
    )
    reflect = ("ctrl[0]", "p", "spin")
    return layout_register_score(
        widths,
        tuple(widths),
        tuple(
            RegisterStep(call.component.name, ports, ports)
            for call, ports in zip(
                model.calls,
                (outer, data, body, reflect, body, data, outer),
                strict=True,
            )
        ),
        input_registers=tuple(signature),
        output_registers=tuple(signature),
    )


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
