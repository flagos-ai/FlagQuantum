# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 hamiltonian_simulation_by_gqsp.ipynb.

"""Hubbard examples for the GQSP Hamiltonian simulation notebook."""

from examples.algorithms.hubbard_qubitization_resources import notebook_signature
from flagquantum.algorithms.primitives.hamiltonian_gqsp import (
    hamiltonian_gqsp_source_graph,
)
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt, SymbolicReal, floor


def hubbard_gqsp_signature() -> Signature:
    """Use the native Hubbard name order for the source's named port union."""
    return Signature((Register("signal", quantum_bit_type()), *notebook_signature(2)))


def hubbard_gqsp_scale(
    hopping: SymbolicReal, interaction: SymbolicReal
) -> SymbolicReal:
    """Return the two-by-two source normalization, including floor division."""
    return 16 * hopping + floor(4 * interaction)


def hubbard_gqsp_graph(
    hopping: SymbolicReal,
    interaction: SymbolicReal,
    time: SymbolicReal,
    precision: SymbolicReal,
    *,
    depth: int = 1,
) -> SymbolicCallGraph[str]:
    """Expand one more layer with angle-generalized rotations and scoped calls."""
    graph = hamiltonian_gqsp_source_graph(
        hubbard_gqsp_scale(hopping, interaction), time, precision
    )
    # The two locked examples share the same direct PREPARE structure.
    template = PrepareHubbard(2, t=1, u=1).source_resource_model()
    prepare: dict[str, SymbolicInt] = {}
    for call in template.calls:
        child = call.component
        name = child.name
        if name == "CSwap":
            name = f"CSwap({child.gates.cswap})"
        elif name == "Controlled":
            name = "ControlledRy"
        elif name == "Adjoint":
            name = "MultiAnd†"
        prepare[name] = call.repetitions
    inverse_keys = {
        "ControlledRy": "ControlledRy†",
        "PrepareUniformSuperposition(2)": "PrepareUniformSuperposition(2)†",
        "ModAddK": "ModAddK†",
    }
    return symbolic_call_graph(
        graph.root,
        {
            graph.root: graph.leaf_counts,
            "Always(PREPARE)": prepare,
            "Always(PREPARE†)": {
                inverse_keys.get(name, name): count for name, count in prepare.items()
            },
            "ZeroControlledWalk": {"PositiveControlledWalk": 1, "XGate": 2},
            "PositiveControlledWalk†": {
                "ControlledSelectBlock†": 1,
                "PositiveControlledReflection": 1,
            },
            "SU2Rotation": {"GlobalPhase": 1, "Rz": 2, "Rx": 1},
        },
        max_depth=depth,
    )
