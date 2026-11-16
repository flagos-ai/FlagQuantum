# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 tutorial_guided_hamiltonian.ipynb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/hamiltonian_simulation/tutorial_guided_hamiltonian.ipynb

"""Generalized leaf counts for the tutorial's declared graph oracles."""

from collections.abc import Mapping

from sympy import Dummy, Expr, pi, simplify, sympify

from flagquantum.algorithms.primitives.guided_hamiltonian_resource_model import (
    GuidedHamiltonianModel,
)
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.core.value_math import SymbolicInt, ceil, floor, log2, ssum

_WALK_IDENTITY = "PrepareIdentity[walk]"
_ARBITRARY = "ArbitraryGate"
_CONTROLLED_ARBITRARY = "ControlledArbitraryGate"


def _reflection_leaves(
    identity: str, identity_calls: int, gate_calls: int, and_calls: SymbolicInt
) -> dict[str, SymbolicInt]:
    return {
        identity: identity_calls,
        _ARBITRARY: gate_calls,
        "And": and_calls,
        "And†": and_calls,
    }


def tutorial_query_graph(
    model: GuidedHamiltonianModel,
    guide_gates: SymbolicInt,
    *,
    amplify: bool,
) -> SymbolicCallGraph[str]:
    """Expand the existing guided model with the tutorial's generalized leaves.

    The tutorial uses an N+1 walk-ancilla register, entry and position oracles,
    and a declared guide cost. The entry oracle is self-adjoint. The position
    oracle keeps both inverse and control as distinct identities.
    """
    n = model.system_width
    if model.walk_ancilla_width != n + 1:
        raise ValueError("Tutorial walk ancilla width must equal system width plus one")
    phase = model.phase_estimation
    base = model.source_call_graph() if amplify else phase.source_call_graph()
    calls: dict[str, dict[str, SymbolicInt]] = {}
    for (parent, child), count in base.direct_calls.items():
        calls.setdefault(parent, {})[child] = count

    m = phase.phase_width
    window_size = 2**m
    block_gates = 2 * ceil(log2(model.block_scale))
    walk_reflection = _reflection_leaves(_WALK_IDENTITY, 2, 3, n)
    controlled_reflection = _reflection_leaves(_WALK_IDENTITY, 2, 2, n)
    kaiser = {
        _ARBITRARY: 4 * window_size * m + 2 * ceil(log2(floor(window_size))) + 2,
        "And": 4 * window_size - 4,
        "And†": 4 * window_size - 4,
        "Toffoli": 2 * m - 4,
    }
    qft_ands = (m - 1) * floor(m / 2)
    qft = {
        _ARBITRARY: qft_ands + m + floor(m / 2),
        "And": qft_ands,
        "And†": qft_ands,
    }
    calls.update(
        {
            "BlackBoxPrepare": {_ARBITRARY: guide_gates},
            "KaiserWindowState": kaiser,
            "QFTWithReverse†": qft,
            "Walk": {
                **walk_reflection,
                _ARBITRARY: block_gates + walk_reflection[_ARBITRARY],
                "O_F": 2,
                "O_H": 2,
            },
            "ControlledWalk": {
                **controlled_reflection,
                _CONTROLLED_ARBITRARY: block_gates,
                "ControlledO_F": 2,
                "ControlledO_H": 2,
            },
            "ZeroControlledReflection": _reflection_leaves(_WALK_IDENTITY, 2, 4, n),
        }
    )
    if amplify:
        calls.update(
            {
                "BlackBoxPrepare†": calls["BlackBoxPrepare"],
                "KaiserWindowState†": kaiser,
                "QFTWithReverse": qft,
                "Walk†": {**calls["Walk"], "O_F†": calls["Walk"]["O_F"]},
                "ControlledWalk†": {
                    **calls["ControlledWalk"],
                    "ControlledO_F†": calls["ControlledWalk"]["ControlledO_F"],
                },
                "ReflectGuideAncilla(-1)": _reflection_leaves(
                    "PrepareIdentity[guide]", 1, 3, model.guide_ancilla_width - 1
                ),
                "ReflectAll(-1)": _reflection_leaves(
                    "PrepareIdentity[all]",
                    1,
                    3,
                    ssum(register.bitsize for register in model.signature) - 1,
                ),
            }
        )
        del calls["Walk†"]["O_F"]
        del calls["ControlledWalk†"]["ControlledO_F"]
    return symbolic_call_graph(base.root, calls)


def tutorial_query_upper_bound(
    model: GuidedHamiltonianModel, exact: Mapping[str, SymbolicInt]
) -> dict[str, Expr]:
    """Replace 2**ceil(log2(T)) by the tutorial bound 2*T for symbolic T."""
    phase = model.phase_estimation
    scale = pi * phase.window_alpha / phase.precision
    # Mark the phase width as an integer so width expressions simplify before
    # replacing the power-of-two query budget.
    width = Dummy("phase_width", positive=True, integer=True)
    return {
        leaf: simplify(
            sympify(count)
            .replace(phase.phase_width, width)
            .replace(2**width, 2 * scale)
            .replace(width, phase.phase_width)
        )
        for leaf, count in exact.items()
    }
