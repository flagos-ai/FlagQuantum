# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's sparse chemistry notebook and test fixture.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/sparse

"""Build the original sparse chemistry examples and their declared call graphs."""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from flagquantum.algorithms.primitives.multiplexers.select_sparse import SelectSparse
from flagquantum.algorithms.primitives.sparse_chemistry_prepare import (
    PrepareSparseChemistry,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)


def sparse_integral_fixture(
    spatial: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reproduce the tutorial's seeded symmetric integrals."""
    rng = np.random.RandomState(7)
    one = rng.normal(size=(spatial, spatial))
    one = 0.5 * (one + one.T)
    two = rng.normal(scale=4, size=(spatial,) * 4)
    two += two.transpose(0, 1, 3, 2)
    two += two.transpose(1, 0, 2, 3)
    two += two.transpose(2, 3, 0, 1)
    return one, two


def sparse_prepare_graph(prepare: PrepareSparseChemistry) -> ComponentCost:
    """Expand the model's direct Clifford and swap categories for the tutorial."""
    model = prepare.resource_model()
    uniform, qroam, comparator = model.calls
    precision = prepare.data.probability_bits
    width = (prepare.data.spatial_orbitals - 1).bit_length()
    one_swap = (
        ()
        if width == 1
        else (CostCall(ComponentCost("CSwap(1)", 3, GateCost(cswap=1))),)
    )
    return replace(
        model,
        gates=GateCost(),
        calls=(
            uniform,
            qroam,
            CostCall(
                ComponentCost(
                    f"OnEach(H, {precision})", precision, GateCost(clifford=precision)
                )
            ),
            CostCall(ComponentCost("Hadamard", 1, GateCost(clifford=1)), 3),
            CostCall(ComponentCost("Controlled Z (0)", 2, GateCost(clifford=3))),
            CostCall(ComponentCost("Controlled Z (1)", 2, GateCost(clifford=1))),
            *one_swap,
            CostCall(
                ComponentCost(f"CSwap({width})", 1 + 2 * width, GateCost(cswap=width)),
                8,
            ),
            comparator,
        ),
    )


def sparse_select_graph(spin_orbitals: int) -> ComponentCost:
    """Display the S phase and distinguish controlled and free Majorana pairs."""
    model = SelectSparse.source_resource_model(spin_orbitals)
    phases = CostCall(ComponentCost("SGate", 1, GateCost(clifford=1)))
    majoranas = tuple(
        replace(
            call,
            component=replace(
                call.component,
                name=call.component.name + (" controlled" if at < 2 else ""),
            ),
        )
        for at, call in enumerate(model.calls)
    )
    return replace(model, gates=GateCost(), calls=(phases, *majoranas))
