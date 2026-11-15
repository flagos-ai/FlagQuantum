# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's projectile.ipynb.

"""Concrete register signatures and graphs from the projectile tutorial."""

from dataclasses import replace

from flagquantum.algorithms.first_quantization_projectile import (
    projectile_prepare_signature,
    projectile_select_signature,
)
from flagquantum.algorithms.first_quantization_projectile_resource_model import (
    source_projectile_prepare_first_quantization_cost,
    source_projectile_select_first_quantization_cost,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, CostCall
from flagquantum.core.ir.program.signatures import Signature


def notebook_cases() -> dict[str, tuple[Signature, ComponentCost]]:
    """Preserve both examples, including the SELECT bookkeeping vertices."""
    p, n, eta, atoms, charge = 6, 8, 10, 10, 10
    m, nuclear = 256, 16
    select = source_projectile_select_first_quantization_cost(p, n, eta, atoms)
    return {
        "prepare": (
            projectile_prepare_signature(n, eta, atoms, m),
            source_projectile_prepare_first_quantization_cost(p, n, eta, atoms, charge),
        ),
        "select": (
            projectile_select_signature(p, n, eta, atoms, m),
            replace(
                select,
                calls=(
                    CostCall(ComponentCost("Allocate", n), 3),
                    CostCall(ComponentCost("Allocate", p), 3),
                    CostCall(ComponentCost("Allocate", nuclear)),
                    *select.calls,
                    CostCall(ComponentCost("Free", n), 3),
                    CostCall(ComponentCost("Free", p), 3),
                    CostCall(ComponentCost("Free", nuclear)),
                ),
            ),
        ),
    }
