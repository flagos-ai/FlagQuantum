# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's projectile.ipynb.

"""Concrete register signatures and graphs from the projectile tutorial."""

from dataclasses import replace

from flagquantum.algorithms.first_quantization_projectile_resource_model import (
    ProjectilePrepareDeclaration,
    ProjectileSelectDeclaration,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, CostCall
from flagquantum.core.ir.program.signatures import Signature


def notebook_cases() -> dict[str, tuple[Signature, ComponentCost]]:
    """Preserve both examples, including the SELECT bookkeeping vertices."""
    p, n, eta, atoms, charge = 6, 8, 10, 10, 10
    prepare = ProjectilePrepareDeclaration(p, n, eta, atoms, charge)
    select = ProjectileSelectDeclaration(p, n, eta, atoms, charge)
    model = select.resource_model()
    nuclear = select.num_bits_nuc_pos
    return {
        "prepare": (prepare.signature, prepare.resource_model()),
        "select": (
            select.signature,
            replace(
                model,
                calls=(
                    CostCall(ComponentCost("Allocate", n), 3),
                    CostCall(ComponentCost("Allocate", p), 3),
                    CostCall(ComponentCost("Allocate", nuclear)),
                    *model.calls,
                    CostCall(ComponentCost("Free", n), 3),
                    CostCall(ComponentCost("Free", p), 3),
                    CostCall(ComponentCost("Free", nuclear)),
                ),
            ),
        ),
    }
