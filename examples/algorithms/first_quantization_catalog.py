# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's first_quantization.ipynb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/first_quantization.ipynb

"""Concrete signatures and call graphs from the first-quantization tutorial."""

from dataclasses import replace

from flagquantum.algorithms.first_quantization import first_quantization_signature
from flagquantum.algorithms.first_quantization_kinetic import prepare_t_signature
from flagquantum.algorithms.first_quantization_potential import (
    prepare_uv_signature,
    prepare_zeta_signature,
)
from flagquantum.algorithms.first_quantization_resource_model import (
    source_prepare_first_quantization_cost,
    source_prepare_t_cost,
    source_prepare_uv_cost,
    source_prepare_zeta_cost,
    source_select_first_quantization_cost,
    source_select_t_cost,
    source_select_uv_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


def _word(name: str, width: int, shape: tuple[int, ...] = ()) -> Register:
    return Register(name, opaque_register_type(width), shape=shape)


def _bit(name: str) -> Register:
    return Register(name, quantum_bit_type())


def _toffoli_graph(model: ComponentCost) -> ComponentCost:
    """Draw a declared Toffoli leaf as an explicit child of its original model."""
    child = ComponentCost("Toffoli", 3, GateCost(toffoli=1))
    return replace(
        model,
        gates=GateCost(),
        calls=(*model.calls, CostCall(child, model.gates.toffoli)),
    )


def _select_graph() -> ComponentCost:
    model = source_select_first_quantization_cost(6, 10, 10)
    # The source's bookkeeping allocates six momentum words and one nuclear word.
    return replace(
        model,
        calls=(
            CostCall(ComponentCost("Allocate", 6), 6),
            CostCall(ComponentCost("Allocate", 16)),
            *model.calls,
            CostCall(ComponentCost("Free", 6), 6),
            CostCall(ComponentCost("Free", 16)),
        ),
    )


def notebook_cases() -> dict[str, tuple[Signature, ComponentCost]]:
    """Return the complete seven concrete examples in their original order."""
    return {
        "prepare": (
            first_quantization_signature(6, 10, 10, 256, prepare=True),
            source_prepare_first_quantization_cost(6, 10, 10, 10),
        ),
        "select": (
            first_quantization_signature(6, 10, 10, 256, prepare=False),
            _select_graph(),
        ),
        "prepare_t": (
            prepare_t_signature(5),
            source_prepare_t_cost(5),
        ),
        "prepare_uv": (
            prepare_uv_signature(5, 10, 256),
            source_prepare_uv_cost(5, 10, 256, 10),
        ),
        "select_t": (
            Signature(
                (
                    _bit("flag_T"),
                    _bit("plus"),
                    _word("w", 3),
                    _word("r", 5),
                    _word("s", 5),
                    _word("p", 5, (3,)),
                )
            ),
            _toffoli_graph(source_select_t_cost(5)),
        ),
        "select_uv": (
            Signature(
                (
                    _bit("flag_tuv"),
                    _bit("flag_uv"),
                    _word("l", 4),
                    _word("rl", 16),
                    _word("nu", 6, (3,)),
                    _word("p", 5, (3,)),
                    _word("q", 5, (3,)),
                )
            ),
            _toffoli_graph(source_select_uv_cost(5, 10, 16)),
        ),
        "zeta": (
            prepare_zeta_signature(10),
            _toffoli_graph(source_prepare_zeta_cost(10, 10)),
        ),
    }
