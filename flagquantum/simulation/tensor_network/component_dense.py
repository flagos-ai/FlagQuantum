# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Numerical composition adapted from Qualtran v0.7.0 bloq_to_dense.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_dense.py

"""Contract explicitly selected component graphs through native numerical owners."""

from collections.abc import Callable, Iterable, Mapping

import torch

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.signatures import Signature
from .component_superoperator_nodes import component_superoperator_nodes
from .component_tensor_nodes import component_tensor_nodes
from .models import TensorNetworkNode
from .register_superoperator_binding import contract_register_superoperator
from .register_tensor_binding import contract_register_tensors


def contract_component_tensors(
    record: ProgramRecord,
    order: tuple[int, ...],
    supply: Callable[
        [
            FunctionRecord,
            Mapping[str, tuple[tuple[int, ...], ...]],
            Mapping[str, tuple[tuple[int, ...], ...]],
        ],
        Iterable[TensorNetworkNode | int],
    ],
    like: torch.Tensor,
    entry: str = "main",
) -> torch.Tensor:
    """Wire and contract a selected pure-state graph to a scalar/vector/matrix.

    Compiler already owns root wrapping, full/partial selection and call order.
    Reuse component_tensor_nodes for validation, SSA wiring and idle identities,
    then contract_register_tensors for output/input grouping and contraction.
    Supplier exceptions, dtype/device behavior and autograd are unchanged.
    Classical ports retain the pure graph's explicit refusal. An empty network
    raises ValueError; a scalar result requires an explicit scalar tensor node.
    Integer trace requests fail at supply with guidance to the mixed entry;
    they never reach contraction or become a projection.
    No decomposition, provider discovery, simulator fallback or normalization
    occurs. Dense output storage is exponential in the declared live bit axes.
    """
    nodes, incoming, outgoing = component_tensor_nodes(
        record, order, supply, like, entry
    )
    function = next(function for function in record.functions if function.name == entry)
    assert function.registers is not None
    return contract_register_tensors(
        nodes, Signature(function.registers), incoming, outgoing
    )


def contract_component_superoperator(
    record: ProgramRecord,
    order: tuple[int, ...],
    supply: Callable[
        [
            FunctionRecord,
            Mapping[str, tuple[tuple[int, ...], ...]],
            Mapping[str, tuple[tuple[int, ...], ...]],
            Callable[[], int],
        ],
        Iterable[TensorNetworkNode | int],
    ],
    like: torch.Tensor,
    entry: str = "main",
) -> torch.Tensor:
    """Wire, conjugate-double, trace and contract a selected mixed graph.

    The supplier retains component_superoperator_nodes' fresh-label and explicit
    discard protocol. Reuse its validation and graph construction, followed by
    contract_register_superoperator's right-forward/backward then left-forward/
    backward grouping. Return a scalar, density/effect matrix or full channel.
    Empty networks are refused; scalar tensors and source-unsupported direct
    input trace effects keep their existing, explicitly documented semantics.
    Supplier and trace/contraction errors propagate without retry or fallback.
    No Compiler traversal, backend selection, physical reset or resource model
    occurs. Dense n-input/n-output-bit channels occupy 16**n elements.
    """
    nodes, incoming, outgoing = component_superoperator_nodes(
        record, order, supply, like, entry
    )
    function = next(function for function in record.functions if function.name == entry)
    assert function.registers is not None
    return contract_register_superoperator(
        nodes, Signature(function.registers), incoming, outgoing
    )
