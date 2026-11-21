"""Report an explicitly selected native CNOT catalog using existing owners."""

from functools import partial
from multiprocessing.connection import Connection

import torch

from examples.algorithms.factorized_controlled_not import (
    controlled_not_chain,
    factorized_cnot_supply,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes_with_labels,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from tools.development.tensor_report import report_on_tensors


def flatten_calls(record: ProgramRecord) -> tuple[ProgramRecord, tuple[int, ...]]:
    """Select already primitive calls through the Compiler graph owner."""
    order = tuple(index for index, _, _ in iter_component_calls(record, "cnot_chain"))
    return record, order


def build_nodes(
    flat: tuple[ProgramRecord, tuple[int, ...]],
) -> tuple[TensorNetworkNode, ...]:
    """Build factorized COPY/XOR leaves through the Simulation owner."""
    record, order = flat
    like = torch.empty((), dtype=torch.complex128)
    return component_tensor_nodes_with_labels(
        record,
        order,
        partial(factorized_cnot_supply, like=like),
        like,
        "cnot_chain",
    )[0]


def report_component(
    name: str, cls_name: str, component: ProgramRecord, cxn: Connection
) -> None:
    """Keep the process task importable under spawn, with explicit providers."""
    report_on_tensors(
        name,
        cls_name,
        component,
        cxn,
        flatten=flatten_calls,
        build_nodes=build_nodes,
    )


def selected_programs() -> tuple[tuple[str, ProgramRecord], ...]:
    """Return three concrete graphs, not a whole-library performance survey."""
    return tuple(
        (f"cnot_chain_{repetitions}", controlled_not_chain(repetitions))
        for repetitions in (1, 2, 3)
    )
