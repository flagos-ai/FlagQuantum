# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/tensor_report_card.py.

"""Measure selected graph expansion, tensor construction and contraction width."""

import time
from collections.abc import Callable, Sequence
from multiprocessing.connection import Connection
from typing import TypeVar

from flagquantum.simulation.tensor_network.contraction_width import (
    tensor_contraction_width,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode

Component = TypeVar("Component")
Flat = TypeVar("Flat")


def report_on_tensors(
    name: str,
    cls_name: str,
    component: Component,
    cxn: Connection,
    *,
    flatten: Callable[[Component], Flat],
    build_nodes: Callable[[Flat], Sequence[TensorNetworkNode]],
) -> None:
    """Send the source-equivalent timing/width record over the task pipe.

    Compiler-owned flattening and Simulation-owned tensor construction are
    supplied explicitly. Width uses the existing pinned Cotengra shape planner.
    Durations measure separate operations, not primitive gate counts or model
    costs. A failure retains completed fields and sends its original message.
    The sending endpoint remains owned by the executor or caller.
    """
    record: dict[str, object] = {"name": name, "cls": cls_name}
    try:
        start = time.perf_counter()
        flat = flatten(component)
        record["flat_dur"] = time.perf_counter() - start
        start = time.perf_counter()
        nodes = build_nodes(flat)
        record["tn_dur"] = time.perf_counter() - start
        start = time.perf_counter()
        record["width"] = tensor_contraction_width(nodes)
        record["width_dur"] = time.perf_counter() - start
    except Exception as error:
        record["err"] = str(error)
    cxn.send(record)
