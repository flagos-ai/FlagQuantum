# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 cirq_interop/jupyter_tools.py.

"""Optional Cirq diagram utilities and independent notebook execution."""

from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from traceback import extract_stack
from typing import Any

from ...core.ir.program.registers import Register
from ._decompose import decompose_cirq_once
from ._legacy_t_complexity import cirq_t_complexity
from ._register_export import _named_qubits, _register_shape
from .conversion import _cirq


def circuit_with_costs(circuit: Any) -> Any:
    """Annotate a Cirq circuit without changing its quantum operations."""
    cirq = _cirq()

    def annotate(operation: Any, _: int) -> Any:
        cost = cirq_t_complexity(operation)
        return operation.with_tags(f"t:{cost.t:g},r:{cost.rotations:g}")

    return cirq.map_operations(circuit, map_func=annotate)


def svg_circuit(
    circuit: Any, registers: Iterable[Register] = (), *, include_costs: bool = False
) -> Any:
    """Render a nonempty circuit in register order with optional legacy costs."""
    if not len(circuit):
        raise ValueError("Circuit is empty")
    cirq = _cirq()
    named = [
        qubit
        for register in registers
        for qubit in _named_qubits(cirq, register, _register_shape(register)).flat
    ]
    order = cirq.QubitOrder.explicit(named, fallback=cirq.QubitOrder.DEFAULT)
    if include_costs:
        circuit = circuit_with_costs(circuit)
    diagram = circuit.to_text_diagram_drawer(transpose=False, qubit_order=order)
    if not diagram.horizontal_lines:
        raise ValueError("No non-empty moments")
    svg = import_module("cirq.contrib.svg.svg").tdd_to_svg(diagram)
    return import_module("IPython.display").SVG(svg)


def display_gate_and_compilation(
    operation: Any,
    *,
    registers: Iterable[Register] = (),
    vertical: bool = False,
    include_costs: bool = True,
) -> None:
    """Display one operation and its Cirq decomposition in a widget pair."""
    widgets = import_module("ipywidgets")
    cirq = _cirq()
    leaves = decompose_cirq_once(operation)
    if leaves is None:
        raise ValueError("Cirq operation has no decomposition")
    order = tuple(registers)
    panels = []
    for circuit in (cirq.Circuit(operation), cirq.Circuit(leaves)):
        panel = widgets.Output()
        panel.append_display_data(
            svg_circuit(circuit, order, include_costs=include_costs)
        )
        panels.append(panel)
    layout = widgets.VBox if vertical else widgets.HBox
    import_module("IPython.display").display(layout(panels))


def execute_notebook(
    name: str | Path, *, timeout: int = 600, kernel_name: str = "python3"
) -> Any:
    """Execute a notebook from an explicit path or a caller-relative stem."""
    from ...testing.notebook_execution import execute_notebook_node

    path = Path(name)
    if not path.is_absolute():
        caller = Path(extract_stack(limit=2)[0].filename)
        path = caller.parent / path
    if not path.suffix:
        path = path.with_suffix(".ipynb")
    notebook = import_module("nbformat").read(path, as_version=4)
    executed, _ = execute_notebook_node(
        notebook, timeout=timeout, kernel_name=kernel_name
    )
    return executed
