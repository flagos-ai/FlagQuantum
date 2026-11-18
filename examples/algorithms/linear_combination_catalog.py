"""Bind the original four-term signed block encoding for the tutorial."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from math import pi
from typing import Literal

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.linear_combination_block_encoding import (
    LinearCombinationBlockEncoding,
)
from flagquantum.algorithms.primitives.unitary_block_encoding import (
    UnitaryBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_workspace,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit

Gate = Literal["t", "h", "x", "z"]
_GATES: tuple[Gate, ...] = ("t", "h", "x", "z")


def _unitary(label: Gate) -> UnitaryBlockEncoding:
    def emit(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        return (Instruction("tdg" if inverse and label == "t" else label, ports["q"]),)

    return UnitaryBlockEncoding((("q", 1),), emit)


def _controlled(
    label: Gate,
) -> Callable[
    [int, Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
    FieldOperation,
]:
    def bind(
        predicate: int,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> FieldOperation:
        wires = (predicate, *ports["system"])
        gate = (
            Instruction(
                "cphase", wires, params={"theta": -pi / 4 if inverse else pi / 4}
            )
            if label == "t"
            else Instruction("c" + label, wires)
        )
        return FieldOperation(max(wires) + 1, (gate,))

    return bind


def linear_combination_example() -> LinearCombinationBlockEncoding:
    """Construct (T - H + X - Z)/4 at the original coefficient precision."""
    return LinearCombinationBlockEncoding.of_terms(
        (0.25, _unitary("t")),
        (-0.25, _unitary("h")),
        (0.25, _unitary("x")),
        (-0.25, _unitary("z")),
        coefficient_bits=1,
    )


def linear_combination_example_circuit(
    *,
    adjoint: bool = False,
) -> DynamicCircuit:
    """Append the original block on system, ancilla, coefficient junk and work."""
    encoding = linear_combination_example()
    prep_work, prep_slots = encoding.coefficient_preparation.workspace_requirements
    traversal = indexed_component_workspace((encoding.select.selection_bitsize,), 0)
    ancilla_end = 1 + encoding.ancilla_bitsize
    through = ancilla_end + encoding.resource_bitsize
    work = tuple(range(through, through + max(prep_work, traversal)))
    slots = tuple(range(max(prep_slots, traversal)))
    circuit = DynamicCircuit(through + len(work), dtype=torch.complex128)
    circuit.append(
        encoding.make_on(
            tuple(_controlled(label) for label in _GATES),
            system=(0,),
            ancilla=tuple(range(1, ancilla_end)),
            resource=tuple(range(ancilla_end, through)),
            prepare_workspace=work[:prep_work],
            prepare_classical_bits=slots[:prep_slots],
            traversal_workspace=work[:traversal],
            traversal_classical_bits=slots[:traversal],
            workspaces=((),) * len(_GATES),
            classical_bits=((),) * len(_GATES),
            adjoint=adjoint,
        )
    )
    return circuit


def linear_combination_example_model() -> ComponentCost:
    """Price the source controlled T, H, X and Z independently of emission."""
    encoding = linear_combination_example()
    prepare = encoding.coefficient_preparation.resource_model()
    children = (
        ComponentCost("Controlled T", 2, GateCost(rotation=1)),
        ComponentCost("Controlled Hadamard", 2, GateCost(clifford=4, rotation=2)),
        ComponentCost("CNOT", 2, GateCost(clifford=1)),
        ComponentCost("CZ", 2, GateCost(clifford=1)),
    )
    return encoding.resource_model(
        encoding.select.resource_model(children),
        prepare,
        replace(prepare, name="Coefficient PREPARE adjoint"),
    )
