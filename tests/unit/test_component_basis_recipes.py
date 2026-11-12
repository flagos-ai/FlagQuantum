"""Canonical vector recipes preserve exact basis tensors without a side table."""

from dataclasses import replace

import pytest
import torch

from flagquantum.compiler.registers.component_basis_vectors import (
    integer_basis_function,
    integer_basis_function_data,
)
from flagquantum.core.ir.program import BlockRecord, FunctionRecord, Operation, ValueRef
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.register_basis_vectors import (
    register_basis_vector_nodes,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,value", [(w, v) for w in range(1, 6) for v in range(1 << w)]
)
@pytest.mark.parametrize("state", [False, True])
def test_recipe_matches_locked_integer_vector_tensor(
    width: int, value: int, state: bool
) -> None:
    from qualtran.bloqs.basic_gates import IntEffect, IntState

    function = integer_basis_function(
        "arbitrary_name", RegisterType("uint", width), value, state=state
    )
    assert integer_basis_function_data(function) == (value, state)
    assert integer_basis_function_data(replace(function, name="renamed")) == (
        value,
        state,
    )
    assert function.registers is not None
    signature = Signature(function.registers)
    connections = {"val": (tuple(range(width)),)}
    like = torch.empty((), dtype=torch.complex128)
    nodes = register_basis_vector_nodes(
        signature, connections, {"val": value}, like, state=state
    )
    actual = contract_register_tensors(
        nodes, signature, {} if state else connections, connections if state else {}
    )
    original = IntState(value, width) if state else IntEffect(value, width)
    torch.testing.assert_close(
        actual,
        torch.as_tensor(original.tensor_contract(), dtype=like.dtype),
        atol=0,
        rtol=0,
    )
    assert len(nodes) == width and all(node.tensor.numel() == 2 for node in nodes)


@pytest.mark.parametrize(
    "defect",
    [
        "untyped",
        "ports",
        "name",
        "shape",
        "through",
        "width",
        "encoding",
        "blocks",
        "no_split",
        "two_splits",
        "split_width",
        "x_arity",
        "x_operand",
        "dirty",
        "phase",
        "permutation",
        "return",
        "extra_x",
    ],
)
def test_noncanonical_body_is_never_classified_by_a_vector_name(defect: str) -> None:
    function = integer_basis_function(
        "__fq_vector_0_state_0", RegisterType("uint", 2), 2
    )
    assert function.registers is not None
    register = function.registers[0]
    block = function.blocks[0]
    operations = list(block.operations)
    if defect == "untyped":
        function = replace(function, registers=None)
    elif defect == "ports":
        function = replace(function, registers=())
    elif defect in {"name", "shape", "through", "width", "encoding"}:
        register = (
            Register("q", register.dtype, side=register.side)
            if defect == "name"
            else register
        )
        register = (
            Register("val", register.dtype, (2,), register.side)
            if defect == "shape"
            else register
        )
        register = (
            Register("val", register.dtype, side=RegisterSide.THRU)
            if defect == "through"
            else register
        )
        register = (
            Register("val", RegisterType("uint", 0), side=register.side)
            if defect == "width"
            else register
        )
        if defect == "encoding":
            register = Register("val", RegisterType("signed", 2), side=register.side)
        function = replace(function, registers=(register,))
    elif defect == "blocks":
        function = replace(function, blocks=(block, BlockRecord("other")))
    else:
        split = operations[1]
        if defect == "no_split":
            operations.pop(1)
        elif defect == "two_splits":
            operations.append(split)
        elif defect == "split_width":
            operations[1] = replace(split, results=split.results[:1])
        elif defect == "x_arity":
            operations[2] = replace(operations[2], operands=())
        elif defect == "x_operand":
            operations[2] = replace(
                operations[2], operands=(ValueRef("absent", 999, "qubit"),)
            )
        elif defect == "dirty":
            operations[0] = replace(operations[0], workspace_dirty=True)
        elif defect == "phase":
            operations.insert(
                3, Operation("z", operations[2].results, operations[2].results)
            )
        elif defect == "permutation":
            operations[-1] = replace(
                operations[-1], operands=operations[-1].operands[::-1]
            )
        elif defect == "return":
            block = replace(block, return_values=())
        else:
            operations.insert(3, operations[2])
        function = replace(
            function, blocks=(replace(block, operations=tuple(operations)),)
        )
    assert integer_basis_function_data(function) is None


def test_unrelated_atomic_function_does_not_become_a_basis_recipe() -> None:
    assert (
        integer_basis_function_data(
            FunctionRecord("state_1", (BlockRecord("entry"),), registers=())
        )
        is None
    )
