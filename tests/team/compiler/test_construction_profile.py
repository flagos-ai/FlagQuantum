"""Coherent arithmetic contracts expressed in the shared structured records."""

from dataclasses import replace
from typing import Literal

import pytest

from flagquantum.compiler._construction.bindings import verify_bindings
from flagquantum.compiler._construction.profile import uint_type, verify_construction
from flagquantum.compiler._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
)
from flagquantum.compiler._hybrid.verifier import (
    HybridVerificationError,
    verify_program,
)

pytestmark = pytest.mark.unit


def value(scope: str, index: int, width: int) -> Value:
    return Value(ValueId(scope, index), uint_type(width))


def function(
    name: str, arguments: tuple[Value, ...], operations: tuple[Operation, ...]
) -> HybridProgram:
    return HybridProgram(name, Region((Block(arguments, operations),)))


@pytest.fixture
def polynomial() -> tuple[HybridProgram, HybridProgram]:
    x = value("square", 0, 2)
    squared = value("square", 1, 4)
    square = function(
        "square",
        (x,),
        (
            Operation("qint.multiply", (x, x), (squared,)),
            Operation("qint.return", (squared,)),
        ),
    )
    x, y = value("main", 0, 2), value("main", 1, 2)
    temporary, result = value("main", 2, 4), value("main", 3, 4)
    main = function(
        "main",
        (x, y),
        (
            Operation("qint.call", (x,), (temporary,), {"callee": "square"}),
            Operation(
                "qint.add",
                (temporary, y),
                (result,),
                location=SourceLocation("user.py", 8, 4),
            ),
            Operation("qint.return", (result,)),
        ),
    )
    return square, main


def test_composable_square_keeps_resource_identity_and_bound(
    polynomial: tuple[HybridProgram, HybridProgram],
) -> None:
    square, main = polynomial
    assert verify_construction((square, main), entry="main") is main
    operation = square.body.blocks[0].operations[0]
    assert operation.operands[0].id == operation.operands[1].id
    # Existing classical hybrid verification must not silently reinterpret this profile.
    with pytest.raises(HybridVerificationError, match="operation.unknown"):
        verify_program(main)


def test_classical_constant_is_not_a_quantum_register() -> None:
    x, result = value("main", 0, 2), value("main", 2, 3)
    one = Value(ValueId("main", 1), INDEX)
    main = function(
        "main",
        (x,),
        (
            Operation("arith.constant", results=(one,), attributes={"value": 1}),
            Operation("qint.add", (x, one), (result,)),
            Operation("qint.return", (result,)),
        ),
    )
    assert verify_construction((main,), entry="main") is main


@pytest.mark.parametrize("constant", [True, -1, 1.5, "1"])
def test_classical_values_are_checked_before_conversion(constant: object) -> None:
    x = value("main", 0, 2)
    number = Value(ValueId("main", 1), INDEX)
    main = function(
        "main",
        (x,),
        (
            Operation(
                "arith.constant", results=(number,), attributes={"value": constant}
            ),
            Operation("qint.return", (x,)),
        ),
    )
    with pytest.raises(HybridVerificationError, match="construction.constant"):
        verify_construction((main,), entry="main")


@pytest.mark.parametrize("width", [True, 0, -1, 2.0])
def test_quantum_width_is_not_coerced(width: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        uint_type(width)


def test_overflow_reports_the_original_source(
    polynomial: tuple[HybridProgram, HybridProgram],
) -> None:
    square, main = polynomial
    block = main.body.blocks[0]
    narrow = value("main", 3, 3)
    add = replace(block.operations[1], results=(narrow,))
    broken = function(
        "main",
        block.arguments,
        (block.operations[0], add, Operation("qint.return", (narrow,))),
    )
    with pytest.raises(HybridVerificationError, match=r"width.overflow: user.py:8:4"):
        verify_construction((square, broken), entry="main")


def test_source_location_is_diagnostic_not_structure(
    polynomial: tuple[HybridProgram, HybridProgram],
) -> None:
    _, main = polynomial
    block = main.body.blocks[0]
    operations = tuple(
        replace(op, location=SourceLocation("moved.py", 100)) for op in block.operations
    )
    relocated = function("main", block.arguments, operations)
    assert main.semantic_identity == relocated.semantic_identity


@pytest.mark.parametrize(
    "opcode", ["arith.add", "quantum.reset", "qint.free", "qint.drop"]
)
def test_unimplemented_effects_are_rejected(opcode: str) -> None:
    x, y = value("main", 0, 2), value("main", 1, 3)
    main = function(
        "main", (x,), (Operation(opcode, (x,), (y,)), Operation("qint.return", (y,)))
    )
    with pytest.raises(HybridVerificationError, match="operation.unsupported"):
        verify_construction((main,), entry="main")


def test_recursive_function_is_not_expanded() -> None:
    x, result = value("recur", 0, 2), value("recur", 1, 2)
    main = function(
        "recur",
        (x,),
        (
            Operation("qint.call", (x,), (result,), {"callee": "recur"}),
            Operation("qint.return", (result,)),
        ),
    )
    with pytest.raises(HybridVerificationError, match="function.recursion"):
        verify_construction((main,), entry="recur")


def test_undefined_values_and_duplicate_definitions_fail() -> None:
    x, unknown, result = value("main", 0, 2), value("main", 1, 2), value("main", 2, 3)
    main = function(
        "main",
        (x,),
        (
            Operation("qint.add", (x, unknown), (result,)),
            Operation("qint.return", (result,)),
        ),
    )
    with pytest.raises(HybridVerificationError, match="ssa.use"):
        verify_construction((main,), entry="main")
    duplicate = function(
        "main",
        (x,),
        (Operation("qint.add", (x, x), (x,)), Operation("qint.return", (x,))),
    )
    with pytest.raises(HybridVerificationError, match="ssa.duplicate"):
        verify_construction((duplicate,), entry="main")


def test_budgets_bound_width_and_expansion(
    polynomial: tuple[HybridProgram, HybridProgram],
) -> None:
    with pytest.raises(HybridVerificationError, match="budget.width"):
        verify_construction(polynomial, entry="main", max_width=3)
    with pytest.raises(HybridVerificationError, match="budget.nodes"):
        verify_construction(polynomial, entry="main", max_nodes=2)
    with pytest.raises(HybridVerificationError, match="budget"):
        verify_construction(polynomial, entry="main", max_call_depth=1)


def test_duplicate_function_and_bad_call_signature(
    polynomial: tuple[HybridProgram, HybridProgram],
) -> None:
    square, main = polynomial
    with pytest.raises(HybridVerificationError, match="function.duplicate"):
        verify_construction((square, square), entry="square")
    block = main.body.blocks[0]
    call = replace(block.operations[0], operands=block.arguments)
    broken = function("main", block.arguments, (call, *block.operations[1:]))
    with pytest.raises(HybridVerificationError, match="function.signature"):
        verify_construction((square, broken), entry="main")


@pytest.mark.parametrize("state", ["fresh_zero", "caller_asserted_zero"])
def test_external_zero_is_a_precondition_not_state_inspection(
    polynomial: tuple[HybridProgram, HybridProgram],
    state: Literal["fresh_zero", "caller_asserted_zero"],
) -> None:
    main = verify_construction(polynomial, entry="main")
    x, y = main.body.blocks[0].arguments
    verify_bindings(
        polynomial,
        {x.id: (0, 1), y.id: (2, 3)},
        (4, 5, 6, 7),
        entry="main",
        n_wires=9,
        workspace=(8,),
        workspace_state=state,
    )
    with pytest.raises(HybridVerificationError, match="workspace.precondition"):
        verify_bindings(
            polynomial,
            {x.id: (0, 1), y.id: (2, 3)},
            (4, 5, 6, 7),
            entry="main",
            n_wires=9,
            workspace=(8,),
        )


@pytest.mark.parametrize("other", [(1, 2), (0, 1), (2, 2)])
def test_input_ports_cannot_alias(
    polynomial: tuple[HybridProgram, HybridProgram], other: tuple[int, int]
) -> None:
    main = verify_construction(polynomial, entry="main")
    x, y = main.body.blocks[0].arguments
    with pytest.raises(HybridVerificationError, match="binding.alias"):
        verify_bindings(
            polynomial,
            {x.id: (0, 1), y.id: other},
            (4, 5, 6, 7),
            entry="main",
            n_wires=8,
        )


def test_output_and_workspace_cannot_overlap_input(
    polynomial: tuple[HybridProgram, HybridProgram],
) -> None:
    main = verify_construction(polynomial, entry="main")
    x, y = main.body.blocks[0].arguments
    inputs = {x.id: (0, 1), y.id: (2, 3)}
    with pytest.raises(HybridVerificationError, match="binding.alias"):
        verify_bindings(polynomial, inputs, (0, 5, 6, 7), entry="main", n_wires=9)
    with pytest.raises(HybridVerificationError, match="binding.alias"):
        verify_bindings(
            polynomial,
            inputs,
            (4, 5, 6, 7),
            entry="main",
            n_wires=9,
            workspace=(7,),
            workspace_state="fresh_zero",
        )


def test_binding_count_and_wire_validation(
    polynomial: tuple[HybridProgram, HybridProgram],
) -> None:
    main = verify_construction(polynomial, entry="main")
    x, y = main.body.blocks[0].arguments
    with pytest.raises(HybridVerificationError, match="binding.inputs"):
        verify_bindings(
            polynomial, {x.id: (0, 1)}, (4, 5, 6, 7), entry="main", n_wires=8
        )
    with pytest.raises(HybridVerificationError, match="binding.width"):
        verify_bindings(
            polynomial,
            {x.id: (0,), y.id: (2, 3)},
            (4, 5, 6, 7),
            entry="main",
            n_wires=8,
        )
    with pytest.raises(HybridVerificationError, match="binding.wire"):
        verify_bindings(
            polynomial,
            {x.id: (False, 1), y.id: (2, 3)},
            (4, 5, 6, 7),
            entry="main",
            n_wires=8,
        )


def test_repeated_calls_have_a_separate_expansion_budget() -> None:
    arg = value("identity", 0, 2)
    identity = function("identity", (arg,), (Operation("qint.return", (arg,)),))
    x = value("main", 0, 2)
    results = tuple(value("main", i, 2) for i in range(1, 9))
    calls = tuple(
        Operation("qint.call", (x,), (result,), {"callee": "identity"})
        for result in results
    )
    main = function("main", (x,), (*calls, Operation("qint.return", (results[-1],))))
    with pytest.raises(HybridVerificationError, match="budget.expansion"):
        verify_construction((identity, main), entry="main", max_nodes=15)


def test_missing_entry_and_unknown_callee(
    polynomial: tuple[HybridProgram, HybridProgram],
) -> None:
    _, main = polynomial
    with pytest.raises(HybridVerificationError, match="function.entry"):
        verify_construction(polynomial, entry="missing")
    with pytest.raises(HybridVerificationError, match="function.callee"):
        verify_construction((main,), entry="main")


def test_call_result_type_cannot_lie_about_callee(
    polynomial: tuple[HybridProgram, HybridProgram],
) -> None:
    square, main = polynomial
    block = main.body.blocks[0]
    bad = value("main", 2, 3)
    call = replace(block.operations[0], results=(bad,))
    broken = function("main", block.arguments, (call, Operation("qint.return", (bad,))))
    with pytest.raises(HybridVerificationError, match="function.signature"):
        verify_construction((square, broken), entry="main")


def test_classical_argument_is_not_a_quantum_borrow() -> None:
    number = Value(ValueId("main", 0), INDEX)
    main = function("main", (number,), (Operation("qint.return", (number,)),))
    with pytest.raises(HybridVerificationError, match="construction.type"):
        verify_construction((main,), entry="main")


def test_trainable_tensor_cannot_be_coerced_to_width() -> None:
    import torch

    parameter = torch.tensor(2.0, requires_grad=True)
    with pytest.raises(TypeError, match="Python integer"):
        uint_type(parameter)
    assert parameter.requires_grad


@pytest.mark.parametrize(
    "case",
    [
        "missing_return",
        "early_return",
        "missing_result",
        "binary_arity",
        "nested_region",
        "duplicate_argument",
        "multiple_blocks",
    ],
)
def test_malformed_function_structure_fails_closed(case: str) -> None:
    x, result = value("main", 0, 2), value("main", 1, 3)
    add = Operation("qint.add", (x, x), (result,))
    ret = Operation("qint.return", (result,))
    main = function("main", (x,), (add, ret))
    if case == "missing_return":
        main = function("main", (x,), (add,))
    elif case == "early_return":
        main = function("main", (x,), (Operation("qint.return", (x,)), add, ret))
    elif case == "missing_result":
        main = function("main", (x,), (replace(add, results=()), ret))
    elif case == "binary_arity":
        main = function("main", (x,), (replace(add, operands=(x,)), ret))
    elif case == "nested_region":
        main = function(
            "main", (x,), (replace(add, regions=(Region((Block(),)),)), ret)
        )
    elif case == "duplicate_argument":
        main = function("main", (x, x), (add, ret))
    else:
        main = HybridProgram("main", Region((Block((x,), (add, ret)), Block())))
    with pytest.raises(HybridVerificationError):
        verify_construction((main,), entry="main")


def test_raw_quantum_type_and_constant_obey_safety_width() -> None:
    from flagquantum.compiler._hybrid.model import IRType

    forged = Value(ValueId("main", 0), IRType("quantum.uint", (True,)))
    main = function("main", (forged,), (Operation("qint.return", (forged,)),))
    with pytest.raises(HybridVerificationError, match="construction.type"):
        verify_construction((main,), entry="main")
    x = value("main", 0, 2)
    number = Value(ValueId("main", 1), INDEX)
    main = function(
        "main",
        (x,),
        (
            Operation(
                "arith.constant", results=(number,), attributes={"value": 1 << 256}
            ),
            Operation("qint.return", (x,)),
        ),
    )
    with pytest.raises(HybridVerificationError, match="budget.width"):
        verify_construction((main,), entry="main")


def test_bindings_require_verified_function_and_matching_output(
    polynomial: tuple[HybridProgram, HybridProgram],
) -> None:
    _, main = polynomial
    x, y = main.body.blocks[0].arguments
    inputs = {x.id: (0, 1), y.id: (2, 3)}
    with pytest.raises(HybridVerificationError, match="binding.width"):
        verify_bindings(polynomial, inputs, (4, 5, 6), entry="main", n_wires=8)
    with pytest.raises(HybridVerificationError, match="function.return"):
        verify_bindings((function("empty", (), ()),), {}, (), entry="empty", n_wires=1)
    multi = HybridProgram("multi", Region((Block(), Block())))
    with pytest.raises(HybridVerificationError, match="function.blocks"):
        verify_bindings((multi,), {}, (), entry="multi", n_wires=1)
