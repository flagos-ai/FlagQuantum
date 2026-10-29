"""Exact polynomial permutations and local scratch obligations."""

import pytest

from flagquantum.compiler._arithmetic_kernels import multiply_accumulate
from flagquantum.compiler._construction.lowering import lower_arithmetic
from flagquantum.compiler._construction.profile import uint_type
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
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.unit


def _value(scope: str, index: int, width: int) -> Value:
    return Value(ValueId(scope, index), uint_type(width))


def _function(
    name: str, args: tuple[Value, ...], ops: tuple[Operation, ...]
) -> HybridProgram:
    return HybridProgram(name, Region((Block(args, ops),)))


def _image(program: CircuitIR, basis: int) -> int:
    for instruction in program.instructions:
        assert instruction.name in {"x", "cx", "ccx"}
        controls, target = instruction.wires[:-1], instruction.wires[-1]
        if all(basis & (1 << (program.n_wires - 1 - q)) for q in controls):
            basis ^= 1 << (program.n_wires - 1 - target)
    return basis


@pytest.mark.parametrize("a,b", [(1, 1), (1, 3), (3, 1), (2, 2), (3, 2), (4, 1)])
def test_kernel_adds_partial_products_for_all_nonzero_accumulators(
    a: int, b: int
) -> None:
    width = a + b
    begin = a + b
    program = CircuitIR(
        n_wires=begin + 2 * width,
        instructions=tuple(
            multiply_accumulate(
                tuple(range(a)),
                tuple(range(a, begin)),
                tuple(range(begin, begin + width)),
                tuple(range(begin + width, begin + 2 * width)),
            )
        ),
    )
    for x in range(1 << a):
        for y in range(1 << b):
            for accumulator in range(1 << width):
                initial = (((x << b) | y) << width | accumulator) << width
                expected = (
                    ((x << b) | y) << width | ((accumulator + x * y) % (1 << width))
                ) << width
                assert _image(program, initial) == expected


@pytest.mark.parametrize("width", [1, 2, 3])
def test_square_collapses_duplicate_controls_without_cloning(width: int) -> None:
    product = 2 * width
    program = CircuitIR(
        n_wires=width + 2 * product,
        instructions=tuple(
            multiply_accumulate(
                tuple(range(width)),
                tuple(range(width)),
                tuple(range(width, width + product)),
                tuple(range(width + product, width + 2 * product)),
            )
        ),
    )
    assert all(len(set(g.wires)) == len(g.wires) for g in program.instructions)
    for x in range(1 << width):
        for target in range(1 << product):
            initial = ((x << product) | target) << product
            expected = ((x << product) | ((target + x * x) % (1 << product))) << product
            assert _image(program, initial) == expected


@pytest.mark.parametrize(
    "a,b,polynomial",
    [
        (1, 1, False),
        (1, 3, False),
        (3, 1, False),
        (3, 2, False),
        (1, 1, True),
        (2, 2, True),
        (1, 3, True),
        (3, 1, True),
    ],
)
def test_full_expression_xor_and_cleanup_for_all_basis_values(
    a: int,
    b: int,
    polynomial: bool,
) -> None:
    x, y = _value("main", 0, a), _value("main", 1, b)
    bound = ((1 << a) - 1) ** 2 if polynomial else ((1 << a) - 1) * ((1 << b) - 1)
    width = max(1, bound.bit_length())
    product = _value("main", 2, width)
    ops = [Operation("qint.multiply", (x, x if polynomial else y), (product,))]
    result = product
    work = 2 * width
    if polynomial:
        width = max(1, (bound + (1 << b) - 1).bit_length())
        result = _value("main", 3, width)
        ops.append(Operation("qint.add", (product, y), (result,)))
        work = max(work, 2 * max(product.type.parameters[0], b) + 2)
    main = _function("main", (x, y), (*ops, Operation("qint.return", (result,))))
    begin = a + b + width
    program = lower_arithmetic(
        (main,),
        {x.id: tuple(range(a)), y.id: tuple(range(a, a + b))},
        tuple(range(a + b, begin)),
        entry="main",
        n_qubits=begin + work,
        workspace=tuple(range(begin, begin + work)),
        workspace_state="fresh_zero",
    )
    for left in range(1 << a):
        for right in range(1 << b):
            value = left * left + right if polynomial else left * right
            for target in range(1 << width):
                initial = (((left << b) | right) << width | target) << work
                expected = (((left << b) | right) << width | (target ^ value)) << work
                assert _image(program, initial) == expected


@pytest.mark.parametrize("constant", [0, 1, 3, 5])
@pytest.mark.parametrize("constant_first", [False, True])
def test_constant_multiplication_in_both_positions(
    constant: int, constant_first: bool
) -> None:
    x = _value("main", 0, 2)
    number = Value(ValueId("main", 1), INDEX)
    width = max(1, (3 * constant).bit_length())
    product = _value("main", 2, width)
    operands = (number, x) if constant_first else (x, number)
    main = _function(
        "main",
        (x,),
        (
            Operation(
                "arith.constant", results=(number,), attributes={"value": constant}
            ),
            Operation("qint.multiply", operands, (product,)),
            Operation("qint.return", (product,)),
        ),
    )
    begin = 2 + width
    program = lower_arithmetic(
        (main,),
        {x.id: (0, 1)},
        tuple(range(2, begin)),
        entry="main",
        n_qubits=begin + 2 * width,
        workspace=tuple(range(begin, begin + 2 * width)),
        workspace_state="caller_asserted_zero",
    )
    for value in range(4):
        for target in range(1 << width):
            assert _image(program, ((value << width) | target) << (2 * width)) == (
                ((value << width) | (target ^ (value * constant))) << (2 * width)
            )


def test_same_callee_rebinds_values_and_retains_all_results() -> None:
    arg, squared = _value("square", 0, 1), _value("square", 1, 1)
    square = _function(
        "square",
        (arg,),
        (
            Operation("qint.multiply", (arg, arg), (squared,)),
            Operation("qint.return", (squared,)),
        ),
    )
    x, y, a, b, result = (_value("main", i, w) for i, w in enumerate((1, 1, 1, 1, 2)))
    main = _function(
        "main",
        (x, y),
        (
            Operation("qint.call", (x,), (a,), {"callee": "square"}),
            Operation("qint.call", (y,), (b,), {"callee": "square"}),
            Operation("qint.add", (a, b), (result,)),
            Operation("qint.return", (result,)),
        ),
    )
    program = lower_arithmetic(
        (square, main),
        {x.id: (0,), y.id: (1,)},
        (2, 3),
        entry="main",
        n_qubits=9,
        workspace=tuple(range(4, 9)),
        workspace_state="fresh_zero",
    )
    for x_value in range(2):
        for y_value in range(2):
            for target in range(4):
                prefix = (x_value << 1) | y_value
                assert _image(program, (prefix << 2 | target) << 5) == (
                    (prefix << 2 | (target ^ (x_value**2 + y_value**2))) << 5
                )


@pytest.mark.parametrize("budget", [1, 87, 91])
def test_multiplication_stream_respects_complete_operation_budget(budget: int) -> None:
    x, y, result = _value("main", 0, 2), _value("main", 1, 2), _value("main", 2, 4)
    main = _function(
        "main",
        (x, y),
        (
            Operation(
                "qint.multiply",
                (x, y),
                (result,),
                location=SourceLocation("user.py", 12, 2),
            ),
            Operation("qint.return", (result,)),
        ),
    )
    with pytest.raises(HybridVerificationError, match="lowering.operations"):
        lower_arithmetic(
            (main,),
            {x.id: (0, 1), y.id: (2, 3)},
            (4, 5, 6, 7),
            entry="main",
            n_qubits=16,
            workspace=tuple(range(8, 16)),
            workspace_state="fresh_zero",
            max_operations=budget,
        )


def test_undersized_result_is_rejected_before_modular_kernel_can_truncate() -> None:
    x, result = _value("main", 0, 2), _value("main", 1, 3)
    main = _function(
        "main",
        (x,),
        (
            Operation("qint.multiply", (x, x), (result,)),
            Operation("qint.return", (result,)),
        ),
    )
    with pytest.raises(HybridVerificationError, match="width.overflow"):
        lower_arithmetic((main,), {x.id: (0, 1)}, (2, 3, 4), entry="main", n_qubits=5)


@pytest.mark.parametrize(
    "a,b,product,scratch,error",
    [
        (True, (0,), (1,), (2,), TypeError),
        (-1, (0,), (1,), (2,), ValueError),
        ((), (0,), (1,), (2,), ValueError),
        ((0, 0), (1,), (2,), (3,), ValueError),
        ((0,), (1,), (), (), ValueError),
        ((0,), (1,), (2,), (), ValueError),
        ((True,), (1,), (2,), (3,), TypeError),
        ((None,), (1,), (2,), (3,), TypeError),
        ((0,), (1,), (True,), (3,), TypeError),
        ((-1,), (1,), (2,), (3,), ValueError),
        ((0,), (1,), (2,), (2,), ValueError),
        ((0,), (1,), (0,), (2,), ValueError),
    ],
)
def test_kernel_preconditions(
    a: tuple[int, ...] | int,
    b: tuple[int, ...] | int,
    product: tuple[int, ...],
    scratch: tuple[int, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        tuple(multiply_accumulate(a, b, product, scratch))


def test_modular_kernel_handles_compile_time_pairs_and_high_partial_bits() -> None:
    program = CircuitIR(
        n_wires=2, instructions=tuple(multiply_accumulate(3, 3, (0,), (1,)))
    )
    assert _image(program, 0) == 2  # 9 modulo 2 = 1, scratch remains zero.


def test_trainable_values_are_not_silently_fixed_as_discrete_integers() -> None:
    import torch

    parameter = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    with pytest.raises(TypeError, match="Python integer"):
        uint_type(parameter)
    with pytest.raises(TypeError, match="unsupported hybrid IR attribute"):
        Operation("arith.constant", attributes={"value": parameter})
    with pytest.raises(TypeError, match="Python integer"):
        lower_arithmetic((), {}, (), entry="main", n_qubits=1, max_operations=parameter)
    assert parameter.requires_grad


def test_polynomial_cannot_reuse_a_live_product_as_later_scratch() -> None:
    x, y = _value("main", 0, 2), _value("main", 1, 2)
    square, result = _value("main", 2, 4), _value("main", 3, 4)
    main = _function(
        "main",
        (x, y),
        (
            Operation("qint.multiply", (x, x), (square,)),
            Operation("qint.add", (square, y), (result,)),
            Operation("qint.return", (result,)),
        ),
    )
    # The multiplication fits, but this schedule needs one more qubit to retain
    # its result and finish the addition. It must fail without aliasing a result.
    with pytest.raises(HybridVerificationError, match="lowering.workspace"):
        lower_arithmetic(
            (main,),
            {x.id: (0, 1), y.id: (2, 3)},
            (4, 5, 6, 7),
            entry="main",
            n_qubits=17,
            workspace=tuple(range(8, 17)),
            workspace_state="fresh_zero",
        )
