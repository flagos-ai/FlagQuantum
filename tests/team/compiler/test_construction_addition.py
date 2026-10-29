"""Addition lowering is an exact XOR permutation with clean workspace."""

from dataclasses import replace

import pytest

from flagquantum.compiler._arithmetic_kernels import ripple_add
from flagquantum.compiler._construction.lowering import lower_addition
from flagquantum.compiler._construction.profile import uint_type
from flagquantum.compiler._construction.ranges import unsigned_width
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


def _value(name: str, index: int, width: int) -> Value:
    return Value(ValueId(name, index), uint_type(width))


def _function(
    name: str, arguments: tuple[Value, ...], operations: tuple[Operation, ...]
) -> HybridProgram:
    return HybridProgram(name, Region((Block(arguments, operations),)))


def _program(a: int = 2, b: int = 2, out: int = 3) -> HybridProgram:
    x, y, result = (_value("main", i, w) for i, w in enumerate((a, b, out)))
    return _function(
        "main",
        (x, y),
        (Operation("qint.add", (x, y), (result,)), Operation("qint.return", (result,))),
    )


def _lower(program: HybridProgram, *, extra: int = 0, **limits: int) -> CircuitIR:
    args = program.body.blocks[0].arguments
    a, b = (int(v.type.parameters[0]) for v in args)
    out = int(program.body.blocks[0].operations[-1].operands[0].type.parameters[0])
    start = a + b + out
    count = start + 2 * max(a, b) + 2 + extra
    return lower_addition(
        (program,),
        {args[0].id: tuple(range(a)), args[1].id: tuple(range(a, a + b))},
        tuple(range(a + b, start)),
        entry="main",
        n_wires=count,
        workspace=tuple(range(start, count)),
        workspace_state="fresh_zero",
        **limits,
    )


def _image(program: CircuitIR, bits: int) -> int:
    """Independent Boolean action of the emitted phase-free primitive set."""
    for instruction in program.instructions:
        controls, target = instruction.wires[:-1], instruction.wires[-1]
        assert instruction.name in {"x", "cx", "ccx"}
        if all(bits & (1 << (program.n_wires - 1 - wire)) for wire in controls):
            bits ^= 1 << (program.n_wires - 1 - target)
    return bits


@pytest.mark.parametrize("a,b,out", [(1, 1, 2), (1, 3, 4), (3, 1, 4), (2, 2, 4)])
def test_all_addends_and_nonzero_xor_targets_are_exact_and_clean(
    a: int, b: int, out: int
) -> None:
    program = _lower(_program(a, b, out), extra=1)
    workspace = program.n_wires - a - b - out
    for x in range(1 << a):
        for y in range(1 << b):
            for z in range(1 << out):
                initial = (((x << b) | y) << out | z) << workspace
                expected = (((x << b) | y) << out | (z ^ (x + y))) << workspace
                assert _image(program, initial) == expected


def test_calls_repeated_reads_constants_and_retained_intermediates() -> None:
    a, doubled = _value("double", 0, 2), _value("double", 1, 3)
    double = _function(
        "double",
        (a,),
        (
            Operation("qint.add", (a, a), (doubled,)),
            Operation("qint.return", (doubled,)),
        ),
    )
    x, twice, answer = _value("main", 0, 2), _value("main", 1, 3), _value("main", 3, 4)
    constant = Value(ValueId("main", 2), INDEX)
    main = _function(
        "main",
        (x,),
        (
            Operation("qint.call", (x,), (twice,), {"callee": "double"}),
            Operation("arith.constant", results=(constant,), attributes={"value": 3}),
            Operation("qint.add", (constant, twice), (answer,)),
            Operation("qint.return", (answer,)),
        ),
    )
    # Six wires for double, eight for the final sum. Retain all until cleanup.
    program = lower_addition(
        (double, main),
        {x.id: (0, 1)},
        (2, 3, 4, 5),
        entry="main",
        n_wires=20,
        workspace=tuple(range(6, 20)),
        workspace_state="fresh_zero",
    )
    for value in range(4):
        for target in range(16):
            initial = ((value << 4) | target) << 14
            expected = ((value << 4) | (target ^ (2 * value + 3))) << 14
            assert _image(program, initial) == expected


@pytest.mark.parametrize("constant,width", [(0, 2), (1, 3), (7, 4)])
def test_constant_widths_do_not_truncate_and_zero_uses_one_bit(
    constant: int, width: int
) -> None:
    x, result = _value("main", 0, 2), _value("main", 2, width)
    number = Value(ValueId("main", 1), INDEX)
    main = _function(
        "main",
        (x,),
        (
            Operation(
                "arith.constant", results=(number,), attributes={"value": constant}
            ),
            Operation("qint.add", (x, number), (result,)),
            Operation("qint.return", (result,)),
        ),
    )
    start = 2 + width
    work = 2 * max(2, unsigned_width(constant)) + 2
    program = lower_addition(
        (main,),
        {x.id: (1, 0)},
        tuple(range(2, start)),
        entry="main",
        n_wires=start + work,
        workspace=tuple(range(start, start + work)),
        workspace_state="caller_asserted_zero",
    )
    for x_value in range(4):
        physical_x = ((x_value & 1) << 1) | (x_value >> 1)
        for z in range(1 << width):
            assert _image(program, ((physical_x << width) | z) << work) == (
                ((physical_x << width) | (z ^ (x_value + constant))) << work
            )


@pytest.mark.parametrize("bound,expected", [(0, 1), (1, 1), (3, 2), (4, 3), (8, 4)])
def test_unsigned_range_width(bound: int, expected: int) -> None:
    assert unsigned_width(bound) == expected


@pytest.mark.parametrize("bound", [True, 1.5, -1])
def test_range_rejects_lossy_or_negative_bounds(bound: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        unsigned_width(bound)


@pytest.mark.parametrize("budget", [1, 41, 44])
def test_operation_budget_includes_inverse_and_final_copy(budget: int) -> None:
    with pytest.raises(HybridVerificationError, match="lowering.operations"):
        _lower(_program(), max_operations=budget)


def test_exact_operation_budget_is_accepted() -> None:
    program = _lower(_program())
    assert len(program.instructions) == 45
    assert _lower(_program(), max_operations=45).instructions == program.instructions


def test_insufficient_workspace_reports_source_before_returning_ir() -> None:
    program = _program()
    block = program.body.blocks[0]
    add = replace(block.operations[0], location=SourceLocation("addition.py", 9, 4))
    program = _function("main", block.arguments, (add, block.operations[1]))
    with pytest.raises(
        HybridVerificationError, match=r"lowering.workspace: addition.py:9:4"
    ):
        _lower(program, extra=-1)


def test_multiplication_is_verified_but_not_lowered() -> None:
    program = _program(out=4)
    block = program.body.blocks[0]
    multiply = replace(block.operations[0], name="qint.multiply")
    with pytest.raises(HybridVerificationError, match="lowering.unsupported"):
        _lower(_function("main", block.arguments, (multiply, block.operations[1])))


@pytest.mark.parametrize("opcode", ["qint.inplace_add", "qint.inplace_xor"])
def test_source_wraparound_and_truncation_are_rejected(opcode: str) -> None:
    program = _program(3, 1, 3)
    block = program.body.blocks[0]
    update = replace(block.operations[0], name=opcode)
    with pytest.raises(HybridVerificationError, match="operation.unsupported"):
        _lower(_function("main", block.arguments, (update, block.operations[1])))


def test_full_precision_three_bit_seven_plus_one_cannot_return_three_bits() -> None:
    with pytest.raises(HybridVerificationError, match="width.overflow"):
        _lower(_program(3, 1, 3))


def test_shared_kernel_accepts_permuted_maps_and_rejects_aliasing() -> None:
    gates = ripple_add((4, 0), (6, 2), 3, 1)
    assert len(gates) == 17
    with pytest.raises(ValueError, match="disjoint"):
        ripple_add((0,), (0,), 1, 2)


@pytest.mark.parametrize(
    "n_wires,budget,error",
    [
        (257, 100000, HybridVerificationError),
        (13, 100001, HybridVerificationError),
        (True, 100000, TypeError),
        (13, True, TypeError),
        (13, 0, ValueError),
    ],
)
def test_safety_limits_fail_before_allocation(
    n_wires: int,
    budget: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        lower_addition((), {}, (), entry="main", n_wires=n_wires, max_operations=budget)


def test_unknown_external_workspace_state_is_rejected_at_lowering_entry() -> None:
    program = _program()
    x, y = program.body.blocks[0].arguments
    with pytest.raises(HybridVerificationError, match="workspace.precondition"):
        lower_addition(
            (program,),
            {x.id: (0, 1), y.id: (2, 3)},
            (4, 5, 6),
            entry="main",
            n_wires=13,
            workspace=tuple(range(7, 13)),
        )


def test_identity_return_needs_no_workspace_and_is_an_xor_copy() -> None:
    x = _value("main", 0, 2)
    program = _function("main", (x,), (Operation("qint.return", (x,)),))
    lowered = lower_addition(
        (program,),
        {x.id: (0, 1)},
        (2, 3),
        entry="main",
        n_wires=4,
    )
    for a in range(4):
        for z in range(4):
            assert _image(lowered, (a << 2) | z) == (a << 2) | (z ^ a)


@pytest.mark.parametrize(
    "a,b,carry,out,error",
    [
        ((), (), 0, 1, ValueError),
        ((0,), (1, 2), 3, 4, ValueError),
        ((True,), (1,), 2, 3, TypeError),
        ((-1,), (1,), 2, 3, ValueError),
    ],
)
def test_kernel_map_preconditions(
    a: tuple[int, ...],
    b: tuple[int, ...],
    carry: int,
    out: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ripple_add(a, b, carry, out)
