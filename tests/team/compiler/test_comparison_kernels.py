"""Independent exhaustive gate permutations for the shared comparator kernel."""

import itertools

import pytest

from flagquantum.compiler._arithmetic_kernels import comparison_xor

pytestmark = pytest.mark.unit


def execute(gates, ones):
    state = set(ones)
    for gate in gates:
        assert gate.name in {"x", "cx", "ccx"}
        if all(wire in state for wire in gate.wires[:-1]):
            target = gate.wires[-1]
            state.symmetric_difference_update((target,))
    return state


@pytest.mark.parametrize("operation", ["eq", "lt", "gt"])
@pytest.mark.parametrize(
    "kind",
    [
        "registers",
        "left_constant",
        "right_constant",
        "alias",
        "partial_alias",
        "unequal",
        "both_constants",
        "padding_constants",
    ],
)
def test_all_small_inputs_targets_and_clean_workspace(operation, kind):
    a, b = (0, 1), (2, 3)
    if kind == "left_constant":
        a = 2
    if kind == "right_constant":
        b = 2
    if kind == "alias":
        b = a
    if kind == "partial_alias":
        b = (1, 2)
    if kind == "unequal":
        b = (2,)
    if kind == "both_constants":
        a, b = 3, 2
    if kind == "padding_constants":
        a, b = (0,), 8
    registers = tuple(
        dict.fromkeys(q for value in (a, b) if isinstance(value, tuple) for q in value)
    )
    width = max(
        len(value) if isinstance(value, tuple) else max(1, value.bit_length())
        for value in (a, b)
    )
    flags = tuple(range(5, 6 + width))
    gates = comparison_xor(a, b, 4, flags, 5 + len(flags), operation=operation)
    for values in itertools.product((0, 1), repeat=len(registers)):
        ones = {q for q, value in zip(registers, values, strict=True) if value}

        def integer(value, selected=ones):
            return (
                value
                if isinstance(value, int)
                else sum(
                    (q in selected) << (len(value) - i - 1) for i, q in enumerate(value)
                )
            )

        left, right = integer(a), integer(b)
        expected = {"eq": left == right, "lt": left < right, "gt": left > right}[
            operation
        ]
        for target in (0, 1):
            initial = ones | ({4} if target else set())
            final = ones | ({4} if target ^ expected else set())
            assert execute(gates, initial) == final
            assert execute(tuple(reversed(gates)), final) == initial


@pytest.mark.parametrize(
    "a,b,target,flags,scratch,operation,error",
    [
        ((0,), (1,), 2, (3, 4), 5, "ne", ValueError),
        (True, (1,), 2, (3, 4), 5, "eq", TypeError),
        ((0,), -1, 2, (3, 4), 5, "lt", ValueError),
        ((), (1,), 2, (3, 4), 5, "eq", ValueError),
        ((0, 0), (1,), 2, (3, 4, 5), 6, "eq", ValueError),
        ((False,), (1,), 2, (3, 4), 5, "eq", TypeError),
        ((-1,), (1,), 2, (3, 4), 5, "eq", ValueError),
        ((0,), (1,), 2, (3,), 5, "eq", ValueError),
        ((0,), (1,), 2, (3, 3), 5, "eq", ValueError),
        ((0,), (1,), 0, (3, 4), 5, "eq", ValueError),
    ],
)
def test_bad_kernel_bindings_are_refused(
    a, b, target, flags, scratch, operation, error
):
    with pytest.raises(error):
        comparison_xor(a, b, target, flags, scratch, operation=operation)
