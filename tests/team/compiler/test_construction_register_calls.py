"""Const calls retain basis values and validate complete fresh-output contracts."""

from dataclasses import replace

import pytest

from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.register_expansion import expand_register_calls
from flagquantum.compiler._construction.register_profile import (
    register_type,
    verify_register_program,
)
from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.compiler._hybrid.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.register_program_cases import Registers
from tools.synthesis_compiler_examples.const_calls.main import const_composition

pytestmark = pytest.mark.unit


def identity(x: int) -> int:
    return x


def chain(
    count: int, repeat: int = 1
) -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    functions = []
    for i in range(count):
        name = f"f{i}"
        e = Value(ValueId(name, 0), QUANTUM_EFFECT)
        x = Value(ValueId(name, 1), register_type(1))
        effect, ops = e, []
        for j in range(repeat if i else 0):
            following = Value(ValueId(name, j + 2), QUANTUM_EFFECT)
            ops.append(
                Operation("qreg.call", (effect, x), (following,), {"callee": f"f{i-1}"})
            )
            effect = following
        functions.append(
            HybridProgram(
                name,
                Region((Block((e, x), (*ops, Operation("qreg.return", (effect, x)))),)),
            )
        )
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.call", x, callee=f"f{count-1}")
    return p.finish(), tuple(functions)


def test_mixed_calls_expand_to_existing_actions_and_share_scratch():
    model, functions = const_composition()
    assert verify_register_program(model, functions) is model
    expanded = expand_register_calls(model, functions)
    assert "qreg.call" not in {op.name for op in expanded.body.blocks[0].operations}
    assert {op.name for op in expanded.body.blocks[0].operations} >= {
        "qreg.compute",
        "qreg.phase",
    }
    selection = select_register_program(model, functions)
    assert selection.selected.arithmetic.peak_workspace == max(
        s.peak_workspace for s in selection.selected.arithmetic.segments
    )


@pytest.mark.parametrize(
    "case,code",
    [
        ("missing_attribute", "operation"),
        ("extra_attribute", "operation"),
        ("regions", "operation"),
        ("unknown", "callee"),
        ("callee_type", "callee"),
        ("arithmetic_callee", "callee"),
        ("wrong_arity", "signature"),
        ("wrong_width", "signature"),
        ("initialized_output", "initialized"),
        ("uninitialized_const", "uninitialized"),
        ("stale_effect", "use"),
        ("no_result", "effect"),
    ],
)
def test_malformed_root_calls_fail_before_lowering(case, code):
    model, functions = const_composition()
    block = model.body.blocks[0]
    ops = list(block.operations)
    pos = next(i for i, op in enumerate(ops) if op.name == "qreg.call")
    op = ops[pos]
    if case == "missing_attribute":
        op = replace(op, attributes={})
    elif case == "extra_attribute":
        op = replace(op, attributes={"callee": "combined_call", "const": True})
    elif case == "regions":
        op = replace(op, regions=(Region((Block(),)),))
    elif case == "unknown":
        op = replace(op, attributes={"callee": "missing"})
    elif case == "callee_type":
        op = replace(op, attributes={"callee": 42})
    elif case == "arithmetic_callee":
        op = replace(op, attributes={"callee": "energy"})
    elif case == "wrong_arity":
        op = replace(op, operands=op.operands[:-1])
    elif case == "wrong_width":
        op = replace(op, operands=(op.operands[0], op.operands[-1], *op.operands[2:]))
    elif case == "initialized_output":
        previous = ops[pos - 1]
        ops.insert(
            pos,
            replace(
                previous,
                name="qreg.allocate",
                operands=(op.operands[0], op.operands[-1]),
                results=(Value(ValueId("insert", 0), QUANTUM_EFFECT),),
                attributes={},
            ),
        )
        pos += 1
        op = replace(op, operands=(ops[pos - 1].results[0], *op.operands[1:]))
    elif case == "uninitialized_const":
        fresh = Value(ValueId("fresh", 0), register_type(2))
        ops.insert(
            pos,
            Operation("qreg.declare", results=(fresh,), attributes={"name": "fresh"}),
        )
        pos += 1
        op = replace(op, operands=(op.operands[0], fresh, *op.operands[2:]))
        ops[-1] = replace(ops[-1], operands=(*ops[-1].operands, fresh))
    elif case == "stale_effect":
        op = replace(op, operands=(op.results[0], *op.operands[1:]))
    elif case == "no_result":
        op = replace(op, results=())
    ops[pos] = op
    with pytest.raises(HybridVerificationError, match=f"register.{code}"):
        verify_register_program(
            replace(model, body=Region((replace(block, operations=tuple(ops)),))),
            functions,
        )


@pytest.mark.parametrize(
    "opcode,attributes,code",
    [
        ("qreg.gate", {"gate": "x", "index": 0}, "const"),
        ("qreg.gate", {"gate": "h", "index": 0}, "const"),
        ("qreg.initialize", {"value": 0}, "const"),
    ],
)
def test_unused_callee_cannot_mutate_const_even_if_state_would_be_restored(
    opcode, attributes, code
):
    root, functions = chain(1)
    helper = functions[0]
    block = helper.body.blocks[0]
    e, x = block.arguments
    next_effect = Value(ValueId("bad", 0), QUANTUM_EFFECT)
    bad = replace(
        helper,
        body=Region(
            (
                replace(
                    block,
                    operations=(
                        Operation(opcode, (e, x), (next_effect,), attributes),
                        Operation("qreg.return", (next_effect, x)),
                    ),
                ),
            )
        ),
    )
    p = Registers()
    q = p.declare("q", 1)
    p.emit("qreg.allocate", q)
    with pytest.raises(HybridVerificationError, match=f"register.{code}"):
        verify_register_program(p.finish(), (bad,))


@pytest.mark.parametrize(
    "case,code",
    [
        ("type", "type"),
        ("ssa", "ssa"),
        ("return", "return"),
        ("capacity", "budget"),
        ("entry", "entry"),
    ],
)
def test_invalid_const_entry_and_return_are_rejected(case, code):
    root, functions = chain(1)
    h = functions[0]
    b = h.body.blocks[0]
    e, x = b.arguments
    if case == "type":
        b = replace(b, arguments=(e, Value(x.id, QUANTUM_EFFECT)))
    elif case == "ssa":
        b = replace(b, arguments=(e, Value(e.id, register_type(1))))
    elif case == "return":
        b = replace(b, operations=(Operation("qreg.return", (e,)),))
    elif case == "capacity":
        b = replace(b, arguments=(e, Value(x.id, register_type(257))))
    elif case == "entry":
        b = replace(b, arguments=())
    with pytest.raises(HybridVerificationError, match=f"register.{code}"):
        verify_register_program(root, (replace(h, body=Region((b,))),))


def test_empty_const_call_still_checks_initialized_actual():
    _, functions = chain(1)
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.call", x, callee="f0")
    p.emit("qreg.allocate", x)
    with pytest.raises(HybridVerificationError, match="register.uninitialized"):
        verify_register_program(p.finish(), functions)


def test_borrowed_callee_is_not_a_fresh_root_and_symbols_are_unique():
    root, functions = chain(1)
    with pytest.raises(HybridVerificationError, match="register.entry"):
        select_register_program(functions[0])
    with pytest.raises(HybridVerificationError, match="register.callee"):
        verify_register_program(root, functions * 2)


def test_output_actuals_cannot_alias_each_other():
    helper = Registers()
    a, b = helper.declare("a", 1), helper.declare("b", 1)
    helper.emit("qreg.allocate", a)
    helper.emit("qreg.allocate", b)
    callee = replace(helper.finish(), name="fresh_pair")
    root = Registers()
    x = root.declare("x", 1)
    root.emit("qreg.call", x, x, callee="fresh_pair")
    with pytest.raises(HybridVerificationError, match="register.alias"):
        verify_register_program(root.finish(), (callee,))


def test_output_cannot_alias_a_const_actual_with_matching_widths():
    root, functions = chain(1)
    helper = functions[0]
    e, x = helper.body.blocks[0].arguments
    out = Value(ValueId("out", 0), register_type(1))
    after = Value(ValueId("after", 0), QUANTUM_EFFECT)
    graph = capture(identity, {"x": 1})
    helper = replace(
        helper,
        body=Region(
            (
                Block(
                    (e, x),
                    (
                        Operation(
                            "qreg.declare", results=(out,), attributes={"name": "out"}
                        ),
                        Operation(
                            "qreg.compute",
                            (e, out, x),
                            (after,),
                            {"callee": graph.name},
                        ),
                        Operation("qreg.return", (after, x, out)),
                    ),
                ),
            )
        ),
    )
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.call", x, x, callee="f0")
    with pytest.raises(HybridVerificationError, match="register.alias"):
        verify_register_program(p.finish(), (graph, helper))


def test_cancelling_gates_do_not_bypass_structural_const_restrictions():
    root, functions = chain(1)
    h = functions[0]
    e, x = h.body.blocks[0].arguments
    first, second = (Value(ValueId("cancel", i), QUANTUM_EFFECT) for i in range(2))
    h = replace(
        h,
        body=Region(
            (
                Block(
                    (e, x),
                    (
                        Operation(
                            "qreg.gate", (e, x), (first,), {"gate": "x", "index": 0}
                        ),
                        Operation(
                            "qreg.gate",
                            (first, x),
                            (second,),
                            {"gate": "x", "index": 0},
                        ),
                        Operation("qreg.return", (second, x)),
                    ),
                ),
            )
        ),
    )
    with pytest.raises(HybridVerificationError, match="register.const"):
        verify_register_program(root, (h,))


def test_expansion_avoids_existing_ssa_identifiers():
    _, functions = chain(1)
    e = Value(ValueId("collision", 0), QUANTUM_EFFECT)
    x = Value(ValueId("collision.calls", 0), register_type(1))
    first, second = (Value(ValueId("original", i), QUANTUM_EFFECT) for i in range(2))
    root = HybridProgram(
        "collision",
        Region(
            (
                Block(
                    (e,),
                    (
                        Operation(
                            "qreg.declare", results=(x,), attributes={"name": "x"}
                        ),
                        Operation("qreg.allocate", (e, x), (first,)),
                        Operation("qreg.call", (first, x), (second,), {"callee": "f0"}),
                        Operation("qreg.return", (second, x)),
                    ),
                ),
            )
        ),
    )
    verify_register_program(root, functions)
    expanded = expand_register_calls(root, functions)
    assert expanded.body.blocks[0].operations[1].results[0].id == ValueId(
        "collision.calls", 1
    )


def test_const_cannot_be_passed_to_a_nested_output():
    p = Registers()
    out = p.declare("out", 1)
    p.emit("qreg.allocate", out)
    fresh = replace(p.finish(), name="fresh")
    root, functions = chain(1)
    h = functions[0]
    e, x = h.body.blocks[0].arguments
    after = Value(ValueId("bad", 0), QUANTUM_EFFECT)
    bad = replace(
        h,
        body=Region(
            (
                Block(
                    (e, x),
                    (
                        Operation("qreg.call", (e, x), (after,), {"callee": "fresh"}),
                        Operation("qreg.return", (after, x)),
                    ),
                ),
            )
        ),
    )
    with pytest.raises(HybridVerificationError, match="register.const"):
        verify_register_program(root, (fresh, bad))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("count", [32, 33])
def test_nesting_budget_is_independent_of_definition_order(reverse, count):
    root, functions = chain(count)
    if reverse:
        functions = functions[::-1]
    if count == 32:
        verify_register_program(root, functions)
    else:
        with pytest.raises(HybridVerificationError, match="register.budget"):
            verify_register_program(root, functions)


def test_cycles_and_exponential_empty_call_visits_are_bounded():
    root, functions = chain(1)
    h = functions[0]
    b = h.body.blocks[0]
    e, x = b.arguments
    after = Value(ValueId("cycle", 0), QUANTUM_EFFECT)
    cyclic = replace(
        h,
        body=Region(
            (
                replace(
                    b,
                    operations=(
                        Operation("qreg.call", (e, x), (after,), {"callee": "f0"}),
                        Operation("qreg.return", (after, x)),
                    ),
                ),
            )
        ),
    )
    with pytest.raises(HybridVerificationError, match="register.recursion"):
        verify_register_program(root, (cyclic,))
    root, functions = chain(13, 2)
    with pytest.raises(HybridVerificationError, match="expansion exceeds visit budget"):
        verify_register_program(root, functions)


@pytest.mark.parametrize("callee", ["missing", 42])
def test_unknown_or_malformed_nested_callee_is_rejected(callee):
    root, functions = chain(1)
    h = functions[0]
    e, x = h.body.blocks[0].arguments
    after = Value(ValueId("nested", 0), QUANTUM_EFFECT)
    h = replace(
        h,
        body=Region(
            (
                Block(
                    (e, x),
                    (
                        Operation("qreg.call", (e, x), (after,), {"callee": callee}),
                        Operation("qreg.return", (after, x)),
                    ),
                ),
            )
        ),
    )
    with pytest.raises(HybridVerificationError, match="register.callee"):
        verify_register_program(root, (h,))


def test_expanded_nodes_include_shared_arithmetic_beyond_visit_count():
    graph = capture(identity, {"x": 1})
    _, functions = chain(1)
    h = functions[0]
    b = h.body.blocks[0]
    e, x = b.arguments
    ops = []
    effect = e
    for i in range(4):
        following = Value(ValueId("phase", i), QUANTUM_EFFECT)
        ops.append(
            Operation("qreg.phase", (effect, x), (following,), {"callee": "identity"})
        )
        effect = following
    helper = replace(
        h,
        body=Region(
            (replace(b, operations=(*ops, Operation("qreg.return", (effect, x)))),)
        ),
    )
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    for _ in range(4):
        p.emit("qreg.call", x, callee="f0")
    with pytest.raises(
        HybridVerificationError, match="expanded register model exceeds node budget"
    ):
        verify_register_program(p.finish(), (graph, helper), max_nodes=21)
