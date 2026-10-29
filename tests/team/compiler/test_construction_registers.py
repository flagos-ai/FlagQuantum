"""Initialization, ordered effects and complete-program candidate decisions."""

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import pytest

from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._construction.selection import (
    Constraints,
    ConstructionBudget,
    select_register_program,
)
from flagquantum.compiler._construction.target import Target
from flagquantum.compiler._hybrid.model import (
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler.resource_estimation import estimate_resources
from tests.register_program_cases import (
    Registers,
    pressure_program,
    product,
    tutorial_program,
)

pytestmark = pytest.mark.unit


def _changed(
    program: HybridProgram, name: str, change: Callable[[Operation], Operation]
) -> HybridProgram:
    block = program.body.blocks[0]
    operations = list(block.operations)
    position = next(i for i, op in enumerate(operations) if op.name == name)
    operations[position] = change(operations[position])
    return replace(program, body=Region((Block(block.arguments, tuple(operations)),)))


@pytest.mark.parametrize("scheduling", ["retain", "recompute"])
def test_two_schedules_keep_all_outputs_live_and_real_segment_evidence(
    scheduling: Literal["retain", "recompute"],
) -> None:
    program, functions = pressure_program()
    lowered = schedule_register_program(
        program, functions, n_qubits=20, scheduling=scheduling
    )
    assert tuple(lowered.registers) == ("x", "y", "result")
    assert tuple(lowered.registers.values()) == ((0,), (1,), (2, 3))
    assert lowered.peak_workspace == (6 if scheduling == "retain" else 5)
    assert len(lowered.segments) == 1
    assert lowered.segments[0].peak_workspace == lowered.peak_workspace
    assert len(lowered.circuit.instructions) == (43 if scheduling == "retain" else 61)
    assert lowered.recomputations == sum(s.recomputations for s in lowered.segments)


def test_whole_program_constraints_include_preparation_and_exact_target_costs() -> None:
    program, functions = pressure_program()
    narrow = select_register_program(
        program, functions, constraints=Constraints(CircuitLimits(max_qubits=9))
    )
    shallow = select_register_program(
        program, functions, constraints=Constraints(CircuitLimits(max_operations=43))
    )
    assert narrow.selected.strategy == "recompute"
    assert shallow.selected.strategy == "retain"
    for selection in (narrow, shallow):
        assert selection.complete
        for row in selection.rows:
            assert row.source_identities == (
                program.semantic_identity,
                functions[0].semantic_identity,
            )
            assert row.resources == estimate_resources(row.target.executable)
            assert row.resources.n_qubits == 4 + row.arithmetic.peak_workspace
            assert row.stage_resources[-1][1] == row.resources
    assert not select_register_program(
        program, functions, constraints=Constraints(CircuitLimits(max_operations=40))
    ).selected


def test_complete_failures_have_real_retries_but_budget_exhaustion_does_not() -> None:
    program, functions = pressure_program()
    rejected = select_register_program(
        program, functions, constraints=Constraints(CircuitLimits(max_qubits=8))
    )
    assert (
        rejected.complete
        and rejected.selected is None
        and len(rejected.recoveries) == 2
    )
    for recovery in rejected.recoveries:
        assert select_register_program(
            program, functions, constraints=recovery.constraints
        ).selected
    for budget in (
        ConstructionBudget(candidate_checks=1),
        ConstructionBudget(qubits=3),
        ConstructionBudget(operations=2),
        ConstructionBudget(operations=3),
    ):
        result = select_register_program(program, functions, budget=budget)
        assert not result.complete and result.selected is None and not result.recoveries


def test_constants_are_live_quantum_outputs_and_do_not_need_expression_inputs() -> None:
    p = Registers()
    a, b, zero = p.declare("a", 3), p.declare("b", 3), p.declare("zero", 2)
    for register, value in ((a, 4), (b, 5), (zero, 0)):
        p.emit("qreg.initialize", register, value=value)
    selection = select_register_program(p.finish())
    row = selection.selected
    assert selection.registers == {"a": (0, 1, 2), "b": (3, 4, 5), "zero": (6, 7)}
    assert row.resources.n_qubits == 8 and row.resources.n_operations == 3
    assert row.arithmetic.peak_workspace == 0 and not row.arithmetic.segments


@pytest.mark.parametrize("bad", [-1, 2, True, 0.5, "1"])
def test_initialization_refuses_invalid_unsigned_constants(bad: object) -> None:
    p = Registers()
    q = p.declare("q", 1)
    p.emit("qreg.initialize", q, value=bad)
    with pytest.raises(HybridVerificationError, match="register.constant"):
        verify_register_program(p.finish())


@pytest.mark.parametrize("bad", [-1, 1, True, "0"])
def test_gate_index_is_a_checked_numeric_bit_weight(bad: object) -> None:
    p = Registers()
    q = p.declare("q", 1)
    p.emit("qreg.allocate", q)
    p.emit("qreg.gate", q, gate="x", index=bad)
    with pytest.raises(HybridVerificationError, match="register.gate"):
        verify_register_program(p.finish())


@pytest.mark.parametrize(
    "case",
    [
        "double_allocate",
        "double_initialize",
        "read_before_initialize",
        "fresh_assign_to_initialized",
    ],
)
def test_initialization_errors_fail_before_any_lowering(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = Registers()
    q = p.declare("q", 1)
    if case == "read_before_initialize":
        p.emit("qreg.gate", q, gate="h", index=0)
    else:
        p.emit("qreg.allocate", q)
        if case == "double_allocate":
            p.emit("qreg.allocate", q)
        elif case == "double_initialize":
            p.emit("qreg.initialize", q, value=0)
        else:
            graph = capture(product, {"a": 1, "b": 1})
            p.emit("qreg.compute", q, q, q, callee=graph.name)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid semantics must fail before gate emission")

    monkeypatch.setattr(
        "flagquantum.compiler._construction.register_lowering._copy", forbidden
    )
    with pytest.raises(
        HybridVerificationError, match="register.(uninitialized|initialized)"
    ):
        schedule_register_program(
            p.finish(), (graph,) if case == "fresh_assign_to_initialized" else ()
        )


@pytest.mark.parametrize(
    "case",
    [
        "stale_effect",
        "duplicate_effect",
        "undefined",
        "drop",
        "uninitialized_return",
        "duplicate_name",
        "wrong_type",
        "unknown_gate",
        "role",
        "nested",
        "unknown_operation",
        "callee",
        "narrow_output",
    ],
)
def test_closed_profile_rejects_invalid_effects_roles_and_aliases(case: str) -> None:
    program, functions = pressure_program()
    block = program.body.blocks[0]
    if case == "stale_effect":
        program = _changed(
            program,
            "qreg.gate",
            lambda op: replace(op, operands=(block.arguments[0], *op.operands[1:])),
        )
    elif case == "duplicate_effect":
        program = _changed(
            program,
            "qreg.allocate",
            lambda op: replace(op, results=(block.arguments[0],)),
        )
    elif case == "undefined":
        program = _changed(
            program,
            "qreg.gate",
            lambda op: replace(
                op,
                operands=(
                    op.operands[0],
                    replace(op.operands[1], id=replace(op.operands[1].id, index=999)),
                ),
            ),
        )
    elif case == "drop":
        program = _changed(
            program, "qreg.return", lambda op: replace(op, operands=op.operands[:-1])
        )
    elif case == "uninitialized_return":
        p = Registers()
        p.declare("q", 1)
        program, functions = p.finish(), ()
    elif case == "duplicate_name":
        program = _changed(
            program, "qreg.declare", lambda op: replace(op, attributes={"name": "y"})
        )
    elif case == "wrong_type":
        program = _changed(
            program,
            "qreg.declare",
            lambda op: replace(
                op, results=(replace(op.results[0], type=IRType("quantum.uint", (1,))),)
            ),
        )
    elif case == "unknown_gate":
        program = _changed(
            program,
            "qreg.gate",
            lambda op: replace(op, attributes={"gate": {"unknown": "x"}, "index": 0}),
        )
    elif case == "role":
        program = _changed(
            program,
            "qreg.declare",
            lambda op: replace(op, attributes={**op.attributes, "role": "Const"}),
        )
    elif case == "nested":
        program = _changed(
            program,
            "qreg.allocate",
            lambda op: replace(op, regions=(Region((Block(),)),)),
        )
    elif case == "unknown_operation":
        program = _changed(
            program, "qreg.allocate", lambda op: replace(op, name="qreg.free")
        )
    elif case == "callee":
        program = _changed(
            program,
            "qreg.compute",
            lambda op: replace(op, attributes={"callee": "missing"}),
        )
    else:
        operations = tuple(
            (
                replace(
                    op,
                    results=(
                        replace(op.results[0], type=IRType("quantum.register", (1,))),
                    ),
                )
                if op.name == "qreg.declare" and op.attributes["name"] == "result"
                else op
            )
            for op in block.operations
        )
        # Update all uses too, so this is a width error rather than malformed SSA.
        narrow = next(
            op.results[0]
            for op in operations
            if op.name == "qreg.declare" and op.attributes["name"] == "result"
        )
        operations = tuple(
            replace(
                op,
                operands=tuple(
                    narrow if value.id == narrow.id else value for value in op.operands
                ),
            )
            for op in operations
        )
        program = replace(program, body=Region((Block(block.arguments, operations),)))
    with pytest.raises(HybridVerificationError, match="construction.register"):
        verify_register_program(program, functions)


def test_original_declared_tutorial_size_remains_structural_evidence_only() -> None:
    program, functions = tutorial_program(3)
    assert verify_register_program(program, functions) is program
    selected = select_register_program(program, functions)
    assert selected.selected is not None
    assert len(selected.registers["x"]) == 3 and len(selected.registers["y"]) == 6
    assert selected.registers["x"] == (0, 1, 2)
    assert selected.selected.resources.n_qubits > 9
    # This constructs the model at its declared width, not a native-source run.


def test_target_failure_and_invalid_selector_controls_remain_fail_closed() -> None:
    program, functions = pressure_program()
    rejected = select_register_program(
        program, functions, target=Target(basis=("x", "cx", "ccx"))
    )
    assert rejected.selected is None
    assert all(row.target is None for row in rejected.rows)
    with pytest.raises(ValueError, match="objective"):
        select_register_program(program, functions, objective="operations")
    with pytest.raises(TypeError, match="Constraints"):
        select_register_program(program, functions, constraints={})


@pytest.mark.parametrize(
    "case",
    [
        "blocks",
        "entry",
        "return",
        "capacity",
        "nodes",
        "cap",
        "record",
        "function_record",
        "declare",
        "operand",
        "effect_result",
        "destination",
        "extra_initializer_input",
    ],
)
def test_malformed_program_boundaries_have_typed_refusals(case: str) -> None:
    program, functions = pressure_program()
    block = program.body.blocks[0]
    kwargs = {}
    expected = "construction.register"
    if case == "blocks":
        program = replace(program, body=Region((block, block)))
    elif case == "entry":
        program = replace(program, body=Region((replace(block, arguments=()),)))
    elif case == "return":
        program = replace(
            program, body=Region((replace(block, operations=block.operations[:-1]),))
        )
    elif case == "capacity":
        kwargs = {"max_qubits": 3}
    elif case == "nodes":
        kwargs = {"max_nodes": 2}
    elif case == "cap":
        kwargs = {"max_qubits": 257}
    elif case == "record":
        with pytest.raises(TypeError, match="HybridProgram"):
            verify_register_program(None)
        return
    elif case == "function_record":
        with pytest.raises(TypeError, match="HybridProgram"):
            verify_register_program(program, (None,))
        return
    elif case == "declare":
        program = _changed(program, "qreg.declare", lambda op: replace(op, results=()))
    elif case == "operand":
        program = _changed(
            program,
            "qreg.allocate",
            lambda op: replace(op, operands=(*op.operands, block.arguments[0])),
        )
    elif case == "effect_result":
        program = _changed(program, "qreg.allocate", lambda op: replace(op, results=()))
    elif case == "destination":
        program = _changed(
            program, "qreg.allocate", lambda op: replace(op, operands=op.operands[:1])
        )
    else:
        program = _changed(
            program,
            "qreg.allocate",
            lambda op: replace(op, operands=(*op.operands, op.operands[-1])),
        )
    with pytest.raises(HybridVerificationError, match=expected):
        verify_register_program(program, functions, **kwargs)


def test_compute_refuses_uninitialized_inputs_aliases_and_mismatched_widths() -> None:
    from tests.register_program_cases import double

    graph = capture(double, {"x": 1})
    for case in ("uninitialized", "alias", "width"):
        p = Registers()
        x, result = p.declare("x", 2 if case == "width" else 1), p.declare("result", 2)
        if case != "uninitialized":
            p.emit("qreg.allocate", x)
        if case == "alias":
            p.emit("qreg.compute", result, result, callee=graph.name)
            # Match the callee input so alias refusal is not hidden by width drift.
            graph_case = capture(double, {"x": 2})
        else:
            p.emit("qreg.compute", result, x, callee=graph.name)
            graph_case = graph
        with pytest.raises(
            HybridVerificationError, match="register.(uninitialized|alias|signature)"
        ):
            verify_register_program(p.finish(), (graph_case,))


def test_arithmetic_wrapper_names_cannot_collide_with_source_functions() -> None:
    program, functions = pressure_program()
    ordinal = next(
        i
        for i, op in enumerate(program.body.blocks[0].operations)
        if op.name == "qreg.compute"
    )
    renamed = replace(functions[0], name=f"{program.name}.compute{ordinal}")
    program = _changed(
        program,
        "qreg.compute",
        lambda op: replace(op, attributes={"callee": renamed.name}),
    )
    result = select_register_program(program, (renamed,))
    assert result.selected is not None


@pytest.mark.parametrize(
    "kwargs", [{"max_operations": 100001}, {"scheduling": "optimal"}]
)
def test_lowering_never_enlarges_safety_caps_or_guesses_a_schedule(kwargs) -> None:
    program, functions = pressure_program()
    with pytest.raises(HybridVerificationError, match="register.budget"):
        schedule_register_program(program, functions, **kwargs)
