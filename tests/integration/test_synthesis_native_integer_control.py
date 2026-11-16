"""End-to-end integer specialization and checked classical branches for native qfunc."""

from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.main import (
    ALICE_BASES,
    BOB_BASES,
    MESSAGE,
    build_program,
    main,
    prepare_key,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/bb84/bb84_implementation.qmod"
)
ERRORS = (synth.SynthesisError, HybridVerificationError)
SECOND = ((1, 0, 1, 0, 1, 1, 0, 1), (0, 1, 0, 1, 0, 1, 1, 0), (1, 0, 1, 1, 0, 0, 1, 0))


def definition(
    body: str, signature: str = "q:Output[QBit]", prelude: str = ""
) -> _QuantumFunction:
    return program_definition(
        "from flagquantum.synth import *\n"
        + prelude
        + f"\n@qfunc\ndef main({signature}):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_fixed_bb84_matches_original_and_independent_full_state(
    objective: Literal["width", "depth"],
) -> None:
    original = synth.synthesize(SOURCE, objective=objective)
    native = synth.synthesize(main, objective=objective)
    for program in (native, original):
        compare_states(program, reference_circuit(), ("qba",))
        assert program.report["workspace_qubits"] == 0
    assert native.report["resources"] == original.report["resources"]
    assert native.report["source_resources"] == original.report["source_resources"]


@pytest.mark.parametrize("data", [(MESSAGE, ALICE_BASES, BOB_BASES), SECOND])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_integer_array_root_and_helper_binding_preserve_all_eight_signals(
    data: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    objective: Literal["width", "depth"],
) -> None:
    message, alice, bob = data
    native = synth.synthesize(
        prepare_key,
        classical_inputs={"message": message, "alice_bases": alice, "bob_bases": bob},
        objective=objective,
        max_workspace=0,
    )
    compare_states(native, reference_circuit(message, alice, bob), ("qba",))
    assert native.report["workspace_qubits"] == 0
    compare_states(
        build_program(message, alice, bob),
        reference_circuit(message, alice, bob),
        ("qba",),
    )


def test_borrowed_bb84_keeps_coherent_input_and_caller_binding_scopes() -> None:
    entry = program_definition(
        "from flagquantum.synth import *\n"
        "from examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.main import bb84\n"
        "@qfunc\ndef main(message:list[int],q:Output[QArray[QBit]]):\n"
        "    allocate(2,q)\n    hadamard_transform(q)\n    S(q[0])\n"
        "    bb84(message,[1,0],[0,1],q)\n"
        "    if message[0] == 1:\n        RZ(0.37,q[1])\n"
    )
    native = synth.synthesize(entry, classical_inputs={"message": [1, 0]})
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .h(0)
        .h(1)
        .s(1)
        .x(1)
        .h(1)
        .h(0)
        .rz(0, theta=0.37)
    )
    compare_states(native, reference, ("q",))


@pytest.mark.parametrize("count", [-2, 0, 1, 3])
def test_integer_scalar_drives_nested_helpers_range_and_branch_order(
    count: int,
) -> None:
    entry = definition(
        "allocate(q)\napply_steps(count,q)\nif count >= 1:\n    RZ(0.31,q)\nelse:\n    H(q)",
        "count:int,q:Output[QBit]",
        "@qfunc\ndef apply_steps(n:int,q:QBit):\n    for i in range(n):\n        if i == 1:\n            S(q)\n        else:\n            H(q)\n",
    )
    native = synth.synthesize(entry, classical_inputs={"count": count})
    reference = fq.Circuit(1, dtype=torch.complex128)
    for index in range(count):
        reference.s(0) if index == 1 else reference.h(0)
    if count >= 1:
        reference.rz(0, theta=0.31)
    else:
        reference.h(0)
    compare_states(native, reference, ("q",))


@pytest.mark.parametrize(
    "expression,selected",
    [
        ("n == 2", True),
        ("n != 2", False),
        ("n < 3", True),
        ("n <= 1", False),
        ("n > 1", True),
        ("n >= 3", False),
        ("not (n < 2)", True),
        ("n == 2 and n > 0", True),
        ("n < 0 or n >= 2", True),
        ("False", False),
        ("True and not False", True),
        ("values[0] == 0.25", True),
    ],
)
def test_closed_boolean_combinations_and_real_comparisons(
    expression: str, selected: bool
) -> None:
    entry = definition(
        f"allocate(q)\nH(q)\nif {expression}:\n    S(q)\nelif n == 2:\n    RZ(-0.41,q)\nelse:\n    X(q)",
        "n:int,values:list[float],q:Output[QBit]",
    )
    reference = fq.Circuit(1, dtype=torch.complex128).h(0)
    reference.s(0) if selected else reference.rz(0, theta=-0.41)
    compare_states(
        synth.synthesize(entry, classical_inputs={"n": 2, "values": [0.25]}),
        reference,
        ("q",),
    )


@pytest.mark.parametrize("integer", [-(2**53), 2**53, 2**53 - 1])
def test_integer_identity_is_preserved_until_real_gate_use(integer: int) -> None:
    entry = definition(
        f"allocate(q)\nif n == {integer}:\n    H(q)\nelse:\n    X(q)",
        "n:int,q:Output[QBit]",
    )
    compare_states(
        synth.synthesize(entry, classical_inputs={"n": integer}),
        fq.Circuit(1, dtype=torch.complex128).h(0),
        ("q",),
    )


@pytest.mark.parametrize(
    "value",
    [True, False, 1.0, 0.25, float("inf"), float("nan"), 2**53 + 1, -(2**53) - 1],
)
def test_integer_root_rejects_bool_float_nonfinite_and_out_of_range(
    value: int | float,
) -> None:
    entry = definition("allocate(q)", "n:int,q:Output[QBit]")
    with pytest.raises(ERRORS, match="integer bindings"):
        synth.synthesize(entry, classical_inputs={"n": value})


@pytest.mark.parametrize("values", [[1.0], [True], [2**53 + 1], [], [1] * 4097])
def test_integer_arrays_keep_element_types_and_shared_binding_budget(
    values: list[int] | list[float],
) -> None:
    entry = definition("allocate(q)", "values:list[int],q:Output[QBit]")
    with pytest.raises(ERRORS, match="integer bindings|4096"):
        synth.synthesize(entry, classical_inputs={"values": values})


@pytest.mark.parametrize("actual", ["1.0", "1 / 1", "angles[0]", "[1.0]", "[True]"])
def test_integer_helpers_never_narrow_real_or_bool_actuals(actual: str) -> None:
    array = actual.startswith("[")
    formal = "list[int]" if array else "int"
    entry = definition(
        f"allocate(q)\nhelper({actual},q)",
        "angles:list[float],q:Output[QBit]",
        f"@qfunc\ndef helper(n:{formal},q:QBit):\n    H(q)\n",
    )
    with pytest.raises(ERRORS, match="integer bindings|static numbers"):
        synth.synthesize(entry, classical_inputs={"angles": [1.0]})


def test_real_formals_remain_real_even_when_root_value_is_an_integer() -> None:
    entry = definition(
        "allocate(q)\nhelper(value,q)",
        "value:float,q:Output[QBit]",
        "@qfunc\ndef helper(n:int,q:QBit):\n    H(q)\n",
    )
    with pytest.raises(ERRORS, match="integer bindings"):
        synth.synthesize(entry, classical_inputs={"value": 1})
    for body in ("allocate(q)\nfor i in range(n):\n    H(q)", "allocate(2,q)\nH(q[n])"):
        signature = (
            "n:float,q:Output[QBit]"
            if "range" in body
            else "n:float,q:Output[QArray[QBit]]"
        )
        with pytest.raises(ERRORS, match="real parameters|static integer"):
            synth.synthesize(definition(body, signature), classical_inputs={"n": 1})


@pytest.mark.parametrize("index", [-1, 2])
def test_classical_integer_element_reads_are_checked(index: int) -> None:
    entry = definition(
        f"allocate(q)\nif values[{index}] == 1:\n    X(q)",
        "values:list[int],q:Output[QBit]",
    )
    with pytest.raises(ERRORS, match="out of bounds"):
        synth.synthesize(entry, classical_inputs={"values": [1, 0]})


@pytest.mark.parametrize(
    "condition",
    [
        "q == 0",
        "q",
        "n",
        "0 < n < 3",
        "True or host()",
        "False and host()",
        "host() == 1",
        "n / 0 == 1",
    ],
)
def test_conditions_cannot_read_qubits_or_execute_host_code(condition: str) -> None:
    entry = definition(
        f"allocate(q)\nif {condition}:\n    X(q)",
        "n:int,q:Output[QBit]",
        "def host():\n    raise AssertionError('Host code must never execute')\n",
    )
    with pytest.raises(ERRORS, match="program.condition"):
        synth.synthesize(entry, classical_inputs={"n": 1})


@pytest.mark.parametrize(
    "body,diagnostic",
    [
        ("if False:\n    return\nallocate(q)", "branch return"),
        ("allocate(q)\nif True:\n    q += 1", "writable raw unsigned QNum"),
        ("if False:\n    allocate(q)\nH(q)", "initializ|width"),
        ("allocate(q)\nfor i in range(3):\n    if False:\n        break", "break"),
    ],
)
def test_classical_branches_preserve_flow_storage_and_assignment_restrictions(
    body: str, diagnostic: str
) -> None:
    with pytest.raises(ERRORS, match=diagnostic):
        synth.synthesize(definition(body))


def test_selected_classical_arm_controls_captured_resource_cost() -> None:
    entry = definition(
        "allocate(2,q)\nif n == 0:\n    H(q[0])\nelse:\n    for i in range(31):\n        H(q[0])\n        CX(q[0],q[1])\n        RZ(0.13,q[0])",
        "n:int,q:Output[QArray[QBit]]",
    )
    compare_states(
        synth.synthesize(entry, classical_inputs={"n": 0}, max_operations=2),
        fq.Circuit(2, dtype=torch.complex128).h(1),
        ("q",),
    )
    with pytest.raises((*ERRORS, PlanningError)):
        synth.synthesize(entry, classical_inputs={"n": 1}, max_operations=2)


def test_shared_total_binding_and_loop_budgets_are_not_reset_by_if_or_helpers() -> None:
    entry = definition("allocate(q)", "first:list[int],second:list[int],q:Output[QBit]")
    with pytest.raises(ERRORS, match="4096"):
        synth.synthesize(
            entry, classical_inputs={"first": [0] * 2048, "second": [1] * 2049}
        )
    for count, diagnostic in ((257, "256 iterations"), (4097, "static integer")):
        entry = definition(
            "allocate(q)\nif n > 0:\n    for i in range(n):\n        H(q)",
            "n:int,q:Output[QBit]",
        )
        with pytest.raises(ERRORS, match=diagnostic):
            synth.synthesize(entry, classical_inputs={"n": count})
    entry = definition(
        "allocate(q)\nfor i in range(17):\n    if True:\n        for j in range(256):\n            H(q)"
    )
    with pytest.raises(ERRORS, match="4096"):
        synth.synthesize(entry)


@pytest.mark.parametrize("mode", ["condition", "branch"])
def test_condition_and_branch_nesting_are_bounded(mode: str) -> None:
    if mode == "condition":
        expression = "not " * 33 + "True"
        body = f"allocate(q)\nif {expression}:\n    H(q)"
    else:
        body = (
            "allocate(q)\n"
            + "\n".join("    " * i + "if True:" for i in range(33))
            + "\n"
            + "    " * 33
            + "H(q)"
        )
    with pytest.raises(ERRORS, match="32"):
        synth.synthesize(definition(body))


def test_integer_arrays_snapshot_and_rebinding_remain_per_program() -> None:
    message = list(MESSAGE)
    inputs = {
        "message": message,
        "alice_bases": list(ALICE_BASES),
        "bob_bases": list(BOB_BASES),
    }
    program = synth.synthesize(prepare_key, classical_inputs=inputs)
    message[0] = 1
    changed = synth.synthesize(prepare_key, classical_inputs=inputs)
    compare_states(program, reference_circuit(), ("qba",))
    compare_states(
        changed, reference_circuit(tuple(message), ALICE_BASES, BOB_BASES), ("qba",)
    )


@pytest.mark.parametrize("case", ["alias", "escape"])
def test_conditional_helpers_preserve_alias_and_output_escape_refusals(
    case: str,
) -> None:
    if case == "alias":
        prelude = "@qfunc\ndef helper(n:int,a:QBit,b:QBit):\n    if n == 1:\n        CX(a,b)\n"
        body = "allocate(q)\nhelper(1,q,q)"
        diagnostic = "disjoint"
    else:
        prelude = "@qfunc\ndef helper(n:int,a:Output[QBit]):\n    if n == 1:\n        allocate(a)\n"
        body = "helper(1,q)"
        diagnostic = "writable quantum ports|publish Outputs"
    with pytest.raises(ERRORS, match=diagnostic):
        synth.synthesize(definition(body, prelude=prelude))


def test_integer_element_as_helper_parameter_and_index_keeps_exact_type() -> None:
    entry = definition(
        "allocate(2,q)\nhelper(indices[0],q)\nH(q[indices[1]])",
        "indices:list[int],q:Output[QArray[QBit]]",
        "@qfunc\ndef helper(index:int,q:QArray[QBit]):\n    X(q[index])\n",
    )
    compare_states(
        synth.synthesize(entry, classical_inputs={"indices": [1, 0]}),
        fq.Circuit(2, dtype=torch.complex128).x(0).h(1),
        ("q",),
    )


@pytest.mark.parametrize("annotation", ['"int"', "Integer", '"Integer"'])
def test_integer_annotations_resolve_builtin_identity(annotation: str) -> None:
    entry = definition(
        "allocate(q)\nif n == 1:\n    X(q)",
        f"n:{annotation},q:Output[QBit]",
        "Integer = int\n",
    )
    compare_states(
        synth.synthesize(entry, classical_inputs={"n": 1}),
        fq.Circuit(1, dtype=torch.complex128).x(0),
        ("q",),
    )


@pytest.mark.parametrize("binding", ["int = bool", "int = float", "int = object"])
def test_shadowed_integer_annotation_is_not_inferred_from_input(binding: str) -> None:
    with pytest.raises((ValueError, *ERRORS)):
        entry = definition(
            "allocate(q)\nhelper(n,q)",
            "n:float,q:Output[QBit]",
            binding
            + "\n@qfunc\ndef helper(n:int,q:QBit):\n    for i in range(n):\n        X(q)\n",
        )
        synth.synthesize(entry, classical_inputs={"n": 1})


def test_classical_array_index_recursion_shares_expression_depth_budget() -> None:
    index = "0"
    for _ in range(35):
        index = f"indices[{index}]"
    entry = definition(
        f"allocate(q)\nif indices[{index}] == 0:\n    H(q)",
        "indices:list[int],q:Output[QBit]",
    )
    with pytest.raises(ERRORS, match="deep|32"):
        synth.synthesize(entry, classical_inputs={"indices": [0]})
