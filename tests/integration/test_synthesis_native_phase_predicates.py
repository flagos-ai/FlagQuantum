"""Bounded native phase predicates match frozen Grover originals and references."""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.grover.grover.main import (
    build_program as build_graph_grover,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.grover_operator.main import (
    build_program as build_product_grover,
)
from examples.synthesis.functions.grover_operator.main import (
    main as product_grover_main,
)
from examples.synthesis.tutorials.workshops.oracle_workshop.oracles_workshop.main import (
    build_program as build_difference_grover,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_polynomial_phase import definition

pytestmark = pytest.mark.integration
ERRORS = (synth.SynthesisError, HybridVerificationError)
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus/original"


def _reference(marked: set[int], width: int, iterations: int) -> fq.Circuit:
    size = 1 << width
    diagonal = torch.ones(size, dtype=torch.complex128)
    diagonal[list(marked)] = -1
    reflection = torch.full((size, size), 2 / size, dtype=torch.complex128) - torch.eye(
        size, dtype=torch.complex128
    )
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        circuit.h(wire)
    for _ in range(iterations):
        circuit.any(*range(width), matrix=torch.diag(diagonal))
        circuit.any(*range(width), matrix=reflection)
    return circuit


def _product_marked() -> set[int]:
    return {
        x * 4 + y for x in range(4) for y in range(4) if x + y < 4 and (x * y) % 4 == 2
    }


def _difference_marked() -> set[int]:
    return {x * 4 + y for x in range(4) for y in range(4) if x - y == 2}


def _graph_marked() -> set[int]:
    edges = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))
    return {
        basis
        for basis in range(32)
        if sum(((basis >> left) ^ (basis >> right)) & 1 for left, right in edges) >= 4
    }


@pytest.mark.parametrize(
    ("native", "reference", "source", "registers"),
    [
        (
            build_product_grover,
            lambda: _reference(_product_marked(), 4, 2),
            "functions/qmod_library_reference/classiq_open_library/grover_operator/grover_operator.qmod",
            ("vars.x", "vars.y"),
        ),
        (
            build_difference_grover,
            lambda: _reference(_difference_marked(), 4, 2),
            "tutorials/workshops/oracle_workshop/oracles_workshop.qmod",
            ("vars.x", "vars.y"),
        ),
    ],
)
def test_fixed_predicate_grovers_match_original_and_independent_circuit(
    native: Callable[[], synth.Program],
    reference: Callable[[], fq.Circuit],
    source: str,
    registers: tuple[str, ...],
) -> None:
    native_program = native()
    original = synth.synthesize(CORPUS / source)
    compare_states(
        native_program,
        reference(),
        tuple(name.removeprefix("vars.") for name in registers),
    )
    compare_states(original, reference(), registers)
    assert native_program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_graph_predicate_grover_matches_original_and_independent_circuit(
    iterations: int,
) -> None:
    native = build_graph_grover(iterations)
    original = synth.synthesize(
        CORPUS / "algorithms/search_and_optimization/grover/grover.qmod",
        classical_inputs={"r": iterations},
    )
    reference = _reference(_graph_marked(), 5, iterations)
    compare_states(native, reference, ("q",))
    compare_states(original, reference, ("nodes",))
    assert native.report["workspace_qubits"] == 0


def test_predicate_truth_tables_include_product_modulo_negative_difference_and_boundary() -> (
    None
):
    assert _product_marked() == {6, 9}
    assert _difference_marked() == {8, 13}
    assert len(_graph_marked()) == 10
    assert 0 not in _graph_marked() and 6 in _graph_marked()


def test_root_and_borrowed_helper_predicate_paths_agree() -> None:
    entry = definition(
        "allocate(2,a)\nallocate(2,b)\nhadamard_transform(a)\nhadamard_transform(b)\n"
        "control((a-b)==2,lambda:phase(1,0.37))\nhelper(a,b)",
        "a:Output[QNum[2]],b:Output[QNum[2]]",
        "@qfunc\ndef helper(a:QNum,b:QNum):\n    control((a-b)==2,lambda:phase(1,0.23))\n",
    )
    program = synth.synthesize(entry, max_workspace=0)
    phases = torch.tensor(
        [0.37 * (a - b == 2) + 0.23 * (a - b == 2) for a in range(4) for b in range(4)],
        dtype=torch.float64,
    )
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.any(0, 1, 2, 3, matrix=torch.diag(torch.exp(1j * phases)))
    compare_states(program, reference, ("a", "b"))


@pytest.mark.parametrize(
    "predicate",
    ["(a==0) ^ (a==0)", "(a==0) & (a==0)"],
)
def test_false_or_canceled_predicates_do_not_hide_uninitialized_reads(
    predicate: str,
) -> None:
    entry = definition(
        "allocate(a)\ncontrol((" + predicate + ") & (b<1),lambda:phase(1,0.2))",
        "a:Output[QBit],b:Output[QNum[2]]",
    )
    with pytest.raises(ERRORS, match="initializ"):
        synth.synthesize(entry)


@pytest.mark.parametrize(
    ("expression", "predicate"),
    [
        ("(a&3)==1", lambda a: (a & 3) == 1),
        ("(a|1)==3", lambda a: (a | 1) == 3),
        ("(a^3)==0", lambda a: (a ^ 3) == 0),
        ("(a%2)==1", lambda a: (a % 2) == 1),
    ],
)
def test_unsigned_bitwise_and_power_of_two_modulo_predicates_are_supported(
    expression: str, predicate: Callable[[int], bool]
) -> None:
    entry = definition(
        "allocate(2,a)\nhadamard_transform(a)\ncontrol("
        + expression
        + ",lambda:phase(1,0.37))",
        "a:Output[QNum[2]]",
    )
    program = synth.synthesize(entry, max_workspace=0)
    mask = [predicate(a) for a in range(4)]
    phases = torch.tensor(
        [0.37 if active else 0.0 for active in mask], dtype=torch.float64
    )
    reference = fq.Circuit(2, dtype=torch.complex128)
    reference.h(0).h(1).any(0, 1, matrix=torch.diag(torch.exp(1j * phases)))
    compare_states(program, reference, ("a",))


@pytest.mark.parametrize(
    "body",
    [
        "allocate(2,a)\ncontrol((a%3)==0,lambda:phase(1,0.2))",
        "allocate(2,a)\ncontrol((a%0)==0,lambda:phase(1,0.2))",
        "allocate(2,a)\ncontrol(((a-3)&1)==0,lambda:phase(1,0.2))",
        "allocate(2,a)\nphase(a&1,0.2)",
        "allocate(2,a)\nphase(a%2,0.2)",
        "allocate(2,a)\ncontrol((a==0) and (a==1),lambda:phase(1,0.2))",
        "allocate(2,a)\ncontrol(not (a==0),lambda:phase(1,0.2))",
    ],
)
def test_unsupported_predicate_forms_fail_closed(body: str) -> None:
    with pytest.raises(ERRORS):
        synth.synthesize(definition(body, "a:Output[QNum[2]]"))


def test_predicate_input_bound_is_checked_before_expansion() -> None:
    with pytest.raises(ERRORS, match="eight input bits"):
        synth.synthesize(
            definition(
                "allocate(9,a)\ncontrol(a<3,lambda:phase(1,0.2))",
                "a:Output[QNum[9]]",
            )
        )


@pytest.mark.parametrize(
    ("constraint", "value"),
    [("max_qubits", 3), ("max_operations", 1), ("memory_limit_bytes", 16)],
)
def test_native_product_search_obeys_actual_resource_ceilings(
    constraint: str, value: int
) -> None:
    with pytest.raises((*ERRORS, PlanningError)):
        if constraint == "max_qubits":
            synth.synthesize(product_grover_main, max_qubits=value)
        elif constraint == "max_operations":
            synth.synthesize(product_grover_main, max_operations=value)
        else:
            synth.synthesize(product_grover_main, memory_limit_bytes=value)
