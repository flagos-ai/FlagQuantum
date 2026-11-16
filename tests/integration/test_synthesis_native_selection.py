"""Native conditional values preserve coherence through shared scalar selection."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.arithmetic.extremum.minimum_2vars_example.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.functions.arithmetic.extremum.minimum_2vars_example.main import (
    build_program,
)
from examples.synthesis.functions.arithmetic.extremum.minimum_2vars_example.main import (
    main as minimum,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "tests/fixtures/classiq_qmod/corpus/original/functions/"
    "function_usage_examples/arithmetic/extremum/minimum_2vars_example.qmod"
)


def truth_table(selector: Callable[[int, int], int]) -> torch.Tensor:
    """Build a classical XOR permutation independently of compiler expression IR."""
    matrix = torch.zeros((64, 64), dtype=torch.complex128)
    for column in range(64):
        a, b = column >> 4, (column >> 2) & 3
        matrix[column ^ selector(a, b), column] = 1
    return matrix


@pytest.mark.parametrize(
    ("expression", "selector"),
    [
        ("a if a < b else b", min),
        ("a if a >= b else b", max),
        ("a if 1 else b", lambda a, b: a),
        ("a if 0 else b", lambda a, b: b),
        ("3 if a == b else 1", lambda a, b: 3 if a == b else 1),
        (
            "a if a < b else (b if b < 2 else 2)",
            lambda a, b: a if a < b else min(b, 2),
        ),
    ],
)
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_conditional_tables_on_coherent_inputs_preserve_absolute_phase(
    expression: str,
    selector: Callable[[int, int], int],
    objective: Literal["width", "depth"],
) -> None:
    entry = program_definition(
        "from flagquantum.synth import phase\n"
        "@qfunc\ndef main(a: Output[QNum[2]],b: Output[QNum[2]],res: Output[QNum]):\n"
        "    allocate(a)\n    allocate(b)\n    hadamard_transform(a)\n"
        "    hadamard_transform(b)\n    phase(1,0.37)\n    phase(a,-0.29)\n"
        f"    res |= {expression}\n"
    )
    reference = fq.Circuit(6, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.any(
        0,
        matrix=torch.eye(2, dtype=torch.complex128)
        * torch.exp(torch.tensor(0.37j, dtype=torch.complex128)),
    )
    reference.any(
        0,
        1,
        matrix=torch.diag(torch.exp(-0.29j * torch.arange(4, dtype=torch.float64))),
    )
    compare_states(
        synth.synthesize(entry, objective=objective),
        reference.any(*range(6), matrix=truth_table(selector)),
        ("a", "b", "res"),
    )


def test_unchanged_minimum_original_and_native_match_an_independent_circuit() -> None:
    for program in (build_program(), synth.synthesize(SOURCE)):
        compare_states(program, reference_circuit(), ("a", "b", "res"))


@pytest.mark.parametrize(
    "expression",
    [
        "a if 3 else b",
        "a if 1 else host()",
        "a if a else b",
        "a if missing else b",
        "a if a < b else host()",
        "a if a < b else -1",
    ],
)
def test_conditional_values_reject_unsupported_predicates_names_and_effects(
    expression: str,
) -> None:
    entry = program_definition(
        "def host():\n    raise AssertionError('host execution')\n"
        "@qfunc\ndef main(a: Output[QNum[2]],b: Output[QNum[2]],res: Output[QNum]):\n"
        "    allocate(a)\n    allocate(b)\n"
        f"    res |= {expression}\n"
    )
    with pytest.raises(HybridVerificationError):
        synth.synthesize(entry)


def test_conditional_result_still_requires_fresh_output_storage() -> None:
    entry = program_definition(
        "@qfunc\ndef main(a: Output[QNum[2]],b: Output[QNum[2]],res: Output[QNum]):\n"
        "    allocate(a)\n    allocate(b)\n    res |= 1\n"
        "    res |= a if a < b else b\n"
    )
    with pytest.raises(HybridVerificationError):
        synth.synthesize(entry)


def test_conditional_program_reuses_existing_resource_budgets() -> None:
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(minimum, max_operations=1)


def test_conditional_values_compose_through_a_checked_quantum_helper() -> None:
    entry = program_definition(
        "from flagquantum.synth import Const\n"
        "@qfunc\ndef choose(a: Const[QNum[2]],b: Const[QNum[2]],res: Output[QNum]):\n"
        "    res |= a if a <= b else b\n"
        "@qfunc\ndef main(a: Output[QNum[2]],b: Output[QNum[2]],res: Output[QNum]):\n"
        "    allocate(a)\n    allocate(b)\n    hadamard_transform(a)\n"
        "    hadamard_transform(b)\n    choose(a,b,res)\n"
    )
    reference = fq.Circuit(6, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    compare_states(
        synth.synthesize(entry),
        reference.any(*range(6), matrix=truth_table(min)),
        ("a", "b", "res"),
    )


def test_conditional_helper_rejects_result_input_aliasing() -> None:
    entry = program_definition(
        "from flagquantum.synth import Const\n"
        "@qfunc\ndef choose(a: Const[QNum[2]],b: Const[QNum[2]],res: Output[QNum[2]]):\n"
        "    res |= a if a < b else b\n"
        "@qfunc\ndef main(a: Output[QNum[2]],b: Output[QNum[2]]):\n"
        "    allocate(a)\n    allocate(b)\n    choose(a,b,a)\n"
    )
    with pytest.raises(HybridVerificationError):
        synth.synthesize(entry)


def test_conditional_result_uses_the_wider_branch_width() -> None:
    entry = program_definition(
        "@qfunc\ndef main(a: Output[QNum[1]],res: Output[QNum]):\n"
        "    allocate(a)\n    hadamard_transform(a)\n    res |= a if a else 7\n"
    )
    reference = fq.Circuit(4, dtype=torch.complex128).h(0)
    columns = torch.arange(16)
    rows = columns ^ torch.where(columns >= 8, 1, 7)
    matrix = torch.zeros((16, 16), dtype=torch.complex128)
    matrix[rows, columns] = 1
    program = synth.synthesize(entry)
    assert len(program.registers["res"]) == 3
    compare_states(program, reference.any(*range(4), matrix=matrix), ("a", "res"))
