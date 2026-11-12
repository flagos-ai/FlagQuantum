"""Two frontend consumers can replace shared expansion without changing capture."""

from collections.abc import Callable
from typing import NoReturn, TypeVar

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._construction import frontend_phase
from flagquantum.compiler._phase_polynomial import Location, Polynomial
from flagquantum.ecosystem._qmod import phase as qmod_phase
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.unit
Node = TypeVar("Node")


def truth_table_expansion(
    expression: Node,
    quantum_word: Callable[[Node], tuple[Location, ...] | None],
    number: Callable[[Node], float],
    *,
    operator_of: Callable[[Node], str],
    operands_of: Callable[[Node], tuple[Node, ...]],
    fail: Callable[[Node, str, str], NoReturn],
    condition: Node | None = None,
    **options: object,
) -> tuple[Polynomial, tuple[Location, ...]]:
    """Independent two-bit test implementation; never used in production."""
    reads: list[Location] = []

    def evaluate(node: Node, basis: int) -> float:
        word = quantum_word(node)
        if word is not None:
            for location in word:
                if location not in reads:
                    reads.append(location)
            assert len(reads) <= 2
            return float(
                sum(
                    ((basis >> reads.index(location)) & 1) << bit
                    for bit, location in enumerate(word)
                )
            )
        children = operands_of(node)
        if not children:
            return number(node)
        values = tuple(evaluate(child, basis) for child in children)
        symbol = operator_of(node)
        if len(values) == 1 and symbol == "-":
            return -values[0]
        first, second = values
        comparisons = {
            "==": first == second,
            "!=": first != second,
            "<": first < second,
            "<=": first <= second,
            ">": first > second,
            ">=": first >= second,
        }
        if symbol in comparisons:
            return float(comparisons[symbol])
        if symbol == "+":
            return first + second
        if symbol == "-":
            return first - second
        if symbol == "*":
            return first * second
        if symbol == "/" and second:
            return first / second
        fail(node, "test", "unsupported replacement-test operation")

    if condition is not None:
        evaluate(condition, 0)
    evaluate(expression, 0)
    values = [
        evaluate(expression, basis)
        * (evaluate(condition, basis) if condition is not None else 1.0)
        for basis in range(1 << len(reads))
    ]
    for bit in range(len(reads)):
        for basis in range(len(values)):
            if basis & (1 << bit):
                values[basis] -= values[basis ^ (1 << bit)]
    return {mask: weight for mask, weight in enumerate(values) if weight}, tuple(reads)


@pytest.mark.parametrize("guarded", [False, True])
def test_qmod_and_python_consumers_accept_an_independent_expansion_replacement(
    monkeypatch: pytest.MonkeyPatch,
    guarded: bool,
) -> None:
    native_effect = "phase(0.3+q[0]/2-q[0]*q[1],0.37)"
    source_effect = native_effect + ";"
    if guarded:
        native_effect = "control(q[0]==1,lambda:" + native_effect + ")"
        source_effect = "control(q[0]==1) {" + source_effect + "}"
    native = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
        "    allocate(2,q)\n    hadamard_transform(q)\n"
        "    " + native_effect + "\n"
    )
    source = (
        "qfunc main(output q:qbit[2]) {allocate(q);hadamard_transform(q);"
        + source_effect
        + "}"
    )
    values = torch.tensor(
        [0.0 if guarded else 0.3, 0.8, 0.0 if guarded else 0.3, -0.2],
        dtype=torch.float64,
    )
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .h(0)
        .h(1)
        .any(0, 1, matrix=torch.diag(torch.exp(0.37j * values)))
    )
    baseline = [
        synth.synthesize(native),
        synth.synthesize(source, source_format="qmod"),
    ]
    monkeypatch.setattr(frontend_phase, "expand_phase", truth_table_expansion)
    monkeypatch.setattr(qmod_phase, "_expand_phase", truth_table_expansion)
    replacements = [
        synth.synthesize(native),
        synth.synthesize(source, source_format="qmod"),
    ]
    for before, after in zip(baseline, replacements, strict=True):
        compare_states(after, reference, ("q",))
        assert after.report["resources"] == before.report["resources"]
