"""Whole Python qfunc workflows share preparation, arithmetic and checked run/read."""

from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.composition.python_program.comparisons.circuit import (
    circuit_reference,
)
from examples.synthesis.tutorials.composition.python_program.main import main
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.synthesis_workflow_helpers import check_state
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
QMOD = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/example_references/tutorials/composition/python_program/comparisons/source.qmod"
).read_text(encoding="utf-8")


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_native_program_and_qmod_have_the_same_full_coherent_state(
    objective: Literal["width", "depth"],
) -> None:
    expected = {(x, y, x * x + y): 0.25 for x in range(4) for y in range(4)}
    programs = [
        synth.synthesize(main, max_qubits=24, objective=objective),
        synth.synthesize(
            QMOD, source_format="qmod", max_qubits=24, objective=objective
        ),
    ]
    for p in programs:
        result = fq.run(p)
        assert tuple(p.registers) == ("x", "y", "result")
        assert len(p.registers["result"]) == 4
        check_state(p, result, expected)
        assert dict(
            p.probabilities(result, registers=("x", "y", "result"))
        ) == pytest.approx(dict.fromkeys(expected, 1 / 16), abs=1e-12)
        assert p.report["memory_limit_bytes"] == 512 * 1024**2
        assert p.report["memory_limit_origin"] == "profile_default"
    assert programs[0].report["source_format"] == "python"
    reference = fq.run(circuit_reference()).statevector().reshape(4, 4, 16, 2)
    oracle = torch.zeros_like(reference)
    for x, y, z in expected:
        oracle[x, y, z, 0] = expected[x, y, z]
    torch.testing.assert_close(reference, oracle, atol=1e-12, rtol=0)


@pytest.mark.parametrize("x", range(4))
@pytest.mark.parametrize("y", range(4))
def test_all_sixteen_basis_inputs_use_declared_domains_and_clean_work(
    x: int, y: int
) -> None:
    source = f"""@qfunc
def energy(x: Const[QNum],y: Const[QNum],result: Output[QNum]):
    result |= x*x+y
@qfunc
def main(x: Output[QNum[2]],y: Output[QNum[2]],result: Output[QNum]):
    x |= {x}
    y |= {y}
    energy(x,y,result)
"""
    p = synth.synthesize(program_definition(source))
    assert len(p.registers["result"]) == 4
    check_state(p, fq.run(p), {(x, y, x * x + y): 1})


def test_composition_keeps_interference_and_result_mutation_after_clean_compute() -> (
    None
):
    source = """@qfunc
def energy(x: Const[QNum],y: Const[QNum],result: Output[QNum]):
    result |= x*x+y
@qfunc
def main(x: Output[QNum],y: Output[QNum],result: Output[QNum]):
    allocate(2,x)
    allocate(2,y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x,y,result)
    hadamard_transform(result)
"""
    p = synth.synthesize(program_definition(source))
    expected = {
        (x, y, z): (-1) ** ((z & (x * x + y)).bit_count()) / 16
        for x in range(4)
        for y in range(4)
        for z in range(16)
    }
    check_state(p, fq.run(p), expected)
    reference = circuit_reference()
    for wire in range(4, 8):
        reference.h(wire)
    state = fq.run(reference).statevector().reshape(4, 4, 16, 2)
    oracle = torch.zeros_like(state)
    for (x, y, z), amplitude in expected.items():
        oracle[x, y, z, 0] = amplitude
    torch.testing.assert_close(state, oracle, atol=1e-12, rtol=0)


def test_const_aliasing_and_repeated_generic_specializations() -> None:
    source = """@qfunc
def energy(x: Const[QNum],y: Const[QNum],result: Output[QNum]):
    result |= x*x+y
@qfunc
def main(x: Output[QNum[2]],y: Output[QNum[1]],a: Output[QNum],b: Output[QNum]):
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x,x,a)
    energy(y,x,b)
"""
    p = synth.synthesize(program_definition(source), max_qubits=32)
    assert (len(p.registers["a"]), len(p.registers["b"])) == (4, 3)
    check_state(
        p,
        fq.run(p),
        {
            (x, y, x * x + x, y * y + x): 1 / (8**0.5)
            for x in range(4)
            for y in range(2)
        },
    )


def test_aliases_closures_and_star_import_remain_familiar() -> None:
    from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc

    @qfunc
    def prepare(q: Output[QNum]) -> None:
        allocate(2, q)
        hadamard_transform(q)

    @qfunc
    def root(q: Output[QNum]) -> None:
        prepare(q)

    p = synth.synthesize(root)
    check_state(p, fq.run(p), {(q,): 0.5 for q in range(4)})
    source = """from flagquantum.synth import *
from flagquantum.synth import allocate as alloc, hadamard_transform as uniform
@qfunc
def main(q: Output[QNum]):
    alloc(2,q)
    uniform(q)
"""
    p = synth.synthesize(program_definition(source))
    check_state(p, fq.run(p), {(q,): 0.5 for q in range(4)})


@pytest.mark.parametrize(
    "option,value",
    [
        ("inputs", {}),
        ("input_state", synth.InputState.uniform()),
        ("constants", {}),
        ("functions", (main,)),
        ("output_mode", "phase"),
        ("preparation", fq.Circuit(1).to_ir()),
    ],
)
def test_root_rejects_duplicate_scalar_state_configuration(
    option: str, value: object
) -> None:
    with pytest.raises(ValueError, match="typed qfunc programs own"):
        synth.synthesize(main, **{option: value})


def test_real_plan_budget_errors_and_routed_named_readout() -> None:
    source = """@qfunc
def main(x: Output[QBit],result: Output[QBit]):
    allocate(x)
    hadamard_transform(x)
    result |= x
"""
    simple = program_definition(source)
    with pytest.raises(PlanningError, match="memory budget"):
        synth.synthesize(simple, memory_limit_bytes=16)
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(simple, max_qubits=1)
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(simple, max_operations=0)
    routed = synth.synthesize(simple, topology="line", memory_limit_bytes=1024**2)
    check_state(routed, fq.run(routed), {(0, 0): 2**-0.5, (1, 1): 2**-0.5})
    assert routed.report["memory_limit_origin"] == "explicit"


def test_const_root_is_not_an_executable_input_state() -> None:
    source = """@qfunc
def main(x: Const[QNum[2]]):
    hadamard_transform(x)
"""
    definition = program_definition(source)
    with pytest.raises(HybridVerificationError, match="Output-only"):
        synth.synthesize(definition)


def test_definition_bindings_freeze_at_synthesis_not_execution() -> None:
    source = """@qfunc
def prepare(q: Output[QNum]):
    allocate(2,q)
    hadamard_transform(q)
@qfunc
def main(q: Output[QNum]):
    prepare(q)
"""
    definition = program_definition(source)
    before = synth.synthesize(definition)
    replacement = program_definition(
        """@qfunc
def prepare(q: Output[QNum[2]]):
    q |= 3
""",
        entry="prepare",
    )
    from types import FunctionType

    assert isinstance(definition.function, FunctionType)
    definition.function.__globals__["prepare"] = replacement
    after = synth.synthesize(definition)
    assert before.report["source_identities"] != after.report["source_identities"]
    assert before.execution.identity != after.execution.identity
    check_state(before, fq.run(before), {(q,): 0.5 for q in range(4)})
    check_state(after, fq.run(after), {(3,): 1})


def test_scoped_quoted_type_annotations_are_captured_without_evaluation() -> None:
    from flagquantum.synth import Output as Out
    from flagquantum.synth import QNum as Number
    from flagquantum.synth import allocate as alloc
    from flagquantum.synth import qfunc

    @qfunc
    def root(q: "Out[Number]") -> None:
        alloc(2, q)

    p = synth.synthesize(root)
    check_state(p, fq.run(p), {(0,): 1})
