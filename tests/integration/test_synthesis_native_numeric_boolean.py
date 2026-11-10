"""Bounded raw-word synthesis preserves exact numeric frontend semantics."""

import cmath
from collections.abc import Callable
from fractions import Fraction
from itertools import product

import pytest

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from flagquantum import synth
from flagquantum.compiler._construction.boolean_lowering import lower_boolean_value
from flagquantum.compiler._construction.frontend_program import capture_program
from tests.integration.test_synthesis_native_exact_numbers import definition
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration

CASES: tuple[tuple[str, Callable[[Fraction, int], Fraction | int]], ...] = (
    ("a-b", lambda a, b: a - b),
    ("-a", lambda a, b: -a),
    ("a+0.5", lambda a, b: a + Fraction(1, 2)),
    ("a*b", lambda a, b: a * b),
    ("max(a+b,-0.5)", lambda a, b: max(a + b, Fraction(-1, 2))),
    ("min(1.5,a-b)", lambda a, b: min(Fraction(3, 2), a - b)),
    ("a<b", lambda a, b: int(a < b)),
    ("a>=b", lambda a, b: int(a >= b)),
)


@pytest.mark.parametrize("expression,oracle", CASES)
def test_exact_domains_arbitrary_xor_and_absolute_coherent_phases(
    expression: str, oracle: Callable[[Fraction, int], Fraction | int]
) -> None:
    entry = definition(
        "a:Output[QNum[2,True,1]],b:Output[QNum[2]],result:Output[QNum]",
        "allocate(a)\nallocate(b)\nhadamard_transform(a)\nhadamard_transform(b)\n"
        "phase(1,0.37)\n" + f"result |= {expression}",
    )
    root, functions = capture_program(entry)
    declarations = {
        op.attributes["name"]: op
        for op in root.body.blocks[0].operations
        if op.name == "qreg.declare"
    }
    compute = next(
        op for op in root.body.blocks[0].operations if op.name == "qreg.compute"
    )
    graph = next(f for f in functions if f.name == compute.attributes["callee"])
    width = graph.body.blocks[0].operations[-1].operands[0].type.parameters[0]
    names = tuple(
        next(name for name, op in declarations.items() if op.results[0] == value)
        for value in compute.operands[2:]
    )
    inputs = {
        arg.id: (2 * i, 2 * i + 1)
        for i, arg in enumerate(graph.body.blocks[0].arguments)
    }
    offset = 2 * len(inputs)
    output = tuple(range(offset, offset + width))
    candidate = lower_boolean_value(
        (graph,),
        inputs,
        output,
        entry=graph.name,
        n_qubits=offset + width + 4,
        workspace=tuple(range(offset + width, offset + width + 4)),
        workspace_state="fresh_zero",
    )
    fraction = declarations["result"].attributes["numeric_format"][1]
    for words in product(*(range(4) for _ in names)):
        values = dict(zip(names, words, strict=True))
        a = values["a"]
        exact = oracle(Fraction(a - 4 if a >= 2 else a, 2), values.get("b", 0)) * (
            1 << fraction
        )
        assert int(exact) == exact
        for target in range(1 << width):
            initial = encode((*words, target), (*inputs.values(), output))
            expected = encode(
                (*words, target ^ (int(exact) & ((1 << width) - 1))),
                (*inputs.values(), output),
            )
            assert classical_apply(candidate.circuit, initial) == expected
    program = synth.synthesize(entry, max_qubits=24, memory_limit_bytes=512 * 1024**2)
    expected_amplitudes = {}
    for a, b in product(range(4), range(4)):
        value = oracle(Fraction(a - 4 if a >= 2 else a, 2), b)
        encoded = int(value * (1 << fraction)) & ((1 << width) - 1)
        expected_amplitudes[(a, b, encoded)] = cmath.exp(0.37j) / 4
    check_state(program, fq.run(program), expected_amplitudes)


def test_affine_max_fits_bounded_memory_without_quantizing() -> None:
    entry = definition(
        "x:Output[QNum[2]],y:Output[QNum[2]],result:Output[QNum[5,False,2]]",
        "allocate(x)\nallocate(y)\nhadamard_transform(x)\nhadamard_transform(y)\n"
        "result |= max(x*1.0,x*0.75+y*0.75-1.25)",
    )
    program = synth.synthesize(
        entry, max_qubits=11, max_workspace=2, memory_limit_bytes=2**20
    )
    assert program.circuit.n_wires == 11
    assert program.report["workspace_qubits"] == 2
    expected = {
        (a, b, int(4 * max(Fraction(a), Fraction(3 * (a + b) - 5, 4)))): 0.25 + 0j
        for a, b in product(range(4), range(4))
    }
    check_state(program, fq.run(program), expected)
    for workspace, qubits, operations in ((1, 11, 1000), (2, 10, 1000), (2, 11, 10)):
        with pytest.raises(synth.SynthesisError):
            synth.synthesize(
                entry,
                memory_limit_bytes=2**20,
                max_workspace=workspace,
                max_qubits=qubits,
                max_operations=operations,
            )
