"""Exact native numeric workflows, independent states and full-domain oracles."""

import importlib
import math
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest

import flagquantum as fq
from benchmarks.synthesis.verification import classical_apply, encode
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._construction.lowering import schedule_arithmetic
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_register_workflow import check_state
from tests.team.compiler.test_construction_python_program import program_definition
from tools.synthesis_catalog import load_catalog

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "bitwise_or_2vars_example",
    "bitwise_xor_2vars_example",
    "comparator_2vars_example",
    "maximum_float_example",
    "maximum_integer_example",
    "multiplication_float_example",
    "negation_example",
    "subtraction_2vars_example",
    "subtraction_float_example",
    "arithmetic_expressions",
}


def definition(ports: str, body: str, helper: str = "") -> _QuantumFunction:
    return program_definition(
        "from flagquantum.synth import *\n"
        + helper
        + f"@qfunc\ndef main({ports}):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )


@pytest.mark.parametrize("name", sorted(TARGETS))
def test_originals_native_and_independent_circuit(name: str) -> None:
    case = next(c for c in load_catalog()["cases"] if Path(c["id"]).stem == name)
    module = ".".join(Path(case["directory"]).parts)
    entry = importlib.import_module(module + ".main")
    reference = importlib.import_module(module + ".comparisons.circuit")
    native = entry.build_program()
    assert "qmod_signed_registers" not in native.report
    assert "qmod_fractional_bits" not in native.report
    for source in (native, synth.synthesize(ROOT / case["original"]["path"])):
        compare_states(source, reference.reference_circuit(), tuple(source.registers))
    result = fq.run(native)
    raw = native.probabilities(result, registers=tuple(native.registers))
    if name == "arithmetic_expressions":
        assert dict(raw) == pytest.approx(
            {
                (x, y, int(x >= 2 or (x == 1 and y == 1))): 1 / 8
                for x in range(4)
                for y in range(2)
            }
        )
        return
    numeric = native.numeric_probabilities(result, registers=tuple(native.registers))
    assert sum(raw.values()) == pytest.approx(1)
    assert sum(numeric.values()) == pytest.approx(1)
    assert all(isinstance(value, Fraction) for key in numeric for value in key)
    if name.startswith("maximum"):
        assert {key[-1] for key in numeric} == {Fraction(7, 2), 4, 5, 6, 7}
    elif name == "subtraction_float_example":
        assert dict(raw) == {(4, 1): 1.0}
        assert dict(numeric) == {(4, Fraction(1, 2)): 1.0}
    elif name == "subtraction_2vars_example":
        assert dict(numeric) == {(4, 5, -1): 1.0}
    elif name == "negation_example":
        assert {key[0] for key in numeric} == set(range(-4, 4))
        assert all(b == -a for a, b in numeric)


CASES: tuple[tuple[str, bool, Callable[[int, int], int | Fraction]], ...] = (
    ("a|b", True, lambda a, b: a | b),
    ("a^b", True, lambda a, b: a ^ b),
    ("a==b", True, lambda a, b: int(a == b)),
    ("a!=b", True, lambda a, b: int(a != b)),
    ("-a", True, lambda a, b: -a),
    ("a-b", True, lambda a, b: a - b),
    ("a-3.5", False, lambda a, b: a - Fraction(7, 2)),
    ("3.5*a", False, lambda a, b: Fraction(7, 2) * a),
    ("max(3.5,a)", False, lambda a, b: Fraction(7, 2) if a < Fraction(7, 2) else a),
    ("min(3.5,a)", False, lambda a, b: Fraction(7, 2) if a > Fraction(7, 2) else a),
)


@pytest.mark.parametrize("expression,signed,oracle", CASES)
def test_native_capture_full_domains_nonzero_xor_and_clean_workspace(
    expression: str, signed: bool, oracle: Callable[[int, int], int | Fraction]
) -> None:
    entry = definition(
        f"a:Output[QNum[3,{signed},0]],b:Output[QNum[2]],result:Output[QNum]",
        "allocate(a)\nallocate(b)\nhadamard_transform(a)\nhadamard_transform(b)\n"
        "phase(1,0.37)\nphase(b,0.19)\n" + f"result |= {expression}",
    )
    root, functions = capture_program(entry)
    compute = next(
        op for op in root.body.blocks[0].operations if op.name == "qreg.compute"
    )
    graph = next(f for f in functions if f.name == compute.attributes["callee"])
    declarations = {
        op.attributes["name"]: op
        for op in root.body.blocks[0].operations
        if op.name == "qreg.declare"
    }
    result_format = declarations["result"].attributes.get("numeric_format", (False, 0))
    width = declarations["result"].results[0].type.parameters[0]
    names = tuple(
        next(name for name, op in declarations.items() if op.results[0] == value)
        for value in compute.operands[2:]
    )
    views, bindings, cursor = [], {}, 0
    for name, formal in zip(names, graph.body.blocks[0].arguments, strict=True):
        bits = 3 if name == "a" else 2
        wires = tuple(range(cursor, cursor + bits))
        views.append(wires)
        bindings[formal.id] = wires
        cursor += bits
    output = tuple(range(cursor, cursor + width))
    candidate = schedule_arithmetic(
        (graph,),
        bindings,
        output,
        entry=graph.name,
        n_qubits=96,
        workspace=tuple(range(cursor + width, 96)),
        workspace_state="fresh_zero",
    )
    maximum = (1 << width) - 1
    for words in product(*(range(1 << len(view)) for view in views)):
        values = dict(zip(names, words, strict=True))
        raw_a = values["a"]
        a = raw_a - 8 if signed and raw_a & 4 else raw_a
        answer = oracle(a, values.get("b", 0)) * (1 << result_format[1])
        assert int(answer) == answer
        encoded = int(answer) & maximum
        for target in {0, 1, maximum}:
            initial = encode((*words, target), (*views, output))
            expected = encode((*words, target ^ encoded), (*views, output))
            assert classical_apply(candidate.circuit, initial) == expected
    program = synth.synthesize(entry, memory_limit_bytes=256 * 1024**2)
    expected_amplitudes = {}
    for raw_a, b in product(range(8), range(4)):
        a = raw_a - 8 if signed and raw_a & 4 else raw_a
        encoded = int(oracle(a, b) * (1 << result_format[1])) & maximum
        expected_amplitudes[(raw_a, b, encoded)] = complex(
            math.cos(0.37 + 0.19 * b), math.sin(0.37 + 0.19 * b)
        ) / math.sqrt(32)
    check_state(program, fq.run(program), expected_amplitudes)


@pytest.mark.parametrize("value", [-4, -1, 0, 3])
def test_signed_initialization_and_explicit_raw_xor(value: int) -> None:
    entry = definition("a:Output[QNum[3,True,0]]", f"a |= {value}\na ^= 5")
    program = synth.synthesize(entry)
    result = fq.run(program)
    raw = (value & 7) ^ 5
    expected = raw - 8 if raw & 4 else raw
    assert dict(program.probabilities(result, registers=("a",))) == {(raw,): 1.0}
    assert dict(program.numeric_probabilities(result, registers=("a",))) == {
        (Fraction(expected),): 1.0
    }


def test_scaled_initialization_and_signed_extension() -> None:
    entry = definition("a:Output[QNum[4,True,1]]", "a |= -1.5\na ^= 1")
    program = synth.synthesize(entry)
    assert dict(program.numeric_probabilities(fq.run(program), registers=("a",))) == {
        (Fraction(-2),): 1.0
    }
    entry = definition(
        "a:Output[QNum[3,True,0]],result:Output[QNum[6,True,0]]",
        "allocate(a)\nhadamard_transform(a)\nresult |= -a",
    )
    program = synth.synthesize(entry)
    check_state(
        program,
        fq.run(program),
        {
            (raw, (-(raw if raw < 4 else raw - 8)) & 63): 1 / math.sqrt(8)
            for raw in range(8)
        },
    )


@pytest.mark.parametrize(
    "ports,body,diagnostic",
    [
        ("a:Output[QNum[3,True,0]]", "a |= 4", "initial value"),
        ("a:Output[QNum[3,True,0]]", "a |= -5", "initial value"),
        ("a:Output[QNum[3]]", "a |= -1", "initial value"),
        ("a:Output[QNum[3,True,1]]", "a |= 0.25", "fractional bits"),
        ("a:Output[QNum[3,True,0]]", "a ^= 0", "uninitialized"),
        ("a:Output[QNum[3,True,0]]", "allocate(a)\na ^= 8", "fitting"),
        ("a:Output[QNum[3,True,0]]", "allocate(a)\na ^= -1", "fitting"),
        ("a:Output[QNum[3,True,0]]", "allocate(a)\na ^= True", "fitting"),
        ("a:Output[QNum[3,True,0]]", "allocate(a)\na |= 0", "initialized"),
        ("a:Output[QNum[3,True,0]]", "a |= a", "alias|uninitialized"),
        (
            "a:Output[QNum[3]],result:Output[QNum]",
            "allocate(a)\nresult |= 0.1*a",
            "exactly binary",
        ),
        (
            "a:Output[QNum[3,True,0]],result:Output[QNum]",
            "allocate(a)\nresult |= a+1",
            "signed operands",
        ),
        (
            "a:Output[QNum[3,True,0]]",
            "allocate(a)\nphase(a<0,0.5)",
            "numeric comparisons",
        ),
        (
            "a:Output[QNum[3,False,1]],result:Output[QNum]",
            "allocate(a)\nresult |= a-1",
            "unscaled",
        ),
        (
            "a:Output[QNum[3]],result:Output[QNum[3]]",
            "allocate(a)\nresult |= a-1",
            "numeric formats",
        ),
        (
            "a:Output[QNum[3]],result:Output[QNum]",
            "allocate(a)\nresult |= max(a, 2, 3)",
            "two-argument",
        ),
        (
            "a:Output[QNum[3]],result:Output[QNum]",
            "allocate(a)\nresult |= a**0 - 1",
            "literal exponent",
        ),
        (
            "a:Output[QNum[3]],result:Output[QNum]",
            "allocate(a)\nresult |= a**a - 1",
            "literal exponent",
        ),
    ],
)
def test_numeric_refusals_have_source_locations(
    ports: str, body: str, diagnostic: str
) -> None:
    with pytest.raises(HybridVerificationError, match=diagnostic) as error:
        synth.synthesize(definition(ports, body))
    assert "<up01-" in str(error.value)


def test_helpers_locals_controls_and_shadowed_builtins_refuse() -> None:
    for helper, body in [
        ("@qfunc\ndef helper(a:QNum):\n    hadamard_transform(a)\n", "helper(a)"),
        (
            "@qfunc\ndef helper(a:Const[QNum],result:Output[QNum]):\n    result |= -a\n",
            "helper(a,result)",
        ),
        (
            "",
            "with within():\n    score:QNum[4] = -a\nwith apply():\n    phase(score,0.5)",
        ),
        ("", "control(a==0,lambda:phase(1,0.5))"),
    ]:
        with pytest.raises(HybridVerificationError, match="numeric"):
            synth.synthesize(
                definition(
                    "a:Output[QNum[3,True,0]],result:Output[QNum]",
                    "allocate(a)\n" + body,
                    helper,
                )
            )
    with pytest.raises(HybridVerificationError, match="shadowed"):
        synth.synthesize(
            definition(
                "a:Output[QNum[3]],result:Output[QNum]",
                "allocate(a)\nresult |= max(3.5,a)",
                "def max(a,b):\n    raise AssertionError('host callback ran')\n",
            )
        )


@pytest.mark.parametrize(
    "numeric", [True, (1, 0), (True, True), (False, -1), (True, 4), (True, 0, 0)]
)
def test_compiler_checks_canonical_numeric_metadata(numeric: object) -> None:
    root, functions = capture_program(
        definition("a:Output[QNum[3,True,0]]", "allocate(a)")
    )
    block = root.body.blocks[0]
    declaration = replace(
        block.operations[0], attributes={"name": "a", "numeric_format": numeric}
    )
    malformed = replace(
        root,
        body=replace(
            root.body,
            blocks=(replace(block, operations=(declaration, *block.operations[1:])),),
        ),
    )
    with pytest.raises(HybridVerificationError, match="register.numeric"):
        verify_register_program(malformed, functions)


def test_decoded_readout_is_checked_bounded_and_ignores_report_relabeling() -> None:
    entry = definition("a:Output[QNum],result:Output[QNum]", "a |= 4\nresult |= a-3.5")
    program = synth.synthesize(entry)
    result = fq.run(program)
    changed = replace(program, report={"numeric_formats": (("result", False, 0),)})
    assert dict(changed.numeric_probabilities(result)) == {(Fraction(1, 2),): 1.0}
    with pytest.raises(ValueError, match="budget"):
        program.numeric_probabilities(result, max_bits=1)
    with pytest.raises(ValueError):
        program.numeric_probabilities(result, registers=("missing",))
    unsigned = synth.synthesize(definition("result:Output[QNum]", "result |= 1"))
    with pytest.raises(ValueError, match="verified native numeric"):
        unsigned.numeric_probabilities(fq.run(unsigned))
    with pytest.raises(ValueError, match="identity"):
        program.numeric_probabilities(fq.run(unsigned))
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(entry, max_operations=0)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 1},
        {"max_operations": 1},
        {"max_workspace": 0},
        {"max_depth": 1},
    ],
)
def test_numeric_resource_limits_use_actual_lowered_costs(
    limits: dict[str, int],
) -> None:
    entry = definition(
        "a:Output[QNum[3,True,0]],result:Output[QNum]",
        "allocate(a)\nhadamard_transform(a)\nresult |= -a",
    )
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            entry,
            max_qubits=limits.get("max_qubits", 256),
            max_operations=limits.get("max_operations", 100000),
            max_workspace=limits.get("max_workspace"),
            max_depth=limits.get("max_depth"),
        )


def test_binary_literals_and_typed_extrema_never_execute_on_host() -> None:
    for marker in (synth.min, synth.max):
        with pytest.raises(TypeError, match="captured"):
            marker(1.5, 2.0)
    for expression, expected in [
        ("-1", Fraction(-1)),
        ("-0.125", Fraction(-1, 8)),
        ("0.125", Fraction(1, 8)),
    ]:
        program = synth.synthesize(
            definition("result:Output[QNum]", "result |= " + expression)
        )
        assert dict(program.numeric_probabilities(fq.run(program))) == {
            (expected,): 1.0
        }
    for imported in ("from builtins import min", "from builtins import max"):
        expression = "min(3.5,a)" if imported.endswith("min") else "max(3.5,a)"
        program = synth.synthesize(
            definition(
                "a:Output[QNum[3]],result:Output[QNum]",
                "allocate(a)\nhadamard_transform(a)\nresult |= " + expression,
                imported + "\n",
            )
        )
        assert sum(
            program.numeric_probabilities(fq.run(program)).values()
        ) == pytest.approx(1)


@pytest.mark.parametrize(
    "annotation",
    [
        "QNum[3,1,0]",
        "QNum[3,True,True]",
        "QNum[3,False,4]",
        "QNum[3,False,-1]",
        "QNum[True,True,0]",
        "Annotated[QNum,(3,True,4)]",
        "Annotated[QNum,(3,1,0)]",
    ],
)
def test_numeric_annotation_validation_is_not_coercive(annotation: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                f"a:Output[{annotation}]",
                "allocate(a)",
                "from typing import Annotated\n",
            )
        )


def test_numeric_metadata_cannot_share_record_fields_and_memory_is_bounded() -> None:
    root, functions = capture_program(
        definition("a:Output[QNum[3,True,0]]", "allocate(a)")
    )
    block = root.body.blocks[0]
    declaration = replace(
        block.operations[0],
        attributes={
            "name": "a",
            "numeric_format": (True, 0),
            "record_type": "Record",
            "fields": (("field", "qnum", 3),),
        },
    )
    malformed = replace(
        root,
        body=replace(
            root.body,
            blocks=(replace(block, operations=(declaration, *block.operations[1:])),),
        ),
    )
    with pytest.raises(HybridVerificationError, match="register.numeric"):
        verify_register_program(malformed, functions)
    with pytest.raises(ValueError, match="memory"):
        synth.synthesize(
            definition(
                "a:Output[QNum[3,True,0]],result:Output[QNum]",
                "allocate(a)\nhadamard_transform(a)\nresult |= -a",
            ),
            memory_limit_bytes=1,
        )


@pytest.mark.parametrize(
    "imports,expression",
    [
        ("from flagquantum import synth", "synth.max(3.5,a)"),
        ("from flagquantum.synth import max as maximum", "maximum(3.5,a)"),
    ],
)
def test_typed_extrema_use_import_identity(imports: str, expression: str) -> None:
    program = synth.synthesize(
        definition(
            "a:Output[QNum[3]],result:Output[QNum]",
            "allocate(a)\nhadamard_transform(a)\nresult |= " + expression,
            imports + "\n",
        )
    )
    assert dict(program.numeric_probabilities(fq.run(program))) == pytest.approx(
        {(Fraction(7, 2),): 0.5, **{(Fraction(value),): 0.125 for value in range(4, 8)}}
    )
