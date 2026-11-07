"""One-bit logic is coherent, eager and built from existing arithmetic."""

from typing import Literal

import pytest

import flagquantum as fq
from examples.synthesis.tutorials.qmod.qmod_bit_predicates.main import SOURCE
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = pytest.mark.integration
CASES = {
    "x&y": lambda x, y: x & y,
    "x|y": lambda x, y: x | y,
    "x^y": lambda x, y: x ^ y,
    "~x": lambda x, y: 1 - x,
    "x^x": lambda x, y: 0,
    "x|x": lambda x, y: x,
    "x&x": lambda x, y: x,
    "~(x|y)": lambda x, y: 1 - (x | y),
    "x|y&x": lambda x, y: x | (y & x),
    "x^y|x": lambda x, y: (x ^ y) | x,
}


@pytest.mark.parametrize("expression", CASES)
@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("phase", [False, True])
def test_logic_truth_tables_coherent_reference_and_clean_work(
    expression: str,
    objective: Literal["width", "depth"],
    phase: bool,
) -> None:
    outputs = "output x:qbit,output y:qbit,output ref:qbit" + (
        ",output result:qnum" if not phase else ""
    )
    action = f"phase({expression},pi);" if phase else f"result={expression};"
    source = f"qfunc main({outputs}) {{allocate(x);H(x);S(x);allocate(y);H(y);ref=x;{action}}}"
    p = synth.synthesize(source, source_format="qmod", objective=objective)
    if phase:
        expected = {
            (x, y, x): 1j**x * (-1) ** CASES[expression](x, y) / 2
            for x in range(2)
            for y in range(2)
        }
    else:
        expected = {
            (x, y, x, CASES[expression](x, y)): 1j**x / 2
            for x in range(2)
            for y in range(2)
        }
    check_state(p, fq.run(p), expected)


@pytest.mark.parametrize("expression", ["~0", "~1", "0&1", "0|1", "1^1"])
def test_classical_bit_literals_fold(expression: str) -> None:
    expected = {"~0": 1, "~1": 0, "0&1": 0, "0|1": 1, "1^1": 0}[expression]
    p = synth.synthesize(
        f"qfunc main(output result:qnum) {{result={expression};}}", source_format="qmod"
    )
    check_state(p, fq.run(p), {(expected,): 1})


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("x&1", 0), ("x|0", 0), ("~x", 3), ("x^1", 1), ("2&1", 0)],
)
def test_multibit_width_is_independent_of_initial_value(expression, expected):
    program = synth.synthesize(
        f"qfunc main(output x:qnum<2>,output result:qnum) {{x=0;result={expression};}}",
        source_format="qmod",
    )
    check_state(program, fq.run(program), {(0, expected): 1})


@pytest.mark.parametrize("expression", ["0&x", "1|x", "x^x", "~x"])
def test_logic_does_not_short_circuit_uninitialized_reads(expression: str) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            f"qfunc main(output x:qbit,output result:qnum) {{result={expression};}}",
            source_format="qmod",
        )


@pytest.mark.parametrize("expression", ["(x<y)|(x==y)", "x<=y"])
def test_compound_predicate_replaces_existing_comparison(expression: str) -> None:
    p = synth.synthesize(
        f"qfunc main(output x:qnum<2>,output y:qnum<2>) {{allocate(x);allocate(y);hadamard_transform(x);hadamard_transform(y);phase({expression},pi);}}",
        source_format="qmod",
    )
    check_state(
        p,
        fq.run(p),
        {(x, y): (-1) ** int(x <= y) / 4 for x in range(4) for y in range(4)},
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("max_qubits", [13, 14])
def test_user_example_honors_width_depth_tradeoff(
    objective: Literal["width", "depth"],
    max_qubits: int,
) -> None:
    p = synth.synthesize(
        SOURCE, source_format="qmod", objective=objective, max_qubits=max_qubits
    )
    check_state(
        p,
        fq.run(p),
        {(x, y): (-1) ** int(x < 2 or y == 3) / 4 for x in range(4) for y in range(4)},
    )
    resource = p.report["resources"]
    expected_width = 14 if objective == "depth" and max_qubits == 14 else 13
    assert resource.n_qubits == expected_width
    assert resource.depth == (53 if expected_width == 14 else 69)
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(SOURCE, source_format="qmod", max_qubits=12)
