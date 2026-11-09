"""Shared inverse regions preserve absolute amplitudes and reject ownership leaks."""

import ast
import cmath
import importlib
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from tests.team.compiler.test_construction_python_program import program_definition
from tools.synthesis_catalog import load_catalog

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
CREDIT = "examples.synthesis.applications.finance.credit_card_fraud"
RB = "examples.synthesis.applications.benchmarking.randomized_benchmarking"


def definition(body: str, helper: str = "", width: int = 2) -> _QuantumFunction:
    return program_definition(
        "import math\nfrom flagquantum.synth import *\n"
        + helper
        + "@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
        + f"    allocate({width},q)\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )


@pytest.mark.parametrize(
    "name", ["algo_design_QCE_tutorial_part_I", "randomized_benchmarking"]
)
def test_original_native_and_independent_circuit(name: str) -> None:
    case = next(c for c in load_catalog()["cases"] if c["id"].endswith(f"/{name}.qmod"))
    module = ".".join(Path(case["directory"]).parts)
    entry = importlib.import_module(module + ".main")
    reference = importlib.import_module(module + ".comparisons.circuit")
    registers = ("x", "y") if name.startswith("algo") else ("target",)
    for source in (entry.main, ROOT / case["original"]["path"]):
        program = synth.synthesize(source, max_workspace=0)
        compare_states(program, reference.reference_circuit(), registers)
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("data_case", [0, 1, 2])
def test_credit_absolute_full_state_two_nonzero_pairs_and_same_data(
    data_case: int,
) -> None:
    native = importlib.import_module(CREDIT + ".main")
    reference = importlib.import_module(CREDIT + ".comparisons.circuit")
    data1, data2 = native.DATA_PAIRS[data_case % 2]
    if data_case == 2:
        data2 = data1
    case = next(
        c
        for c in load_catalog()["cases"]
        if c["id"].endswith("/credit_card_fraud.qmod")
    )
    for source in (native.main, ROOT / case["original"]["path"]):
        program = synth.synthesize(
            source, classical_inputs={"data1": data1, "data2": data2}, max_workspace=0
        )
        compare_states(program, reference.reference_circuit(data1, data2), ("qba",))
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("basis", [0, 1, None])
@pytest.mark.parametrize("inverse", [False, True])
def test_sx_helper_keeps_absolute_global_phase(
    basis: int | None, inverse: bool
) -> None:
    preparation = "H(q[0])\nS(q[0])" if basis is None else "X(q[0])" if basis else ""
    helper = "@qfunc\ndef sx(q:QBit):\n" "    phase(1,math.pi/4)\n    RX(math.pi/2,q)\n"
    body = preparation + (
        "\nwith adjoint():\n    sx(q[0])" if inverse else "\nsx(q[0])"
    )
    circuit = fq.Circuit(1, dtype=torch.complex128)
    if basis is None:
        circuit.h(0).s(0)
    elif basis:
        circuit.x(0)
    circuit.gate("sxdg" if inverse else "sx", 0)
    compare_states(synth.synthesize(definition(body, helper, 1)), circuit, ("q",))


@pytest.mark.parametrize("inverse", [False, True])
def test_entire_random_sequence_on_coherent_input(inverse: bool) -> None:
    reference_module = importlib.import_module(RB + ".comparisons.circuit")
    case = next(
        c
        for c in load_catalog()["cases"]
        if c["id"].endswith("/randomized_benchmarking.qmod")
    )
    raw = (ROOT / case["original"]["path"]).read_text().split("invert", 1)[0]
    assert tuple(re.findall(r"\b(RZ|X|I|SX)\(", raw)) == reference_module.SEQUENCE
    body = "H(q[0])\nS(q[0])\n" + (
        "with adjoint():\n    sequence(q)" if inverse else "sequence(q)"
    )
    entry = definition(body, f"from {RB}.main import sequence\n", 1)
    reference = fq.Circuit(1, dtype=torch.complex128).h(0).s(0)
    sequence = (
        reversed(reference_module.SEQUENCE) if inverse else reference_module.SEQUENCE
    )
    for gate in sequence:
        if gate == "RZ":
            reference.rz(0, theta=-math.pi if inverse else math.pi)
        else:
            reference.gate("sxdg" if inverse and gate == "SX" else gate.lower(), 0)
    compare_states(synth.synthesize(entry, max_workspace=0), reference, ("q",))


@pytest.mark.parametrize("helper_call", [False, True])
def test_noncommuting_inverse_only_reverses_order(helper_call: bool) -> None:
    helper = "@qfunc\ndef twist(q:QArray[QBit]):\n    RZ(0.71,q[0])\n    CX(q[0],q[1])\n    phase(1,0.23)\n    RX(-0.37,q[1])\n"
    calls = (
        ["twist(q)"]
        if helper_call
        else ["RZ(0.71,q[0])", "CX(q[0],q[1])", "phase(1,0.23)", "RX(-0.37,q[1])"]
    )
    body = "hadamard_transform(q)\nS(q[1])\nwith adjoint():\n" + "\n".join(
        "    " + line for line in calls
    )
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1).s(0)
    reference.rx(0, theta=0.37).any(
        0, matrix=torch.eye(2, dtype=torch.complex128) * cmath.exp(-0.23j)
    ).cx(1, 0).rz(1, theta=-0.71)
    compare_states(synth.synthesize(definition(body, helper)), reference, ("q",))


@pytest.mark.parametrize("word", [0, 3, 7])
@pytest.mark.parametrize("width", [3, 4])
def test_fourier_arithmetic_nonzero_targets_and_sizes(word: int, width: int) -> None:
    preparation = f"    y |= {word}"
    entry = program_definition(
        "import math\nfrom flagquantum.synth import *\n@qfunc\n"
        f"def main(x:Output[QNum[2]],y:Output[QNum[{width}]]):\n"
        "    allocate(x)\n    hadamard_transform(x)\n"
        + preparation
        + "\n    with within():\n        qft(y)\n    with apply():\n"
        + f"        phase(y*x**2,2*math.pi/{1 << width})\n"
    )
    size = 1 << width
    matrix = torch.zeros((4 * size, 4 * size), dtype=torch.complex128)
    for x in range(4):
        for y in range(size):
            matrix[size * x + (y + x * x) % size, size * x + y] = 1
    reference = fq.Circuit(width + 2, dtype=torch.complex128).h(0).h(1)
    for bit in range(width):
        if word & (1 << bit):
            reference.x(width + 1 - bit)
    reference.any(*range(width + 2), matrix=matrix)
    compare_states(synth.synthesize(entry, max_workspace=0), reference, ("x", "y"))


def test_nested_regions_inside_mutable_helper() -> None:
    helper = "@qfunc\ndef twist(q:QArray[QBit]):\n    with within():\n        H(q[0])\n    with apply():\n        RZ(0.43,q[0])\n        with adjoint():\n            RX(0.19,q[1])\n"
    program = synth.synthesize(
        definition(
            "hadamard_transform(q)\nS(q[0])\nwith adjoint():\n    twist(q)", helper
        )
    )
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1).s(1)
    reference.h(1).rx(0, theta=0.19).rz(1, theta=-0.43).h(1)
    compare_states(program, reference, ("q",))


def test_conjugate_transpose_of_complex_matrix_and_double_adjoint() -> None:
    program = synth.synthesize(
        definition(
            "hadamard_transform(q)\nwith adjoint():\n    unitary(((0,1j),(1,0)),q[0])\n    with adjoint():\n        RZ(0.31,q[1])"
        )
    )
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1)
    matrix = torch.tensor([[0, 1j], [1, 0]], dtype=torch.complex128)
    reference.rz(0, theta=0.31).any(1, matrix=matrix.mH.contiguous())
    compare_states(program, reference, ("q",))


@pytest.mark.parametrize(
    "body,helper,reason",
    [
        ("with adjoint():\n    allocate(q)", "", "pure effects"),
        (
            "with within():\n    allocate(q)\nwith apply():\n    X(q[0])",
            "",
            "pure effects",
        ),
        (
            "with within():\n    H(q[0])\nwith apply():\n    publish(q)",
            "@qfunc\ndef publish(q:Output[QArray[QBit]]):\n    allocate(2,q)\n",
            "pure effects",
        ),
        (
            "with adjoint():\n    pair(q[0],q[0])",
            "@qfunc\ndef pair(a:QBit,b:QBit):\n    CX(a,b)\n",
            "disjoint",
        ),
        ("with adjoint():\n    print(q)", "", "host calls"),
        (
            "with within():\n    H(q[0])\nwith apply():\n    local: QNum[1] = 1",
            "",
            "captured operations",
        ),
        ("with adjoint():\n    return", "", "return|supported"),
    ],
)
def test_ownership_alias_and_host_effect_refusals(
    body: str, helper: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        synth.synthesize(definition(body, helper))


def test_uninitialized_and_const_mutation_refuse() -> None:
    for declaration in (
        "@qfunc\ndef main(q:Output[QBit]):\n    with adjoint():\n        H(q)\n",
        "@qfunc\ndef borrowed(q:Const[QBit]):\n    with adjoint():\n        H(q)\n@qfunc\ndef main(q:Output[QBit]):\n    allocate(q)\n    borrowed(q)\n",
    ):
        with pytest.raises(ValueError, match="initialized|const|pure effects"):
            synth.synthesize(
                program_definition("from flagquantum.synth import *\n" + declaration)
            )


class _Limits(TypedDict, total=False):
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


@pytest.mark.parametrize(
    "budget", [{"max_operations": 1}, {"max_depth": 1}, {"memory_limit_bytes": 1}]
)
def test_inverse_regions_still_enforce_actual_budgets(budget: _Limits) -> None:
    entry = definition(
        "hadamard_transform(q)\nwith adjoint():\n    RZ(0.7,q[0])\n    CX(q[0],q[1])\n    RX(0.31,q[1])"
    )
    with pytest.raises(ValueError, match="memory budget|candidate"):
        synth.synthesize(entry, **budget)


def test_nested_and_expanded_operation_budgets() -> None:
    body = "X(q[0])"
    for _ in range(33):
        body = "with adjoint():\n" + "\n".join(
            "    " + line for line in body.splitlines()
        )
    # CPython refuses more than20 compiled with blocks. Compiler must still
    # reject a deeper retained AST without relying on CPython execution.
    retained = definition("X(q[0])")
    node = ast.parse(
        "def main(q:Output[QArray[QBit]]):\n    allocate(2,q)\n"
        + "\n".join("    " + line for line in body.splitlines())
    ).body[0]
    assert isinstance(node, ast.FunctionDef)
    retained = replace(retained, source=replace(retained.source, node=node))
    with pytest.raises(ValueError, match="nesting exceeds 32"):
        synth.synthesize(retained)
    with pytest.raises(ValueError, match="4096"):
        synth.synthesize(
            definition(
                "with adjoint():\n    for i in range(256):\n        for j in range(20):\n            X(q[0])"
            )
        )
