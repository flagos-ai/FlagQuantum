"""Native all-ones X control shares checked scratch and preserves exact phases."""

import importlib
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


class _Limits(TypedDict, total=False):
    max_workspace: int
    max_qubits: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


def definition(body: str, width: int = 3, helper: str = "") -> _QuantumFunction:
    return program_definition(
        "from flagquantum.synth import *\n"
        + helper
        + "@qfunc\ndef main(c:Output[QArray[QBit]],t:Output[QBit]):\n"
        + f"    allocate({width},c)\n    allocate(t)\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )


def permutation(width: int) -> torch.Tensor:
    result = torch.eye(1 << (width + 1), dtype=torch.complex128)
    result[-2:, -2:] = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    return result


@pytest.mark.parametrize("name", ("mcx_example", "hardware_aware_mcx"))
def test_exact_original_native_and_independent_circuit(name: str) -> None:
    case = next(c for c in load_catalog()["cases"] if c["id"].endswith(f"/{name}.qmod"))
    module = ".".join(Path(case["directory"]).parts)
    entry = importlib.import_module(module + ".main")
    reference = importlib.import_module(module + ".comparisons.circuit")
    for source in (entry.main, ROOT / case["original"]["path"]):
        program = synth.synthesize(source, memory_limit_bytes=512 * 1024**2)
        compare_states(program, reference.reference_circuit(), ("cntrl", "target"))
        assert program.report["workspace_qubits"] == (4 if name == "mcx_example" else 6)


@pytest.mark.parametrize("width", (1, 2, 3))
def test_all_basis_columns_and_nonzero_target(width: int) -> None:
    for word in range(1 << (width + 1)):
        prep = [f"X(c[{bit}])" for bit in range(width) if word & (1 << (bit + 1))]
        if word & 1:
            prep.append("X(t)")
        program = synth.synthesize(
            definition("\n".join([*prep, "control(c,lambda:X(t))"]), width)
        )
        reference = fq.Circuit(width + 1, dtype=torch.complex128)
        for bit in range(width + 1):
            if word & (1 << bit):
                reference.x(width - bit)
        reference.any(*range(width + 1), matrix=permutation(width))
        compare_states(program, reference, ("c", "t"))


@pytest.mark.parametrize("width", (1, 2, 3, 7))
@pytest.mark.parametrize("helper", (False, True))
def test_coherent_phase_and_clean_scratch(width: int, helper: bool) -> None:
    helper_source = (
        "@qfunc\ndef toggle(c:QArray[QBit],t:QBit):\n    control(c,lambda:X(t))\n"
        if helper
        else ""
    )
    call = "toggle(c,t)" if helper else "control(c,lambda:X(t))"
    program = synth.synthesize(
        definition(
            "hadamard_transform(c)\nS(c[0])\nH(t)\nRZ(0.73,t)\n" + call,
            width,
            helper_source,
        )
    )
    reference = fq.Circuit(width + 1, dtype=torch.complex128)
    for bit in range(width):
        reference.h(bit)
    reference.s(width - 1).h(width).rz(width, theta=0.73)
    reference.any(*range(width + 1), matrix=permutation(width))
    compare_states(program, reference, ("c", "t"))


def test_disjoint_views_within_one_register() -> None:
    program = synth.synthesize(
        definition("hadamard_transform(c)\nS(c[0])\ncontrol(c[1:3],lambda:X(c[0]))")
    )
    reference = fq.Circuit(4, dtype=torch.complex128).h(0).h(1).h(2).s(2).ccx(0, 1, 2)
    compare_states(program, reference, ("c", "t"))


@pytest.mark.parametrize("word", (1, 3))
def test_numeric_condition_means_all_ones_not_nonzero(word: int) -> None:
    vector = [int(index == word) for index in range(4)]
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef main(c:Output[QNum[2]],t:Output[QBit]):\n"
        f"    prepare_amplitudes({vector!r},0,c)\n    allocate(t)\n"
        "    control(c,lambda:X(t))\n"
    )
    reference = fq.Circuit(3, dtype=torch.complex128).x(1)
    if word == 3:
        reference.x(0).x(2)
    compare_states(synth.synthesize(entry), reference, ("c", "t"))


@pytest.mark.parametrize("enabled", (14, 15))
@pytest.mark.parametrize("target", (0, 1))
def test_full_fifteen_control_active_and_near_active_inputs(
    enabled: int, target: int
) -> None:
    prep = [f"X(c[{bit}])" for bit in range(enabled)]
    prep += ["S(c[0])"]
    if target:
        prep.append("X(t)")
    program = synth.synthesize(
        definition("\n".join([*prep, "control(c,lambda:X(t))"]), 15)
    )
    reference = fq.Circuit(16, dtype=torch.complex128)
    for bit in range(enabled):
        reference.x(14 - bit)
    reference.s(14)
    if target ^ (enabled == 15):
        reference.x(15)
    compare_states(program, reference, ("c", "t"))


def test_const_target_and_cross_call_alias_refuse() -> None:
    protected = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef protected(c:Const[QNum[2]],t:Const[QBit]):\n    control(c,lambda:X(t))\n@qfunc\ndef main(c:Output[QNum[2]],t:Output[QBit]):\n    allocate(c)\n    allocate(t)\n    protected(c,t)\n"
    )
    with pytest.raises(ValueError, match="const|writable"):
        synth.synthesize(protected)
    helper = "@qfunc\ndef toggle(c:QArray[QBit],t:QBit):\n    control(c,lambda:X(t))\n"
    with pytest.raises(ValueError, match="disjoint"):
        synth.synthesize(definition("toggle(c,c[0])", helper=helper))


@pytest.mark.parametrize(
    "body,diagnostic",
    [
        ("control(c,lambda:X(c[0]))", "disjoint"),
        ("control(c,lambda:X(c))", "scalar"),
        ("control(c[:0],lambda:X(t))", "slice"),
        ("control(c[3],lambda:X(t))", "bounds"),
        ("control(c,lambda:X())", "target"),
        ("control(c,lambda:X(t,t))", "target"),
        ("control(c,lambda i:X(t))", "lambda"),
        ("control(c,lambda:allocate(t))", "supported qfunc"),
        ("control(c,lambda:control(c,lambda:X(t)))", "supported qfunc"),
        ("control(c==1,lambda:X(t))", "quantum port"),
    ],
)
def test_invalid_control_views_and_callbacks_refuse(body: str, diagnostic: str) -> None:
    with pytest.raises(ValueError, match=diagnostic):
        synth.synthesize(definition(body))


@pytest.mark.parametrize(
    "budget",
    [
        {"max_workspace": 3},
        {"max_qubits": 11},
        {"max_operations": 1},
        {"max_depth": 1},
        {"memory_limit_bytes": 1},
    ],
)
def test_actual_resource_budgets_refuse(budget: _Limits) -> None:
    with pytest.raises(ValueError, match="budget|workspace|memory|limit|candidate"):
        synth.synthesize(definition("control(c,lambda:X(t))", 7), **budget)


def test_uninitialized_target_and_control_refuse() -> None:
    for initialization in ("allocate(3,c)", "allocate(t)"):
        entry = program_definition(
            "from flagquantum.synth import *\n@qfunc\ndef main(c:Output[QArray[QBit]],t:Output[QBit]):\n    "
            + initialization
            + "\n    control(c,lambda:X(t))\n"
        )
        with pytest.raises(ValueError, match="width|initialized"):
            synth.synthesize(entry)
