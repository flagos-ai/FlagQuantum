"""Native bounded transforms preserve Fourier phases and checked ownership."""

import importlib
import json
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qct_qst.reference import transform_matrix
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
NAMES = ("qct_type2", "qst_type2", "qct_qst_type2", "qct_qst_type1")


def definition(body: str, *, combined: bool = False) -> _QuantumFunction:
    ports = (
        "q: Output[QArray[QBit]], selector: Output[QBit]"
        if combined
        else "q: Output[QArray[QBit]]"
    )
    return program_definition(
        "from flagquantum.synth import *\n@qfunc\n"
        f"def main({ports}):\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complete_original_native_and_independent_circuit(
    name: str, objective: Literal["width", "depth"]
) -> None:
    module = importlib.import_module(
        f"examples.synthesis.functions.qct_qst.{name}.main"
    )
    reference = importlib.import_module(
        f"examples.synthesis.functions.qct_qst.{name}.comparisons.circuit"
    )
    source = (
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original/functions/qmod_library_reference/classiq_open_library/qct_qst"
        / f"{name}.qmod"
    )
    registers = ("x", "q") if name == "qct_qst_type2" else ("x",)
    for entry in (module.main, source):
        program = synth.synthesize(entry, objective=objective, max_workspace=0)
        compare_states(program, reference.reference_circuit(), registers)
        assert program.report["workspace_qubits"] == 0
    root, _ = capture_program(module.main)
    expected = (
        "qreg.type1_transform" if name == "qct_qst_type1" else "qreg.type2_transform"
    )
    assert sum(op.name == expected for op in root.body.blocks[0].operations) == 1


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("width", range(1, 5))
def test_all_basis_columns_and_complex_interference(name: str, width: int) -> None:
    combined = name == "qct_qst_type2"
    # Execute every basis column in one batch, keeping the full boundary width.
    body = [f"allocate({width}, q)"]
    if combined:
        body.append("allocate(selector)")
    body.append(f"{name}(q, selector)" if combined else f"{name}(q)")
    program = synth.synthesize(
        definition("\n".join(body), combined=combined), max_workspace=0
    )
    total = width + int(combined)
    basis = torch.eye(1 << total, dtype=torch.complex128)
    generated = fq.Circuit.from_ir(program.circuit, bsz=len(basis), inputs=basis)
    reference = fq.Circuit(total, bsz=len(basis), inputs=basis, dtype=torch.complex128)
    data = program.registers["q"]
    ports = (*program.registers["selector"], *data) if combined else data
    reference.any(*ports, matrix=transform_matrix(width, name))
    torch.testing.assert_close(
        fq.run(generated).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0

    body = [
        f"allocate({width}, q)",
        "hadamard_transform(q)",
        "RZ(0.73, q[0])",
        "RX(-0.41, q[0])",
        "phase(1, 0.29)",
    ]
    if combined:
        body += ["allocate(selector)", "H(selector)", "RZ(-0.37, selector)"]
    body += [f"{name}(q, selector)" if combined else f"{name}(q)"]
    program = synth.synthesize(definition("\n".join(body), combined=combined))
    reference = fq.Circuit(width + int(combined), dtype=torch.complex128)
    for bit in range(width):
        reference.h(bit)
    reference.rz(width - 1, theta=0.73).rx(width - 1, theta=-0.41)
    reference.any(
        0,
        matrix=torch.eye(2, dtype=torch.complex128)
        * torch.exp(torch.tensor(0.29j, dtype=torch.complex128)),
    )
    if combined:
        reference.h(width).rz(width, theta=-0.37)
    reference.any(
        *((width, *range(width)) if combined else range(width)),
        matrix=transform_matrix(width, name),
    )
    compare_states(program, reference, ("q", "selector") if combined else ("q",))


@pytest.mark.parametrize("name", NAMES)
def test_mutable_helpers_and_complete_aliases(name: str) -> None:
    combined = name == "qct_qst_type2"
    helper_ports = "q: QArray[QBit], selector: QBit" if combined else "q: QArray[QBit]"
    call = f"{name}(q, selector)" if combined else f"{name}(q)"
    root_ports = (
        "q: Output[QArray[QBit]], selector: Output[QBit]"
        if combined
        else "q: Output[QArray[QBit]]"
    )
    source = "from flagquantum.synth import *\n@qfunc\n"
    source += f"def helper({helper_ports}):\n    {call}\n@qfunc\n"
    source += f"def main({root_ports}):\n    allocate(2, q)\n    X(q[0])\n"
    source += (
        "    allocate(selector)\n    H(selector)\n    helper(q[:], selector)\n"
        if combined
        else "    helper(q[:])\n"
    )
    program = synth.synthesize(program_definition(source))
    reference = fq.Circuit(2 + int(combined), dtype=torch.complex128).x(1)
    if combined:
        reference.h(2)
    reference.any(
        *((2, 0, 1) if combined else (0, 1)), matrix=transform_matrix(2, name)
    )
    compare_states(program, reference, ("q", "selector") if combined else ("q",))


@pytest.mark.parametrize("name", NAMES)
def test_invalid_ownership_views_width_and_host_calls(name: str) -> None:
    combined = name == "qct_qst_type2"
    suffix = ", selector" if combined else ""
    prefix = "allocate(selector)\n" if combined else ""
    for body, diagnostic in (
        (prefix + f"{name}(q{suffix})", "uninitialized|width"),
        (prefix + f"allocate(5,q)\n{name}(q{suffix})", "bounded Type"),
        (prefix + f"allocate(3,q)\n{name}(q[1:]{suffix})", "program.view"),
        (prefix + f"allocate(3,q)\n{name}()", "program.signature"),
        (prefix + f"allocate(3,q)\n{name}(target=q)", "program.signature"),
    ):
        with pytest.raises(HybridVerificationError, match=diagnostic):
            synth.synthesize(definition(body, combined=combined))
    entry = program_definition(
        f"from flagquantum.synth import {name}\n@qfunc\n"
        f"def helper(q: Const[QNum[2]]):\n    {name}(q{', q' if combined else ''})\n"
        "@qfunc\ndef main(q: Output[QNum[2]]):\n    allocate(q)\n    helper(q)\n"
    )
    with pytest.raises(HybridVerificationError):
        synth.synthesize(entry)
    with pytest.raises(TypeError, match="captured inside qfunc"):
        getattr(synth, name)(None, *([None] if combined else []))


def test_selector_alias_views_and_initialization_are_rejected() -> None:
    for body, diagnostic in (
        ("allocate(1,q)\nqct_qst_type2(q,q[0])", "disjoint"),
        ("allocate(2,q)\nqct_qst_type2(q,q[0])", "complete qubit"),
        ("allocate(2,q)\nqct_qst_type2(q,selector)", "initialized"),
    ):
        with pytest.raises(HybridVerificationError, match=diagnostic):
            synth.synthesize(definition(body, combined=True))


@pytest.mark.parametrize("name", NAMES)
def test_existing_resource_limits_and_public_exports(name: str) -> None:
    module = importlib.import_module(
        f"examples.synthesis.functions.qct_qst.{name}.main"
    )
    program = synth.synthesize(module.main)
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    for qubits, operations, depth, memory in (
        (resources.n_qubits - 1, None, None, None),
        (None, resources.n_operations - 1, None, None),
        (None, None, resources.depth - 1, None),
        (None, None, None, 16),
    ):
        with pytest.raises(
            (synth.SynthesisError, HybridVerificationError, PlanningError, ValueError)
        ):
            synth.synthesize(
                module.main,
                max_qubits=qubits,
                max_operations=operations,
                max_depth=depth,
                memory_limit_bytes=memory,
            )
    from flagquantum.services import synthesis

    assert getattr(synth, name) is getattr(synthesis, name)
    contract = json.loads(
        (ROOT / "contracts/synthesis-python-transforms-up06-candidate.json").read_text()
    )
    assert name in contract["signatures"]
