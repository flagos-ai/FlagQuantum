"""Native QFT preserves the independent Fourier definition and storage contract."""

from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qft.comparisons.circuit import (
    fourier_matrix,
    reference_circuit,
)
from examples.synthesis.functions.qft.main import build_program
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration


def definition(body: str, annotation: str = "QArray[QBit]") -> _QuantumFunction:
    return program_definition(
        "from flagquantum.synth import *\n"
        "@qfunc\n"
        f"def main(sentinel: Output[QBit], q: Output[{annotation}]):\n"
        "    allocate(sentinel)\n    X(sentinel)\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )


def test_original_native_notebook_resources_and_complex_amplitudes() -> None:
    program = build_program()
    compare_states(program, reference_circuit(), ("x",))
    resource = program.report["resources"]
    assert isinstance(resource, ResourceEstimate)
    assert resource.n_qubits == 4
    assert sum(resource.operation_counts.values()) == 12
    assert resource.depth == 8
    assert resource.operation_counts == {"h": 4, "cphase": 6, "swap": 2}
    assert program.report["workspace_qubits"] == 0
    assert program.report["source_format"] == "python"
    root, helpers = capture_program(
        program_definition(
            "from flagquantum.synth import qft\n"
            "@qfunc\ndef main(q: Output[QNum[4]]):\n"
            "    allocate(q)\n    qft(q)\n"
        )
    )
    assert not helpers
    assert sum(op.name == "qreg.qft" for op in root.body.blocks[0].operations) == 1


@pytest.mark.parametrize(
    ("width", "basis"),
    [(width, value) for width in range(1, 5) for value in range(1 << width)],
)
def test_all_fourier_columns_and_offset_registers(width: int, basis: int) -> None:
    preparation = [f"X(q[{bit}])" for bit in range(width) if basis & (1 << bit)]
    entry = definition("\n".join([f"allocate({width}, q)", *preparation, "qft(q)"]))
    program = synth.synthesize(entry)
    reference = fq.Circuit(width + 1, dtype=torch.complex128).x(0)
    for bit in range(width):
        if basis & (1 << bit):
            reference.x(width - bit)
    reference.any(*range(1, width + 1), matrix=fourier_matrix(width))
    compare_states(program, reference, ("sentinel", "q"))
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("width", range(1, 5))
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_complex_coherence_and_absolute_phase(
    width: int, objective: Literal["width", "depth"]
) -> None:
    entry = definition(
        f"allocate({width}, q)\nhadamard_transform(q)\n"
        "RX(0.73,q[0])\nRZ(-0.49,q[0])\nphase(1,0.31)\nqft(q)"
    )
    program = synth.synthesize(entry, objective=objective)
    reference = fq.Circuit(width + 1, dtype=torch.complex128).x(0)
    for wire in range(1, width + 1):
        reference.h(wire)
    reference.rx(width, theta=0.73).rz(width, theta=-0.49)
    reference.any(
        0,
        matrix=torch.eye(2, dtype=torch.complex128)
        * torch.exp(torch.tensor(0.31j, dtype=torch.complex128)),
    )
    reference.any(*range(1, width + 1), matrix=fourier_matrix(width))
    compare_states(program, reference, ("sentinel", "q"))


def test_full_register_mutable_helper_and_alias_import() -> None:
    entry = program_definition(
        "from flagquantum.synth import qft as transform, QArray, X\n"
        "@qfunc\ndef helper(q: QArray[QBit]):\n    transform(q)\n"
        "@qfunc\ndef main(q: Output[QArray[QBit]]):\n"
        "    allocate(4,q)\n    X(q[1])\n    helper(q[:])\n"
    )
    reference = fq.Circuit(4, dtype=torch.complex128).x(2)
    reference.any(*range(4), matrix=fourier_matrix(4))
    compare_states(synth.synthesize(entry), reference, ("q",))


def test_unsigned_numeric_register_uses_same_transform() -> None:
    entry = definition("allocate(4,q)\nq |= 1\nqft(q)", "QNum[4]")
    # Fresh assignment cannot overwrite initialized storage, even before QFT.
    with pytest.raises(HybridVerificationError):
        synth.synthesize(entry)
    entry = definition("q |= 5\nqft(q)", "QNum[4]")
    reference = fq.Circuit(5, dtype=torch.complex128).x(0).x(2).x(4)
    reference.any(1, 2, 3, 4, matrix=fourier_matrix(4))
    compare_states(synth.synthesize(entry), reference, ("sentinel", "q"))


@pytest.mark.parametrize(
    ("body", "diagnostic"),
    [
        ("allocate(4,q)\nqft()", "program.signature"),
        ("allocate(4,q)\nqft(q,q)", "program.signature"),
        ("allocate(4,q)\nqft(target=q)", "program.signature"),
        ("allocate(4,q)\nqft(*[q])", "program.signature"),
        ("allocate(4,q)\nqft(q[1:])", "program.view"),
        ("allocate(4,q)\nqft(q[0])", "program.view"),
    ],
)
def test_malformed_calls_and_partial_views_are_refused(
    body: str, diagnostic: str
) -> None:
    with pytest.raises(HybridVerificationError, match=diagnostic):
        synth.synthesize(definition(body))


def test_uninitialized_storage_and_const_borrows_are_refused() -> None:
    with pytest.raises(HybridVerificationError, match="register.uninitialized"):
        synth.synthesize(definition("qft(q)", "QNum[4]"))
    entry = program_definition(
        "from flagquantum.synth import qft\n"
        "@qfunc\ndef helper(q: Const[QNum[4]]):\n    qft(q)\n"
        "@qfunc\ndef main(q: Output[QNum[4]]):\n    allocate(q)\n    helper(q)\n"
    )
    with pytest.raises(HybridVerificationError, match="register.const"):
        synth.synthesize(entry)


def test_helper_partial_alias_is_refused_before_lowering() -> None:
    entry = program_definition(
        "from flagquantum.synth import qft, QArray\n"
        "@qfunc\ndef helper(q: QArray[QBit]):\n    qft(q)\n"
        "@qfunc\ndef main(q: Output[QArray[QBit]]):\n"
        "    allocate(4,q)\n    helper(q[1:])\n"
    )
    with pytest.raises(HybridVerificationError, match="program.view"):
        synth.synthesize(entry)


@pytest.mark.parametrize(
    ("qubits", "operations", "depth", "memory"),
    [
        (3, None, None, None),
        (None, 11, None, None),
        (None, None, 7, None),
        (None, None, None, 16),
    ],
)
def test_qft_respects_existing_construction_and_execution_budgets(
    qubits: int | None, operations: int | None, depth: int | None, memory: int | None
) -> None:
    from examples.synthesis.functions.qft.main import main

    with pytest.raises(
        (synth.SynthesisError, HybridVerificationError, PlanningError),
        match="budget|constraint|memory|qubit|depth|operation",
    ):
        synth.synthesize(
            main,
            max_qubits=qubits,
            max_operations=operations,
            max_depth=depth,
            memory_limit_bytes=memory,
        )
