"""Raw gate views preserve numeric scale while numerical borrows remain checked."""

from fractions import Fraction

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._construction.frontend.python import _QuantumFunction
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration

IMPORTS = "from flagquantum.synth import H, RZ, CX, SWAP, control, adjoint\n"


def source(
    body: str, profile: str = "QNum[3,False,3]", helper: str = ""
) -> _QuantumFunction:
    return program_definition(
        IMPORTS
        + helper
        + f"\n@qfunc\ndef main(x:Output[{profile}], flag:Output[QBit]):\n    allocate(x)\n    allocate(flag)\n"
        + "\n".join("    " + line for line in body.splitlines())
    )


@pytest.mark.parametrize(
    "profile", ("QNum[3,True,0]", "QNum[3,True,1]", "QNum[3,False,3]")
)
def test_all_raw_words_preserve_scale_and_sign(profile: str) -> None:
    for word in range(8):
        p = synth.synthesize(
            source(
                f"x ^= {word}\nH(flag)\nRZ(0.37,x[1])\nCX(x[0],x[2])\nSWAP(x[1],x[2])",
                profile,
            )
        )
        expected = fq.Circuit(4, dtype=torch.complex128)
        for bit in range(3):
            if word & (1 << bit):
                expected.x(2 - bit)
        expected.h(3).rz(1, 0.37).cnot(2, 0).swap(1, 0)
        torch.testing.assert_close(
            fq.run(p).statevector().flatten(),
            expected.state().flatten(),
            atol=1e-12,
            rtol=0,
        )
        assert p.report["workspace_qubits"] == 0
        updated = word ^ ((word & 1) << 2)
        updated = (updated & 1) | ((updated & 2) << 1) | ((updated & 4) >> 1)
        signed = "True" in profile
        fractional = 3 if "False,3" in profile else int("True,1" in profile)
        number = updated - 8 if signed and updated & 4 else updated
        decoded = p.numeric_probabilities(fq.run(p), registers=("x",))
        assert dict(decoded) == pytest.approx(
            {(Fraction(number, 1 << fractional),): 1.0}, abs=1e-12
        )


def test_coherent_helper_inverse_and_conditional_phase() -> None:
    helper = """
@qfunc
def swap_bits(x:QNum[3,False,3],y:QNum[3,False,3]):
    for bit in range(3):
        CX(x[bit],y[bit])
        CX(y[bit],x[bit])
        CX(x[bit],y[bit])
"""
    body = """
@qfunc
def main(c:Output[QBit],x:Output[QNum[3,False,3]],y:Output[QNum[3,False,3]]):
    allocate(c)
    allocate(x)
    allocate(y)
    H(c)
    hadamard_transform(x)
    y ^= 5
    RZ(0.29,x[2])
    control(c,lambda:swap_bits(x,y))
    with adjoint():
        swap_bits(x,y)
    swap_bits(x,y)
    H(c)
"""
    definition = program_definition(IMPORTS + helper + body)
    p = synth.synthesize(definition, max_workspace=0)
    reference = (
        fq.Circuit(7, dtype=torch.complex128).h(0).h(1).h(2).h(3).x(4).x(6).rz(1, 0.29)
    )
    matrix = torch.eye(8, dtype=torch.complex128)
    matrix[5, 5] = matrix[6, 6] = 0
    matrix[5, 6] = matrix[6, 5] = 1
    for bit in range(3):
        reference.any(0, 3 - bit, 6 - bit, matrix=matrix)
    reference.h(0)
    torch.testing.assert_close(
        fq.run(p).statevector().flatten(),
        reference.state().flatten(),
        atol=2e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "body",
    (
        "RZ(0.1,x[3])",
        "RZ(0.1,x[-1])",
        "RZ(0.1,x[True])",
        "RZ(0.1,x[0:2])",
        "CX(x[1],x[1])",
        "CX(x[0],x[flag])",
    ),
)
def test_invalid_raw_scalar_views_still_refuse(body: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(source(body))


@pytest.mark.parametrize(
    "profile", ("QNum[3,True,0]", "QNum[3,True,1]", "QNum[3,False,3]")
)
def test_raw_gate_admission_does_not_allow_unsigned_arithmetic_reinterpretation(
    profile: str,
) -> None:
    helper = "\n@qfunc\ndef increment(word:QNum):\n    word += 1\n"
    with pytest.raises(ValueError):
        synth.synthesize(source("increment(x[0:2])", profile, helper))


def test_const_and_uninitialized_scalar_gate_still_refuse() -> None:
    for text in (
        "@qfunc\ndef main(x:Output[QNum[3,False,3]]):\n    H(x[0])",
        "@qfunc\ndef bad(x:Const[QNum[3,False,3]]):\n    H(x[0])\n@qfunc\ndef main(x:Output[QNum[3,False,3]]):\n    allocate(x)\n    bad(x)",
    ):
        with pytest.raises(ValueError):
            synth.synthesize(program_definition(IMPORTS + text))


@pytest.mark.parametrize(
    "profile", ("QNum[3,True,0]", "QNum[3,True,1]", "QNum[3,False,3]")
)
def test_complex_coherent_raw_gates_keep_absolute_phase(profile: str) -> None:
    definition = source(
        "H(x[0])\nH(x[1])\nH(x[2])\nRZ(0.37,x[1])\nCX(x[0],x[2])\nH(x[2])", profile
    )
    program = synth.synthesize(definition, max_workspace=0)
    reference = (
        fq.Circuit(4, dtype=torch.complex128).h(2).h(1).h(0).rz(1, 0.37).cnot(2, 0).h(0)
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        reference.state().flatten(),
        atol=1e-12,
        rtol=0,
    )
