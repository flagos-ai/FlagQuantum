"""Pure QPE callbacks agree with independent matrix powers and Fourier."""

import math

import pytest
import torch

import flagquantum as fq
from examples.synthesis.qpe_reference import dense_qpe
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("angle", [0.37, -0.61])
@pytest.mark.parametrize("inverse", [False, True])
def test_nested_word_and_predicate_phase_callbacks_keep_pure_effects(
    angle: float, inverse: bool
) -> None:
    helper = f"""
@qfunc
def diagonal(a:QNum,t:QBit):
    phase(0.19*a+0.23,1)
    control(a<=1,lambda:phase(1.0,{-angle}))
    control(a>=2,lambda:phase(t,{angle}))
@qfunc
def kick(a:QNum,t:QBit):
"""
    helper += (
        "    with adjoint():\n        diagonal(a,t)\n"
        if inverse
        else "    diagonal(a,t)\n"
    )
    program = synth.synthesize(
        definition(
            "p:Output[QNum[3,False,3]]",
            "a:QNum[2]=QNum()\nt:QBit=QBit()\nallocate(a)\nallocate(t)\n"
            "hadamard_transform(a)\nH(t)\nRZ(0.17,a[0])\nRZ(-0.29,t)\n"
            "allocate(p)\nqpe(lambda:kick(a,t),p)\ndrop(a)\ndrop(t)",
            helper,
        ),
        max_workspace=0,
    )
    phases = torch.tensor(
        [
            0.19 * a + 0.23 - angle * int(a <= 1) + angle * t * int(a >= 2)
            for a in range(4)
            for t in range(2)
        ],
        dtype=torch.float64,
    )
    unitary = torch.diag(torch.exp((-1j if inverse else 1j) * phases))
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2)
    initial_phases = torch.tensor(
        [
            0.085 * (2 * (a & 1) - 1) - 0.145 * (2 * t - 1)
            for a in range(4)
            for t in range(2)
        ],
        dtype=torch.float64,
    )
    preparation = torch.diag(torch.exp(1j * initial_phases)) @ torch.kron(
        torch.kron(hadamard, hadamard), hadamard
    )
    expected = fq.run(dense_qpe(unitary, preparation, 3)).statevector().flatten()
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-11, rtol=0
    )
    assert program.report["workspace_qubits"] == 0


def reference(matrix: torch.Tensor, word: int, bits: int = 3) -> torch.Tensor:
    """Build an independent dense controlled-unitary Circuit."""
    circuit = fq.Circuit(bits + 1, dtype=torch.complex128)
    if word:
        circuit.x(bits)
    for bit in range(bits):
        controlled = torch.eye(4, dtype=torch.complex128)
        controlled[2:, 2:] = torch.linalg.matrix_power(matrix, 1 << bit)
        circuit.h(bits - 1 - bit).any(bits - 1 - bit, bits, matrix=controlled)
    axis = torch.arange(1 << bits, dtype=torch.float64)
    inverse = torch.exp(-2j * math.pi * torch.outer(axis, axis) / (1 << bits))
    circuit.any(*range(bits), matrix=inverse / math.sqrt(1 << bits))
    state = fq.run(circuit).statevector()
    assert isinstance(state, torch.Tensor)
    return state.flatten()


@pytest.mark.parametrize("word", [0, 1])
@pytest.mark.parametrize("operation", ["H", "RX", "RY", "RZ", "S", "helper"])
def test_all_input_columns_and_conditional_global_phase(
    operation: str, word: int
) -> None:
    angle = 0.73
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    matrices = {
        "H": torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) / math.sqrt(2),
        "RX": torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex128),
        "RY": torch.tensor([[c, -s], [s, c]], dtype=torch.complex128),
        "RZ": torch.diag(
            torch.exp(
                torch.tensor([-0.5j * angle, 0.5j * angle], dtype=torch.complex128)
            )
        ),
        "S": torch.diag(torch.tensor([1, 1j], dtype=torch.complex128)),
    }
    helper = ""
    if operation == "helper":
        helper = "@qfunc\ndef kick(s:QBit):\n    with adjoint():\n        R(0.73,1.5707963267948966,s)\n    phase(1.0,0.41)\n"
        call = "kick(s)"
        matrix = matrices["RY"].mH * complex(math.cos(0.41), math.sin(0.41))
    else:
        call = (
            f"{operation}({angle},s)"
            if operation.startswith("R")
            else f"{operation}(s)"
        )
        if operation == "RY":
            call = f"R({angle},1.5707963267948966,s)"
        matrix = matrices[operation]
    setup = "X(s)\n" if word else ""
    program = synth.synthesize(
        definition(
            "p:Output[QNum[3,False,3]]",
            "s:QBit=QBit()\nallocate(s)\n"
            + setup
            + f"allocate(p)\nqpe(lambda:{call},p)\ndrop(s)",
            helper,
        ),
        max_workspace=0,
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        reference(matrix, word),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("callback", ["drop(s)", "allocate(s)", "H(p)", "kick(s)"])
def test_callback_cannot_change_ownership_or_capture_counter(callback: str) -> None:
    helper = "@qfunc\ndef kick(s:QBit):\n    hidden:QBit=QBit()\n    allocate(hidden)\n    H(s)\n    drop(hidden)\n"
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "p:Output[QNum[3,False,3]]",
                "s:QBit=QBit()\nallocate(s)\nallocate(p)\n"
                + f"qpe(lambda:{callback},p)\ndrop(s)",
                helper,
            )
        )


@pytest.mark.parametrize("angle", [0.41, -1.23])
def test_scalar_phase_becomes_observable_under_qpe(angle: float) -> None:
    program = synth.synthesize(
        definition(
            "p:Output[QNum[3,False,3]]",
            "s:QBit=QBit()\nallocate(s)\nallocate(p)\n"
            + f"qpe(lambda:phase(1.0,{angle}),p)\ndrop(s)",
        )
    )
    matrix = torch.eye(2, dtype=torch.complex128) * complex(
        math.cos(angle), math.sin(angle)
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        reference(matrix, 0),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("body", ["H(s)", "kick(s)"])
def test_uninitialized_data_and_const_mutation_are_rejected(body: str) -> None:
    helper = "@qfunc\ndef kick(s:Const[QBit]):\n    H(s)\n"
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "p:Output[QNum[2,False,2]]",
                "s:QBit=QBit()\nallocate(p)\n" + f"qpe(lambda:{body},p)\ndrop(s)",
                helper,
            )
        )


@pytest.mark.parametrize("budget", ["operations", "depth", "width"])
def test_callback_cost_is_charged_to_real_budget(budget: str) -> None:
    program = definition(
        "p:Output[QNum[2,False,2]]",
        "s:QBit=QBit()\nallocate(s)\nallocate(p)\nqpe(lambda:H(s),p)\ndrop(s)",
    )
    with pytest.raises((HybridVerificationError, synth.SynthesisError)):
        if budget == "operations":
            synth.synthesize(program, max_operations=1)
        elif budget == "depth":
            synth.synthesize(program, max_depth=1)
        else:
            synth.synthesize(program, max_qubits=2)


@pytest.mark.parametrize(
    "helper,call",
    [
        ("@qfunc\ndef kick(s:Const[QBit]):\n    H(s)\n", "kick(s)"),
        ("@qfunc\ndef kick(a:QBit,b:QBit):\n    CX(a,b)\n", "kick(s,s)"),
        ("@qfunc\ndef kick(s:QBit):\n    pass\n", "kick(s)"),
        ("", "print(s)"),
    ],
)
def test_initialized_data_does_not_admit_illegal_callbacks(
    helper: str, call: str
) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "p:Output[QNum[2,False,2]]",
                "s:QBit=QBit()\nallocate(s)\nallocate(p)\n"
                + f"qpe(lambda:{call},p)\ndrop(s)",
                helper,
            )
        )


def test_repeated_callback_expansion_respects_effect_bound() -> None:
    helper = "@qfunc\ndef burst(s:QBit):\n    for i in range(32):\n        H(s)\n"
    with pytest.raises(HybridVerificationError, match="4096"):
        synth.synthesize(
            definition(
                "p:Output[QNum[8,False,8]]",
                "s:QBit=QBit()\nallocate(s)\nallocate(p)\nqpe(lambda:burst(s),p)\ndrop(s)",
                helper,
            )
        )
