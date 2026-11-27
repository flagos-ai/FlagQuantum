"""Static QPE unitary families preserve per-power semantics and lexical scope."""

import math

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.frontends._structured.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


def family_reference(angle: float, bits: int, *, matrix: bool = False) -> torch.Tensor:
    """Assemble each controlled family member and inverse DFT independently."""
    circuit = fq.Circuit(bits + 1, dtype=torch.complex128)
    circuit.h(bits).rz(bits, theta=0.29)
    for bit in range(bits):
        power = 1 << bit
        radians = angle if matrix else angle * power * power + 0.17
        rotation = torch.diag(
            torch.tensor(
                [
                    complex(math.cos(-radians / 2), math.sin(-radians / 2)),
                    complex(math.cos(radians / 2), math.sin(radians / 2)),
                ],
                dtype=torch.complex128,
            )
        )
        if not matrix:
            rotation *= complex(math.cos(0.13 * power), math.sin(0.13 * power))
        controlled = torch.eye(4, dtype=torch.complex128)
        controlled[2:, 2:] = rotation
        circuit.h(bits - bit - 1).any(bits - bit - 1, bits, matrix=controlled)
    axis = torch.arange(1 << bits, dtype=torch.float64)
    inverse = torch.exp(-2j * math.pi * torch.outer(axis, axis) / (1 << bits))
    circuit.any(*range(bits), matrix=inverse / math.sqrt(1 << bits))
    result = fq.run(circuit).statevector()
    assert isinstance(result, torch.Tensor)
    return result.flatten()


@pytest.mark.parametrize("angle", [0.37, -0.61])
@pytest.mark.parametrize("bits", [1, 3, 8])
def test_nonlinear_family_helper_and_conditional_global_phase(
    angle: float, bits: int
) -> None:
    helper = """
@qfunc
def family(radians:float,power:int,s:QBit):
    RZ(radians*power*power+0.17,s)
    phase(1.0,0.13*power)
"""
    program = synth.synthesize(
        definition(
            f"angle:float,p:Output[QNum[{bits},True,{bits}]]",
            "s:QBit=QBit()\nallocate(s)\nH(s)\nRZ(0.29,s)\nallocate(p)\n"
            "qpe(lambda power:family(angle,power,s),p)\ndrop(s)",
            helper,
        ),
        classical_inputs={"angle": angle},
        max_workspace=0,
    )
    actual = fq.run(program).statevector().flatten()
    torch.testing.assert_close(
        actual, family_reference(angle, bits), atol=2e-11, rtol=0
    )
    assert program.circuit.n_wires == bits + 1
    assert program.report["workspace_qubits"] == 0


def test_one_argument_matrix_is_not_powered_again() -> None:
    angle = 0.53
    diagonal = [complex(math.cos(a), math.sin(a)) for a in (-angle / 2, angle / 2)]
    call = f"unitary([[{diagonal[0]!r},0],[0,{diagonal[1]!r}]],s)"
    program = synth.synthesize(
        definition(
            "p:Output[QNum[3,False,3]]",
            "s:QBit=QBit()\nallocate(s)\nH(s)\nRZ(0.29,s)\nallocate(p)\n"
            f"qpe(lambda power:{call},p)\ndrop(s)",
        )
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        family_reference(angle, 3, matrix=True),
        atol=2e-11,
        rtol=0,
    )


@pytest.mark.parametrize(
    "callback",
    [
        "lambda power=1:RZ(power,s)",
        "lambda *power:H(s)",
        "lambda *,power:H(s)",
        "lambda power,extra:H(s)",
        "lambda **power:H(s)",
        "lambda s:H(s)",
        "lambda p:H(s)",
        "lambda angle:H(s)",
        "lambda power:drop(s)",
        "lambda power:allocate(s)",
        "lambda power:H(p)",
        "lambda power:kick(s,s)",
        "lambda power:escape(s)",
        "lambda power:print(s)",
        "lambda power:RESET(s)",
    ],
)
def test_power_callback_rejects_unsupported_scopes_effects_and_escapes(
    callback: str,
) -> None:
    helpers = """
@qfunc
def kick(a:QBit,b:QBit):
    CX(a,b)
@qfunc
def escape(result:Output[QBit]):
    allocate(result)
"""
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "angle:float,p:Output[QNum[3,False,3]]",
                "s:QBit=QBit()\nallocate(s)\nallocate(p)\n"
                f"qpe({callback},p)\ndrop(s)",
                helpers,
            ),
            classical_inputs={"angle": 0.4},
        )


def test_power_binding_is_removed_before_following_expressions() -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "p:Output[QNum[2,False,2]]",
                "s:QBit=QBit()\nallocate(s)\nallocate(p)\n"
                "qpe(lambda power:RZ(0.17*power,s),p)\nRZ(power,s)\ndrop(s)",
            )
        )


@pytest.mark.parametrize(
    "constraint",
    [
        {"max_qubits": 3},
        {"max_operations": 3},
        {"max_depth": 2},
        {"memory_limit_bytes": 1},
    ],
)
def test_each_family_member_counts_against_resource_budgets(
    constraint: dict[str, int],
) -> None:
    with pytest.raises((HybridVerificationError, synth.SynthesisError, PlanningError)):
        synth.synthesize(
            definition(
                "p:Output[QNum[3,False,3]]",
                "s:QBit=QBit()\nallocate(s)\nallocate(p)\n"
                "qpe(lambda power:RZ(0.17*power,s),p)\ndrop(s)",
            ),
            max_qubits=constraint.get("max_qubits"),
            max_operations=constraint.get("max_operations"),
            max_depth=constraint.get("max_depth"),
            memory_limit_bytes=constraint.get("memory_limit_bytes"),
        )
