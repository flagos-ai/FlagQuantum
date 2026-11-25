"""Phase representation reduction preserves canceled reads and exact unitary effects."""

import cmath
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("binding", ["root", "whole", "partial", "controlled"])
@pytest.mark.parametrize("coefficient", [0.0, 0.17, -0.23])
def test_retained_and_canceled_reads_keep_absolute_and_controlled_phase(
    binding: Literal["root", "whole", "partial", "controlled"], coefficient: float
) -> None:
    width = 4 if binding == "partial" else 2
    view = "a[1:3]" if binding == "partial" else "a"
    expression = f"phase(a*a+b-b+1,{coefficient})"
    helpers = "@qfunc\ndef kick(a:QNum,b:QNum):\n    " + expression + "\n"
    call = expression if binding == "root" else f"kick({view},b)"
    if binding == "controlled":
        call = "control(c,lambda:" + call + ")"
    program = synth.synthesize(
        definition(
            f"a:Output[QNum[{width}]],b:Output[QNum[2]],c:Output[QBit]",
            "allocate(a)\nallocate(b)\nallocate(c)\nhadamard_transform(a)\nhadamard_transform(b)\nH(c)\nRZ(0.29,c)\n"
            + call,
            helpers,
        ),
        max_workspace=0,
    )
    phases = []
    for raw in range(1 << width):
        word = (raw >> 1) & 3 if binding == "partial" else raw
        for b in range(4):
            for c in range(2):
                phases.append(
                    cmath.exp(
                        1j
                        * coefficient
                        * (word * word + 1)
                        * (c if binding == "controlled" else 1)
                    )
                )
    reference = fq.Circuit(width + 3, dtype=torch.complex128)
    for wire in range(width + 3):
        reference.h(wire)
    reference.rz(width + 2, 0.29)
    reference.any(
        *range(width + 3),
        matrix=torch.diag(torch.tensor(phases, dtype=torch.complex128)),
    )
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        reference.state().flatten(),
        atol=1e-11,
        rtol=0,
    )


@pytest.mark.parametrize("coefficient", [0.0, 0.37])
@pytest.mark.parametrize("expression", ["a+b-b", "a-a+b", "a-a+b-b"])
@pytest.mark.parametrize("borrowed", [False, True])
def test_uninitialized_reads_survive_cancellation_and_zero_coefficient(
    coefficient: float, expression: str, borrowed: bool
) -> None:
    effect = f"phase({expression},{coefficient})"
    helpers = "@qfunc\ndef kick(a:QNum,b:QNum):\n    " + effect + "\n"
    with pytest.raises(
        (synth.SynthesisError, HybridVerificationError), match="initializ"
    ):
        synth.synthesize(
            definition(
                "a:Output[QNum[2]],b:Output[QNum[2]]",
                "allocate(b)\n" + ("kick(a,b)" if borrowed else effect),
                helpers,
            )
        )


def test_repeated_phase_specializations_fit_existing_node_budget() -> None:
    entry = definition(
        "word:Output[QNum[8]]",
        "allocate(word)\nhadamard_transform(word)\nfor layer in range(76):\n    kick((layer+1)*0.000001,word)",
        "@qfunc\ndef kick(gamma:float,word:QNum):\n    phase(word*word,gamma)\n",
    )
    program = synth.synthesize(entry, max_workspace=0)
    total = sum((layer + 1) * 0.000001 for layer in range(76))
    reference = fq.Circuit(8, dtype=torch.complex128)
    for wire in range(8):
        reference.h(wire)
    reference.any(
        *range(8),
        matrix=torch.diag(
            torch.tensor(
                [cmath.exp(1j * total * word * word) for word in range(256)],
                dtype=torch.complex128,
            )
        ),
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        reference.state().flatten(),
        atol=1e-11,
        rtol=0,
    )
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert 0 < resources.n_operations <= 100000
    assert program.report["workspace_qubits"] == 0
    with pytest.raises(ValueError, match="operation|budget"):
        synth.synthesize(entry, max_operations=resources.n_operations - 1)


def test_larger_phase_specializations_still_refuse_node_budget() -> None:
    entry = definition(
        "word:Output[QNum[8]]",
        "allocate(word)\nhadamard_transform(word)\nfor layer in range(96):\n    kick((layer+1)*0.000001,word)",
        "@qfunc\ndef kick(gamma:float,word:QNum):\n    phase(word*word,gamma)\n",
    )
    with pytest.raises(HybridVerificationError, match="node budget"):
        synth.synthesize(entry, max_workspace=0)
