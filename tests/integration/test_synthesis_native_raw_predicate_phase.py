"""Raw integer phase predicates retain mathematical differences across borrows."""

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


@pytest.mark.parametrize("binding", ["root", "whole", "slice", "packed"])
@pytest.mark.parametrize("inverse", [False, True])
def test_difference_phase_preserves_every_complex_amplitude(
    binding: Literal["root", "whole", "slice", "packed"], inverse: bool
) -> None:
    partial = binding in {"slice", "packed"}
    width = 4 if partial else 2
    view = {"root": "a", "whole": "a", "slice": "a[1:3]", "packed": "[a[3],a[0]]"}[
        binding
    ]
    effect = "control((a-b)==-1,lambda:phase(1.0,0.37))"
    helper = "@qfunc\ndef kick(a:QNum,b:QNum):\n    " + effect + "\n"
    call = effect if binding == "root" else f"kick({view},b)"
    if inverse:
        call = "with adjoint():\n    " + call
    program = synth.synthesize(
        definition(
            f"a:Output[QNum[{width}]],b:Output[QNum[2]]",
            "allocate(a)\nallocate(b)\nhadamard_transform(a)\nhadamard_transform(b)\n"
            "RZ(0.29,b[0])\n" + call,
            helper,
        ),
        max_workspace=0,
    )
    expected = []
    for raw in range(1 << width):
        a = raw
        if binding == "slice":
            a = (raw >> 1) & 3
        elif binding == "packed":
            a = (raw >> 3) | ((raw & 1) << 1)
        for b in range(4):
            angle = 0.29 * ((b & 1) - 0.5)
            angle += (-1 if inverse else 1) * 0.37 * (a - b == -1)
            expected.append(cmath.exp(1j * angle) / (1 << (width + 2)) ** 0.5)
    assert program.report["workspace_qubits"] == 0
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        torch.tensor(expected, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize(
    "predicate",
    ["((a-3)&1)==0", "((a-3)%2)==0", "(a-True)==0"],
)
def test_invalid_integer_operands_refuse_before_phase_cancellation(
    borrowed: bool, predicate: str
) -> None:
    effect = f"control(({predicate}) ^ ({predicate}),lambda:phase(1.0,0.37))"
    helper = "@qfunc\ndef kick(a:QNum):\n    " + effect + "\n"
    with pytest.raises((synth.SynthesisError, HybridVerificationError)):
        synth.synthesize(
            definition(
                "a:Output[QNum[2]]",
                "allocate(a)\n" + ("kick(a)" if borrowed else effect),
                helper,
            )
        )


def test_raw_difference_phase_keeps_actual_operation_budget() -> None:
    entry = definition(
        "a:Output[QNum[2]],b:Output[QNum[2]]",
        "allocate(a)\nallocate(b)\nhadamard_transform(a)\nhadamard_transform(b)\n"
        "control((a-b)==-1,lambda:phase(1.0,0.37))",
    )
    program = synth.synthesize(entry, max_workspace=0)
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    with pytest.raises(ValueError, match="operation|budget"):
        synth.synthesize(entry, max_operations=resources.n_operations - 1)
