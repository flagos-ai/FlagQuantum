"""Closed predicate scratch composes with inverse regions without escaping."""

import math

import pytest
import torch

import flagquantum as fq
from examples.synthesis.functions.qft.comparisons.circuit import fourier_matrix
from flagquantum import synth
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


def compare(body: str, matrix: torch.Tensor) -> synth.Program:
    program = synth.synthesize(
        definition(
            "c:Output[QNum[2]],q:Output[QArray[QBit]]",
            "allocate(c)\nallocate(2,q)\nhadamard_transform(c)\nhadamard_transform(q)\nphase(c,0.37)\nS(q[0])\n"
            + body,
        )
    )
    reference = (
        fq.Circuit(4, dtype=torch.complex128)
        .h(0)
        .h(1)
        .h(2)
        .h(3)
        .any(
            0,
            1,
            matrix=torch.diag(torch.exp(0.37j * torch.arange(4, dtype=torch.float64))),
        )
        .s(3)
        .any(*range(4), matrix=matrix)
    )
    expected = fq.run(reference).statevector().flatten()
    work = program.report["workspace_qubits"]
    assert isinstance(work, int) and work > 0
    complete = torch.zeros((16, 1 << work), dtype=torch.complex128)
    complete[:, 0] = expected
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), complete.flatten(), atol=1e-12, rtol=0
    )
    return program


def controlled_fourier() -> torch.Tensor:
    result = torch.eye(16, dtype=torch.complex128)
    result[4:8, 4:8] = fourier_matrix(2)
    return result


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_controlled_fourier_inverse_and_composite_conditions(
    inverse: bool, composite: bool
) -> None:
    condition = "(c>=1)&(c<=1)" if composite else "c==1"
    call = f"control({condition},lambda:qft(q))"
    body = "with adjoint():\n    " + call if inverse else call
    matrix = controlled_fourier()
    compare(body, matrix.mH if inverse else matrix)


def test_within_and_nested_adjoint_freshen_predicate_storage() -> None:
    body = "with within():\n    control(c==1,lambda:qft(q))\nwith apply():\n    with adjoint():\n        RZ(0.29,q[0])\n        phase(1,0.17)"
    rotation = torch.diag(
        torch.tensor(
            [
                complex(math.cos(0.29 / 2), math.sin(0.29 / 2)),
                complex(math.cos(-0.29 / 2), math.sin(-0.29 / 2)),
            ],
            dtype=torch.complex128,
        )
    )
    action = torch.kron(torch.eye(8, dtype=torch.complex128), rotation) * complex(
        math.cos(-0.17), math.sin(-0.17)
    )
    forward = controlled_fourier()
    compare(body, forward.mH @ action @ forward)


def test_double_inverse_and_adjacent_predicates_preserve_phase() -> None:
    body = "with adjoint():\n    with adjoint():\n        control(c==1,lambda:qft(q))\n        control(c>=2,lambda:phase(1,0.31))"
    phase = torch.eye(16, dtype=torch.complex128)
    phase[8:, 8:] *= complex(math.cos(0.31), math.sin(0.31))
    compare(body, phase @ controlled_fourier())


@pytest.mark.parametrize(
    "body",
    [
        "with adjoint():\n    allocate(q)",
        "with adjoint():\n    drop(q)",
        "with adjoint():\n    control(c==1,lambda:qft(c))",
        "with adjoint():\n    control(c==1,lambda:qft(q))\n    allocate(q)",
    ],
)
def test_ownership_and_overlap_still_fail(body: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            definition(
                "c:Output[QNum[2]],q:Output[QNum[2]]",
                "allocate(c)\nallocate(q)\n" + body,
            )
        )


def test_predicate_input_must_be_initialized() -> None:
    with pytest.raises(ValueError, match="initialized"):
        synth.synthesize(
            definition(
                "c:Output[QNum[2]],q:Output[QNum[2]]",
                "allocate(q)\nwith adjoint():\n    control(c==1,lambda:qft(q))\nallocate(c)",
            )
        )
