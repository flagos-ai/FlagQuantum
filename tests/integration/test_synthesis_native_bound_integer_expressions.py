"""Bound integer names specialize before existing unsigned semantic checks."""

import cmath
import math

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("offset", [1, 3])
def test_two_integer_bindings_preserve_coherent_xor_and_nonzero_target(
    offset: int,
) -> None:
    function = definition(
        "offset:int,x:Output[QNum[2]],target:Output[QNum[3]]",
        "allocate(x)\nallocate(target)\nhadamard_transform(x)\nhadamard_transform(target)\nphase(0.13*x+0.17*target,1)\ntarget ^= x+offset",
    )
    program = synth.synthesize(function, classical_inputs={"offset": offset})
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        reference.h(wire)
    values = [cmath.exp(1j * (0.13 * x + 0.17 * t)) for x in range(4) for t in range(8)]
    reference.any(
        *range(5), matrix=torch.diag(torch.tensor(values, dtype=torch.complex128))
    )
    permutation = torch.zeros((32, 32), dtype=torch.complex128)
    for x in range(4):
        for t in range(8):
            permutation[8 * x + (t ^ (x + offset)), 8 * x + t] = 1
    reference.any(*range(5), matrix=permutation)
    compare_states(program, reference, ("x", "target"))


def test_static_loop_predicate_matches_independent_table_without_index_escape() -> None:
    function = definition(
        "q:Output[QNum[2]],target:Output[QBit]",
        "allocate(q)\nallocate(target)\nhadamard_transform(q)\nfor index in range(4):\n    control(q==index,lambda:R(0.3*(index+1),0,target))",
    )
    program = synth.synthesize(function)
    matrix = torch.zeros((8, 8), dtype=torch.complex128)
    for word in range(4):
        angle = 0.15 * (word + 1)
        matrix[2 * word, 2 * word] = matrix[2 * word + 1, 2 * word + 1] = math.cos(
            angle
        )
        matrix[2 * word, 2 * word + 1] = matrix[2 * word + 1, 2 * word] = torch.tensor(
            -1j * math.sin(angle), dtype=torch.complex128
        )
    reference = (
        fq.Circuit(3, dtype=torch.complex128).h(0).h(1).any(0, 1, 2, matrix=matrix)
    )
    compare_states(program, reference, ("q", "target"))
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "q:Output[QNum[2]],target:Output[QBit]",
                "allocate(q)\nallocate(target)\nfor index in range(2):\n    H(target)\ncontrol(q==index,lambda:R(0.3,0,target))",
            )
        )


@pytest.mark.parametrize(
    "parameter,binding",
    [("offset:float", 1.5), ("offset:float", 2.0), ("offset:int", -1)],
)
def test_real_and_negative_unsigned_constants_do_not_coerce(
    parameter: str, binding: float
) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                parameter + ",x:Output[QNum[2]],target:Output[QNum[3]]",
                "allocate(x)\nallocate(target)\ntarget ^= x+offset",
            ),
            classical_inputs={"offset": binding},
        )


def test_bound_integer_cannot_hide_target_alias() -> None:
    with pytest.raises(HybridVerificationError, match="alias"):
        synth.synthesize(
            definition(
                "offset:int,x:Output[QNum[2]],target:Output[QNum[3]]",
                "allocate(x)\nallocate(target)\ntarget ^= x+offset+0*target",
            ),
            classical_inputs={"offset": 1},
        )
