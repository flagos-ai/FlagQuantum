"""In-place preparation is a specific unitary on every input column."""

import math

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
VECTORS = (
    (0.5, -0.5, 0.5, 0.5),
    (-0.5, 0.5, -0.5, -0.5),
    (0.0, 0.6, 0.8, 0.0),
    tuple(v / math.sqrt(8) for v in (1, -1, 1, 1, -1, 1, -1, -1)),
)


def independent_unitary(vector: tuple[float, ...]) -> torch.Tensor:
    """Multiply analytical prefix-conditioned RY matrices, without gate synthesis."""
    size = len(vector)
    width = size.bit_length() - 1
    unitary = torch.eye(size, dtype=torch.complex128)
    for level in range(width):
        stage = torch.zeros((size, size), dtype=torch.complex128)
        stride = 1 << (width - level - 1)
        for prefix in range(1 << level):
            start = prefix * 2 * stride
            if stride == 1:
                low, high = vector[start : start + 2]
            else:
                low = math.hypot(*vector[start : start + stride])
                high = math.hypot(*vector[start + stride : start + 2 * stride])
            half = math.atan2(high, low)
            cosine, sine = math.cos(half), math.sin(half)
            for tail in range(stride):
                lo, hi = start + tail, start + stride + tail
                stage[lo, lo] = stage[hi, hi] = cosine
                stage[lo, hi] = -sine
                stage[hi, lo] = sine
        unitary = stage @ unitary
    return unitary


@pytest.mark.parametrize(
    "vector,word", [(v, w) for v in VECTORS for w in range(len(v))]
)
def test_every_nonzero_column_and_global_sign(
    vector: tuple[float, ...], word: int
) -> None:
    width = len(vector).bit_length() - 1
    program = synth.synthesize(
        definition(
            f"q:Output[QNum[{width}]]",
            f"q |= {word}\ninplace_prepare_amplitudes({list(vector)!r},0,q)",
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        if word & (1 << (width - wire - 1)):
            reference.x(wire)
    reference.any(*range(width), matrix=independent_unitary(vector))
    compare_states(program, reference, ("q",))


@pytest.mark.parametrize(
    "probabilities", [(0.25, 0.25, 0.25, 0.25), (0.0, 0.36, 0.64, 0.0)]
)
def test_probabilities_apply_exact_square_root_unitary(
    probabilities: tuple[float, ...],
) -> None:
    vector = tuple(math.sqrt(v) for v in probabilities)
    program = synth.synthesize(
        definition(
            "q:Output[QNum[2]]",
            f"q |= 3\ninplace_prepare_state({list(probabilities)!r},0,q)",
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(2, dtype=torch.complex128).x(0).x(1)
    reference.any(0, 1, matrix=independent_unitary(vector))
    compare_states(program, reference, ("q",))


def test_entangled_storage_and_exact_adjoint_round_trip() -> None:
    vector = VECTORS[0]
    body = "allocate(q)\nallocate(ref)\nH(ref)\nCX(ref,q)\ninplace_prepare_amplitudes([0.6,-0.8],0,q)"
    program = synth.synthesize(
        definition("q:Output[QBit],ref:Output[QBit]", body), max_workspace=0
    )
    reference = fq.Circuit(2, dtype=torch.complex128).h(1).cx(1, 0)
    reference.any(0, matrix=independent_unitary((0.6, -0.8)))
    compare_states(program, reference, ("q", "ref"))
    roundtrip = synth.synthesize(
        definition(
            "q:Output[QNum[2]]",
            f"allocate(q)\nhadamard_transform(q)\nphase(0.17*q,1)\ninplace_prepare_amplitudes({list(vector)!r},0,q)\nwith adjoint():\n    inplace_prepare_amplitudes({list(vector)!r},0,q)",
        ),
        max_workspace=0,
    )
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .h(0)
        .h(1)
        .phase(0, theta=0.34)
        .phase(1, theta=0.17)
    )
    compare_states(roundtrip, reference, ("q",))


def test_mutable_helper_and_within_preserve_full_inverse() -> None:
    helper = "@qfunc\ndef prepare(angles:list[float],q:QNum):\n    inplace_prepare_amplitudes(angles,0,q)\n"
    vector = VECTORS[0]
    body = f"q |= 3\nwith within():\n    prepare({list(vector)!r},q)\nwith apply():\n    phase(0.37*q,1)"
    program = synth.synthesize(
        definition("q:Output[QNum[2]]", body, helper), max_workspace=0
    )
    u = independent_unitary(vector)
    diagonal = torch.diag(torch.exp(0.37j * torch.arange(4, dtype=torch.float64)))
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .x(0)
        .x(1)
        .any(0, 1, matrix=u.conj().T @ diagonal @ u)
    )
    compare_states(program, reference, ("q",))


@pytest.mark.parametrize(
    "call",
    [
        "inplace_prepare_amplitudes([1,0,0],0,q)",
        "inplace_prepare_amplitudes([0,0,0,0],0,q)",
        "inplace_prepare_amplitudes([1,0],0,q)",
        "inplace_prepare_amplitudes([0.5,0.5,0.5,0.5],-1,q)",
        "inplace_prepare_state([0.2,0.2,0.2,0.2],0,q)",
        "inplace_prepare_state([1.1,-0.1,0,0],0,q)",
    ],
)
def test_invalid_vectors_and_width_refuse(call: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(definition("q:Output[QNum[2]]", "allocate(q)\n" + call))


def test_uninitialized_and_const_storage_refuse() -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "q:Output[QNum[2]]", "inplace_prepare_amplitudes([0.5,0.5,0.5,0.5],0,q)"
            )
        )
    helper = "@qfunc\ndef bad(q:Const[QNum[2]]):\n    inplace_prepare_amplitudes([0.5,0.5,0.5,0.5],0,q)\n"
    with pytest.raises(HybridVerificationError):
        synth.synthesize(definition("q:Output[QNum[2]]", "allocate(q)\nbad(q)", helper))


@pytest.mark.parametrize("view", ["q[0]", "q[0:1]", "q[1:3:2]"])
def test_wrong_width_or_strided_views_refuse(view: str) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(
            definition(
                "q:Output[QArray[QBit]]",
                f"allocate(3,q)\ninplace_prepare_amplitudes([0.5,0.5,0.5,0.5],0,{view})",
            )
        )


def test_operation_and_memory_budget_refuse() -> None:
    function = definition(
        "q:Output[QNum[2]]",
        "allocate(q)\ninplace_prepare_amplitudes([0.5,-0.5,0.5,0.5],0,q)",
    )
    with pytest.raises(synth.SynthesisError):
        synth.synthesize(function, max_operations=1)
    with pytest.raises((synth.SynthesisError, PlanningError)):
        synth.synthesize(function, memory_limit_bytes=16)


def test_public_exports_preserve_single_implementation_identity() -> None:
    from flagquantum.compiler._construction import program_types
    from flagquantum.services import synthesis

    for name in ("inplace_prepare_amplitudes", "inplace_prepare_state"):
        assert (
            getattr(synth, name)
            is getattr(synthesis, name)
            is getattr(program_types, name)
        )
        assert name in synth.__all__


def test_complete_array_uses_same_big_endian_unitary() -> None:
    program = synth.synthesize(
        definition(
            "q:Output[QArray[QBit]]",
            "allocate(2,q)\nX(q[0])\ninplace_prepare_amplitudes([0.5,-0.5,0.5,0.5],0,q)",
        ),
        max_workspace=0,
    )
    reference = (
        fq.Circuit(2, dtype=torch.complex128)
        .x(1)
        .any(0, 1, matrix=independent_unitary(VECTORS[0]))
    )
    compare_states(program, reference, ("q",))


@pytest.mark.parametrize(
    "operation", ["inplace_prepare_amplitudes([0.5,-0.5,0.5,0.5],0,q)", "prepare(q)"]
)
def test_quantum_control_of_preparation_matches_independent_unitary(
    operation: str,
) -> None:
    helper = "@qfunc\ndef prepare(q:QNum):\n    inplace_prepare_amplitudes([0.5,-0.5,0.5,0.5],0,q)\n"
    program = synth.synthesize(
        definition(
            "q:Output[QNum[2]],c:Output[QBit]",
            f"allocate(q)\nallocate(c)\nH(c)\nX(q[0])\ncontrol(c,lambda:{operation})",
            helper,
        )
    )
    controlled = torch.eye(8, dtype=torch.complex128)
    controlled[4:, 4:] = independent_unitary(VECTORS[0])
    reference = fq.Circuit(3, dtype=torch.complex128).h(2).x(1)
    reference.any(2, 0, 1, matrix=controlled)
    compare_states(program, reference, ("q", "c"))
