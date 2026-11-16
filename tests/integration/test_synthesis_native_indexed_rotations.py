"""Indexed controls preserve affine rotations, absolute phase and ownership."""

import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.linear_pauli_rotations.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.functions.linear_pauli_rotations.main import build_program
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_controlled_rotations import (
    controlled_reference,
)
from tests.integration.test_synthesis_native_polynomial_phase import definition

pytestmark = pytest.mark.integration
ERRORS = (synth.SynthesisError, HybridVerificationError)
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/functions/qmod_library_reference/classiq_open_library/linear_pauli_rotations/linear_pauli_rotations.qmod"
)


@pytest.mark.parametrize("value", range(64))
def test_all_unsigned_values_match_independent_affine_angle_matrix(value: int) -> None:
    native = build_program(input_value=value)
    compare_states(native, reference_circuit(input_value=value), ("x", "ind"))
    assert native.report["workspace_qubits"] == 0


def test_original_and_coherent_native_match_absolute_amplitudes() -> None:
    for coherent in (False, True):
        source = SOURCE.read_text()
        if coherent:
            source = source.replace(
                "linear_pauli_rotations(",
                "hadamard_transform(x); Z(x[0]); linear_pauli_rotations(",
                1,
            )
        original = synth.synthesize(source, source_format="qmod")
        expected = reference_circuit(coherent=coherent)
        compare_states(original, expected, ("x", "ind"))
        compare_states(build_program(coherent=coherent), expected, ("x", "ind"))


def test_nonzero_negative_parameters_compose_without_source_specific_rewrites() -> None:
    slopes, offsets = (-0.37, 0.81, -1.13), (0.29, -0.43, 0.67)
    compare_states(
        build_program(slopes, offsets, coherent=True),
        reference_circuit(slopes, offsets, coherent=True),
        ("x", "ind"),
    )


@pytest.mark.parametrize("index", [0, 1, 2])
def test_indexed_controls_preserve_spectators_and_relative_phase(index: int) -> None:
    entry = definition(
        "allocate(3,c)\nallocate(2,q)\nhadamard_transform(c)\n"
        "hadamard_transform(q)\nPHASE(0.29,c[1])\n"
        f"control(c[{index}],lambda:R(-0.73,{math.pi / 2!r},q[0]))",
        "c:Output[QArray[QBit]],q:Output[QArray[QBit]]",
    )
    native = synth.synthesize(entry, max_workspace=0)
    expected = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        expected.h(wire)
    expected.phase(1, theta=0.29)
    expected.any(2 - index, 4, matrix=controlled_reference(-0.73, math.pi / 2))
    compare_states(native, expected, ("c", "q"))


@pytest.mark.parametrize(
    "body,diagnostic",
    [
        (
            "allocate(3,c)\nallocate(2,q)\ncontrol(c[3],lambda:R(0.2,0.0,q[0]))",
            "bounds",
        ),
        (
            "allocate(3,c)\nallocate(2,q)\ncontrol(c[-1],lambda:R(0.2,0.0,q[0]))",
            "bounds",
        ),
        (
            "allocate(3,c)\nallocate(2,q)\ncontrol(c[:2],lambda:R(0.2,0.0,q[0]))",
            "exactly one bit",
        ),
        (
            "allocate(3,c)\nallocate(2,q)\ncontrol(c[0],lambda:R(0.2,0.0,c[1]))",
            "disjoint",
        ),
        (
            "allocate(2,q)\ncontrol(c[0],lambda:R(0.2,0.0,q[0]))",
            "initialize the port first",
        ),
        (
            "allocate(3,c)\ncontrol(c[0],lambda:R(0.2,0.0,q[0]))",
            "initialize the port first",
        ),
    ],
)
def test_invalid_locations_and_storage_fail_closed(body: str, diagnostic: str) -> None:
    with pytest.raises(ERRORS, match=diagnostic):
        synth.synthesize(
            definition(body, "c:Output[QArray[QBit]],q:Output[QArray[QBit]]")
        )


def test_indexed_control_obeys_actual_operation_budget() -> None:
    entry = definition(
        "allocate(3,c)\nallocate(2,q)\nhadamard_transform(c)\n"
        "control(c[1],lambda:R(0.73,0.0,q[0]))",
        "c:Output[QArray[QBit]],q:Output[QArray[QBit]]",
    )
    with pytest.raises((*ERRORS, PlanningError)):
        synth.synthesize(entry, max_operations=1)


def test_mutable_helper_keeps_bound_scalar_locations_and_rejects_aliases() -> None:
    prelude = (
        "@qfunc\ndef rotate(theta:float,c:QBit,t:QBit):\n"
        "    control(c,lambda:R(theta,0.0,t))\n"
    )
    body = "allocate(2,c)\nallocate(2,q)\nhadamard_transform(c)\nrotate(0.37,c[1],q[0])"
    signature = "c:Output[QArray[QBit]],q:Output[QArray[QBit]]"
    native = synth.synthesize(definition(body, signature, prelude), max_workspace=0)
    reference = fq.Circuit(4, dtype=torch.complex128).h(0).h(1)
    reference.any(0, 3, matrix=controlled_reference(0.37, 0.0))
    compare_states(native, reference, ("c", "q"))
    for arguments in ("c[1],c[1]", "c[1],c[0]"):
        with pytest.raises(ERRORS, match="disjoint"):
            synth.synthesize(
                definition(body.replace("c[1],q[0]", arguments), signature, prelude)
            )
