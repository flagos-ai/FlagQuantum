"""Checkerboard block encoding preserves source field packing and controlled RX."""

from pathlib import Path

import pytest

from examples.synthesis.community.paper_implementation_project.block_encoding.checkerboard.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.community.paper_implementation_project.block_encoding.checkerboard.main import (
    ANGLES,
    build_program,
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
HELPER = "from examples.synthesis.community.paper_implementation_project.block_encoding.checkerboard.main import block\n"


def test_original_and_native_match_independent_controlled_rx_table() -> None:
    original = synth.synthesize(
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/block_encoding/checkerboard_BE_N8.qmod"
    )
    compare_states(original, reference_circuit(), ("data", "s", "j"))
    program = build_program()
    compare_states(program, reference_circuit(), ("data", "s1", "s0", "j1", "j0"))
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("angles", [(0.37, -0.91), (1.13, 2.07)])
def test_two_nonzero_rotation_bindings_match_full_state(
    angles: tuple[float, float],
) -> None:
    compare_states(
        build_program(angles),
        reference_circuit(angles),
        ("data", "s1", "s0", "j1", "j0"),
    )


@pytest.mark.parametrize("constraint", ["operations", "qubits", "depth", "memory"])
def test_checkerboard_budget_refusals(constraint: str) -> None:
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "operations":
            synth.synthesize(
                main, classical_inputs={"angles": list(ANGLES)}, max_operations=1
            )
        elif constraint == "qubits":
            synth.synthesize(
                main, classical_inputs={"angles": list(ANGLES)}, max_qubits=6
            )
        elif constraint == "depth":
            synth.synthesize(
                main, classical_inputs={"angles": list(ANGLES)}, max_depth=1
            )
        else:
            synth.synthesize(
                main, classical_inputs={"angles": list(ANGLES)}, memory_limit_bytes=16
            )


@pytest.mark.parametrize("word", range(128))
def test_every_packed_input_including_nonzero_data(word: int) -> None:
    body = f"data |= {word >> 6}\nallocate(2,s1)\ns0 |= {(word >> 3)&1}\nj1 |= {(word >> 2)&1}\nallocate(2,j0)\n"
    for bit in range(2):
        if word & (1 << (4 + bit)):
            body += f"X(s1[{bit}])\n"
        if word & (1 << bit):
            body += f"X(j0[{bit}])\n"
    body += "hadamard_transform(j1)\nhadamard_transform(j0)\nwith within():\n    hadamard_transform(s1)\n    H(s0)\nwith apply():\n    block([0.37,-0.91],data,s1,s0,j1,j0)"
    program = synth.synthesize(
        definition(
            "data:Output[QBit],s1:Output[QArray[QBit]],s0:Output[QBit],j1:Output[QBit],j0:Output[QArray[QBit]]",
            body,
            HELPER,
        ),
        max_workspace=0,
    )
    compare_states(
        program,
        reference_circuit((0.37, -0.91), word),
        ("data", "s1", "s0", "j1", "j0"),
    )


def test_phase_weighted_coherent_packed_registers() -> None:
    body = "allocate(data)\nallocate(2,s1)\nallocate(s0)\nallocate(j1)\nallocate(2,j0)\nH(data)\nhadamard_transform(s1)\nH(s0)\nH(j1)\nhadamard_transform(j0)\nphase(0.13*data+0.26*s1[1]+0.39*s1[0]+0.52*s0+0.65*j1+0.78*j0[1]+0.91*j0[0],1)\nhadamard_transform(j1)\nhadamard_transform(j0)\nwith within():\n    hadamard_transform(s1)\n    H(s0)\nwith apply():\n    block([1.13,2.07],data,s1,s0,j1,j0)"
    program = synth.synthesize(
        definition(
            "data:Output[QBit],s1:Output[QArray[QBit]],s0:Output[QBit],j1:Output[QBit],j0:Output[QArray[QBit]]",
            body,
            HELPER,
        ),
        max_workspace=0,
    )
    compare_states(
        program,
        reference_circuit((1.13, 2.07), coherent=True),
        ("data", "s1", "s0", "j1", "j0"),
    )
