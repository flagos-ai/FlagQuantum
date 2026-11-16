"""The split-coordinate native Laplacian preserves packed source bit order."""

from pathlib import Path

import pytest

from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_nd.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_nd.main import (
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
HELPER = "from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_nd.main import shift_coordinate\n"


def test_original_and_split_coordinate_native_match_independent_full_state() -> None:
    original = synth.synthesize(
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/Block_encoding-ND_Laplacian/ND_Periodic_Laplacian_BE.qmod"
    )
    compare_states(original, reference_circuit(), ("j", "l", "k"))
    program = build_program()
    compare_states(program, reference_circuit(), ("j1", "j0", "selector", "dimension"))
    assert program.report["workspace_qubits"] == 0
    assert tuple(map(len, program.registers.values())) == (2, 1, 2, 1)


@pytest.mark.parametrize("position", range(8))
@pytest.mark.parametrize("selector", range(4))
@pytest.mark.parametrize("dimension", range(2))
def test_all_split_boundaries_and_selector_interference(
    position: int, selector: int, dimension: int
) -> None:
    body = f"j1 |= {position >> 1}\nj0 |= {position & 1}\nallocate(2,selector)\n"
    for bit in range(2):
        if selector & (1 << bit):
            body += f"X(selector[{bit}])\n"
    body += f"dimension |= {dimension}\n"
    body += (
        "hadamard_transform(selector)\nphase(selector[0]+selector[1],3.141592653589793)\nH(dimension)\n"
        "shift_coordinate(j0,selector,dimension,0,2.0)\n"
        "shift_coordinate(j1,selector,dimension,1,4.0)\n"
        "hadamard_transform(selector)\nH(dimension)"
    )
    program = synth.synthesize(
        definition(
            "j1:Output[QNum[2]],j0:Output[QBit],selector:Output[QArray[QBit]],dimension:Output[QBit]",
            body,
            HELPER,
        ),
        max_workspace=0,
    )
    compare_states(
        program,
        reference_circuit(position, selector, dimension),
        ("j1", "j0", "selector", "dimension"),
    )


def test_phase_weighted_coherence_across_both_coordinate_wraps() -> None:
    body = (
        "allocate(2,j1)\nallocate(j0)\nallocate(2,selector)\nallocate(dimension)\n"
        "hadamard_transform(j1)\nH(j0)\nhadamard_transform(selector)\nH(dimension)\n"
        "phase(0.13*j1+0.39*j0+0.52*selector[1]+0.65*selector[0]+0.78*dimension,1)\n"
        "hadamard_transform(selector)\nphase(selector[0]+selector[1],3.141592653589793)\nH(dimension)\n"
        "shift_coordinate(j0,selector,dimension,0,2.0)\n"
        "shift_coordinate(j1,selector,dimension,1,4.0)\n"
        "hadamard_transform(selector)\nH(dimension)"
    )
    program = synth.synthesize(
        definition(
            "j1:Output[QNum[2]],j0:Output[QBit],selector:Output[QArray[QBit]],dimension:Output[QBit]",
            body,
            HELPER,
        ),
        max_workspace=0,
    )
    # Choose an independent vector whose two coordinate-bit phases follow the word weights.
    reference = reference_circuit(coherent=True)
    compare_states(program, reference, ("j1", "j0", "selector", "dimension"))


@pytest.mark.parametrize("constraint", ["operations", "qubits", "depth", "memory"])
def test_nd_laplacian_budget_refusals(constraint: str) -> None:
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "operations":
            synth.synthesize(main, max_operations=10)
        elif constraint == "qubits":
            synth.synthesize(main, max_qubits=5)
        elif constraint == "depth":
            synth.synthesize(main, max_depth=1)
        else:
            synth.synthesize(main, memory_limit_bytes=16)
