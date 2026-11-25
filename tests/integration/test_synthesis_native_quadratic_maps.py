"""Preserve original kicked-map and periodic-Laplacian quantum semantics."""

import importlib
from fractions import Fraction
from pathlib import Path

import pytest

import flagquantum as fq
from examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.comparisons.circuit import (
    reference_circuit as sawtooth_reference,
)
from examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.main import (
    build_program as sawtooth_program,
)
from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_1d.comparisons.circuit import (
    reference_circuit as laplacian_reference,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition
from tools.synthesis_catalog import load_catalog

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
LAPLACIAN = "examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_1d.main"


@pytest.mark.parametrize("name", ["quantum_sawtooth_map", "1D_Periodic_Laplacian_BE"])
def test_untouched_original_native_and_independent_circuit(name: str) -> None:
    case = next(c for c in load_catalog()["cases"] if Path(c["id"]).stem == name)
    module = ".".join(Path(case["directory"]).parts)
    native = importlib.import_module(module + ".main").build_program()
    reference = importlib.import_module(
        module + ".comparisons.circuit"
    ).reference_circuit()
    original = synth.synthesize(
        ROOT / case["original"]["path"], classical_inputs=case["classical_inputs"]
    )
    for program in (native, original):
        compare_states(program, reference, tuple(program.registers))
    assert native.report["workspace_qubits"] == 0


@pytest.mark.parametrize("steps", [0, 1, 2])
@pytest.mark.parametrize("unused_kick_mag", [0.23, -0.61])
def test_signed_kicked_map_keeps_literal_magnitude_and_absolute_phase(
    steps: int, unused_kick_mag: float
) -> None:
    case = next(
        c
        for c in load_catalog()["cases"]
        if c["id"].endswith("/quantum_sawtooth_map.qmod")
    )
    native = sawtooth_program(steps)
    original = synth.synthesize(
        ROOT / case["original"]["path"],
        classical_inputs={"num_kicks": steps, "kick_mag": unused_kick_mag},
    )
    for program in (native, original):
        compare_states(program, sawtooth_reference(steps), ("p",))
    if steps == 0:
        assert native.numeric_probabilities(fq.run(native), registers=("p",)) == {
            (Fraction(-2),): 1.0
        }


@pytest.mark.parametrize("position", [0, 1, 15, 16, 31])
@pytest.mark.parametrize("selector", range(4))
def test_conditional_laplacian_shifts_wrap_in_both_directions(
    position: int, selector: int
) -> None:
    preparation = "".join(f"X(l[{bit}])\n" for bit in range(2) if selector & (1 << bit))
    program = synth.synthesize(
        definition(
            "j:Output[QNum[5]],l:Output[QArray[QBit]]",
            f"j |= {position}\nallocate(2,l)\n" + preparation + "laplacian_block(j,l)",
            f"from {LAPLACIAN} import laplacian_block\n",
        ),
        max_workspace=0,
    )
    compare_states(program, laplacian_reference(position, selector), ("j", "l"))


def test_coherent_laplacian_preserves_position_phases_and_selector_interference() -> (
    None
):
    program = synth.synthesize(
        definition(
            "j:Output[QNum[5]],l:Output[QArray[QBit]]",
            "allocate(j)\nallocate(2,l)\nhadamard_transform(j)\n"
            "phase(0.13*j,1)\nlaplacian_block(j,l)",
            f"from {LAPLACIAN} import laplacian_block\n",
        ),
        max_workspace=0,
    )
    compare_states(program, laplacian_reference(coherent=True), ("j", "l"))


@pytest.mark.parametrize("steps", [-1, 17, True, 1.5])
def test_kicked_map_binding_is_bounded_and_not_coerced(steps: object) -> None:
    with pytest.raises(ValueError, match="integer between"):
        getattr(
            importlib.import_module(
                "examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.main"
            ),
            "build_program",
        )(steps)


@pytest.mark.parametrize("constraint", ["operations", "qubits", "depth", "memory"])
def test_phase_map_budgets_use_actual_lowered_resources(constraint: str) -> None:
    module = importlib.import_module(LAPLACIAN)
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "operations":
            synth.synthesize(module.main, max_operations=1)
        elif constraint == "qubits":
            synth.synthesize(module.main, max_qubits=6)
        elif constraint == "depth":
            synth.synthesize(module.main, max_depth=1)
        else:
            synth.synthesize(module.main, memory_limit_bytes=16)
