"""Dicke preparation and coherent Grover mixer preserve absolute amplitudes."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.grover_mixer_qaoa.gm_qaoa.comparisons.circuit import (
    initial_circuit,
    reference_circuit,
)
from examples.synthesis.algorithms.search_and_optimization.grover_mixer_qaoa.gm_qaoa.main import (
    PARAMETERS,
    build_program,
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition
from tools.synthesis_catalog import load_catalog

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
HELPER = "from examples.synthesis.algorithms.search_and_optimization.grover_mixer_qaoa.gm_qaoa.main import initial_state\n"


@pytest.mark.parametrize("params", [PARAMETERS, (0.31, -0.22, 0.57, -0.4, 0.28, 0.63)])
def test_untouched_original_native_and_independent_circuit(
    params: tuple[float, ...],
) -> None:
    case = next(c for c in load_catalog()["cases"] if c["id"].endswith("/gm_qaoa.qmod"))
    native = build_program(params)
    original = synth.synthesize(
        ROOT / case["original"]["path"], classical_inputs={"params": list(params)}
    )
    for program in (native, original):
        compare_states(program, reference_circuit(params), ("x",))
    assert native.report["workspace_qubits"] == 0
    probabilities = native.probabilities(fq.run(native), registers=("x",))
    assert all(value.bit_count() == 1 for (value,) in probabilities)
    assert sum(probabilities.values()) == pytest.approx(1)


@pytest.mark.parametrize("basis", range(16))
def test_dicke_helper_full_unitary_extension(basis: int) -> None:
    preparation = "".join(f"X(x[{bit}])\n" for bit in range(4) if basis & (1 << bit))
    program = synth.synthesize(
        definition(
            "x:Output[QArray[QBit]]",
            "allocate(4,x)\n" + preparation + "initial_state(x)",
            HELPER,
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(4, dtype=torch.complex128)
    for bit in range(4):
        if basis & (1 << bit):
            reference.x(3 - bit)
    reference.compose(initial_circuit())
    compare_states(program, reference, ("x",))


def test_dicke_helper_inverse_preserves_a_phase_weighted_coherent_input() -> None:
    program = synth.synthesize(
        definition(
            "x:Output[QArray[QBit]]",
            "allocate(4,x)\nhadamard_transform(x)\nphase(x[0]+2*x[1]-x[2],0.23)\n"
            "initial_state(x)\nwith adjoint():\n    initial_state(x)",
            HELPER,
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    for wire, coefficient in ((3, 1), (2, 2), (1, -1)):
        reference.phase(wire, theta=coefficient * 0.23)
    compare_states(program, reference, ("x",))


@pytest.mark.parametrize("params", [(), (0.2,) * 5])
def test_parameter_count_is_explicit(params: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="three cost and three mixer"):
        build_program(params)


@pytest.mark.parametrize("constraint", ["operations", "qubits", "depth", "memory"])
def test_actual_mixer_budgets_are_enforced(constraint: str) -> None:
    bindings = {"params": list(PARAMETERS)}
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=100)
        elif constraint == "qubits":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=3)
        elif constraint == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=10)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)
