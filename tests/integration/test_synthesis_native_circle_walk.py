"""Original circle widths and exact signed modular walk semantics."""

import importlib
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition
from tools.synthesis_catalog import Case, load_catalog

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
NAMES = (
    "discrete_quantum_walk",
    "classiq_discrete_quantum_walk",
    "discrete_quantum_walk_circle",
    "quantum_walk_circle",
    "quantum_walk_circle_balanced_coin",
)


def entry(name: str) -> tuple[Case, ModuleType]:
    case = next(c for c in load_catalog()["cases"] if Path(c["id"]).stem == name)
    return case, importlib.import_module(
        ".".join(Path(case["directory"]).parts) + ".main"
    )


@pytest.mark.parametrize("name", NAMES)
def test_original_width_native_and_independent_circuit(name: str) -> None:
    case, module = entry(name)
    reference_module = importlib.import_module(
        ".".join(Path(case["directory"]).parts) + ".comparisons.circuit"
    )
    native = module.build_program()
    original = synth.synthesize(
        ROOT / case["original"]["path"], classical_inputs=case["classical_inputs"]
    )
    for program in (native, original):
        compare_states(program, reference_module.reference_circuit(), ("x", "coin"))
    assert native.report["workspace_qubits"] == 0
    assert native.circuit.n_wires == len(native.registers["x"]) + 1
    work = original.report["workspace_qubits"]
    assert isinstance(work, int) and work > 0


@pytest.mark.parametrize("name", NAMES[-2:])
@pytest.mark.parametrize("steps", [0, 1, 2])
def test_signed_steps_keep_raw_source_and_numeric_readout(
    name: str, steps: int
) -> None:
    case, module = entry(name)
    native = module.build_program(steps)
    reference = reference_circuit(7, steps, name.endswith("balanced_coin"))
    original = synth.synthesize(
        ROOT / case["original"]["path"], classical_inputs={"t": steps}
    )
    for program in (native, original):
        compare_states(program, reference, ("x", "coin"))
    raw = native.probabilities(fq.run(native), registers=("x", "coin"))
    expected = {
        (Fraction(x - 128 if x >= 64 else x), Fraction(coin)): probability
        for (x, coin), probability in raw.items()
    }
    assert native.numeric_probabilities(
        fq.run(native), registers=("x", "coin")
    ) == pytest.approx(expected)
    if steps == 1:
        assert any(x < 0 for x, _ in expected)


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("position", [0, 3, 4, 7])
@pytest.mark.parametrize("steps", [1, 2])
def test_every_wrap_and_sign_boundary_preserves_absolute_phase(
    signed: bool, position: int, steps: int
) -> None:
    value = position - 8 if signed and position >= 4 else position
    program = synth.synthesize(
        definition(
            f"x:Output[QNum[3,{signed},0]],coin:Output[QBit]",
            f"x |= {value}\nallocate(coin)\nphase(1,0.37)\n"
            + "H(coin)\nS(coin)\n"
            + "for step in range("
            + str(steps)
            + "):\n"
            + "    H(coin)\n    qft(x)\n"
            + "    phase((1-2*coin)*x,0.7853981633974483)\n"
            + "    with adjoint():\n        qft(x)",
        ),
        max_workspace=0,
    )
    reference = reference_circuit(3, steps, True, position)
    reference.any(
        3,
        matrix=torch.eye(2, dtype=torch.complex128)
        * torch.exp(torch.tensor(0.37j, dtype=torch.complex128)),
    )
    compare_states(program, reference, ("x", "coin"))


@pytest.mark.parametrize("signed", [False, True])
def test_coherent_input_and_coin_match_modular_permutation(signed: bool) -> None:
    program = synth.synthesize(
        definition(
            f"x:Output[QNum[3,{signed},0]],coin:Output[QBit]",
            "allocate(x)\nallocate(coin)\nhadamard_transform(x)\nphase(0.19*x,1)\n"
            "H(coin)\nS(coin)\n"
            "H(coin)\nqft(x)\nphase((1-2*coin)*x,0.7853981633974483)\n"
            "with adjoint():\n    qft(x)",
        ),
        max_workspace=0,
    )
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire, weight in enumerate((-4 if signed else 4, 2, 1)):
        reference.h(wire).phase(wire, theta=0.19 * weight)
    reference.h(3).s(3).compose(reference_circuit(3))
    compare_states(program, reference, ("x", "coin"))


@pytest.mark.parametrize("value", [-64, -1, 63])
@pytest.mark.parametrize("balanced", [False, True])
def test_original_seven_bit_sign_and_wrap_boundaries(
    value: int, balanced: bool
) -> None:
    body = f"x |= {value}\nallocate(coin)\n"
    if balanced:
        body += "H(coin)\nS(coin)\n"
    body += (
        "for step in range(2):\n"
        "    H(coin)\n    qft(x)\n"
        "    phase((1-2*coin)*x,0.04908738521234052)\n"
        "    with adjoint():\n        qft(x)"
    )
    program = synth.synthesize(
        definition("x:Output[QNum[7,True,0]],coin:Output[QBit]", body),
        max_workspace=0,
    )
    compare_states(
        program, reference_circuit(7, 2, balanced, value & 127), ("x", "coin")
    )


@pytest.mark.parametrize("name", NAMES[-2:])
@pytest.mark.parametrize("steps", [-1, 17, True, 1.5])
def test_step_binding_is_bounded_and_not_coerced(name: str, steps: object) -> None:
    _, module = entry(name)
    with pytest.raises(ValueError, match="integer between"):
        module.build_program(steps)


@pytest.mark.parametrize("constraint", ["operations", "qubits", "depth", "memory"])
def test_actual_native_walk_budgets_refuse(constraint: str) -> None:
    _, module = entry("quantum_walk_circle")
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "operations":
            synth.synthesize(module.main, classical_inputs={"t": 2}, max_operations=1)
        elif constraint == "qubits":
            synth.synthesize(module.main, classical_inputs={"t": 2}, max_qubits=7)
        elif constraint == "depth":
            synth.synthesize(module.main, classical_inputs={"t": 2}, max_depth=1)
        else:
            synth.synthesize(
                module.main, classical_inputs={"t": 2}, memory_limit_bytes=16
            )
