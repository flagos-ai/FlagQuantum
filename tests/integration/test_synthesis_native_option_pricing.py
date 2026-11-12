"""Original option models preserve guarded tables, full loaders and Grover phase."""

import importlib
import math

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.option_pricing.physics import (
    ANGLE_TABLE,
    PROBABILITIES,
    effective_angles,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_amplification import (
    test_original_native_independent_state_and_named_marginal as check_original,
)
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
CASES = (
    (
        "applications.finance.option_pricing",
        "applications/finance/option_pricing/option_pricing.qmod",
    ),
    (
        "tutorials.workshops.finance_workshops.option_pricing_workshop",
        "tutorials/workshops/finance_workshops/Option_Pricing_Workshop.qmod",
    ),
)


@pytest.mark.parametrize("module,original", CASES)
@pytest.mark.parametrize("k", [0, 1, 2])
def test_unchanged_source_native_and_independent_full_states(
    module: str, original: str, k: int
) -> None:
    check_original(module, original, k)


@pytest.mark.parametrize("strike", [0, 13, 32])
@pytest.mark.parametrize("word", [12, 13, 31])
@pytest.mark.parametrize("marker", [0, 1])
def test_conditional_table_boundary_and_nonzero_marker(
    strike: int, word: int, marker: int
) -> None:
    amplitudes = [0.0] * 32
    amplitudes[word] = 1.0
    program = synth.synthesize(
        definition(
            "angles:list[float],asset:Output[QNum[5]],indicator:Output[QBit]",
            f"allocate(asset)\nallocate(indicator)\ninplace_prepare_amplitudes({amplitudes!r},0,asset)\n"
            + ("X(indicator)\n" if marker else "")
            + 'select_rotation("Y",angles,asset,indicator)',
        ),
        classical_inputs={"angles": effective_angles(strike)},
        max_workspace=0,
    )
    angle = ANGLE_TABLE[word] if word >= strike else 0.0
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    expected = torch.zeros(64, dtype=torch.complex128)
    expected[2 * word : 2 * word + 2] = torch.tensor(
        [-s, c] if marker else [c, s], dtype=torch.complex128
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("variant", [0, 1])
def test_complete_loading_on_nonzero_coherent_vectors(variant: int) -> None:
    reference = importlib.import_module(
        "examples.synthesis.applications.finance.option_pricing.comparisons.circuit"
    )
    values = tuple(
        (-1) ** (index + variant) * (index + 1 + variant) for index in range(64)
    )
    amplitudes = tuple(value / math.hypot(*values) for value in values)
    strike = 12 + 2 * variant
    program = synth.synthesize(
        definition(
            "probabilities:list[float],angles:list[float],indicator:Output[QBit]",
            "asset:QNum[5]=QNum()\nallocate(asset)\nallocate(indicator)\n"
            + f"inplace_prepare_amplitudes({amplitudes!r},0,(asset,indicator))\nprepare(probabilities,angles,asset,indicator)\ndrop(asset)",
            "from examples.synthesis.applications.finance.option_pricing.operations import prepare\n",
        ),
        classical_inputs={
            "probabilities": list(PROBABILITIES),
            "angles": effective_angles(strike),
        },
        max_workspace=0,
    )
    expected = reference.preparation_matrix(strike) @ torch.tensor(
        amplitudes, dtype=torch.complex128
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("module,original", CASES)
@pytest.mark.parametrize(
    "k,strike", [(-1, 13), (17, 13), (True, 13), (1, -1), (1, 33), (1, True)]
)
def test_invalid_iteration_or_strike_is_refused(
    module: str, original: str, k: int, strike: int
) -> None:
    entry = importlib.import_module("examples.synthesis." + module + ".main")
    with pytest.raises(ValueError):
        entry.build_program(k, strike)


@pytest.mark.parametrize("module,original", CASES)
@pytest.mark.parametrize("budget", ["operations", "width", "depth", "memory"])
def test_actual_program_resources_are_budgeted(
    module: str, original: str, budget: str
) -> None:
    entry = importlib.import_module("examples.synthesis." + module + ".main")
    bindings: dict[str, float | list[float]] = {
        "k": 1,
        "probabilities": list(PROBABILITIES),
        "angles": effective_angles(),
    }
    with pytest.raises((HybridVerificationError, synth.SynthesisError, PlanningError)):
        if budget == "operations":
            synth.synthesize(entry.main, classical_inputs=bindings, max_operations=1)
        elif budget == "width":
            synth.synthesize(entry.main, classical_inputs=bindings, max_qubits=5)
        elif budget == "depth":
            synth.synthesize(entry.main, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(
                entry.main, classical_inputs=bindings, memory_limit_bytes=16
            )
