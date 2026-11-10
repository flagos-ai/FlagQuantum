"""Native Rainbow programs retain exact scaled payoff, order and full unitary."""

import importlib
import math
from fractions import Fraction

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.physics import (
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
        "applications.finance.rainbow_options.rainbow_options_bruteforce_method",
        "applications/finance/rainbow_options/rainbow_options_bruteforce_method.qmod",
    ),
    (
        "tutorials.workshops.finance_workshops.rainbow_options_workshop_bruteforce",
        "tutorials/workshops/finance_workshops/rainbow_options_workshop_bruteforce.qmod",
    ),
)
APP = "examples.synthesis." + CASES[0][0]


@pytest.mark.parametrize("module,original", CASES)
@pytest.mark.parametrize("k", [0, 1, 2])
def test_original_native_and_independent_full_states(
    module: str, original: str, k: int
) -> None:
    check_original(module, original, k)


@pytest.mark.parametrize("word", range(16))
@pytest.mark.parametrize("marker", [0, 1])
def test_every_joint_selector_and_nonzero_marker(word: int, marker: int) -> None:
    # Joint index x1 + 4*x2 checks the nonsymmetric payoff's physical bit order.
    x1, x2 = word % 4, word // 4
    raw = 2 * max(Fraction(x1), Fraction(x1 + x2 - 3, 2))
    assert raw.denominator == 1 and 0 <= raw < 8
    assert effective_angles()[word] == ANGLE_TABLE[int(raw)]
    program = synth.synthesize(
        definition(
            "angles:list[float],indicator:Output[QBit]",
            "asset:QArray[QBit]=QArray()\nallocate(4,asset)\nallocate(indicator)\n"
            + "\n".join(f"X(asset[{i}])" for i in range(4) if word & (1 << i))
            + ("\nX(indicator)" if marker else "")
            + '\nselect_rotation("Y",angles,asset,indicator)\ndrop(asset)',
        ),
        classical_inputs={"angles": effective_angles()},
        max_workspace=0,
    )
    angle = ANGLE_TABLE[int(raw)]
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    expected = torch.zeros(32, dtype=torch.complex128)
    expected[word], expected[16 + word] = (-s, c) if marker else (c, s)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("variant", [0, 1])
def test_nonzero_coherent_columns_of_complete_loading(variant: int) -> None:
    reference = importlib.import_module(APP + ".comparisons.circuit")
    values = tuple((-1) ** (i + variant) * (i + 1 + variant) for i in range(32))
    amplitudes = tuple(v / math.hypot(*values) for v in values)
    program = synth.synthesize(
        definition(
            "probabilities:list[float],angles:list[float],indicator:Output[QBit]",
            "asset:QArray[QBit]=QArray()\nallocate(4,asset)\nallocate(indicator)\n"
            + f"inplace_prepare_amplitudes({amplitudes!r},0,(asset,indicator))\n"
            + "prepare(probabilities,angles,asset,indicator)\ndrop(asset)",
            "from " + APP + ".operations import prepare\n",
        ),
        classical_inputs={
            "probabilities": list(PROBABILITIES),
            "angles": effective_angles(),
        },
        max_workspace=0,
    )
    expected = reference.preparation_matrix() @ torch.tensor(
        amplitudes, dtype=torch.complex128
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("module,original", CASES)
@pytest.mark.parametrize("k", [-1, 17, True])
def test_invalid_iteration_count_is_refused(module: str, original: str, k: int) -> None:
    entry = importlib.import_module("examples.synthesis." + module + ".main")
    with pytest.raises(ValueError, match="0..16"):
        entry.build_program(k)


@pytest.mark.parametrize("module,original", CASES)
@pytest.mark.parametrize("budget", ["operations", "width", "depth", "memory"])
def test_final_program_resources_are_budgeted(
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
            synth.synthesize(entry.main, classical_inputs=bindings, max_qubits=4)
        elif budget == "depth":
            synth.synthesize(entry.main, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(
                entry.main, classical_inputs=bindings, memory_limit_bytes=16
            )
