"""Native matrix HHL preserves full states, source data and retained environments."""

import importlib
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem import InteropConversionError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
CASES = (
    "tutorials/workshops/hhl_workshop/hhl_workshop",
    "tutorials/technology_demonstrations/hhl/hhl_example",
    "applications/finance/portfolio_optimization_hhl/HHL_portfolio",
    "applications/finance/hybrid_hhl_for_portfolio_optimization/portfolio_optimization_with_hhl",
)


def native_module(directory: str) -> str:
    path = (
        directory.removesuffix("/hhl_workshop")
        if directory.endswith("/hhl_workshop/hhl_workshop")
        else directory
    )
    return "examples.synthesis." + path.replace("/", ".").replace(
        ".HHL_portfolio", ".hhl_portfolio"
    )


@pytest.mark.parametrize("directory", CASES)
def test_original_native_and_independent_complete_hhl_state(directory: str) -> None:
    module = native_module(directory)
    entry = importlib.import_module(module + ".main")
    reference = importlib.import_module(module + ".comparisons.circuit")
    source = (
        ROOT / "tests/fixtures/classiq_qmod/corpus/original" / (directory + ".qmod")
    )
    if directory.endswith("/hhl_example"):
        with pytest.raises(
            InteropConversionError, match="normalized finite real vector"
        ):
            synth.synthesize(source)
        catalog = json.loads(
            (ROOT / "tests/fixtures/classiq_qmod/corpus_catalog.json").read_text()
        )
        case = next(
            case for case in catalog["cases"] if case["id"] == directory + ".qmod"
        )
        source = ROOT / case["approved"]["path"]
    original = synth.synthesize(source)
    native = entry.build_program()
    native_work = int(native.report["workspace_qubits"])
    expected = fq.run(reference.reference_circuit()).statevector().flatten()
    native_state = fq.run(native).statevector().reshape(-1, 1 << native_work)
    torch.testing.assert_close(native_state[:, 0], expected, atol=1e-12, rtol=0)
    assert float(native_state[:, 1:].abs().square().sum()) < 1e-24
    workspace = original.report["workspace_qubits"]
    assert isinstance(workspace, int)
    original_state = fq.run(original).statevector().reshape(-1, 1 << workspace)
    torch.testing.assert_close(original_state[:, 0], expected, atol=1e-12, rtol=0)
    assert float(original_state[:, 1:].abs().square().sum()) < 1e-24
    assert native_work == (3 if directory.endswith("/HHL_portfolio") else 0)
    if reference.ENVIRONMENT_BITS:
        assert tuple(native.registers) == ("indicator", "test")
        assert float(expected.reshape(4, -1)[:, 1:].abs().square().sum()) > 0
    else:
        assert set(native.registers) == {
            "res",
            "indicator",
            (
                "phase"
                if "workshops" in directory
                else (
                    "phase_var"
                    if directory.endswith("/hhl_example")
                    else "rescaled_eig"
                )
            ),
        }


@pytest.mark.parametrize("directory", CASES)
@pytest.mark.parametrize("variant", [0, 1])
def test_two_nonzero_inputs_absolute_phase_bit_order_and_marginals(
    directory: str, variant: int
) -> None:
    module = native_module(directory)
    entry = importlib.import_module(module + ".main")
    reference = importlib.import_module(module + ".comparisons.circuit")
    values = tuple(
        (-1) ** (i + variant) * (i + 1 + variant) for i in range(len(entry.RAW_INPUT))
    )
    amplitudes = tuple(v / math.hypot(*values) for v in values)
    kwargs: dict[str, tuple[float, ...]] = {"amplitudes": amplitudes}
    if reference.ENVIRONMENT_BITS:
        comparison = tuple(((-1) ** i) * (5 - i + variant) for i in range(4))
        kwargs["comparison"] = tuple(v / math.hypot(*comparison) for v in comparison)
    program = entry.build_program(**kwargs)
    result = fq.run(program)
    workspace = int(program.report["workspace_qubits"])
    expected = (
        fq.run(reference.reference_circuit(**kwargs, workspace=workspace))
        .statevector()
        .flatten()
    )
    torch.testing.assert_close(
        result.statevector().flatten(), expected, atol=1e-12, rtol=0
    )
    indicator = reference.INDICATOR
    tensor = expected.reshape(
        (2,)
        * (
            len(reference.DATA)
            + len(reference.PHASE)
            + 1
            + (3 if reference.ENVIRONMENT_BITS else 0)
            + workspace
        )
    )
    marginal = tensor.abs().square().movedim(indicator, 0).reshape(2, -1).sum(dim=1)
    measured = dict(program.probabilities(result, registers=("indicator",)))
    for word in range(2):
        assert measured.get((word,), 0) == pytest.approx(
            float(marginal[word]), abs=1e-12
        )
    assert (
        float(
            result.statevector().reshape(-1, 1 << workspace)[:, 1:].abs().square().sum()
        )
        < 1e-24
    )


@pytest.mark.parametrize("directory", CASES)
@pytest.mark.parametrize(
    "budget", ["max_qubits", "max_operations", "max_depth", "memory_limit_bytes"]
)
def test_hhl_actual_resource_budgets_fail_closed(directory: str, budget: str) -> None:
    entry = importlib.import_module(native_module(directory) + ".main")
    amplitudes = entry.RAW_INPUT
    if directory.endswith("/hhl_example"):
        amplitudes = tuple(v / math.hypot(*amplitudes) for v in amplitudes)
    bindings = {"amplitudes": list(amplitudes)}
    if hasattr(entry, "COMPARISON_INPUT"):
        bindings["comparison"] = list(entry.COMPARISON_INPUT)
    with pytest.raises((synth.SynthesisError, PlanningError, HybridVerificationError)):
        synth.synthesize(
            entry.main,
            classical_inputs=bindings,
            max_qubits=1 if budget == "max_qubits" else None,
            max_operations=1 if budget == "max_operations" else None,
            max_depth=1 if budget == "max_depth" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )
