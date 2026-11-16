"""The native Drude program preserves every variational layer, not training claims."""

import json
from pathlib import Path

import pytest

from examples.synthesis.applications.chemistry.quantum_drude_oscillator.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.applications.chemistry.quantum_drude_oscillator.main import (
    build_program,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/fixtures/classiq_qmod/corpus/original/applications/chemistry/quantum_drude_oscillator/quantum_drude_oscillator.qmod"
)


@pytest.mark.parametrize("scale", [0.37, -0.81, 8.7])
def test_all_ten_layers_and_complex_absolute_phase_match_independent_circuit(
    scale: float,
) -> None:
    params = [scale * 0.1 * (index + 1) for index in range(40)]
    for program in (
        build_program(params),
        synth.synthesize(SOURCE, classical_inputs={"params": params}),
    ):
        compare_states(program, reference_circuit(params), ("state",))
        assert len(program.registers["state"]) == 6
        assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("size", [0, 39, 41])
def test_fixed_source_parameter_count_is_preserved(size: int) -> None:
    with pytest.raises(ValueError, match="forty"):
        build_program([0.1] * size)


@pytest.mark.parametrize("value", [True, float("inf"), float("nan")])
def test_nonfinite_or_boolean_parameters_are_refused(value: float) -> None:
    with pytest.raises(ValueError, match="finite reals"):
        build_program([value] + [0.1] * 39)


def test_complete_optimization_notebook_is_still_pending() -> None:
    inventory = json.loads(
        (ROOT / "tests/fixtures/classiq_qmod/native_library_inventory.json").read_text()
    )
    group = next(
        g
        for g in inventory["groups"]
        if g["id"].endswith("/quantum_drude_oscillator/quantum_drude_oscillator")
    )
    assert group["native_program"]
    assert group["workflow_status"] == "pending"
