"""Unchanged Lanchester input and nonzero variations retain both logical fields."""

import math
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.physical_systems.hhl_lanchester.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.applications.physical_systems.hhl_lanchester.main import (
    build_program,
    main,
)
from examples.synthesis.applications.physical_systems.hhl_lanchester.physics import (
    INITIAL_VECTOR,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/applications/physical_systems/hhl_lanchester/hhl_lanchester.qmod"
)


@pytest.mark.parametrize("variant", [None, 0, 1])
def test_complete_lanchester_state_global_phase_field_order_and_clean_work(
    variant: int | None,
) -> None:
    if variant is None:
        amplitudes = INITIAL_VECTOR
    else:
        values = tuple(((-1) ** (i + variant)) * (i + 1 + variant) for i in range(32))
        amplitudes = tuple(v / math.hypot(*values) for v in values)
    program = build_program(amplitudes)
    result = fq.run(program)
    expected = fq.run(reference_circuit(amplitudes)).statevector().flatten()
    torch.testing.assert_close(
        result.statevector().flatten(), expected, atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == 4
    assert dict(program.registers) == {
        "indicator": (0,),
        "time_index": (1, 2, 3, 4),
        "rabbits": (5,),
        "rescaled_eig": (6, 7, 8, 9, 10, 11),
    }
    assert (
        float(result.statevector().reshape(-1, 16)[:, 1:].abs().square().sum()) < 1e-24
    )
    logical = expected.reshape(2, 16, 2, 64, 16).abs().square().sum(dim=(0, 4))
    measured = dict(
        program.numeric_probabilities(
            result, registers=("time_index", "rabbits", "rescaled_eig")
        )
    )
    for time in range(16):
        for group in range(2):
            for phase in range(64):
                assert measured.get(
                    (Fraction(time), Fraction(group), Fraction(phase, 64)), 0
                ) == pytest.approx(float(logical[time, group, phase]), abs=1e-12)
    if variant is None:
        original = synth.synthesize(SOURCE)
        torch.testing.assert_close(
            fq.run(original).statevector().flatten(), expected, atol=1e-12, rtol=0
        )
        assert dict(original.registers) == {
            "indicator": (0,),
            "time_index_and_group.rabbits": (5,),
            "time_index_and_group.time_index": (1, 2, 3, 4),
            "rescaled_eig": (6, 7, 8, 9, 10, 11),
        }
        original_resources = original.report["resources"]
        native_resources = program.report["resources"]
        assert isinstance(original_resources, ResourceEstimate)
        assert isinstance(native_resources, ResourceEstimate)
        assert original_resources.n_operations == native_resources.n_operations == 20213
        assert original_resources.n_qubits == native_resources.n_qubits == 16
        assert original_resources.depth == native_resources.depth == 18016


@pytest.mark.parametrize(
    "budget", ["width", "workspace", "operations", "depth", "memory"]
)
def test_lanchester_actual_resource_refusals(budget: str) -> None:
    with pytest.raises((synth.SynthesisError, PlanningError, HybridVerificationError)):
        synth.synthesize(
            main,
            classical_inputs={"amplitudes": list(INITIAL_VECTOR)},
            max_qubits=15 if budget == "width" else None,
            max_workspace=3 if budget == "workspace" else None,
            max_operations=20212 if budget == "operations" else None,
            max_depth=18015 if budget == "depth" else None,
            memory_limit_bytes=1024 if budget == "memory" else None,
        )
