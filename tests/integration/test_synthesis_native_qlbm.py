"""Thirteen original QLBM steps preserve coherent amplitudes and clean workspace."""

import math
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qlbm import main as lattice
from examples.synthesis.applications.cfd.qlbm import physics
from examples.synthesis.applications.cfd.qlbm.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem.contracts import InteropConversionError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod/corpus"
RELATIVE = "applications/CFD/qlbm/qlbm.qmod"


def check(program: synth.Program, expected: torch.Tensor) -> None:
    result = fq.run(program)
    actual = result.statevector().reshape(4096, -1)
    torch.testing.assert_close(actual[:, 0], expected.flatten(), atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    probabilities = expected.flatten().abs().square().reshape(8, 8, 2, 2, 4, 4)
    for axis, name in enumerate(("x", "y", "vx", "vy", "ux", "uy")):
        marginal = probabilities.sum(dim=tuple(i for i in range(6) if i != axis))
        observed = program.probabilities(result, registers=(name,))
        for word in range(marginal.numel()):
            assert observed.get((word,), 0) == pytest.approx(
                float(marginal[word]), abs=1e-12, rel=0
            )


def test_approved_source_native_and_independent_complete_state() -> None:
    expected = fq.run(reference_circuit()).statevector()
    program = lattice.build_program()
    check(program, expected)
    source = synth.synthesize(FIXTURES / "approved" / RELATIVE)
    result = fq.run(source)
    fields = ("g_x", "g_y", "v_dir_x", "v_dir_y", "u_x", "u_y")
    data_wires = tuple(q for field in fields for q in source.registers["qs." + field])
    resources_source = source.report["resources"]
    assert isinstance(resources_source, ResourceEstimate)
    scratch = tuple(q for q in range(resources_source.n_qubits) if q not in data_wires)
    actual = (
        result.statevector()
        .reshape((2,) * resources_source.n_qubits)
        .permute(*data_wires, *scratch)
        .reshape(4096, -1)
    )
    torch.testing.assert_close(actual[:, 0], expected.flatten(), atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 19
    assert program.report["workspace_qubits"] == 7
    assert resources.n_operations == 3342
    assert resources.depth == 2770
    assert physics.ACTIVE_VELOCITIES == (3, 2, 3, 1, 2, 3, 3, 2, 3, 1, 2, 3, 3)


@pytest.mark.parametrize("variant", [0, 1])
def test_signed_boundary_and_coherent_inputs(variant: int) -> None:
    x: tuple[float, ...]
    if variant == 0:
        x = (1 / math.sqrt(2), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1 / math.sqrt(2))
        y = (0.0, 0.0, 0.6, 0.0, 0.0, -0.8, 0.0, 0.0)
        ux, uy = (0.6, 0.0, -0.8, 0.0), (0.0, -0.8, 0.0, 0.6)
    else:
        x = tuple((-1.0) ** i / math.sqrt(8) for i in range(8))
        y = (0.0, -0.8, 0.0, 0.0, 0.0, 0.6, 0.0, 0.0)
        ux, uy = (0.5, -0.5, 0.5, -0.5), (-0.5, 0.5, 0.5, -0.5)
    expected = fq.run(reference_circuit(x, y, ux, uy)).statevector()
    check(lattice.build_program(x, y, ux, uy), expected)


def test_original_unnormalized_amplitudes_still_refuse() -> None:
    with pytest.raises(InteropConversionError, match="normalized"):
        synth.synthesize(FIXTURES / "original" / RELATIVE)


def test_invalid_amplitude_width_refuses() -> None:
    with pytest.raises(ValueError, match="amplitudes"):
        lattice.build_program((1.0, 0.0))


def test_unnormalized_native_amplitudes_refuse() -> None:
    with pytest.raises(HybridVerificationError, match="normalized"):
        lattice.build_program(ux_amplitudes=(0.25, 0.25, 0.25, 0.25))


@pytest.mark.parametrize(
    "constraint", ["qubits", "workspace", "operations", "depth", "memory"]
)
def test_resource_budgets_refuse(constraint: str) -> None:
    classical: dict[str, float | list[float]] = {
        "x_amplitudes": list(physics.X_AMPLITUDES),
        "y_amplitudes": list(physics.Y_AMPLITUDES),
        "ux_amplitudes": list(physics.VELOCITY_AMPLITUDES),
        "uy_amplitudes": list(physics.VELOCITY_AMPLITUDES),
        "active": list(physics.ACTIVE_VELOCITIES),
    }
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "qubits":
            synth.synthesize(lattice.main, classical_inputs=classical, max_qubits=12)
        elif constraint == "workspace":
            synth.synthesize(lattice.main, classical_inputs=classical, max_workspace=0)
        elif constraint == "operations":
            synth.synthesize(lattice.main, classical_inputs=classical, max_operations=1)
        elif constraint == "depth":
            synth.synthesize(lattice.main, classical_inputs=classical, max_depth=1)
        else:
            synth.synthesize(
                lattice.main, classical_inputs=classical, memory_limit_bytes=16
            )


def test_native_schedule_preserves_every_original_step() -> None:
    source = (FIXTURES / "original" / RELATIVE).read_text()
    body = source[source.index("qfunc main(") :]
    suffixes = re.findall(r"stream_expanded___([012])\(qs\);", body)
    expected = tuple({"0": 3, "1": 2, "2": 1}[suffix] for suffix in suffixes)
    assert len(expected) == 13
    assert expected == physics.ACTIVE_VELOCITIES
    assert re.findall(r"reflection_expanded___([012])\(qs,", body) == suffixes
