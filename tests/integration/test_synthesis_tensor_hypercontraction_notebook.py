"""Check source-sized tensor-hypercontraction compilation and reversible regions."""

import json
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("numpy")

import flagquantum as fq
from examples.synthesis.applications.chemistry.tensor_hypercontraction.overview import (
    model,
    quantum,
    verification,
)
from examples.synthesis.applications.chemistry.tensor_hypercontraction.overview.layout import (
    allocate_layout,
)
from flagquantum import synth
from flagquantum.compiler.resource_estimation import estimate_resources

pytestmark = pytest.mark.integration


def test_original_table_generation_and_layout() -> None:
    generated = model.generate()
    assert generated == model.generate()
    assert generated.rotation == pytest.approx(3.448296944257913)
    assert tuple(
        (len(table), len(table[0]))
        for table in (
            generated.mu_alt,
            generated.nu_alt,
            generated.keep,
            generated.rot_mu,
            generated.rot_nu,
        )
    ) == ((31, 5), (31, 5), (31, 18), (31, 12), (31, 12))
    assert len(generated.theta) == 31 and len(generated.clock) == 16
    assert math.fsum(a * a for a in generated.clock) == pytest.approx(1)
    assert model.CLOCK_STEPS == 15
    clock = synth.synthesize(
        quantum.clock_state,
        classical_inputs={"weights": list(generated.clock)},
        max_qubits=4,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
    actual = fq.run(clock).statevector().reshape(-1)
    torch.testing.assert_close(
        actual,
        torch.tensor(generated.clock, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    layout = allocate_layout()
    assert len(layout.ell) == 71 and len(layout.clock) == 4
    assert layout.logical_width == 83


def test_subcircuit_trace_includes_dirty_targets_and_boundary_cases() -> None:
    checks = verification.verify(model.generate(), allocate_layout())
    assert checks == verification.Checks(1024, 96, 16, 72, 146)


def test_full_compile_only_resources_and_stage_map() -> None:
    compiled = quantum.compile_all(model.generate())
    resources = [
        estimate_resources(circuit.to_ir())
        for circuit in (
            compiled.prepare_one,
            compiled.prepare_two,
            compiled.select,
            compiled.complete,
        )
    ]
    assert all(item.n_qubits == compiled.layout.width for item in resources)
    assert [item.n_operations for item in resources] == [711, 5517, 1546, 218838]
    assert resources[-1].used_qubits == 133
    assert resources[-1].depth == 173381
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/applications/chemistry/tensor_hypercontraction/overview/walkthrough.ipynb"
        ).read_text()
    )
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert [cell["metadata"]["source_stage"] for cell in cells] == [
        1,
        3,
        5,
        7,
        9,
        11,
        12,
        13,
        15,
        16,
        18,
        20,
        21,
        23,
        25,
        26,
        27,
    ]
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
