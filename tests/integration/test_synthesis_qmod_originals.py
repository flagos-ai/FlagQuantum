"""Original-size CPU acceptance; larger than the routine 22-qubit grid."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum.ecosystem._qmod.adapter import SUPPORTED_SOURCE_VERSION, parse_qmod
from flagquantum.services._synthesis_registers import synthesize_registers
from tests.integration.test_synthesis_register_workflow import check_state

pytestmark = [pytest.mark.integration, pytest.mark.slow]
CORPUS = Path(__file__).resolve().parents[1] / "fixtures/classiq_qmod"


@pytest.mark.parametrize(
    "filename,widths,total,expected",
    [
        ("X.qmod", {"q": 1}, 1, {(1,): 1.0}),
        (
            "synthesis_tutorial.qmod",
            {"x": 3, "y": 6},
            23,
            {(x, x * x + 1): 1 / math.sqrt(8) for x in range(8)},
        ),
        (
            "multiplication_2vars_example.qmod",
            {"a": 3, "b": 3, "res": 6},
            24,
            {(4, 5, 20): 1.0},
        ),
    ],
)
def test_unmodified_originals_at_actual_dimensions(filename, widths, total, expected):
    source = CORPUS / filename
    imported = parse_qmod(
        source.read_text(),
        filename=str(source),
        source_version=SUPPORTED_SOURCE_VERSION,
    )
    program = synthesize_registers(
        imported.program, imported.functions, memory_limit_bytes=2 * 1024**3
    )
    assert dict(program.report["output_widths"]) == widths
    assert program.report["resources"].n_qubits == total
    # Bound this independent full-state test's working set before allocation.
    # Runtime's own sealed memory decision is separate from this test harness.
    assert total <= 24
    assert 4 * program.execution.state_bytes <= 1024**3
    result = fq.run(program)
    check_state(program, result, expected)
    observed = program.probabilities(
        result, registers=tuple(widths), max_amplitudes=1 << 24
    )
    assert dict(observed) == pytest.approx(
        {values: abs(amplitude) ** 2 for values, amplitude in expected.items()},
        abs=1e-12,
    )
    row = program.selection.selected
    print(
        "ORIGINAL_ACCEPTANCE "
        + json.dumps(
            {
                "file": filename,
                "source_version": imported.source_version,
                "output_widths": widths,
                "logical_ports": dict(program.registers),
                "physical_ports": dict(program._checked.registers),
                "source_identities": row.source_identities,
                "candidate": row.strategy,
                "workspace_qubits": program.report["workspace_qubits"],
                "source_resources": asdict(program.report["source_resources"]),
                "actual_resources": asdict(program.report["resources"]),
                "state_bytes": program.execution.state_bytes,
                "read_amplitudes": observed.diagnostics["scanned_amplitudes"],
                "parsed": True,
                "lowered": True,
                "executed": True,
                "full_state_verified": True,
                "tolerance": 1e-12,
            },
            sort_keys=True,
        )
    )
