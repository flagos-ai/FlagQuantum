"""Constrained 15-control native programs preserve amplitudes, XOR and clean work."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import pytest

from examples.synthesis.tutorials.technology_demonstrations.hardware_aware_mcx.main import (
    main,
)
from examples.synthesis.tutorials.technology_demonstrations.hardware_aware_mcx.overview.workflow import (
    Scenario,
    Topology,
    build_program,
    verify,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


class Limits(TypedDict, total=False):
    max_qubits: int
    max_workspace: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


@pytest.mark.parametrize("topology", (None, "line"))
@pytest.mark.parametrize(
    "scenario", ("zero", "coherent", "active_zero", "active_one", "near_active")
)
def test_full_original_width_and_nontrivial_states(
    topology: Topology, scenario: Scenario
) -> None:
    check = verify(build_program(topology, scenario), topology, scenario)
    assert check.workspace_qubits == 1
    assert check.maximum_state_error < 2e-11
    assert check.dirty_workspace_probability < 1e-22
    assert check.resources.n_qubits == 17
    if scenario == "zero":
        assert check.resources.n_operations == (2461 if topology == "line" else 975)
        assert check.resources.depth == (1314 if topology == "line" else 681)


@pytest.mark.parametrize("topology", (None, "line"))
@pytest.mark.parametrize(
    "limit",
    (
        {"max_qubits": 16},
        {"max_workspace": 0},
        {"max_operations": 10},
        {"max_depth": 10},
        {"memory_limit_bytes": 1},
    ),
)
def test_actual_resource_constraints_refuse(topology: Topology, limit: Limits) -> None:
    options: Limits = {"max_qubits": 18, "memory_limit_bytes": 512 * 1024**2}
    options.update(limit)
    with pytest.raises(ValueError, match="budget|workspace|memory|limit|candidate"):
        synth.synthesize(
            main, objective="depth", basis=("cx", "u"), topology=topology, **options
        )


def test_pinned_cells_have_complete_native_mapping() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/technology_demonstrations/hardware_aware_mcx/overview"
    )
    provenance = json.loads((directory / "provenance.json").read_text())
    assert [cell["index"] for cell in provenance["reviewed_code_cells"]] == [1, 3, 5]
    assert len(provenance["cell_mapping"]) == 3
    assert provenance["upstream_commit"] == "b69c4893d8264d1e59ce0909df2bd88efed25e00"
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert "from flagquantum" in code
    assert "classiq" not in code and ".qmod" not in code
    assert (
        hashlib.sha256((directory.parent / "main.py").read_bytes()).hexdigest()
        == provenance["frozen_quantum_entry_sha256"]
    )
