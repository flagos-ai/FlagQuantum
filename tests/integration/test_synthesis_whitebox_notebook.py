"""Original-size whitebox Notebook sampling, predicate, phases and provenance."""

import hashlib
import json
import math
from pathlib import Path

import pytest

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name == "numpy":
        pytest.skip(
            "optional numpy is required for the Notebook", allow_module_level=True
        )
    raise

from examples.synthesis.applications.cybersecurity.whitebox_fuzzing.main import main
from examples.synthesis.applications.cybersecurity.whitebox_fuzzing.overview.operations import (
    build_program,
)
from examples.synthesis.applications.cybersecurity.whitebox_fuzzing.overview.workflow import (
    classical_samples,
    foo,
    marked_pairs,
    output_counts,
    quantum_samples,
    source_recommendation,
    verify,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("iterations", (None, 0, 1, 2, 10))
def test_original_oracle_search_and_clean_work(iterations: int | None) -> None:
    check = verify(build_program(iterations), iterations)
    assert check.maximum_state_error < 3e-11
    assert check.dirty_workspace_probability < 1e-22
    assert check.reference_dirty_probability < 1e-22
    assert check.resources.n_qubits <= 25


def test_predicate_exhaustion_and_branch_priority() -> None:
    actual = {(x, y) for x in range(64) for y in range(64) if foo(x, y) == "c"}
    assert (
        actual
        == set(marked_pairs())
        == {(1, 1), (1, 3), (1, 5), (1, 7), (5, 1), (5, 3)}
    )
    assert foo(12, 4) == "a" and foo(12, 3) == "b"
    assert foo(5, 3) == "c" and foo(5, 4) == "e"
    assert foo(0, 0) == "d" and foo(63, 63) == "e"
    assert source_recommendation() == 36


def test_original_classical_rng_and_500_draws() -> None:
    rng = np.random.RandomState(3)
    xs, ys = rng.randint(0, 63, 500), rng.randint(0, 63, 500)
    expected = dict.fromkeys(("a", "b", "c", "d", "e"), 0)
    for x, y in zip(xs, ys, strict=True):
        expected[foo(int(x), int(y))] += 1
    observed = classical_samples()
    assert sum(row.count for row in observed) == 500
    assert all(0 <= row.x < 63 and 0 <= row.y < 63 for row in observed)
    assert output_counts(observed) == expected


def test_quantum_named_sampling_counts_and_reproducibility() -> None:
    program = build_program(10)
    observed = quantum_samples(program, 1913)
    assert observed == quantum_samples(program, 1913)
    assert sum(row.count for row in observed) == 500
    counts = output_counts(observed)
    theta = math.asin(math.sqrt(6 / 4096))
    probability = math.sin(21 * theta) ** 2
    assert abs(counts["c"] - 500 * probability) <= 6 * math.sqrt(
        500 * probability * (1 - probability)
    )
    assert all(row.output == foo(row.x, row.y) and row.count > 0 for row in observed)


def test_source_mapping_and_frozen_parent() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = (
        root / "examples/synthesis/applications/cybersecurity/whitebox_fuzzing/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    assert [c["index"] for c in proof["reviewed_code_cells"]] == [
        4,
        6,
        10,
        12,
        18,
        21,
        23,
        25,
        27,
        29,
        30,
    ]
    assert (
        hashlib.sha256((directory.parent / "main.py").read_bytes()).hexdigest()
        == proof["frozen_quantum_entry_sha256"]
    )
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert "source_format" not in code
    assert "classiq" not in code and ".qmod" not in code


@pytest.mark.parametrize("budget", ("qubits", "workspace", "memory"))
def test_rejects_insufficient_resource_budget(budget: str) -> None:
    with pytest.raises(ValueError, match="budget|workspace|memory|limit|candidate"):
        synth.synthesize(
            main,
            classical_inputs={"r": 10},
            max_qubits=11 if budget == "qubits" else 25,
            max_workspace=0 if budget == "workspace" else 8,
            memory_limit_bytes=1 if budget == "memory" else 64 * 1024**2,
        )


@pytest.mark.parametrize("iterations", (-1, 17))
def test_rejects_out_of_range_iterations(iterations: int) -> None:
    with pytest.raises(ValueError, match="Grover count"):
        build_program(iterations)
