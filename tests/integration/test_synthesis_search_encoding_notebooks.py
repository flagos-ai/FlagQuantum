"""Full native search, arithmetic, encoding and partial-uniform lessons."""

import importlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.data_encoding.encode_in_angle.comparisons.circuit import (
    reference_circuit as angle_reference,
)
from examples.synthesis.functions.data_encoding.encode_on_bloch.comparisons.circuit import (
    reference_circuit as bloch_reference,
)
from examples.synthesis.functions.grover_operator.comparisons.circuit import (
    reference_circuit as grover_reference,
)
from examples.synthesis.tutorials.technology_demonstrations.arithmetic_expressions.comparisons.circuit import (
    reference_circuit as arithmetic_reference,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    ("functions/grover_operator", 3, (3, 5), 1),
    ("tutorials/technology_demonstrations/arithmetic_expressions", 2, (3, 6, 8), 2),
    ("functions/data_encoding/variational_data_encoding", 4, (4, 5, 10, 11), 4),
    (
        "functions/state_preparation/prepare_partial_uniform_state",
        6,
        (2, 3, 4, 7, 8, 9),
        2,
    ),
)


def notebook(relative: str) -> Path:
    return ROOT / "examples/synthesis" / relative / "overview/walkthrough.ipynb"


def test_grover_full_state_and_sampled_predicate_postprocessing() -> None:
    namespace, cells = execute_code_cells(notebook(WORKFLOWS[0][0]))
    assert cells == 3
    program = namespace["program"]
    assert isinstance(program, synth.Program)
    compare_states(program, grover_reference(), ("x", "y"))
    probabilities = dict(program.probabilities(fq.run(program), registers=("x", "y")))
    success = sum(
        probability
        for (x, y), probability in probabilities.items()
        if x + y < 4 and (x * y) % 4 == 2
    )
    assert success == pytest.approx(math.sin(5 * math.asin(math.sqrt(2 / 16))) ** 2)
    rows = cast(list[dict[str, int | bool]], namespace["rows"])
    assert sum(row["counts"] for row in rows) == 1000
    for row in rows:
        x, y = row["x"], row["y"]
        assert (x, y) in probabilities
        assert row["predicate"] is (x + y < 4 and (x * y) % 4 == 2)


def test_arithmetic_both_depth_objective_width_budgets() -> None:
    namespace, cells = execute_code_cells(notebook(WORKFLOWS[1][0]))
    assert cells == 2
    programs = cast(dict[int, synth.Program], namespace["programs"])
    assert set(programs) == {10, 12}
    expected = {
        (x, y, int(2 * x + y + max(3 * y, 2) > 4)): 1 / 8
        for x in range(4)
        for y in range(2)
    }
    for width, program in programs.items():
        assert isinstance(program, synth.Program)
        compare_states(program, arithmetic_reference(), ("x", "y", "z"))
        probabilities = dict(
            program.probabilities(fq.run(program), registers=("x", "y", "z"))
        )
        assert probabilities == pytest.approx(expected, abs=1e-12)
        resources = program.report["resources"]
        assert isinstance(resources, ResourceEstimate)
        assert resources.n_qubits <= width
    counts = cast(dict[tuple[int, ...], int], namespace["counts"])
    assert sum(counts.values()) == 1000 and set(counts) == set(expected)


@pytest.mark.parametrize("mode", ["angles", "bloch"])
def test_encoding_both_nonzero_bindings_and_odd_bloch_tail(mode: str) -> None:
    namespace, cells = execute_code_cells(notebook(WORKFLOWS[2][0]))
    assert cells == 4
    data = cast(list[list[float]], namespace[f"{mode}_data"])
    programs = cast(list[synth.Program], namespace[f"{mode}_programs"])
    assert len(data) == len(programs) == 2 and data[0] != data[1]
    reference = angle_reference if mode == "angles" else bloch_reference
    expected_length = 4 if mode == "angles" else 7
    for vector, program in zip(data, programs, strict=True):
        assert len(vector) == expected_length
        assert all(math.isfinite(value) and value != 0 for value in vector)
        assert isinstance(program, synth.Program)
        assert len(program.registers["x"]) == 4
        compare_states(program, reference(vector), ("x",))


def interval_reference(start: int, stop: int, width: int) -> fq.Circuit:
    amplitudes = torch.zeros(1 << width, dtype=torch.complex128)
    amplitudes[start:stop] = 1 / math.sqrt(stop - start)
    identity = torch.eye(1 << width, dtype=torch.complex128)
    delta = identity[:, 0] - amplitudes
    reflection = identity - 2 * torch.outer(delta, delta.conj()) / delta.norm().square()
    return fq.Circuit(width, dtype=torch.complex128).any(
        *range(width), matrix=reflection
    )


def test_both_uniform_states_samples_and_plot_contents() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    try:
        namespace, cells = execute_code_cells(notebook(WORKFLOWS[3][0]))
        assert cells == 6
        programs = cast(dict[str, synth.Program], namespace["programs"])
        figures = cast(dict[str, Figure], namespace["figures"])
        for name, start, stop, width, size in (
            ("trimmed", 0, 3, 4, (4, 3)),
            ("interval", 2, 8, 5, (5, 3)),
        ):
            program = programs[name]
            assert isinstance(program, synth.Program)
            assert len(program.registers["x"]) == width
            compare_states(program, interval_reference(start, stop, width), ("x",))
            probabilities = cast(
                dict[tuple[int, ...], float], namespace[f"{name}_probabilities"]
            )
            assert probabilities == pytest.approx(
                {(value,): 1 / (stop - start) for value in range(start, stop)},
                abs=1e-12,
            )
            counts = cast(dict[tuple[int, ...], int], namespace[f"{name}_counts"])
            assert sum(counts.values()) == 1000
            assert set(counts) == {(value,) for value in range(start, stop)}
            figure = figures[name]
            assert isinstance(figure, Figure)
            assert tuple(figure.get_size_inches()) == size
            axes = figure.axes[0]
            assert (axes.get_xlabel(), axes.get_ylabel()) == ("state", "shots")
            assert len(axes.patches) == stop - start
            for patch, ((value,), count) in zip(
                axes.patches, sorted(counts.items()), strict=True
            ):
                assert isinstance(patch, Rectangle)
                assert patch.get_x() + patch.get_width() / 2 == pytest.approx(value)
                assert patch.get_height() == count
    finally:
        plt.close("all")


@pytest.mark.parametrize("relative,cells,source_cells,stages", WORKFLOWS)
def test_python_launcher_runs_every_stage(
    relative: str,
    cells: int,
    source_cells: tuple[int, ...],
    stages: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if "partial_uniform" in relative:
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg", force=True)
    launcher = importlib.import_module(
        "examples.synthesis." + relative.replace("/", ".") + ".overview.main"
    )
    run = cast(Callable[[], None], launcher.run_example)
    try:
        run()
        output = capsys.readouterr().out
        assert output.count("Probabilities:") == stages
        assert output.count("Resources:") == stages
        if "grover_operator" in relative:
            assert "'predicate': True" in output
        if "arithmetic_expressions" in relative:
            assert "Width budget: 10" in output and "Width budget: 12" in output
    finally:
        if "partial_uniform" in relative:
            import matplotlib.pyplot as plt

            plt.close("all")


@pytest.mark.parametrize("relative,cells,source_cells,stages", WORKFLOWS)
def test_original_cell_provenance_and_clean_native_sources(
    relative: str, cells: int, source_cells: tuple[int, ...], stages: int
) -> None:
    path = notebook(relative)
    proof = json.loads(path.with_name("provenance.json").read_text())
    assert tuple(cell["index"] for cell in proof["reviewed_code_cells"]) == source_cells
    assert all(len(cell["sha256"]) == 64 for cell in proof["reviewed_code_cells"])
    document = json.loads(path.read_text())
    assert sum(cell["cell_type"] == "code" for cell in document["cells"]) == cells
    assert len({cell["id"] for cell in document["cells"]}) == len(document["cells"])
    for cell in document["cells"]:
        source = "".join(cell["source"])
        assert "from classiq" not in source and "source_format" not in source
        assert "Staged prototype" not in source
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
