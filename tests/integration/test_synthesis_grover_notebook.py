"""Independent complete-state and finite-shot references for adaptive Grover."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.grover.grover.main import (
    EDGES,
)
from examples.synthesis.algorithms.search_and_optimization.grover.grover.main import (
    main as graph_search,
)
from examples.synthesis.algorithms.search_and_optimization.grover.three_sat_grover.main import (
    main as large_search,
)
from flagquantum import synth
from flagquantum.synth import SynthesisError

try:
    from examples.synthesis.algorithms.search_and_optimization.grover.grover.overview import (
        workflow as grover,
    )
except ModuleNotFoundError as error:
    if error.name not in {"networkx", "matplotlib"}:
        raise
    pytest.skip(str(error), allow_module_level=True)

pytestmark = pytest.mark.integration
PROBLEMS: tuple[grover.Problem, ...] = ("small_3sat", "large_3sat", "max_cut")


def _marked(problem: grover.Problem, word: int) -> bool:
    if problem == "small_3sat":
        return word in {2, 6}
    if problem == "large_3sat":
        return word in {8, 12}
    # Evaluate crossings independently from the classical and quantum predicates.
    return (
        sum(
            ((word >> a) ^ (word >> b)) & 1
            for a, b in ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))
        )
        >= 4
    )


def _reference(problem: grover.Problem, repetitions: int) -> fq.Circuit:
    n = grover.width(problem)
    size = 1 << n
    oracle = torch.diag(
        torch.tensor(
            [-1 if _marked(problem, word) else 1 for word in range(size)],
            dtype=torch.complex128,
        )
    )
    reflection = torch.full((size, size), 2 / size, dtype=torch.complex128) - torch.eye(
        size, dtype=torch.complex128
    )
    circuit = fq.Circuit(n, dtype=torch.complex128)
    for wire in range(n):
        circuit.h(wire)
    for _ in range(repetitions):
        circuit.any(*range(n), matrix=oracle)
        circuit.any(*range(n), matrix=reflection)
    return circuit


@pytest.mark.parametrize("problem", PROBLEMS)
@pytest.mark.parametrize("repetitions", range(4))
def test_full_absolute_states(problem: grover.Problem, repetitions: int) -> None:
    program = grover.build_program(problem, repetitions)
    expected = fq.run(_reference(problem, repetitions)).statevector()
    actual = fq.run(program).statevector()
    assert isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual.flatten(), expected.flatten(), atol=1e-11, rtol=0)
    assert program.circuit.n_wires == grover.width(problem)
    assert program.report["workspace_qubits"] == 0
    for bits, value in grover.truth_table(problem):
        word = sum(bit << i for i, bit in enumerate(bits))
        assert value == _marked(problem, word)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_every_adaptive_sample(problem: grover.Problem) -> None:
    search = grover.run_search(problem, seed=1100 + 100 * PROBLEMS.index(problem))
    assert search.succeeded
    for attempt in search.attempts:
        state = fq.run(_reference(problem, attempt.repetitions)).statevector()
        assert isinstance(state, torch.Tensor)
        draws = torch.multinomial(
            state.flatten().abs().square(),
            1000,
            replacement=True,
            generator=torch.Generator().manual_seed(attempt.seed),
        )
        expected = torch.bincount(draws, minlength=1 << grover.width(problem))
        counts = {
            sum(bit << i for i, bit in enumerate(row.bits)): row.count
            for row in attempt.outcomes
        }
        assert counts == {
            word: count for word, count in enumerate(expected.tolist()) if count
        }
        assert sum(row.count for row in attempt.outcomes) == 1000
        assert all(row.probability == row.count / 1000 for row in attempt.outcomes)
        assert all(
            row.marked == _marked(problem, word)
            for word, row in (
                (sum(bit << i for i, bit in enumerate(row.bits)), row)
                for row in attempt.outcomes
            )
        )
    assert grover.exceeds_threshold(search.attempts[-1].outcomes, search.threshold)
    assert all(
        not grover.exceeds_threshold(a.outcomes, search.threshold)
        for a in search.attempts[:-1]
    )


@pytest.mark.parametrize("problem", PROBLEMS)
def test_exhaustion_duplicate_and_strict_row_threshold(problem: grover.Problem) -> None:
    search = grover.run_search(problem, threshold=1.0)
    assert not search.succeeded
    assert [a.loop_index for a in search.attempts] == [0, 1, 3]
    assert [a.repetitions for a in search.attempts] == [1, 2, 3]
    rows = (
        grover.Outcome((0,), 450, 0.45, True),
        grover.Outcome((1,), 450, 0.45, True),
    )
    assert not grover.exceeds_threshold(rows, 0.45)
    assert grover.exceeds_threshold(rows, 0.449)
    assert not grover.exceeds_threshold((grover.Outcome((0,), 1000, 1.0, False),), 0.1)
    limited = grover.run_search(problem, threshold=1.0, max_iter=1)
    assert not limited.succeeded and len(limited.attempts) == 1


@pytest.mark.parametrize("count", [-1, 4, True, 1.5])
def test_repetition_refusal(count: int) -> None:
    with pytest.raises(ValueError, match="repetition"):
        grover.build_program("small_3sat", count)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_threshold_refusal(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        grover.run_search("small_3sat", threshold=threshold)


@pytest.mark.parametrize("problem", PROBLEMS)
def test_actual_budget_refusal(problem: grover.Problem) -> None:
    entry = {
        "small_3sat": grover.small_search,
        "large_3sat": large_search,
        "max_cut": graph_search,
    }[problem]
    with pytest.raises(SynthesisError, match="max_qubits"):
        synth.synthesize(
            entry, classical_inputs={"r": 1}, max_qubits=grover.width(problem) - 1
        )


def test_graph_plot(tmp_path: Path) -> None:
    search = grover.run_search("max_cut", seed=1300)
    figure = grover.plot_graph(search)
    assert len(figure.axes) == 2
    rows = search.attempts[-1].outcomes
    assert rows[0].count == max(row.count for row in rows)
    bits = rows[0].bits
    array = figure.axes[1].collections[0].get_array()
    assert array is not None
    assert list(array) == list(bits)
    expected_widths = [float(bits[a] != bits[b]) + 0.5 for a, b in EDGES]
    widths = figure.axes[1].collections[1].get_linewidth()
    assert not isinstance(widths, float)
    assert list(widths) == expected_widths
    figure.savefig(tmp_path / "grover.png")
    assert (tmp_path / "grover.png").stat().st_size > 10000


def test_clean_complete_notebook() -> None:
    import json

    from tests.integration.synthesis_notebook_support import execute_code_cells

    root = Path(__file__).resolve().parents[2]
    directory = (
        root
        / "examples/synthesis/algorithms/search_and_optimization/grover/grover/overview"
    )
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
    namespace, executed = execute_code_cells(directory / "walkthrough.ipynb")
    assert executed == 6
    for name, problem, seed in (
        ("small", "small_3sat", 1100),
        ("large", "large_3sat", 1200),
        ("graph", "max_cut", 1300),
    ):
        search = namespace[name]
        assert isinstance(search, grover.Search)
        assert search.problem == problem and search.succeeded
        assert search.attempts[0].seed == seed
        assert all(
            sum(row.count for row in a.outcomes) == 1000 for a in search.attempts
        )
    failed = namespace["not_found"]
    assert isinstance(failed, grover.Search) and not failed.succeeded
    assert [a.repetitions for a in failed.attempts] == [1, 2, 3]
    proof = json.loads((directory / "provenance.json").read_text())
    assert (
        proof["upstream_path"]
        == "algorithms/search_and_optimization/grover/grover.ipynb"
    )
    assert len(proof["reviewed_code_cells"]) == 18
    assert proof["native_notebook"] == str(
        (directory / "walkthrough.ipynb").relative_to(root)
    )


@pytest.mark.parametrize("shots", [0, -1, 100001, True])
def test_shot_budget_refusal(shots: int) -> None:
    with pytest.raises(ValueError, match="shot"):
        grover.run_search("small_3sat", shots=shots)


@pytest.mark.parametrize("iterations", [0, -1, 5, True])
def test_classical_loop_budget_refusal(iterations: int) -> None:
    with pytest.raises(ValueError, match="loop budget"):
        grover.run_search("small_3sat", max_iter=iterations)


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_seed_refusal(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        grover.run_search("small_3sat", seed=seed)


@pytest.mark.parametrize(
    "counts",
    [
        torch.zeros(8, dtype=torch.int64),
        torch.ones(7, dtype=torch.int64),
        -torch.ones(8, dtype=torch.int64),
        torch.ones(8),
    ],
)
def test_count_vector_refusal(counts: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="count vector"):
        grover.observed_outcomes("small_3sat", counts)


@pytest.mark.parametrize("bits", [(0, 1), (0, 1, 2), (False, 1, 0)])
def test_logical_assignment_refusal(bits: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="binary"):
        grover.predicate("small_3sat", bits)


def test_result_ties_and_plot_refusal() -> None:
    counts = torch.tensor([2, 0, 2, 0, 0, 0, 0, 0], dtype=torch.int64)
    rows = grover.observed_outcomes("small_3sat", counts)
    assert [row.bits for row in rows] == [(0, 0, 0), (0, 1, 0)]
    assert not rows[0].marked and rows[1].marked
    with pytest.raises(ValueError, match="max_cut"):
        grover.plot_graph(grover.Search("small_3sat", 0.45, (), False))
