"""Verify complete native experiments, full proposal batches and Bayesian updates."""

import cmath
import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.integration


def test_all_hypotheses_and_two_absolute_vectors() -> None:
    np = pytest.importorskip("numpy")
    from examples.synthesis.algorithms.search_and_optimization.quantum_likelihood_estimation.overview.quantum import (
        HYPOTHESES,
        batch_state,
        build_experiment,
    )

    identity = torch.eye(2, dtype=torch.complex128)
    matrices = {
        "x": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
        "z": torch.diag(torch.tensor([1, -1], dtype=torch.complex128)),
    }
    controls = np.asarray(((0.31, 0.17, 0.21, 0.61, 0.92), (1.2, 0.7, 1.3, 2.1, 0.2)))
    for hypothesis, (axis, coefficient) in enumerate(HYPOTHESES):
        batch = batch_state(hypothesis, controls)
        for row, binding in enumerate(controls):
            time, alpha, beta, theta, phi = map(float, binding)
            prepared = torch.tensor(
                [math.cos(alpha), cmath.exp(1j * beta) * math.sin(alpha)],
                dtype=torch.complex128,
            )
            rotation = (
                math.cos(coefficient * time) * identity
                - 1j * math.sin(coefficient * time) * matrices[axis]
            )
            lam = math.pi - phi
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            readout = torch.tensor(
                [[c, -cmath.exp(1j * lam) * s], [cmath.exp(1j * phi) * s, -c]],
                dtype=torch.complex128,
            )
            expected = readout @ rotation @ prepared
            actual = (
                fq.run(build_experiment(hypothesis, tuple(map(float, binding))))
                .statevector()
                .flatten()
            )
            torch.testing.assert_close(expected, batch[row], atol=1e-12, rtol=0)
            torch.testing.assert_close(expected, actual, atol=1e-12, rtol=0)


def test_original_source_evaluates_every_proposal_and_updates_actual_shots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("numpy")
    from examples.synthesis.algorithms.search_and_optimization.quantum_likelihood_estimation.overview import (
        workflow,
    )
    from examples.synthesis.algorithms.search_and_optimization.quantum_likelihood_estimation.overview.quantum import (
        RealArray,
    )

    observed: list[int] = []
    from examples.synthesis.algorithms.search_and_optimization.quantum_likelihood_estimation.overview.quantum import (
        likelihoods,
    )

    def counted(controls: RealArray) -> RealArray:
        observed.append(len(controls))
        return likelihoods(controls)

    monkeypatch.setattr(workflow, "likelihoods", counted)
    result = workflow.run_workflow()
    assert len(result.standard) == 16 and len(result.optimized) == 3
    assert observed.count(20) == 8 * len(result.optimized)
    assert observed.count(1) == len(result.standard) + 2 * len(result.optimized)
    for mode in (result.standard, result.optimized):
        assert mode[-1].posterior[0] > 0.99
        weights: tuple[float, ...] = (0.25,) * 4
        for row in mode:
            assert row.outcome in (0, 1)
            values = [p * q for p, q in zip(weights, row.likelihood, strict=True)]
            assert row.posterior == pytest.approx(
                [value / math.fsum(values) for value in values], abs=1e-12
            )
            weights = row.posterior


def test_all_source_stages_are_unexecuted() -> None:
    from examples.synthesis.algorithms.search_and_optimization.quantum_likelihood_estimation.overview import (
        __file__ as path,
    )

    assert path is not None
    root = Path(path).parent
    book = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 21
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
