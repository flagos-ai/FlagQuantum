"""Train the three-feature quantum classifier and all four classical baselines."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import flagquantum as fq
from examples.synthesis.algorithms.machine_learning.qsvm.overview.datasets import (
    Dataset,
    Split,
)
from examples.synthesis.algorithms.machine_learning.qsvm.overview.workflow import (
    SHOTS,
    Score,
    StateArray,
    sample_kernel,
)
from examples.synthesis.applications.finance.credit_card_fraud.comparisons.circuit import (
    reference_circuit,
)

from . import model, quantum

SAMPLE_SEED = 1297


@dataclass(frozen=True)
class Run:
    data: model.Data
    score: Score
    classical: tuple[tuple[str, float], ...]
    verification: tuple[float, ...]


def feature_states(dataset: Dataset) -> StateArray:
    return np.asarray(
        [
            quantum.state(tuple(map(float, row))).detach().cpu().numpy()
            for row in dataset.points
        ],
        dtype=np.complex128,
    )


def classify(split: Split) -> Score:
    from sklearn.svm import SVC

    rng = np.random.default_rng(SAMPLE_SEED)
    train = feature_states(split.train)
    matrix, pairs = sample_kernel(train, None, rng)
    classifier = SVC(kernel="precomputed").fit(matrix, split.train.labels)
    support = train[classifier.support_]
    labels = []
    for group in (split.test, split.predict):
        kernel, visits = sample_kernel(feature_states(group), support, rng)
        decision = kernel @ classifier.dual_coef_[0] + classifier.intercept_[0]
        labels.append(
            np.asarray(
                np.where(decision >= 0, classifier.classes_[1], classifier.classes_[0]),
                dtype=np.int64,
            )
        )
        pairs += visits
    test, predict = labels
    return Score(
        float(np.mean(test == split.test.labels)),
        float(np.mean(predict == split.predict.labels)),
        test,
        predict,
        len(support),
        pairs,
    )


def classical(split: Split) -> tuple[tuple[str, float], ...]:
    from sklearn.svm import SVC

    return tuple(
        (
            kernel,
            float(
                SVC(kernel=kernel)
                .fit(split.train.points, split.train.labels)
                .score(split.test.points, split.test.labels)
            ),
        )
        for kernel in model.CLASSICAL_KERNELS
    )


def compare(split: Split) -> tuple[float, ...]:
    """Check both complete overlap states for distinct newly transformed points."""
    errors = []
    for first, second in ((0, 3), (104, 115)):
        left = tuple(map(float, split.train.points[first]))
        right = tuple(map(float, split.train.points[second]))
        program = quantum.overlap(left, right)
        actual = fq.run(program).statevector().flatten()
        expected = fq.run(reference_circuit(left, right)).statevector().flatten()
        error = float((actual - expected).abs().max())
        if error >= 1e-12 or program.report["workspace_qubits"] != 0:
            raise AssertionError(f"Overlap full-state error {error}")
        if (
            abs(
                float(actual[0].abs().square())
                - float(quantum.state(left).vdot(quantum.state(right)).abs().square())
            )
            >= 1e-12
        ):
            raise AssertionError(
                "Batched feature overlap disagrees with the executed kernel"
            )
        errors.append(error)
    return tuple(errors)


def run(path: Path = model.DATA) -> Run:
    data = model.prepare(path)
    return Run(
        data, classify(data.features), classical(data.features), compare(data.features)
    )


def trial_count(result: Run) -> int:
    return result.score.sampled_pairs * SHOTS
