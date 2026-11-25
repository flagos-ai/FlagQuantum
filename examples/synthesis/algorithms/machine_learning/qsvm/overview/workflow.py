"""Fit support vector classifiers from measured local feature states."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from . import datasets, quantum

SHOTS = 2048
SAMPLE_SEED = 107
StateArray = NDArray[np.complex128]


@dataclass(frozen=True)
class Score:
    test_accuracy: float
    prediction_accuracy: float
    test_labels: datasets.LabelArray
    predictions: datasets.LabelArray
    support_size: int
    sampled_pairs: int


@dataclass(frozen=True)
class Result:
    synthetic: datasets.Split
    iris: datasets.Split
    synthetic_bloch: Score
    iris_pauli: Score
    iris_bloch: Score
    iris_rbf: Score


def feature_states(kind: quantum.FeatureMap, dataset: datasets.Dataset) -> StateArray:
    return np.asarray(
        [
            quantum.state(kind, (float(x), float(y))).detach().cpu().numpy()
            for x, y in dataset.points
        ],
        dtype=np.complex128,
    )


def exact_kernel(left: StateArray, right: StateArray) -> datasets.FloatArray:
    overlaps = left.conjugate() @ right.T
    return np.asarray(np.clip(np.abs(overlaps) ** 2, 0.0, 1.0), dtype=np.float64)


def sample_kernel(
    left: StateArray,
    right: StateArray | None,
    rng: np.random.Generator,
) -> tuple[datasets.FloatArray, int]:
    """Draw independent all-zero event counts from executed feature probabilities."""
    if right is not None:
        probabilities = exact_kernel(left, right)
        sampled = rng.binomial(SHOTS, probabilities)
        return np.asarray(sampled / SHOTS, dtype=np.float64), probabilities.size
    probabilities = exact_kernel(left, left)
    upper = np.triu_indices(len(left))
    counts = rng.binomial(SHOTS, probabilities[upper])
    matrix = np.empty_like(probabilities)
    matrix[upper] = matrix[upper[1], upper[0]] = counts / SHOTS
    return matrix, len(counts)


def classify(
    kind: quantum.FeatureMap, data: datasets.Split, seed: int = SAMPLE_SEED
) -> Score:
    from sklearn.svm import SVC

    rng = np.random.default_rng(seed)
    train = feature_states(kind, data.train)
    kernel, train_pairs = sample_kernel(train, None, rng)
    classifier = SVC(kernel="precomputed")
    classifier.fit(kernel, data.train.labels)
    support = train[classifier.support_]

    def predict(rows: datasets.Dataset) -> tuple[datasets.LabelArray, int]:
        states = feature_states(kind, rows)
        evaluation, pairs = sample_kernel(states, support, rng)
        decision = evaluation @ classifier.dual_coef_[0] + classifier.intercept_[0]
        choices = np.where(
            decision >= 0, classifier.classes_[1], classifier.classes_[0]
        )
        return np.asarray(choices, dtype=np.int64), pairs

    test_labels, test_pairs = predict(data.test)
    predictions, prediction_pairs = predict(data.predict)
    return Score(
        float(np.mean(test_labels == data.test.labels)),
        float(np.mean(predictions == data.predict.labels)),
        test_labels,
        predictions,
        len(support),
        train_pairs + test_pairs + prediction_pairs,
    )


def classical(data: datasets.Split) -> Score:
    from sklearn.svm import SVC

    classifier = SVC(kernel="rbf", random_state=42)
    classifier.fit(data.train.points, data.train.labels)
    test = np.asarray(classifier.predict(data.test.points), dtype=np.int64)
    predicted = np.asarray(classifier.predict(data.predict.points), dtype=np.int64)
    return Score(
        float(np.mean(test == data.test.labels)),
        float(np.mean(predicted == data.predict.labels)),
        test,
        predicted,
        len(classifier.support_),
        0,
    )


def run() -> Result:
    synthetic, iris = datasets.synthetic(), datasets.iris()
    return Result(
        synthetic,
        iris,
        classify(quantum.FeatureMap.BLOCH, synthetic),
        classify(quantum.FeatureMap.PAULI, iris),
        classify(quantum.FeatureMap.BLOCH, iris),
        classical(iris),
    )
