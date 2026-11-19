"""Regenerate the three synthetic sets and the full binary Iris workflow."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
LabelArray = NDArray[np.int64]


@dataclass(frozen=True)
class Dataset:
    points: FloatArray
    labels: LabelArray


@dataclass(frozen=True)
class Split:
    train: Dataset
    test: Dataset
    predict: Dataset


def synthetic() -> Split:
    """Preserve the source's per-feature draw order, clipping and scale."""
    centres = ((1.23016026, 1.72327701), (3.20331931, 5.32365722))
    rng = np.random.RandomState(0)
    sets = []
    for _ in range(3):
        points = np.concatenate(
            [
                np.column_stack([rng.normal(center, 0.5, 20) for center in pair])
                for pair in centres
            ]
        )
        points = np.clip(points, 0, 2 * math.pi)
        scaled = (2 * points - 6 * math.pi) / (6 * math.pi)
        labels = np.repeat(np.arange(len(centres), dtype=np.int64), 20)
        sets.append(Dataset(scaled, labels))
    return Split(*sets)


def iris() -> Split:
    """Load full original Iris petal columns and match both seeded stratified splits."""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    data, target = load_iris(return_X_y=True)
    keep = target != 0
    features = data[keep, 2:]
    labels = target[keep] - 1
    standard = StandardScaler().fit_transform(features)
    scaled = MinMaxScaler(feature_range=(0, 2 * math.pi)).fit_transform(standard)
    train, held, train_labels, held_labels = train_test_split(
        scaled,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    test, predict, test_labels, prediction_labels = train_test_split(
        held,
        held_labels,
        test_size=0.5,
        random_state=42,
        stratify=held_labels,
    )
    return Split(
        Dataset(
            np.asarray(train, dtype=np.float64),
            np.asarray(train_labels, dtype=np.int64),
        ),
        Dataset(
            np.asarray(test, dtype=np.float64), np.asarray(test_labels, dtype=np.int64)
        ),
        Dataset(
            np.asarray(predict, dtype=np.float64),
            np.asarray(prediction_labels, dtype=np.int64),
        ),
    )
