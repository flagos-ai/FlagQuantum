"""Use every archived transaction and preserve the original deterministic splits."""

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from examples.synthesis.algorithms.machine_learning.qsvm.overview.datasets import (
    Dataset,
    Split,
)

DATA = Path("creditcard.csv")
SOURCE_DIGEST = "cb6baeefbfd24b7b8f865b55efa366e2023b35bb125bc88a9f08a4c03724e5cd"
FEATURES = 29
SPLIT_COUNTS = ((100, 25), (50, 10), (50, 10))
LABELS = ("Nominal", "Fraud")
CLASSICAL_KERNELS = ("linear", "poly", "rbf", "sigmoid")
SEED_TSNE = 20150101


@dataclass(frozen=True)
class Data:
    rows: int
    by_class: tuple[int, int]
    raw: Split
    features: Split


def read(path: Path = DATA) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    buckets: tuple[list[list[float]], list[list[float]]] = ([], [])
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if columns is None or columns[0] != "Time" or columns[-1] != "Class":
            raise ValueError("The source asset must contain Time, features and Class")
        fields = columns[1:-1]
        if len(fields) != FEATURES:
            raise ValueError(
                f"The source asset must contain {FEATURES} feature columns"
            )
        for row in reader:
            label = int(row["Class"])
            if label not in (0, 1):
                raise ValueError("Source labels must be binary")
            point = [float(row[name]) for name in fields]
            if not all(math.isfinite(value) for value in point):
                raise ValueError("The source transaction features must be finite")
            buckets[label].append(point)
    return np.asarray(buckets[0], dtype=np.float64), np.asarray(
        buckets[1], dtype=np.float64
    )


def partition(classes: tuple[NDArray[np.float64], NDArray[np.float64]]) -> Split:
    if len(classes[0]) < 200 or len(classes[1]) < 45:
        raise ValueError(
            "The source split requires at least 200 nominal and 45 fraud rows"
        )
    groups = []
    first, second = 0, 0
    for n0, n1 in SPLIT_COUNTS:
        selected = np.concatenate(
            (classes[0][first : first + n0], classes[1][second : second + n1])
        )
        labels = np.repeat(np.arange(2, dtype=np.int64), (n0, n1))
        groups.append(Dataset(selected, labels))
        first, second = first + n0, second + n1
    return Split(*groups)


def transform(split: Split) -> Split:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    source = (split.train, split.test, split.predict)
    pca = PCA(n_components=3).fit(split.train.points)
    reduced = [pca.transform(group.points) for group in source]
    standard = StandardScaler().fit(reduced[0])
    standardized = [standard.transform(points) for points in reduced]
    scaler = MinMaxScaler((-math.pi, math.pi)).fit(np.concatenate(standardized))
    return Split(
        *(
            Dataset(
                np.asarray(scaler.transform(points), dtype=np.float64), group.labels
            )
            for points, group in zip(standardized, source, strict=True)
        )
    )


def prepare(path: Path = DATA) -> Data:
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_DIGEST:
        raise ValueError(
            "Use the pinned 948-row source CSV listed in the example README"
        )
    classes = read(path)
    raw = partition(classes)
    totals = (len(classes[0]), len(classes[1]))
    return Data(sum(totals), totals, raw, transform(raw))


def embed(group: Dataset) -> NDArray[np.float64]:
    """Apply the source's original t-SNE declaration to one raw split."""
    from sklearn.manifold import TSNE

    return np.asarray(
        TSNE(random_state=SEED_TSNE).fit_transform(group.points), dtype=np.float64
    )
