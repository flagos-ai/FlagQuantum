"""Example-local SU(2) Solovay--Kitaev words with measured operator errors."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

Quaternion = NDArray[np.float64]
Word = tuple[str, ...]


def product(first: Quaternion, second: Quaternion) -> Quaternion:
    return np.concatenate(
        (
            np.array([first[0] * second[0] - first[1:] @ second[1:]]),
            first[0] * second[1:]
            + second[0] * first[1:]
            + np.cross(first[1:], second[1:]),
        )
    )


def inverse(value: Quaternion) -> Quaternion:
    result: Quaternion = value * np.array([1, -1, -1, -1])
    return result


def adjoint_word(word: Word) -> Word:
    names = {"h": "h", "t": "tdg", "tdg": "t"}
    return tuple(names[name] for name in reversed(word))


def canonical(value: Quaternion) -> Quaternion:
    nonzero = np.flatnonzero(np.abs(value) > 1e-12)
    return -value if nonzero.size and value[nonzero[0]] < 0 else value


GATES = {
    "h": np.array([0.0, -1 / np.sqrt(2), 0.0, -1 / np.sqrt(2)]),
    "t": np.array([np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)]),
    "tdg": np.array([np.cos(np.pi / 8), 0.0, 0.0, -np.sin(np.pi / 8)]),
}


@lru_cache(maxsize=1)
def net() -> tuple[NDArray[np.float64], tuple[Word, ...]]:
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    rows: dict[tuple[float, ...], tuple[Quaternion, Word]] = {
        tuple(identity): (identity, ())
    }
    frontier: list[tuple[Quaternion, Word]] = [(identity, ())]
    for _ in range(12):
        next_frontier: list[tuple[Quaternion, Word]] = []
        for value, word in frontier:
            for name, gate in GATES.items():
                candidate = canonical(product(gate, value))
                key = tuple(np.round(candidate, 10))
                if key not in rows:
                    entry = (candidate, word + (name,))
                    rows[key] = entry
                    next_frontier.append(entry)
        frontier = next_frontier
    values, words = zip(*rows.values(), strict=True)
    return np.stack(values), tuple(words)


def balanced(value: Quaternion) -> tuple[Quaternion, Quaternion]:
    value = value / np.linalg.norm(value)
    if value[0] < 0:
        value = -value
    magnitude = float(np.linalg.norm(value[1:]))
    if magnitude < 1e-14:
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        return identity, identity
    deficit = 1 - float(value[0])
    if deficit <= 0:
        deficit = magnitude * magnitude / (1 + float(value[0]))
    sine = (deficit / 2) ** 0.25
    cosine = np.sqrt(1 - sine * sine)
    first = np.array([cosine, sine, 0.0, 0.0])
    second = np.array([cosine, 0.0, sine, 0.0])
    commutator = product(
        product(product(first, second), inverse(first)), inverse(second)
    )
    a = commutator[1:] / np.linalg.norm(commutator[1:])
    b = value[1:] / magnitude
    dot = float(np.clip(a @ b, -1, 1))
    if dot < -1 + 1e-12:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-12:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        rotation = np.concatenate((np.array([0.0]), axis / np.linalg.norm(axis)))
    else:
        rotation = np.concatenate((np.array([1 + dot]), np.cross(a, b)))
        rotation /= np.linalg.norm(rotation)
    return (
        product(product(rotation, first), inverse(rotation)),
        product(product(rotation, second), inverse(rotation)),
    )


@dataclass(frozen=True)
class Approximation:
    word: Word
    quaternion: Quaternion
    iterations: int
    error: float


def _nearest_index(overlaps: Quaternion) -> int:
    # Equivalent words can differ by BLAS roundoff. Resolve overlap ties in
    # insertion order at the same 1e-12 precision used to validate the target.
    return int(np.argmax(np.round(np.abs(overlaps), 12)))


def approximate(value: Quaternion, iterations: int) -> Approximation:
    """Return actual H/T/Tdg words; errors are computed, never assumed from depth."""
    if type(iterations) is not int or not 0 <= iterations <= 3:
        raise ValueError("This source experiment admits zero to three SK iterations")
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("The target must be a finite unit quaternion")
    if abs(float(np.linalg.norm(value)) - 1) > 1e-12:
        raise ValueError("The target quaternion must have unit norm")
    values, words = net()

    def recurse(target: Quaternion, depth: int) -> tuple[Quaternion, Word]:
        if depth == 0:
            overlaps = values @ target
            index = _nearest_index(overlaps)
            chosen = values[index] * (1 if overlaps[index] >= 0 else -1)
            return chosen, words[index]
        previous, word = recurse(target, depth - 1)
        delta = product(target, inverse(previous))
        v, w = balanced(delta)
        v_approx, v_word = recurse(v, depth - 1)
        w_approx, w_word = recurse(w, depth - 1)
        result = product(
            product(
                product(product(v_approx, w_approx), inverse(v_approx)),
                inverse(w_approx),
            ),
            previous,
        )
        # Matrix products act right to left; these words are chronological.
        return (
            result,
            word + adjoint_word(w_word) + adjoint_word(v_word) + w_word + v_word,
        )

    actual, word = recurse(value, iterations)
    error = min(
        float(np.linalg.norm(actual - value)), float(np.linalg.norm(actual + value))
    )
    return Approximation(word, actual, iterations, error)
