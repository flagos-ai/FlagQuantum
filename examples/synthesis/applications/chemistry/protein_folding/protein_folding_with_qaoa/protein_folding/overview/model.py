"""Generate the six-residue lattice model and its exact discrete objective."""

from itertools import pairwise

RESIDUES = "ABCDEF"
DIRECTION_BITS = 2 * (len(RESIDUES) - 3)
WIDTH = DIRECTION_BITS + 1
STEPS = ((-1, 1, -1), (-1, -1, 1), (1, -1, -1), (1, 1, 1))


def directions(word: int) -> tuple[int, ...]:
    """Decode the two fixed turns and three free little-endian direction pairs."""
    return (1, 2, *(word >> bit & 3 for bit in range(0, DIRECTION_BITS, 2)))


def energy(word: int) -> int:
    """Evaluate the source penalties, including its reverse empty ranges."""
    turns = directions(word)

    def distance_squared(start: int, stop: int) -> int:
        displacement = [0] * 4
        for site in range(start, stop):
            displacement[turns[site]] += 1 if site % 2 == 0 else -1
        return sum(component**2 for component in displacement)

    overlap = 500 * sum(left == right for left, right in pairwise(turns))
    activity = word >> DIRECTION_BITS & 1
    contact = -5000 + 500 * (distance_squared(0, 5) - 1) ** 2
    contact += 300 * sum(
        (2 - distance_squared(start, stop)) ** 2
        for start, stop in ((4, 0), (6, 0), (-1, 5), (1, 5))
    )
    return overlap + activity * contact


def lattice(word: int) -> tuple[tuple[int, int, int], ...]:
    """Return the source tetrahedral walk from the initial residue."""
    x = y = z = 0
    points = [(x, y, z)]
    for index, turn in enumerate(directions(word)):
        dx, dy, dz = STEPS[turn]
        sign = (-1) ** index
        x, y, z = x + sign * dx, y + sign * dy, z + sign * dz
        points.append((x, y, z))
    return tuple(points)


def costs() -> tuple[int, ...]:
    return tuple(energy(word) for word in range(1 << WIDTH))
