"""Finite-field point translations decomposed into bounded basis transpositions."""

Point = tuple[int, int] | None
P = 7
A = 5
B = 4


def add(left: Point, right: Point) -> Point:
    if left is None:
        return right
    if right is None:
        return left
    x, y = left
    u, v = right
    if x == u and (y + v) % P == 0:
        return None
    slope = (
        (3 * x * x + A) * pow(2 * y, -1, P)
        if left == right
        else (v - y) * pow((u - x) % P, -1, P)
    ) % P
    result = (slope * slope - x - u) % P
    return result, (slope * (x - result) - y) % P


def encode(point: Point) -> int:
    return 63 if point is None else point[0] + 8 * point[1]


def translation_steps(addend: Point) -> list[int]:
    points: list[Point] = [None] + [
        (x, y)
        for y in range(P)
        for x in range(P)
        if (y * y - x * x * x - A * x - B) % P == 0
    ]
    mapping = list(range(64))
    for pt in points:
        mapping[encode(pt)] = encode(add(pt, addend))
    assert sorted(mapping) == list(range(64))
    seen: set[int] = set()
    steps = []
    for origin in range(64):
        if origin in seen:
            continue
        cycle = []
        current = origin
        while current not in seen:
            seen.add(current)
            cycle.append(current)
            current = mapping[current]
        for end in cycle[1:]:
            start = cycle[0]
            path = [start]
            for bit in range(6):
                if (start ^ end) >> bit & 1:
                    start ^= 1 << bit
                    path.append(start)
            edges = list(zip(path, path[1:], strict=False))
            edges = edges + list(reversed(edges[:-1]))
            for left, right in edges:
                target = (left ^ right).bit_length() - 1
                swapped = left
                if target != 5 and ((left >> target) ^ (left >> 5)) & 1:
                    swapped ^= (1 << target) | (1 << 5)
                steps.append(target * 64 + (swapped & 31))
    return [
        entry
        for step in steps
        for entry in (step // 64, *[(step % 64 >> bit) & 1 for bit in range(5)])
    ]
