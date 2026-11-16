"""Generate all nine source CFL events and their active magnitudes."""

from fractions import Fraction

GRID = 8
MAGNITUDES = (0, 1, 2, 3)
EVENTS = 9
X_LIMITS = (2, 5)
Y_LIMITS = (1, 5)
SHOTS = 2048


def schedule() -> tuple[tuple[Fraction, tuple[int, ...]], ...]:
    """Advance to the next crossing of a grid spacing by a source magnitude."""
    crossings = {speed: Fraction(1, speed) for speed in MAGNITUDES if speed}
    events = []
    for _ in range(EVENTS):
        instant = min(crossings.values())
        moving = tuple(
            speed for speed, crossing in crossings.items() if crossing == instant
        )
        for speed in moving:
            crossings[speed] += Fraction(1, speed)
        events.append((instant, moving))
    return tuple(events)


def active_magnitudes() -> tuple[int, ...]:
    """Flatten simultaneous moves in increasing magnitude as in the source."""
    return tuple(speed for _, active in schedule() for speed in active)
