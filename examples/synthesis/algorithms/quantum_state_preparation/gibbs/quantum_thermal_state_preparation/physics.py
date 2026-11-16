"""Retained Gaussian/thermal data and bounded jump-selection operators."""

GAUSSIAN = (
    0.008448505968335,
    0.022863578200213,
    0.053671305267896,
    0.10928832979566,
    0.193036348649671,
    0.295759070777593,
    0.393070748484186,
    0.453144853594257,
    0.453144853594257,
    0.393070748484186,
    0.295759070777593,
    0.193036348649671,
    0.10928832979566,
    0.053671305267896,
    0.022863578200213,
    0.008448505968335,
)
ANGLES = (
    1.570796326794897,
    1.508336977207666,
    1.446120581805428,
    1.384385401181443,
    1.323360527812465,
    1.263261830882205,
    1.204288485710753,
    1.14662020638434,
    2.051177405928626,
    1.994972447205453,
    1.93730416787904,
    1.878330822707588,
    1.818232125777328,
    1.75720725240835,
    1.695472071784365,
    1.633255676382128,
)
_X = ((0, 1), (1, 0))
_Y = ((0, -1j), (1j, 0))
_I = ((1, 0), (0, 1))


def _selection(
    blocks: tuple[tuple[tuple[complex | float, ...], ...], ...],
) -> tuple[tuple[complex | float, ...], ...]:
    return tuple(
        tuple(
            blocks[row // 2][row % 2][column % 2] if row // 2 == column // 2 else 0
            for column in range(8)
        )
        for row in range(8)
    )


# The first state bit receives X/Y for jump labels 0/1; the other for 2/3.
SELECT_FIRST = _selection((_X, _Y, _I, _I))
SELECT_SECOND = _selection((_I, _I, _X, _Y))
