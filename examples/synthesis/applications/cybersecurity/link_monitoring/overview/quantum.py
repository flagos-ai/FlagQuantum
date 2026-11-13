"""Original link-monitoring model with resource node one fixed to one."""

from flagquantum.synth import RX, H, Output, QArray, QBit, X, allocate, phase, qfunc


@qfunc
def main(parameters: list[float], target: Output[QArray[QBit]]) -> None:
    allocate(12, target)
    X(target[0])
    for bit in range(1, 12):
        H(target[bit])
    phase(
        target[0]
        + target[1]
        + target[2]
        + target[3]
        + target[4]
        + target[5]
        + target[6]
        + target[7]
        + target[8]
        + target[9]
        + target[10]
        + target[11]
        + 2
        * (
            (1 - target[0]) * (1 - target[1])
            + (1 - target[0]) * (1 - target[2])
            + (1 - target[0]) * (1 - target[3])
            + (1 - target[0]) * (1 - target[4])
            + (1 - target[1]) * (1 - target[2])
            + (1 - target[1]) * (1 - target[3])
            + (1 - target[2]) * (1 - target[6])
            + (1 - target[2]) * (1 - target[11])
            + (1 - target[3]) * (1 - target[5])
            + (1 - target[4]) * (1 - target[10])
            + (1 - target[5]) * (1 - target[7])
            + (1 - target[7]) * (1 - target[8])
            + (1 - target[7]) * (1 - target[9])
        ),
        -parameters[0],
    )
    for bit in range(1, 12):
        RX(parameters[1], target[bit])
