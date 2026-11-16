"""Bind QML angles statically; no gate wires or bit-string decoding are needed."""

import flagquantum as fq
from flagquantum.synth import RX, Output, QArray, QBit, allocate, qfunc, synthesize


@qfunc
def encoding(theta: float, q: QArray[QBit]) -> None:
    RX(theta, q[0])


@qfunc
def mixing(theta: float, q: QArray[QBit]) -> None:
    RX(theta, q[0])


@qfunc
def main(input_0: float, weight_0: float, res: Output[QArray[QBit]]) -> None:
    allocate(1, res)
    encoding(input_0, res)
    mixing(weight_0, res)


def run_example() -> None:
    program = synthesize(
        main,
        classical_inputs={"input_0": 0.37, "weight_0": -0.81},
        memory_limit_bytes=64 * 1024**2,
    )
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("res",))))


if __name__ == "__main__":
    run_example()
