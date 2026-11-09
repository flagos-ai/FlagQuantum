"""Prepare and compose x*x+y using native high-level qfunc."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform


@synth.qfunc
def prepare(x: Output[QNum], y: Output[QNum]) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)


@synth.qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= x * x + y


@synth.qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    prepare(x, y)
    energy(x, y, result)


def run() -> None:
    program = synth.synthesize(main, memory_limit_bytes=64 * 1024**2)
    result = fq.run(program)
    outcomes = dict(program.probabilities(result, registers=("x", "y", "result")))
    expected = {(x, y, x * x + y): 1 / 16 for x in range(4) for y in range(4)}
    assert outcomes.keys() == expected.keys()
    assert all(abs(outcomes[key] - value) < 1e-12 for key, value in expected.items())
    print(outcomes)
    print(program.report["resources"])


if __name__ == "__main__":
    run()
