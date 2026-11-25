"""Native coefficient-quantized autocallable preparation and amplification."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QBit, QNum, allocate, drop, phase, qfunc

from .operations import prepare, reflect
from .physics import DISTRIBUTION, EXPONENTIAL, predicate_bounds


@qfunc
def main(
    k: int,
    probabilities: list[float],
    exponential: list[float],
    bounds: list[int],
    indicator: Output[QBit],
) -> None:
    x0: QNum = QNum()
    x1: QNum = QNum()
    x2: QNum = QNum()
    aux: QNum = QNum()
    allocate(2, x0)
    allocate(2, x1)
    allocate(2, x2)
    allocate(4, aux)
    allocate(indicator)
    prepare(
        probabilities,
        exponential,
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        bounds[4],
        bounds[5],
        bounds[6],
        x0,
        x1,
        x2,
        aux,
        indicator,
    )
    for iteration in range(k):
        phase(indicator, 3.141592653589793)
        reflect(
            probabilities,
            exponential,
            bounds[0],
            bounds[1],
            bounds[2],
            bounds[3],
            bounds[4],
            bounds[5],
            bounds[6],
            x0,
            x1,
            x2,
            aux,
            indicator,
        )
    drop(x0)
    drop(x1)
    drop(x2)
    drop(aux)


def build_program(k: int = 1, *, shift: float = 0.1274075) -> synth.Program:
    """Build the finite source model at its original two-fractional-bit precision."""
    if type(k) is not int or not 0 <= k <= 4:
        raise ValueError("Iteration count must be an integer in 0..4")
    return synth.synthesize(
        main,
        classical_inputs={
            "k": k,
            "probabilities": list(DISTRIBUTION),
            "exponential": list(EXPONENTIAL),
            "bounds": predicate_bounds(shift),
        },
        max_qubits=24,
        memory_limit_bytes=1024**3,
        max_operations=100_000,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(
        dict(
            program.probabilities(
                result, registers=("indicator",), max_amplitudes=2**24
            )
        )
    )
    print(program.report["resources"])
