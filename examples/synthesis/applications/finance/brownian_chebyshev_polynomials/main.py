"""Coherent Brownian/Chebyshev recurrence with explicit binary quantization."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Const,
    Output,
    QNum,
    allocate,
    drop,
    hadamard_transform,
    prepare_state,
    qfunc,
    quantize,
)


@qfunc
def two_cos(t: Const[QNum], out: Output[QNum]) -> None:
    out |= quantize(-6.283185307179586 * (t - 0.5), fractional_bits=8, literal_bits=20)


@qfunc
def uk(
    two_x: Const[QNum], previous: Const[QNum], earlier: Const[QNum], out: Output[QNum]
) -> None:
    out |= quantize(two_x * previous - earlier, fractional_bits=8, literal_bits=20)


@qfunc
def sine(t: Const[QNum], out: Output[QNum]) -> None:
    out |= quantize(-5 * (t - 0.5) ** 2 + 1, fractional_bits=8, literal_bits=20)


@qfunc
def series(t: Const[QNum], out: Output[QNum]) -> None:
    a0: QNum = QNum()
    a1: QNum = QNum()
    a2: QNum = QNum()
    a3: QNum = QNum()
    prepare_state([0.5, 0.5], 0, a0)
    prepare_state([0.5, 0.5], 0, a1)
    prepare_state([0.5, 0.5], 0, a2)
    prepare_state([0.5, 0.5], 0, a3)
    cosine: QNum = QNum()
    sin_t: QNum = QNum()
    u0: QNum = QNum()
    u1: QNum = QNum()
    u2: QNum = QNum()
    u3: QNum = QNum()
    two_cos(t, cosine)
    u0 |= 1
    u1 |= cosine
    uk(cosine, u1, u0, u2)
    uk(cosine, u2, u1, u3)
    sine(t, sin_t)
    out |= quantize(
        a0 * t + (0.450158158078553 * sin_t) * (a1 * u1 + a2 * u2 + a3 * u3),
        fractional_bits=8,
        literal_bits=20,
    )
    drop(a0)
    drop(a1)
    drop(a2)
    drop(a3)
    drop(cosine)
    drop(sin_t)
    drop(u0)
    drop(u1)
    drop(u2)
    drop(u3)


@qfunc
def return_to_price_space(
    returns: Const[QNum], t: Const[QNum], out: Output[QNum]
) -> None:
    out |= quantize(2 * returns - t + 1, fractional_bits=8, literal_bits=20)


@qfunc
def main(t: Output[QNum], price: Output[QNum]) -> None:
    b: QNum = QNum()
    allocate(1, t)
    hadamard_transform(t)
    series(t, b)
    return_to_price_space(b, t, price)
    drop(b)


def build_program() -> synth.Program:
    return synth.synthesize(
        main,
        max_qubits=23,
        max_workspace=2,
        max_operations=3000,
        memory_limit_bytes=256 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(
        dict(
            program.numeric_probabilities(
                result, registers=("t", "price"), max_amplitudes=1 << 23
            )
        )
    )
    print(program.report["resources"])
