"""Native high-level qct_type2 over the complete retained amplitude vector."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QNum, prepare_amplitudes, qct_type2, qfunc

AMPLITUDES = (
    0.112068721799421,
    0.107745452355844,
    0.326427301057128,
    0.326596924449695,
    0.389587541071017,
    0.521638625386058,
    0.444890198893713,
    0.375231379859592,
)


@qfunc
def main(x: Output[QNum]) -> None:
    prepare_amplitudes(
        [
            0.112068721799421,
            0.107745452355844,
            0.326427301057128,
            0.326596924449695,
            0.389587541071017,
            0.521638625386058,
            0.444890198893713,
            0.375231379859592,
        ],
        0.0,
        x,
    )
    qct_type2(x)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",))))
