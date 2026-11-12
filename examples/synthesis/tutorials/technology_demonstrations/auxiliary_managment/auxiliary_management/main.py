"""Prepare every retained signed amplitude with a declared error budget."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, prepare_amplitudes, qfunc

AMPLITUDES = (
    0.245034472820708,
    -0.170081350544172,
    0.167697366233728,
    -0.023905223276522,
    0.343936056437118,
    -0.296692966947323,
    -0.283916724581065,
    0.330582853784989,
    -0.323760293689634,
    0.257046476143475,
    0.153163141682669,
    -0.075162649766642,
    -0.314744872342388,
    -0.249922153985921,
    0.352662065451226,
    -0.015039183255469,
)


@qfunc
def main(amplitudes: list[float], out: Output[QArray[QBit]]) -> None:
    prepare_amplitudes(amplitudes, 0.2, out)


def build_program() -> synth.Program:
    return synth.synthesize(
        main, classical_inputs={"amplitudes": AMPLITUDES}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("out",))))

    from examples.synthesis.tutorials.technology_demonstrations.auxiliary_managment.auxiliary_management.workflow import (
        run_example,
    )

    run_example()
