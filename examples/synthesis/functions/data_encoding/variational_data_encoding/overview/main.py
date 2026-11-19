"""Execute both symbolic-input lessons using explicit demonstration bindings."""

import flagquantum as fq
from examples.synthesis.functions.data_encoding.encode_in_angle.main import (
    build_program as build_angles,
)
from examples.synthesis.functions.data_encoding.encode_on_bloch.main import (
    build_program as build_bloch,
)

ANGLE_DATA = ((0.2, -0.3, 0.4, 0.5), (0.17, 2.63, -2.49, 0.91))
BLOCH_DATA = (
    (0.2, -0.3, 0.4, 0.5, 0.6, -0.7, 0.8),
    (0.17, 2.63, -2.49, 0.91, -0.25, 0.37, 1.58),
)


def run_example() -> None:
    for build, vectors in ((build_angles, ANGLE_DATA), (build_bloch, BLOCH_DATA)):
        for data in vectors:
            program = build(data)
            result = fq.run(program)
            print("Data:", data)
            print(
                "Probabilities:", dict(program.probabilities(result, registers=("x",)))
            )
            print("Resources:", program.report["resources"])


if __name__ == "__main__":
    run_example()
