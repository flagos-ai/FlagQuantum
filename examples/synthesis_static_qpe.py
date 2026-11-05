"""One-bit phase estimation with Qmod, existing scalar qfunc, and Circuit.

The native example composes existing operations. It does not advertise a native
array QPE export; the three unchanged full QPE originals have a separate validator.
"""

import math

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    prepare_uniform_interval_state,
    qfunc,
)

ANGLE = math.pi / 2


@qfunc
def estimate(counter: Output[QBit], data: Output[QBit]) -> None:
    allocate(counter)
    allocate(data)
    prepare_uniform_interval_state(1, 2, data)
    hadamard_transform(counter)
    phase(counter * data, ANGLE)
    hadamard_transform(counter)


QMOD = """qfunc main(output counter:qnum<1,UNSIGNED,1>,output data:qbit) {
 allocate(counter);allocate(data);X(data);
 qpe(lambda(){PHASE(1.5707963267948966,data);},counter);
}"""


def main() -> None:
    reference = fq.Circuit(2, dtype=torch.complex128).x(1).h(0)
    reference.any(
        0, 1, matrix=torch.diag(torch.tensor([1, 1, 1, 1j], dtype=torch.complex128))
    ).h(0)
    expected = fq.run(reference).statevector().reshape(-1)
    programs = (
        synth.synthesize(estimate, memory_limit_bytes=1024**2),
        synth.synthesize(QMOD, source_format="qmod", memory_limit_bytes=1024**2),
    )
    for program in programs:
        result = fq.run(program)
        state = result.statevector().reshape(4, -1)
        torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
        assert state[:, 1:].abs().square().sum() < 1e-24
        print(dict(program.probabilities(result, registers=("counter",))))
        print(program.report["resources"])


if __name__ == "__main__":
    main()
