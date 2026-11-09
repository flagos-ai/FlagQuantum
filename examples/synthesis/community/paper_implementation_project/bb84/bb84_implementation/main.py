"""Prepare and rotate all eight BB84 signals with classical integer helpers."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import H, Output, QArray, QBit, X, allocate, qfunc

MESSAGE = (0, 0, 1, 1, 0, 0, 0, 1)
ALICE_BASES = (1, 1, 0, 1, 1, 1, 0, 1)
BOB_BASES = (1, 0, 0, 0, 0, 1, 1, 0)


@qfunc
def bb84(
    message: list[int], alice_bases: list[int], bob_bases: list[int], qba: QArray[QBit]
) -> None:
    for i in range(len(message)):
        if message[i] == 1:
            X(qba[i])
    for i in range(len(alice_bases)):
        if alice_bases[i] == 1:
            H(qba[i])
    for i in range(len(bob_bases)):
        if bob_bases[i] == 1:
            H(qba[i])


@qfunc
def main(qba: Output[QArray[QBit]]) -> None:
    allocate(8, qba)
    bb84(
        [0, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 0],
        qba,
    )


@qfunc
def prepare_key(
    message: list[int],
    alice_bases: list[int],
    bob_bases: list[int],
    qba: Output[QArray[QBit]],
) -> None:
    allocate(8, qba)
    bb84(message, alice_bases, bob_bases, qba)


def build_program(
    message: tuple[int, ...] = MESSAGE,
    alice_bases: tuple[int, ...] = ALICE_BASES,
    bob_bases: tuple[int, ...] = BOB_BASES,
) -> synth.Program:
    """Bind eight binary signals; this example does not measure or certify a key."""
    if any(
        len(values) != 8
        or any(type(value) is not int or value not in (0, 1) for value in values)
        for values in (message, alice_bases, bob_bases)
    ):
        raise ValueError(
            "BB84 example needs three eight-element binary integer sequences"
        )
    return synth.synthesize(
        prepare_key,
        classical_inputs={
            "message": message,
            "alice_bases": alice_bases,
            "bob_bases": bob_bases,
        },
        max_workspace=0,
    )


def run_example() -> None:
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("qba",))))


if __name__ == "__main__":
    run_example()
