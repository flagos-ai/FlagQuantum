"""Native dyadic XOR agrees with independent permutations on nonzero targets."""

from fractions import Fraction
from typing import Annotated

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration


@qfunc
def update(a: float, b: float, source: QNum, target: QNum) -> None:
    target ^= a * source + b


@qfunc
def main(
    a: float,
    b: float,
    initial: int,
    repeats: int,
    x: Output[Annotated[QNum, (4, False, 4)]],
    y: Output[Annotated[QNum, (4, False, 4)]],
) -> None:
    allocate(x)
    hadamard_transform(x)
    allocate(y)
    y ^= initial / 16
    for _ in range(repeats):
        update(a, b, x, y)


@pytest.mark.parametrize("initial", [0, 5, 15])
@pytest.mark.parametrize(
    "a,b", [(0.9375, 0.03125), (-0.375, 0.03125), (0.625, 0.21875)]
)
@pytest.mark.parametrize("repeats", [1, 2])
def test_exact_floor_nonzero_targets_and_inverse(
    initial: int, a: float, b: float, repeats: int
) -> None:
    program = synth.synthesize(
        main, classical_inputs={"a": a, "b": b, "initial": initial, "repeats": repeats}
    )
    circuit = fq.Circuit(8, dtype=torch.complex128)
    for bit in range(4):
        circuit.h(bit)
        if initial >> bit & 1:
            circuit.x(7 - bit)
    matrix = torch.zeros((256, 256), dtype=torch.complex128)
    for x in range(16):
        exact = (Fraction(str(a)) * Fraction(x, 16) + Fraction(str(b))) * 16
        word = (exact.numerator // exact.denominator) & 15 if repeats % 2 else 0
        for target in range(16):
            matrix[x * 16 + (target ^ word), x * 16 + target] = 1
    circuit.any(*range(8), matrix=matrix)
    expected = fq.run(circuit).statevector().flatten()
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    actual = fq.run(program).statevector().reshape(256, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@pytest.mark.parametrize("a", [0.9, 1 / 3])
def test_nondyadic_coefficients_refuse(a: float) -> None:
    with pytest.raises(HybridVerificationError, match="binary rationals"):
        synth.synthesize(
            main, classical_inputs={"a": a, "b": 0.0, "initial": 0, "repeats": 1}
        )


@qfunc
def alias(x: Output[Annotated[QNum, (4, False, 4)]]) -> None:
    allocate(x)
    x ^= x


def test_alias_refuses() -> None:
    with pytest.raises(HybridVerificationError, match="disjoint"):
        synth.synthesize(alias)


@qfunc
def coherent_target(
    x: Output[Annotated[QNum, (4, False, 4)]],
    y: Output[Annotated[QNum, (4, False, 4)]],
) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    synth.phase(x, 0.37)
    synth.phase(y, -0.23)
    update(-0.375, 0.03125, x, y)


def test_complex_coherent_target_full_permutation() -> None:
    program = synth.synthesize(coherent_target)
    expected = torch.zeros(256, dtype=torch.complex128)
    for x in range(16):
        value = (Fraction(-3, 8) * Fraction(x, 16) + Fraction(1, 32)) * 16
        mask = (value.numerator // value.denominator) & 15
        for y in range(16):
            expected[x * 16 + (y ^ mask)] = (
                torch.exp(
                    torch.tensor(
                        1j * (0.37 * x / 16 - 0.23 * y / 16), dtype=torch.complex128
                    )
                )
                / 16
            )
    actual = fq.run(program).statevector().reshape(256, -1)
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


@qfunc
def uninitialized_target(
    x: Output[Annotated[QNum, (4, False, 4)]], y: Output[Annotated[QNum, (4, False, 4)]]
) -> None:
    allocate(x)
    y ^= x - x


def test_cancelled_update_still_checks_target_initialization() -> None:
    with pytest.raises(HybridVerificationError, match="initialize"):
        synth.synthesize(uninitialized_target)


@pytest.mark.parametrize(
    "ports,body",
    [
        (
            "x: Output[Annotated[QNum,(4,True,3)]], y: Output[Annotated[QNum,(4,False,4)]]",
            "allocate(x)\nallocate(y)\ny ^= x*0.5",
        ),
        (
            "x: Output[Annotated[QNum,(4,False,4)]], y: Output[Annotated[QNum,(4,True,3)]]",
            "allocate(x)\nallocate(y)\ny ^= x*0.5",
        ),
        (
            "x: Output[Annotated[QNum,(4,False,4)]], y: Const[Annotated[QNum,(4,False,4)]]",
            "allocate(x)\ny ^= x*0.5",
        ),
        (
            "x: Output[Annotated[QNum,(4,False,4)]], y: Output[Annotated[QNum,(4,False,4)]]",
            "allocate(y)\ny ^= x-x",
        ),
    ],
)
def test_invalid_numeric_borrow_and_cancelled_uninitialized_read_refuse(
    ports: str, body: str
) -> None:
    with pytest.raises(HybridVerificationError):
        synth.synthesize(definition(ports, body, "from typing import Annotated\n"))


def test_integer_literal_remains_raw_mask_on_fractional_target() -> None:
    program = synth.synthesize(
        definition(
            "y: Output[Annotated[QNum,(4,False,4)]]",
            "allocate(y)\ny ^= 5",
            "from typing import Annotated\n",
        )
    )
    assert dict(program.probabilities(fq.run(program), registers=("y",))) == {(5,): 1}
