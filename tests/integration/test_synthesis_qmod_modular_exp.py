"""Verify bounded modular exponentiation as a coherent permutation."""

import hashlib
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.ecosystem import InteropConversionError

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/modular_exp_example.qmod"
)


def _permutation(modulus: int, base: int, width: int) -> torch.Tensor:
    size = 1 << width
    matrix = torch.zeros((8 * size, 8 * size), dtype=torch.complex128)
    for exponent in range(8):
        for value in range(size):
            result = (
                value * pow(base, exponent, modulus) % modulus
                if value < modulus
                else value
            )
            matrix[exponent * size + result, exponent * size + value] = 1
    return matrix


def test_original_modular_exponentiation_full_state() -> None:
    raw = SOURCE.read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "878bfca97f83fcdbfc1d27efcdd342ea3d53b8d9"
    )
    program = synth.synthesize(SOURCE)
    reference = fq.Circuit(program.circuit.n_wires, dtype=torch.complex128)
    reference.h(0).h(1).h(2).x(5)
    reference.any(*range(6), matrix=_permutation(5, 4, 3))
    result = fq.run(program)
    torch.testing.assert_close(
        result.statevector(), fq.run(reference).statevector(), atol=1e-12, rtol=0
    )
    assert dict(
        program.probabilities(result, registers=("power", "x"))
    ) == pytest.approx(
        {(power, 1 if power % 2 == 0 else 4): 0.125 for power in range(8)}
    )


@pytest.mark.parametrize(("modulus", "base", "width"), [(7, 3, 3), (15, 2, 4)])
def test_other_modular_permutations_preserve_relative_phase(modulus, base, width):
    program = synth.synthesize(
        f"""qfunc main(output power:qnum<3>, output x:qnum<{width}>) {{
            allocate(power); allocate(x);
            hadamard_transform(power); hadamard_transform(x);
            phase(power==3,pi); phase(x==2,pi);
            modular_exp({modulus},{base},x,power);
        }}""",
        source_format="qmod",
    )
    reference = fq.Circuit(program.circuit.n_wires, dtype=torch.complex128)
    for wire in range(3 + width):
        reference.h(wire)
    control_phases = torch.ones(8, dtype=torch.complex128)
    control_phases[3] = -1
    target_phases = torch.ones(1 << width, dtype=torch.complex128)
    target_phases[2] = -1
    reference.any(*range(3), matrix=torch.diag(control_phases))
    reference.any(*range(3, 3 + width), matrix=torch.diag(target_phases))
    reference.any(*range(3 + width), matrix=_permutation(modulus, base, width))
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference).statevector(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("old", "new", "diagnostic"),
    [
        ("modular_exp(5, 4", "modular_exp(6, 4", "coprime"),
        ("modular_exp(5, 4", "modular_exp(9, 4", "fitting the target"),
        ("x, power);", "x, x);", "distinct initialized"),
        ("log(5, 2)", "log(1, 2)", "at least two"),
    ],
)
def test_modular_exponentiation_refuses_invalid_domains(old, new, diagnostic):
    with pytest.raises(InteropConversionError, match=diagnostic):
        synth.synthesize(SOURCE.read_text().replace(old, new, 1), source_format="qmod")
