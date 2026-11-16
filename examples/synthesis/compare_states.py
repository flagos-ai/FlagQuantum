"""Compare declared register amplitudes and clean workspace across producers."""

import torch

import flagquantum as fq
from flagquantum import synth


def compare_states(
    program: synth.Program,
    reference: fq.Circuit,
    registers: tuple[str, ...],
) -> None:
    """Check absolute amplitudes in named register order, including hidden zeros."""
    expected = fq.run(reference).statevector().flatten()
    assert tuple(program.registers) == registers
    state = fq.run(program).statevector().flatten()
    width = state.numel().bit_length() - 1
    logical = tuple(wire for name in registers for wire in program.registers[name])
    assert 1 << len(logical) == expected.numel()
    hidden = tuple(wire for wire in range(width) if wire not in logical)
    ordered = (
        state.reshape((2,) * width)
        .permute((*logical, *hidden))
        .reshape(expected.numel(), -1)
    )
    torch.testing.assert_close(ordered[:, 0], expected, atol=1e-12, rtol=0)
    assert float(ordered[:, 1:].abs().square().sum()) < 1e-24
    print(program.report["resources"])
    print("Native qfunc and independent Circuit amplitudes agree.")
