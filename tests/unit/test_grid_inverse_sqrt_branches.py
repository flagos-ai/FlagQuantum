"""Certify every erasure history of complete grid lookup/interpolation chains."""

from itertools import product

import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.grid_cubic_inverse_sqrt import (
    grid_cubic_inverse_sqrt_requirements,
    with_grid_cubic_inverse_sqrt_ir,
)
from flagquantum.compiler.registers.bits.bitwise import xor_ir
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.test_grid_cubic_inverse_sqrt import cubic_word

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width,precision", ((4, 5), (3, 16)))
def test_complete_chain_all_branch_scalars_and_occupied_targets(
    width: int, precision: int
) -> None:
    work, slots = grid_cubic_inverse_sqrt_requirements(width, precision)
    inputs = tuple(range(width))
    out = tuple(range(width, width + precision + 1))
    spectator = width + len(out)
    scratch = tuple(range(spectator + 1, spectator + 1 + work))
    gates = with_grid_cubic_inverse_sqrt_ir(
        inputs,
        scratch,
        tuple(range(slots)),
        precision=precision,
        consume=lambda estimate: xor_ir(estimate, out),
    )
    total = spectator + 1 + work
    circuit = DynamicCircuit(total).append(FieldOperation(total, gates))
    assert circuit.to_ir().instructions == gates
    # Target wires are only XOR destinations. This structural property extends
    # zero/basis-vector target images to every occupied target in the wide case.
    assert all(
        gate.name == "cx" and gate.wires[1] in out and gate.wires[0] not in out
        for gate in gates
        if set(gate.wires) & set(out)
    )
    targets = (
        tuple(range(1 << len(out)))
        if precision == 5
        else (0, *(1 << bit for bit in range(len(out))), (1 << len(out)) - 1)
    )
    before, expected = [], []
    for distance, target, bystander in product(range(1 << width), targets, (0, 1)):
        logical = (distance << (len(out) + 1)) | (target << 1) | bystander
        result = target ^ cubic_word(distance, precision)
        output = (distance << (len(out) + 1)) | (result << 1) | bystander
        before.append(
            tuple((logical << work) >> bit & 1 for bit in reversed(range(total)))
        )
        expected.append(
            tuple((output << work) >> bit & 1 for bit in reversed(range(total)))
        )
    observed, measurements = certified_and_channel_map(gates, tuple(before))
    assert observed == tuple(expected)
    assert measurements == sum(g.name == "measure" for g in gates)
    assert measurements > 1000
    assert all(not any(word[-work:]) for word in observed)
    # The certificate checks each literal compute and erasure block and its
    # actual And invariant. All combined raw histories have this same positive
    # permutation times 2**(-measurements/2), hence the same complex amplitudes.
