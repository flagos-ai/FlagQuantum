"""Resolve the original sparse declaration and replace it with owned operations."""

from dataclasses import replace
from hashlib import sha1
from math import sqrt, tau
from pathlib import Path

import pytest
import torch

from examples.algorithms.sparse_rotation_preparation_catalog import (
    SparseRotationExample,
)
from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.sparse_rotation_preparation import (
    SparseRotationPreparation,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_foreign_phase_addition import frozen_source as frozen_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "size,indices,coefficients,bits",
    [(4, (0, 3), (1, 1j), 2), (8, (0, 4, 6), (1, 2j, 3), 3)],
)
def test_symbolic_sparse_resolution_owned_roundtrip_and_every_logical_column(
    size: int, indices: tuple[int, ...], coefficients: tuple[complex, ...], bits: int
) -> None:
    from attrs import evolve
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.state_preparation import (
        sparse_state_preparation_via_rotations as reference,
    )
    from sympy import Symbol

    raw = Path(reference.__file__).read_bytes()
    assert (
        sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
        == "e5d631263c199129e23c246bfab0fd36fd5f774a"
    )
    n, d, b = (Symbol(name, positive=True, integer=True) for name in ("N", "d", "b"))
    declaration = reference.SparseStatePreparationViaRotations.from_n_coeffs(n, d, b)
    assert declaration.is_symbolic()
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        declaration.decompose_bloq()
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(declaration)
    values = {n: size, d: len(indices), b: bits}
    source = evolve(
        declaration,
        sparse_indices=indices,
        nonzero_coeffs=coefficients,
        N=size,
        phase_bitsize=bits,
        target_bitsize=int(declaration.target_bitsize.subs(values)),
    )
    native = SparseRotationPreparation(indices, coefficients, size, bits)
    example = SparseRotationExample(native, source.target_bitsize)
    from tests.unit.test_rotation_resource_model import assert_source_cost

    assert_source_cost(example.resource_model(), source)
    original = import_register_emitter(source)
    emitter = replace(
        original,
        snapshots=tuple(CircuitIR.from_json(ir.to_json()) for ir in original.snapshots),
    )
    packed = BlackBoxPrepare(
        (("target_state", source.target_bitsize),), (("phase_gradient", bits),), emitter
    )
    logical = example.data_width
    work, slots = emitter.requirements()

    def imported(adjoint: bool) -> CircuitIR:
        return (
            Circuit(logical + work)
            .append(
                packed.make_on(
                    selection=tuple(range(source.target_bitsize)),
                    junk=tuple(range(source.target_bitsize, logical)),
                    workspace=tuple(range(logical, logical + work)),
                    classical_bits=tuple(range(slots)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )

    forward, inverse = imported(False), imported(True)
    actual = Circuit(example.qubit_count).append(example).to_ir()
    channels = (forward, inverse, actual)
    events = tuple(
        certify_erasure_histories(
            ir.instructions,
            n_wires=ir.n_wires,
            zero_wires=tuple(range(logical, ir.n_wires)),
            # Coherent source equality circuits conservatively enlarge Boolean
            # support. Prove branch identity here and final cleanup separately
            # on every logical basis column below; neither obligation is waived.
            check_clean_output=False,
        )
        for ir in channels
    )
    columns = []
    for column in range(1 << logical):
        results = []
        for ir, count in zip((forward, actual), (events[0], events[2]), strict=True):
            stride = 1 << (ir.n_wires - logical)
            initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
            initial[column * stride] = 1
            result = forced_branch(initial, ir.instructions, (0,) * count) * sqrt(
                2**count
            )
            clean = torch.zeros_like(result)
            clean[::stride] = result[::stride]
            torch.testing.assert_close(result, clean, atol=8e-12, rtol=0)
            results.append(result[::stride])
        torch.testing.assert_close(results[0], results[1], atol=8e-12, rtol=0)
        columns.append(results[0])
        start = torch.zeros(1 << inverse.n_wires, dtype=torch.complex128)
        start[:: 1 << (inverse.n_wires - logical)] = results[0]
        back = forced_branch(start, inverse.instructions, (0,) * events[1]) * sqrt(
            2 ** events[1]
        )
        expected = torch.zeros_like(back)
        expected[column << (inverse.n_wires - logical)] = 1
        torch.testing.assert_close(back, expected, atol=1e-11, rtol=0)

    operator = torch.stack(columns, dim=1)
    torch.testing.assert_close(
        operator.mH @ operator,
        torch.eye(1 << logical, dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
    gradient = torch.exp(
        -1j * tau * torch.arange(1 << bits, dtype=torch.float64) / (1 << bits)
    ) / sqrt(1 << bits)
    for target in range(1 << source.target_bitsize):
        initial = torch.zeros(1 << logical, dtype=torch.complex128)
        initial[target << bits : (target + 1) << bits] = gradient
        prepared = (operator @ initial).reshape(1 << source.target_bitsize, 1 << bits)
        state = prepared @ gradient.conj()
        torch.testing.assert_close(
            prepared, state[:, None] * gradient[None, :], atol=1e-11, rtol=0
        )
