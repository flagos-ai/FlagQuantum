"""Bind both original dense symbolic factories to owned executable operations."""

from collections import Counter
from dataclasses import replace
from hashlib import sha1
from math import sqrt, tau
from pathlib import Path

import pytest
import torch

from examples.algorithms.rotation_preparation_catalog import RotationPreparationExample
from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_foreign_phase_addition import frozen_source as frozen_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("factory", ["shape", "phase"])
@pytest.mark.parametrize("controls", [0, 1, 2])
@pytest.mark.parametrize("uncompute", [False, True])
def test_dense_symbolic_factories_full_operators_and_owned_roundtrip(
    factory: str, controls: int, uncompute: bool
) -> None:
    from attrs import evolve
    from qualtran import DecomposeTypeError
    from qualtran.bloqs.state_preparation import (
        state_preparation_via_rotation as reference,
    )

    raw = Path(reference.__file__).read_bytes()
    assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == (
        "2466a11c3303e23c066602dce535750e650f83bf"
    )
    declaration = (
        reference._state_prep_via_rotation_symb.make()
        if factory == "shape"
        else reference._state_prep_via_rotation_symb_phasegrad.make()
    )
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        declaration.decompose_bloq()
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(declaration)
    # Resolve constructor parameters, rather than coercing symbolic widths in an
    # execution callback. The phase-only factory retains its original vector.
    coefficients = (
        (-0.5 - 0.5j, 0, 0.5, -0.5)
        if factory == "shape"
        else declaration.state_coefficients
    )
    bits = 2
    source = evolve(
        declaration,
        state_coefficients=coefficients,
        phase_bitsize=bits,
        control_bitsize=controls,
        uncompute=uncompute,
    )
    example = RotationPreparationExample(coefficients, bits, controls, uncompute)
    from tests.unit.test_rotation_resource_model import assert_source_cost

    assert_source_cost(example.resource_model(), source)
    original = import_register_emitter(source)
    emitter = replace(
        original,
        snapshots=(
            CircuitIR.from_json(original.snapshots[0].to_json()),
            CircuitIR.from_json(original.snapshots[1].to_json()),
        ),
    )
    packed = BlackBoxPrepare(
        tuple(
            (name, width)
            for name, width in (
                ("prepare_control", controls),
                ("target_state", source.state_bitsize),
            )
            if width
        ),
        (("phase_gradient", bits),),
        emitter,
    )
    logical = controls + example.target_width + bits
    work, slots = emitter.requirements()

    def imported(adjoint: bool) -> CircuitIR:
        circuit = Circuit(logical + work).append(
            packed.make_on(
                selection=tuple(range(controls + example.target_width)),
                junk=tuple(range(controls + example.target_width, logical)),
                workspace=tuple(range(logical, logical + work)),
                classical_bits=tuple(range(slots)),
                adjoint=adjoint,
            )
        )
        ir = circuit.to_ir()
        assert isinstance(ir, CircuitIR)
        return ir

    forward, inverse = imported(False), imported(True)
    actual = Circuit(example.qubit_count).append(example).to_ir()
    captured_counts = Counter(g.name for g in forward.instructions)
    assert captured_counts == Counter(g.name for g in actual.instructions)
    if factory == "phase":
        # Source AddIntoPhaseGrad has no physical recipe. Both routes use the
        # accepted native add; compare its cost separately from the source model.
        events_expected = 14 + 8 * controls + 4 * uncompute
        assert captured_counts["measure"] == events_expected
        assert captured_counts["t"] + captured_counts["tdg"] == 4 * events_expected
        assert captured_counts["cx"] == 122 + 48 * controls + 36 * uncompute
        assert captured_counts["rx"] == 6
    channels = (forward, inverse, actual)
    events = tuple(
        certify_erasure_histories(
            ir.instructions,
            n_wires=ir.n_wires,
            zero_wires=tuple(range(logical, ir.n_wires)),
            # Prove all histories here and final cleanup on every logical
            # basis column below. Branch identity alone does not prove cleanup.
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
    for target in range(1 << (controls + example.target_width)):
        initial = torch.zeros(1 << logical, dtype=torch.complex128)
        initial[target << bits : (target + 1) << bits] = gradient
        prepared = (operator @ initial).reshape(
            1 << (controls + example.target_width), 1 << bits
        )
        state = prepared @ gradient.conj()
        torch.testing.assert_close(
            prepared, state[:, None] * gradient[None, :], atol=1e-11, rtol=0
        )
