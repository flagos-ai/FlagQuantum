"""Trace the actual source H front end, including extracted control phases."""

from hashlib import sha256
from itertools import product
from pathlib import Path

import pytest
import torch

from flagquantum.compiler.registers.primitives.controlled_hadamard import (
    controlled_hadamard_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._component_snapshot import _snapshot
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_controlled_unitary_synthesis import (
    assert_source_stream,
)
from tests.unit.test_controlled_unitary_synthesis import (
    reference_versions as reference_versions,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PATTERNS = tuple(
    tuple(values) for count in range(5) for values in product((0, 1), repeat=count)
)


@pytest.fixture(scope="module", autouse=True)
def frozen_hadamard_frontend() -> None:
    from cirq.ops import common_gates

    folder = Path(common_gates.__file__).parent
    for name, digest in (
        (
            "common_gates.py",
            "dd01adbca42e57cfc606016300f7995d42c240f3299d915eed21bc99cf88df69",
        ),
        (
            "eigen_gate.py",
            "e038f627047ba7ca065b7b48f602dff6b21faf58f82d1c84f259acdd654c55ad",
        ),
        (
            "global_phase_op.py",
            "78685dc7cc5b2688274964191febf8ed334949e8a51e71bfbc1d987d8dff3f44",
        ),
        (
            "controlled_gate.py",
            "665e05967a5a024101b38507586b850165bb93d7a14ff12d3a91449838a15908",
        ),
        (
            "diagonal_gate.py",
            "ea4392b4837a4aedbd5adf67c1f2b93897432e18675f9b5e3d0be9506249ac19",
        ),
        (
            "three_qubit_gates.py",
            "ef56b158711d27d9bab98173d73e4bcf7c85c46ddf581a4ec691872e6fdeea86",
        ),
    ):
        assert sha256((folder / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("values", PATTERNS)
def test_hadamard_front_end_matches_actual_source_calls(
    values: tuple[int, ...],
    adjoint: bool,
) -> None:
    import cirq
    from qualtran.cirq_interop import CirqGateAsBloq

    source_gate = (
        cirq.H.controlled(num_controls=len(values), control_values=values)
        if values
        else cirq.H
    )
    if adjoint:
        source_gate = source_gate**-1
    source = CirqGateAsBloq(source_gate)
    ports = tuple((reg.name, int(reg.total_bits())) for reg in source.signature)
    original, swaps = _snapshot(source, ports)
    assert swaps == 0 and original.n_wires == len(values) + 1
    actual = CircuitIR(
        len(values) + 1,
        controlled_hadamard_ir(
            tuple(range(len(values))),
            len(values),
            control_values=values,
            adjoint=adjoint,
        ),
    )
    assert_source_stream(actual, original)
    expected = torch.as_tensor(cirq.unitary(source_gate), dtype=torch.complex128)
    for restored in (actual, CircuitIR.from_json(actual.to_json())):
        torch.testing.assert_close(
            get_unitary(restored, dtype=torch.complex128), expected, atol=2e-11, rtol=0
        )
    assert all(g.name not in {"measure", "reset"} for g in actual.instructions)


@pytest.mark.parametrize("values", [(), (2,), (True,), (0, 1)])
def test_invalid_polarity_is_rejected(values: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="binary value"):
        controlled_hadamard_ir((0,), 1, control_values=values)


@pytest.mark.parametrize("adjoint", (False, True))
@pytest.mark.parametrize("controls,target", [((0, 2), 3), ((0, 1), 3)])
def test_source_ccz_adjacency_on_noncontiguous_line_wires(
    controls: tuple[int, ...], target: int, adjoint: bool
) -> None:
    import cirq

    from flagquantum.ecosystem.cirq._component_snapshot import _has_explicit_matrix
    from flagquantum.ecosystem.cirq._feedback_import import (
        _lower_feedback_gate,
        import_cirq_feedback,
    )

    wires = cirq.LineQubit.range(4)
    source = cirq.H(wires[target]).controlled_by(*(wires[q] for q in controls))
    if adjoint:
        source = source**-1
    leaves = cirq.decompose(
        source,
        keep=lambda op: _has_explicit_matrix(cirq, op.gate)
        or _lower_feedback_gate(cirq, op.gate, [], 0) is not None,
    )
    original = import_cirq_feedback(
        cirq.Circuit(leaves, strategy=cirq.InsertStrategy.NEW), qubit_order=wires
    ).ir
    actual = CircuitIR(4, controlled_hadamard_ir(controls, target, adjoint=adjoint))
    assert_source_stream(actual, original)
    torch.testing.assert_close(
        get_unitary(actual, dtype=torch.complex128),
        get_unitary(original, dtype=torch.complex128),
        atol=2e-11,
        rtol=0,
    )


def test_invalid_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="direction must be Boolean"):
        controlled_hadamard_ir((0,), 1, adjoint=1)
