"""Compare ordered lookup against the unchanged random-oracle source path."""

from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.lookup.ordered_pauli_select import (
    ordered_pauli_select_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._component_snapshot import _snapshot
from flagquantum.simulation.unitary import get_unitary
from tests.unit.test_controlled_unitary_synthesis import assert_source_stream

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
np = pytest.importorskip("numpy")


@pytest.mark.parametrize("width,control", tuple(product((1, 2, 3), (None, 0, 1))))
@pytest.mark.parametrize("adjoint", (False, True))
def test_ordered_pauli_source_stream_and_complete_complex_operator(
    width: int, control: int | None, adjoint: bool
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        TestPauliSelectOracle,
    )

    words = tuple("IXYZ"[i % 4] + "ZYXI"[i % 4] for i in range(1 << width))
    source = TestPauliSelectOracle(
        width, 2, tuple(cirq.DensePauliString(word) for word in words), control
    )
    offset = int(control is not None)
    ports = tuple((r.name, int(r.total_bits())) for r in source.signature)
    original, swaps = _snapshot(source.adjoint() if adjoint else source, ports)
    assert swaps == 0
    actual = CircuitIR(
        width + 2 + offset,
        ordered_pauli_select_ir(
            words,
            tuple(range(offset, width + offset)),
            (width + offset, width + offset + 1),
            control=(0,) if offset else (),
            control_value=1 if control is None else control,
            adjoint=adjoint,
        ),
    )
    assert_source_stream(actual, original)
    expected = torch.as_tensor(source.tensor_contract().copy(), dtype=torch.complex128)
    if adjoint:
        expected = expected.mH
    for restored in (actual, CircuitIR.from_json(actual.to_json())):
        torch.testing.assert_close(
            get_unitary(restored, dtype=torch.complex128), expected, atol=3e-11, rtol=0
        )
    assert all(g.name not in {"measure", "reset"} for g in actual.instructions)


@pytest.mark.parametrize("control", (None, 0, 1))
@pytest.mark.parametrize("adjoint", (False, True))
def test_ordered_pauli_signed_rows(control: int | None, adjoint: bool) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        TestPauliSelectOracle,
    )

    words = ("II", "XY", "YZ", "ZX")
    coefficients = (1 + 0j, -1 + 0j, 1j, np.exp(0.37j))
    source = TestPauliSelectOracle(
        2,
        2,
        tuple(
            cirq.DensePauliString(w, coefficient=c)
            for w, c in zip(words, coefficients, strict=True)
        ),
        control,
    )
    offset = int(control is not None)
    original, _ = _snapshot(
        source.adjoint() if adjoint else source,
        tuple((r.name, int(r.total_bits())) for r in source.signature),
    )
    actual = CircuitIR(
        4 + offset,
        ordered_pauli_select_ir(
            words,
            (offset, offset + 1),
            (offset + 2, offset + 3),
            control=(0,) if offset else (),
            control_value=1 if control is None else control,
            coefficients=coefficients,
            adjoint=adjoint,
        ),
    )
    assert_source_stream(actual, original)
    expected = torch.as_tensor(source.tensor_contract().copy(), dtype=torch.complex128)
    torch.testing.assert_close(
        get_unitary(actual, dtype=torch.complex128),
        expected.mH if adjoint else expected,
        atol=3e-11,
        rtol=0,
    )


@pytest.mark.parametrize("width", (1, 2))
def test_random_selector_actual_append_owns_no_work_and_reuses_original_stream(
    width: int,
) -> None:
    from collections import Counter

    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        random_qubitization_walk_operator,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum import Circuit
    from tests.unit.pauli_walk_fixtures import RandomPauliFixture

    source, _ = random_qubitization_walk_operator(
        width, 2, random_state=np.random.RandomState(42)
    )
    native = RandomPauliFixture.random(width, 2, np.random.RandomState(42))
    for control in (None, 0, 1):
        from attrs import evolve

        offset = int(control is not None)
        selector = evolve(source.block_encoding.select, control_val=control)
        original, _ = _snapshot(
            selector, tuple((r.name, int(r.total_bits())) for r in selector.signature)
        )
        circuit = Circuit(width + 2 + offset, dtype=torch.complex128)
        circuit.append(
            native.selector(control).make_on(
                selection=tuple(range(width + offset)),
                target=(width + offset, width + offset + 1),
            )
        )
        assert_source_stream(circuit.to_ir(), original)
        assert Counter(g.name for g in circuit.to_ir().instructions) == Counter(
            g.name for g in original.instructions
        )
        assert circuit.to_ir().n_wires == width + 2 + offset
        assert not any(
            g.name in {"measure", "reset"} for g in circuit.to_ir().instructions
        )
    # These logical model values follow Qualtran's composite control graph,
    # independently of its Cirq SDK decomposition's primitive counts.
    cost = get_cost_value(source.block_encoding.select, QECGatesCost())
    assert cost.and_bloq == (0 if width == 1 else 4)
    assert cost.measurement == (0 if width == 1 else 4)
    assert get_cost_value(source.block_encoding.select, QubitCount()) == width + 2 + (
        width - 1
    )


@pytest.mark.parametrize("depth", (0, 1, 2))
def test_original_foreign_factories_retain_transport_failures(depth: int) -> None:
    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        TestPauliSelectOracle,
        TestPrepareOracle,
    )
    from qualtran.serialization.bloq import bloqs_from_proto, bloqs_to_proto

    from flagquantum.compiler.formats.legacy_component_library import (
        deserialize_component_library,
        serialize_component_library,
    )

    prepare = TestPrepareOracle.random(2, random_state=np.random.RandomState(42))
    proto = bloqs_to_proto(prepare, max_depth=depth)
    wire = proto.SerializeToString(deterministic=True)
    assert serialize_component_library(deserialize_component_library(wire)) == wire
    with pytest.raises(ValueError, match="Unable to find a Bloq"):
        bloqs_from_proto(proto)
    select = TestPauliSelectOracle.random(2, 2, random_state=np.random.RandomState(31))
    with pytest.raises(ValueError, match="Object arrays cannot be saved"):
        bloqs_to_proto(select, max_depth=depth)


@pytest.mark.parametrize("control", (0, 1))
def test_original_specialized_select_control_callback(control: int) -> None:
    from qualtran import BloqBuilder, CtrlSpec, QBit
    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        TestPauliSelectOracle,
    )

    source = TestPauliSelectOracle.random(2, 2, random_state=np.random.RandomState(31))
    controlled, callback = source.get_ctrl_system(CtrlSpec(cvs=control))
    assert controlled.control_val == control
    bb = BloqBuilder()
    ctrl = bb.add_register("ctrl", QBit())
    ports = {reg.name: bb.add_register(reg) for reg in source.signature}
    (returned,), outputs = callback(bb, (ctrl,), ports)
    result = dict(zip((r.name for r in source.signature), outputs, strict=True))
    record = bb.finalize(ctrl=returned, **result)
    calls = tuple(record.bloq_instances)
    assert len(calls) == 1 and calls[0].bloq == controlled
    for nested in (0, 1):
        outer, _ = controlled.get_ctrl_system(CtrlSpec(cvs=nested))
        assert outer != controlled


@pytest.mark.parametrize(
    "kwargs",
    (
        {"selection": ()},
        {"target": ()},
        {"control": (3, 4)},
        {"control_value": True},
        {"adjoint": 1},
        {"words": ("XI", "II", "II")},
        {"words": ("AI", "II")},
        {"coefficients": (1 + 0j,)},
        {"coefficients": (0j, 1 + 0j)},
        {"target": (0, 2)},
    ),
)
def test_invalid_ordered_lookup_bindings_fail(kwargs: dict[str, Any]) -> None:
    arguments = {"words": ("XI", "II"), "selection": (0,), "target": (1, 2)} | kwargs
    with pytest.raises(ValueError):
        ordered_pauli_select_ir(**arguments)


@pytest.mark.parametrize("rows", (0, 1, 3))
@pytest.mark.parametrize("adjoint", (False, True))
def test_omitted_rows_keep_original_identity_addresses(
    rows: int, adjoint: bool
) -> None:
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        TestPauliSelectOracle,
    )

    words = ("XI", "IZ", "YX")[:rows]
    source = TestPauliSelectOracle(2, 2, tuple(cirq.DensePauliString(w) for w in words))
    original, _ = _snapshot(
        source.adjoint() if adjoint else source, (("selection", 2), ("target", 2))
    )
    actual = CircuitIR(
        4, ordered_pauli_select_ir(words, (0, 1), (2, 3), adjoint=adjoint)
    )
    assert_source_stream(actual, original)
    torch.testing.assert_close(
        get_unitary(actual, dtype=torch.complex128),
        get_unitary(original, dtype=torch.complex128),
        atol=3e-11,
        rtol=0,
    )


@pytest.mark.parametrize("width", (1, 2, 3))
def test_prepare_composite_matrix_leaf_and_nonreal_rejection(width: int) -> None:
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.bloqs.for_testing.random_select_and_prepare import TestPrepareOracle

    from tests.unit.matrix_register_fixture import MatrixRegisterFixture
    from tests.unit.pauli_walk_fixtures import RandomPauliFixture

    source = TestPrepareOracle.random(width, random_state=np.random.RandomState(42))
    native = RandomPauliFixture.random(width, 2, np.random.RandomState(42))
    (call,) = source.decompose_bloq().bloq_instances
    assert call.bloq == source.U
    prep = native.walk().block_encoding.signal_state
    for adjoint in (False, True):
        (gate,) = prep.emit(tuple(range(width)), (), (), (), adjoint=adjoint)
        assert gate.name == "matrix_fixture" and gate.wires == tuple(range(width))
        expected = source.U.adjoint() if adjoint else source.U
        torch.testing.assert_close(
            torch.as_tensor(gate.matrix).reshape(1 << width, 1 << width),
            torch.as_tensor(np.array(expected.matrix)),
            atol=0,
            rtol=0,
        )
    matrix = np.array(source.U.matrix) * 1j
    with pytest.raises(AssertionError):
        TestPrepareOracle(MatrixGate(width, matrix))
    with pytest.raises(AssertionError):
        RandomPauliFixture(
            MatrixRegisterFixture(width, tuple(tuple(row) for row in matrix)),
            native.words,
            2,
        )
