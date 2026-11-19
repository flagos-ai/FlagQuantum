"""Compare feedback imports with independent source instruments, branch by branch."""

import subprocess
import sys
from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback
from flagquantum.ecosystem.cirq.models import CirqConversionError
from tests.unit.measurement_checks import forced_branch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.mcmt import and_bloq

    raw = Path(and_bloq.__file__).read_bytes()
    assert (
        sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
        == "2e6efeb53938ac43f6f0330d79140f04138861eb"
    )


def test_optional_feedback_import_module_does_not_load_cirq() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from flagquantum.ecosystem.cirq._feedback_import import import_cirq_feedback; assert 'cirq' not in sys.modules",
        ],
        check=True,
        timeout=30,
    )


def source_branch(
    circuit: Any, order: tuple[Any, ...], initial: np.ndarray, outcomes: tuple[int, ...]
) -> np.ndarray:
    """Use Cirq's own predicate resolution and gate matrices, not native IR."""
    import cirq

    state = initial.copy()
    histories = cirq.ClassicalDataDictionaryStore()
    results = iter(outcomes)
    indices = np.arange(state.size)
    for original in circuit.all_operations():
        if not all(
            condition.resolve(histories) for condition in original.classical_controls
        ):
            continue
        operation = original.without_classical_controls()
        gate = operation.gate
        if isinstance(gate, cirq.MeasurementGate):
            raw: list[int] = []
            for qubit in operation.qubits:
                result = next(results)
                raw.append(result)
                mask = (indices >> (len(order) - 1 - order.index(qubit))) & 1
                state *= mask == result
            recorded = tuple(
                bit ^ (int(gate.invert_mask[i]) if i < len(gate.invert_mask) else 0)
                for i, bit in enumerate(raw)
            )
            histories.record_measurement(
                cirq.measurement_key_obj(operation), recorded, operation.qubits
            )
        elif isinstance(gate, cirq.ResetChannel):
            bit = 1 << (len(order) - 1 - order.index(operation.qubits[0]))
            zero, one = state[(indices & bit) == 0], state[(indices & bit) != 0]
            populated = (np.any(abs(zero) > 1e-14), np.any(abs(one) > 1e-14))
            assert not all(populated), "Source reset fixture must be projected"
            if np.any(abs(one) > 1e-14):
                state = state[indices ^ bit]
        else:
            state = cirq.Circuit(operation).unitary(qubit_order=order) @ state
    assert next(results, None) is None
    return np.asarray(state)


def compare_every_branch(circuit: Any, order: tuple[Any, ...]) -> None:
    import cirq

    result = import_cirq_feedback(circuit, qubit_order=order)
    assert result.report.lossless
    count = sum(
        len(op.qubits) for op in circuit.all_operations() if cirq.is_measurement(op)
    )
    values = np.arange(1 << len(order))
    initial = (values + 1 + 1j * (values % 5 - 2)).astype(np.complex128)
    initial /= np.linalg.norm(initial)
    for outcomes in product((0, 1), repeat=count):
        actual = forced_branch(
            torch.from_numpy(initial), result.ir.instructions, outcomes
        )
        expected = torch.from_numpy(source_branch(circuit, order, initial, outcomes))
        torch.testing.assert_close(actual, expected, atol=4e-12, rtol=0)


def test_original_and_erasure_including_phase_feedback_and_reset() -> None:
    import cirq
    from qualtran.bloqs.mcmt import And

    source = And().adjoint().to_clifford_t_circuit().unfreeze()
    order = tuple(cirq.NamedQubit(name) for name in ("ctrl_0", "ctrl_1", "target"))
    compare_every_branch(source, order)
    imported = import_cirq_feedback(source, qubit_order=order)
    assert [g.name for g in imported.ir.instructions] == ["h", "measure", "cz", "reset"]
    assert imported.ir.instructions[2].metadata["conditions"] == ((0, 1),)


def test_repeated_measurement_key_old_record_inversion_and_conditional_reset() -> None:
    import cirq

    a, b, c = cirq.NamedQubit.range(3, prefix="q")
    key = cirq.MeasurementKey("result")
    source = cirq.Circuit(
        cirq.measure(a, key=key, invert_mask=(True,)),
        cirq.reset(a),
        cirq.H(a),
        cirq.measure(a, key=key),
        cirq.X(b).with_classical_controls(cirq.KeyCondition(key, index=0)),
        cirq.Y(c).with_classical_controls(key),
        cirq.reset(a).with_classical_controls(cirq.KeyCondition(key, index=0)),
        cirq.global_phase_operation(1j),
    )
    compare_every_branch(source, (c, a, b))
    records = import_cirq_feedback(source, qubit_order=(c, a, b)).ir.metadata[
        "interop"
    ]["measurement_records"]
    assert records == {"result": ((0,), (1,))}


def test_explicit_order_rejects_aliases_missing_wires_and_nonbinary_qids() -> None:
    import cirq

    a, b = cirq.NamedQubit.range(2, prefix="q")
    for order in ((a, a), (b,), (cirq.LineQid(0, dimension=3),)):
        with pytest.raises(ValueError):
            import_cirq_feedback(cirq.Circuit(cirq.X(a)), qubit_order=order)


@pytest.mark.parametrize(
    "kind",
    (
        "nonzero",
        "equal",
        "unequal",
        "words",
        "boolean",
        "indexed",
        "implies",
        "equivalent",
        "true",
        "false",
        "too_large",
        "negative",
        "indexed_words",
        "reversed",
    ),
)
def test_multibit_keys_and_composed_boolean_guards(kind: str) -> None:
    import cirq
    import sympy as s

    a, b, c, d = cirq.LineQubit.range(4)
    x, y = s.symbols("x y")
    expressions = {
        "equal": s.Eq(x, 2),
        "unequal": s.Ne(x, 1),
        "words": s.Eq(x, y),
        "boolean": s.ITE(s.Eq(x, 3), s.Not(s.Eq(y, 0)), s.Xor(s.Eq(x, 0), s.Eq(y, 1))),
        "indexed": s.Eq(s.IndexedBase("x")[1], 1),
        "implies": s.Implies(s.Eq(x, 2), s.Ne(y, 0)),
        "equivalent": s.Equivalent(s.Eq(x, 0), s.Eq(y, 1)),
        "true": s.true,
        "false": s.false,
        "too_large": s.Eq(x, 9),
        "negative": s.Eq(x, -1),
        "indexed_words": s.Eq(s.IndexedBase("x")[-1], s.IndexedBase("y")[0]),
        "reversed": s.Eq(1, s.IndexedBase("x")[0], evaluate=False),
    }
    condition = (
        cirq.KeyCondition(cirq.MeasurementKey("x"))
        if kind == "nonzero"
        else cirq.SympyCondition(expressions[kind])
    )
    source = cirq.Circuit(
        cirq.measure(a, b, key="x"),
        cirq.measure(c, key="y"),
        cirq.Z(d).with_classical_controls(condition),
        cirq.global_phase_operation(np.exp(0.31j)).with_classical_controls(condition),
    )
    compare_every_branch(source, (d, b, c, a))


@pytest.mark.parametrize(
    "bad", ("undefined", "arithmetic", "gate", "tag", "empty_phase", "assumptions")
)
def test_unsupported_or_undefined_semantics_fail_with_report(bad: str) -> None:
    import cirq
    import sympy as s

    a = cirq.LineQubit(0)
    if bad == "undefined":
        source = cirq.Circuit(cirq.X(a).with_classical_controls("missing"))
    elif bad == "arithmetic":
        source = cirq.Circuit(
            cirq.measure(a, key="x"),
            cirq.X(a).with_classical_controls(
                cirq.SympyCondition(s.Gt(s.Symbol("x"), 0))
            ),
        )
    elif bad == "gate":

        class UnsupportedGate(cirq.Gate):
            def _num_qubits_(self) -> int:
                return 1

        source = cirq.Circuit(UnsupportedGate().on(a))
    elif bad == "tag":
        source = cirq.Circuit(cirq.X(a).with_tags("uninterpreted"))
    elif bad == "assumptions":
        source = cirq.Circuit(
            cirq.measure(a, key="x"),
            cirq.X(a).with_classical_controls(
                cirq.SympyCondition(s.Symbol("x", integer=True))
            ),
        )
    else:
        source = cirq.Circuit(cirq.global_phase_operation(1j))
    with pytest.raises(CirqConversionError) as error:
        import_cirq_feedback(source, qubit_order=() if bad == "empty_phase" else (a,))
    assert not error.value.report.lossless


def test_static_entry_still_rejects_midcircuit_measurement() -> None:
    import cirq

    from flagquantum.ecosystem.cirq import import_cirq

    a = cirq.LineQubit(0)
    source = cirq.Circuit(cirq.measure(a, key="key"), cirq.X(a))
    with pytest.raises(CirqConversionError):
        import_cirq(source)
    assert import_cirq_feedback(source, qubit_order=(a,)).report.lossless


def test_decomposed_foreign_sqrt_entry_matches_native_callback_stream() -> None:
    import cirq

    from examples.algorithms.custom_sparse_oracles import ConjunctionSqrtEntry
    from flagquantum.ecosystem.cirq.conversion import _lower_gate
    from tests.unit.register_source_graph import assert_source_wire_streams
    from tests.unit.test_hermitian_custom_oracles import source_oracles

    source = source_oracles(1).entry_oracle
    ports = tuple(cirq.LineQubit.range(3))
    circuit, outputs = source.decompose_bloq().to_cirq_circuit_and_quregs(
        qubit_manager=cirq.GreedyQubitManager(prefix="scratch", maximize_reuse=True),
        q=np.asarray(ports[:1]),
        i=np.asarray(ports[1:2]),
        j=np.asarray(ports[2:]),
    )
    assert {name: tuple(word.flat) for name, word in outputs.items()} == {
        "q": ports[:1],
        "i": ports[1:2],
        "j": ports[2:],
    }

    def admitted(operation: Any) -> bool:
        if (
            isinstance(operation, cirq.ClassicallyControlledOperation)
            or cirq.is_measurement(operation)
            or isinstance(operation.gate, (cirq.ResetChannel, cirq.GlobalPhaseGate))
        ):
            return True
        return _lower_gate(cirq, operation.gate, [], 0) is not None

    primitives = cirq.Circuit(cirq.decompose(circuit, keep=admitted))
    scratch = tuple(sorted(set(primitives.all_qubits()) - set(ports)))
    assert len(scratch) == 1
    imported = import_cirq_feedback(primitives, qubit_order=(*ports, *scratch)).ir
    native = ConjunctionSqrtEntry(1).emit(0, (1,), (2,), (3,), (0,))
    assert_source_wire_streams(imported.instructions, native)
    compare_every_branch(primitives, (*ports, *scratch))


@pytest.mark.parametrize("axis", ("rx", "ry", "rz"))
@pytest.mark.parametrize("angle", (0.31, -0.52))
def test_quantum_controlled_rotations_preserve_full_complex_phase(
    axis: str, angle: float
) -> None:
    import cirq

    a, b, c = cirq.NamedQubit.range(3, prefix="q")
    gate = getattr(cirq, axis)(angle).controlled().on(a, b)
    source = cirq.Circuit(cirq.measure(c, key="m"), gate.with_classical_controls("m"))
    compare_every_branch(source, (b, c, a))
    result = import_cirq_feedback(source, qubit_order=(b, c, a))
    assert result.ir.instructions[-1].name == "c" + axis


def test_scoped_qubit_reservation_restores_after_nested_exception() -> None:
    import cirq

    from flagquantum.ecosystem.cirq._qubit_manager import InteropQubitManager

    base = cirq.GreedyQubitManager(prefix="q", maximize_reuse=True)
    (external,) = base.qalloc(1)
    base.qfree([external])
    manager = InteropQubitManager(base)
    with pytest.raises(RuntimeError, match="probe"), manager.reserve_qubits([external]):
        (first,) = manager.qalloc(1)
        assert first != external
        with manager.reserve_qubits([first]):
            (second,) = manager.qalloc(1)
            assert second not in (first, external)
            raise RuntimeError("probe")
    (restored,) = manager.qalloc(1)
    assert restored == external
    manager.qfree([first, second, restored])


@pytest.mark.parametrize("exponent", (0.125, -0.37, 1.25))
def test_fractional_phase_powers_and_controlled_h_preserve_absolute_phase(
    exponent: float,
) -> None:
    import cirq

    a, b, c = cirq.LineQubit.range(3)
    source = cirq.Circuit(
        cirq.H(c),
        cirq.measure(c, key="record"),
        cirq.ZPowGate(exponent=exponent)(a),
        cirq.CZPowGate(exponent=exponent)(a, b),
        cirq.H(b).controlled_by(a),
        cirq.CZPowGate(exponent=-exponent)(a, b).with_classical_controls("record"),
        cirq.ZPowGate(exponent=exponent)(b).controlled_by(a),
    )
    compare_every_branch(source, (a, b, c))


@pytest.mark.parametrize("controlled", (False, True))
def test_declared_dense_matrix_leaf_including_feedback_and_global_phase(
    controlled: bool,
) -> None:
    import cirq

    a, b, c = cirq.LineQubit.range(3)
    matrix = np.exp(0.29j) * cirq.testing.random_unitary(4, random_state=1234)
    operation = cirq.MatrixGate(matrix)(a, b)
    if controlled:
        operation = operation.controlled_by(c, control_values=(0,))
    source = cirq.Circuit(
        cirq.H(c),
        cirq.measure(c, key="branch"),
        operation.with_classical_controls("branch"),
    )
    compare_every_branch(source, (c, b, a))
    imported = import_cirq_feedback(source, qubit_order=(c, b, a))
    leaf = imported.ir.instructions[-1]
    assert leaf.name == "any" and isinstance(leaf.matrix, torch.Tensor)
    assert leaf.metadata["conditions"] == ((0, 1),)
    assert torch.equal(
        leaf.matrix, torch.as_tensor(cirq.unitary(operation), dtype=torch.complex128)
    )


def test_matrix_leaf_is_copied_and_invalid_matrix_cannot_be_imported() -> None:
    import cirq

    q = cirq.LineQubit(0)
    original = np.array([[0, 1j], [1j, 0]], dtype=np.complex128)
    gate = cirq.MatrixGate(original)
    imported = import_cirq_feedback(cirq.Circuit(gate(q)), qubit_order=(q,))
    imported_matrix = imported.ir.instructions[0].matrix
    assert isinstance(imported_matrix, torch.Tensor)
    before = imported_matrix.clone()
    original[:] = np.eye(2)
    assert torch.equal(imported_matrix, before)
    invalid = cirq.MatrixGate(
        np.array([[1, 1], [0, 1]], dtype=np.complex128), unitary_check=False
    )
    with pytest.raises(CirqConversionError, match="not unitary"):
        import_cirq_feedback(cirq.Circuit(invalid(q)), qubit_order=(q,))


@pytest.mark.parametrize("exponent", (0.5, -0.5, 2.5, -2.5))
@pytest.mark.parametrize("controlled", (False, True))
def test_source_half_x_powers_use_native_sqrt_x_with_absolute_phase(
    exponent: float, controlled: bool
) -> None:
    import cirq

    a, b, c = cirq.LineQubit.range(3)
    gate = cirq.XPowGate(exponent=exponent)
    op = gate(b).controlled_by(a) if controlled else gate(b)
    source = cirq.Circuit(
        cirq.H(c), cirq.measure(c, key="record"), op.with_classical_controls("record")
    )
    compare_every_branch(source, (a, b, c))
    imported = import_cirq_feedback(source, qubit_order=(a, b, c))
    expected = ("c" if controlled else "") + ("sx" if exponent % 2 == 0.5 else "sxdg")
    assert imported.ir.instructions[-1].name == expected
    assert imported.ir.instructions[-1].metadata["conditions"] == ((0, 1),)


def test_shifted_half_x_keeps_its_distinct_rotation_phase_convention() -> None:
    import cirq

    from flagquantum.simulation.unitary import get_unitary

    q = cirq.LineQubit(0)
    gate = cirq.XPowGate(exponent=0.5, global_shift=-0.5)
    imported = import_cirq_feedback(cirq.Circuit(gate(q)), qubit_order=(q,))
    assert imported.ir.instructions[0].name == "rx"
    torch.testing.assert_close(
        get_unitary(imported.ir, dtype=torch.complex128),
        torch.as_tensor(cirq.unitary(gate), dtype=torch.complex128),
        atol=1e-14,
        rtol=0,
    )
