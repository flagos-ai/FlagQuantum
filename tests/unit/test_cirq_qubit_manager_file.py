"""Verify Cirq ownership, release and the one-step decomposition call site."""

from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from flagquantum.ecosystem.cirq import _decompose
from flagquantum.ecosystem.cirq._qubit_manager import InteropQubitManager

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def cirq() -> Any:
    return pytest.importorskip("cirq")


@pytest.fixture(scope="module")
def source_manager() -> Any:
    import kickmix
    from qualtran.cirq_interop import _interop_qubit_manager

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    source = Path(_interop_qubit_manager.__file__)
    for path, digest in (
        (source, "da3e8d2ee92618a6c46c9c3e8ffbfc9d200dad07"),
        (
            source.with_name("_interop_qubit_manager_test.py"),
            "c0132d612e4a90972c5986b5cc77f47a67a0743e",
        ),
    ):
        raw = path.read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == digest
    return _interop_qubit_manager.InteropQubitManager


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("kind", ("greedy", "simple"))
def test_original_manager_lifetime(cirq: Any, source_manager: Any, kind: str) -> None:
    base = cirq.GreedyQubitManager if kind == "greedy" else cirq.SimpleQubitManager
    original = source_manager(base("anc"))
    native = InteropQubitManager(base("anc"))
    allocated = native.qalloc(1)
    reference = original.qalloc(1)
    assert allocated == reference
    native.qfree(allocated)
    original.qfree(reference)
    external = cirq.NamedQubit("junk")
    for manager in (original, native):
        with pytest.raises((ValueError, AssertionError)):
            manager.qfree([external])
        manager.manage_qubits([external])
        manager.qfree([external])
        with pytest.raises((ValueError, AssertionError)):
            manager.qfree([external])


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("allocate", ("qalloc", "qborrow"))
def test_reserved_candidate_does_not_reenter_allocation(
    cirq: Any, source_manager: Any, allocate: str
) -> None:
    native = InteropQubitManager(cirq.GreedyQubitManager("anc", maximize_reuse=True))
    source = source_manager(cirq.GreedyQubitManager("anc", maximize_reuse=True))
    reserved = cirq.NamedQubit("anc_0")
    for manager in (native, source):
        manager.manage_qubits([reserved])
    actual = getattr(native, allocate)(2)
    expected = getattr(source, allocate)(2)
    assert actual == expected
    assert reserved not in actual
    assert len(set(actual)) == 2
    assert native.qalloc(1) == source.qalloc(1)
    for manager in (native, source):
        manager.qfree([reserved])
        assert manager.qalloc(1) == [reserved]


def test_mixed_generator_release_preserves_every_internal_work_wire(cirq: Any) -> None:
    base = cirq.GreedyQubitManager("anc", maximize_reuse=True)
    manager = InteropQubitManager(base)
    reserved = cirq.NamedQubit("outside")
    manager.manage_qubits(iter([reserved]))
    clean = manager.qalloc(2)
    dirty = manager.qborrow(1)
    manager.qfree(qubit for qubit in [clean[0], reserved, clean[1], dirty[0]])
    assert set(base.qalloc(3)) == {*clean, *dirty}
    with pytest.raises((ValueError, AssertionError)):
        manager.qfree([reserved])


def test_failed_mixed_release_keeps_external_reservation(cirq: Any) -> None:
    manager = InteropQubitManager(cirq.GreedyQubitManager("anc"))
    external, unknown = cirq.NamedQubit.range(2, prefix="port")
    manager.manage_qubits([external])
    with pytest.raises((ValueError, AssertionError)):
        manager.qfree([external, unknown])
    manager.qfree([external])
    with pytest.raises((ValueError, AssertionError)):
        manager.qfree([external])


def test_default_manager_keeps_clean_dirty_and_dimension(cirq: Any) -> None:
    manager = InteropQubitManager()
    (clean,) = manager.qalloc(1, 3)
    (borrowed,) = manager.qborrow(1, 3)
    assert clean.dimension == borrowed.dimension == 3
    assert clean != borrowed
    assert isinstance(manager.qalloc(1)[0], type(clean))
    assert isinstance(manager.qborrow(1)[0], type(borrowed))
    manager.qfree([clean, borrowed])


def test_allocation_failure_releases_skipped_candidates(cirq: Any) -> None:
    base = cirq.GreedyQubitManager("anc", maximize_reuse=True)
    (external,) = base.qalloc(1)
    base.qfree([external])

    class FailingBase:
        def __init__(self) -> None:
            self.called = False

        def qalloc(self, n: int, dim: int) -> list[Any]:
            if self.called:
                raise ValueError("allocation failed")
            self.called = True
            return list(base.qalloc(n, dim))

        def qfree(self, qubits: list[Any]) -> None:
            base.qfree(qubits)

    manager = InteropQubitManager(FailingBase())
    manager.manage_qubits([external])
    with pytest.raises(ValueError, match="allocation failed"):
        manager.qalloc(1)
    assert base.qalloc(1) == [external]


def test_decomposition_reserves_input_even_when_base_name_collides(
    cirq: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RoundTripGate(cirq.testing.SingleQubitGate):  # type: ignore[misc]
        def _decompose_with_context_(
            self, qubits: tuple[Any, ...], context: Any
        ) -> Any:
            (data,) = qubits
            (clean,) = context.qubit_manager.qalloc(1)
            yield cirq.CNOT(data, clean)
            yield cirq.CNOT(data, clean)
            context.qubit_manager.qfree(iter([clean]))

    monkeypatch.setattr(_decompose, "uuid4", lambda: "fixture")
    q = cirq.NamedQubit("_fixture_0")
    leaves = _decompose.decompose_cirq_once(RoundTripGate().on(q))
    assert leaves is not None
    clean = cirq.NamedQubit("_fixture_1")
    assert leaves == [cirq.CNOT(q, clean), cirq.CNOT(q, clean)]
    state = np.array([1, 0, 1j, 0], dtype=np.complex128) / np.sqrt(2)
    result = cirq.Simulator(dtype=np.complex128).simulate(
        cirq.Circuit(leaves), qubit_order=[q, clean], initial_state=state.copy()
    )
    np.testing.assert_allclose(result.final_state_vector, state, atol=2e-12, rtol=0)
    with pytest.raises(ValueError, match="lifetime binding"):
        _decompose.import_cirq_decomposition(RoundTripGate().on(q))
