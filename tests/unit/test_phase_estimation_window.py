"""Complete rectangular window source and existing PREPARE append binding."""

import inspect
from collections import Counter
from hashlib import sha1
from pathlib import Path

import cirq
import numpy as np
import pytest
import sympy
import torch

from flagquantum.algorithms.primitives.phase_estimation_window import (
    PhaseEstimationWindow,
    RectangularWindowState,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.bits.bitwise import on_each_ir
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.registers import RegisterSide

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [1, 2, 4, 5])
def test_complete_preparation_source_capture_amplitudes_cleanup_and_resources(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.bloqs.phase_estimation.qpe_window_state import (
        RectangularWindowState as Source,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    original = Source(width)
    native = RectangularWindowState(width)
    assert isinstance(native, PhaseEstimationWindow)
    assert native.m_bits == original.m_bits
    assert native.m_qdtype.encoding == "fixed_point"
    assert (
        dict(native.m_qdtype.parameters)["fractional_width"]
        == original.m_qdtype.num_frac
    )
    assert native.m_register.side == RegisterSide.RIGHT
    assert [(r.name, r.bitsize, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.side.name) for r in original.signature
    ]
    target = tuple(reversed(range(width)))
    bound = native.bind()
    gates = bound.emit(target, (), (), ())
    source_circuit = original.decompose_bloq().to_cirq_circuit()
    ordered = tuple(sorted(source_circuit.all_qubits()))
    captured = tuple(
        Instruction("h", (target[ordered.index(op.qubits[0])],))
        for op in cirq.decompose(
            source_circuit, keep=lambda operation: operation.gate == cirq.H
        )
        if op.gate == cirq.H
    )
    assert len(captured) == width
    assert gates == captured == on_each_ir(Instruction("h", (0,)), target)
    circuit = Circuit(width + 1, dtype=torch.complex128)
    circuit.append(bound.make_on(selection=target, junk=()))
    state = circuit.state().detach().numpy().reshape(2**width, 2)
    np.testing.assert_allclose(
        state[:, 0], original.tensor_contract(), atol=1e-14, rtol=0
    )
    np.testing.assert_array_equal(state[:, 1], 0)
    assert Counter(g.name for g in gates) == {"h": width}
    spectator = np.array([np.sqrt(0.3), 1j * np.sqrt(0.7)])
    zero = np.zeros(2**width, dtype=complex)
    zero[0] = 1
    initial = np.kron(zero, spectator)
    coherent = Circuit(width + 1, inputs=torch.tensor(initial), dtype=torch.complex128)
    coherent.append(bound.make_on(selection=target, junk=()))
    np.testing.assert_allclose(
        coherent.state().detach().numpy().reshape(-1),
        np.kron(original.tensor_contract(), spectator),
        atol=1e-14,
        rtol=0,
    )
    coherent.append(bound.make_on(selection=target, junk=(), adjoint=True))
    np.testing.assert_allclose(
        coherent.state().detach().numpy().reshape(-1), initial, atol=1e-14, rtol=0
    )
    source_cost = get_cost_value(original, QECGatesCost())
    assert source_cost.clifford == width
    assert source_cost.total_t_count() == 0
    graph = native.source_graph()
    assert graph.leaf_counts == {"Allocate(phase)": 1, "H": width}
    assert graph.estimate_leaf_cost(lambda key: int(key == "H")) == width

    # Replace only the Compiler owner used by the same bound PREPARE consumer.
    def source_capture(
        gate: Instruction, wires: tuple[int, ...]
    ) -> tuple[Instruction, ...]:
        assert gate == Instruction("h", (0,)) and wires == target
        return captured

    monkeypatch.setattr(
        "flagquantum.algorithms.primitives.phase_estimation_window.on_each_ir",
        source_capture,
    )
    replaced = Circuit(width + 1, dtype=torch.complex128)
    replaced.append(bound.make_on(selection=target, junk=()))
    np.testing.assert_array_equal(
        replaced.state().detach().numpy(), circuit.state().detach().numpy()
    )


@pytest.mark.parametrize("precision,delta", [(0, 0.5), (3, 0.1), (8, 0.01)])
def test_precision_and_delta_factory_matches_complete_formula(
    precision: int, delta: float
) -> None:
    from qualtran.bloqs.phase_estimation.qpe_window_state import (
        RectangularWindowState as Source,
    )

    assert RectangularWindowState.from_precision_and_delta(
        precision, delta
    ).bitsize == (Source.from_precision_and_delta(precision, delta).bitsize)


@pytest.mark.parametrize(
    "eps", [0.01, 0.1, 1.0, sympy.Symbol("epsilon", positive=True)]
)
def test_standard_deviation_factory_numeric_and_symbolic(
    eps: float | sympy.Expr,
) -> None:
    from qualtran.bloqs.phase_estimation.qpe_window_state import (
        RectangularWindowState as Source,
    )

    native = RectangularWindowState.from_standard_deviation_eps(eps)
    original = Source.from_standard_deviation_eps(eps)
    assert native.bitsize == original.bitsize


def test_symbolic_base_and_original_examples() -> None:
    from qualtran.bloqs.phase_estimation.qpe_window_state import (
        RectangularWindowState as Source,
    )
    from qualtran.bloqs.phase_estimation.qpe_window_state import (
        _rectangular_window_state_small,
        _rectangular_window_state_symbolic,
    )

    n, delta = sympy.symbols("n delta", positive=True)
    native = RectangularWindowState.from_precision_and_delta(n, delta)
    original = Source.from_precision_and_delta(n, delta)
    assert native.bitsize == original.bitsize
    assert native.m_register.bitsize == original.m_register.bitsize
    assert native.source_graph().leaf_counts["H"] == original.bitsize
    with pytest.raises(ValueError, match="concrete width"):
        native.bind()
    assert (
        RectangularWindowState(5).bitsize == _rectangular_window_state_small().bitsize
    )
    assert (
        RectangularWindowState(sympy.Symbol("n")).bitsize
        == _rectangular_window_state_symbolic().bitsize
    )
    assert inspect.isabstract(PhaseEstimationWindow)


@pytest.mark.parametrize(
    "factory,args",
    [("from_precision_and_delta", (2, 0)), ("from_standard_deviation_eps", (0,))],
)
def test_original_division_failure(factory: str, args: tuple[int, ...]) -> None:
    from qualtran.bloqs.phase_estimation.qpe_window_state import (
        RectangularWindowState as Source,
    )

    for owner in (Source, RectangularWindowState):
        with pytest.raises(ZeroDivisionError):
            getattr(owner, factory)(*args)


def test_zero_width_declared_graph_and_physical_source_failure() -> None:
    from qualtran.bloqs.phase_estimation.qpe_window_state import (
        RectangularWindowState as Source,
    )

    native = RectangularWindowState(0)
    assert native.m_register.bitsize == Source(0).m_register.bitsize == 0
    assert native.source_graph().leaf_counts == {"Allocate(phase)": 1, "H": 0}
    with pytest.raises(ValueError):
        Source(0).tensor_contract()
    with pytest.raises(ValueError, match="zero-width"):
        native.bind()


def test_frozen_complete_source_and_tests() -> None:
    from qualtran.bloqs.phase_estimation import qpe_window_state

    root = Path(qpe_window_state.__file__).parent
    for name, expected in (
        ("qpe_window_state.py", "075147c04eef8ff5e47067d992c9af4c47372099"),
        ("qpe_window_state_test.py", "0ca33cf4ab254b52dac9a605e0efe1a1bda98d84"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected
