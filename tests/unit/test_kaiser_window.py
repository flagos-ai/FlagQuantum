"""Complete Kaiser source declaration, tensor and quantized-child evidence."""

from collections import Counter
from hashlib import sha1
from itertools import product
from math import sqrt
from pathlib import Path

import numpy as np
import pytest
import sympy
import torch

from flagquantum.algorithms.primitives.kaiser_window import KaiserWindowState
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.simulation.value_dimensions import SymbolicSequenceLength
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_rotation_preparation import source_rotation_preparation_gates
from tests.unit.test_symbolic_rotation_resource_model import assert_source_cost

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_complete_sources() -> None:
    from qualtran.bloqs.phase_estimation import kaiser_window_state

    root = Path(kaiser_window_state.__file__).parent
    for name, blob in (
        ("kaiser_window_state.py", "38e851f815ca3d6833215563738b481f39801a47"),
        ("kaiser_window_state_test.py", "6a4917cf85cd2f8f50c016a8b930d598680ee775"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("n,alpha", [(1, 0), (2, 2), (3, 0.5), (5, 2)])
def test_complete_coefficients_tensor_signature_and_source_callee(
    n: int, alpha: float
) -> None:
    from qualtran.bloqs.phase_estimation.kaiser_window_state import (
        KaiserWindowState as Source,
    )
    from qualtran.resource_counting import SympySymbolAllocator

    native, original = KaiserWindowState(n, alpha), Source(n, alpha)
    np.testing.assert_array_equal(
        native.kaiser_state_coeff, original.kaiser_state_coeff
    )
    np.testing.assert_array_equal(native.kaiser_state_coeff, original.tensor_contract())
    assert native.kaiser_state_coeff is native.kaiser_state_coeff
    assert [(r.name, r.bitsize, r.side.name) for r in native.signature] == [
        (r.name, r.bitsize, r.side.name) for r in original.signature
    ]
    if n < 2:
        with pytest.raises(AssertionError):
            original.build_call_graph(SympySymbolAllocator())
        return
    ((child, count),) = original.build_call_graph(SympySymbolAllocator()).items()
    assert count == 1 and child.phase_bitsize == n
    np.testing.assert_array_equal(
        native.source_preparation_coefficients, child.state_coefficients
    )
    graph = native.source_graph(root="Kaiser", preparation="RotationPreparation")
    assert graph.leaf_counts == {"RotationPreparation": 1}
    if n >= 2:
        assert native.rotation_preparation().phase_width == n
        assert_source_model(native.source_resource_model(), original)


@pytest.mark.parametrize(
    "precision,delta",
    [
        (1, 0.01),
        (4, 0.1),
        (8, 1e-5),
        (
            sympy.Symbol("p", integer=True, positive=True),
            sympy.Symbol("delta", positive=True),
        ),
    ],
)
def test_complete_factory_numeric_and_symbolic(
    precision: int | sympy.Expr, delta: float | sympy.Expr
) -> None:
    from qualtran.bloqs.phase_estimation.kaiser_window_state import (
        KaiserWindowState as Source,
    )

    native = KaiserWindowState.from_precision_and_delta(precision, delta)
    original = Source.from_precision_and_delta(precision, delta)
    assert native.bitsize == original.bitsize
    assert native.alpha == original.alpha


def test_complete_symbolic_declarations_and_shape_costs() -> None:
    from qualtran.bloqs.phase_estimation.kaiser_window_state import (
        KaiserWindowState as Source,
    )
    from qualtran.resource_counting import SympySymbolAllocator

    n, alpha = sympy.symbols("n alpha", positive=True)
    for width, shape in ((n, alpha), (3, alpha), (n, 2)):
        native, original = KaiserWindowState(width, shape), Source(width, shape)
        coefficients = native.source_preparation_coefficients
        assert isinstance(coefficients, SymbolicSequenceLength)
        ((child, count),) = original.build_call_graph(SympySymbolAllocator()).items()
        assert count == 1 and coefficients.length == child.state_coefficients.n
        assert child.phase_bitsize == width
        assert_source_cost(native.source_shape_resource_formula(), original)
        with pytest.raises(ValueError, match="concrete"):
            native.rotation_preparation()
    with pytest.raises(ValueError, match="symbolic"):
        KaiserWindowState(n, 2).kaiser_state_coeff


@pytest.mark.parametrize("uncompute", [False, True])
def test_quantized_child_exact_stream_all_branches_cleanup_and_resource(
    uncompute: bool,
) -> None:
    from qualtran.bloqs.phase_estimation.kaiser_window_state import (
        KaiserWindowState as Source,
    )
    from qualtran.resource_counting import (
        QECGatesCost,
        SympySymbolAllocator,
        get_cost_value,
    )

    native = KaiserWindowState(2, 2)
    ((child, _),) = Source(2, 2).build_call_graph(SympySymbolAllocator()).items()
    forward_child = child
    if uncompute:
        from attrs import evolve

        child = evolve(child, uncompute=True)
    work, slots = native.rotation_preparation().requirements(0, uncompute=uncompute)
    target, gradient = (0, 1), (2, 3)
    workspace = tuple(range(4, 4 + work))
    operation = native.make_on(
        qpe_reg=target,
        phase_gradient=gradient,
        workspace=workspace,
        classical_bits=tuple(range(slots)),
        uncompute=uncompute,
    )
    circuit = DynamicCircuit(4 + work, dtype=torch.complex128).append(operation)
    gates = circuit.to_ir().instructions
    expected_stream = source_rotation_preparation_gates(
        child,
        {"target_state": target, "phase_gradient": gradient},
        workspace,
        tuple(range(slots)),
    )
    assert_source(gates, expected_stream)
    assert CircuitIR.from_json(circuit.to_ir().to_json()).instructions == gates
    phase = np.exp(-2j * np.pi * np.arange(4) / 4) / 2
    target_state = np.array([1, 2j, -1j, 2], dtype=complex) / sqrt(10)
    logical = np.kron(target_state, phase)
    if uncompute:
        logical = forward_child.tensor_contract() @ logical
    mapped = child.tensor_contract() @ logical
    if uncompute:
        np.testing.assert_allclose(
            mapped, np.kron(target_state, phase), atol=1e-13, rtol=0
        )
    reduced = mapped.reshape(4, 4) @ phase.conj()
    np.testing.assert_allclose(mapped, np.kron(reduced, phase), atol=1e-13, rtol=0)
    clean = np.zeros(2**work, dtype=complex)
    clean[0] = 1
    initial = torch.tensor(np.kron(logical, clean))
    expected = torch.tensor(np.kron(mapped, clean))
    counts = Counter(g.name for g in gates)
    events = counts["measure"]
    assert events == 7 and counts["t"] + counts["tdg"] == 28
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / sqrt(2**events),
            atol=1e-12,
            rtol=0,
        )
    assert get_cost_value(child, QECGatesCost()).rotation == 0
    assert (work, slots) == (5, 2)


def test_original_missing_decomposition_and_precision_boundaries() -> None:
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.phase_estimation.kaiser_window_state import (
        KaiserWindowState as Source,
    )

    with pytest.raises(DecomposeNotImplementedError):
        Source(2, 2).decompose_bloq()
    with pytest.raises(ValueError, match="at least two phase bits"):
        KaiserWindowState(1, 2).rotation_preparation().requirements(0)
    with pytest.raises(ZeroDivisionError):
        KaiserWindowState.from_precision_and_delta(3, 0)


def test_numerical_owner_replacement_preserves_append_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from numpy.typing import NDArray
    from qualtran.bloqs.phase_estimation.kaiser_window_state import (
        KaiserWindowState as Source,
    )

    from flagquantum.core.value_math import SymbolicReal

    native = KaiserWindowState(2, 2)
    work, slots = native.rotation_preparation().requirements(0)
    workspace = tuple(range(4, 4 + work))
    bits = tuple(range(slots))
    original_gates = native.make_on(
        qpe_reg=(0, 1),
        phase_gradient=(2, 3),
        workspace=workspace,
        classical_bits=bits,
    ).instructions
    original_model = native.source_resource_model()

    def original_coefficients(width: int, alpha: SymbolicReal) -> NDArray[np.float64]:
        return np.asarray(Source(width, alpha).kaiser_state_coeff, dtype=np.float64)

    monkeypatch.setattr(
        "flagquantum.algorithms.primitives.kaiser_window.kaiser_window_coefficients",
        original_coefficients,
    )
    replaced = KaiserWindowState(2, 2)
    assert (
        replaced.make_on(
            qpe_reg=(0, 1),
            phase_gradient=(2, 3),
            workspace=workspace,
            classical_bits=bits,
        ).instructions
        == original_gates
    )
    assert replaced.source_resource_model() == original_model
