"""Complete GQSP metadata and source declarations reuse accepted execution owners."""

from collections.abc import Mapping
from hashlib import sha1
from pathlib import Path

import numpy as np
import pytest
import sympy
import torch

from flagquantum.algorithms.primitives.generalized_qsp import (
    GeneralizedQSP,
    source_gqsp_signature,
)
from flagquantum.algorithms.primitives.generalized_qsp_model import (
    source_generalized_qsp_graph,
)
from flagquantum.circuit import Circuit
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.value_dimensions import SymbolicArrayShape
from tests.unit.test_generalized_qsp_factory import _unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "length", [2, 5, sympy.Symbol("n", positive=True, integer=True)]
)
@pytest.mark.parametrize(
    "power", [0, 3, sympy.Symbol("k", positive=True, integer=True)]
)
def test_shape_only_factory_precision_and_physical_refusal(
    length: int | sympy.Expr, power: int | sympy.Expr
) -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source
    from qualtran.symbolics import Shaped

    def forbidden(
        signal: int,
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        value: int | None,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        raise AssertionError("Symbolic declarations must stop before unitary emission")

    shape = SymbolicArrayShape((length,))
    source = Source.from_qsp_polynomial(
        XGate(), Shaped((length,)), negative_power=power, precision=0.125, verify=True
    )
    native = GeneralizedQSP.from_qsp_polynomial(
        (("q", 1),),
        forbidden,
        shape,
        negative_power=power,
        precision=0.125,
        verify=True,
    )
    assert native.p is native.q is shape
    assert native.precision == source.precision == 1e-11
    assert native.precision_per_rotation == source._eps_per_rotation
    assert native.is_symbolic() == source.is_symbolic() is True
    with pytest.raises(ValueError, match="shape-only"):
        _ = native._phases
    with pytest.raises(ValueError):
        _ = source._qsp_phases
    circuit = Circuit(3, dtype=torch.complex128).h(2)
    before = circuit.to_ir()
    with pytest.raises(ValueError, match="symbolic"):
        circuit.append(native.make_on(signal=0, target=(1,)))
    assert circuit.to_ir() == before


@pytest.mark.parametrize("length", [0, 1])
def test_concrete_shape_degree_validation(length: int) -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source
    from qualtran.symbolics import Shaped

    for which in (0, 1):
        native = [(0.2, 0.3), (0.2, 0.3)]
        source = [(0.2, 0.3), (0.2, 0.3)]
        with pytest.raises(ValueError, match="degree at least 1"):
            Source(
                XGate(),
                *(Shaped((length,)) if i == which else source[i] for i in (0, 1)),
            )
        with pytest.raises(ValueError, match="degree at least 1"):
            GeneralizedQSP(
                (("q", 1),),
                _unitary,
                *(
                    SymbolicArrayShape((length,)) if i == which else native[i]
                    for i in (0, 1)
                ),
            )


@pytest.mark.parametrize("degree", [2, 4, 6])
@pytest.mark.parametrize("power", [0, 2, 3, 5])
def test_original_shape_call_graph_and_caller_replacement(
    degree: int, power: int
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source
    from qualtran.resource_counting import SympySymbolAllocator
    from qualtran.symbolics import Shaped

    source = Source.from_qsp_polynomial(
        XGate(), Shaped((degree + 1,)), negative_power=power
    )
    expected = {}
    for key, repetitions in source.build_call_graph(SympySymbolAllocator()).items():
        if isinstance(key, SU2RotationGate):
            name = "rotation"
        elif key == XGate().controlled(ctrl_spec=CtrlSpec(cvs=0)):
            name = "zero-U"
        elif key == XGate().adjoint():
            name = "adjoint"
        else:
            assert key == XGate().adjoint().controlled()
            name = "controlled-adjoint"
        expected[name] = repetitions
    actual = source_generalized_qsp_graph(
        degree + 1,
        negative_power=power,
        root="GQSP",
        signal_rotation="rotation",
        zero_control_unitary="zero-U",
        unitary_adjoint="adjoint",
        controlled_adjoint="controlled-adjoint",
    )
    assert actual.leaf_counts == expected
    assert actual.estimate_leaf_cost(lambda _: 1) == sum(expected.values())
    # Replacing all unitary-role keys proves equality-based source Counter merging.
    replaced = source_generalized_qsp_graph(
        degree + 1,
        negative_power=power,
        root="GQSP",
        signal_rotation="rotation",
        zero_control_unitary="same-U",
        unitary_adjoint="same-U",
        controlled_adjoint="same-U",
    )
    assert replaced.leaf_counts == {
        "rotation": degree + 1,
        "same-U": sum(value for key, value in expected.items() if key != "rotation"),
    }


def test_symbolic_expression_graph_and_typed_signature() -> None:
    from qualtran import QBit, QUInt
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.basic_gates.su2_rotation import SU2RotationGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source
    from qualtran.resource_counting import SympySymbolAllocator
    from qualtran.symbolics import Shaped

    n, k = sympy.symbols("n k", positive=True, integer=True)
    graph = source_generalized_qsp_graph(
        n + 1,
        negative_power=k,
        root="GQSP",
        signal_rotation="rotation",
        zero_control_unitary="zero-U",
        unitary_adjoint="adjoint",
        controlled_adjoint="controlled-adjoint",
    )
    source = Source.from_qsp_polynomial(XGate(), Shaped((n + 1,)), negative_power=k)
    values = source.build_call_graph(SympySymbolAllocator())
    rotation_count = next(
        v for key, v in values.items() if isinstance(key, SU2RotationGate)
    )
    assert graph.leaf_counts["rotation"] == rotation_count == n + 1
    assert graph.leaf_counts["zero-U"] == sympy.Max(0, n - k)
    assert graph.leaf_counts["adjoint"] == sympy.Max(0, k - n)
    assert graph.leaf_counts["controlled-adjoint"] == sympy.Min(n, k)
    signature = source_gqsp_signature(
        Signature((Register("word", unsigned_register_type(n), (2,)),))
    )
    expected = (
        SourceRegister("signal", QBit()),
        SourceRegister("word", QUInt(n), (2,)),
    )
    assert [(r.name, r.bitsize, r.shape, r.side.name) for r in signature] == [
        (r.name, r.bitsize, r.shape, r.side.name) for r in expected
    ]
    finite = GeneralizedQSP((("q", 1),), _unitary, (0.2, 0.3), (0.4, 0.1))
    assert finite.signature[0] == Register("signal", quantum_bit_type())
    assert [r.name for r in finite.signature] == ["signal", "q"]


@pytest.mark.parametrize(
    "p,power", [((0.5, 0.5), 0), ((0.5, 0, 0.5), 1), ((0.5, 0, 0.5), 5)]
)
def test_all_original_factories_through_existing_append(
    p: tuple[complex, ...], power: int
) -> None:
    import cirq

    from flagquantum.compiler.registers.primitives.pauli_rotation import pauli_power_ir

    def unitary_x(
        signal: int,
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        value: int | None,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        flip = (Instruction("x", (signal,)),) if value == 0 else ()
        return (
            *flip,
            *pauli_power_ir(
                "x",
                ports["q"][0],
                -1 if adjoint else 1,
                control=None if value is None else signal,
            ),
            *flip,
        )

    from qualtran.bloqs.qsp.generalized_qsp import (
        _gqsp,
        _gqsp_with_large_negative_power,
        _gqsp_with_negative_power,
    )

    original_factory = {
        0: _gqsp,
        1: _gqsp_with_negative_power,
        5: _gqsp_with_large_negative_power,
    }[power]
    source = original_factory()
    native = GeneralizedQSP.from_qsp_polynomial(
        (("q", 1),), unitary_x, p, negative_power=power
    )
    assert tuple(p) == source.P
    np.testing.assert_array_equal(native.q, source.Q)
    assert native.precision == source.precision == 0
    expected = cirq.unitary(source)
    coherent = np.array([1, 2j, -1j, 2], dtype=complex) / np.sqrt(10)
    for initial in (*np.eye(4, dtype=complex), coherent):
        circuit = Circuit(
            3, inputs=torch.tensor(np.kron(initial, [1, 0])), dtype=torch.complex128
        )
        circuit.append(native.make_on(signal=0, target=(1,)))
        state = circuit.state().detach().numpy().reshape(4, 2)
        np.testing.assert_allclose(state[:, 0], expected @ initial, atol=1e-13, rtol=0)
        np.testing.assert_array_equal(state[:, 1], 0)
    assert not native.is_symbolic()


def test_frozen_complete_source_and_test_blobs() -> None:
    from qualtran.bloqs.qsp import generalized_qsp

    root = Path(generalized_qsp.__file__).parent
    for name, expected in (
        ("__init__.py", "15780de77b1637deab59d60722c84f30b3d8e75d"),
        ("generalized_qsp.py", "5848dcb4785697563fca6173e499015e504b233b"),
        ("generalized_qsp_test.py", "c925da2a851888c4ceb3c8dbc1893c93e3c7602d"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


@pytest.mark.parametrize("shape_left", [False, True])
def test_mixed_shape_direct_constructor_preserves_explicit_precision(
    shape_left: bool,
) -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source
    from qualtran.symbolics import Shaped

    p, q = ((0.2, 0.3), (0.4, 0.1))
    native = GeneralizedQSP(
        (("q", 1),),
        _unitary,
        SymbolicArrayShape((2,)) if shape_left else p,
        q if shape_left else SymbolicArrayShape((2,)),
        precision=0.125,
    )
    source = Source(
        XGate(),
        Shaped((2,)) if shape_left else p,
        q if shape_left else Shaped((2,)),
        precision=0.125,
    )
    assert native.precision == source.precision == 0.125
    assert native.precision_per_rotation == source._eps_per_rotation
    assert native.is_symbolic() == source.is_symbolic() is True
    with pytest.raises(ValueError, match="shape-only"):
        _ = native.signal_rotation_calls


def test_finite_phases_symbolic_power_and_precision_keep_distinct_roles() -> None:
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as Source

    k, eps = sympy.symbols("k eps", positive=True)
    p, q = (0.2, 0.3), (0.4, 0.1)
    native = GeneralizedQSP((("q", 1),), _unitary, p, q, k, precision=eps)
    source = Source(XGate(), p, q, negative_power=k, precision=eps)
    assert native.is_symbolic() == source.is_symbolic() is True
    for actual, expected in zip(native._phases, source._qsp_phases, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert native.precision_per_rotation == source._eps_per_rotation
    with pytest.raises(ValueError, match="symbolic"):
        native.emit(0, (1,))
    concrete = GeneralizedQSP((("q", 1),), _unitary, p, q, precision=eps)
    assert not concrete.is_symbolic()
    assert len(concrete.emit(0, (1,))) > 0
