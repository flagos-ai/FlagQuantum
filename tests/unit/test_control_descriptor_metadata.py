"""Control metadata reuses native registers, values and immutable patterns."""

from collections import Counter
from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest
import torch

try:
    import numpy as np
except ModuleNotFoundError:
    pytest.importorskip("numpy")

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.controlled_programs import controlled_x_program
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_unsigned_register_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_parameters import is_symbolic_value
from flagquantum.simulation.classical_control_activation import (
    classical_controls_active,
)
from flagquantum.simulation.classical_control_patterns import normalize_control_patterns
from flagquantum.simulation.tensor_network.controlled_tensor import (
    controlled_register_tensor,
)
from flagquantum.simulation.value_dimensions import SymbolicArrayShape
from tests.unit.test_register_control_tutorial import _source_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "kind,width,shape,values",
    [
        ("QBit", 1, (), 1),
        ("QBit", 1, (1,), [1]),
        ("QBit", 1, (3, 3, 3), np.ones((3, 3, 3), dtype=np.intc)),
        ("QBit", 1, (0, 2), np.zeros((0, 2), dtype=int)),
        ("QInt", 64, (), np.int64(234234)),
        ("QUInt", 3, (2,), [1, 5]),
        ("QUInt", 0, (), 0),
        ("CBit", 1, (), 0),
        ("CBit", 1, (2, 2), np.asarray([[1, 0], [0, 1]])),
        ("CUInt", 4, (2,), [3, 9]),
        ("CUInt", 0, (), 0),
    ],
)
def test_constructor_shapes_counts_and_actual_activation(
    kind: str, width: int, shape: tuple[int, ...], values: Any
) -> None:
    import qualtran as q

    source_type = (
        getattr(q, kind)() if kind.endswith("Bit") else getattr(q, kind)(width)
    )
    source = q.CtrlSpec(source_type, values)
    dtype = {
        "QBit": quantum_bit_type(),
        "QInt": signed_register_type(width),
        "QUInt": unsigned_register_type(width),
        "CBit": classical_bit_type(),
        "CUInt": classical_unsigned_register_type(width),
    }[kind]
    patterns = normalize_control_patterns(values)
    assert len(patterns) == source.num_ctrl_reg == 1
    ports = (Register("ctrl", dtype, patterns[0].shape),)
    signature = Signature(ports)
    assert tuple(r.shape_symbolic for r in ports) == source.shapes == (shape,)
    assert tuple(r.shape for r in ports) == source.concrete_shapes
    assert signature.n_bits() == source.num_bits
    assert signature.n_qubits() == source.num_qubits
    assert signature.n_cbits() == source.num_cbits
    assert [(r.dtype, r.shape_symbolic) for r in ports] == [(dtype, shape)]
    assert [(t.num_bits, sh) for t, sh in source.activation_function_dtypes()] == [
        (dtype.num_bits, shape)
    ]
    assert is_symbolic_value(*ports, *patterns) == source.is_symbolic()
    assert classical_controls_active(
        (values,), patterns, symbolic_parameters=ports
    ) == source.is_active(values)
    assert classical_controls_active((values,), patterns, symbolic_parameters=ports)


@pytest.mark.parametrize("unknown_shape", [False, True])
@pytest.mark.parametrize("classical", [False, True])
def test_symbolic_counts_shapes_and_activation_refusals(
    unknown_shape: bool, classical: bool
) -> None:
    from qualtran import CtrlSpec, CUInt, QUInt
    from qualtran.symbolics import Shaped
    from sympy import Symbol

    n = Symbol("n", integer=True, positive=True)
    shape = (n, 2) if unknown_shape else (2,)
    source_values = Shaped(shape) if unknown_shape else np.asarray([1, 0])
    native_values = SymbolicArrayShape(shape) if unknown_shape else source_values
    source = CtrlSpec(CUInt(n) if classical else QUInt(n), source_values)
    patterns = normalize_control_patterns(native_values)
    dtype = (
        classical_unsigned_register_type(n) if classical else unsigned_register_type(n)
    )
    ports = (Register("ctrl", dtype, shape),)
    signature = Signature(ports)
    assert source.is_symbolic() and is_symbolic_value(*ports, *patterns)
    assert source.shapes == tuple(r.shape_symbolic for r in ports)
    assert signature.n_bits() == source.num_bits
    assert signature.n_qubits() == source.num_qubits
    assert signature.n_cbits() == source.num_cbits
    for evaluate in (
        lambda: source.is_active(),
        lambda: classical_controls_active((), patterns, symbolic_parameters=ports),
    ):
        with pytest.raises(ValueError, match="symbolic"):
            evaluate()
    if unknown_shape:
        # Source checks shape tuples as opaque objects and leaks symbolic dimensions.
        assert source.concrete_shapes == (shape,)
        with pytest.raises(ValueError, match="symbolic"):
            _ = ports[0].shape
        assert patterns[0] is native_values
    else:
        assert source.concrete_shapes == (ports[0].shape,)


def test_multiple_groups_identity_shape_dtype_and_hash_equivalence() -> None:
    from qualtran import CBit, CtrlSpec, QBit, QInt, QUInt

    sources = [
        CtrlSpec(),
        CtrlSpec(QBit(), 1),
        CtrlSpec(QBit(), [1]),
        CtrlSpec(QUInt(1), 1),
        CtrlSpec(QInt(1), -1),
        CtrlSpec(CBit(), 1),
        CtrlSpec((QBit(), QUInt(2)), (np.array(1), np.array([1, 2]))),
        CtrlSpec((QBit(), QUInt(2)), (np.array(1), np.array([1, 2]))),
    ]
    native = [
        ((Register("ctrl", quantum_bit_type()),), (ControlRegister(1, (1,)),)),
        ((Register("ctrl", quantum_bit_type()),), (ControlRegister(1, (1,)),)),
        ((Register("ctrl", quantum_bit_type(), (1,)),), (ControlRegister(1, (1,)),)),
        ((Register("ctrl", unsigned_register_type(1)),), (ControlRegister(1, (1,)),)),
        (
            (Register("ctrl", signed_register_type(1)),),
            (ControlRegister(1, (-1,), "twos_complement"),),
        ),
        ((Register("ctrl", classical_bit_type()),), (ControlRegister(1, (1,)),)),
        (
            (
                Register("ctrl1", quantum_bit_type()),
                Register("ctrl2", unsigned_register_type(2), (2,)),
            ),
            (ControlRegister(1, (1,)), ControlRegister(2, (1, 2))),
        ),
        (
            (
                Register("ctrl1", quantum_bit_type()),
                Register("ctrl2", unsigned_register_type(2), (2,)),
            ),
            (ControlRegister(1, (1,)), ControlRegister(2, (1, 2))),
        ),
    ]
    # Reuse the actual native records' equality/hash, with no foreign-object key.
    for i, (source, actual) in enumerate(zip(sources, native, strict=True)):
        for other_source, other_actual in zip(sources, native, strict=True):
            assert (source == other_source) == (actual == other_actual)
            if actual == other_actual:
                assert hash(actual) == hash(other_actual)
                assert hash(source) == hash(other_source)
        assert source != object()
        assert actual != object()
        if i >= 6:
            signature = Signature(actual[0])
            assert signature.n_qubits() == source.num_qubits == 5
            assert source.num_ctrl_reg == len(actual[0]) == 2
    assert len(set(sources)) == len(set(native)) == 6


def test_frozen_array_cache_divergence_and_explicit_native_snapshot_execution() -> None:
    from qualtran import BloqBuilder, CtrlSpec, QBit
    from qualtran import Signature as SourceSignature
    from qualtran._infra.controlled import make_ctrl_system_with_correct_metabloq
    from qualtran.bloqs.basic_gates import XGate

    raw = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/_infra/controlled.py"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "6fabe760e793f1b39f825d76f80e92b6e5fa5c28"
    )
    array = np.array(1)
    source = CtrlSpec(QBit(), array)
    patterns = normalize_control_patterns(array)
    assert patterns[0] is array and source.cvs[0] is array
    port = Register("ctrl", quantum_bit_type())
    frozen = ControlRegister(1, (int(array),))
    old_hash = hash(source)
    assert source == CtrlSpec()
    array[...] = 0
    assert source == CtrlSpec() and hash(source) == old_hash
    assert source.is_active(0) and not source.is_active(1)
    assert classical_controls_active((0,), patterns)
    assert not classical_controls_active((1,), patterns)
    assert frozen.values == frozen.bits() == (1,)
    bb, inputs = BloqBuilder.from_signature(SourceSignature.build(q=1))
    inputs["q"] = bb.add(XGate(), q=inputs["q"])
    child = bb.finalize(**inputs)
    controlled, _ = make_ctrl_system_with_correct_metabloq(child, source)
    assert type(controlled).__name__ == "Controlled"
    positive = bind_register_program(
        controlled_x_program((port,), (frozen,)), ((0,), (1,))
    )
    assert_source(
        positive.circuit.instructions,
        _source_gates(controlled, {"ctrl": (0,), "q": (1,)}, (), {}),
    )
    x = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
    live_tensor = controlled_register_tensor(
        Signature((port,)), Signature.build(q=1), patterns, lambda: x
    )
    np.testing.assert_array_equal(live_tensor, controlled._tensor_data())
    assert not np.array_equal(live_tensor.reshape(4, 4), controlled.tensor_contract())
    current = ControlRegister(1, (int(array),))
    negative = bind_register_program(
        controlled_x_program((port,), (current,)), ((0,), (1,))
    )
    assert Counter(g.name for g in positive.circuit.instructions) == {"cx": 1}
    assert Counter(g.name for g in negative.circuit.instructions) == {"x": 2, "cx": 1}
    state = torch.tensor(
        [1 + 2j, 3 - 1j, -2j, 4 + 1j, 2 - 3j, 1j, -1 + 4j, 3j], dtype=torch.complex128
    )
    state /= state.norm()
    for bound, matrix in (
        (positive, controlled.tensor_contract()),
        (negative, live_tensor.reshape(4, 4)),
    ):
        circuit = Circuit(3, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(3, bound.circuit.instructions))
        expected = (
            torch.as_tensor(matrix.copy(), dtype=torch.complex128) @ state.reshape(4, 2)
        ).flatten()
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=3e-14
        )
    # Both explicit native activation snapshots remain distinct, stable hash keys.
    assert frozen != current and len({frozen, current}) == 2


def test_unknown_activation_identity_and_mixed_metadata_counts() -> None:
    from qualtran import CBit, CtrlSpec, QBit, QUInt
    from qualtran.symbolics import Shaped
    from sympy import Symbol

    n = Symbol("n", integer=True, positive=True)
    sources = [
        CtrlSpec(QBit(), Shaped((n, 2))),
        CtrlSpec(QBit(), Shaped((n, 2))),
        CtrlSpec(QUInt(1), Shaped((n, 2))),
    ]
    native = [
        (
            (Register("ctrl", quantum_bit_type(), (n, 2)),),
            (SymbolicArrayShape((n, 2)),),
        ),
        (
            (Register("ctrl", quantum_bit_type(), (n, 2)),),
            (SymbolicArrayShape((n, 2)),),
        ),
        (
            (Register("ctrl", unsigned_register_type(1), (n, 2)),),
            (SymbolicArrayShape((n, 2)),),
        ),
    ]
    assert len(set(sources)) == len(set(native)) == 2
    for i, source in enumerate(sources):
        for j, other in enumerate(sources):
            assert (source == other) == (native[i] == native[j])
    values = (np.array([1, 0]), np.array(1))
    source = CtrlSpec((QBit(), CBit()), values)
    patterns = normalize_control_patterns(values)
    signature = Signature(
        (
            Register("ctrl1", quantum_bit_type(), (2,)),
            Register("ctrl2", classical_bit_type()),
        )
    )
    assert signature.n_qubits() == source.num_qubits == 2
    assert signature.n_cbits() == source.num_cbits == 1
    assert signature.n_bits() == source.num_bits == 3
    assert classical_controls_active(
        (values[0], 1), patterns, symbolic_parameters=tuple(signature)
    ) == source.is_active(values[0], 1)
