"""Raw register tensor source audit and replacement of validated consumers."""

from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import pytest
import sympy
from numpy.typing import NDArray

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.controlled_tensor import (
    controlled_register_tensor,
)
from flagquantum.simulation.tensor_network.register_tensor import (
    register_identity_tensor,
    register_tensor_from_matrix,
    register_tensor_shape,
)
from flagquantum.simulation.tensor_network.register_tensor_data import (
    _declared_bits,
    project_register_tensor,
    register_active_indices,
    register_identity_data,
    register_word_shape,
    register_word_shapes,
)
from flagquantum.simulation.value_dimensions import SymbolicArrayShape

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def native_signature(source: Any) -> Signature:
    return Signature(
        Register(
            reg.name,
            RegisterType("opaque", reg.bitsize),
            reg.shape_symbolic,
            RegisterSide[reg.side.name],
        )
        for reg in source
    )


def result(function: Callable[..., Any], *args: Any) -> Any:
    try:
        return function(*args)
    except (AssertionError, ValueError, TypeError, IndexError, AttributeError) as error:
        return type(error), str(error)


@pytest.mark.parametrize("sides", list(product(("LEFT", "RIGHT", "THRU"), repeat=3)))
@pytest.mark.parametrize(
    "dtype",
    [np.bool_, np.int8, np.int64, np.float32, np.complex64, np.complex128, object],
)
def test_complete_projection_preserves_source_dtype_subspace_and_views(
    sides: tuple[str, ...], dtype: Any
) -> None:
    from qualtran import QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    signature = SourceSignature(
        SourceRegister(name, QAny(width), shape=shape, side=Side[side])
        for name, width, shape, side in zip(
            ("a", "b", "c"), (1, 1, 0), ((1, 2), (), ()), sides, strict=True
        )
    )
    native = native_signature(signature)
    matrix = np.arange(64).reshape(8, 8).astype(dtype)
    if np.issubdtype(matrix.dtype, np.complexfloating):
        matrix += 1j * matrix[::-1]
    for data in (matrix, matrix.T, np.asfortranarray(matrix)):
        original = data.copy()
        actual = project_register_tensor(data, native)
        expected = source.tensor_data_from_unitary_and_signature(data, signature)
        np.testing.assert_array_equal(actual, expected)
        assert type(actual) is type(expected) and actual.dtype == expected.dtype
        assert np.shares_memory(actual, data) == np.shares_memory(expected, data)
        np.testing.assert_array_equal(data, original)
    assert register_word_shapes(native) == source.tensor_out_inp_shape_from_signature(
        signature
    )
    assert register_word_shape(native) == source.tensor_shape_from_signature(signature)
    assert _declared_bits(native) == source._n_qubits(signature)
    np.testing.assert_array_equal(
        register_identity_data(native), source.eye_tensor_for_signature(signature)
    )


@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (1, 2)])
@pytest.mark.parametrize("side", ["LEFT", "RIGHT", "THRU"])
def test_zero_width_and_empty_arrays_include_scalar_result_identity(
    shape: tuple[int, ...], side: str
) -> None:
    from qualtran import QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    signature = SourceSignature(
        [SourceRegister("q", QAny(0), shape=shape, side=Side[side])]
    )
    actual = register_identity_data(native_signature(signature))
    expected = source.eye_tensor_for_signature(signature)
    np.testing.assert_array_equal(actual, expected)
    assert type(actual) is type(expected) and actual.shape == expected.shape


@pytest.mark.parametrize(
    "width",
    [-1, 0, 1, sympy.Integer(2), sympy.Symbol("n", integer=True, positive=True)],
)
def test_raw_word_shape_preserves_symbolic_and_negative_powers(width: Any) -> None:
    from qualtran import QAny
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    signature = SourceSignature([SourceRegister("q", QAny(width), shape=(1, 2))])
    native = native_signature(signature)
    assert register_word_shapes(native) == source.tensor_out_inp_shape_from_signature(
        signature
    )
    assert register_word_shape(native) == source.tensor_shape_from_signature(signature)


def test_input_side_shape_failure_precedes_output_side() -> None:
    from qualtran import QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    signature = SourceSignature(
        [
            SourceRegister("out", QAny(1), shape=(sympy.Symbol("r"),), side=Side.RIGHT),
            SourceRegister("in", QAny(1), shape=(sympy.Symbol("l"),), side=Side.LEFT),
        ]
    )
    original = result(source.tensor_out_inp_shape_from_signature, signature)
    native = result(register_word_shapes, native_signature(signature))
    # Metadata repr names are native; the first failing register remains input.
    assert original[0] is native[0] is ValueError
    assert "name='in'" in original[1] and "name='in'" in native[1]


@pytest.mark.parametrize("child_side", ["LEFT", "RIGHT", "THRU"])
@pytest.mark.parametrize(
    "pattern",
    [
        np.array(0),
        np.array(1),
        np.array(-1),
        np.array([0, 1]),
        np.array([[1, 0]]),
        np.empty((0,), dtype=int),
    ],
)
def test_active_indices_source_word_offsets_and_negative_indexing(
    child_side: str, pattern: NDArray[Any]
) -> None:
    from qualtran import CtrlSpec, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    # Construct spec without the signed codec: active-space helper consumes raw
    # arrays, independently of CtrlSpec's public validation of those values.
    spec = CtrlSpec(cvs=np.zeros(pattern.shape, dtype=int))
    object.__setattr__(spec, "cvs", (pattern,))
    signature = SourceSignature(
        [
            SourceRegister("ctrl", QAny(1), shape=pattern.shape),
            SourceRegister("q", QAny(1), side=Side[child_side]),
        ]
    )
    actual = register_active_indices(native_signature(signature), (pattern,))
    expected = source.active_space_for_ctrl_spec(signature, spec)
    assert actual == expected


@pytest.mark.parametrize(
    "patterns", [(np.array(2), np.array([[0, 1]])), (), (np.array([0, 1]),)]
)
def test_raw_active_space_consumes_patterns_without_native_count_validation(
    patterns: tuple[NDArray[Any], ...],
) -> None:
    from qualtran import CtrlSpec, QAny
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    signature = SourceSignature(
        [
            SourceRegister("a", QAny(2)),
            SourceRegister("b", QAny(1), shape=(1, 2)),
            SourceRegister("q", QAny(1)),
        ]
    )
    spec = CtrlSpec()
    object.__setattr__(spec, "cvs", patterns)
    assert result(
        register_active_indices, native_signature(signature), patterns
    ) == result(source.active_space_for_ctrl_spec, signature, spec)


def test_symbolic_and_nonarray_patterns_retain_rejection_boundary() -> None:
    with pytest.raises(ValueError, match="symbolic"):
        register_active_indices(Signature.build(q=1), (SymbolicArrayShape(()),))
    with pytest.raises(AssertionError):
        invalid: Any = 1
        register_active_indices(Signature.build(q=1), (invalid,))


@pytest.mark.parametrize("matrix", [np.eye(3), np.zeros((2, 2, 1)), np.zeros((0, 0))])
def test_raw_shape_assertion_is_distinct_from_validated_matrix_error(
    matrix: NDArray[Any],
) -> None:
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    assert result(project_register_tensor, matrix, Signature.build(q=1)) == result(
        source.tensor_data_from_unitary_and_signature,
        matrix,
        SourceSignature.build(q=1),
    )
    with pytest.raises(ValueError, match="matrix shape"):
        register_tensor_from_matrix(matrix, Signature.build(q=1))


def test_paired_directional_name_inherits_thru_reconstruction_failure() -> None:
    from qualtran import QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    signature = SourceSignature(
        [
            SourceRegister("q", QAny(1), side=Side.LEFT),
            SourceRegister("q", QAny(1), side=Side.RIGHT),
        ]
    )
    native = native_signature(signature)
    matrix = np.eye(4, dtype=np.complex128)
    assert result(project_register_tensor, matrix, native) == result(
        source.tensor_data_from_unitary_and_signature, matrix, signature
    )
    assert result(register_identity_data, native) == result(
        source.eye_tensor_for_signature, signature
    )
    # The validated native interface keeps its pre-existing same-name support.
    expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    np.testing.assert_array_equal(register_tensor_from_matrix(matrix, native), expected)
    np.testing.assert_array_equal(register_identity_tensor(native), expected)


def test_matrix_subclass_and_scalar_object_values_preserve_original_behavior() -> None:
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    class Array(np.ndarray[Any, np.dtype[Any]]):
        pass

    for matrix in (np.eye(2).view(Array), np.array([[sympy.I]], dtype=object)):
        signature = (
            SourceSignature.build(q=1)
            if matrix.shape == (2, 2)
            else SourceSignature([])
        )
        actual = result(project_register_tensor, matrix, native_signature(signature))
        expected = result(
            source.tensor_data_from_unitary_and_signature, matrix, signature
        )
        assert type(actual) is type(expected)
        if isinstance(expected, tuple):
            assert actual == expected
        else:
            np.testing.assert_array_equal(actual, expected)


def test_source_consumers_with_only_native_tensor_helper_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.simulation.tensor import _tensor_data_manipulation_test as original

    def shape(signature: Any) -> tuple[Any, ...]:
        return register_word_shape(native_signature(signature))

    def shapes(signature: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        return register_word_shapes(native_signature(signature))

    def eye(signature: Any) -> object:
        return register_identity_data(native_signature(signature))

    def active(signature: Any, spec: Any) -> tuple[int | slice, ...]:
        return register_active_indices(native_signature(signature), spec.cvs)

    for name, function in (
        ("tensor_shape_from_signature", shape),
        ("tensor_out_inp_shape_from_signature", shapes),
        ("eye_tensor_for_signature", eye),
        ("active_space_for_ctrl_spec", active),
    ):
        monkeypatch.setattr(original, name, function)
    original.test_tensor_shape_from_signature()
    original.test_eye_tensor_for_signature()
    original.test_active_space_for_ctrl_spec(0)
    original.test_active_space_for_ctrl_spec(1)


def test_existing_validated_owners_use_shared_projection_and_active_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.simulation.tensor_network import controlled_tensor, register_tensor

    matrix = np.array([[1, 1j], [-1j, -1]], dtype=np.complex128)
    calls = []

    def project(data: NDArray[Any], signature: Signature) -> NDArray[Any]:
        calls.append(("project", signature))
        return data

    def active(signature: Signature, patterns: Any) -> tuple[int | slice, ...]:
        calls.append(("active", signature))
        return (1, slice(2), 1, slice(2))

    monkeypatch.setattr(register_tensor, "project_register_tensor", project)
    assert register_tensor_from_matrix(matrix, Signature.build(q=1)) is matrix
    monkeypatch.setattr(controlled_tensor, "register_active_indices", active)
    actual = controlled_register_tensor(
        Signature.build(ctrl=1), Signature.build(q=1), (np.array(1),), lambda: matrix
    )
    np.testing.assert_array_equal(actual[1, :, 1, :], matrix)
    assert tuple(c[0] for c in calls) == ("project", "active")
    assert register_tensor_shape(Signature.build(q=1)) == (2, 2)


def test_symbolic_control_dtype_is_rejected_before_shape_slicing() -> None:
    from qualtran import CtrlSpec, QUInt
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    width = sympy.Symbol("n", positive=True, integer=True)
    spec = CtrlSpec(qdtypes=QUInt(width), cvs=0)
    signature = SourceSignature([SourceRegister("ctrl", QUInt(width))])
    expected = result(source.active_space_for_ctrl_spec, signature, spec)
    actual = result(register_active_indices, native_signature(signature), spec.cvs)
    assert expected[0] is actual[0] is ValueError
    assert "symbolic" in expected[1] and "symbolic" in actual[1]


@pytest.mark.parametrize("symbolic_dtype", [False, True])
def test_control_dtype_classification_precedes_pattern_query(
    monkeypatch: pytest.MonkeyPatch, symbolic_dtype: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran import Signature as SourceSignature
    from qualtran._infra import controlled as source_control
    from qualtran.simulation.tensor import _tensor_data_manipulation as source

    from flagquantum.simulation.tensor_network import register_tensor_data as native

    traces = []
    calls: list[str] = []
    spec = CtrlSpec(cvs=1)
    for module, name, invoke in (
        (
            source_control,
            "is_symbolic",
            lambda: source.active_space_for_ctrl_spec(
                SourceSignature.build(ctrl=1), spec
            ),
        ),
        (
            native,
            "is_symbolic_value",
            lambda: register_active_indices(Signature.build(ctrl=1), spec.cvs),
        ),
    ):
        calls.clear()

        def classify(*values: object) -> bool:
            calls.append("patterns" if isinstance(values[0], np.ndarray) else "dtypes")
            return symbolic_dtype and len(calls) == 1

        with monkeypatch.context() as context:
            context.setattr(module, name, classify)
            if symbolic_dtype:
                with pytest.raises(ValueError, match="symbolic"):
                    invoke()
            else:
                assert invoke() == (1, 1)
        traces.append(calls.copy())
    expected = ["dtypes"] if symbolic_dtype else ["dtypes", "patterns"]
    assert traces == [expected, expected]
