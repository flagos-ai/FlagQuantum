"""Dense register-axis order and zero-port projections match pinned source."""

from itertools import product
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest
import sympy

from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.register_tensor import (
    register_identity_tensor,
    register_tensor_from_matrix,
    register_tensor_shape,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def native_signature(source: Any) -> Signature:
    return Signature(
        Register(
            reg.name,
            RegisterType("opaque", int(reg.bitsize)),
            reg.shape,
            RegisterSide[reg.side.name],
        )
        for reg in source
    )


@pytest.mark.parametrize("sides", list(product(("LEFT", "RIGHT", "THRU"), repeat=3)))
def test_directional_matrix_and_identity_projection(sides: tuple[str, ...]) -> None:
    from qualtran._infra.data_types import QAny
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Side
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_data_manipulation import (
        eye_tensor_for_signature,
        tensor_data_from_unitary_and_signature,
        tensor_shape_from_signature,
    )

    source = SourceSignature(
        SourceRegister(name, QAny(width), shape=shape, side=Side[side])
        for name, width, shape, side in zip(
            ("a", "b", "c"), (1, 2, 0), ((1, 2), (), ()), sides, strict=True
        )
    )
    native = native_signature(source)
    rng = np.random.default_rng(109)
    matrix = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    unchanged = matrix.copy()
    assert register_tensor_shape(native) == tensor_shape_from_signature(source)
    np.testing.assert_array_equal(
        register_tensor_from_matrix(matrix, native),
        tensor_data_from_unitary_and_signature(matrix, source),
    )
    np.testing.assert_array_equal(
        register_identity_tensor(native), eye_tensor_for_signature(source)
    )
    np.testing.assert_array_equal(matrix, unchanged)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0), (1, 2)])
def test_zero_width_and_empty_array_have_distinct_axes(shape: tuple[int, ...]) -> None:
    from qualtran._infra.data_types import QAny
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_data_manipulation import (
        eye_tensor_for_signature,
    )

    source = SourceSignature([SourceRegister("empty", QAny(0), shape=shape)])
    actual = register_identity_tensor(native_signature(source))
    np.testing.assert_array_equal(actual, eye_tensor_for_signature(source))
    assert actual.shape == (1,) * (2 * len(tuple(source[0].all_idxs())))


@pytest.mark.parametrize("width", [-1, 1.5, sympy.Symbol("n")])
def test_nonfinite_or_invalid_width_fails_before_allocating(width: Any) -> None:
    with pytest.raises(ValueError, match="widths"):
        register_identity_tensor(
            Signature([Register("q", RegisterType("opaque", width))])
        )


@pytest.mark.parametrize("shape", [(-1,), (True,), (sympy.Symbol("n"),)])
def test_invalid_array_dimensions_fail_before_allocating(shape: Any) -> None:
    with pytest.raises(ValueError):
        register_identity_tensor(
            Signature([Register("q", RegisterType("opaque", 1), shape)])
        )


def test_matrix_arity_is_checked() -> None:
    with pytest.raises(ValueError, match="matrix shape"):
        register_tensor_from_matrix(
            np.eye(3, dtype=np.complex128), Signature.build(q=1)
        )
