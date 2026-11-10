"""Raw leaf errors and contextual dense errors follow distinct source paths."""

from typing import Any

import pytest
import torch

from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
    classical_action_tensor_node,
)
from tests.unit.test_classical_action_tensor import decode, encode

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "exception", [ValueError, KeyError, IndexError, NotImplementedError]
)
def test_original_raw_leaf_error_identity_and_dense_error_context(
    exception: type[Exception],
) -> None:
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
        my_tensors_from_classical_action,
    )

    error = exception("action failed")

    def fail(**values: Any) -> tuple[object, ...]:
        raise error

    class Source:
        signature = SourceSignature.build(q=1)
        call_classically = staticmethod(fail)

        def __str__(self) -> str:
            return "fixture"

    source = Source()
    with pytest.raises(exception) as original_leaf:
        my_tensors_from_classical_action(source, {"q": "in"}, {"q": "out"})
    with pytest.raises(exception) as native_leaf:
        classical_action_tensor_node(
            Signature.build(q=1),
            {"q": ((3,),)},
            {"q": ((5,),)},
            fail,
            decode,
            encode,
            torch.empty(()),
            operation_name=str(source),
        )
    assert native_leaf.value is original_leaf.value is error
    with pytest.raises(exception) as original_dense:
        bloq_to_dense_via_classical_action(source)
    with pytest.raises(exception) as native_dense:
        classical_action_tensor(
            Signature.build(q=1), fail, decode, encode, operation_name=str(source)
        )
    assert str(native_dense.value) == str(original_dense.value)
    if exception is ValueError:
        assert native_dense.value.__cause__ is original_dense.value.__cause__ is error
        assert native_dense.value is not error
    else:
        assert native_dense.value is original_dense.value is error


def test_original_raw_and_dense_size_refusals_precede_callbacks() -> None:
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
        my_tensors_from_classical_action,
    )

    def fail(**values: Any) -> tuple[object, ...]:
        raise AssertionError("must not enumerate")

    class Source:
        signature = SourceSignature.build(q=21)
        call_classically = staticmethod(fail)

        def __str__(self) -> str:
            return "huge"

    source = Source()
    with pytest.raises(ValueError) as original_leaf:
        my_tensors_from_classical_action(source, {}, {})
    with pytest.raises(ValueError) as native_leaf:
        classical_action_tensor_node(
            Signature.build(q=21),
            {"q": (tuple(range(21)),)},
            {"q": (tuple(range(21, 42)),)},
            fail,
            decode,
            encode,
            torch.empty(()),
            operation_name=str(source),
        )
    assert (
        str(native_leaf.value)
        == str(original_leaf.value)
        == "tensor is too large: 42 total qubits"
    )
    with pytest.raises(ValueError) as original_dense:
        bloq_to_dense_via_classical_action(source)
    with pytest.raises(ValueError) as native_dense:
        classical_action_tensor(
            Signature.build(q=21), fail, decode, encode, operation_name=str(source)
        )
    assert (
        str(native_dense.value)
        == str(original_dense.value)
        == "cannot compute tensor for huge: tensor is too large: 42 total qubits"
    )
