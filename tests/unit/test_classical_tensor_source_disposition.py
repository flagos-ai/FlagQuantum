"""Complete classical tensor source boundaries without repairing the reference."""

from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.simulation.tensor_network.classical_action_tensor import (
    classical_action_tensor,
    classical_action_tensor_node,
)
from tests.unit.test_classical_action_tensor import (
    decode,
    encode,
    signature_from_source,
)
from tests.unit.test_register_tensor_binding import source_bindings

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (2, 0)])
@pytest.mark.parametrize("side", ["LEFT", "RIGHT", "THRU"])
def test_full_direction_layout_dense_and_leaf_disposition(
    classical: bool, shape: tuple[int, ...], side: str
) -> None:
    from qualtran import CBit, QBit, Register, Side
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.tensor._tensor_from_classical import (
        bloq_to_dense_via_classical_action,
        my_tensors_from_classical_action,
    )

    source_calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    native_calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def action(
        log: list[tuple[str, tuple[int, ...], tuple[int, ...]]], **values: Any
    ) -> tuple[object, ...]:
        for name, value in values.items():
            array = np.asarray(value)
            log.append((name, array.shape, tuple(int(i) for i in array.flat)))
        if side == "LEFT":
            return ()
        return (values["bits"] if side == "THRU" else np.zeros(shape, dtype=int),)

    class Source:
        signature = SourceSignature(
            [
                Register(
                    "bits",
                    CBit() if classical else QBit(),
                    shape=shape,
                    side=Side[side],
                )
            ]
        )

        def call_classically(self, **values: Any) -> tuple[object, ...]:
            return action(source_calls, **values)

        def __str__(self) -> str:
            return "direction-layout"

    source = Source()
    signature = signature_from_source(source.signature)

    def native_action(**values: Any) -> tuple[object, ...]:
        return action(native_calls, **values)

    # Raw action objects isolate enumeration from foreign builder validation.
    # Compare exact exceptions and callback prefixes, including failures before
    # the first action for classical arrays and empty codec inputs.
    try:
        expected = bloq_to_dense_via_classical_action(source)
    except (ValueError, IndexError, AssertionError, TypeError) as error:
        with pytest.raises(type(error)) as native_error:
            classical_action_tensor(
                signature, native_action, operation_name=str(source)
            )
        assert str(native_error.value) == str(error)
        assert type(native_error.value.__cause__) is type(error.__cause__)
    else:
        actual = classical_action_tensor(
            signature, native_action, operation_name=str(source)
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype == np.float64
    assert native_calls == source_calls

    source_calls.clear()
    native_calls.clear()
    labels: dict[tuple[str, int], int] = {}
    src_in, incoming = source_bindings(source.signature.lefts(), "in", labels)
    src_out, outgoing = source_bindings(source.signature.rights(), "out", labels)
    if classical:
        # The original leaf counts total bits for data but only quantum bits
        # for indices. Its raw rank/reshape failure is not silently fixed.
        with pytest.raises((ValueError, IndexError, AssertionError, TypeError)):
            my_tensors_from_classical_action(source, src_in, src_out)
        with pytest.raises(ValueError, match="require quantum ports"):
            classical_action_tensor_node(
                signature,
                incoming,
                outgoing,
                native_action,
                decode,
                encode,
                torch.empty(()),
                operation_name=str(source),
            )
        assert native_calls == []
    else:
        try:
            original = my_tensors_from_classical_action(source, src_in, src_out)[0]
        except (ValueError, IndexError, AssertionError, TypeError) as error:
            with pytest.raises(type(error)) as native_error:
                classical_action_tensor_node(
                    signature,
                    incoming,
                    outgoing,
                    native_action,
                    decode,
                    encode,
                    torch.empty(()),
                    operation_name=str(source),
                )
            assert str(native_error.value) == str(error)
        else:
            node = classical_action_tensor_node(
                signature,
                incoming,
                outgoing,
                native_action,
                decode,
                encode,
                torch.empty((), dtype=torch.float64),
                operation_name=str(source),
            )
            np.testing.assert_array_equal(node.tensor.numpy(), original.data)
            assert node.labels == tuple(labels[index] for index in original.inds)
        assert native_calls == source_calls
