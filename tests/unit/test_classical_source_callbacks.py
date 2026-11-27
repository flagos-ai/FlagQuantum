"""Actual source callbacks retain duck owners, label identity and exact errors."""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as error:
    unavailable_reference(error)

from tests.unit.classical_source_callbacks import (
    source_classical_action_tensors,
    source_tensors,
    source_truth_table,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_truth_table_retains_original_structural_owner_and_rows() -> None:
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Signature
    from qualtran.simulation.classical_sim import get_classical_truth_table

    from flagquantum.core.ir.program.signatures import Signature as NativeSignature
    from tests.unit.test_classical_truth_table import source_adapter

    owner = source_adapter(
        NativeSignature.build(q=1),
        lambda register: (0, 1),
        lambda values: (values["q"],),
    )
    expected = get_classical_truth_table(owner)
    assert source_truth_table(owner) == expected
    # A separately built structural owner is also a source-supported input.
    structural = SimpleNamespace(
        signature=Signature.build_from_dtypes(q=QBit()),
        call_classically=lambda **values: (values["q"],),
    )
    assert source_truth_table(structural) == expected


def test_tensor_callbacks_retain_source_data_and_opaque_label_objects() -> None:
    from qualtran.bloqs.basic_gates.x_basis import XGate

    source = XGate()
    incoming, outgoing = {"q": object()}, {"q": object()}
    expected = source.tensor_contract()
    actual = source_classical_action_tensors(source, incoming, outgoing)
    assert len(actual) == 1
    np.testing.assert_array_equal(actual[0].data, expected)
    assert actual[0].inds[0][0] is outgoing["q"]
    assert actual[0].inds[1][0] is incoming["q"]
    direct = source_tensors(source, incoming, outgoing)
    assert direct[0].inds[0][0] is outgoing["q"]
    assert direct[0].inds[1][0] is incoming["q"]
    np.testing.assert_array_equal(direct[0].data, expected)


def test_bound_callback_preserves_input_mapping_result_and_exception_identity() -> None:
    from quimb.tensor import Tensor

    incoming, outgoing = {"q": object()}, {"q": object()}
    result = [Tensor(np.eye(2), inds=("out", "in"))]
    failure = RuntimeError("original callback failure")

    class Owner:
        fail = False

        def my_tensors(
            self, left: Mapping[str, object], right: Mapping[str, object]
        ) -> list[Tensor]:
            assert left is incoming and right is outgoing
            if self.fail:
                raise failure
            return result

    owner = Owner()
    assert source_tensors(owner, incoming, outgoing) is result
    owner.fail = True
    with pytest.raises(RuntimeError) as caught:
        source_tensors(owner, incoming, outgoing)
    assert caught.value is failure
