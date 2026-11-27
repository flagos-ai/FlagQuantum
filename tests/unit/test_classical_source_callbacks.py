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


@pytest.mark.parametrize("basis", ["x", "z"])
def test_original_measurement_instrument_preserves_both_owned_elements(
    basis: str,
) -> None:
    from qualtran.bloqs.basic_gates.x_basis import MeasureX
    from qualtran.bloqs.basic_gates.z_basis import MeasureZ
    from qualtran.simulation.tensor._quimb import DiscardInd

    from tests.unit.classical_source_callbacks import source_measurement_tensors

    owner = MeasureX() if basis == "x" else MeasureZ()
    incoming, outgoing = {"q": object()}, {"c": object()}
    raw = getattr(owner, "my_tensors")(incoming, outgoing)
    tensor, discarded = source_measurement_tensors(owner, incoming, outgoing)
    assert isinstance(discarded, DiscardInd)
    np.testing.assert_array_equal(tensor.data, raw[0].data)
    # Each original invocation allocates its own private UUID. Keep that index
    # tied to its own discard marker and compare the caller-owned connections.
    assert discarded.ind_tuple in tensor.inds and raw[1].ind_tuple in raw[0].inds
    assert discarded.ind_tuple[0] not in (*incoming.values(), *outgoing.values())
    assert raw[1].ind_tuple[0] not in (*incoming.values(), *outgoing.values())
    actual_connections = tuple(
        "private" if index == discarded.ind_tuple else index for index in tensor.inds
    )
    expected_connections = tuple(
        "private" if index == raw[1].ind_tuple else index for index in raw[0].inds
    )
    assert actual_connections == expected_connections


def test_original_discard_marker_and_controlled_matrix_keep_source_labels() -> None:
    from qualtran._infra.controlled import CtrlSpec
    from qualtran._infra.data_types import CBit
    from qualtran.bloqs.basic_gates.discard import Discard
    from qualtran.bloqs.basic_gates.x_basis import XGate

    from tests.unit.classical_source_callbacks import (
        source_controlled_tensor,
        source_discard_index,
        source_discard_tensors,
    )

    label = object()
    markers = source_discard_tensors(Discard(), {"c": label}, {})
    assert markers == [source_discard_index(label, 0)]
    assert markers[0].ind_tuple[0] is label
    owner = XGate().controlled(CtrlSpec(CBit(), cvs=1))
    expected = getattr(owner, "_tensor_data")()
    actual = source_controlled_tensor(owner)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_synthetic_soquet_keeps_source_constructor_and_nominal_register() -> None:
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register

    from tests.unit.classical_source_callbacks import source_placeholder_soquet

    register = Register("q", QBit(), shape=(2,))
    actual = source_placeholder_soquet(register, (1,))
    assert actual.reg is register and actual.idx == (1,)
    assert getattr(actual, "binst") is None


def test_original_join_accepts_its_split_array_without_unsigned_coercion() -> None:
    from qualtran._infra.data_types import QGF
    from qualtran.bloqs.bookkeeping.join import Join
    from qualtran.bloqs.bookkeeping.split import Split

    from tests.unit.classical_source_callbacks import source_classical_values

    dtype = QGF(2, 3)
    for value in range(8):
        bits = Split(dtype).on_classical_vals(reg=dtype.gf_type(value))["reg"]
        assert isinstance(bits, np.ndarray)
        original = bits.copy()
        for array in (bits, bits.astype(np.int64)):
            owner = Join(dtype)
            expected = getattr(owner, "on_classical_vals")(reg=array)
            actual = source_classical_values(owner, reg=array)
            assert type(actual["reg"]) is type(expected["reg"])
            np.testing.assert_array_equal(actual["reg"], expected["reg"])
        np.testing.assert_array_equal(bits, original)
        assert bits.dtype == original.dtype
