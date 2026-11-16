"""Audit source PrepareOracle through existing named and packed consumers."""

from hashlib import sha1
from pathlib import Path

import pytest

from examples.algorithms.hubbard_lcu_catalog import HubbardLCUExample
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_prepare_oracle_contract_and_existing_hubbard_binding() -> None:
    from qualtran.bloqs.state_preparation import prepare_base
    from qualtran.bloqs.state_preparation.black_box_prepare import _black_box_prepare
    from qualtran.bloqs.state_preparation.black_box_prepare_test import (
        TestPrepareOracle,
    )

    raw = Path(prepare_base.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "6870d7752aeb7da0c251ad1abb3b90519f2d0556"
    )
    with pytest.raises(TypeError, match="selection_registers"):
        prepare_base.PrepareOracle()

    source = _black_box_prepare.make().prepare
    native = HubbardLCUExample(2)
    selected, junk = source.selection_registers, source.junk_registers
    assert tuple(source.signature) == (*selected, *junk)
    assert (
        tuple((r.name, r.total_bits()) for r in selected)
        == native.prepare_oracle.selection_layout
    )
    assert (
        tuple((r.name, r.total_bits()) for r in junk)
        == native.prepare_oracle.junk_layout
    )
    assert native.prepare_oracle.normalization == source.l1_norm_of_coeffs == 32

    dummy = TestPrepareOracle()
    assert tuple(dummy.signature) == (*dummy.selection_registers, *dummy.junk_registers)
    with pytest.raises(NotImplementedError):
        _ = dummy.l1_norm_of_coeffs


def test_prepare_oracle_default_junk_and_missing_normalization() -> None:
    from qualtran import QAny, Register
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    class SelectionOnly(PrepareOracle):  # type: ignore[misc]
        @property
        def selection_registers(self) -> tuple[object, ...]:
            return (Register("data", QAny(2)),)

    source = SelectionOnly()
    assert source.junk_registers == ()
    assert tuple(source.signature) == source.selection_registers
    with pytest.raises(NotImplementedError):
        _ = source.l1_norm_of_coeffs
    native = BlackBoxPrepare((("data", 2),), (), lambda *_: ())
    assert native.junk_layout == ()
    assert native.normalization is None
    assert native.selection_width == source.signature.n_qubits() == 2
