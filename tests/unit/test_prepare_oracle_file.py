"""Audit source PrepareOracle through existing named and packed consumers."""

from collections.abc import Mapping
from hashlib import sha1
from pathlib import Path

import pytest
from sympy import Symbol

from examples.algorithms.hubbard_lcu_catalog import HubbardLCUExample
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.algorithms.primitives.prepare_oracle import PrepareOracle
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
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


@pytest.mark.ftqc_reference
def test_prepare_oracle_default_junk_and_missing_normalization() -> None:
    from qualtran import QAny, Register
    from qualtran.bloqs.state_preparation.prepare_base import (
        PrepareOracle as SourcePrepare,
    )

    class SelectionOnly(SourcePrepare):  # type: ignore[misc]
        @property
        def selection_registers(self) -> tuple[object, ...]:
            return (Register("data", QAny(2)),)

    source = SelectionOnly()
    typed = _SelectionOnly()
    assert PrepareOracle.__abstractmethods__ == SourcePrepare.__abstractmethods__
    assert tuple(reg.name for reg in typed.signature) == ("i", "j")
    assert typed.junk_registers == source.junk_registers == ()
    with pytest.raises(NotImplementedError):
        _ = typed.l1_norm_of_coeffs
    assert tuple(source.signature) == source.selection_registers
    with pytest.raises(NotImplementedError):
        _ = source.l1_norm_of_coeffs
    native = BlackBoxPrepare((("data", 2),), (), lambda *_: ())
    assert native.junk_layout == ()
    assert native.normalization is None
    assert native.selection_width == source.signature.n_qubits() == 2


class _SelectionOnly(PrepareOracle):
    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (Register("i", quantum_bit_type()), Register("j", quantum_bit_type()))


def test_prepare_signature_keeps_symbolic_and_output_descriptors() -> None:
    n = Symbol("n", integer=True, positive=True)
    junk = Register(
        "work", opaque_register_type(n), shape=(2,), side=RegisterSide.RIGHT
    )

    class _DeclaredJunk(_SelectionOnly):
        @property
        def junk_registers(self) -> tuple[Register, ...]:
            return (junk,)

    oracle = _DeclaredJunk()
    assert tuple(oracle.signature) == (*oracle.selection_registers, junk)
    assert oracle.signature is oracle.signature
    assert oracle.signature.get_right("work") is junk
    assert tuple(oracle.signature.lefts()) == oracle.selection_registers
    assert junk.total_bits() == 2 * n


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_hubbard_uses_the_ordered_prepare_signature(dimension: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard import (
        PrepareHubbard as SourceHubbard,
    )

    native = PrepareHubbard(dimension, t=1.5, u=2.0)
    reference = SourceHubbard(dimension, dimension, t=1.5, u=2.0)
    assert native.l1_norm_of_coeffs == native.alpha == reference.l1_norm_of_coeffs
    assert [
        (reg.name, reg.bitsize, reg.side.value, dict(reg.dtype.parameters).get("bound"))
        for reg in native.signature
    ] == [
        (
            reg.name,
            reg.bitsize,
            reg.side.value,
            getattr(reg.dtype, "iteration_length", None),
        )
        for reg in reference.signature
    ]
    assert (
        tuple((reg.name, reg.total_bits()) for reg in native.signature) == native.layout
    )


def test_packed_black_box_exposes_its_distinct_binding() -> None:
    oracle = _SelectionOnly()
    names = tuple(reg.name for reg in oracle.selection_registers)
    bound: list[tuple[int, ...]] = []

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[()]:
        bound.extend(ports[name] for name in names)
        return ()

    box = BlackBoxPrepare(tuple((name, 1) for name in names), (), emit)
    assert tuple(reg.name for reg in box.signature) == ("selection",)
    assert box.normalization is None
    assert box.emit((1, 0), (), (), ()) == ()
    assert bound == [(1,), (0,)]
