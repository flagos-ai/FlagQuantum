"""Whole-source modular declarations preserve native typed signature metadata."""

from hashlib import sha1
from pathlib import Path

import pytest
import sympy

from flagquantum.compiler.registers.modular.modular_declarations import (
    ModularDeclaration,
    modular_arithmetic_registers,
    modular_arithmetic_signature,
)
from flagquantum.core.ir.program.registers import RegisterSide

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
KINDS: tuple[ModularDeclaration, ...] = (
    "multi_controlled_x",
    "controlled_subtraction",
    "less_than",
    "controlled_half",
)


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.arithmetic import _shims

    raw = Path(_shims.__file__).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "5c83286454c465065266b20aecf2db9c385089e2"
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("width", (-2, 0, 1, 3, sympy.Symbol("n", integer=True)))
@pytest.mark.parametrize("signed", (False, True))
def test_complete_typed_signatures_and_existing_flat_admission(
    kind: ModularDeclaration, width: int | sympy.Expr, signed: bool
) -> None:
    from qualtran import QBit, QUInt
    from qualtran.bloqs.arithmetic._shims import CHalf, CSub, Lt, MultiCToffoli

    factories = {
        "multi_controlled_x": MultiCToffoli,
        "controlled_subtraction": CSub,
        "less_than": Lt,
        "controlled_half": CHalf,
    }
    source = Lt(width, signed=signed) if kind == "less_than" else factories[kind](width)
    actual = modular_arithmetic_signature(kind, width)
    assert len(actual) == len(source.signature)
    for native, original in zip(actual, source.signature, strict=True):
        assert native.name == original.name
        assert native.shape_symbolic == original.shape_symbolic
        assert native.bitsize == original.bitsize
        assert native.total_bits() == original.total_bits()
        assert native.side is RegisterSide.THRU and native.adjoint() is native
        assert native.dtype.encoding == (
            "bit" if isinstance(original.dtype, QBit) else "uint"
        )
        assert isinstance(original.dtype, (QBit, QUInt))
        assert not native.dtype.parameters
        assert native.is_symbolic() == original.is_symbolic()
        if native.shape_symbolic:
            assert native.shape_symbolic[0] is width
        elif native.dtype.encoding == "uint":
            assert native.bitsize is width
    if isinstance(width, int) and width < 0:
        with pytest.raises(ValueError, match="nonnegative integer width"):
            modular_arithmetic_registers(kind, width)
    else:
        assert modular_arithmetic_registers(kind, width) == tuple(
            (register.name, register.total_bits()) for register in source.signature
        )


def test_existing_flat_query_consumes_typed_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.registers.modular import modular_declarations
    from flagquantum.core.ir.program.register_types import unsigned_register_type
    from flagquantum.core.ir.program.registers import Register
    from flagquantum.core.ir.program.signatures import Signature

    observed: list[tuple[ModularDeclaration, int | sympy.Expr]] = []

    def replacement(kind: ModularDeclaration, width: int | sympy.Expr) -> Signature:
        observed.append((kind, width))
        return Signature((Register("replacement", unsigned_register_type(width)),))

    monkeypatch.setattr(
        modular_declarations, "modular_arithmetic_signature", replacement
    )
    assert modular_arithmetic_registers("less_than", 5) == (("replacement", 5),)
    assert observed == [("less_than", 5)]
