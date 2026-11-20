"""Exact stored ModDbl adjoint preserves source gates and changed word ownership."""

import pytest

from flagquantum.compiler.registers.modular.modular_halving_component import (
    modular_halving_program,
)
from flagquantum.core.ir.program.register_types import montgomery_register_type
from tests.unit.inverse_component_checks import check_inverse_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("n,p", [(1, 1), (2, 3), (3, 5), (5, 31), (8, 251)])
def test_source_adjoint_stream_aliases_and_resources(n: int, p: int) -> None:
    from qualtran import QMontgomeryUInt
    from qualtran.bloqs.mod_arithmetic.mod_multiplication import ModDbl

    check_inverse_source(
        modular_halving_program(montgomery_register_type(n), p),
        ModDbl(QMontgomeryUInt(n), p).adjoint(),
    )
