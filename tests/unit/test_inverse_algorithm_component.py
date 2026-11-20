"""Complete stored inverse algorithm retains frozen connections, feedback and resources."""

import pytest

from flagquantum.compiler.registers.modular.inversion.inverse_algorithm_component import (
    inverse_algorithm_program,
)
from tests.unit.inverse_component_checks import check_inverse_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("n,p", [(2, 1), (3, 5), (5, 13), (8, 251)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_source_stream_aliases_and_models(
    n: int, p: int, adjoint: bool
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiModInverseImpl

    source = _KaliskiModInverseImpl(n, p)
    if adjoint:
        source = source.adjoint()
    check_inverse_source(inverse_algorithm_program(n, p, adjoint=adjoint), source)
