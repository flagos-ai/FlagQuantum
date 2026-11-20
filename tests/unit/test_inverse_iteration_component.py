"""Complete stored inverse iteration retains frozen connections, feedback and resources."""

import pytest

from flagquantum.compiler.registers.modular.inversion.inverse_iteration_component import (
    inverse_iteration_program,
)
from tests.unit.inverse_component_checks import check_inverse_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("n,p", [(2, 3), (3, 5), (5, 31), (8, 251)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_source_stream_aliases_and_models(
    n: int, p: int, adjoint: bool
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIteration

    source = _KaliskiIteration(n, p)
    if adjoint:
        source = source.adjoint()
    check_inverse_source(inverse_iteration_program(n, p, adjoint=adjoint), source)
