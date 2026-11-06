"""Compare the projectile cost graphs against the locked source."""

import pytest

from flagquantum.algorithms.first_quantization_projectile_resource_model import (
    source_projectile_power_two_cost,
    source_projectile_prepare_t_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("p", "n", "eta", "precision"),
    [(1, 1, 2, 1), (2, 2, 3, 4), (2, 4, 5, 6), (6, 8, 10, 8)],
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.ftqc_reference
def test_projectile_kinetic_prepare(
    p: int, n: int, eta: int, precision: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_t import (
        PreparePowerTwoStateWithProj,
        PrepareTFirstQuantizationWithProj,
    )

    power = PreparePowerTwoStateWithProj(n, p, is_adjoint=adjoint)
    source = PrepareTFirstQuantizationWithProj(p, n, eta, precision, is_adjoint=adjoint)
    model = source_projectile_prepare_t_cost(p, n, adjoint=adjoint)
    assert_source_model(source_projectile_power_two_cost(n, p, adjoint=adjoint), power)
    assert_source_model(model, source)
    assert [(reg.name, reg.bitsize) for reg in source.signature.lefts()] == [
        ("w", 2),
        ("w_mean", 2),
        ("r", n),
        ("s", n),
    ]
    assert source.bloq_counts() == {Toffoli(): 15 if adjoint else 31, power: 2}
    assert [call.repetitions for call in model.calls] == [15 if adjoint else 31, 2]
    for operation in (power, source):
        with pytest.raises(DecomposeNotImplementedError):
            operation.decompose_bloq()


def test_projectile_kinetic_roundtrip_and_width_validation() -> None:
    forward = estimate_ftqc_resources(source_projectile_prepare_t_cost(6, 8))
    reverse = estimate_ftqc_resources(
        source_projectile_prepare_t_cost(6, 8, adjoint=True)
    )
    assert forward.logical.gates.toffoli == 47
    assert reverse.logical.gates.toffoli == 27
    assert (
        forward.logical.peak_logical_qubits == reverse.logical.peak_logical_qubits == 20
    )
    assert forward.emitted is reverse.emitted is None
    with pytest.raises(ValueError, match="at least the electron width"):
        source_projectile_prepare_t_cost(8, 6)
