"""Native physical programs refuse before digital deployment side effects."""

from __future__ import annotations

from collections.abc import Callable

import pytest

import flagquantum.deployment as deployment
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.errors import CapabilityError
from flagquantum.experimental.simulation import RydbergProgram

pytestmark = pytest.mark.unit


@pytest.fixture
def program() -> RydbergProgram:
    return RydbergProgram(
        atoms=(("a", "trap"),),
        sites={"trap": (0.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=64.0,
        loss_rate=0.2,
        minimum_separation=1.0,
    )


def forbidden(*args: object, **kwargs: object) -> None:
    pytest.fail("native source reached a digital target, compiler or emitter")


@pytest.mark.parametrize("as_record", [False, True])
@pytest.mark.parametrize("measurement_plan", [False, True])
def test_package_intake_refuses_before_target_compilation_and_emission(
    program: RydbergProgram,
    as_record: bool,
    measurement_plan: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("flagquantum.deployment.cloud.compile_program", forbidden)
    monkeypatch.setattr("flagquantum.deployment.cloud.emit_openqasm", forbidden)
    monkeypatch.setattr("flagquantum.deployment.cloud.emit_qcis", forbidden)
    monkeypatch.setattr(deployment.CloudBackendProfile, "simulator", forbidden)
    source = program.to_ir() if as_record else program
    if measurement_plan:
        entry: Callable[..., object] = deployment.create_pauli_measurement_plan
        with pytest.raises(CapabilityError, match="flagquantum.lindblad"):
            entry(source, Hamiltonian([pauli_term(1.0, "Z", (0,))]))
    else:
        package: Callable[..., object] = deployment.create_deployment_package
        with pytest.raises(CapabilityError, match="flagquantum.lindblad"):
            package(source)
