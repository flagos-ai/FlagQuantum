"""Check the service/transport roster and its remaining capability gaps.

The exact exported adapter set must match the parity row. A shared HTTP
transport is not hardware execution evidence; retiring a generic physics-route
placeholder must not be reported as a new cloud integration.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

import pytest

import flagquantum.remote as remote
import flagquantum.remote.qpu as qpu

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "contracts" / "cudaq-parity-matrix.toml"

#: Every adapter the package declares, measured by subclassing ``QuantumProvider``.
DECLARED_PROVIDERS = frozenset(
    {
        "AmazonBraketProvider",
        "AzureQuantumProvider",
        "HttpQuantumProvider",
        "IQMProvider",
        "IonQProvider",
        "QuafuProvider",
        "QuantinuumProvider",
    }
)

#: The declaration readers, one per adapter that reads a vendor listing.
DECLARED_PROFILES = frozenset(
    {
        "azure_backend_profile",
        "braket_backend_profile",
        "ionq_backend_profile",
        "iqm_backend_profile",
        "quantinuum_backend_profile",
    }
)


def _rows() -> dict[str, dict[str, Any]]:
    document = tomllib.loads(MATRIX.read_text())
    return {
        str(row["id"]): row
        for domain in document["domains"]
        for row in domain["capabilities"]
    }


def test_the_roster_is_the_set_the_row_names() -> None:
    """Seven declared adapters, including the generic HTTP transport."""

    exported = {name for name in qpu.__all__ if name.endswith("Provider")}
    assert exported == DECLARED_PROVIDERS | {"QuantumProvider"}

    concrete = {
        name
        for name in DECLARED_PROVIDERS
        if not getattr(getattr(qpu, name), "__abstractmethods__", frozenset())
    }
    assert concrete == DECLARED_PROVIDERS, sorted(DECLARED_PROVIDERS - concrete)

    for name in DECLARED_PROVIDERS:
        assert issubclass(getattr(qpu, name), qpu.QuantumProvider), name

    # Every adapter is declared by this package, which is what keeps the census a
    # statement about the roster rather than about whatever else in the tree
    # happens to subclass the protocol.
    for name in DECLARED_PROVIDERS:
        assert getattr(qpu, name).__module__.startswith("flagquantum.remote.qpu"), name

    # ``QuafuProvider`` is the one adapter that reaches ``QuantumProvider`` through
    # another adapter rather than directly, which is why counting direct
    # subclasses alone undercounts the roster by one.
    assert qpu.QuafuProvider.__bases__ == (qpu.HttpQuantumProvider,)

    # ``QuantumProvider`` does have a ninth direct subclass, and it is deliberately
    # not an adapter: the in-memory target is a testing double that lives in
    # ``flagquantum.testing`` and is exported from neither ``remote.qpu`` nor
    # ``remote``. A census that walked ``__subclasses__()`` would count it as a
    # provider and over-report the roster by one.
    from flagquantum.testing import InMemoryRemoteTarget

    assert issubclass(InMemoryRemoteTarget, qpu.QuantumProvider)
    assert InMemoryRemoteTarget.__module__ == "flagquantum.testing.remote"
    exported_provider_classes = {
        getattr(qpu, name) for name in qpu.__all__ if name.endswith("Provider")
    }
    assert InMemoryRemoteTarget not in exported_provider_classes
    assert "InMemoryRemoteTarget" not in qpu.__all__
    assert "InMemoryRemoteTarget" not in remote.__all__
    # Neither is it reachable as an attribute: ``__all__`` is what a ``*`` import
    # carries, so a name the module merely imported would still be one attribute
    # access away from being read as a roster member.
    assert not hasattr(qpu, "InMemoryRemoteTarget")
    assert not hasattr(remote, "InMemoryRemoteTarget")

    assert remote.__all__ == qpu.__all__


def test_the_declaration_readers_are_the_ones_that_read_a_vendor_listing() -> None:
    """Six profile builders, and a Quafu route that is a calibration reader."""

    profiles = {name for name in qpu.__all__ if name.endswith("_backend_profile")}
    assert profiles == DECLARED_PROFILES

    # Quafu is the adapter without a profile builder: its declaration route reads a
    # chip listing into a noise model instead, which is a different artefact and not
    # a missing profile.
    assert not hasattr(qpu, "quafu_backend_profile")
    assert "quafu_noise_model_from_chip_info" in qpu.__all__
    for name in ("fetch_calibration", "fetch_chip_info"):
        assert callable(getattr(qpu.QuafuProvider, name)), name


@pytest.mark.parametrize(
    "builder,message",
    [
        (lambda: qpu.ionq_backend_profile({}), "IonQ declaration does not expose"),
        (lambda: qpu.iqm_backend_profile({}), "IQM declaration does not expose"),
        (
            lambda: qpu.quantinuum_backend_profile({}),
            "Quantinuum declaration does not expose",
        ),
        (lambda: qpu.braket_backend_profile({}), "Braket device does not expose"),
    ],
)
def test_every_reader_refuses_an_empty_declaration(
    builder: Callable[[], object], message: str
) -> None:
    """An empty listing is refused, in the adapter's own words."""

    with pytest.raises(ValueError, match=message):
        builder()


def test_the_azure_reader_refuses_a_width_it_cannot_use() -> None:
    """Azure takes its width from the caller, so the refusal is on the width."""

    with pytest.raises(ValueError, match="width must be positive"):
        qpu.azure_backend_profile({"id": "ionq.simulator"}, n_qubits=0)
    with pytest.raises(ValueError, match="does not expose an id"):
        qpu.azure_backend_profile({}, n_qubits=3)


def test_a_profile_carries_the_devices_own_declaration_not_a_default() -> None:
    """The declared facts survive the crossing, which is what makes it a roster."""

    ionq = qpu.ionq_backend_profile({"backend": "simulator", "qubits": 32})
    assert (ionq.provider, ionq.name, ionq.n_qubits) == ("ionq", "simulator", 32)
    assert ionq.is_simulator is True

    iqm = qpu.iqm_backend_profile(
        {"name": "garnet", "qubits": 20, "basis_gates": ["rx"]}
    )
    assert (iqm.provider, iqm.name, iqm.n_qubits) == ("iqm", "garnet", 20)
    assert iqm.basis_gates == ("rx",)

    quantinuum = qpu.quantinuum_backend_profile({"machine": "H2-1", "n_qubits": 56})
    assert (quantinuum.name, quantinuum.n_qubits) == ("H2-1", 56)

    azure = qpu.azure_backend_profile({"id": "ionq.simulator"}, n_qubits=3)
    assert (azure.name, azure.n_qubits) == ("ionq.simulator", 3)
    assert azure.is_simulator is True


def test_no_photonic_or_control_system_adapter_is_exported() -> None:
    """The two absent provider families are absent from the exported surface.

    The scan is over names rather than over prose, because the row's claim is
    about adapters it does not have, and a name is what an adapter would need to
    be reachable.
    """

    exported = " ".join(qpu.__all__).lower()
    for absent in ("photonic", "control_system", "qcs"):
        assert absent not in exported, absent

    package = ROOT / "flagquantum"
    occurrences = sum(
        path.read_text(errors="ignore").lower().count("photonic")
        for path in package.rglob("*.py")
    )
    assert occurrences == 0


def test_the_row_still_names_the_roster_and_cites_this_test() -> None:
    """The row's remaining distance is unchanged and its evidence is no longer empty."""

    row = _rows()["provider_roster_breadth"]
    assert row["status"] == "partial"
    assert row["priority"] == "now"
    assert row["dependency_class"] == "none"
    assert row["maturity_ref"] == "hardware_provider_roster"

    evidence = row["evidence"]
    assert evidence, "the row must not claim breadth without an evidence list"
    assert "flagquantum/remote/qpu" in evidence
    assert "tests/unit/test_parity_provider_roster.py" in evidence

    reason = row["reason"]
    for absent in ("photonic", "quantum-control-system", "analog-schedule"):
        assert absent in reason, absent
    assert "seven" in reason.lower()
