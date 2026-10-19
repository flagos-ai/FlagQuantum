"""Pin the provider roster the parity row claims, so the row cannot drift.

``provider_roster_breadth`` is a ``priority = "now"`` row with no ``evidence``
list at all, and its prose names seven adapters. The package declares eight: the
shared HTTP adapter is a concrete ``QuantumProvider`` with its own ``submit`` and
``fetch_result``, and ``QuafuProvider`` derives from it. That is the kind of
off-by-one a roster claim drifts into, so the census here is a set equality
rather than a count -- adding an adapter fails this module until the row is
updated, and so does removing one.

The refusals are the second half. A roster is only breadth if every member reads
a vendor's declaration fail-closed, so every profile builder is handed an empty
declaration and must refuse it in its own words. A builder that started
accepting an empty declaration would make the row's "read fail-closed" clause
false, and this is where that would show up.
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

import flagquantum as fq
import flagquantum.remote as remote
import flagquantum.remote.qpu as qpu
from flagquantum.deployment.cloud import DeploymentPackage
from flagquantum.remote.qpu import (
    interaction_edges,
    neutral_atom_backend_profile,
)

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
        "NeutralAtomProvider",
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
        "neutral_atom_backend_profile",
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
    """Eight concrete adapters, seven of which build a backend profile."""

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
            lambda: neutral_atom_backend_profile({}, interaction_radius=4.0),
            "Neutral atom declaration does not expose",
        ),
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


def test_the_neutral_atom_connectivity_is_derived_from_the_declared_geometry() -> None:
    """The register's coordinates are the connectivity, not a separate declaration.

    Two of the three sites sit inside the interaction radius and the third does
    not, so a profile that reported the register as fully connected, or as
    isolated, would disagree with the geometry it was given.
    """

    sites = ((0.0, 0.0), (4.0, 0.0), (20.0, 0.0))
    profile = neutral_atom_backend_profile(
        {"name": "aquila", "sites": [list(site) for site in sites]},
        interaction_radius=5.0,
    )
    assert profile.coupling_map is not None
    assert profile.coupling_map.edges == ((0, 1),)
    assert profile.metadata["connectivity_source"] == "derived_from_declared_geometry"
    assert profile.metadata["derived_coupling_count"] == 1
    assert interaction_edges(sites, interaction_radius=5.0) == ((0, 1),)

    # The radius is inclusive: a pair exactly at the radius interacts, so a
    # radius equal to the pair's own distance still yields the edge and a radius
    # a hair smaller does not. A strict comparison would answer "isolated" here.
    assert interaction_edges(sites, interaction_radius=4.0) == ((0, 1),)

    profile_at_boundary = neutral_atom_backend_profile(
        {"name": "aquila", "sites": [list(site) for site in sites]},
        interaction_radius=4.0,
    )
    assert profile_at_boundary.coupling_map is not None
    assert profile_at_boundary.coupling_map.edges == ((0, 1),)

    # A non-positive radius is refused rather than read as "no interactions",
    # which would report a connected register as isolated.
    with pytest.raises(ValueError, match="must be positive and finite"):
        interaction_edges(sites, interaction_radius=0.0)


def test_the_neutral_atom_adapter_declares_the_submission_it_cannot_emit() -> None:
    """The absent analog path is a declared refusal, not a silent gap.

    The refusal has to survive the run, not merely appear in the source: the
    blocker is read off ``dry_run``'s own preview and the format it names is
    read off the same preview, so dropping the blocker from the returned tuple
    is a different measurement from deleting the constant.
    """

    sites = [[0.0, 0.0], [4.0, 0.0]]
    provider = qpu.NeutralAtomProvider(
        register={"sites": sites}, interaction_radius=5.0
    )
    profile = neutral_atom_backend_profile(
        {"name": "aquila", "sites": sites}, interaction_radius=5.0
    )
    preview = provider.dry_run(
        DeploymentPackage(
            name="roster",
            ir=fq.Circuit(1).h(0).to_ir(),
            qasm="OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n",
            shots=100,
            qasm_version=3.0,
            backend=profile,
        )
    )
    assert preview.compatible is False
    assert preview.program_format == "analog_register_schedule"
    assert "neutral_atom_submission_requires_an_analog_program_schedule" in (
        preview.blockers
    )

    with pytest.raises(RuntimeError, match="analog program schedule"):
        provider.submit(object())  # type: ignore[arg-type]


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
    assert "eight" in reason.lower()
