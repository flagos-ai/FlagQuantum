"""The hardware provider roster: declaration reading and offline preflight.

Every adapter in this file is exercised against a fake transport and a fake
vendor listing. No test contacts a provider, and a test that would reach the
network fails on the autouse guard below. What the tests prove is the part that
can be proven without hardware: that a vendor declaration is read fail-closed,
that connectivity is never invented, and that a submission is refused before a
job exists when the adapter's own policy cannot satisfy it.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest

import flagquantum as fq
import flagquantum.deployment as fqd
import flagquantum.remote as remote
from flagquantum.remote.qpu import http
from flagquantum.remote.qpu.contracts import (
    ProviderTaskHandle,
    preflight_submission,
)
from flagquantum.remote.qpu.declaration import (
    connectivity_edges,
    declared_count,
    declared_flag,
    declared_gate_names,
    declared_text,
    declared_value,
    declared_width,
    edge_pair,
)
from flagquantum.remote.qpu.ionq import IonQProvider, ionq_backend_profile
from flagquantum.remote.qpu.iqm import IQMProvider, iqm_backend_profile
from flagquantum.remote.qpu.neutral_atom import (
    NeutralAtomProvider,
    interaction_edges,
    neutral_atom_backend_profile,
    neutral_atom_sites,
)
from flagquantum.remote.qpu.quantinuum import (
    QuantinuumProvider,
    _language,
    quantinuum_backend_profile,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _refuse_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a provider test attempted a real network call")

    monkeypatch.setattr(http.request, "urlopen", _refuse)


class FakeTransport:
    """A queue-free transport that answers from a URL table and records calls."""

    def __init__(self, responses: Mapping[tuple[str, str], Mapping[str, Any]]) -> None:
        self.responses = dict(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]:
        self.calls.append(("POST", url, dict(payload)))
        return self.responses[("POST", url)]

    def get_json(
        self, url: str, headers: Mapping[str, str], timeout: float
    ) -> Mapping[str, Any]:
        self.calls.append(("GET", url, {}))
        return self.responses[("GET", url)]

    def post_form(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def put_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, Any]:
        self.calls.append(("PUT", url, dict(payload)))
        return self.responses[("PUT", url)]

    def urls(self) -> list[str]:
        return [f"{method} {url}" for method, url, _payload in self.calls]


def _package(profile: Any, *, shots: int = 100, qasm_version: float = 3.0) -> Any:
    circuit = fq.Circuit(profile.n_qubits).h(0).cx(0, 1)
    return fqd.create_deployment_package(
        circuit, backend=profile, shots=shots, qasm_version=qasm_version
    )


# --------------------------------------------------------------------------
# Declaration readers
# --------------------------------------------------------------------------


def test_declared_value_finds_the_first_spelling_and_refuses_absence() -> None:
    row = {"n_qubits": 5, "qubits": 8}

    assert declared_value(row, "n_qubits", "qubits") == 5
    assert declared_value(row, "qubits", "n_qubits") == 8
    assert declared_value(row, "missing", default=None) is None
    with pytest.raises(KeyError, match="exposes none of"):
        declared_value(row, "missing", "also_missing")


def test_declared_value_reads_an_object_and_rejects_a_null_fact() -> None:
    class Row:
        name = "H2-1"
        n_qubits = None

    assert declared_value(Row(), "name") == "H2-1"
    assert declared_value(Row(), "n_qubits", default="absent") == "absent"


def test_declared_text_refuses_a_missing_or_blank_fact() -> None:
    assert declared_text(
        {"name": "  IQM Garnet "}, "name", owner="IQM", fact="name"
    ) == ("IQM Garnet")
    with pytest.raises(
        ValueError, match=r"^IQM declaration does not expose a device name$"
    ):
        declared_text({}, "name", owner="IQM", fact="device name")
    with pytest.raises(ValueError, match="does not expose a device name"):
        declared_text({"name": "   "}, "name", owner="IQM", fact="device name")


def test_declared_count_refuses_a_boolean_a_string_and_a_non_positive_width() -> None:
    assert declared_count({"qubits": 5}, "qubits", owner="IQM", fact="count") == 5
    with pytest.raises(ValueError, match=r"^IQM count must be a positive integer$"):
        declared_count({"qubits": True}, "qubits", owner="IQM", fact="count")
    with pytest.raises(ValueError, match="must be a positive integer"):
        declared_count({"qubits": "5"}, "qubits", owner="IQM", fact="count")
    with pytest.raises(ValueError, match="must be a positive integer"):
        declared_count({"qubits": 0}, "qubits", owner="IQM", fact="count")


def test_declared_flag_defaults_only_when_the_listing_is_silent() -> None:
    assert declared_flag({}, "is_simulator", owner="IQM", fact="marker", default=True)
    assert (
        declared_flag(
            {"is_simulator": False},
            "is_simulator",
            owner="IQM",
            fact="marker",
            default=True,
        )
        is False
    )
    with pytest.raises(ValueError, match=r"^IQM marker must be a boolean$"):
        declared_flag(
            {"is_simulator": "yes"},
            "is_simulator",
            owner="IQM",
            fact="marker",
            default=False,
        )


def test_declared_gate_names_normalises_and_refuses_a_non_name() -> None:
    assert declared_gate_names(["PRX", "cz", "prx"], owner="IQM") == ("prx", "cz")
    assert declared_gate_names(None, owner="IQM") == ()
    with pytest.raises(ValueError, match="must be a sequence of gate names"):
        declared_gate_names("prx", owner="IQM")
    with pytest.raises(ValueError, match="entries must be non-empty gate names"):
        declared_gate_names(["prx", ""], owner="IQM")
    with pytest.raises(ValueError, match="entries must be non-empty gate names"):
        declared_gate_names(["prx", 3], owner="IQM")


def test_declared_width_reads_a_count_or_a_qubit_list() -> None:
    assert declared_width({"n_qubits": 5}, "n_qubits", owner="IQM") == (
        5,
        "declared_count",
    )
    assert declared_width({"qubits": [[0, 0], [0, 1]]}, "qubits", owner="IQM") == (
        2,
        "declared_qubit_list",
    )
    with pytest.raises(ValueError, match="must be a positive integer or a list"):
        declared_width({"qubits": True}, "qubits", owner="IQM")
    with pytest.raises(ValueError, match=r"^IQM qubit count must be a positive int"):
        declared_width({"qubits": 0}, "qubits", owner="IQM")
    with pytest.raises(ValueError, match="must be a positive integer"):
        declared_width({"qubits": []}, "qubits", owner="IQM")
    with pytest.raises(ValueError, match="entry 0 must be a qubit coordinate"):
        declared_width({"qubits": [0, 1]}, "qubits", owner="IQM")
    with pytest.raises(ValueError, match="does not expose a qubit count"):
        declared_width({}, "qubits", owner="IQM")


def test_edge_pair_refuses_a_malformed_coupling() -> None:
    assert edge_pair((3, 1), owner="IQM") == (1, 3)
    with pytest.raises(
        ValueError, match=r"^IQM coupling entry must be a pair of wires$"
    ):
        edge_pair((0, 1, 2), owner="IQM")
    with pytest.raises(ValueError, match="must be a pair of integers"):
        edge_pair((0, "1"), owner="IQM")
    with pytest.raises(ValueError, match="must not contain a negative wire"):
        edge_pair((-1, 0), owner="IQM")


def test_connectivity_edges_reads_both_listing_shapes_and_bounds_them() -> None:
    assert connectivity_edges([[0, 1], [1, 2]], owner="IQM", n_qubits=3) == (
        (0, 1),
        (1, 2),
    )
    assert connectivity_edges({0: [1, 2], 1: [2]}, owner="IQM", n_qubits=3) == (
        (0, 1),
        (0, 2),
        (1, 2),
    )
    # A self-loop is a vendor typo, not a coupling, and is dropped rather than
    # reported as connectivity the device does not have.
    assert connectivity_edges([[0, 0], [0, 1]], owner="IQM", n_qubits=2) == ((0, 1),)
    with pytest.raises(ValueError, match="is outside a 2-wire device"):
        connectivity_edges([[0, 2]], owner="IQM", n_qubits=2)
    with pytest.raises(ValueError, match="must be a mapping or pairs"):
        connectivity_edges(5, owner="IQM", n_qubits=2)
    with pytest.raises(ValueError, match="targets must be a sequence"):
        connectivity_edges({0: 1}, owner="IQM", n_qubits=2)


# --------------------------------------------------------------------------
# Shared preflight policy
# --------------------------------------------------------------------------


def test_preflight_reports_each_policy_failure_the_adapter_owns() -> None:
    profile = iqm_backend_profile({"name": "IQM Garnet", "n_qubits": 2})
    package = _package(profile)
    preview = preflight_submission(
        package, provider="iqm", backend_name="IQM Star", qasm_versions=(2.0,)
    )

    assert preview.compatible is False
    assert preview.blockers == (
        "deployment_package_targets_a_different_backend",
        "iqm_provider_requires_openqasm_2",
    )
    assert preview.summary()["compatible"] is False


def test_preflight_declares_no_qasm_policy_when_none_is_implemented() -> None:
    profile = neutral_atom_backend_profile(
        {"name": "aquila", "sites": [[0.0, 0.0], [4.0, 0.0]]}, interaction_radius=5.0
    )
    preview = preflight_submission(
        _package(profile, shots=10), provider="neutral-atom", qasm_versions=None
    )

    assert preview.blockers == ()
    assert preview.compatible is True


def test_preflight_reports_a_package_that_fails_its_own_validator() -> None:
    """A package whose identity was altered is refused by the deployment
    validator, and that refusal is reported as a blocker like any other rather
    than raising out of the preflight."""

    profile = iqm_backend_profile({"name": "IQM Garnet", "n_qubits": 2})
    preview = preflight_submission(
        replace(_package(profile), metadata={}), provider="iqm"
    )

    assert preview.compatible is False
    assert preview.blockers == (
        "invalid_deployment_package:unsupported deployment package schema",
    )


def test_preflight_records_a_provider_mismatch_and_a_dynamic_circuit() -> None:
    profile = iqm_backend_profile({"name": "IQM Garnet", "n_qubits": 2})
    dynamic = fqd.CloudBackendProfile(
        provider=profile.provider,
        name=profile.name,
        n_qubits=profile.n_qubits,
        supports_dynamic_circuits=True,
    )
    preview = preflight_submission(
        _package(dynamic, shots=10), provider="iqm", dynamic_circuits_supported=False
    )
    assert preview.blockers == ("iqm_dynamic_circuits_are_not_supported",)

    other = preflight_submission(_package(profile, shots=10), provider="ionq")
    assert other.blockers == ("backend_provider_is_not_ionq",)


# --------------------------------------------------------------------------
# IonQ
# --------------------------------------------------------------------------


def test_ionq_profile_records_the_width_and_gate_set_it_was_given() -> None:
    profile = ionq_backend_profile(
        {"backend": "qpu.aria-1", "qubits": 25, "gates": ["X", "H", "CX"]},
        gateset="qis",
    )

    assert profile.provider == "ionq"
    assert profile.name == "qpu.aria-1"
    assert profile.n_qubits == 25
    assert profile.basis_gates == ("x", "h", "cx")
    assert profile.is_simulator is False
    assert profile.metadata["gateset"] == "qis"
    assert profile.metadata["simulator_source"] == "backend_name"


def test_ionq_does_not_fabricate_all_to_all_connectivity() -> None:
    """A trapped-ion device is all-to-all, but a listing that states no
    connectivity has not stated it, so the profile reports none and the
    compiler is left without a lattice to route against instead of being handed
    an invented complete graph."""

    profile = ionq_backend_profile({"backend": "qpu.forte-1", "qubits": 36})

    assert profile.coupling_map is None
    assert profile.metadata["connectivity_declared"] is False


def test_ionq_reads_a_declared_connectivity_when_the_listing_carries_one() -> None:
    profile = ionq_backend_profile(
        {
            "backend": "qpu.forte-enterprise",
            "qubits": 3,
            "connectivity": [[0, 1], [1, 2]],
        }
    )

    assert profile.coupling_map is not None
    assert profile.coupling_map.edges == ((0, 1), (1, 2))
    assert profile.metadata["connectivity_declared"] is True


def test_ionq_marks_a_simulation_backend_from_its_own_marking() -> None:
    profile = ionq_backend_profile(
        {"backend": "simulator", "qubits": 29, "is_simulator": True}
    )
    inferred = ionq_backend_profile({"backend": "simulator", "qubits": 29})

    assert profile.is_simulator is True
    assert profile.metadata["simulator_source"] == "declared"
    assert inferred.is_simulator is True
    assert inferred.metadata["simulator_source"] == "backend_name"


def test_ionq_refuses_an_unknown_gate_set_and_a_missing_width() -> None:
    with pytest.raises(ValueError, match="gateset must be one of qis, native"):
        ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25}, gateset="ibm")
    with pytest.raises(ValueError, match=r"^IonQ declaration does not expose a qubit"):
        ionq_backend_profile({"backend": "qpu.aria-1"})
    with pytest.raises(ValueError, match="gateset must be one of qis, native"):
        IonQProvider(gateset="ibm")


def test_ionq_dry_run_validates_locally_and_touches_no_transport() -> None:
    transport = FakeTransport({})
    provider = IonQProvider(transport=transport)
    profile = ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25})

    preview = provider.dry_run(_package(profile))

    assert preview.compatible is True
    assert preview.program_format == "openqasm-3"
    assert preview.program == _package(profile).qasm
    assert preview.summary()["provider"] == "ionq"
    assert transport.calls == []


def test_ionq_submits_its_declared_gate_set_and_decodes_probabilities() -> None:
    base = "https://api.ionq.co/v0.3"
    transport = FakeTransport(
        {
            ("POST", f"{base}/jobs"): {"id": "job-7"},
            ("GET", f"{base}/jobs/job-7"): {"status": "completed"},
            ("GET", f"{base}/jobs/job-7/results"): {
                "probabilities": {"00": 0.6, "11": 0.4}
            },
        }
    )
    provider = IonQProvider(transport=transport)
    profile = ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25})
    package = _package(profile, shots=100)

    result = provider.run(package)

    assert result.counts == {"00": 60, "11": 40}
    assert result.shots == 100
    assert result.handle.task_id == "job-7"
    assert transport.urls() == [
        f"POST {base}/jobs",
        f"GET {base}/jobs/job-7",
        f"GET {base}/jobs/job-7/results",
    ]
    submitted = transport.calls[0][2]
    assert submitted["target"] == "qpu.aria-1"
    assert submitted["shots"] == 100
    assert submitted["gateset"] == "qis"
    assert submitted["input"] == {"format": "qasm3", "value": package.qasm}


def test_ionq_maps_a_qasm_two_package_onto_the_vendor_qasm_two_format() -> None:
    provider = IonQProvider(transport=FakeTransport({}))
    profile = ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25})

    assert provider.dry_run(_package(profile, qasm_version=2.0)).program_format == (
        "openqasm-2"
    )
    assert (
        provider._submit_payload(_package(profile, qasm_version=2.0))["input"]["format"]
        == "qasm2"
    )


def test_ionq_refuses_a_shot_budget_it_was_told_about() -> None:
    provider = IonQProvider(transport=FakeTransport({}), shots_limit=10)
    profile = ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25})

    preview = provider.dry_run(_package(profile, shots=100))

    assert preview.compatible is False
    assert preview.blockers == ("shots_exceed_the_declared_ionq_shot_budget",)


def test_ionq_maps_vendor_status_and_refuses_an_unfinished_job() -> None:
    base = "https://api.ionq.co/v0.3"
    transport = FakeTransport(
        {
            ("POST", f"{base}/jobs"): {"id": "job-9"},
            ("GET", f"{base}/jobs/job-9"): {"status": "running"},
        }
    )
    provider = IonQProvider(transport=transport)
    profile = ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25})

    with pytest.raises(RuntimeError, match="ended with status 'Running'"):
        provider.run(_package(profile))


def test_ionq_refuses_a_malformed_result_outcome() -> None:
    base = "https://api.ionq.co/v0.3"
    transport = FakeTransport(
        {
            ("POST", f"{base}/jobs"): {"id": "job-8"},
            ("GET", f"{base}/jobs/job-8"): {"status": "completed"},
            ("GET", f"{base}/jobs/job-8/results"): {"probabilities": {"0x": 1.0}},
        }
    )
    provider = IonQProvider(transport=transport)
    profile = ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25})

    with pytest.raises(RuntimeError, match="invalid outcome"):
        provider.run(_package(profile))


def test_ionq_lists_devices_and_filters_by_requested_width() -> None:
    transport = FakeTransport(
        {
            ("GET", "https://api.ionq.co/v0.3/backends"): {
                "backends": [
                    {"backend": "qpu.aria-1", "qubits": 25},
                    {"backend": "simulator", "qubits": 29},
                ]
            }
        }
    )
    provider = IonQProvider(transport=transport)

    everything = provider.list_devices()
    narrow = provider.list_devices(n_qubits=26)

    assert [profile.name for profile in everything] == ["qpu.aria-1", "simulator"]
    assert [profile.name for profile in narrow] == ["simulator"]


def test_ionq_refuses_a_listing_that_is_not_a_list_of_rows() -> None:
    transport = FakeTransport(
        {("GET", "https://api.ionq.co/v0.3/backends"): {"backends": {"a": 1}}}
    )
    with pytest.raises(ValueError, match="is a mapping, not a list of rows"):
        IonQProvider(transport=transport).list_devices()


def test_ionq_refuses_to_submit_a_package_its_own_policy_rejects() -> None:
    """The dry run is not advisory: submission runs the same policy and refuses
    before any job is created."""

    provider = IonQProvider(transport=FakeTransport({}), shots_limit=10)
    profile = ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25})

    with pytest.raises(RuntimeError, match=r"^IonQ preflight failed: shots_exceed"):
        provider.submit(_package(profile, shots=100))


def test_ionq_refuses_a_submit_response_without_a_job_id() -> None:
    transport = FakeTransport(
        {("POST", "https://api.ionq.co/v0.3/jobs"): {"status": "ready"}}
    )
    provider = IonQProvider(transport=transport)
    profile = ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25})

    with pytest.raises(RuntimeError, match="does not contain a job id"):
        provider.submit(_package(profile))


def test_ionq_refuses_to_decode_a_result_without_a_shot_count() -> None:
    """A job handle that does not carry the submitted shot count cannot be read:
    probabilities are meaningless without the number of shots they describe."""

    transport = FakeTransport(
        {
            ("GET", "https://api.ionq.co/v0.3/jobs/job-1/results"): {
                "probabilities": {"00": 1.0}
            }
        }
    )
    handle = ProviderTaskHandle(
        provider="ionq", task_id="job-1", backend_name="qpu.aria-1"
    )

    with pytest.raises(RuntimeError, match="without its shot count"):
        IonQProvider(transport=transport).fetch_result(handle)


def test_ionq_refuses_a_payload_it_cannot_read_as_outcomes() -> None:
    """A payload whose weights are not numbers, and one that carries no outcomes
    at all, are both refused rather than read as an empty result."""

    base = "https://api.ionq.co/v0.3"
    profile = ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25})

    def _run(payload: dict[str, Any]) -> None:
        transport = FakeTransport(
            {
                ("POST", f"{base}/jobs"): {"id": "job-2"},
                ("GET", f"{base}/jobs/job-2"): {"status": "completed"},
                ("GET", f"{base}/jobs/job-2/results"): payload,
            }
        )
        IonQProvider(transport=transport).run(_package(profile, shots=10))

    with pytest.raises(RuntimeError, match="does not contain outcomes"):
        _run({"probabilities": {"00": "0.6"}})
    with pytest.raises(RuntimeError, match="does not contain outcomes"):
        _run({"probabilities": {}})


def test_ionq_refuses_a_probability_distribution_it_cannot_allocate() -> None:
    """Allocation is a largest-remainder split of the reported shots, so a
    distribution it can only round to a different total, or one that is not a
    distribution at all, is refused rather than rounded."""

    base = "https://api.ionq.co/v0.3"
    profile = ionq_backend_profile({"backend": "qpu.aria-1", "qubits": 25})

    def _counts(payload: dict[str, Any], shots: int) -> dict[str, int]:
        transport = FakeTransport(
            {
                ("POST", f"{base}/jobs"): {"id": "job-3"},
                ("GET", f"{base}/jobs/job-3"): {"status": "completed"},
                ("GET", f"{base}/jobs/job-3/results"): payload,
            }
        )
        return (
            IonQProvider(transport=transport).run(_package(profile, shots=shots)).counts
        )

    with pytest.raises(RuntimeError, match="invalid probability"):
        _counts({"probabilities": {"00": -0.5, "11": 1.5}}, 10)
    with pytest.raises(RuntimeError, match="do not sum to one"):
        _counts({"probabilities": {"00": 0.6, "11": 0.6}}, 10)
    assert _counts({"probabilities": {"00": 0.5, "11": 0.5}}, 3) == {"00": 2, "11": 1}


# --------------------------------------------------------------------------
# Quantinuum
# --------------------------------------------------------------------------


def test_quantinuum_reads_a_row_and_an_api_version_that_lists_names_only() -> None:
    row = quantinuum_backend_profile({"name": "H2-1", "n_qubits": 32})
    name_only = quantinuum_backend_profile("H2-1E", n_qubits=32)

    assert (row.n_qubits, row.is_simulator) == (32, False)
    assert row.metadata["qubit_count_source"] == "declaration"
    assert name_only.name == "H2-1E"
    assert name_only.n_qubits == 32
    assert name_only.is_simulator is True
    assert name_only.metadata["simulator_source"] == "machine_name"
    assert name_only.metadata["qubit_count_source"] == "caller"


def test_quantinuum_refuses_a_width_it_cannot_know() -> None:
    with pytest.raises(
        ValueError, match="does not expose a qubit count; pass n_qubits"
    ):
        quantinuum_backend_profile({"name": "H2-1"})


def test_quantinuum_declares_no_connectivity_for_a_shared_trap() -> None:
    profile = quantinuum_backend_profile({"name": "H2-1", "n_qubits": 32})

    assert profile.coupling_map is None
    assert profile.metadata["connectivity_declared"] is False


def test_quantinuum_accepts_a_caller_simulator_decision_and_a_gate_set() -> None:
    profile = quantinuum_backend_profile(
        {"name": "H2-1", "n_qubits": 32, "is_simulator": "unset"},
        is_simulator=True,
        basis_gates=("RZZ", "RZ", "RX"),
    )

    assert profile.is_simulator is True
    assert profile.metadata["simulator_source"] == "caller"
    assert profile.basis_gates == ("rzz", "rz", "rx")
    with pytest.raises(ValueError, match="is_simulator must be a boolean"):
        quantinuum_backend_profile({"name": "H2-1", "n_qubits": 32}, is_simulator=1)


def test_quantinuum_submits_the_language_it_has_a_mapping_for() -> None:
    base = "https://api.quantinuum.com/v1"
    transport = FakeTransport(
        {
            ("POST", f"{base}/job"): {"job": "j1"},
            ("GET", f"{base}/job/j1"): {"status": "completed"},
            ("GET", f"{base}/job/j1/results"): {"results": [{"c": ["00", "11", "00"]}]},
        }
    )
    provider = QuantinuumProvider(transport=transport, machine="H2-1")
    profile = quantinuum_backend_profile({"name": "H2-1", "n_qubits": 32})
    package = _package(profile, shots=3, qasm_version=2.0)

    result = provider.run(package)

    assert result.counts == {"00": 2, "11": 1}
    assert result.shots == 3
    assert transport.calls[0][2] == {
        "name": package.name,
        "machine": "H2-1",
        "language": "QASM",
        "program": package.qasm,
        "count": 3,
    }


def test_quantinuum_refuses_a_qasm_version_it_cannot_submit() -> None:
    provider = QuantinuumProvider(transport=FakeTransport({}), machine="H2-1")
    profile = quantinuum_backend_profile({"name": "H2-1", "n_qubits": 32})

    preview = provider.dry_run(_package(profile, qasm_version=3.0))

    assert preview.compatible is False
    assert preview.blockers == ("quantinuum_provider_requires_openqasm_2",)


def test_quantinuum_refuses_a_package_for_another_machine() -> None:
    provider = QuantinuumProvider(transport=FakeTransport({}), machine="H2-2")
    profile = quantinuum_backend_profile({"name": "H2-1", "n_qubits": 32})

    package = _package(profile, qasm_version=2.0)
    assert provider.dry_run(package).blockers == (
        "deployment_package_targets_a_different_backend",
    )


def test_quantinuum_tallies_an_aggregated_result_without_reversing_it() -> None:
    base = "https://api.quantinuum.com/v1"
    transport = FakeTransport(
        {
            ("POST", f"{base}/job"): {"job": "j2"},
            ("GET", f"{base}/job/j2"): {"status": "completed"},
            ("GET", f"{base}/job/j2/results"): {"results": [{"c": {"01": 4, "10": 1}}]},
        }
    )
    provider = QuantinuumProvider(transport=transport, machine="H2-1")
    profile = quantinuum_backend_profile({"name": "H2-1", "n_qubits": 32})

    result = provider.run(_package(profile, shots=5, qasm_version=2.0))

    assert result.counts == {"01": 4, "10": 1}
    assert result.metadata["bitstring_order"] == "as_returned"


def test_quantinuum_refuses_a_qasm_version_it_has_no_language_for() -> None:
    """Preflight refuses an unsupported version first, so this refusal is the
    second line of defence and is exercised directly."""

    assert _language(2.0) == "QASM"
    with pytest.raises(ValueError, match="does not accept QASM 3"):
        _language(3.0)


def test_quantinuum_refuses_to_submit_a_package_for_another_machine() -> None:
    provider = QuantinuumProvider(transport=FakeTransport({}), machine="H2-2")
    profile = quantinuum_backend_profile({"name": "H2-1", "n_qubits": 32})

    with pytest.raises(RuntimeError, match=r"^Quantinuum preflight failed: "):
        provider.submit(_package(profile, qasm_version=2.0))


class _Register:
    """A vendor register object that is not a mapping."""

    def __init__(self, value: Any) -> None:
        self.c = value


def _quantinuum_result(payload: Any, *, shots: int) -> None:
    base = "https://api.quantinuum.com/v1"
    transport = FakeTransport(
        {
            ("POST", f"{base}/job"): {"job": "j4"},
            ("GET", f"{base}/job/j4"): {"status": "completed"},
            ("GET", f"{base}/job/j4/results"): {"results": payload},
        }
    )
    provider = QuantinuumProvider(transport=transport, machine="H2-1")
    profile = quantinuum_backend_profile({"name": "H2-1", "n_qubits": 32})
    provider.run(_package(profile, shots=shots, qasm_version=2.0))


def test_quantinuum_refuses_a_result_entry_that_is_not_a_register_mapping() -> None:
    """A register is read from a mapping or from an object that exposes one; an
    object that is neither is refused rather than guessed at."""

    with pytest.raises(RuntimeError, match="must be a classical register"):
        _quantinuum_result([_Register(["00", "11"])], shots=2)


def test_quantinuum_refuses_a_result_that_mixes_register_shapes() -> None:
    """One response cannot carry both one entry per shot and one aggregate per
    register; merging the two would invent a shot count."""

    with pytest.raises(RuntimeError, match="mixes per-shot and aggregated"):
        _quantinuum_result([{"c": {"00": 1}}, {"c": ["00"]}], shots=2)


def test_quantinuum_refuses_a_register_that_is_not_a_bitstring_list() -> None:
    with pytest.raises(RuntimeError, match="must hold bitstrings"):
        _quantinuum_result([{"c": "00"}], shots=2)


def test_quantinuum_refuses_an_outcome_that_is_not_a_bitstring() -> None:
    with pytest.raises(RuntimeError, match="invalid outcome"):
        _quantinuum_result([{"c": ["0x"]}], shots=1)


def test_quantinuum_refuses_a_result_entry_without_a_register() -> None:
    base = "https://api.quantinuum.com/v1"
    transport = FakeTransport(
        {
            ("POST", f"{base}/job"): {"job": "j3"},
            ("GET", f"{base}/job/j3"): {"status": "completed"},
            ("GET", f"{base}/job/j3/results"): {"results": [{"unexpected": "00"}]},
        }
    )
    provider = QuantinuumProvider(transport=transport, machine="H2-1")
    profile = quantinuum_backend_profile({"name": "H2-1", "n_qubits": 32})

    with pytest.raises(RuntimeError, match="does not contain a register"):
        provider.run(_package(profile, shots=2, qasm_version=2.0))


# --------------------------------------------------------------------------
# IQM
# --------------------------------------------------------------------------


def test_iqm_reads_its_lattice_and_counts_its_own_qubit_list() -> None:
    profile = iqm_backend_profile(
        {
            "name": "IQM Garnet",
            "qubits": [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            "connectivity": [[0, 1], [1, 2]],
            "gates": ["PRX", "CZ"],
        }
    )

    assert profile.provider == "iqm"
    assert profile.n_qubits == 3
    assert profile.metadata["qubit_count_source"] == "declared_qubit_list"
    assert profile.basis_gates == ("prx", "cz")
    assert profile.coupling_map is not None
    assert profile.coupling_map.edges == ((0, 1), (1, 2))
    assert profile.metadata["declared_coupling_count"] == 2


def test_iqm_reports_an_absent_lattice_rather_than_inventing_one() -> None:
    profile = iqm_backend_profile({"name": "IQM Star", "n_qubits": 5})

    assert profile.coupling_map is None
    assert profile.metadata["connectivity_declared"] is False
    assert profile.metadata["qubit_count_source"] == "declared_count"
    assert profile.metadata["declared_coupling_count"] == 0


def test_iqm_refuses_a_coupling_outside_the_device() -> None:
    with pytest.raises(ValueError, match="is outside a 2-wire device"):
        iqm_backend_profile(
            {"name": "IQM Garnet", "qubits": [[0, 0], [0, 1]], "connectivity": [[0, 2]]}
        )


def test_iqm_reads_its_own_simulator_marking() -> None:
    typed = iqm_backend_profile({"name": "qsim", "n_qubits": 5, "type": "simulator"})
    named = iqm_backend_profile({"name": "IQM-Simulator", "n_qubits": 5})
    spelled = iqm_backend_profile(
        {"name": "IQM Garnet", "n_qubits": 5, "simulator": True}
    )

    assert typed.is_simulator is True
    assert typed.metadata["simulator_source"] == "device_type"
    assert named.is_simulator is True
    assert named.metadata["simulator_source"] == "device_name"
    assert spelled.is_simulator is True
    assert spelled.metadata["simulator_source"] == "declared"


def test_iqm_prefers_a_caller_gate_set_over_the_listing() -> None:
    """A caller who states the native gate set knows the device revision, so the
    declared list is overridden rather than merged."""

    profile = iqm_backend_profile(
        {"name": "IQM Garnet", "n_qubits": 5, "gates": ["PRX"]},
        basis_gates=("CZ", "PRX"),
    )

    assert profile.basis_gates == ("cz", "prx")


def test_iqm_submits_openqasm_three_and_reads_integer_counts() -> None:
    base = "https://resonance.iqm.tech/v1"
    transport = FakeTransport(
        {
            ("POST", f"{base}/jobs"): {"id": "j2"},
            ("GET", f"{base}/jobs/j2"): {"status": "completed"},
            ("GET", f"{base}/jobs/j2/results"): {"counts": {"00": 3, "11": 2}},
        }
    )
    provider = IQMProvider(transport=transport)
    profile = iqm_backend_profile(
        {"name": "IQM Garnet", "qubits": [[0, 0], [0, 1]], "connectivity": [[0, 1]]}
    )
    package = _package(profile, shots=5)

    result = provider.run(package)

    assert result.counts == {"00": 3, "11": 2}
    assert transport.calls[0][2] == {
        "name": package.name,
        "device": "IQM Garnet",
        "shots": 5,
        "qasm": package.qasm,
    }


def test_iqm_refuses_a_qasm_two_package() -> None:
    provider = IQMProvider(transport=FakeTransport({}))
    profile = iqm_backend_profile({"name": "IQM Garnet", "n_qubits": 5})

    assert provider.dry_run(_package(profile, qasm_version=2.0)).blockers == (
        "iqm_provider_requires_openqasm_3",
    )


def test_iqm_reads_a_per_shot_result_and_refuses_a_malformed_one() -> None:
    base = "https://resonance.iqm.tech/v1"
    transport = FakeTransport(
        {
            ("POST", f"{base}/jobs"): {"id": "j4"},
            ("GET", f"{base}/jobs/j4"): {"status": "completed"},
            ("GET", f"{base}/jobs/j4/results"): {"measurements": ["01", "01", "10"]},
        }
    )
    provider = IQMProvider(transport=transport)
    profile = iqm_backend_profile({"name": "IQM Garnet", "n_qubits": 5})

    assert provider.run(_package(profile, shots=3)).counts == {"01": 2, "10": 1}

    broken = FakeTransport(
        {
            ("POST", f"{base}/jobs"): {"id": "j5"},
            ("GET", f"{base}/jobs/j5"): {"status": "completed"},
            ("GET", f"{base}/jobs/j5/results"): {"measurements": [[0, 2]]},
        }
    )
    with pytest.raises(RuntimeError, match="invalid outcome"):
        IQMProvider(transport=broken).run(_package(profile, shots=1))


def test_iqm_refuses_an_unfinished_job_a_shot_budget_and_a_missing_shot_count() -> None:
    """The status mapping decides whether a result exists at all, the shot
    budget is the adapter's own policy, and a handle without a shot count
    cannot be decoded."""

    base = "https://resonance.iqm.tech/v1"
    running = FakeTransport(
        {
            ("POST", f"{base}/jobs"): {"id": "j6"},
            ("GET", f"{base}/jobs/j6"): {"status": "executing"},
        }
    )
    profile = iqm_backend_profile({"name": "IQM Garnet", "n_qubits": 5})

    with pytest.raises(RuntimeError, match="ended with status 'Running'"):
        IQMProvider(transport=running).run(_package(profile))

    budgeted = IQMProvider(transport=FakeTransport({}), shots_limit=2)
    assert budgeted.dry_run(_package(profile, shots=5)).blockers == (
        "shots_exceed_the_declared_iqm_shot_budget",
    )
    with pytest.raises(RuntimeError, match=r"^IQM preflight failed: shots_exceed"):
        budgeted.submit(_package(profile, shots=5))

    results = FakeTransport({("GET", f"{base}/jobs/j7/results"): {"counts": {"00": 1}}})
    handle = ProviderTaskHandle(provider="iqm", task_id="j7", backend_name="IQM Garnet")
    with pytest.raises(RuntimeError, match="without its shot count"):
        IQMProvider(transport=results).fetch_result(handle)


def test_iqm_refuses_a_result_payload_with_no_outcomes() -> None:
    base = "https://resonance.iqm.tech/v1"
    transport = FakeTransport(
        {
            ("POST", f"{base}/jobs"): {"id": "j8"},
            ("GET", f"{base}/jobs/j8"): {"status": "completed"},
            ("GET", f"{base}/jobs/j8/results"): {"measurements": []},
        }
    )
    profile = iqm_backend_profile({"name": "IQM Garnet", "n_qubits": 5})

    with pytest.raises(RuntimeError, match="does not contain outcomes"):
        IQMProvider(transport=transport).run(_package(profile, shots=1))


def test_iqm_refuses_a_per_shot_result_that_disagrees_with_the_shot_count() -> None:
    """The tally is returned as-is only when it accounts for every shot the job
    was submitted with."""

    base = "https://resonance.iqm.tech/v1"
    transport = FakeTransport(
        {
            ("POST", f"{base}/jobs"): {"id": "j9"},
            ("GET", f"{base}/jobs/j9"): {"status": "completed"},
            ("GET", f"{base}/jobs/j9/results"): {"measurements": ["01", "01", "10"]},
        }
    )
    profile = iqm_backend_profile({"name": "IQM Garnet", "n_qubits": 5})

    with pytest.raises(RuntimeError, match="returned 3 outcomes for a 2-shot job"):
        IQMProvider(transport=transport).run(_package(profile, shots=2))


# --------------------------------------------------------------------------
# Neutral atom
# --------------------------------------------------------------------------


def test_neutral_atom_reads_the_register_in_every_declared_shape() -> None:
    assert neutral_atom_sites(
        {"sites": [{"x": 0.0, "y": 0.0}, {"position": [4.0, 0.0]}]}
    ) == ((0.0, 0.0), (4.0, 0.0))
    assert neutral_atom_sites({"atoms": [[0, 0], [0, 1]]}) == ((0.0, 0.0), (0.0, 1.0))
    assert neutral_atom_sites({"register": {"a": [0.0, 0.0], "b": [4.0, 0.0]}}) == (
        (0.0, 0.0),
        (4.0, 0.0),
    )


def test_neutral_atom_refuses_a_register_it_cannot_place() -> None:
    with pytest.raises(ValueError, match="does not expose a register"):
        neutral_atom_sites({"name": "aquila"})
    with pytest.raises(ValueError, match="must declare x and y coordinates"):
        neutral_atom_sites({"sites": [{"x": 0.0}]})
    with pytest.raises(ValueError, match="coordinates must be numbers"):
        neutral_atom_sites({"sites": [[0.0, "1.0"]]})
    with pytest.raises(ValueError, match="coordinates must be finite"):
        neutral_atom_sites({"sites": [[0.0, float("inf")]]})
    with pytest.raises(ValueError, match="two atoms at the same coordinate"):
        neutral_atom_sites({"sites": [[0.0, 0.0], [0.0, 0.0]]})
    with pytest.raises(ValueError, match="does not declare any sites"):
        neutral_atom_sites({"sites": []})


def test_neutral_atom_requires_an_explicit_interaction_radius() -> None:
    """The radius is a property of the atom species and the excitation scheme,
    not of this package, so it has no default and a non-physical value is
    refused rather than read as "no interactions"."""

    signature = inspect.signature(neutral_atom_backend_profile)
    assert signature.parameters["interaction_radius"].default is inspect.Parameter.empty

    sites = ((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError, match="must be positive and finite"):
        interaction_edges(sites, interaction_radius=0.0)
    with pytest.raises(ValueError, match="must be positive and finite"):
        interaction_edges(sites, interaction_radius=float("nan"))
    with pytest.raises(ValueError, match="must be a number"):
        interaction_edges(sites, interaction_radius="1.0")
    # A boolean is an integer in Python, but it is not a distance.
    with pytest.raises(ValueError, match="must be a number"):
        interaction_edges(sites, interaction_radius=True)


def test_neutral_atom_refuses_a_register_it_cannot_read_as_sites() -> None:
    """A register is a sequence of sites, and a site is a coordinate pair; a
    string is neither, however much it looks like one."""

    with pytest.raises(ValueError, match="must be a sequence of sites"):
        neutral_atom_sites({"sites": "01"})
    with pytest.raises(ValueError, match="must declare x and y coordinates"):
        neutral_atom_sites({"sites": ["01"]})


def test_neutral_atom_derives_connectivity_from_the_declared_geometry() -> None:
    sites = ((0.0, 0.0), (4.0, 0.0), (8.0, 0.0), (4.0, 1.0))

    # The radius is inclusive, so a pair exactly at the radius interacts.
    assert interaction_edges(sites, interaction_radius=4.0) == (
        (0, 1),
        (1, 2),
        (1, 3),
    )
    assert interaction_edges(sites, interaction_radius=1.0) == ((1, 3),)


def test_neutral_atom_profile_reports_where_its_connectivity_came_from() -> None:
    profile = neutral_atom_backend_profile(
        {"name": "aquila", "atoms": [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]},
        interaction_radius=5.0,
        register_units="um",
    )

    assert profile.provider == "neutral-atom"
    assert profile.n_qubits == 3
    assert profile.coupling_map is not None
    assert profile.coupling_map.edges == ((0, 1), (1, 2))
    assert profile.metadata["connectivity_source"] == "derived_from_declared_geometry"
    assert profile.metadata["register_units"] == "um"
    assert profile.metadata["interaction_radius"] == 5.0
    # The adapter submits no OpenQASM to a neutral-atom machine, so it declares
    # no OpenQASM support rather than implying it can.
    assert profile.supports_openqasm is False


def test_neutral_atom_refuses_a_count_that_disagrees_with_the_register() -> None:
    with pytest.raises(ValueError, match="states 5 atoms but places 2 sites"):
        neutral_atom_backend_profile(
            {"name": "aquila", "n_atoms": 5, "atoms": [[0.0, 0.0], [4.0, 0.0]]},
            interaction_radius=5.0,
        )
    with pytest.raises(ValueError, match="register_units must be one of"):
        neutral_atom_backend_profile(
            {"name": "aquila", "atoms": [[0.0, 0.0], [4.0, 0.0]]},
            interaction_radius=5.0,
            register_units="furlongs",
        )


def test_neutral_atom_preflight_names_the_program_it_cannot_emit() -> None:
    register = {"name": "aquila", "atoms": [[0.0, 0.0], [4.0, 0.0]]}
    profile = neutral_atom_backend_profile(register, interaction_radius=5.0)
    provider = NeutralAtomProvider(register=register, interaction_radius=5.0)

    assert [item.name for item in provider.list_devices()] == ["aquila"]
    preview = provider.dry_run(_package(profile, shots=10))

    assert preview.compatible is False
    assert preview.program_format == "analog_register_schedule"
    assert preview.blockers == (
        "neutral_atom_submission_requires_an_analog_program_schedule",
    )
    with pytest.raises(RuntimeError, match="requires an analog program schedule"):
        provider.submit(_package(profile, shots=10))


def test_neutral_atom_applies_no_qasm_policy_to_a_program_it_cannot_emit() -> None:
    """The adapter emits an analog register schedule, not OpenQASM, so the
    package's QASM version is not a submission requirement here."""

    register = {"name": "aquila", "atoms": [[0.0, 0.0], [4.0, 0.0]]}
    profile = neutral_atom_backend_profile(register, interaction_radius=5.0)
    provider = NeutralAtomProvider(register=register, interaction_radius=5.0)

    preview = provider.dry_run(_package(profile, shots=10, qasm_version=2.0))

    assert preview.blockers == (
        "neutral_atom_submission_requires_an_analog_program_schedule",
    )


def test_neutral_atom_list_devices_honours_a_requested_width() -> None:
    register = {"name": "aquila", "atoms": [[0.0, 0.0], [4.0, 0.0]]}
    provider = NeutralAtomProvider(register=register, interaction_radius=5.0)

    assert provider.list_devices(n_qubits=2)[0].n_qubits == 2
    assert provider.list_devices(n_qubits=3) == ()


# --------------------------------------------------------------------------
# Roster surface
# --------------------------------------------------------------------------


def test_the_roster_adapters_are_reachable_from_the_remote_facade() -> None:
    providers = (
        "IonQProvider",
        "QuantinuumProvider",
        "IQMProvider",
        "NeutralAtomProvider",
    )
    profiles = (
        "ionq_backend_profile",
        "quantinuum_backend_profile",
        "iqm_backend_profile",
        "neutral_atom_backend_profile",
    )

    assert all(getattr(remote, name).provider for name in providers)
    assert all(callable(getattr(remote, name)) for name in profiles)
    assert remote.preflight_submission is preflight_submission
    assert remote.ProviderSubmissionPreview.__name__ == "ProviderSubmissionPreview"


def test_result_decoding_delegates_to_the_shared_boundary() -> None:
    """ARCH-012 replacement: Azure's probability allocation and Quantinuum's
    per-shot tally are the shared helpers rather than adapter-local copies, so
    the boundary was admitted by replacing implementations rather than by
    declaring an interface beside them."""

    from flagquantum.remote.qpu import azure, quantinuum, result_parsing

    assert azure._probability_counts is result_parsing._probability_counts
    assert quantinuum._tally_outcomes is result_parsing._tally_outcomes
    assert not hasattr(quantinuum, "_tally_bitstrings")


def test_no_new_adapter_claims_hardware_identical_behaviour() -> None:
    """Each vendor-specific text in these modules is a vendor's own word.

    The guard is deliberately coarse: the adapters may name a vendor, but none
    may claim its behaviour was verified on hardware, because no hardware was
    contacted while writing them.
    """

    claims = ("verified on hardware", "hardware-identical", "calibrated against")
    for module in (
        ionq_backend_profile,
        quantinuum_backend_profile,
        iqm_backend_profile,
    ):
        source = inspect.getsource(module)
        assert not any(claim in source for claim in claims)
