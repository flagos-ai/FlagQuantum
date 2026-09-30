"""The admission contract stays true to the code that implements it."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from flagquantum.ecosystem import extensions as extension_namespace
from flagquantum.ecosystem.extensions import admission, sdk
from flagquantum.runtime import backend_registry

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "backend-execution-admission-v1-candidate.json"


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_the_contract_names_only_existing_authorities() -> None:
    contract = _contract()
    assert contract["status"] == "candidate"
    assert contract["public_api_change"] is False
    assert contract["stable_core_change"] is False
    assert contract["root_manifest_change"] is False
    assert contract["default_path_change"] is False
    assert contract["existing_authorities"] == {
        "extension_discovery_and_negotiation": (
            "flagquantum.ecosystem.extensions.ExtensionRegistry"
        ),
        "extension_entry_point_group": "flagquantum.extensions",
        "backend_capability_record": (
            "flagquantum.runtime.backend_registry.BackendCapabilities"
        ),
        "backend_capability_registry": (
            "flagquantum.runtime.backend_registry.register_backend"
        ),
        "planning": "flagquantum.runtime.planner.plan",
        "single_execution_attempt": "flagquantum.runtime.plan_execution.execute_plan",
        "distribution_semantics_vocabulary": "flagquantum.runtime.audit.vocabulary",
    }


def test_the_declaration_member_and_host_owned_keys_match_the_code() -> None:
    contract = _contract()
    assert contract["admission"]["declaration_member"] == admission.DECLARATION_MEMBER
    assert set(contract["declaration"]["host_owned_keys"]) == set(
        admission.HOST_OWNED_DECLARATION_KEYS
    )


def test_every_declared_capability_addition_exists_on_the_record() -> None:
    contract = _contract()
    fields = backend_registry.BackendCapabilities.__dataclass_fields__
    additions = contract["capability_record_additions"]

    assert set(additions) <= set(fields)
    for name, entry in additions.items():
        assert "meaning" in entry
        assert entry["serialized"] is (name != "executor")


def test_the_declared_gradient_key_is_the_one_the_planner_consults() -> None:
    contract = _contract()
    declaration = contract["declaration"]
    assert declaration["gradient_capability_key"] == "supports_autograd"
    assert (
        "supports_autograd" in backend_registry.BackendCapabilities.__dataclass_fields__
    )
    assert "supports_gradients" not in declaration


def test_the_execution_route_matches_the_executor_protocol() -> None:
    contract = _contract()
    route = contract["execution_route"]

    assert route["protocol"] == "flagquantum.runtime.backend_registry.BackendExecutor"
    execute = inspect.signature(backend_registry.BackendExecutor.execute)
    assert list(execute.parameters) == ["self", "program", "options"]
    assert execute.parameters["options"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "close" in backend_registry.BackendExecutor.__protocol_attrs__


def test_the_sdk_protocol_is_unchanged_and_separate_from_the_route() -> None:
    contract = _contract()
    assert contract["admission"]["sdk_namespace_change"] == "none"
    assert admission.DECLARATION_MEMBER not in extension_namespace.__all__
    assert not set(admission.__all__) & set(extension_namespace.__all__)

    parameters = inspect.signature(sdk.ExecutionBackendExtension.execute).parameters
    assert list(parameters) == ["self", "program", "parameters"]


def test_the_builtin_fast_path_claim_matches_the_registry() -> None:
    contract = _contract()
    builtin = contract["builtin_fast_path"]
    assert set(builtin["builtin_backends"]) == set(
        backend_registry.BUILTIN_BACKEND_NAMES
    )
    for name in builtin["builtin_backends"]:
        assert backend_registry.resolve_backend_executor(name) is None
    assert builtin["registry_access_on_the_builtin_path"] is False


def test_the_contract_declares_no_new_registry_or_fallback() -> None:
    contract = _contract()
    admission_block = contract["admission"]
    fail_closed = contract["fail_closed"]

    assert admission_block["second_registry_created"] is False
    assert admission_block["negotiation_per_execution"] is False
    assert admission_block["version_check_per_execution"] is False
    assert admission_block["admission_trigger"] == "explicit_call_only"
    assert "extension_scope" not in admission_block["admission_trigger"]
    assert fail_closed["execution_time_fallback"] is False
    assert fail_closed["backend_substitution"] is False
    assert fail_closed["silent_cpu_path"] is False
    assert fail_closed["admission_error_type"] == (
        "flagquantum.ecosystem.extensions.admission.BackendAdmissionError"
    )
    assert admission.BackendAdmissionError.__module__ == admission.__name__


def test_the_sdk_version_check_is_delegated_to_the_sdk_authority() -> None:
    contract = _contract()
    assert contract["admission"]["sdk_version_check"] == (
        "flagquantum.ecosystem.extensions.ExtensionRegistry.with_extension"
    )
    source = (ROOT / "flagquantum/ecosystem/extensions/admission.py").read_text(
        encoding="utf-8"
    )
    assert "ExtensionRegistry().with_extension(extension)" in source
    # The check must not be restated: one authority, one implementation.
    assert "SDK_API_VERSION" not in source


def test_every_declared_invariant_is_enforced_by_a_test() -> None:
    contract = _contract()
    invariants = set(contract["invariants"])

    assert (
        "scalability_claim_allowed is false unless distribution_semantics is sharded_across_ranks"
        in invariants
    )
    with pytest.raises(ValueError, match="sharded_across_ranks"):
        backend_registry.BackendCapabilities(
            name="invariant_probe",
            tensor_backend="torch",
            devices=("cpu",),
            dtypes=("complex64",),
            supports_autograd=True,
            supports_distributed=True,
            supports_statevector=True,
            supports_density_matrix=False,
            supports_mps=False,
            distribution_semantics="replicated_per_rank",
            scalability_claim_allowed=True,
        )
    assert "capability_summary never exposes executor" in invariants
    summary = backend_registry.capability_summary("pytorch")
    assert "executor" not in summary


def test_the_replacement_proof_chain_names_reachable_entry_points() -> None:
    contract = _contract()
    proof = contract["replacement_proof"]

    assert proof["consumer_left_unmodified"] == "fq.run"
    assert proof["second_implementation_required"] is True
    assert proof["scenario_test_required"] is True
    assert proof["chain"] == [
        "admit_backend_extension",
        "fq.ExecutionOptions(backend=<extension name>)",
        "fq.plan",
        "execute_plan",
        "ExecutionResult",
    ]
    assert callable(admission.admit_backend_extension)


def test_the_contract_keeps_its_stated_non_goals_out_of_scope() -> None:
    contract = _contract()
    non_goals = " ".join(contract["non_goals"]).lower()

    assert "c abi" in non_goals
    assert "second extension registry" in non_goals
    assert "per-execution" in non_goals
    assert "numeric" in non_goals
    assert not hasattr(backend_registry, "BackendArtifactFormats")
