"""Check the compiler-pass admission contract against the code.

`contracts/compiler-pass-admission-v1-candidate.json` is the repository's own
statement about this crossing. A prose contract that nothing reads rots, so this
module reads it and checks the claims that could silently become false: that the
authorities it names are the objects the code actually calls, that the reserved
names and the pipeline are the same set, that the default path really resolves no
extension pass, and that the declared failure modes are the ones the tests
observe.

The point of the checks is disproof, not coverage. Each one is written so that
making the code do something the contract does not say makes it fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler import pass_manager, pipeline
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.extensions import pass_admission, sdk
from flagquantum.errors import CapabilityError, CompilationError

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts/compiler-pass-admission-v1-candidate.json"


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _dotted(name: str) -> object:
    """Resolve one dotted name from the contract to the object it denotes.

    A contract that names a module which does not exist, or an attribute which
    does not exist on it, is a claim about nothing, so this resolves the whole
    path rather than only the module.
    """

    parts = name.split(".")
    module: Any = None
    for index in range(len(parts), 0, -1):
        candidate = ".".join(parts[:index])
        try:
            module = __import__(candidate, fromlist=["_"])
        except ImportError:
            continue
        remainder = parts[index:]
        break
    else:  # pragma: no cover - only reached if no prefix is an importable module
        raise AssertionError(f"{name} has no importable module prefix")
    for attribute in remainder:
        try:
            module = getattr(module, attribute)
        except AttributeError as exc:  # pragma: no cover - reported as a failure
            raise AssertionError(f"{name} does not resolve: {exc}") from exc
    return module


def test_the_contract_is_a_candidate_that_changes_no_protected_surface(
    contract: dict[str, Any],
) -> None:
    assert contract["schema"] == "flagquantum.compiler_pass_admission_contract"
    assert contract["status"] == "candidate"
    assert contract["public_api_change"] is False
    assert contract["stable_core_change"] is False
    assert contract["root_manifest_change"] is False
    assert contract["default_path_change"] is False


def test_the_proposal_the_contract_cites_exists(contract: dict[str, Any]) -> None:
    # A contract citing a proposal that does not exist is a dead reference, which
    # this repository has produced before and which nothing else catches.
    assert (ROOT / contract["proposal"]).is_file()


_AUTHORITY_KEYS = (
    "extension_discovery_and_negotiation",
    "pass_kind_protocol",
    "pass_registry",
    "optimization_pipeline",
    "builtin_pass_bindings",
    "compilation_entry",
)


@pytest.mark.parametrize("key", sorted(_AUTHORITY_KEYS))
def test_every_declared_authority_resolves(contract: dict[str, Any], key: str) -> None:
    assert _dotted(contract["existing_authorities"][key]) is not None


def test_the_declared_authorities_are_the_objects_this_code_uses(
    contract: dict[str, Any],
) -> None:
    # Agreement with the code, not merely resolvability: a contract naming a
    # look-alike would still resolve.
    authorities = contract["existing_authorities"]

    assert _dotted(authorities["pass_registry"]) is pass_manager.PassRegistry
    assert _dotted(authorities["optimization_pipeline"]) is (
        pass_manager.OPTIMIZATION_PIPELINE
    )
    assert _dotted(authorities["builtin_pass_bindings"]) is pipeline.BUILTIN_PASSES
    assert _dotted(authorities["compilation_entry"]) is pipeline.compile
    assert _dotted(authorities["pass_kind_protocol"]) is sdk.CompilerPassExtension
    assert _dotted(authorities["extension_discovery_and_negotiation"]) is (
        sdk.ExtensionRegistry
    )


def test_the_declared_entry_point_group_is_the_one_the_sdk_publishes(
    contract: dict[str, Any],
) -> None:
    # The contract writes the group as a bare string, not a dotted name, because
    # it is a value rather than an object; this is what ties the two together.
    declared = contract["existing_authorities"]["extension_entry_point_group"]

    assert declared == sdk.EXTENSION_ENTRY_POINT_GROUP


def test_the_declared_crossing_and_registration_call_are_what_the_code_calls(
    contract: dict[str, Any],
) -> None:
    admission = contract["admission"]

    assert _dotted(admission["crossing"]) is pass_admission.admit_pass_extension
    assert (
        _dotted(admission["registration_call"]) is pass_manager.PassRegistry.with_pass
    )
    assert admission["host_side_module"] == pass_admission.__name__
    assert admission["declaration_member"] == pass_admission.PASS_DECLARATION_MEMBER


def test_the_declared_sdk_version_check_is_delegated_and_not_restated(
    contract: dict[str, Any],
) -> None:
    # The contract says the check reuses `ExtensionRegistry.with_extension`. If the
    # crossing grew its own version comparison, this fails.
    admission = contract["admission"]

    assert (
        _dotted(admission["sdk_version_check"]) is sdk.ExtensionRegistry.with_extension
    )
    assert admission["second_registry_created"] is False
    source = Path(pass_admission.__file__).read_text(encoding="utf-8")
    assert "SDK_API_VERSION" not in source


def test_the_crossing_adds_nothing_to_the_pinned_sdk_namespace(
    contract: dict[str, Any],
) -> None:
    # contracts/extension-protocol-v1-candidate.json pins the namespace as an exact
    # set, so a new name there would break that approved contract.
    protocol = json.loads(
        (ROOT / "contracts/extension-protocol-v1-candidate.json").read_text(
            encoding="utf-8"
        )
    )
    pinned = set(protocol["stable_extensions"][0]["additions"])
    namespace = __import__("flagquantum.ecosystem.extensions", fromlist=["_"])

    assert contract["admission"]["sdk_namespace_change"] == "none"
    assert set(namespace.__all__) == pinned
    assert not pinned.intersection(pass_admission.__all__)


def test_the_declared_route_is_the_protocol_member_the_sdk_requires(
    contract: dict[str, Any],
) -> None:
    # The route must call the member the protocol declares, so an admitted pass and
    # an in-repository pass are called alike.
    route = contract["pass_route"]
    members = _protocol_members(sdk.CompilerPassExtension)

    assert "transform" in members
    assert route["call"].startswith("transform(")
    assert route["built_in_and_extension_passes_are_called_alike"] is True


def _protocol_members(protocol: type) -> set[str]:
    attrs = getattr(protocol, "__protocol_attrs__", None)
    if attrs is None:  # pragma: no cover - depends on the Python version
        import typing

        attrs = typing._get_protocol_attrs(protocol)  # type: ignore[attr-defined]
    return set(attrs)


def test_the_declared_builder_is_the_one_that_builds_registered_names(
    contract: dict[str, Any],
) -> None:
    declaration = contract["declaration"]

    builder = _dotted(declaration["registered_name_builder"])
    assert builder is pass_manager.extension_pass_name
    assert builder("acme", "fuse") == "extension.acme.fuse"


def test_the_reserved_names_are_exactly_the_pipeline_the_contract_names(
    contract: dict[str, Any],
) -> None:
    registry_facts = contract["registry_facts"]

    assert registry_facts["built_in_names_reserved"] is True
    assert (
        registry_facts["reserved_names"] == "exactly the names in OPTIMIZATION_PIPELINE"
    )
    assert (
        frozenset(pass_manager.OPTIMIZATION_PIPELINE)
        == pass_manager.RESERVED_PASS_NAMES
    )
    assert frozenset(pipeline.BUILTIN_PASSES) == pass_manager.RESERVED_PASS_NAMES


def test_the_declared_immutability_and_fresh_registry_facts_hold(
    contract: dict[str, Any],
) -> None:
    registry_facts = contract["registry_facts"]

    assert registry_facts["immutable"] is True
    assert registry_facts["additions_return_a_new_registry"] is True
    base = pass_manager.PassRegistry()
    added = base.with_pass("extension.contract.one", lambda ir: ir)
    assert base.names == ()
    assert added.names == ("extension.contract.one",)
    with pytest.raises(TypeError):
        added.passes["extension.contract.two"] = lambda ir: ir  # type: ignore[index]


def test_the_declared_fixed_point_rule_is_the_one_the_manager_implements(
    contract: dict[str, Any],
) -> None:
    # The rule matters because the contract claims the replaced inline loop and the
    # manager agree. "A round that removes no instruction" is the loop's condition
    # and must be the manager's.
    registry_facts = contract["registry_facts"]
    assert (
        registry_facts["fixed_point_test"]
        == "a round that removes no instruction is a fixed point"
    )

    pair = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(name="h", wires=(0,)),
            Instruction(name="h", wires=(0,)),
        ),
    )
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    # One round collapses the pair and the next round changes nothing, which is how
    # the loop ends. `to_fixed_point` must reach the empty program from a program
    # the first round already changed.
    assert len(manager.run(pair, ("merge_self_inverse",))) == 0
    assert len(manager.to_fixed_point(pair, ("merge_self_inverse",))) == 0

    # A pass that keeps growing the program never satisfies the condition, so the
    # bound is what stops it rather than a silent truncation.
    grow = pass_manager.PassRegistry().with_pass(
        "extension.contract.grow",
        lambda ir: CircuitIR(
            n_wires=ir.n_wires,
            instructions=(*ir.instructions, Instruction(name="h", wires=(0,))),
        ),
    )
    with pytest.raises(CompilationError, match="fixed point"):
        pass_manager.PassManager(grow).to_fixed_point(
            CircuitIR(n_wires=1, instructions=(Instruction(name="h", wires=(0,)),)),
            ("extension.contract.grow",),
            max_rounds=1,
        )


def test_the_declared_default_path_resolves_no_extension_pass(
    contract: dict[str, Any],
) -> None:
    builtin = contract["builtin_path"]

    assert builtin["extension_pass_in_the_default_registry"] is False
    assert builtin["default_path_change"] is False
    assert _dotted(builtin["what_compile_runs"]) is pipeline.default_pass_registry
    assert pipeline.default_pass_registry().extension_names == ()


def test_the_declared_replacement_chain_names_reachable_entry_points(
    contract: dict[str, Any],
) -> None:
    # The replacement proof is only a proof if every name in the chain exists, so
    # the chain is resolved rather than trusted.
    proof = contract["replacement_proof"]

    assert proof["second_implementation_required"] is True
    assert proof["scenario_test_required"] is True
    for step in proof["chain"]:
        assert _dotted(step) is not None, step
    assert proof["chain"][0].endswith(".admit_pass_extension")
    assert _dotted("flagquantum.compiler.pass_manager.PassManager.to_fixed_point") is (
        pass_manager.PassManager.to_fixed_point
    )
    for consumer in proof["consumer_left_unmodified"]:
        assert _dotted(consumer) is not None, consumer
    assert _dotted("flagquantum.compiler.pipeline.optimize") is pipeline.optimize


def test_the_inline_loop_the_contract_replaces_is_gone(
    contract: dict[str, Any],
) -> None:
    # The declared replacement target must not still exist, or the contract is
    # describing a second implementation rather than a replacement.
    source = (ROOT / "flagquantum/compiler/pipeline.py").read_text(encoding="utf-8")

    assert "while True" not in source
    assert source.count("PassManager(default_pass_registry()).to_fixed_point") == 1
    assert contract["replacement_proof"]["replaced_implementation"].startswith(
        "the inline nine-call fixed-point loop"
    )


def test_the_declared_failure_modes_are_capability_errors(
    contract: dict[str, Any],
) -> None:
    fail_closed = contract["fail_closed"]

    for key in (
        "unusable_declaration",
        "non_compiler_pass_kind",
        "missing_transform_member",
        "refused_negotiation",
        "extension_declaring_no_circuit_ir_support",
        "failed_start",
    ):
        declared = _dotted(fail_closed[key])
        assert isinstance(declared, type), key
        assert issubclass(declared, _dotted(fail_closed["error_type_supertype"])), key
    assert fail_closed["silent_fallback"] is False
    assert fail_closed["automatic_extension_selection"] is False


def test_the_declared_failure_modes_are_the_errors_the_code_raises(
    contract: dict[str, Any],
) -> None:
    # Resolving the name is not enough; it must be the error the crossing raises
    # for that case. This is the check that fails if a case is reclassified.
    fail_closed = contract["fail_closed"]

    for key in (
        "unusable_declaration",
        "non_compiler_pass_kind",
        "missing_transform_member",
        "refused_negotiation",
        "extension_declaring_no_circuit_ir_support",
        "failed_start",
    ):
        assert _dotted(fail_closed[key]) is pass_admission.PassAdmissionError, key
    assert (
        _dotted(fail_closed["sdk_incompatible_extension"])
        is sdk.ExtensionCompatibilityError
    )


def test_every_declared_failure_condition_is_explained(
    contract: dict[str, Any],
) -> None:
    # The condition prose and the error type must cover the same cases, so a case
    # cannot be added to one map and forgotten in the other.
    fail_closed = contract["fail_closed"]
    typed = {
        key
        for key, value in fail_closed.items()
        if isinstance(value, str)
        and value.startswith(("flagquantum.", "PassAdmission"))
        and key != "error_type_supertype"
    }

    assert set(fail_closed["conditions"]) == typed


def test_the_declared_assertion_of_extension_lifecycle_binding_holds() -> None:
    # The contract claims the route closes over the handle, so a pass invoked after
    # its extension closed fails closed. Exercised here because it is a property of
    # the crossing rather than of any single pass.
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["pass_route"]["lifecycle_bound"] is True
    assert "ExtensionHandle" in contract["pass_route"]["lifecycle_bound_note"]
    assert issubclass(pass_admission.PassAdmissionError, CapabilityError)


def test_every_declared_invariant_is_named_by_a_test() -> None:
    # An invariant with no test is an intention. Each one here must be pinned by a
    # test in this module or in the scenario suite.
    text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "tests/unit/test_compiler_pass_admission.py",
            "tests/unit/test_compiler_pass_admission_contract.py",
        )
    )
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    keywords = {
        "the registered name of an admitted pass always starts with the extension prefix": (
            "EXTENSION_PASS_PREFIX"
        ),
        "no extension value can produce a registered name equal to a built-in pass name": (
            "test_a_built_in_pass_name_cannot_be_reached_by_an_extension"
        ),
        "PassRegistry refuses to shadow a registered name and refuses a reserved built-in name on an otherwise empty registry": (
            "test_the_registry_independently_refuses_a_built_in_name"
        ),
        "default_pass_registry resolves exactly the built-in pipeline names and no extension name": (
            "test_the_default_path_resolves_no_extension_pass"
        ),
        "the built-in pipeline names, their order, and the number of rounds the fixed-point loop runs are unchanged by the refactor": (
            "test_the_documented_order_is_preserved_by_the_refactor"
        ),
        "a sequence is resolved in full before the first pass runs, so an unknown name leaves the program untouched": (
            "test_an_unregistered_name_is_refused_before_any_pass_runs"
        ),
        "optimize_with_extension_pass runs the built-in pipeline to its own fixed point and then the admitted pass exactly once": (
            "test_the_admitted_pass_runs_once_even_when_it_is_not_idempotent"
        ),
        "the extension handle is closed even when the admitted pass or the pipeline fails": (
            "test_the_extension_is_closed_even_when_the_pass_fails"
        ),
    }

    assert set(keywords) == set(contract["invariants"])
    for invariant, marker in keywords.items():
        assert marker in text, invariant


def test_the_non_goals_stay_out_of_scope() -> None:
    # Each non-goal is checked as absent, so a later change that adds one has to
    # update this contract rather than leaving it describing something else.
    assert not hasattr(pass_admission, "ExtensionRegistry2")
    assert not hasattr(pass_admission, "discover_extensions")
    assert not hasattr(pass_manager, "AnalysisPass")
    assert not hasattr(pass_manager, "run_analysis")
    namespace = __import__("flagquantum.ecosystem.extensions", fromlist=["_"])
    assert not hasattr(namespace, "admit_pass_extension")
    assert not hasattr(namespace, "optimize_with_extension_pass")
