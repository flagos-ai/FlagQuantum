"""Compiler pass admission: one crossing from the extension SDK to the Compiler.

These tests demonstrate the boundary rather than the mechanism. The decisive one
is :func:`test_a_third_party_pass_changes_compilation_without_a_source_change`:
a pass defined outside the repository is admitted and then runs inside the
optimization pipeline, with no FlagQuantum source modified and no name added to
any list in the Compiler. Everything else here is a failure mode, and every one of
them is rejected before the pass is registered.

The other property these tests pin is the one that makes the crossing safe: an
extension registers a new name and can never redefine a built-in one, so a
third-party pass cannot silently change what `compile` means.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest

import flagquantum as fq
from flagquantum.compiler import pass_manager, pipeline
from flagquantum.compiler.pass_manager import (
    OPTIMIZATION_PIPELINE,
    PassManager,
    PassRegistryError,
)
from flagquantum.compiler.pipeline import default_pass_registry
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionCompatibilityError,
    ExtensionConfig,
    ExtensionManifest,
)
from flagquantum.ecosystem.extensions.pass_admission import (
    IR_CAPABILITY,
    PASS_DECLARATION_MEMBER,
    PassAdmissionError,
    admit_pass_extension,
    admitted_pass_name,
    check_pass_admission,
    optimize_with_extension_pass,
)
from flagquantum.errors import CapabilityError, CompilationError

pytestmark = pytest.mark.unit


class _Lifecycle:
    """The SDK lifecycle members, without `transform`.

    Kept separate from `_CircuitPass` so a test can build an extension of kind
    `compiler_pass` that genuinely has no `transform` member, which is a different
    failure from a `transform` that is not callable.
    """

    manifest = ExtensionManifest(
        name="scripted_pass",
        version="1.0.0",
        kind="compiler_pass",
        capabilities=frozenset({IR_CAPABILITY}),
    )

    def __init__(self) -> None:
        self.active = False
        self.starts = 0
        self.closes = 0
        self.pass_name: Any = "scripted"
        self.behaviour: Any = None
        self.accept = True
        self.blockers: tuple[str, ...] = ()
        self.negotiated: list[CapabilityRequest] = []

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        self.negotiated.append(request)
        return CapabilityResponse(
            self.accept, self.manifest.capabilities, self.blockers
        )

    def start(self, config: ExtensionConfig) -> None:
        self.starts += 1
        self.active = True

    def close(self) -> None:
        self.closes += 1
        self.active = False


class _CircuitPass(_Lifecycle):
    """A minimal compiler_pass extension whose transform is chosen by the test.

    `transform` is the member `CompilerPassExtension` requires, so this class is
    the protocol's shape and not a FlagQuantum-specific convenience. The callable
    the test supplies is a plain instance attribute rather than a class-level
    method, so a one-argument transform is not turned into a bound method.
    """

    def transform(self, program: CircuitIR) -> CircuitIR:
        if self.behaviour is not None:
            return self.behaviour(program)
        return program


def _variant(
    extension_name: str,
    *,
    kind: str = "compiler_pass",
    transform: Any = None,
    pass_name: Any = "scripted",
    accept: bool = True,
    blockers: tuple[str, ...] = (),
) -> _CircuitPass:
    """A `_CircuitPass` that differs only in the field under test.

    `extension_name` is deliberately not called `name`: `name` is one of the
    host-owned keys a test has to be able to pass through `pass_name`.
    """

    extension = type(
        extension_name,
        (_CircuitPass,),
        {
            "manifest": ExtensionManifest(
                name=extension_name,
                version="1.0.0",
                kind=kind,
                capabilities=frozenset({IR_CAPABILITY}),
            )
        },
    )()
    extension.behaviour = transform
    extension.pass_name = pass_name
    extension.accept = accept
    extension.blockers = tuple(blockers)
    return extension


def _swap_h_for_x(program: CircuitIR) -> CircuitIR:
    """A pass whose effect no built-in pass produces, so it is observable."""

    return CircuitIR(
        n_wires=program.n_wires,
        instructions=tuple(
            (
                Instruction(name="x", wires=instruction.wires)
                if instruction.name == "h"
                else instruction
            )
            for instruction in program.instructions
        ),
    )


def _program() -> CircuitIR:
    return CircuitIR(
        n_wires=2,
        instructions=(
            Instruction(name="h", wires=(0,)),
            Instruction(name="h", wires=(1,)),
        ),
    )


def _names(program: CircuitIR) -> list[str]:
    return [instruction.name for instruction in program.instructions]


# --------------------------------------------------------------------------
# Scenario: a third-party pass compiles a program without a source change.
# --------------------------------------------------------------------------


def test_a_third_party_pass_changes_compilation_without_a_source_change() -> None:
    # `optimize_with_extension_pass` is the only name this test needs from
    # FlagQuantum, and no built-in pass turns `h` into `x`, so the change in the
    # output can only have come from the admitted pass.
    before = _program()
    assert _names(before) == ["h", "h"]

    after = optimize_with_extension_pass(
        before, extension=_variant("swapper", transform=_swap_h_for_x)
    )

    assert _names(after) == ["x", "x"]


def test_the_admitted_pass_runs_after_the_built_in_pipeline() -> None:
    # The pipeline runs first, so a pass sees what the pipeline left rather than
    # what the caller wrote. `h h` on one wire is what the pipeline collapses.
    seen: list[list[str]] = []

    def _record(program: CircuitIR) -> CircuitIR:
        seen.append(_names(program))
        return program

    program = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(name="h", wires=(0,)),
            Instruction(name="h", wires=(0,)),
        ),
    )

    optimize_with_extension_pass(
        program, extension=_variant("recorder", transform=_record)
    )

    assert seen == [[]]


def test_the_admitted_pass_runs_once_even_when_it_is_not_idempotent() -> None:
    # The pass runs after the pipeline rather than inside it, so a pass that is not
    # idempotent cannot hold the pipeline in an unbounded loop.
    calls: list[int] = []

    def _append(program: CircuitIR) -> CircuitIR:
        calls.append(len(calls))
        return CircuitIR(
            n_wires=program.n_wires,
            instructions=(*program.instructions, Instruction(name="z", wires=(0,))),
        )

    program = CircuitIR(n_wires=1, instructions=(Instruction(name="x", wires=(0,)),))

    result = optimize_with_extension_pass(
        program, extension=_variant("appender", transform=_append)
    )

    assert calls == [0]
    assert _names(result) == ["x", "z"]


def test_compilation_through_an_admitted_pass_needs_no_registry_edit() -> None:
    # The built-in registry is unchanged by admitting, which is what "adds a pass
    # but does not modify the Compiler" means concretely.
    before = default_pass_registry().names

    _handle, registry = admit_pass_extension(_variant("unchanged"))

    assert default_pass_registry().names == before
    assert registry.names == tuple(sorted((*before, "extension.unchanged.scripted")))


def test_the_registered_name_names_the_extension_and_the_declared_pass() -> None:
    _handle, registry = admit_pass_extension(
        _variant("acme", pass_name="fuse_rotations")
    )

    assert admitted_pass_name(registry) == "extension.acme.fuse_rotations"


def test_the_admitted_registry_still_resolves_every_built_in_pass() -> None:
    _handle, registry = admit_pass_extension(_variant("alongside"))

    for name in OPTIMIZATION_PIPELINE:
        assert callable(registry.resolve(name))


def test_admission_returns_the_lifecycle_handle_it_started() -> None:
    extension = _variant("lifecycle")

    handle, _registry = admit_pass_extension(extension)

    assert handle.active is True
    assert extension.active is True
    assert extension.starts == 1

    handle.close()

    assert extension.active is False
    assert extension.closes == 1


def test_a_pass_invoked_after_its_extension_closed_fails_closed() -> None:
    # The adapter closes over the handle rather than the extension object, so
    # withdrawing the extension withdraws the pass.
    handle, registry = admit_pass_extension(_variant("withdrawn"))
    handle.close()

    with pytest.raises(Exception) as excinfo:
        PassManager(registry).run(_program(), (admitted_pass_name(registry),))

    assert "not active" in str(excinfo.value)


def test_a_config_is_passed_through_to_the_extension() -> None:
    seen: list[ExtensionConfig] = []

    class _Configured(_CircuitPass):
        def start(self, config: ExtensionConfig) -> None:
            seen.append(config)
            super().start(config)

    extension = type(
        "configured",
        (_Configured,),
        {
            "manifest": ExtensionManifest(
                name="configured",
                version="1.0.0",
                kind="compiler_pass",
                capabilities=frozenset({IR_CAPABILITY}),
            )
        },
    )()

    admit_pass_extension(extension, config=ExtensionConfig({"level": 3}))

    assert dict(seen[0].values) == {"level": 3}


def test_a_credential_shaped_config_is_still_refused_at_admission() -> None:
    # The SDK already owns that refusal; admission does not restate it and does not
    # bypass it, because the config travels through `ExtensionHandle.start`.
    with pytest.raises(ValueError, match="credential"):
        admit_pass_extension(
            _variant("credentialed"), config=ExtensionConfig({"api_token": "x"})
        )


# --------------------------------------------------------------------------
# The decisive safety property: a built-in name can never be taken.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(pass_manager.RESERVED_PASS_NAMES))
def test_a_built_in_pass_name_cannot_be_reached_by_an_extension(name: str) -> None:
    # The host builds the registered name, always under the extension prefix, so
    # no value the extension declares lands on a built-in name. A declared name
    # equal to a built-in one is a distinct name, not a shadow.
    registered = pass_manager.extension_pass_name("taker", name)

    assert registered == f"extension.taker.{name}"
    assert registered not in pass_manager.RESERVED_PASS_NAMES
    assert registered not in default_pass_registry().passes


def test_admitting_a_pass_named_after_a_built_in_leaves_the_built_in_alone() -> None:
    # The concrete consequence: admitting the pass registers one new name and
    # redefines nothing, so `compile` still runs the built-in transformation.
    before = default_pass_registry()
    _handle, registry = admit_pass_extension(
        _variant("taker", pass_name="remove_identity_gates")
    )

    assert registry.resolve("remove_identity_gates") is before.resolve(
        "remove_identity_gates"
    )
    assert registry.extension_names == ("extension.taker.remove_identity_gates",)


def test_the_registry_independently_refuses_a_built_in_name() -> None:
    # The second half of the same guarantee, on an empty registry so the refusal
    # cannot come from the name already being present: the reserved-name rule
    # alone refuses it. Neither half relies on the other.
    with pytest.raises(PassRegistryError, match="built-in"):
        pass_manager.PassRegistry().with_pass("remove_identity_gates", _noop)

    with pytest.raises(PassRegistryError, match="already registered"):
        default_pass_registry().with_pass("remove_identity_gates", _noop)


def test_a_name_collision_that_reaches_the_registry_is_reported_as_a_live_pass() -> (
    None
):
    # Reached when the host-built name is already registered by an earlier
    # admission in the same task. The registry, not this module, owns the refusal.
    registry = default_pass_registry().with_pass("extension.duplicate.scripted", _noop)

    with pytest.raises(PassRegistryError, match="already registered"):
        registry.with_pass("extension.duplicate.scripted", _noop)


def _noop(program: CircuitIR) -> CircuitIR:
    return program


def test_admitting_a_name_an_earlier_admission_took_is_refused() -> None:
    # `_validated` consults the live registry, so the second admission of the same
    # extension name is refused rather than displacing the first route.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "flagquantum.ecosystem.extensions.pass_admission.default_pass_registry",
            lambda: default_pass_registry().with_pass(
                "extension.duplicate.scripted", _noop
            ),
        )
        with pytest.raises(PassAdmissionError, match="never displaced silently"):
            admit_pass_extension(_variant("duplicate"))


def test_an_extension_is_closed_when_registration_fails_after_it_started() -> None:
    # The name can be taken between validation and registration. Admission did not
    # happen, so the extension that was already started must not be left running.
    class _Racing:
        passes = MappingProxyType({})

        def with_pass(self, name: str, function: object) -> object:
            raise PassRegistryError(f"pass {name!r} is already registered")

    extension = _variant("racing")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "flagquantum.ecosystem.extensions.pass_admission.default_pass_registry",
            _Racing,
        )
        with pytest.raises(PassRegistryError):
            admit_pass_extension(extension)

    assert extension.active is False
    assert extension.closes == 1


# --------------------------------------------------------------------------
# Failure modes: each is rejected before the pass is registered or runs.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["backend", "compiler", "kernel", "operator"])
def test_a_non_compiler_pass_kind_is_refused(kind: str) -> None:
    with pytest.raises(PassAdmissionError, match="kind"):
        admit_pass_extension(_variant("wrongkind", kind=kind))


def test_a_kind_of_compiler_pass_without_a_transform_member_is_refused() -> None:
    # A distinct failure from a `transform` that is not callable: this extension
    # declares the kind but does not implement the member the kind requires.
    extension = type(
        "notransform",
        (_Lifecycle,),
        {
            "manifest": ExtensionManifest(
                name="notransform",
                version="1.0.0",
                kind="compiler_pass",
                capabilities=frozenset({IR_CAPABILITY}),
            )
        },
    )()

    assert not hasattr(extension, "transform")

    with pytest.raises(PassAdmissionError, match="transform"):
        admit_pass_extension(extension)


def test_a_missing_pass_declaration_is_refused() -> None:
    extension = _variant("undeclared")
    del extension.pass_name

    with pytest.raises(PassAdmissionError, match=PASS_DECLARATION_MEMBER):
        admit_pass_extension(extension)


@pytest.mark.parametrize("declared", ["", "   "])
def test_a_blank_pass_declaration_is_refused_by_this_crossing(declared: str) -> None:
    # The refusal must name the declaration and come from validation rather than
    # from the naming rule below it, so a host can tell "you declared nothing" from
    # "the name you declared is unusable".
    with pytest.raises(PassAdmissionError, match=PASS_DECLARATION_MEMBER) as excinfo:
        admit_pass_extension(_variant("badname", pass_name=declared))

    assert "non-empty string" in str(excinfo.value)


@pytest.mark.parametrize("declared", [3, ["a"]])
def test_a_non_string_pass_declaration_is_refused_by_this_crossing(
    declared: object,
) -> None:
    with pytest.raises(PassAdmissionError, match=PASS_DECLARATION_MEMBER) as excinfo:
        admit_pass_extension(_variant("badname", pass_name=declared))

    assert type(declared).__name__ in str(excinfo.value)


def test_a_declared_name_containing_a_period_is_refused() -> None:
    with pytest.raises(PassAdmissionError, match=r"may not contain"):
        admit_pass_extension(_variant("dotted", pass_name="has.dot"))


def test_an_extension_name_containing_a_period_is_refused() -> None:
    with pytest.raises(PassAdmissionError, match=r"may not contain"):
        admit_pass_extension(_variant("has.dot"))


def test_a_refused_negotiation_blocks_admission() -> None:
    with pytest.raises(PassAdmissionError, match="rejected"):
        admit_pass_extension(
            _variant("refusing", accept=False, blockers=("no free qubits",))
        )


def test_the_negotiation_request_asks_for_flagquantum_ir() -> None:
    extension = _variant("asking")

    check_pass_admission(extension)

    assert extension.negotiated[0].required == frozenset({IR_CAPABILITY})


def test_an_extension_that_declares_no_ir_support_is_refused() -> None:
    # A pass written against some other program representation must not be
    # admitted, because the manager would call it with FlagQuantum IR.
    class _Silent(_CircuitPass):
        def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
            self.negotiated.append(request)
            return CapabilityResponse(True, frozenset(), ())

    extension = type(
        "silent",
        (_Silent,),
        {
            "manifest": ExtensionManifest(
                name="silent",
                version="1.0.0",
                kind="compiler_pass",
                capabilities=frozenset(),
            )
        },
    )()

    with pytest.raises(PassAdmissionError, match=IR_CAPABILITY):
        admit_pass_extension(extension)


def test_a_failing_negotiation_is_contained_as_an_admission_error() -> None:
    class _Exploding(_CircuitPass):
        def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
            raise RuntimeError("negotiation exploded")

    extension = type(
        "exploding",
        (_Exploding,),
        {
            "manifest": ExtensionManifest(
                name="exploding",
                version="1.0.0",
                kind="compiler_pass",
                capabilities=frozenset({IR_CAPABILITY}),
            )
        },
    )()

    with pytest.raises(PassAdmissionError, match="exploded"):
        admit_pass_extension(extension)


def test_a_failing_start_is_contained_and_leaves_no_route() -> None:
    class _Unstartable(_CircuitPass):
        def start(self, config: ExtensionConfig) -> None:
            raise RuntimeError("cannot start")

    extension = type(
        "unstartable",
        (_Unstartable,),
        {
            "manifest": ExtensionManifest(
                name="unstartable",
                version="1.0.0",
                kind="compiler_pass",
                capabilities=frozenset({IR_CAPABILITY}),
            )
        },
    )()

    with pytest.raises(PassAdmissionError, match="cannot start"):
        admit_pass_extension(extension)


def test_an_sdk_incompatible_extension_is_rejected_before_activation() -> None:
    extension = _variant("incompatible")
    object.__setattr__(extension.manifest, "api_version", "99.0")

    with pytest.raises(ExtensionCompatibilityError):
        admit_pass_extension(extension)

    assert extension.starts == 0
    assert extension.active is False


def test_checking_admission_starts_nothing_and_registers_nothing() -> None:
    extension = _variant("checked")
    before = default_pass_registry().names

    name = check_pass_admission(extension)

    assert name == "extension.checked.scripted"
    assert extension.starts == 0
    assert extension.active is False
    assert default_pass_registry().names == before


def test_a_transform_that_returns_a_non_program_is_refused_by_name() -> None:
    with pytest.raises(CompilationError) as excinfo:
        optimize_with_extension_pass(
            _program(), extension=_variant("wrong", transform=lambda program: "nope")
        )

    assert "extension.wrong.scripted" in str(excinfo.value)


def test_a_transform_that_raises_is_contained_by_the_handle() -> None:
    def _explode(program: CircuitIR) -> CircuitIR:
        raise RuntimeError("transform exploded")

    with pytest.raises(Exception) as excinfo:
        optimize_with_extension_pass(
            _program(), extension=_variant("boom", transform=_explode)
        )

    assert "exploded" in str(excinfo.value)


def test_a_pass_error_is_also_a_capability_error() -> None:
    assert issubclass(PassAdmissionError, CapabilityError)


def test_admitted_pass_name_refuses_a_registry_with_no_extension_pass() -> None:
    with pytest.raises(PassAdmissionError, match="none"):
        admitted_pass_name(default_pass_registry())


def test_admitted_pass_name_refuses_two_extension_passes() -> None:
    registry = default_pass_registry()
    registry = registry.with_pass("extension.one.a", _noop)
    registry = registry.with_pass("extension.two.b", _noop)

    with pytest.raises(PassAdmissionError, match="exactly one"):
        admitted_pass_name(registry)


# --------------------------------------------------------------------------
# The crossing is one boundary, not a second pipeline.
# --------------------------------------------------------------------------


def test_the_extension_is_closed_even_when_the_pass_fails() -> None:
    extension = _variant("closed", transform=lambda program: "nope")

    with pytest.raises(CompilationError):
        optimize_with_extension_pass(_program(), extension=extension)

    assert extension.active is False
    assert extension.closes == 1


def test_the_module_publishes_no_second_registry_or_protocol() -> None:
    # The admission module is a crossing, so the SDK types it uses must be the
    # SDK's own objects, and it must define no discovery path or second registry
    # of its own.
    from flagquantum.ecosystem.extensions import pass_admission, sdk

    assert pass_admission.ExtensionRegistry is sdk.ExtensionRegistry
    assert pass_admission.CompilerPassExtension is sdk.CompilerPassExtension
    assert pass_admission.ExtensionHandle is sdk.ExtensionHandle
    assert pass_admission.ExtensionConfig is sdk.ExtensionConfig
    assert not hasattr(pass_admission, "discover_extensions")
    assert not hasattr(pass_admission, "ExtensionManifest")


def test_the_module_declares_one_entry_point_per_direction() -> None:
    # One check that registers nothing, one admission that returns a registry, and
    # one call that runs the pass. More than that would be a second pipeline.
    from flagquantum.ecosystem.extensions import pass_admission

    assert set(pass_admission.__all__) == {
        "IR_CAPABILITY",
        "PASS_DECLARATION_MEMBER",
        "PassAdmissionError",
        "admit_pass_extension",
        "admitted_pass_name",
        "check_pass_admission",
        "optimize_with_extension_pass",
    }


def test_the_compiler_publishes_no_extension_names_of_its_own() -> None:
    # The Compiler owns names, the ecosystem owns discovery; the built-in registry
    # must resolve no extension pass, so `compile` is defined by built-ins alone.
    assert default_pass_registry().extension_names == ()
    assert pipeline.default_pass_registry().extension_names == ()


def test_the_public_optimize_path_is_unaffected_by_the_crossing() -> None:
    # A third-party pass reaches compilation only through admission, never through
    # `fq.compile`.
    program = fq.Circuit(1).h(0).h(0)

    import flagquantum.compiler as compiler

    assert _names(compiler.compile(program)) == []
