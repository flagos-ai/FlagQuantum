"""Managed compiler passes and the pipeline that runs them."""

from dataclasses import replace

import pytest

from flagquantum import CircuitIR, Instruction
from flagquantum.compiler import optimize
from flagquantum.compiler.passes import (
    IDENTITY_REMOVAL,
    ROTATION_MERGE,
    SELF_INVERSE_CANCELLATION,
    PassDeclaration,
    PassManager,
    RegisteredPass,
    pass_from_extension,
)
from flagquantum.compiler.pipeline import DEFAULT_OPTIMIZATION_PASSES
from flagquantum.ecosystem.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)
from flagquantum.errors import CapabilityError, CompilationError

pytestmark = pytest.mark.unit

_SEMANTICS = ("program_equivalence",)


class ReferencePassExtension:
    """An independently installable pass that drops every Hadamard gate."""

    manifest = ExtensionManifest(
        name="reference_pass",
        version="1.0.0",
        kind="compiler_pass",
        capabilities=frozenset({"circuit_ir"}),
    )
    declaration = PassDeclaration(
        name="reference_pass",
        summary="drops Hadamard gates",
        preserved_semantics=_SEMANTICS,
        deterministic=True,
    )

    def __init__(self) -> None:
        self.active = False
        self.activations = 0

    def negotiate(self, request: CapabilityRequest) -> CapabilityResponse:
        missing = request.required - self.manifest.capabilities
        return CapabilityResponse(
            not missing,
            self.manifest.capabilities,
            (
                ("missing capabilities: " + ", ".join(sorted(missing)),)
                if missing
                else ()
            ),
        )

    def start(self, config: ExtensionConfig) -> None:
        self.active = True
        self.activations += 1

    def transform(self, program: CircuitIR) -> CircuitIR:
        if not self.active:
            raise RuntimeError("pass extension is closed")
        return replace(
            program,
            instructions=tuple(
                item for item in program if item.name not in {"h", "hadamard"}
            ),
        )

    def close(self) -> None:
        self.active = False


def _wire_local_ir() -> CircuitIR:
    """A circuit with one cancellable pair and one mergeable rotation pair."""

    return CircuitIR(
        2,
        (
            Instruction("x", (0,)),
            Instruction("rz", (0,), params={"theta": 0.2}),
            Instruction("rz", (0,), params={"theta": -0.2}),
            Instruction("x", (0,)),
            Instruction("h", (1,)),
        ),
    )


def test_default_pipeline_is_the_registered_transform_set() -> None:
    assert tuple(item.name for item in DEFAULT_OPTIMIZATION_PASSES) == (
        "identity_removal",
        "self_inverse_cancellation",
        "rotation_merge",
    )
    assert PassManager(passes=DEFAULT_OPTIMIZATION_PASSES).names == (
        IDENTITY_REMOVAL.name,
        SELF_INVERSE_CANCELLATION.name,
        ROTATION_MERGE.name,
    )


def test_a_replacement_pipeline_changes_the_result_without_touching_consumers() -> None:
    """The swap is the admission test: same entry point, different pass set."""

    def drop_hadamards(program: CircuitIR) -> CircuitIR:
        return replace(
            program,
            instructions=tuple(item for item in program if item.name != "h"),
        )

    replacement = RegisteredPass(
        declaration=PassDeclaration(
            name="drop_hadamards",
            summary="drops Hadamard gates",
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        ),
        transform=drop_hadamards,
    )

    default_result = optimize(_wire_local_ir())
    replaced_result = optimize(_wire_local_ir(), passes=(replacement,))

    assert tuple(item.name for item in default_result) == ("h",)
    assert replaced_result.instructions == ()


def test_the_default_pipeline_is_unchanged_when_no_passes_are_given() -> None:
    ir = _wire_local_ir()

    assert optimize(ir).instructions == optimize(ir, passes=None).instructions


def test_extension_supplied_pass_rewrites_the_program_through_the_manager() -> None:
    extension = ReferencePassExtension()
    managed = pass_from_extension(extension)

    ir = CircuitIR(1, (Instruction("h", (0,)), Instruction("x", (0,))))
    compiled = optimize(ir, passes=(managed,))

    assert compiled.instructions == (Instruction("x", (0,)),)
    assert extension.activations == 1
    assert extension.active is False


def test_a_negotiated_pass_extension_is_not_scheduled_into_a_strict_pipeline() -> None:
    class AnonymousExtension(ReferencePassExtension):
        declaration = None

    managed = pass_from_extension(AnonymousExtension())

    assert managed.declaration.preserved_semantics == ()
    with pytest.raises(CapabilityError, match="does not declare that it preserves"):
        PassManager(passes=(managed,))


def test_a_pass_that_does_not_declare_a_required_property_is_refused() -> None:
    silent = RegisteredPass(
        declaration=PassDeclaration(
            name="silent",
            summary="declares nothing it keeps",
            deterministic=True,
        ),
        transform=lambda program: program,
    )

    with pytest.raises(CapabilityError, match="does not declare that it preserves"):
        PassManager(passes=(silent,))


def test_a_pass_that_does_not_guarantee_deterministic_replay_is_refused() -> None:
    nondeterministic = RegisteredPass(
        declaration=PassDeclaration(
            name="nondeterministic",
            summary="may reorder gates on each run",
            preserved_semantics=_SEMANTICS,
            deterministic=False,
        ),
        transform=lambda program: program,
    )

    with pytest.raises(CapabilityError, match="deterministic replay"):
        PassManager(passes=(nondeterministic,))


def test_a_pass_declaring_a_foreign_ir_level_is_refused() -> None:
    foreign = RegisteredPass(
        declaration=PassDeclaration(
            name="quantum_level",
            summary="consumes a lower level than the manager executes",
            input_level="quantum_ir",
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        ),
        transform=lambda program: program,
    )

    with pytest.raises(CapabilityError, match="this manager executes 'circuit_ir'"):
        PassManager(passes=(foreign,))


def test_a_pass_requiring_an_analysis_an_earlier_pass_invalidated_is_refused() -> None:
    invalidating = RegisteredPass(
        declaration=PassDeclaration(
            name="invalidating",
            summary="keeps no analysis",
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        ),
        transform=lambda program: program,
    )
    dependent = RegisteredPass(
        declaration=PassDeclaration(
            name="dependent",
            summary="reads the def-use relation",
            required_analyses=("def_use",),
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        ),
        transform=lambda program: program,
    )

    with pytest.raises(CapabilityError, match="requires def_use after an earlier"):
        PassManager(passes=(invalidating, dependent))


def test_analysis_survival_is_the_intersection_over_the_pipeline() -> None:
    keeping = RegisteredPass(
        declaration=PassDeclaration(
            name="keeping",
            summary="keeps the def-use relation and the cost analysis",
            preserved_analyses=("def_use", "circuit_cost"),
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        ),
        transform=lambda program: program,
    )
    dropping = RegisteredPass(
        declaration=PassDeclaration(
            name="dropping",
            summary="keeps only the cost analysis",
            preserved_analyses=("circuit_cost",),
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        ),
        transform=lambda program: program,
    )

    manager = PassManager(passes=(keeping, dropping))

    assert manager.surviving_analyses == ("circuit_cost",)


def test_an_invented_analysis_name_is_refused_when_the_pass_is_declared() -> None:
    with pytest.raises(CompilationError, match="unknown required_analyses"):
        PassDeclaration(
            name="invented",
            summary="names an analysis that does not exist",
            required_analyses=("vibes",),
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        )


def test_an_invented_ir_level_is_refused_when_the_pass_is_declared() -> None:
    with pytest.raises(CompilationError, match="unknown input_level"):
        PassDeclaration(
            name="invented",
            summary="names a level that does not exist",
            input_level="quantum_assembly",
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        )


def test_a_pass_without_a_name_or_summary_is_refused() -> None:
    with pytest.raises(CompilationError, match="non-empty name"):
        PassDeclaration(name="  ", summary="s", preserved_semantics=(), deterministic=True)
    with pytest.raises(CompilationError, match="non-empty summary"):
        PassDeclaration(name="n", summary=" ", preserved_semantics=(), deterministic=True)


def test_the_manager_refuses_the_same_pass_twice() -> None:
    with pytest.raises(CompilationError, match="registered twice"):
        PassManager(passes=(IDENTITY_REMOVAL, IDENTITY_REMOVAL))


def test_the_manager_refuses_a_non_pass_entry() -> None:
    with pytest.raises(CompilationError, match="must be a RegisteredPass"):
        PassManager(passes=(lambda program: program,))  # type: ignore[arg-type]


def test_the_manager_refuses_a_program_that_is_not_circuit_ir() -> None:
    with pytest.raises(TypeError, match="consumes CircuitIR, got tuple"):
        PassManager().run((Instruction("h", (0,)),))


def test_a_pass_that_returns_the_wrong_type_is_refused() -> None:
    lying = RegisteredPass(
        declaration=PassDeclaration(
            name="lying",
            summary="returns something other than a program",
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        ),
        transform=lambda program: "not a program",  # type: ignore[arg-type,return-value]
    )

    with pytest.raises(CompilationError, match="must return CircuitIR"):
        PassManager(passes=(lying,)).run(CircuitIR(1))


def test_a_pipeline_that_never_settles_is_refused_rather_than_looping() -> None:
    inflating_ir = CircuitIR(1, (Instruction("h", (0,)),))
    appending = RegisteredPass(
        declaration=PassDeclaration(
            name="appending",
            summary="adds one gate per round",
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        ),
        transform=lambda program: replace(
            program,
            instructions=(*program.instructions, Instruction("h", (0,))),
        ),
    )

    with pytest.raises(CompilationError, match="did not reach a fixed point"):
        PassManager(passes=(appending,)).run_to_fixed_point(inflating_ir)


def test_the_contract_records_what_the_pipeline_declares() -> None:
    manager = PassManager(passes=DEFAULT_OPTIMIZATION_PASSES)
    contract = manager.contract()

    assert contract["kind"] == "flagquantum_pass_contract_v1"
    assert contract["managed_ir_level"] == "circuit_ir"
    assert contract["required_semantics"] == ["program_equivalence"]
    assert [item["name"] for item in contract["passes"]] == list(manager.names)
    assert all(item["deterministic"] for item in contract["passes"])
    assert all(
        item["preserved_semantics"] == ["program_equivalence"]
        for item in contract["passes"]
    )


def test_a_declaration_round_trips_through_its_own_record() -> None:
    record = IDENTITY_REMOVAL.declaration.to_dict()
    rebuilt = PassDeclaration(
        name=record["name"],
        summary=record["summary"],
        deterministic=record["deterministic"],
        input_level=record["input_level"],
        output_level=record["output_level"],
        required_analyses=tuple(record["required_analyses"]),
        preserved_analyses=tuple(record["preserved_analyses"]),
        preserved_semantics=tuple(record["preserved_semantics"]),
    )

    assert rebuilt == IDENTITY_REMOVAL.declaration


def test_the_manager_enforces_a_declaration_not_the_claim_inside_it() -> None:
    """A declaration is a claim. The manager checks that it is made, not that it holds."""

    def corrupt(program: CircuitIR) -> CircuitIR:
        return replace(program, n_wires=program.n_wires + 1)

    claiming = RegisteredPass(
        declaration=PassDeclaration(
            name="claiming",
            summary="claims equivalence while changing the wire count",
            preserved_semantics=_SEMANTICS,
            deterministic=True,
        ),
        transform=corrupt,
    )

    manager = PassManager(passes=(claiming,))

    assert manager.run(CircuitIR(1)).n_wires == 2


def test_the_registered_transforms_still_satisfy_their_declarations() -> None:
    ir = _wire_local_ir()
    compiled = PassManager(passes=DEFAULT_OPTIMIZATION_PASSES).run(ir)

    assert tuple(item.name for item in compiled) == ("h",)
    assert ir.instructions != compiled.instructions
