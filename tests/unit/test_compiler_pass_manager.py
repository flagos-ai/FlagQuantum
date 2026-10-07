"""The optimization pipeline is a named, replaceable sequence of registered passes.

The pipeline used to be nine inline calls in `pipeline._optimize_to_fixed_point`,
and these tests exist so that it stays a nameable sequence: every pass the
pipeline runs resolves in the default registry, the order is the order the inline
loop documented, and the manager reproduces the loop's fixed-point semantics. The
last group pins the two properties that make an extension pass possible at all --
a name resolves to a pass, and a built-in name cannot be taken.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import pytest

import flagquantum as fq
from flagquantum.compiler import pass_manager, pipeline
from flagquantum.core.ir import (
    CircuitIR,
    Instruction,
    ensure_circuit_ir,
)
from flagquantum.errors import CapabilityError, CompilationError

pytestmark = pytest.mark.unit


def _ir(*instructions: Instruction) -> CircuitIR:
    return CircuitIR(n_wires=2, instructions=tuple(instructions))


def _reset(wire: int = 0) -> Instruction:
    """A `reset` as the dynamic front end lowers one.

    `reset` is not a declared opcode, so the IR accepts it only as a dynamic
    instruction; this is the shape `DynamicCircuit.reset` produces.
    """

    return Instruction(name="reset", wires=(wire,), metadata={"is_dynamic": True})


def _names(ir: CircuitIR) -> list[str]:
    return [instruction.name for instruction in ir.instructions]


class _NoOp:
    """A pass that returns its input, used to count how often a round ran."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, program: CircuitIR) -> CircuitIR:
        self.calls += 1
        return program


class _DropsEverything:
    """A pass that empties the program, so the next round sees a fixed point."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, program: CircuitIR) -> CircuitIR:
        self.calls += 1
        return CircuitIR(n_wires=program.n_wires, instructions=())


def _noop(program: CircuitIR) -> CircuitIR:
    """A pass that returns its input, for tests that only build a declaration."""

    return program


def _registry_with(name: str, function: object) -> pass_manager.PassRegistry:
    return pass_manager.PassRegistry().with_pass(name, function)  # type: ignore[arg-type]


def test_every_name_in_the_pipeline_resolves_in_the_default_registry() -> None:
    registry = pipeline.default_pass_registry()

    for name in pass_manager.OPTIMIZATION_PIPELINE:
        assert callable(registry.resolve(name))


def test_the_default_registry_resolves_exactly_the_names_the_pipeline_runs() -> None:
    registry = pipeline.default_pass_registry()

    assert set(registry.names) == set(pass_manager.OPTIMIZATION_PIPELINE)
    assert registry.extension_names == ()


def test_the_pipeline_names_each_built_in_pass_at_least_once() -> None:
    assert set(pass_manager.OPTIMIZATION_PIPELINE) == pass_manager.RESERVED_PASS_NAMES


def test_remove_identity_gates_runs_twice_because_the_sequence_says_so() -> None:
    # The repetition is a property of the sequence, not of the pass: the passes
    # between the two calls expose new identity pairs.
    sequence = pass_manager.OPTIMIZATION_PIPELINE

    assert sequence.count("remove_identity_gates") == 2
    assert sequence[2] == "remove_identity_gates"
    assert sequence[-2] == "remove_identity_gates"


def test_the_documented_order_is_preserved_by_the_refactor() -> None:
    # The whole sequence is pinned, not only its ends: the order is the artifact
    # this refactor made explicit, so reordering it has to be a deliberate edit to
    # this assertion rather than a silent one.
    assert pass_manager.OPTIMIZATION_PIPELINE == (
        "remove_zero_state_resets",
        "remove_diagonal_gates_before_measure",
        "remove_identity_gates",
        "merge_self_inverse",
        "merge_inverse_pairs",
        "merge_adjacent_rotations",
        "cancel_commuting_self_inverse",
        "remove_identity_gates",
        "collapse_one_qubit_runs",
    )


def test_the_manager_reproduces_the_pipeline_the_inline_loop_implemented() -> None:
    # `x(0) x(0) reset(0)` is the case the inline comment named as needing a second
    # round, so it is the case that proves the loop, not just one pass, is intact.
    program = _ir(
        Instruction(name="x", wires=(0,)),
        Instruction(name="x", wires=(0,)),
        _reset(0),
        Instruction(name="h", wires=(1,)),
        Instruction(name="h", wires=(1,)),
    )

    optimized = pipeline.optimize(program)

    assert _names(optimized) == []


def test_a_single_round_leaves_a_program_that_needs_a_second_round() -> None:
    # The pass that removes this reset needs to see the pair gone first, and the
    # pair is removed later in the same round, so one round cannot finish it.
    program = _ir(
        Instruction(name="x", wires=(0,)),
        Instruction(name="x", wires=(0,)),
        _reset(0),
    )
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    one_round = manager.run(program, pass_manager.OPTIMIZATION_PIPELINE)
    fixed = manager.to_fixed_point(program, pass_manager.OPTIMIZATION_PIPELINE)

    assert _names(one_round) == ["reset"]
    assert _names(fixed) == []


def test_a_round_that_removes_nothing_ends_the_loop() -> None:
    program = _ir(Instruction(name="h", wires=(0,)))
    pass_ = _NoOp()
    registry = _registry_with("extension.example.steady", pass_)
    manager = pass_manager.PassManager(registry)

    result = manager.to_fixed_point(program, ("extension.example.steady",))

    assert result is program
    assert pass_.calls == 1


def test_a_round_that_changes_the_program_earns_another_round() -> None:
    program = _ir(Instruction(name="h", wires=(0,)))
    pass_ = _DropsEverything()
    registry = _registry_with("extension.example.empty", pass_)
    manager = pass_manager.PassManager(registry)

    result = manager.to_fixed_point(program, ("extension.example.empty",))

    assert _names(result) == []
    # One round that emptied the program, one round that found it empty already.
    assert pass_.calls == 2


def test_the_fixed_point_bound_defaults_to_one_round_per_instruction_plus_one() -> None:
    # A pass that always removes exactly one instruction reaches the empty program
    # in as many rounds as it had instructions; the bound must admit that.
    program = ensure_circuit_ir(fq.Circuit(1).h(0).h(0).h(0).h(0).h(0))
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    assert (
        manager.to_fixed_point(program, pass_manager.OPTIMIZATION_PIPELINE) is not None
    )


def test_a_sequence_that_cannot_reach_a_fixed_point_is_refused() -> None:
    program = _ir(Instruction(name="h", wires=(0,)))
    registry = _registry_with("extension.example.grow", lambda ir: _grow(ir))
    manager = pass_manager.PassManager(registry)

    with pytest.raises(CompilationError, match="fixed point"):
        manager.to_fixed_point(program, ("extension.example.grow",), max_rounds=3)


def _grow(program: CircuitIR) -> CircuitIR:
    return CircuitIR(
        n_wires=program.n_wires,
        instructions=(*program.instructions, Instruction(name="h", wires=(0,))),
    )


def test_a_non_positive_round_bound_is_refused() -> None:
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    with pytest.raises(ValueError, match="max_rounds"):
        manager.to_fixed_point(
            _ir(Instruction(name="h", wires=(0,))),
            pass_manager.OPTIMIZATION_PIPELINE,
            max_rounds=0,
        )


def test_an_unregistered_name_is_refused_before_any_pass_runs() -> None:
    # The counting pass is what proves the refusal happened first: asserting on the
    # input program would pass even if the first pass had already run, because a
    # pass returns a new program rather than mutating the one it is handed.
    first = _DropsEverything()
    program = _ir(Instruction(name="x", wires=(0,)), Instruction(name="x", wires=(0,)))
    registry = _registry_with("extension.example.first", first)
    manager = pass_manager.PassManager(registry)

    with pytest.raises(pass_manager.PassRegistryError, match="unknown pass"):
        manager.run(program, ("extension.example.first", "extension.example.missing"))

    assert first.calls == 0


def test_the_fixed_point_refuses_an_unregistered_name_before_any_pass_runs() -> None:
    first = _NoOp()
    program = _ir(Instruction(name="x", wires=(0,)))
    registry = _registry_with("extension.example.first", first)
    manager = pass_manager.PassManager(registry)

    with pytest.raises(pass_manager.PassRegistryError, match="unknown pass"):
        manager.to_fixed_point(
            program, ("extension.example.first", "extension.example.missing")
        )

    assert first.calls == 0


def test_a_program_that_is_not_ir_is_refused_before_any_pass_runs() -> None:
    first = _NoOp()
    registry = _registry_with("extension.example.first", first)
    manager = pass_manager.PassManager(registry)

    with pytest.raises(TypeError):
        manager.run("not a program", ("extension.example.first",))

    assert first.calls == 0


def test_the_refusal_names_the_passes_that_are_registered() -> None:
    registry = _registry_with("extension.example.named", _NoOp())

    with pytest.raises(pass_manager.PassRegistryError) as excinfo:
        registry.resolve("extension.example.other")

    assert "extension.example.named" in str(excinfo.value)


def test_a_pass_that_returns_a_non_program_is_refused_by_name() -> None:
    registry = _registry_with("extension.example.wrong", lambda ir: "not a program")
    manager = pass_manager.PassManager(registry)
    program = _ir(Instruction(name="h", wires=(0,)))

    with pytest.raises(CompilationError) as excinfo:
        manager.run(program, ("extension.example.wrong",))

    assert "extension.example.wrong" in str(excinfo.value)
    assert "str" in str(excinfo.value)


def test_the_same_refusal_covers_the_fixed_point_path() -> None:
    registry = _registry_with("extension.example.wrong", lambda ir: None)
    manager = pass_manager.PassManager(registry)

    with pytest.raises(CompilationError, match="extension.example.wrong"):
        manager.to_fixed_point(
            _ir(Instruction(name="h", wires=(0,))), ("extension.example.wrong",)
        )


def test_a_registered_name_cannot_be_shadowed() -> None:
    registry = _registry_with("extension.example.once", _NoOp())

    with pytest.raises(pass_manager.PassRegistryError, match="already registered"):
        registry.with_pass("extension.example.once", _NoOp())


@pytest.mark.parametrize(
    "name",
    [
        "remove_zero_state_resets",
        "remove_diagonal_gates_before_measure",
        "remove_identity_gates",
        "merge_self_inverse",
        "merge_inverse_pairs",
        "merge_adjacent_rotations",
        "cancel_commuting_self_inverse",
        "collapse_one_qubit_runs",
    ],
)
def test_a_built_in_pass_name_is_reserved(name: str) -> None:
    with pytest.raises(pass_manager.PassRegistryError, match="built-in"):
        pass_manager.PassRegistry().with_pass(name, _DropsEverything())


def test_the_reserved_set_is_exactly_the_pipeline_the_compiler_is_defined_by() -> None:
    # If `compile` gained a pass, that pass would be part of what `compile` means
    # and must be reserved too; this is the assertion that pairing is checked.
    registry = pipeline.default_pass_registry()

    assert frozenset(registry.names) == pass_manager.RESERVED_PASS_NAMES


@pytest.mark.parametrize("name", ["", "   "])
def test_an_empty_pass_name_is_refused(name: str) -> None:
    with pytest.raises(pass_manager.PassRegistryError, match="empty"):
        pass_manager.PassRegistry().with_pass(name, _NoOp())


@pytest.mark.parametrize("function", [None, 3, "pass"])
def test_a_non_callable_pass_is_refused(function: object) -> None:
    with pytest.raises(pass_manager.PassRegistryError, match="not callable"):
        pass_manager.PassRegistry().with_pass("extension.example.bad", function)  # type: ignore[arg-type]


def test_the_registry_is_immutable_and_additions_return_a_new_one() -> None:
    base = pass_manager.PassRegistry()

    added = base.with_pass("extension.example.added", _NoOp())

    assert base.names == ()
    assert added.names == ("extension.example.added",)


def test_the_registry_entries_cannot_be_mutated_through_the_mapping() -> None:
    registry = _registry_with("extension.example.frozen", _NoOp())

    with pytest.raises(TypeError):
        registry.passes["extension.example.other"] = _NoOp()  # type: ignore[index]


def test_an_extension_name_and_a_declared_name_may_not_contain_a_period() -> None:
    # A period would make the registered name ambiguous about which dot separates
    # the extension from the pass.
    with pytest.raises(pass_manager.PassRegistryError, match=r"may not contain"):
        pass_manager.extension_pass_name("has.dot", "pass")

    with pytest.raises(pass_manager.PassRegistryError, match=r"may not contain"):
        pass_manager.extension_pass_name("extension", "has.dot")


def test_the_registered_name_names_both_the_extension_and_the_pass() -> None:
    name = pass_manager.extension_pass_name("acme", "fuse")

    assert name == "extension.acme.fuse"
    assert name.startswith(pass_manager.EXTENSION_PASS_PREFIX)


def test_extension_names_are_told_apart_from_built_in_names() -> None:
    registry = pass_manager.PassRegistry().with_pass("extension.example.only", _NoOp())

    assert registry.extension_names == ("extension.example.only",)
    assert "remove_identity_gates" not in registry.extension_names
    assert registry.names == ("extension.example.only",)


def test_a_pass_registry_error_is_a_capability_error() -> None:
    # Same class as every other "this is not available" refusal, so one handler
    # catches both a missing pass and a missing backend.
    assert issubclass(pass_manager.PassRegistryError, CapabilityError)


def test_the_manager_runs_a_caller_supplied_sequence_without_the_pipeline() -> None:
    # The manager holds no sequence of its own, so a caller may name its own.
    program = ensure_circuit_ir(fq.Circuit(1).x(0).x(0))
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    result = manager.run(program, ("merge_self_inverse",))

    assert _names(result) == []


def test_the_public_pipeline_module_publishes_the_registry_entry_points() -> None:
    assert set(pipeline.__all__) >= {
        "BUILTIN_PASSES",
        "default_pass_registry",
        "optimize",
        "compile",
    }
    assert (
        pipeline.BUILTIN_PASSES["remove_identity_gates"]
        is pipeline.remove_identity_gates
    )


def test_every_built_in_binding_agrees_with_the_module_that_implements_it() -> None:
    # The registry adds a name, not a second implementation. Each bound pass must
    # produce exactly what its own module's function produces on the same program,
    # so the pipeline cannot drift away from the pass that owns the transformation.
    from flagquantum.compiler.commutation_cancellation import (
        cancel_commuting_self_inverse,
    )
    from flagquantum.compiler.inverse_cancellation import merge_inverse_pairs
    from flagquantum.compiler.one_qubit_optimization import collapse_one_qubit_runs
    from flagquantum.compiler.zero_state_reset import remove_zero_state_resets

    program = _ir(
        Instruction(name="x", wires=(0,)),
        Instruction(name="x", wires=(0,)),
        Instruction(name="reset", wires=(0,), metadata={"is_dynamic": True}),
        Instruction(name="h", wires=(1,)),
        Instruction(name="h", wires=(1,)),
    )
    expected = {
        "remove_zero_state_resets": remove_zero_state_resets,
        "merge_inverse_pairs": merge_inverse_pairs,
        "cancel_commuting_self_inverse": cancel_commuting_self_inverse,
        "collapse_one_qubit_runs": collapse_one_qubit_runs,
    }
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    for name, function in expected.items():
        assert manager.run(program, (name,)) == function(program), name


def test_a_built_in_binding_is_callable_with_one_program_argument() -> None:
    # `PassRegistry.resolve` returns a binding a caller that does not care about
    # analyses may call with one program argument. A binding that *needed* a second
    # argument, or that was a closure over something else, would fail here. The
    # manager is free to pass the facts a binding declared it reads, and does.
    program = _ir(Instruction(name="h", wires=(0,)))
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    for name in pipeline.BUILTIN_PASSES:
        assert isinstance(pipeline.BUILTIN_PASSES[name](program), CircuitIR), name
        assert isinstance(manager.run(program, (name,)), CircuitIR), name


def test_the_inline_passes_stay_in_the_module_that_owns_their_tables() -> None:
    # These three share `_SELF_INVERSE`, `_ROTATION_PARAM`, and `_WireLocalProgram`,
    # which are `pipeline`'s; a move would duplicate them. The two tables are also
    # what the registered analyses return, so the module that owns the table is the
    # module that owns the fact.
    for name in (
        "remove_identity_gates",
        "merge_self_inverse",
        "merge_adjacent_rotations",
    ):
        assert pipeline.BUILTIN_PASSES[name] is getattr(pipeline, name)


def test_optimize_and_compile_agree_with_the_manager() -> None:
    # The refactor is complete only if the public entry points still go through
    # the sequence rather than through a second copy of it.
    program = fq.Circuit(2).h(0).h(0).x(1).x(1)

    assert _names(pipeline.optimize(program)) == []
    assert _names(pipeline.compile(program)) == []


# The analysis half. A pass used to reach into whichever module happened to hold
# the fact it needed, so nothing could say what a pass read, a fact could be
# computed twice in one round, and no pass could be handed a different fact without
# editing the module that held it. These tests pin the four properties that
# replaces: the fact has a name, the name is resolved before anything runs, a
# program-independent fact is computed once for a whole run, and asking for a fact
# is what computes it.


def _declared_analysis_names() -> set[str]:
    """Every analysis the default registry's own pass specs declare."""

    registry = pipeline.default_pass_registry()
    declared: set[str] = set()
    for name in registry.names:
        spec = registry.spec(name)
        declared.update(spec.uses)
        declared.update(spec.preserves)
    return declared


def test_every_analysis_a_pass_declares_resolves_in_the_default_registry() -> None:
    registry = pipeline.default_pass_registry()

    for name in registry.names:
        spec = registry.spec(name)
        for analysis_name in (*spec.uses, *spec.preserves):
            registry.analysis_order(analysis_name)


def test_the_registry_registers_exactly_the_analyses_the_passes_declare() -> None:
    # A fact nobody reads is machinery with no consumer, and a fact a pass reads
    # without it being registered is a resolution that fails at run time. Both
    # directions are refused here rather than in a review.
    registry = pipeline.default_pass_registry()

    assert set(registry.analysis_names) == _declared_analysis_names()


def test_the_two_opcode_tables_are_the_program_independent_analyses() -> None:
    # Only an analysis whose value follows from the schema rather than from the
    # program may be carried across a transformation, and only these two are. The
    # program handed in differs from the values returned, which is the point.
    program = _ir(Instruction(name="h", wires=(0,)))
    registry = pipeline.default_pass_registry()

    manager = pass_manager.PassManager(registry)
    independent = {
        name
        for name in registry.analysis_names
        if registry.analysis(name).program_independent
    }

    assert independent == {"self_inverse_opcodes", "rotation_parameters"}
    assert manager.analyze(program, "self_inverse_opcodes") == pipeline._SELF_INVERSE
    assert manager.analyze(program, "rotation_parameters") == pipeline._ROTATION_PARAM


def test_an_analysis_that_needs_another_is_resolved_after_it() -> None:
    registry = (
        pass_manager.PassRegistry()
        .with_analysis("leaf", lambda program: "leaf")
        .with_analysis(
            "middle",
            lambda program, facts: ("middle", facts["leaf"]),
            requires=("leaf",),
        )
        .with_analysis(
            "top",
            lambda program, facts: ("top", facts["middle"]),
            requires=("middle",),
        )
    )

    assert registry.analysis_order("top") == ("leaf", "middle", "top")


def test_a_cycle_in_the_analysis_declarations_is_refused() -> None:
    registry = (
        pass_manager.PassRegistry()
        .with_analysis("a", lambda program, facts: facts["b"], requires=("b",))
        .with_analysis("b", lambda program, facts: facts["a"], requires=("a",))
    )

    with pytest.raises(pass_manager.PassRegistryError, match="cycle"):
        registry.analysis_order("a")


def test_a_program_independent_analysis_may_not_rest_on_a_program_derived_one() -> None:
    # A carried value stays true only if everything it was computed from stays
    # true, so the claim is refused rather than trusted.
    registry = (
        pass_manager.PassRegistry()
        .with_analysis("blocks", lambda program: program)
        .with_analysis(
            "cached_blocks",
            lambda program, facts: facts["blocks"],
            requires=("blocks",),
            program_independent=True,
        )
    )

    with pytest.raises(pass_manager.PassRegistryError, match="program-independent"):
        registry.analysis_order("cached_blocks")


def test_a_program_dependent_analysis_cannot_be_declared_preserved() -> None:
    registry = pipeline.default_pass_registry().with_pass(
        "extension.example.claims",
        lambda program: program,
        preserves=("commutation_blocks",),
    )
    manager = pass_manager.PassManager(registry)
    program = _ir(Instruction(name="h", wires=(0,)))

    with pytest.raises(pass_manager.PassRegistryError, match="preserve"):
        manager.run(program, ("extension.example.claims",))


def test_a_reader_of_a_carried_fact_must_declare_it_preserves_it() -> None:
    # The manager caches a program-independent fact for the whole run; a reader
    # that did not vouch for it would be relying on the cache silently.
    registry = pipeline.default_pass_registry().with_pass(
        "extension.example.reads",
        lambda program, _facts: program,
        uses=("self_inverse_opcodes",),
    )
    manager = pass_manager.PassManager(registry)
    program = _ir(Instruction(name="h", wires=(0,)))

    with pytest.raises(pass_manager.PassRegistryError, match="preserves"):
        manager.run(program, ("extension.example.reads",))


def test_an_unregistered_declared_analysis_is_refused_before_any_pass_runs() -> None:
    first = _NoOp()
    registry = (
        pipeline.default_pass_registry()
        .with_pass("extension.example.first", first)
        .with_pass(
            "extension.example.asking",
            lambda program, facts: program,
            uses=("nowhere_registered",),
        )
    )
    manager = pass_manager.PassManager(registry)
    program = _ir(Instruction(name="h", wires=(0,)))

    with pytest.raises(pass_manager.PassRegistryError) as excinfo:
        manager.run(program, ("extension.example.first", "extension.example.asking"))

    assert "extension.example.asking" in str(excinfo.value)
    assert first.calls == 0


def _snooping_pass(program: CircuitIR, facts: object) -> CircuitIR:
    """A pass that reads a fact it never declared, which the manager refuses."""

    assert isinstance(facts, Mapping)
    facts["commutation_blocks"]
    return program


def test_a_reader_cannot_reach_a_fact_it_did_not_declare() -> None:
    registry = pipeline.default_pass_registry().with_pass(
        "extension.example.snooping",
        _snooping_pass,
        uses=("self_inverse_opcodes",),
        preserves=("self_inverse_opcodes",),
    )
    manager = pass_manager.PassManager(registry)

    with pytest.raises(pass_manager.PassRegistryError, match="not declared"):
        manager.run(
            _ir(Instruction(name="h", wires=(0,))), ("extension.example.snooping",)
        )


def test_a_carried_fact_is_computed_once_for_a_whole_fixed_point_run() -> None:
    # `x(0) x(0) reset(0)` is the program that needs a second round, so one round
    # would not tell the two behaviours apart. The opcode table is asked for once
    # even though the sequence reads it in three passes across two rounds.
    program = _ir(
        Instruction(name="x", wires=(0,)),
        Instruction(name="x", wires=(0,)),
        _reset(0),
    )
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    calls = manager.calls(program, pass_manager.OPTIMIZATION_PIPELINE)

    assert calls["self_inverse_opcodes"] == 1
    assert calls["rotation_parameters"] == 1
    assert (
        _names(manager.to_fixed_point(program, pass_manager.OPTIMIZATION_PIPELINE))
        == []
    )


def test_a_program_derived_fact_is_computed_again_for_the_next_program() -> None:
    # A program-independent fact is carried because its value cannot change; a fact
    # that is a function of the program must not be, or the second program in a run
    # would be answered with the first one's blocks. Two readers separated by a
    # growing pass put two different programs in front of the same analysis run.
    seen: list[object] = []

    def reader(program: CircuitIR, facts: Mapping[str, object]) -> CircuitIR:
        seen.append(facts["commutation_blocks"])
        return program

    registry = (
        pipeline.default_pass_registry()
        .with_pass("extension.example.blocks", reader, uses=("commutation_blocks",))
        .with_pass("extension.example.grow", _grow)
    )
    manager = pass_manager.PassManager(registry)
    sequence = (
        "extension.example.grow",
        "extension.example.blocks",
        "extension.example.grow",
        "extension.example.blocks",
    )
    program = _ir(
        Instruction(name="cx", wires=(0, 1)),
        Instruction(name="rz", wires=(0,), params={"theta": 0.4}),
        Instruction(name="cx", wires=(0, 1)),
    )

    calls = manager.calls(program, sequence)

    assert len(seen) == 2
    assert calls["commutation_blocks"] == 2


def test_a_declared_fact_is_not_computed_when_the_reader_short_circuits() -> None:
    # `cancel_commuting_self_inverse` declares `commutation_blocks` and reads it
    # only after the repeat check passes. A circuit with no repeat costs no
    # commutation analysis at all, which is the short circuit the lazy resolution
    # exists to preserve.
    no_repeat = _ir(
        Instruction(name="x", wires=(0,)), Instruction(name="h", wires=(1,))
    )
    repeated = _ir(
        Instruction(name="cx", wires=(0, 1)),
        Instruction(name="rz", wires=(0,), params={"theta": 0.4}),
        Instruction(name="cx", wires=(0, 1)),
    )
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    quiet = manager.calls(no_repeat, ("cancel_commuting_self_inverse",))
    loud = manager.calls(repeated, ("cancel_commuting_self_inverse",))

    assert "commutation_blocks" not in quiet
    assert loud["commutation_blocks"] == 1


def test_the_manager_answers_an_analysis_the_way_the_pass_reads_it() -> None:
    from flagquantum.compiler.commutation import analyze_commutation

    program = _ir(
        Instruction(name="cx", wires=(0, 1)),
        Instruction(name="rz", wires=(0,), params={"theta": 0.4}),
        Instruction(name="cx", wires=(0, 1)),
    )
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    assert manager.analyze(program, "commutation_blocks") == analyze_commutation(
        program
    )


def test_a_replaced_fact_changes_what_a_pass_removes_without_touching_the_pass() -> (
    None
):
    # The replacement demonstration: `merge_self_inverse` is not modified, only
    # the registry it is handed, and the fact it reads changes what it removes.
    narrowed = pass_manager.AnalysisSpec(
        lambda program: frozenset(pipeline._SELF_INVERSE - {"x"}),
        program_independent=True,
    )
    registry = pass_manager.PassRegistry(
        pipeline.BUILTIN_PASSES,
        pipeline._BUILTIN_PASS_SPECS,
        {**pipeline._BUILTIN_ANALYSES, "self_inverse_opcodes": narrowed},
    )
    program = _ir(Instruction(name="x", wires=(0,)), Instruction(name="x", wires=(0,)))

    default = pass_manager.PassManager(pipeline.default_pass_registry())
    replaced = pass_manager.PassManager(registry)

    assert _names(default.run(program, ("merge_self_inverse",))) == []
    assert _names(replaced.run(program, ("merge_self_inverse",))) == ["x", "x"]
    # And the pass object itself is the same one in both registries.
    assert registry.resolve("merge_self_inverse") is pipeline.merge_self_inverse


def test_a_replaced_rotation_table_changes_what_remove_identity_gates_removes() -> None:
    # The same demonstration for the other carried fact, and for a pass defined in
    # the module rather than in another one: the table says which opcode's single
    # angle is `theta`, so narrowing it to nothing stops the zero-angle removal.
    narrowed = pass_manager.AnalysisSpec(
        lambda program: MappingProxyType({}), program_independent=True
    )
    registry = pass_manager.PassRegistry(
        pipeline.BUILTIN_PASSES,
        pipeline._BUILTIN_PASS_SPECS,
        {**pipeline._BUILTIN_ANALYSES, "rotation_parameters": narrowed},
    )
    program = _ir(
        Instruction(name="i", wires=(0,)),
        Instruction(name="rx", wires=(0,), params={"theta": 0.0}),
    )

    default = pass_manager.PassManager(pipeline.default_pass_registry())
    replaced = pass_manager.PassManager(registry)

    # The explicit identity gate is removed either way; only the zero-angle
    # rotation depends on the table.
    assert _names(default.run(program, ("remove_identity_gates",))) == []
    assert _names(replaced.run(program, ("remove_identity_gates",))) == ["rx"]


def test_a_replaced_self_inverse_set_changes_what_the_cancellation_pass_removes() -> (
    None
):
    # A pass whose fact lives in another module reads the declared one too, so the
    # replacement route reaches it without editing `commutation_cancellation`.
    narrowed = pass_manager.AnalysisSpec(
        lambda program: frozenset(pipeline._SELF_INVERSE - {"cx"}),
        program_independent=True,
    )
    registry = pass_manager.PassRegistry(
        pipeline.BUILTIN_PASSES,
        pipeline._BUILTIN_PASS_SPECS,
        {**pipeline._BUILTIN_ANALYSES, "self_inverse_opcodes": narrowed},
    )
    # `cx(0, 1) z(0) cx(0, 1)` is a bare `z(0)` only because the two `cx` cancel
    # across a proven commuting gap, and `cx` is exactly what was dropped.
    program = _ir(
        Instruction(name="cx", wires=(0, 1)),
        Instruction(name="z", wires=(0,)),
        Instruction(name="cx", wires=(0, 1)),
    )

    default = pass_manager.PassManager(pipeline.default_pass_registry())
    replaced = pass_manager.PassManager(registry)

    assert _names(default.run(program, ("cancel_commuting_self_inverse",))) == ["z"]
    assert _names(replaced.run(program, ("cancel_commuting_self_inverse",))) == [
        "cx",
        "z",
        "cx",
    ]
    # The registry binds the pipeline's wrapper rather than the pass itself, because
    # the pass imports `pipeline` and an eager import in this direction is a cycle.
    # What matters is that the wrapper forwards the facts, so the replacement route
    # reaches a pass in another module without editing either one.
    assert registry.resolve("cancel_commuting_self_inverse") is (
        pipeline._cancel_commuting_self_inverse
    )
    assert pipeline.BUILTIN_PASSES["cancel_commuting_self_inverse"] is (
        pipeline._cancel_commuting_self_inverse
    )


def test_a_declared_name_list_is_validated() -> None:
    # A declaration is the whole record of what a pass reads and vouches for, so a
    # duplicated name or a name that is not a string is refused where it is written
    # rather than at the point a resolution walks it.
    with pytest.raises(pass_manager.PassRegistryError, match="twice"):
        pass_manager.PassSpec(
            _noop, uses=("rotation_parameters", "rotation_parameters")
        )
    with pytest.raises(pass_manager.PassRegistryError, match="requires"):
        pass_manager.AnalysisSpec(_noop, requires=("",))


def test_a_declaration_that_names_itself_is_refused() -> None:
    # `_validated_declaration` is given the kind of declaration, not a registered
    # name, so the literal self-reference it catches is a name equal to that kind.
    # The case a reader would actually write -- an analysis requiring its own
    # registered name, or a pass naming itself in `uses` -- is caught one step
    # later by the resolution walk, and both messages name the offending entry.
    with pytest.raises(pass_manager.PassRegistryError, match="cannot declare itself"):
        pass_manager.PassSpec(_noop, uses=("a pass",))
    with pytest.raises(pass_manager.PassRegistryError, match="cannot declare itself"):
        pass_manager.AnalysisSpec(_noop, requires=("an analysis",))


def test_an_analysis_that_requires_its_own_name_is_refused_as_a_cycle() -> None:
    # The declaration itself is accepted: an analysis cannot know its registered
    # name, so the refusal has to come from the walk that sees the name.
    spec = pass_manager.AnalysisSpec(_noop, requires=("loop",))
    registry = pass_manager.PassRegistry().with_analysis(
        "loop", _noop, requires=spec.requires
    )

    assert spec.requires == ("loop",)
    assert registry.analysis("loop").requires == ("loop",)
    with pytest.raises(pass_manager.PassRegistryError, match="cycle"):
        registry.analysis_order("loop")


def test_a_pass_that_names_its_own_registered_name_is_refused() -> None:
    registry = pass_manager.PassRegistry().with_pass(
        "extension.example.self", _noop, uses=("extension.example.self",)
    )

    with pytest.raises(pass_manager.PassRegistryError) as raised:
        registry.resolve_sequence(("extension.example.self",))

    # Both the declaring pass and the name it could not resolve are in the message,
    # which is the difference between a usable refusal and a puzzle. Asserting the
    # joining text is what pins the connection: the name alone appears either way.
    assert "pass 'extension.example.self' declares 'extension.example.self'" in str(
        raised.value
    )
    assert "unknown analysis" in str(raised.value)


def test_an_analysis_declares_its_dependencies() -> None:
    with pytest.raises(pass_manager.PassRegistryError, match="twice"):
        pass_manager.AnalysisSpec(_noop, requires=("leaf", "leaf"))
    with pytest.raises(pass_manager.PassRegistryError, match="requires"):
        pass_manager.AnalysisSpec(_noop, requires=("",))


def test_a_spec_whose_function_is_a_different_object_than_the_pass_is_refused() -> None:
    # `resolve` returns one function and the manager calls another; a registry that
    # allowed those to differ would run a pass under a name it is not bound to.
    registry = pipeline.default_pass_registry()

    with pytest.raises(pass_manager.PassRegistryError, match="different function"):
        pass_manager.PassRegistry(
            pipeline.BUILTIN_PASSES,
            {
                **pipeline._BUILTIN_PASS_SPECS,
                "merge_self_inverse": pass_manager.PassSpec(lambda program: program),
            },
            pipeline._BUILTIN_ANALYSES,
        )
    assert registry.resolve("merge_self_inverse") is pipeline.merge_self_inverse


def _bump_rx(program: CircuitIR) -> CircuitIR:
    """Add one to an `rx` angle, leaving the instruction count alone.

    A same-length rewrite: the count comparison the fixed point used to make read
    this as "nothing changed" and returned a program the sequence would still edit.
    """

    instructions = []
    for instruction in program:
        if instruction.name == "rx" and instruction.params["theta"] < 2.0:
            params = dict(instruction.params)
            params["theta"] = params["theta"] + 1.0
            instruction = Instruction(
                name=instruction.name, wires=instruction.wires, params=params
            )
        instructions.append(instruction)
    return CircuitIR(n_wires=program.n_wires, instructions=tuple(instructions))


def test_a_same_length_rewrite_earns_another_round() -> None:
    program = _ir(Instruction(name="rx", wires=(0,), params={"theta": 0.0}))
    counted = _NoOp()
    registry = (
        pass_manager.PassRegistry()
        .with_pass("extension.example.bump", _bump_rx)
        .with_pass("extension.example.count", counted)
    )
    manager = pass_manager.PassManager(registry)

    # The default bound is one round per instruction plus one, which assumes a
    # round removes something; a same-length rewrite needs the bound raised.
    result = manager.to_fixed_point(
        program, ("extension.example.bump", "extension.example.count"), max_rounds=5
    )

    assert result.instructions[0].params["theta"] == 2.0
    # Two rounds that moved the angle, and one that found it where it left it.
    assert counted.calls == 3
