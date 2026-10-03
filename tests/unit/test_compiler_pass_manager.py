"""The optimization pipeline is a named, replaceable sequence of registered passes.

The pipeline used to be nine inline calls in `pipeline._optimize_to_fixed_point`,
and these tests exist so that it stays a nameable sequence: every pass the
pipeline runs resolves in the default registry, the order is the order the inline
loop documented, and the manager reproduces the loop's fixed-point semantics. The
last group pins the two properties that make an extension pass possible at all --
a name resolves to a pass, and a built-in name cannot be taken.
"""

from __future__ import annotations

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
    # The manager calls a pass as `function(ir)`. A binding that needed a second
    # argument, or that was a closure over something else, would fail here.
    program = _ir(Instruction(name="h", wires=(0,)))
    manager = pass_manager.PassManager(pipeline.default_pass_registry())

    for name in pipeline.BUILTIN_PASSES:
        assert isinstance(manager.run(program, (name,)), CircuitIR), name


def test_the_inline_passes_stay_in_the_module_that_owns_their_tables() -> None:
    # These three share `_SELF_INVERSE`, `_ROTATION_PARAM`, and `_WireLocalProgram`,
    # which are `pipeline`'s; a move would duplicate them.
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
