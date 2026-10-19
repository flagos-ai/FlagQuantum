"""A pass this repository does not ship runs inside the declared pipeline.

The optimization pipeline became a named sequence of registered passes in an
earlier slice, but the registry was Compiler-internal: a transformation written
outside this repository could be registered and run only by reaching into
``flagquantum.compiler.pass_manager``, a module the facade does not publish, and
neither entry point admitted a name of the caller's. These tests hold the claim
that closed it -- an admitted pass runs where the level's stages run, it is
recorded on the program it changed, an analysis behind it is queryable -- and the
claim that makes it safe -- a built-in name, a non-callable, a pass that never
settles, and an admission next to ``optimize=False`` are each refused before they
can run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import pytest

import flagquantum as fq
from flagquantum.compiler import (
    OPTIMIZATION_PIPELINE,
    RESERVED_PASS_NAMES,
    AnalysisSpec,
    PassManager,
    PassRegistry,
    PassRegistryError,
    compile,
    default_pass_registry,
    extension_pass_name,
    optimize,
)
from flagquantum.compiler import pass_manager as pass_manager_module
from flagquantum.compiler import pipeline as pipeline_module
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CapabilityError, CompilationError

pytestmark = pytest.mark.unit

ADMITTED = "acme.lower_x"


def _names(program: CircuitIR) -> list[str]:
    return [held.name for held in program.instructions]


def _lower_x(program: CircuitIR) -> CircuitIR:
    """A pass this repository does not ship: rewrite `x` as `rx(pi)`.

    It is idempotent, which is what a pass running inside a fixed-point loop has to
    be, and it is a transformation no built-in stage performs, so a program it
    touched is distinguishable from one it did not.
    """

    return replace(
        program,
        instructions=tuple(
            (
                Instruction("rx", held.wires, {"theta": math.pi})
                if held.name == "x"
                else held
            )
            for held in program.instructions
        ),
    )


class _Counting:
    """Wrap a pass and count how many rounds it actually ran."""

    def __init__(self) -> None:
        self.rounds = 0

    def __call__(self, program: CircuitIR) -> CircuitIR:
        self.rounds += 1
        return _lower_x(program)


def test_an_admitted_pass_runs_inside_the_declared_pipeline() -> None:
    program = fq.Circuit(1).x(0).to_ir()

    out = optimize(program, optimization_level=0, passes={ADMITTED: _lower_x})

    assert _names(out) == ["rx"]
    assert _names(program) == ["x"]


def test_the_level_runs_and_the_admitted_pass_runs_after_it() -> None:
    """Both halves of the request are visible in one result.

    Level 1 removes the `h` pair and nothing else, so `x` survives it; the admitted
    pass then rewrites `x`. A result carrying both effects is the only shape that
    proves the admitted pass joined the pipeline rather than replacing it.
    """

    program = fq.Circuit(1).h(0).h(0).x(0).to_ir()

    built_in_only = optimize(program, optimization_level=1)
    admitted = optimize(program, optimization_level=1, passes={ADMITTED: _lower_x})

    assert _names(built_in_only) == ["x"]
    assert _names(admitted) == ["rx"]


def test_level_zero_runs_the_admitted_pass_and_nothing_else() -> None:
    """A level with an empty sequence still runs the passes the caller admitted."""

    program = fq.Circuit(1).h(0).h(0).to_ir()

    out = optimize(program, optimization_level=0, passes={ADMITTED: _lower_x})

    assert _names(out) == ["h", "h"]


def test_the_admitted_names_are_recorded_on_the_program() -> None:
    program = fq.Circuit(1).x(0).to_ir()

    out = optimize(program, optimization_level=0, passes={ADMITTED: _lower_x})

    assert out.metadata["admitted_passes"] == [ADMITTED]


def test_a_program_that_ran_only_built_in_stages_carries_no_admission_field() -> None:
    """The record appears when there is something to say and not before."""

    out = optimize(fq.Circuit(1).h(0).h(0).to_ir(), optimization_level=1)

    assert "admitted_passes" not in out.metadata
    assert out.metadata["optimization"] == {"level": 1}


def test_the_level_record_is_unchanged_by_admission() -> None:
    """Admission adds a field; it does not rewrite the level the caller asked for."""

    out = optimize(
        fq.Circuit(1).x(0).to_ir(),
        optimization_level=1,
        passes={ADMITTED: _lower_x},
    )

    assert out.metadata["optimization"] == {"level": 1}
    assert out.metadata["admitted_passes"] == [ADMITTED]


@pytest.mark.parametrize("reserved", sorted(RESERVED_PASS_NAMES))
def test_a_built_in_pass_name_cannot_be_taken(reserved: str) -> None:
    """Every name the level is defined by is refused, not merely some of them."""

    with pytest.raises(PassRegistryError) as caught:
        optimize(
            fq.Circuit(1).x(0).to_ir(),
            optimization_level=1,
            passes={reserved: _lower_x},
        )

    assert reserved in str(caught.value)


def test_a_name_bound_to_a_non_callable_is_refused() -> None:
    with pytest.raises(PassRegistryError):
        optimize(
            fq.Circuit(1).x(0).to_ir(),
            optimization_level=1,
            passes={ADMITTED: object()},  # type: ignore[dict-item]
        )


def test_an_empty_name_is_refused() -> None:
    with pytest.raises(PassRegistryError):
        optimize(
            fq.Circuit(1).x(0).to_ir(), optimization_level=1, passes={"": _lower_x}
        )


def test_a_pass_that_returns_something_else_is_refused_by_name() -> None:
    def _broken(program: CircuitIR) -> CircuitIR:
        return None  # type: ignore[return-value]

    with pytest.raises(CompilationError) as caught:
        optimize(
            fq.Circuit(1).x(0).to_ir(),
            optimization_level=1,
            passes={"acme.broken": _broken},
        )

    assert "acme.broken" in str(caught.value)


def test_a_pass_that_never_settles_is_refused() -> None:
    """A pass that is not idempotent is held to the same verdict as a built-in."""

    def _grows(program: CircuitIR) -> CircuitIR:
        return replace(
            program,
            instructions=(*program.instructions, Instruction("h", (0,))),
        )

    with pytest.raises(CompilationError) as caught:
        optimize(
            fq.Circuit(1).x(0).to_ir(),
            optimization_level=1,
            passes={"acme.grows": _grows},
        )

    assert "fixed point" in str(caught.value)


def test_compile_forwards_the_passes_to_every_point_it_optimizes() -> None:
    """Routing inserts work, so the admitted pass runs on both sides of it."""

    counter = _Counting()
    program = fq.Circuit(3).cx(0, 2).to_ir()

    out = compile(
        program,
        coupling_map=[(0, 1), (1, 2)],
        optimization_level=1,
        passes={ADMITTED: counter},
    )

    assert counter.rounds >= 2
    assert "rx" not in _names(out)
    assert out.metadata["admitted_passes"] == [ADMITTED]


def test_compile_refuses_admission_while_optimization_is_switched_off() -> None:
    """A request for a pass next to a request for no passes is refused, not ignored."""

    with pytest.raises(ValueError) as caught:
        compile(
            fq.Circuit(1).x(0).to_ir(),
            optimize=False,
            passes={ADMITTED: _lower_x},
        )

    assert "optimization is switched off" in str(caught.value)


def test_the_registry_and_the_manager_are_the_same_objects_at_both_paths() -> None:
    """The facade publishes the authority; it does not copy it."""

    assert default_pass_registry is pipeline_module.default_pass_registry
    assert PassManager is pass_manager_module.PassManager
    assert PassRegistry is pass_manager_module.PassRegistry
    assert PassRegistryError is pass_manager_module.PassRegistryError
    assert AnalysisSpec is pass_manager_module.AnalysisSpec
    assert extension_pass_name is pass_manager_module.extension_pass_name
    assert OPTIMIZATION_PIPELINE == pass_manager_module.OPTIMIZATION_PIPELINE
    assert pass_manager_module.PassRegistryError is not CapabilityError


def test_an_analysis_result_is_queryable_by_a_caller() -> None:
    """The other half of the registry: asking it for a fact about a program."""

    registry = default_pass_registry().with_analysis(
        "acme.opcodes",
        lambda program: frozenset(held.name for held in program.instructions),
    )
    manager = PassManager(registry)
    program = fq.Circuit(1).h(0).x(0).to_ir()

    assert manager.analyze(program, "acme.opcodes") == frozenset({"h", "x"})


def test_a_registered_pass_reads_the_analysis_it_declared() -> None:
    """A pass with a declaration is registered on the registry, not through `passes`."""

    seen: list[frozenset[str]] = []

    def _reads(program: CircuitIR, analyses: Mapping[str, object]) -> CircuitIR:
        seen.append(analyses["acme.opcodes"])  # type: ignore[arg-type]
        return program

    registry = default_pass_registry().with_analysis(
        "acme.opcodes",
        lambda program: frozenset(held.name for held in program.instructions),
    )
    registry = registry.with_pass("acme.reads", _reads, uses=("acme.opcodes",))
    program = fq.Circuit(1).h(0).x(0).to_ir()

    PassManager(registry).run(program, ("acme.reads",))

    assert seen == [frozenset({"h", "x"})]


def test_a_program_independent_analysis_is_computed_once_for_a_whole_run() -> None:
    """The carried fact is the reason a declaration is worth making.

    The pass changes the program on the first round and returns it unchanged after
    that, so the loop runs at least twice and the pass is applied at least four
    times; a fact declared program-independent is computed once for all of it.
    """

    computed: list[int] = []

    def _rotation_opcodes(_program: CircuitIR) -> frozenset[str]:
        computed.append(1)
        return frozenset({"rx"})

    registry = default_pass_registry().with_analysis(
        "acme.rotation_opcodes", _rotation_opcodes, program_independent=True
    )

    def _keeps(program: CircuitIR, analyses: Mapping[str, object]) -> CircuitIR:
        assert "rx" in analyses["acme.rotation_opcodes"]  # type: ignore[operator]
        return _lower_x(program)

    registry = registry.with_pass(
        "acme.keeps",
        _keeps,
        uses=("acme.rotation_opcodes",),
        preserves=("acme.rotation_opcodes",),
    )
    manager = PassManager(registry)

    out = manager.to_fixed_point(fq.Circuit(1).x(0).to_ir(), ("acme.keeps",))

    assert _names(out) == ["rx"]
    assert len(computed) == 1


def test_a_program_derived_analysis_is_recomputed_when_the_program_changes() -> None:
    """The opposite declaration: a fact about the program is not carried."""

    computed: list[int] = []

    def _opcodes(program: CircuitIR) -> frozenset[str]:
        computed.append(1)
        return frozenset(held.name for held in program.instructions)

    def _reads(program: CircuitIR, analyses: Mapping[str, object]) -> CircuitIR:
        analyses["acme.opcodes"]
        return program

    registry = default_pass_registry().with_analysis("acme.opcodes", _opcodes)
    registry = registry.with_pass(
        "acme.reads",
        _reads,
        uses=("acme.opcodes",),
    )

    PassManager(registry).run(fq.Circuit(1).h(0).to_ir(), ("acme.reads", "acme.reads"))

    assert len(computed) == 2


def test_a_pass_reading_a_carried_fact_without_preserving_it_is_refused() -> None:
    """The declaration is checked, so the cache cannot be trusted by accident."""

    registry = default_pass_registry().with_analysis(
        "acme.rotation_opcodes",
        lambda: frozenset({"rx"}),
        program_independent=True,
    )
    registry = registry.with_pass(
        "acme.reads_only",
        lambda program, _analyses: program,
        uses=("acme.rotation_opcodes",),
    )

    with pytest.raises(PassRegistryError) as caught:
        PassManager(registry).run(fq.Circuit(1).h(0).to_ir(), ("acme.reads_only",))

    assert "acme.rotation_opcodes" in str(caught.value)
