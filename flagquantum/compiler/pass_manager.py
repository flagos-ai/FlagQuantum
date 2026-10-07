"""Named, replaceable circuit-optimization passes, their analyses, and the manager.

The optimization pipeline used to be nine inline calls inside
``pipeline._optimize_to_fixed_point``. That spelling named no pass, so a pass could
not be replaced, listed, or supplied from outside the repository, and the Compiler
extension protocol's ``compiler_pass`` kind had no consumer. This module supplies
the machinery that makes a pass a name: a registry, a sequence, and a manager that
runs one over a program.

Transformation and analysis are told apart here. A *pass* returns a program; an
*analysis* returns a fact about one and changes nothing. Several of the facts the
optimizer needs were already written as plain functions -- ``analyze_commutation``
returns a commuting-block partition, ``cancellable_positions`` groups the pairs a
pass may remove -- but nothing named them, so each consumer reached into the module
that happened to hold one, no caller could ask what a pass reads before it ran, and
no caller could ask for a fact twice without computing it twice. An analysis is now
a registry entry with its own dependencies, and a pass declares which analyses it
reads and which of them it leaves valid.

A pass keeps the one-argument shape. The manager calls a registered pass as
``function(program)`` when it declares no analysis and as
``function(program, analyses)`` when it declares one, so an in-repository pass and
an admitted extension pass are still called the same way by every caller that does
not care about analyses, and ``ecosystem.extensions.pass_admission`` needs no
second convention. A pass that wants its facts takes ``analyses=None`` and falls
back to the module table the analysis reads, which is the same object.

Two properties are enforced rather than trusted. A declared analysis is resolved
lazily, at the moment the reader asks for it, so a pass that declares a fact and
does not read it pays nothing for it -- which is what lets an analysis carry its
own short circuit. And an analysis that is a function of the program cannot be
preserved by a transformation, because the transformation may return a different
program: the manager refuses the claim instead of trusting it. The converse is
required too, so the value the manager carries from one round to the next is one
the reader vouched for rather than a hidden assumption.

Fail closed, in three places. A sequence is resolved in full before the first pass
runs, so a mistyped name cannot leave a half-applied pipeline behind. A declared
analysis that is not registered is refused before anything runs. And the built-in
names are reserved: a registry refuses to shadow one, so an extension cannot
silently change what ``compile`` does.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from ..core.ir import CircuitIR, ensure_circuit_ir
from ..errors import CapabilityError, CompilationError

PassFunction = Callable[[CircuitIR], CircuitIR]
"""A transformation: one program in, one program out."""

AnalysisFunction = Callable[..., object]
"""A read-only fact: a program in, and the facts it declared it needs."""

AnalysisConsumer = Callable[[CircuitIR, Mapping[str, object]], CircuitIR]
"""A pass that reads the analyses it declares.

It is called with the program and a lazily resolved mapping from each analysis it
named to that analysis' value. Reading a key computes it; leaving it unread costs
nothing.
"""

PassBinding = PassFunction | AnalysisConsumer
"""What a name in a registry resolves to.

A pass that declares no analysis is a `PassFunction` and may be called with the
program alone; a pass that declares some is an `AnalysisConsumer` and is called
with the facts it named. The two are one field because a name resolves to the
implementation of the pass, and `PassSpec.function` is that same object -- a
registry refuses a spec and a binding that disagree.
"""

# An extension pass is registered under this prefix and nothing else may use it.
# The prefix is what lets `PassRegistry` tell "a name a plugin supplied" from "a
# name this repository guarantees".
EXTENSION_PASS_PREFIX = "extension."

# One round of the fixed optimization pipeline. `remove_identity_gates` appears
# twice because the passes between the two calls expose new identity pairs, which
# is a property of the sequence rather than of the pass. Those passes are also the
# reserved names: exactly the names `compile` is defined by, and so exactly the
# names an extension may not redefine.
OPTIMIZATION_PIPELINE: tuple[str, ...] = (
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

RESERVED_PASS_NAMES = frozenset(OPTIMIZATION_PIPELINE)


class PassRegistryError(CapabilityError):
    """A pass or analysis name cannot be registered or resolved as declared."""


def _validated_name(name: object, kind: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise PassRegistryError(f"{kind} name cannot be empty")
    return name.strip()


def _validated_callable(name: str, function: object, kind: str) -> None:
    if not callable(function):
        raise PassRegistryError(
            f"{kind} {name!r} is not callable: {type(function).__name__}"
        )


def _validated_declaration(
    owner: str, field_name: str, names: Sequence[str]
) -> tuple[str, ...]:
    """Check a declared name list: non-empty strings, no repeats, no self-reference."""

    resolved: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise PassRegistryError(f"{owner} declares an empty name in {field_name}")
        candidate = name.strip()
        if candidate == owner:
            raise PassRegistryError(f"{owner} cannot declare itself in {field_name}")
        if candidate in resolved:
            raise PassRegistryError(
                f"{owner} declares {candidate!r} twice in {field_name}"
            )
        resolved.append(candidate)
    return tuple(resolved)


def _validated_program(program: object) -> CircuitIR:
    return ensure_circuit_ir(program)


def _expect_ir(function: Callable[..., object], result: object, name: str) -> CircuitIR:
    if not isinstance(result, CircuitIR):
        raise CompilationError(
            f"pass {name!r} returned {type(result).__name__}; expected CircuitIR"
        )
    return result


@dataclass(frozen=True)
class AnalysisSpec:
    """One registered analysis: how to compute it and what it needs first.

    ``program_independent`` is the analysis' own claim that its value follows from
    the operator schema and not from the program it is handed. It is the only kind
    of analysis a pass may declare it preserves, and the only kind the manager
    carries from one round of the pipeline to the next.
    """

    function: AnalysisFunction
    requires: tuple[str, ...] = ()
    program_independent: bool = False

    def __post_init__(self) -> None:
        _validated_callable("<analysis>", self.function, "analysis")
        object.__setattr__(
            self,
            "requires",
            _validated_declaration("an analysis", "requires", self.requires),
        )
        if not isinstance(self.program_independent, bool):
            raise PassRegistryError("program_independent must be a bool")


@dataclass(frozen=True)
class PassSpec:
    """One registered pass: what it declares about the pipeline.

    ``uses`` names the analyses whose values the pass is called with, and
    ``preserves`` names the analyses the pass leaves valid. A pass that reads a
    program-independent analysis must also declare it preserves it: the manager
    caches such a value across the whole pipeline run, and the declaration is the
    reader's statement that the cache stays true.
    """

    function: PassFunction | AnalysisConsumer
    uses: tuple[str, ...] = ()
    preserves: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validated_callable("<pass>", self.function, "pass")
        for attribute in ("uses", "preserves"):
            object.__setattr__(
                self,
                attribute,
                _validated_declaration("a pass", attribute, getattr(self, attribute)),
            )


@dataclass(frozen=True)
class PassRegistry:
    """Immutable name-to-pass map. Additions return a new registry."""

    passes: Mapping[str, PassBinding] = field(default_factory=dict)
    specs: Mapping[str, PassSpec] = field(default_factory=dict)
    analyses: Mapping[str, AnalysisSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        resolved: dict[str, PassBinding] = {}
        for name, function in dict(self.passes).items():
            candidate = _validated_name(name, "pass")
            _validated_callable(candidate, function, "pass")
            resolved[candidate] = function

        # A pass registered without metadata still has a spec: the empty one. That
        # keeps `PassRegistry({"name": function})` a complete registry and makes
        # `spec` total, so no caller has to special-case "registered but undeclared".
        declared: dict[str, PassSpec] = {}
        for name, spec in dict(self.specs).items():
            candidate = _validated_name(name, "pass")
            if candidate not in resolved:
                raise PassRegistryError(
                    f"pass spec {candidate!r} has no registered pass"
                )
            if not isinstance(spec, PassSpec):
                raise PassRegistryError(
                    f"pass spec {candidate!r} is {type(spec).__name__}, not PassSpec"
                )
            # `resolve` returns one function and the manager calls another; a
            # registry where those disagree would run a pass under a name it is
            # not registered as.
            if spec.function is not resolved[candidate]:
                raise PassRegistryError(
                    f"pass spec {candidate!r} names a different function than the "
                    "one registered under that name"
                )
            declared[candidate] = spec
        for name, function in resolved.items():
            declared.setdefault(name, PassSpec(function))

        analyses: dict[str, AnalysisSpec] = {}
        for name, analysis_spec in dict(self.analyses).items():
            candidate = _validated_name(name, "analysis")
            if not isinstance(analysis_spec, AnalysisSpec):
                raise PassRegistryError(
                    f"analysis {candidate!r} is {type(analysis_spec).__name__}, not "
                    "AnalysisSpec"
                )
            analyses[candidate] = analysis_spec

        object.__setattr__(self, "passes", MappingProxyType(resolved))
        object.__setattr__(self, "specs", MappingProxyType(declared))
        object.__setattr__(self, "analyses", MappingProxyType(analyses))

    def with_pass(
        self,
        name: str,
        function: PassFunction | AnalysisConsumer,
        *,
        uses: Sequence[str] = (),
        preserves: Sequence[str] = (),
    ) -> "PassRegistry":
        """Return a registry that also resolves ``name``.

        Raises:
            PassRegistryError: The name is empty, already taken, or reserved for a
                built-in pass; the function is not callable; or a declared name list
                is malformed.
        """

        candidate = _validated_name(name, "pass")
        _validated_callable(candidate, function, "pass")
        if candidate in self.passes:
            raise PassRegistryError(
                f"pass {candidate!r} is already registered; a live pass is never "
                "displaced silently"
            )
        if candidate in RESERVED_PASS_NAMES:
            raise PassRegistryError(
                f"pass {candidate!r} is a built-in pass name; an extension must "
                f"register under {EXTENSION_PASS_PREFIX!r} so that the built-in "
                "meaning of a name cannot be redefined"
            )
        pass_spec = PassSpec(function, tuple(uses), tuple(preserves))
        return PassRegistry(
            {**self.passes, candidate: function},
            {**self.specs, candidate: pass_spec},
            self.analyses,
        )

    def with_analysis(
        self,
        name: str,
        function: AnalysisFunction,
        *,
        requires: Sequence[str] = (),
        program_independent: bool = False,
    ) -> "PassRegistry":
        """Return a registry that also resolves the analysis ``name``.

        Raises:
            PassRegistryError: The name is empty or already registered, the function
                is not callable, or a declared dependency is malformed.
        """

        candidate = _validated_name(name, "analysis")
        if candidate in self.analyses:
            raise PassRegistryError(
                f"analysis {candidate!r} is already registered; a live analysis is "
                "never displaced silently"
            )
        spec = AnalysisSpec(function, tuple(requires), program_independent)
        return PassRegistry(self.passes, self.specs, {**self.analyses, candidate: spec})

    def resolve(self, name: str) -> PassBinding:
        """Return the function registered under ``name``.

        The return type is the union because a pass that declares analyses is
        called with them; a caller that supplies none is calling a pass whose
        `PassSpec.uses` is empty, and `PassSpec.uses` is what says so.

        Raises:
            PassRegistryError: No pass is registered under that name.
        """

        try:
            return self.passes[name]
        except KeyError as exc:
            raise PassRegistryError(
                f"unknown pass {name!r}; registered passes are "
                + ", ".join(sorted(self.passes))
            ) from exc

    def spec(self, name: str) -> PassSpec:
        """Return what the pass ``name`` declares about the pipeline.

        Raises:
            PassRegistryError: No pass is registered under that name.
        """

        self.resolve(name)
        return self.specs[name]

    def analysis(self, name: str) -> AnalysisSpec:
        """Return the registered analysis ``name``.

        Raises:
            PassRegistryError: No analysis is registered under that name.
        """

        try:
            return self.analyses[name]
        except KeyError as exc:
            raise PassRegistryError(
                f"unknown analysis {name!r}; registered analyses are "
                + ", ".join(sorted(self.analyses))
            ) from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.passes))

    @property
    def analysis_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.analyses))

    @property
    def extension_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.names if name.startswith(EXTENSION_PASS_PREFIX)
        )

    def analysis_order(self, name: str) -> tuple[str, ...]:
        """Return ``name`` and every analysis it needs, dependencies first.

        The result is the dependency graph as a sequence: an analysis appears after
        everything it declares a dependency on, so resolving the result in order
        makes every dependency available before its consumer asks for it.

        Raises:
            PassRegistryError: ``name`` is not registered, a dependency is not
                registered, or the declarations contain a cycle.
        """

        ordered: list[str] = []
        on_path: set[str] = set()

        def visit(current: str, path: tuple[str, ...]) -> None:
            if current in ordered:
                return
            if current in on_path:
                raise PassRegistryError(
                    "analysis dependencies contain a cycle: "
                    + " -> ".join((*path, current))
                )
            spec = self.analysis(current)
            # A carried value stays true only if everything it was computed from
            # stays true, so a program-independent analysis may not rest on one
            # that is a function of the program.
            for dependency in spec.requires:
                if (
                    spec.program_independent
                    and not self.analysis(dependency).program_independent
                ):
                    raise PassRegistryError(
                        f"analysis {current!r} is program-independent but depends on "
                        f"{dependency!r}, which is not"
                    )
            on_path.add(current)
            for dependency in spec.requires:
                visit(dependency, (*path, current))
            on_path.discard(current)
            ordered.append(current)

        visit(name, ())
        return tuple(ordered)

    def resolve_sequence(
        self, sequence: Sequence[str]
    ) -> tuple[tuple[str, PassSpec], ...]:
        """Return the specs of ``sequence`` after checking everything it declares.

        Raises:
            PassRegistryError: A name is not registered, a declared analysis is not
                registered, its dependencies are unsatisfiable, or a pass claims to
                preserve an analysis it cannot prove survives it.
        """

        resolved: list[tuple[str, PassSpec]] = []
        for name in sequence:
            spec = self.spec(name)
            for analysis_name in (*spec.uses, *spec.preserves):
                try:
                    self.analysis_order(analysis_name)
                except PassRegistryError as exc:
                    # Naming the declaring pass costs nothing here and is the
                    # difference between a usable refusal and a puzzle.
                    raise PassRegistryError(
                        f"pass {name!r} declares {analysis_name!r}: {exc}"
                    ) from exc
            for analysis_name in spec.preserves:
                if not self.analysis(analysis_name).program_independent:
                    raise PassRegistryError(
                        f"pass {name!r} claims to preserve {analysis_name!r}, which "
                        "is a function of the program; a transformation may return a "
                        "different program and cannot prove that survives it"
                    )
            for analysis_name in spec.uses:
                if self.analysis(analysis_name).program_independent and (
                    analysis_name not in spec.preserves
                ):
                    raise PassRegistryError(
                        f"pass {name!r} reads the program-independent analysis "
                        f"{analysis_name!r} without declaring that it preserves it"
                    )
            resolved.append((name, spec))
        return tuple(resolved)


class _AnalysisRun:
    """The analyses one pipeline run has established, and how often each was asked.

    An analysis is resolved when a reader asks for it, not before. That is what
    lets an analysis carry a short circuit -- a fact that is expensive on the
    circuits it has nothing to say about stays uncomputed for a reader that never
    reaches the point of asking. A program-independent analysis is computed once
    and carried for the rest of the run: its value cannot change when a pass returns
    a different program, and a pass only reads one if it declared that it preserves
    it.
    """

    def __init__(self, registry: PassRegistry) -> None:
        self._registry = registry
        self._carried: dict[str, object] = {}
        self._calls: dict[str, int] = {}

    @property
    def calls(self) -> Mapping[str, int]:
        """How often each analysis was computed, for evidence and tests."""

        return MappingProxyType(dict(self._calls))

    def value(self, name: str, program: CircuitIR) -> object:
        """Return the value of ``name`` for ``program``, carrying it when it can.

        Raises:
            PassRegistryError: ``name`` is not registered, or its declared
                dependencies are unsatisfiable.
        """

        self._registry.analysis_order(name)
        spec = self._registry.analysis(name)
        if not spec.program_independent:
            return self._compute(name, spec, program)
        if name not in self._carried:
            self._carried[name] = self._compute(name, spec, program)
        return self._carried[name]

    def _compute(self, name: str, spec: AnalysisSpec, program: CircuitIR) -> object:
        self._calls[name] = self._calls.get(name, 0) + 1
        if not spec.requires:
            return spec.function(program)
        return spec.function(program, _DeclaredFacts(self, spec.requires, program))

    def facts_for(
        self, names: Sequence[str], program: CircuitIR
    ) -> Mapping[str, object]:
        """Return a mapping that resolves each of ``names`` on first read."""

        return _DeclaredFacts(self, tuple(names), program)

    def for_pass(self, spec: PassSpec, program: CircuitIR) -> Mapping[str, object]:
        """Return the analyses ``spec`` reads, resolved on first read."""

        if not spec.uses:
            return MappingProxyType({})
        return _DeclaredFacts(self, spec.uses, program)


class _DeclaredFacts(Mapping[str, object]):
    """The facts a reader declared it needs, resolved the first time each is read.

    The mapping is the whole interface between an analysis or a pass and the facts
    it declared: it answers exactly the declared names and raises on anything else,
    so a reader cannot reach a fact it never declared and no caller has to know how
    the facts are ordered.
    """

    def __init__(
        self, run: _AnalysisRun, names: Sequence[str], program: CircuitIR
    ) -> None:
        self._run = run
        self._program = program
        self._names = tuple(names)
        self._resolved: dict[str, object] = {}

    def __getitem__(self, name: str) -> object:
        if name not in self._names:
            raise PassRegistryError(
                f"{name!r} was not declared; the declared facts are "
                + (", ".join(self._names) if self._names else "none")
            )
        if name not in self._resolved:
            self._resolved[name] = self._run.value(name, self._program)
        return self._resolved[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"_DeclaredFacts({self._names!r})"


@dataclass(frozen=True)
class PassManager:
    """Run an ordered sequence of registered passes over one program.

    The manager holds no pass body and no sequence of its own: both arrive from the
    caller, so the same manager serves the fixed optimization pipeline and a caller
    that names its own passes.
    """

    registry: PassRegistry

    def analyze(self, program: CircuitIR, name: str) -> object:
        """Return the analysis ``name`` of ``program``, dependencies resolved first.

        This is what a pass reads, asked directly. A benchmark can report the reach
        a pass had before deciding whether to run it, and a test can hold a pass's
        own table to the fact the manager hands it.

        Raises:
            PassRegistryError: No analysis is registered under that name, or its
                declared dependencies are unsatisfiable.
        """

        return _AnalysisRun(self.registry).value(name, _validated_program(program))

    def calls(self, program: CircuitIR, sequence: Sequence[str]) -> Mapping[str, int]:
        """Run ``sequence`` and report how often each analysis was computed.

        Raises:
            PassRegistryError: The sequence does not resolve.
            CompilationError: A pass returned something that is not a CircuitIR.
        """

        resolved = self.registry.resolve_sequence(sequence)
        ir = _validated_program(program)
        run = _AnalysisRun(self.registry)
        for name, spec in resolved:
            ir = self._applied(name, spec, ir, run)
        return run.calls

    def run(self, program: CircuitIR, sequence: Sequence[str]) -> CircuitIR:
        """Apply ``sequence`` in order, resolving every name before the first pass.

        Raises:
            PassRegistryError: A name in the sequence is not registered, a declared
                analysis is not registered or unsatisfiable, or a pass claims a
                preservation it cannot prove. Nothing has been applied when raised.
            CompilationError: A pass returned something that is not a CircuitIR.
        """

        resolved = self.registry.resolve_sequence(sequence)
        ir = _validated_program(program)
        run = _AnalysisRun(self.registry)
        for name, spec in resolved:
            ir = self._applied(name, spec, ir, run)
        return ir

    def to_fixed_point(
        self,
        program: CircuitIR,
        sequence: Sequence[str],
        *,
        max_rounds: int | None = None,
    ) -> CircuitIR:
        """Apply ``sequence`` repeatedly until the program stops changing.

        A round that returns the program it was handed is a fixed point. The
        comparison is the program, not its length: a pass may rewrite an
        instruction in place, and a length comparison reads that as "nothing
        changed" and returns a program the sequence would still have changed. The
        bound defaults to one round per instruction plus one.

        Raises:
            PassRegistryError: The sequence does not resolve.
            CompilationError: A pass returned something that is not a CircuitIR, or
                no fixed point was reached inside ``max_rounds``.
            ValueError: ``max_rounds`` is not positive.
        """

        resolved = self.registry.resolve_sequence(sequence)
        ir = _validated_program(program)
        limit = len(ir) + 1 if max_rounds is None else int(max_rounds)
        if limit <= 0:
            raise ValueError("max_rounds must be positive")
        # One analysis run for the whole sequence, so an analysis a reader declared
        # it preserves is computed once rather than once per round. Everything else
        # is recomputed from the program the round actually produced.
        run = _AnalysisRun(self.registry)
        for _ in range(limit):
            previous = ir
            for name, spec in resolved:
                ir = self._applied(name, spec, ir, run)
            if ir == previous:
                return ir
        raise CompilationError(
            "compiler optimization passes did not reach a fixed point"
        )

    def _applied(
        self, name: str, spec: PassSpec, program: CircuitIR, run: _AnalysisRun
    ) -> CircuitIR:
        # `PassFunction` and `AnalysisConsumer` are both stored in one field, so the
        # call site names which of the two the declaration selected.
        if not spec.uses:
            function = cast(PassFunction, spec.function)
            return _expect_ir(function, function(program), name)
        consumer = cast(AnalysisConsumer, spec.function)
        analyses = run.for_pass(spec, program)
        return _expect_ir(consumer, consumer(program, analyses), name)


def extension_pass_name(extension: str, pass_name: str) -> str:
    """The registered name of one pass an extension declares.

    The name carries both halves of the declaration -- the extension manifest name
    and the declared pass name both appear, so a trace names the extension that
    supplied a pass and two extensions cannot collide on a bare pass name.

    Raises:
        PassRegistryError: Either name is empty or contains a period, which would
            make the registered name ambiguous.
    """

    extension = _validated_name(extension, "extension")
    pass_name = _validated_name(pass_name, "pass")
    if "." in extension or "." in pass_name:
        raise PassRegistryError(
            "an extension name and a declared pass name may not contain '.'"
        )
    return f"{EXTENSION_PASS_PREFIX}{extension}.{pass_name}"


__all__ = (
    "EXTENSION_PASS_PREFIX",
    "OPTIMIZATION_PIPELINE",
    "RESERVED_PASS_NAMES",
    "AnalysisConsumer",
    "AnalysisFunction",
    "AnalysisSpec",
    "PassBinding",
    "PassFunction",
    "PassManager",
    "PassRegistry",
    "PassRegistryError",
    "PassSpec",
    "extension_pass_name",
)
