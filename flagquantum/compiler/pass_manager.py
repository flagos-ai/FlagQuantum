"""Named, replaceable circuit-optimization passes and the manager that runs them.

The optimization pipeline used to be nine inline calls inside
``pipeline._optimize_to_fixed_point``. That spelling named no pass, so a pass
could not be replaced, listed, or supplied from outside the repository, and the
Compiler extension protocol's ``compiler_pass`` kind had no consumer. This module
supplies the machinery that makes a pass a name: a registry, a sequence, and a
manager that runs one over a program.

Two authorities, one direction. This module owns the *pipeline* -- which pass runs
in which order, and how a pass is named, registered, and resolved. It owns no pass
body: every built-in implementation stays in the module that already implements
it, and ``pipeline`` binds the built-in names to those functions. It also owns no
extension protocol; it accepts a function, and
:mod:`flagquantum.ecosystem.extensions.pass_admission` is the one crossing that
turns a negotiated ``CompilerPassExtension`` into such a function.

Fail closed, in two places. A sequence is resolved in full before the first pass
runs, so a mistyped name cannot leave a half-applied pipeline behind. And the
built-in names are reserved: a registry refuses to shadow one, so an extension
cannot silently change what ``compile`` does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from ..core.ir import CircuitIR, ensure_circuit_ir
from ..errors import CapabilityError, CompilationError

PassFunction = Callable[[CircuitIR], CircuitIR]

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
    """A pass name cannot be registered or resolved exactly as declared."""


def _validated_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise PassRegistryError("pass name cannot be empty")
    return name.strip()


def _validated_function(name: str, function: object) -> PassFunction:
    if not callable(function):
        raise PassRegistryError(
            f"pass {name!r} is not callable: {type(function).__name__}"
        )
    return function


def _validated_program(program: object) -> CircuitIR:
    return ensure_circuit_ir(program)


def _expect_ir(function: PassFunction, result: object, name: str) -> CircuitIR:
    if not isinstance(result, CircuitIR):
        raise CompilationError(
            f"pass {name!r} returned {type(result).__name__}; expected CircuitIR"
        )
    return result


@dataclass(frozen=True)
class PassRegistry:
    """Immutable name-to-pass map. Additions return a new registry."""

    passes: Mapping[str, PassFunction] = field(default_factory=dict)

    def __post_init__(self) -> None:
        resolved: dict[str, PassFunction] = {}
        for name, function in dict(self.passes).items():
            resolved[_validated_name(name)] = _validated_function(name, function)
        object.__setattr__(self, "passes", MappingProxyType(resolved))

    def with_pass(self, name: str, function: PassFunction) -> "PassRegistry":
        """Return a registry that also resolves ``name``.

        Raises:
            PassRegistryError: The name is empty, already taken, or reserved for
                a built-in pass; or the function is not callable.
        """

        name = _validated_name(name)
        _validated_function(name, function)
        if name in self.passes:
            raise PassRegistryError(
                f"pass {name!r} is already registered; a live pass is never "
                "displaced silently"
            )
        if name in RESERVED_PASS_NAMES:
            raise PassRegistryError(
                f"pass {name!r} is a built-in pass name; an extension must "
                f"register under {EXTENSION_PASS_PREFIX!r} so that the built-in "
                "meaning of a name cannot be redefined"
            )
        return PassRegistry({**self.passes, name: function})

    def resolve(self, name: str) -> PassFunction:
        """Return the function registered under ``name``.

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

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.passes))

    @property
    def extension_names(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.names if name.startswith(EXTENSION_PASS_PREFIX)
        )


@dataclass(frozen=True)
class PassManager:
    """Run an ordered sequence of registered passes over one program.

    The manager holds no pass body and no sequence of its own: both arrive from
    the caller, so the same manager serves the fixed optimization pipeline and a
    caller that names its own passes.
    """

    registry: PassRegistry

    def run(self, program: CircuitIR, sequence: Sequence[str]) -> CircuitIR:
        """Apply ``sequence`` in order, resolving every name before the first pass.

        Raises:
            PassRegistryError: A name in the sequence is not registered. Nothing
                has been applied when this is raised.
            CompilationError: A pass returned something that is not a CircuitIR.
        """

        resolved = tuple((name, self.registry.resolve(name)) for name in sequence)
        ir = _validated_program(program)
        for name, function in resolved:
            ir = _expect_ir(function, function(ir), name)
        return ir

    def to_fixed_point(
        self,
        program: CircuitIR,
        sequence: Sequence[str],
        *,
        max_rounds: int | None = None,
    ) -> CircuitIR:
        """Apply ``sequence`` repeatedly until the program stops changing.

        A round that removes no instruction is a fixed point, which is what the
        inline loop this replaces tested. The bound defaults to one round per
        instruction plus one.

        Raises:
            PassRegistryError: A name in the sequence is not registered.
            CompilationError: A pass returned something that is not a CircuitIR,
                or no fixed point was reached inside ``max_rounds``.
            ValueError: ``max_rounds`` is not positive.
        """

        resolved = tuple((name, self.registry.resolve(name)) for name in sequence)
        ir = _validated_program(program)
        limit = len(ir) + 1 if max_rounds is None else int(max_rounds)
        if limit <= 0:
            raise ValueError("max_rounds must be positive")
        for _ in range(limit):
            previous_count = len(ir)
            for name, function in resolved:
                ir = _expect_ir(function, function(ir), name)
            if len(ir) == previous_count:
                return ir
        raise CompilationError(
            "compiler optimization passes did not reach a fixed point"
        )


def extension_pass_name(extension: str, pass_name: str) -> str:
    """The registered name of one pass an extension declares.

    The extension's own manifest name and the declared pass name both appear, so
    a trace names the extension that supplied a pass and two extensions cannot
    collide on a bare pass name.

    Raises:
        PassRegistryError: Either name is empty or contains a period, which would
            make the registered name ambiguous.
    """

    extension = _validated_name(extension)
    pass_name = _validated_name(pass_name)
    if "." in extension or "." in pass_name:
        raise PassRegistryError(
            "an extension name and a declared pass name may not contain '.'"
        )
    return f"{EXTENSION_PASS_PREFIX}{extension}.{pass_name}"


__all__ = (
    "EXTENSION_PASS_PREFIX",
    "OPTIMIZATION_PIPELINE",
    "RESERVED_PASS_NAMES",
    "PassFunction",
    "PassManager",
    "PassRegistry",
    "PassRegistryError",
    "extension_pass_name",
)
