"""Declared ``CircuitIR`` transformation passes and the manager that runs them.

A rewrite of the FlagQuantum IR is admitted here as a declared pass: a name, the
IR levels it consumes and produces, the analyses it reads, the analyses it
leaves valid, the semantic properties it claims to preserve, and whether it is
deterministic. :class:`PassManager` executes declared passes in registration
order and refuses a pipeline whose declarations do not cover what the manager
requires, so a missing guarantee is a construction-time refusal rather than a
silently weaker program.

A declaration is a claim, not a proof: the manager enforces that the claim is
made, that it names known vocabulary, and that it is internally consistent.
Numerical agreement of each built-in pass is established by tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from importlib import import_module
from math import isclose
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

from ..core.ir import CircuitIR, Instruction
from ..errors import CapabilityError, CompilationError

PASS_CONTRACT_SCHEMA = "flagquantum_pass_contract_v1"

MANAGED_IR_LEVEL = "circuit_ir"
IR_LEVELS: frozenset[str] = frozenset(
    {"circuit_ir", "program_ir", "quantum_ir", "target_ir"}
)

# Analysis names from the designed analysis set. A pass may name a requirement
# or a preservation only from this vocabulary, so an invented analysis is
# refused when the pass is declared instead of failing to be supplied later.
ANALYSIS_VOCABULARY: frozenset[str] = frozenset(
    {
        "def_use",
        "dominance",
        "qubit_lifetime",
        "measurement_dependency",
        "parameter_dependency",
        "interaction_graph",
        "circuit_cost",
        "target_legality",
        "liveness_and_memory",
        "distribution",
    }
)

# Properties a transformation may claim to keep. A managed pipeline states the
# properties it requires, so a pass that keeps fewer than the pipeline needs is
# refused before it can rewrite a program.
SEMANTIC_PROPERTIES: frozenset[str] = frozenset(
    {
        "program_equivalence",
        "parameter_gradients",
        "wire_ownership",
        "measurement_recording",
    }
)

_SELF_INVERSE: frozenset[str] = frozenset(
    {"x", "y", "z", "h", "cx", "cy", "cz", "swap", "ccx", "cswap"}
)
_ROTATION_PARAM: Mapping[str, str] = MappingProxyType(
    {
        "rx": "theta",
        "ry": "theta",
        "rz": "theta",
        "phase": "theta",
        "u1": "theta",
        "rxx": "theta",
        "ryy": "theta",
        "rzz": "theta",
        "crx": "theta",
        "cry": "theta",
        "crz": "theta",
        "cphase": "theta",
    }
)
_IDENTITY_OPCODES: frozenset[str] = frozenset({"i", "id"})
_ROTATION_ZERO_ATOL = 1e-12


@dataclass(frozen=True)
class PassDeclaration:
    """What one transformation states about the programs it rewrites."""

    name: str
    summary: str
    # Declared rather than defaulted: a transformation states whether replaying
    # it on the same program produces the same result, and an adapter that has
    # no declaration states nothing.
    deterministic: bool
    input_level: str = MANAGED_IR_LEVEL
    output_level: str = MANAGED_IR_LEVEL
    required_analyses: tuple[str, ...] = ()
    preserved_analyses: tuple[str, ...] = ()
    preserved_semantics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise CompilationError("a pass declaration requires a non-empty name")
        object.__setattr__(self, "name", name)
        summary = str(self.summary).strip()
        if not summary:
            raise CompilationError(f"pass {name!r} requires a non-empty summary")
        object.__setattr__(self, "summary", summary)
        for attribute in ("input_level", "output_level"):
            level = str(getattr(self, attribute))
            if level not in IR_LEVELS:
                raise CompilationError(
                    f"pass {name!r} declares unknown {attribute} {level!r}; "
                    f"known levels are {sorted(IR_LEVELS)}"
                )
            object.__setattr__(self, attribute, level)
        for attribute, vocabulary in (
            ("required_analyses", ANALYSIS_VOCABULARY),
            ("preserved_analyses", ANALYSIS_VOCABULARY),
            ("preserved_semantics", SEMANTIC_PROPERTIES),
        ):
            declared = tuple(str(item) for item in getattr(self, attribute))
            unknown = sorted(set(declared) - vocabulary)
            if unknown:
                raise CompilationError(
                    f"pass {name!r} declares unknown {attribute} {unknown}; "
                    f"known names are {sorted(vocabulary)}"
                )
            if len(set(declared)) != len(declared):
                raise CompilationError(f"pass {name!r} repeats an {attribute} entry")
            object.__setattr__(self, attribute, declared)

    @property
    def preserves_program_equivalence(self) -> bool:
        return "program_equivalence" in self.preserved_semantics

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": PASS_CONTRACT_SCHEMA,
            "name": self.name,
            "summary": self.summary,
            "input_level": self.input_level,
            "output_level": self.output_level,
            "required_analyses": list(self.required_analyses),
            "preserved_analyses": list(self.preserved_analyses),
            "preserved_semantics": list(self.preserved_semantics),
            "deterministic": self.deterministic,
        }


@runtime_checkable
class CompilerPass(Protocol):
    """The shape a manager executes: a declaration plus one transformation."""

    declaration: PassDeclaration

    def transform(self, program: CircuitIR) -> CircuitIR: ...


@dataclass(frozen=True)
class RegisteredPass:
    """One pass under management: a declaration and the transform it names."""

    declaration: PassDeclaration
    transform: Callable[[CircuitIR], CircuitIR]
    origin: str = "pipeline"

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, PassDeclaration):
            raise CompilationError("a registered pass requires a PassDeclaration")
        if not callable(self.transform):
            raise CompilationError(
                f"pass {self.declaration.name!r} has a non-callable transform"
            )

    @property
    def name(self) -> str:
        return self.declaration.name

    def __call__(self, program: CircuitIR) -> CircuitIR:
        return self.transform(program)


@dataclass(frozen=True)
class PassManager:
    """Run declared passes in order, refusing undeclared guarantees."""

    passes: tuple[RegisteredPass, ...] = ()
    required_semantics: tuple[str, ...] = ("program_equivalence",)
    surviving_analyses: tuple[str, ...] = field(
        default=(), init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "passes", tuple(self.passes))
        required = tuple(str(item) for item in self.required_semantics)
        unknown = sorted(set(required) - SEMANTIC_PROPERTIES)
        if unknown:
            raise CompilationError(
                f"a managed pipeline requires unknown semantics {unknown}; "
                f"known properties are {sorted(SEMANTIC_PROPERTIES)}"
            )
        object.__setattr__(self, "required_semantics", required)
        seen: set[str] = set()
        for item in self.passes:
            if not isinstance(item, RegisteredPass):
                raise CompilationError(
                    "every managed pass must be a RegisteredPass; "
                    f"got {type(item).__name__}"
                )
            if item.name in seen:
                raise CompilationError(f"pass {item.name!r} is registered twice")
            seen.add(item.name)
            self._require_managed_level(item)
            self._require_declared_semantics(item)
            self._require_deterministic(item)
        object.__setattr__(self, "surviving_analyses", self._surviving_analyses())

    def _require_managed_level(self, item: RegisteredPass) -> None:
        for attribute in ("input_level", "output_level"):
            level = getattr(item.declaration, attribute)
            if level != MANAGED_IR_LEVEL:
                raise CapabilityError(
                    f"pass {item.name!r} declares {attribute} {level!r}; "
                    f"this manager executes {MANAGED_IR_LEVEL!r} programs"
                )

    def _require_declared_semantics(self, item: RegisteredPass) -> None:
        missing = tuple(
            property_name
            for property_name in self.required_semantics
            if property_name not in item.declaration.preserved_semantics
        )
        if missing:
            raise CapabilityError(
                f"pass {item.name!r} does not declare that it preserves "
                + ", ".join(missing)
                + "; declare the semantic properties the transformation keeps"
            )

    def _require_deterministic(self, item: RegisteredPass) -> None:
        if not item.declaration.deterministic:
            raise CapabilityError(
                f"pass {item.name!r} does not guarantee deterministic replay; "
                "a managed pipeline must compile the same program the same way"
            )

    def _surviving_analyses(self) -> tuple[str, ...]:
        """Fold the pipeline, refusing a pass that reads an invalidated analysis.

        A transformation invalidates every analysis it does not declare
        preserved, so a later pass may require one only if every earlier pass
        kept it.
        """

        kept = set(ANALYSIS_VOCABULARY)
        for item in self.passes:
            missing = sorted(set(item.declaration.required_analyses) - kept)
            if missing:
                raise CapabilityError(
                    f"pass {item.name!r} requires {', '.join(missing)} after an "
                    "earlier pass invalidated it; a pass that keeps an analysis "
                    "must declare it in preserved_analyses"
                )
            kept &= set(item.declaration.preserved_analyses)
        return tuple(sorted(kept))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.passes)

    def with_pass(self, pass_: CompilerPass) -> "PassManager":
        """Return a manager that runs ``pass_`` after every registered pass."""

        registered = (
            pass_
            if isinstance(pass_, RegisteredPass)
            else RegisteredPass(
                declaration=pass_.declaration,
                transform=pass_.transform,
                origin=getattr(pass_, "origin", "extension"),
            )
        )
        return PassManager(
            passes=(*self.passes, registered),
            required_semantics=self.required_semantics,
        )

    def run(self, program: CircuitIR) -> CircuitIR:
        """Apply every registered pass once, in registration order."""

        if not isinstance(program, CircuitIR):
            raise TypeError(
                f"a managed pass consumes CircuitIR, got {type(program).__name__}"
            )
        result = program
        for item in self.passes:
            transformed = item.transform(result)
            if not isinstance(transformed, CircuitIR):
                raise CompilationError(
                    f"pass {item.name!r} returned {type(transformed).__name__}; "
                    "a managed pass must return CircuitIR"
                )
            result = transformed
        return result

    def run_to_fixed_point(
        self,
        program: CircuitIR,
        *,
        measure: Callable[[CircuitIR], int] = len,
    ) -> CircuitIR:
        """Repeat the pipeline until ``measure`` stops changing.

        The round budget is one more than the size the input starts with, so a
        pipeline that never settles is refused instead of looping.
        """

        result = program
        for _ in range(int(measure(program)) + 1):
            previous = measure(result)
            result = self.run(result)
            if measure(result) == previous:
                return result
        raise CompilationError(
            "compiler optimization passes did not reach a fixed point"
        )

    def contract(self) -> dict[str, Any]:
        """Record what this pipeline declares, for evidence and diagnostics."""

        return {
            "kind": PASS_CONTRACT_SCHEMA,
            "managed_ir_level": MANAGED_IR_LEVEL,
            "required_semantics": list(self.required_semantics),
            "surviving_analyses": list(self.surviving_analyses),
            "passes": [item.declaration.to_dict() for item in self.passes],
        }


def pass_from_extension(
    extension: Any,
    *,
    declaration: PassDeclaration | None = None,
) -> RegisteredPass:
    """Adapt one negotiated compiler-pass extension into a managed pass.

    The extension's own lifecycle is honoured: the pass is negotiated and
    started for each transformation and closed afterwards, so a failed
    transformation cannot leave the extension active.

    The declaration is the caller's, the extension's own ``declaration``
    attribute, or an empty one that states no guarantee. A manager that requires
    a semantic property then refuses the pass instead of assuming it.
    """

    extensions = import_module("..ecosystem.extensions", __package__)
    manifest = getattr(extension, "manifest", None)
    if manifest is None:
        raise CapabilityError("a compiler pass extension must expose a manifest")
    name = str(getattr(manifest, "name", ""))
    kind = str(getattr(manifest, "kind", ""))
    if kind != "compiler_pass":
        raise CapabilityError(
            f"extension {name!r} declares kind {kind!r}; "
            "a compiler pass extension must declare 'compiler_pass'"
        )
    declared_transform = getattr(extension, "transform", None)
    if not callable(declared_transform):
        raise CapabilityError(
            f"compiler pass extension {name!r} exposes no callable transform"
        )
    resolved = declaration if declaration is not None else getattr(
        extension, "declaration", None
    )
    if resolved is None:
        resolved = PassDeclaration(
            name=name,
            summary=f"transformation supplied by extension {name!r}",
            deterministic=False,
        )
    if not isinstance(resolved, PassDeclaration):
        raise CapabilityError(
            f"compiler pass extension {name!r} declares "
            f"{type(resolved).__name__}; a PassDeclaration is required"
        )
    if resolved.name != name:
        raise CapabilityError(
            f"compiler pass extension {name!r} carries the declaration of "
            f"{resolved.name!r}; the declaration must name the extension"
        )
    registry = extensions.ExtensionRegistry().with_extension(extension)

    def transform(program: CircuitIR) -> CircuitIR:
        handle = registry.negotiate(
            "compiler_pass",
            name,
            extensions.CapabilityRequest(required=frozenset({MANAGED_IR_LEVEL})),
        )
        handle.start(extensions.ExtensionConfig())
        try:
            return cast(CircuitIR, handle.invoke("transform", program))
        finally:
            handle.close()

    return RegisteredPass(
        declaration=resolved, transform=transform, origin="extension"
    )


def _is_zero(value: Any, atol: float = _ROTATION_ZERO_ATOL) -> bool:
    try:
        if bool(getattr(value, "requires_grad", False)):
            return False
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "numel") and value.numel() != 1:
            return False
        if hasattr(value, "item"):
            value = value.item()
        return isclose(float(value), 0.0, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def _add_values(left: Any, right: Any) -> Any:
    return left + right


def _replace_param(instruction: Instruction, key: str, value: Any) -> Instruction:
    params = dict(instruction.params)
    params[key] = value
    return Instruction(
        name=instruction.name,
        wires=instruction.wires,
        params=params,
        matrix=instruction.matrix,
        metadata=instruction.metadata,
    )


def _remove_identity_gates(ir: CircuitIR) -> CircuitIR:
    instructions = []
    for instruction in ir:
        if instruction.name in _IDENTITY_OPCODES:
            continue
        param_name = _ROTATION_PARAM.get(instruction.name)
        if param_name is not None and _is_zero(instruction.params.get(param_name)):
            continue
        instructions.append(instruction)
    return replace(ir, instructions=tuple(instructions))


def _last_touching_instruction(
    instructions: list[Instruction],
    wires: tuple[int, ...],
) -> int | None:
    target_wires = set(wires)
    for index in range(len(instructions) - 1, -1, -1):
        if not target_wires.isdisjoint(instructions[index].wires):
            return index
    return None


def _merge_self_inverse(ir: CircuitIR) -> CircuitIR:
    out: list[Instruction] = []
    for instruction in ir:
        previous_index = _last_touching_instruction(out, instruction.wires)
        previous = out[previous_index] if previous_index is not None else None
        if (
            instruction.name in _SELF_INVERSE
            and previous is not None
            and previous.name == instruction.name
            and previous.wires == instruction.wires
            and not instruction.params
            and not previous.params
        ):
            assert previous_index is not None
            out.pop(previous_index)
        else:
            out.append(instruction)
    return replace(ir, instructions=tuple(out))


def _merge_adjacent_rotations(ir: CircuitIR) -> CircuitIR:
    out: list[Instruction] = []
    for instruction in ir:
        param_name = _ROTATION_PARAM.get(instruction.name)
        previous_index = _last_touching_instruction(out, instruction.wires)
        previous = out[previous_index] if previous_index is not None else None
        if (
            param_name is not None
            and previous is not None
            and previous.name == instruction.name
            and previous.wires == instruction.wires
            and previous.matrix is None
            and instruction.matrix is None
            and param_name in previous.params
            and param_name in instruction.params
        ):
            assert previous_index is not None
            merged_value = _add_values(
                previous.params[param_name], instruction.params[param_name]
            )
            if _is_zero(merged_value):
                out.pop(previous_index)
            else:
                out[previous_index] = _replace_param(
                    previous,
                    param_name,
                    merged_value,
                )
            continue
        out.append(instruction)
    return replace(ir, instructions=tuple(out))


IDENTITY_REMOVAL = RegisteredPass(
    declaration=PassDeclaration(
        name="identity_removal",
        summary="drops explicit identity gates and zero-angle rotations",
        preserved_analyses=("def_use", "qubit_lifetime", "liveness_and_memory"),
        preserved_semantics=(
            "program_equivalence",
            "parameter_gradients",
            "wire_ownership",
            "measurement_recording",
        ),
    ),
    transform=_remove_identity_gates,
)

SELF_INVERSE_CANCELLATION = RegisteredPass(
    declaration=PassDeclaration(
        name="self_inverse_cancellation",
        summary="drops identical self-inverse gates adjacent on their wires",
        preserved_analyses=(
            "def_use",
            "qubit_lifetime",
            "liveness_and_memory",
            "measurement_dependency",
        ),
        preserved_semantics=(
            "program_equivalence",
            "parameter_gradients",
            "wire_ownership",
            "measurement_recording",
        ),
    ),
    transform=_merge_self_inverse,
)

ROTATION_MERGE = RegisteredPass(
    declaration=PassDeclaration(
        name="rotation_merge",
        summary="merges rotations adjacent on their wires",
        preserved_analyses=(
            "def_use",
            "qubit_lifetime",
            "liveness_and_memory",
            "measurement_dependency",
            "parameter_dependency",
        ),
        preserved_semantics=(
            "program_equivalence",
            "parameter_gradients",
            "wire_ownership",
            "measurement_recording",
        ),
    ),
    transform=_merge_adjacent_rotations,
)


def passes_in_order(*items: RegisteredPass) -> tuple[RegisteredPass, ...]:
    """Return ``items`` as a pipeline, refusing a repeated pass."""

    names = [item.name for item in items]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise CompilationError(
            "a pipeline cannot run " + ", ".join(repeated) + " twice"
        )
    return tuple(items)


__all__ = (
    "ANALYSIS_VOCABULARY",
    "CompilerPass",
    "IDENTITY_REMOVAL",
    "IR_LEVELS",
    "MANAGED_IR_LEVEL",
    "PASS_CONTRACT_SCHEMA",
    "PassDeclaration",
    "PassManager",
    "ROTATION_MERGE",
    "RegisteredPass",
    "SEMANTIC_PROPERTIES",
    "SELF_INVERSE_CANCELLATION",
    "pass_from_extension",
    "passes_in_order",
)
