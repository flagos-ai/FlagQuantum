"""Private bounded target pipeline; Runtime must still confirm its executable."""

from collections import Counter
from dataclasses import dataclass, field, replace
from time import perf_counter

from ...core.ir import CircuitIR
from ...core.runtime_config import RuntimeConfig
from ...errors import CompilationError
from ..basis_conversion import (
    BasisConversionError,
    BasisConversionResult,
    convert_basis,
    require_basis,
)
from ..coupling import CouplingMap
from ..layout import Layout, _routing_swap_phases, final_layout, remove_layout_restore
from ..pipeline import compile, schedule_layers


class TargetError(ValueError):
    """A known target limitation or intermediate construction budget."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


_DEFAULT_CONFIG = RuntimeConfig().with_overrides(complex_dtype="complex128")


# Keep exact one-qubit phase compensation native. The admitted two-qubit
# rewrites are the existing CCX/CZ/SWAP identities, not projective cphase/KAK.
PHASE_BASIS = (
    "x",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "ry",
    "rz",
    "cx",
    "u3",
    "swap",
    "z",
    "phase",
    "cz",
)


@dataclass(frozen=True)
class Target:
    """Explicit expert profiles by allocated width; never resize a device graph."""

    basis: tuple[str, ...] | None = None
    couplings: tuple[CouplingMap, ...] = ()
    optimize: bool = True
    restore_layout: bool = True
    config: RuntimeConfig = _DEFAULT_CONFIG

    def __post_init__(self) -> None:
        if type(self.optimize) is not bool or type(self.restore_layout) is not bool:
            raise TypeError("target optimization and restore flags must be booleans")
        if not isinstance(self.config, RuntimeConfig):
            raise TypeError("target config must be RuntimeConfig")
        if (
            self.config.backend not in {"torch", "pytorch"}
            or self.config.device != "cpu"
        ):
            raise ValueError("this target profile requires local CPU torch")
        if self.basis is not None:
            object.__setattr__(self, "basis", require_basis(self.basis))
        object.__setattr__(self, "couplings", tuple(self.couplings))
        if any(not isinstance(c, CouplingMap) for c in self.couplings):
            raise TypeError("target couplings require undirected CouplingMap records")
        if len({c.n_qubits for c in self.couplings}) != len(self.couplings):
            raise ValueError("provide one unambiguous coupling profile per width")

    def coupling_for(self, width: int) -> CouplingMap | None:
        if not self.couplings:
            return None
        for coupling in self.couplings:
            if coupling.n_qubits == width:
                return CouplingMap(
                    coupling.n_qubits,
                    coupling.edges,
                    path_cache_capacity=coupling.path_cache_capacity,
                )
        raise TargetError(
            "target.unsupported", f"no explicit coupling profile for width {width}"
        )


@dataclass(frozen=True)
class TargetProduct:
    submitted: CircuitIR
    executable: CircuitIR
    stages: tuple[tuple[str, CircuitIR], ...]
    conversions: tuple[BasisConversionResult, ...]
    layout: Layout
    coupling: CouplingMap | None
    elapsed_seconds: float
    content_identities: tuple[tuple[str, str], ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "content_identities",
            tuple((name, program.content_hash) for name, program in self.stages),
        )
        self.validate()

    def validate(self) -> None:
        """Refuse stale programs without recompiling or creating new evidence."""
        current = tuple((name, program.content_hash) for name, program in self.stages)
        if (
            current != self.content_identities
            or self.submitted.content_hash != self.content_identities[-3][1]
            or self.executable.content_hash != self.content_identities[-1][1]
        ):
            raise TargetError(
                "target.identity",
                "a target stage changed after its content identity was recorded",
            )


def compile_target(
    source: CircuitIR, target: Target, max_operations: int
) -> TargetProduct:
    """Reuse exact Compiler transforms and predict normal Runtime compilation.

    Intermediate caps are construction budgets. Final user ceilings are checked
    separately on the executable, including all preparation and cleanup.
    """
    start = perf_counter()
    stages: list[tuple[str, CircuitIR]] = []
    conversions: list[BasisConversionResult] = []

    def record(name: str, program: CircuitIR) -> CircuitIR:
        if len(program.instructions) > max_operations:
            raise TargetError(
                "target.budget",
                f"{name} exceeds intermediate operation budget {max_operations}",
            )
        stages.append((name, program))
        return program

    def convert(program: CircuitIR, name: str) -> CircuitIR:
        if target.basis is None:
            return program
        try:
            conversion = convert_basis(
                program,
                gates=target.basis,
                max_added_operations=max_operations - len(program.instructions),
            )
        except BasisConversionError as error:
            code = (
                "target.budget"
                if "max_added_operations" in str(error)
                else "target.unsupported"
            )
            raise TargetError(code, str(error)) from error
        conversions.append(conversion)
        return record(name, conversion.program)

    record("source", source)
    basis_program = convert(source, "initial_basis")
    coupling = target.coupling_for(source.n_wires)
    try:
        routed = record(
            "target_compile",
            compile(
                basis_program,
                coupling_map=coupling,
                routing_strategy="persistent_layout",
                optimize=target.optimize,
                config=target.config,
            ),
        )
        if coupling is not None and not target.restore_layout:
            routed = record("remove_restore", remove_layout_restore(routed))
        submitted = convert(routed, "post_basis")
        normalized = record(
            "runtime_normalization", compile(submitted, config=target.config)
        )
        executable = record(
            "executable",
            replace(
                normalized,
                # Runtime seals the submitted annotations and the normalized
                # layers separately. Keep compiler pass details in the prior
                # stage, rather than inventing an extra execution annotation.
                metadata=submitted.metadata,
                instructions=tuple(
                    g for layer in schedule_layers(normalized) for g in layer
                ),
            ),
        )
        layout = (
            final_layout(executable)
            if coupling is not None
            else Layout(range(source.n_wires))
        )
    except (ValueError, CompilationError) as error:
        if isinstance(error, TargetError):
            raise
        raise TargetError("target.unsupported", str(error)) from error
    if target.basis is not None:
        outside = sorted({g.name for g in executable.instructions} - set(target.basis))
        if outside:
            raise TargetError(
                "target.basis_drift",
                f"actual normalized leaves {outside} are outside the requested basis",
            )
    if coupling is not None:
        # The route is produced here by the existing Compiler. Later transforms
        # must preserve its SWAP witnesses and layout record. ASAP may reorder
        # disjoint SWAPs, so compare their multiset and replay the final mapping.
        def witness(program: CircuitIR) -> tuple[tuple[str, tuple[int, int]], ...]:
            return tuple(
                (phase, pair) for _, phase, pair in _routing_swap_phases(program)
            )

        if Counter(witness(routed)) != Counter(
            witness(executable)
        ) or routed.metadata.get("routing") != executable.metadata.get("routing"):
            raise TargetError(
                "target.layout", "target transformation lost routed SWAP evidence"
            )
        replayed = Layout(range(source.n_wires)).apply_swaps(
            pair for _, pair in witness(executable)
        )
        if replayed != layout:
            raise TargetError(
                "target.layout", "normalized SWAP replay disagrees with final layout"
            )
        if any(
            len(g.wires) > 2
            or (len(g.wires) == 2 and tuple(sorted(g.wires)) not in coupling.edges)
            for g in executable.instructions
        ):
            raise TargetError(
                "target.topology",
                "normalized executable violates the requested two-qubit topology",
            )
    return TargetProduct(
        submitted,
        executable,
        tuple(stages),
        tuple(conversions),
        layout,
        coupling,
        perf_counter() - start,
    )
