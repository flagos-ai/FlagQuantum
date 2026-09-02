"""Private, provider-free Phase 2 static deployment compilation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from flagquantum.core.ir import CircuitIR

from .diagnostics import Diagnostic, DiagnosticCode
from .exporters.circuit_ir import SealedCircuitIRRoundTrip, seal_circuit_ir_round_trip
from .exporters.text import (
    TextEmissionResult,
    TextEmissionStatus,
    emit_openqasm2,
    emit_openqasm3,
    emit_qcis_v1,
)
from .ir.modules import QuantumModule
from .passes.manager import PassManager
from .passes.placement_routing import DirectedCouplingGraph, PlacementRoutingPass
from .passes.static_canonicalization import StaticCanonicalizationPass
from .passes.target_decomposition import (
    UNIVERSAL_RX_RY_RZ_CX_V1,
    DecomposeToTargetGateSetPass,
)
from .pipeline_cache import (
    BoundedPipelineCache,
    CachedPipelineExecution,
    CachedPipelineRunner,
    CompilationIdentityInputs,
)

OFFLINE_DEPLOYMENT_SCHEMA = "flagquantum.offline_static_compilation.v1"


class OfflineTextFormat(str, Enum):
    OPENQASM2 = "openqasm2"
    OPENQASM3 = "openqasm3"
    QCIS_V1 = "qcis_v1"


class OfflineCompilationStatus(str, Enum):
    COMPILED_EXACT = "compiled_exact"
    UNSUPPORTED_WITH_DIAGNOSTICS = "unsupported_with_diagnostics"
    INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class OfflineStaticTarget:
    """An anonymous, immutable device snapshot; never a cloud backend handle."""

    coupling_graph: DirectedCouplingGraph
    calibration_identity: str
    initial_layout: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        identity = self.calibration_identity
        if (
            len(identity) != 64
            or identity.lower() != identity
            or any(character not in "0123456789abcdef" for character in identity)
        ):
            raise ValueError("calibration identity must be a lowercase SHA-256 digest")
        if self.initial_layout is not None:
            object.__setattr__(
                self, "initial_layout", tuple(int(item) for item in self.initial_layout)
            )

    @property
    def topology_identity(self) -> str:
        payload = json.dumps(
            self.coupling_graph.canonical(), separators=(",", ":"), sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OfflineCompilationResult:
    status: OfflineCompilationStatus
    source_artifact: SealedCircuitIRRoundTrip | None = None
    module: QuantumModule | None = None
    execution: CachedPipelineExecution | None = None
    emissions: tuple[tuple[OfflineTextFormat, TextEmissionResult], ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is OfflineCompilationStatus.COMPILED_EXACT

    def emission(self, output_format: OfflineTextFormat) -> TextEmissionResult:
        for candidate, result in self.emissions:
            if candidate is output_format:
                return result
        raise KeyError(f"output format {output_format.value!r} was not requested")


_EMITTERS = {
    OfflineTextFormat.OPENQASM2: emit_openqasm2,
    OfflineTextFormat.OPENQASM3: emit_openqasm3,
    OfflineTextFormat.QCIS_V1: emit_qcis_v1,
}


def _failure(
    status: OfflineCompilationStatus,
    message: str,
    *,
    artifact: SealedCircuitIRRoundTrip | None = None,
    execution: CachedPipelineExecution | None = None,
    diagnostics: tuple[Diagnostic, ...] = (),
) -> OfflineCompilationResult:
    if not diagnostics:
        diagnostics = (
            Diagnostic(
                DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                message,
                notes=("offline deployment compilation never drops semantics",),
            ),
        )
    return OfflineCompilationResult(
        status, source_artifact=artifact, execution=execution, diagnostics=diagnostics
    )


def _preflight(source: CircuitIR, target: OfflineStaticTarget) -> str | None:
    if source.n_wires != target.coupling_graph.n_qubits:
        return "logical and physical qubit counts must match in the v1 static profile"
    if source.measurements:
        return "measurements are outside the v1 static deployment text profile"
    if source.observables:
        return "observables are outside the v1 static deployment text profile"
    if source.metadata.get("dynamic_circuit") is True:
        return "dynamic circuits are outside the v1 static deployment profile"
    return None


def _pipeline(target: OfflineStaticTarget) -> PassManager:
    return PassManager(
        (
            StaticCanonicalizationPass(),
            DecomposeToTargetGateSetPass(),
            PlacementRoutingPass(
                target.coupling_graph, initial_layout=target.initial_layout
            ),
            StaticCanonicalizationPass(),
        )
    )


def compile_offline_static(
    source: object,
    target: OfflineStaticTarget,
    *,
    output_formats: tuple[OfflineTextFormat, ...] = (
        OfflineTextFormat.OPENQASM2,
        OfflineTextFormat.OPENQASM3,
        OfflineTextFormat.QCIS_V1,
    ),
    cache: BoundedPipelineCache | None = None,
) -> OfflineCompilationResult:
    """Compile one static CircuitIR without provider discovery or submission."""

    if not isinstance(source, CircuitIR):
        return _failure(
            OfflineCompilationStatus.INVALID_INPUT,
            "offline static compilation requires CircuitIR",
        )
    unsupported = _preflight(source, target)
    if unsupported is not None:
        return _failure(
            OfflineCompilationStatus.UNSUPPORTED_WITH_DIAGNOSTICS, unsupported
        )
    if len(output_formats) != len(set(output_formats)) or any(
        not isinstance(item, OfflineTextFormat) for item in output_formats
    ):
        return _failure(
            OfflineCompilationStatus.INVALID_INPUT,
            "output formats must be unique OfflineTextFormat values",
        )
    sealed = seal_circuit_ir_round_trip(source)
    if not sealed.ok or sealed.artifact is None:
        return _failure(
            OfflineCompilationStatus.INVALID_INPUT,
            "CircuitIR import failed",
            diagnostics=sealed.diagnostics,
        )
    artifact = sealed.artifact
    manager = _pipeline(target)
    execution = CachedPipelineRunner(
        manager, cache if cache is not None else BoundedPipelineCache()
    ).run(
        artifact.imported.module,
        CompilationIdentityInputs(
            source_identity=source.content_hash,
            target_profile=(
                f"{UNIVERSAL_RX_RY_RZ_CX_V1.name}:"
                f"{UNIVERSAL_RX_RY_RZ_CX_V1.version}"
            ),
            topology_identity=target.topology_identity,
            calibration_identity=target.calibration_identity,
            compile_options={
                "schema": OFFLINE_DEPLOYMENT_SCHEMA,
                "initial_layout": target.initial_layout,
                "output_formats": tuple(item.value for item in output_formats),
            },
        ),
    )
    if not execution.ok or execution.pipeline is None:
        diagnostics = tuple(
            item.diagnostic
            for item in execution.diagnostics
            if item.diagnostic is not None
        )
        return _failure(
            OfflineCompilationStatus.UNSUPPORTED_WITH_DIAGNOSTICS,
            "private static compilation pipeline failed",
            artifact=artifact,
            execution=execution,
            diagnostics=diagnostics,
        )
    module = execution.pipeline.module
    emissions = tuple((item, _EMITTERS[item](module)) for item in output_formats)
    failed = tuple(
        diagnostic
        for _, emission in emissions
        if emission.status is not TextEmissionStatus.EMITTED_EXACT
        for diagnostic in emission.diagnostics
    )
    if failed:
        return OfflineCompilationResult(
            OfflineCompilationStatus.UNSUPPORTED_WITH_DIAGNOSTICS,
            artifact,
            module,
            execution,
            emissions,
            failed,
        )
    return OfflineCompilationResult(
        OfflineCompilationStatus.COMPILED_EXACT,
        artifact,
        module,
        execution,
        emissions,
    )


__all__ = [
    "OFFLINE_DEPLOYMENT_SCHEMA",
    "OfflineCompilationResult",
    "OfflineCompilationStatus",
    "OfflineStaticTarget",
    "OfflineTextFormat",
    "compile_offline_static",
]
