"""Private deterministic compilation identity and bounded pipeline cache."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from threading import Condition, RLock

from .bindings import RuntimeBindingRef, SymbolicExpression
from .diagnostics import Diagnostic, DiagnosticCode, DiagnosticSeverity
from .ir.modules import QuantumModule
from .ir.operations import FrozenAttributes, canonical_value
from .passes.manager import PassManager, PipelineResult

COMPILATION_IDENTITY_SCHEMA = "flagquantum.compilation_identity.v1"


class IdentityStatus(str, Enum):
    CACHEABLE = "cacheable"
    BYPASS_WITH_DIAGNOSTICS = "bypass_with_diagnostics"


class CacheDisposition(str, Enum):
    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"
    FAILURE_NOT_CACHED = "failure_not_cached"


class PipelineExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class CompilationIdentityInputs:
    source_identity: object
    target_profile: object
    topology_identity: object
    calibration_identity: object
    compile_options: object = None


@dataclass(frozen=True)
class CompilationIdentity:
    digest: str
    canonical_json: str
    source_identity: str
    input_program_identity: str
    pipeline_digest: str


@dataclass(frozen=True)
class CompilationIdentityResult:
    status: IdentityStatus
    identity: CompilationIdentity | None = None
    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def cacheable(self) -> bool:
        return self.status is IdentityStatus.CACHEABLE


@dataclass(frozen=True)
class PipelineDiagnosticRecord:
    stage: str
    source_identity: str
    diagnostic: Diagnostic | None = None
    pass_name: str | None = None
    cause_type: str | None = None
    cause_message: str | None = None


@dataclass(frozen=True)
class PipelineCacheSnapshot:
    capacity: int
    size: int
    hits: int
    misses: int
    bypasses: int
    invalidations: int
    evictions: int
    collisions: int
    computations: int
    failed_compilations: int
    waits: int


@dataclass(frozen=True)
class CachedPipelineExecution:
    status: PipelineExecutionStatus
    cache_disposition: CacheDisposition
    pipeline: PipelineResult | None
    identity: CompilationIdentity | None
    diagnostics: tuple[PipelineDiagnosticRecord, ...]
    cache_snapshot: PipelineCacheSnapshot

    @property
    def ok(self) -> bool:
        return self.status is PipelineExecutionStatus.COMPLETED


def _identity_failure(message: str) -> CompilationIdentityResult:
    return CompilationIdentityResult(
        IdentityStatus.BYPASS_WITH_DIAGNOSTICS,
        diagnostics=(
            Diagnostic(
                DiagnosticCode.ATTRIBUTE_TYPE_MISMATCH,
                message,
                notes=("unsafe or unknown identity inputs bypass the cache",),
                severity=DiagnosticSeverity.WARNING,
            ),
        ),
    )


def _contains_runtime_binding(value: object) -> bool:
    if isinstance(value, RuntimeBindingRef):
        return True
    if isinstance(value, SymbolicExpression):
        return any(_contains_runtime_binding(item) for item in value.args)
    if isinstance(value, FrozenAttributes):
        return any(_contains_runtime_binding(item) for item in value.values())
    if isinstance(value, tuple):
        return any(_contains_runtime_binding(item) for item in value)
    return False


def build_compilation_identity(
    module: QuantumModule,
    pass_manager: PassManager,
    inputs: CompilationIdentityInputs,
) -> CompilationIdentityResult:
    """Build an exact cache identity or return an explicit bypass reason."""

    if any(
        not isinstance(value, str) or not value.strip()
        for value in (
            inputs.source_identity,
            inputs.target_profile,
            inputs.topology_identity,
            inputs.calibration_identity,
        )
    ):
        return _identity_failure(
            "all compilation identity fields must be known strings"
        )
    if any(
        _contains_runtime_binding(operation.attributes)
        for block in module.body.blocks
        for operation in block.operations
    ):
        return _identity_failure("trainable runtime bindings are not cacheable")
    try:
        if inputs.compile_options is None:
            options = FrozenAttributes()
        elif isinstance(inputs.compile_options, Mapping):
            options = FrozenAttributes(inputs.compile_options)
        else:
            raise TypeError("compile options must be a deterministic mapping")
        payload = {
            "schema": COMPILATION_IDENTITY_SCHEMA,
            "source_identity": inputs.source_identity.strip(),
            "input_program_identity": module.program_identity,
            "pipeline_digest": pass_manager.pipeline_digest,
            "target_profile": inputs.target_profile.strip(),
            "topology_identity": inputs.topology_identity.strip(),
            "calibration_identity": inputs.calibration_identity.strip(),
            "compile_options": canonical_value(options),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        return _identity_failure(f"compile options are not deterministic: {exc}")
    identity = CompilationIdentity(
        hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        encoded,
        inputs.source_identity.strip(),
        module.program_identity,
        pass_manager.pipeline_digest,
    )
    return CompilationIdentityResult(IdentityStatus.CACHEABLE, identity)


class BoundedPipelineCache:
    """Thread-safe LRU cache with single-flight compilation per exact identity."""

    def __init__(self, max_entries: int = 128) -> None:
        capacity = int(max_entries)
        if capacity <= 0:
            raise ValueError("pipeline cache capacity must be positive")
        self._capacity = capacity
        self._entries: OrderedDict[str, tuple[str, PipelineResult]] = OrderedDict()
        self._inflight: set[str] = set()
        self._condition = Condition(RLock())
        self._hits = 0
        self._misses = 0
        self._bypasses = 0
        self._invalidations = 0
        self._evictions = 0
        self._collisions = 0
        self._computations = 0
        self._failed_compilations = 0
        self._waits = 0

    def note_bypass(self) -> None:
        with self._condition:
            self._bypasses += 1

    def note_failed_compilation(self) -> None:
        with self._condition:
            self._failed_compilations += 1

    def execute(
        self,
        identity: CompilationIdentity,
        compute: Callable[[], PipelineResult],
    ) -> tuple[PipelineResult, CacheDisposition]:
        while True:
            with self._condition:
                existing = self._entries.get(identity.digest)
                if existing is not None:
                    canonical_json, pipeline = existing
                    if canonical_json != identity.canonical_json:
                        self._collisions += 1
                        self._bypasses += 1
                        collision = True
                    else:
                        self._entries.move_to_end(identity.digest)
                        self._hits += 1
                        return pipeline, CacheDisposition.HIT
                else:
                    collision = False
                if collision:
                    break
                if identity.digest in self._inflight:
                    self._waits += 1
                    self._condition.wait_for(
                        lambda: identity.digest not in self._inflight
                    )
                    continue
                self._inflight.add(identity.digest)
                self._misses += 1
                break
        if collision:
            try:
                pipeline = compute()
            except BaseException:
                self.note_failed_compilation()
                raise
            with self._condition:
                self._computations += 1
                if not pipeline.ok:
                    self._failed_compilations += 1
            return pipeline, CacheDisposition.BYPASS
        try:
            pipeline = compute()
        except BaseException:
            with self._condition:
                self._failed_compilations += 1
                self._inflight.remove(identity.digest)
                self._condition.notify_all()
            raise
        with self._condition:
            self._computations += 1
            if pipeline.ok:
                self._entries[identity.digest] = (identity.canonical_json, pipeline)
                self._entries.move_to_end(identity.digest)
                if len(self._entries) > self._capacity:
                    self._entries.popitem(last=False)
                    self._evictions += 1
                disposition = CacheDisposition.MISS
            else:
                self._failed_compilations += 1
                disposition = CacheDisposition.FAILURE_NOT_CACHED
            self._inflight.remove(identity.digest)
            self._condition.notify_all()
            return pipeline, disposition

    def invalidate(self, identity: CompilationIdentity | None = None) -> int:
        with self._condition:
            if identity is None:
                removed = len(self._entries)
                self._entries.clear()
            else:
                existing = self._entries.get(identity.digest)
                if existing is None or existing[0] != identity.canonical_json:
                    removed = 0
                else:
                    del self._entries[identity.digest]
                    removed = 1
            self._invalidations += removed
            return removed

    def snapshot(self) -> PipelineCacheSnapshot:
        with self._condition:
            return PipelineCacheSnapshot(
                self._capacity,
                len(self._entries),
                self._hits,
                self._misses,
                self._bypasses,
                self._invalidations,
                self._evictions,
                self._collisions,
                self._computations,
                self._failed_compilations,
                self._waits,
            )


def _records(
    pipeline: PipelineResult,
    source_identity: str,
    pass_names_by_index: tuple[str, ...],
) -> tuple[PipelineDiagnosticRecord, ...]:
    pass_names = {
        id(diagnostic): result_index
        for result_index, result in enumerate(pipeline.pass_results)
        for diagnostic in result.diagnostics
    }
    records = []
    for diagnostic in pipeline.diagnostics:
        result_index = pass_names.get(id(diagnostic))
        pass_name = None
        stage = "pipeline_contract"
        if result_index is not None:
            stage = "pass"
            pass_name = pass_names_by_index[result_index]
        records.append(
            PipelineDiagnosticRecord(
                stage,
                source_identity,
                diagnostic=diagnostic,
                pass_name=pass_name,
                cause_type=type(diagnostic).__name__,
                cause_message=diagnostic.message,
            )
        )
    return tuple(records)


class CachedPipelineRunner:
    def __init__(self, pass_manager: PassManager, cache: BoundedPipelineCache) -> None:
        self._pass_manager = pass_manager
        self._cache = cache

    def run(
        self,
        module: QuantumModule,
        inputs: CompilationIdentityInputs,
    ) -> CachedPipelineExecution:
        identity_result = build_compilation_identity(module, self._pass_manager, inputs)
        identity = identity_result.identity
        source_identity = (
            inputs.source_identity
            if isinstance(inputs.source_identity, str) and inputs.source_identity
            else module.program_identity
        )
        disposition = CacheDisposition.FAILURE_NOT_CACHED
        try:
            if identity is None:
                self._cache.note_bypass()
                pipeline = self._pass_manager.run(module)
                disposition = CacheDisposition.BYPASS
            else:
                pipeline, disposition = self._cache.execute(
                    identity, lambda: self._pass_manager.run(module)
                )
        except Exception as exc:
            if identity is None:
                self._cache.note_failed_compilation()
            diagnostic = PipelineDiagnosticRecord(
                "pipeline_exception",
                str(source_identity),
                cause_type=type(exc).__name__,
                cause_message=str(exc),
            )
            return CachedPipelineExecution(
                PipelineExecutionStatus.FAILED,
                disposition,
                None,
                identity,
                (diagnostic,),
                self._cache.snapshot(),
            )
        records = tuple(
            PipelineDiagnosticRecord(
                "identity",
                str(source_identity),
                diagnostic=diagnostic,
                cause_type=type(diagnostic).__name__,
                cause_message=diagnostic.message,
            )
            for diagnostic in identity_result.diagnostics
        ) + _records(
            pipeline,
            str(source_identity),
            tuple(item.descriptor.name for item in self._pass_manager._passes),
        )
        status = (
            PipelineExecutionStatus.COMPLETED
            if pipeline.ok
            else PipelineExecutionStatus.FAILED
        )
        return CachedPipelineExecution(
            status,
            disposition,
            pipeline,
            identity,
            records,
            self._cache.snapshot(),
        )


__all__ = [
    "BoundedPipelineCache",
    "COMPILATION_IDENTITY_SCHEMA",
    "CacheDisposition",
    "CachedPipelineExecution",
    "CachedPipelineRunner",
    "CompilationIdentity",
    "CompilationIdentityInputs",
    "CompilationIdentityResult",
    "IdentityStatus",
    "PipelineCacheSnapshot",
    "PipelineDiagnosticRecord",
    "PipelineExecutionStatus",
    "build_compilation_identity",
]
