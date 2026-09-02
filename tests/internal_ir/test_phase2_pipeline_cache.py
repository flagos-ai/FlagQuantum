from __future__ import annotations

import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.diagnostics import Diagnostic, DiagnosticCode
from flagquantum._compiler.exporters.circuit_ir import seal_circuit_ir_round_trip
from flagquantum._compiler.passes.base import PassDescriptor, PassResult
from flagquantum._compiler.passes.manager import PassManager
from flagquantum._compiler.passes.static_canonicalization import (
    StaticCanonicalizationPass,
)
from flagquantum._compiler.pipeline_cache import (
    BoundedPipelineCache,
    CacheDisposition,
    CachedPipelineRunner,
    CompilationIdentityInputs,
    IdentityStatus,
    PipelineExecutionStatus,
    build_compilation_identity,
)

pytestmark = pytest.mark.unit


def _module(theta: object = 0.2):
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("rx", (0,), {"theta": theta}),
            fq.Instruction("cx", (0, 1)),
        ),
    )
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None
    return sealed.artifact.imported.module


def _inputs(**changes: object) -> CompilationIdentityInputs:
    values = {
        "source_identity": "a" * 64,
        "target_profile": "universal_rx_ry_rz_cx_v1",
        "topology_identity": "b" * 64,
        "calibration_identity": "c" * 64,
        "compile_options": {"optimization_level": 2},
    }
    values.update(changes)
    return CompilationIdentityInputs(**values)


def _manager() -> PassManager:
    return PassManager((StaticCanonicalizationPass(),))


def test_compilation_identity_is_deterministic_and_covers_every_input() -> None:
    module = _module()
    manager = _manager()
    first = build_compilation_identity(module, manager, _inputs())
    second = build_compilation_identity(module, manager, _inputs())

    assert first.status is IdentityStatus.CACHEABLE
    assert first == second and first.identity is not None
    baseline = first.identity.digest
    variants = (
        (_module(0.3), manager, _inputs()),
        (module, PassManager(()), _inputs()),
        (module, manager, _inputs(source_identity="d" * 64)),
        (module, manager, _inputs(target_profile="other-target-v1")),
        (module, manager, _inputs(topology_identity="e" * 64)),
        (module, manager, _inputs(calibration_identity="f" * 64)),
        (module, manager, _inputs(compile_options={"optimization_level": 3})),
    )
    for variant_module, variant_manager, inputs in variants:
        result = build_compilation_identity(variant_module, variant_manager, inputs)
        assert result.identity is not None
        assert result.identity.digest != baseline


def test_unknown_nondeterministic_and_runtime_bound_inputs_bypass_cache() -> None:
    manager = _manager()
    unknown = build_compilation_identity(
        _module(), manager, _inputs(topology_identity=None)
    )
    unordered = build_compilation_identity(
        _module(), manager, _inputs(compile_options={"devices": {0, 1}})
    )
    runtime_bound = build_compilation_identity(
        _module(torch.tensor(0.2, requires_grad=True)), manager, _inputs()
    )

    for result in (unknown, unordered, runtime_bound):
        assert result.status is IdentityStatus.BYPASS_WITH_DIAGNOSTICS
        assert result.identity is None
        assert result.diagnostics


def test_cache_hit_invalidation_and_lru_eviction_are_exact() -> None:
    cache = BoundedPipelineCache(max_entries=2)
    runner = CachedPipelineRunner(_manager(), cache)
    modules = (_module(0.1), _module(0.2), _module(0.3))

    first = runner.run(modules[0], _inputs(source_identity="1" * 64))
    hit = runner.run(modules[0], _inputs(source_identity="1" * 64))
    runner.run(modules[1], _inputs(source_identity="2" * 64))
    runner.run(modules[2], _inputs(source_identity="3" * 64))

    assert first.cache_disposition is CacheDisposition.MISS
    assert hit.cache_disposition is CacheDisposition.HIT
    assert hit.pipeline is first.pipeline
    assert hit.pipeline is not None and hit.pipeline.module == first.pipeline.module
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(hit.pipeline.module, "revision", 99)
    assert cache.snapshot().evictions == 1
    assert cache.snapshot().size == 2
    assert first.identity is not None
    assert cache.invalidate(first.identity) == 0
    assert cache.invalidate() == 2
    assert cache.snapshot().invalidations == 2


def test_digest_collision_never_returns_the_wrong_pipeline() -> None:
    cache = BoundedPipelineCache(max_entries=2)
    manager = _manager()
    left_module = _module(0.1)
    right_module = _module(0.4)
    left = build_compilation_identity(
        left_module, manager, _inputs(source_identity="1" * 64)
    )
    right = build_compilation_identity(
        right_module, manager, _inputs(source_identity="2" * 64)
    )
    assert left.identity is not None and right.identity is not None

    left_pipeline, _ = cache.execute(left.identity, lambda: manager.run(left_module))
    collision = dataclasses.replace(right.identity, digest=left.identity.digest)
    right_pipeline, disposition = cache.execute(
        collision, lambda: manager.run(right_module)
    )

    assert disposition is CacheDisposition.BYPASS
    assert (
        right_pipeline.module.program_identity != left_pipeline.module.program_identity
    )
    assert cache.snapshot().collisions == 1
    assert cache.snapshot().size == 1


class _DiagnosticPass:
    descriptor = PassDescriptor(
        "test.diagnostic", "1", program_identity_policy="preserve"
    )

    def run(self, module) -> PassResult:
        return PassResult(
            module,
            diagnostics=(Diagnostic(DiagnosticCode.UNKNOWN_OPERATION, "deliberate"),),
        )


class _ExceptionPass:
    descriptor = PassDescriptor(
        "test.exception", "1", program_identity_policy="preserve"
    )

    def run(self, module) -> PassResult:
        del module
        raise RuntimeError("root cause retained")


def test_failures_are_not_cached_and_diagnostics_preserve_pass_and_cause() -> None:
    cache = BoundedPipelineCache()
    runner = CachedPipelineRunner(PassManager((_DiagnosticPass(),)), cache)
    first = runner.run(_module(), _inputs())
    second = runner.run(_module(), _inputs())

    assert first.status is PipelineExecutionStatus.FAILED
    assert second.cache_disposition is CacheDisposition.FAILURE_NOT_CACHED
    assert first.diagnostics[0].stage == "pass"
    assert first.diagnostics[0].pass_name == "test.diagnostic"
    assert first.diagnostics[0].cause_message == "deliberate"
    assert cache.snapshot().size == 0
    assert cache.snapshot().computations == 2
    assert cache.snapshot().failed_compilations == 2


def test_exceptions_are_not_cached_and_preserve_root_cause() -> None:
    cache = BoundedPipelineCache()
    result = CachedPipelineRunner(PassManager((_ExceptionPass(),)), cache).run(
        _module(), _inputs()
    )

    assert result.status is PipelineExecutionStatus.FAILED
    assert result.pipeline is None
    assert result.diagnostics[0].stage == "pipeline_exception"
    assert result.diagnostics[0].cause_type == "RuntimeError"
    assert result.diagnostics[0].cause_message == "root cause retained"
    assert cache.snapshot().failed_compilations == 1


class _BlockingPass:
    descriptor = PassDescriptor(
        "test.blocking", "1", program_identity_policy="preserve"
    )

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self._started = started
        self._release = release
        self.calls = 0

    def run(self, module) -> PassResult:
        self.calls += 1
        self._started.set()
        assert self._release.wait(timeout=2.0)
        return PassResult(module)


def test_concurrent_identical_requests_use_one_single_flight_compilation() -> None:
    started = threading.Event()
    release = threading.Event()
    compiler_pass = _BlockingPass(started, release)
    cache = BoundedPipelineCache()
    runner = CachedPipelineRunner(PassManager((compiler_pass,)), cache)
    module = _module()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(runner.run, module, _inputs())
        assert started.wait(timeout=2.0)
        second_future = executor.submit(runner.run, module, _inputs())
        deadline = time.monotonic() + 2.0
        while cache.snapshot().waits == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        release.set()
        first = first_future.result(timeout=2.0)
        second = second_future.result(timeout=2.0)

    assert {first.cache_disposition, second.cache_disposition} == {
        CacheDisposition.MISS,
        CacheDisposition.HIT,
    }
    assert compiler_pass.calls == 1
    assert cache.snapshot().computations == 1
    assert cache.snapshot().waits == 1


def test_bypass_executes_without_populating_cache_and_surface_stays_private() -> None:
    cache = BoundedPipelineCache()
    runner = CachedPipelineRunner(_manager(), cache)
    result = runner.run(_module(), _inputs(calibration_identity=""))

    assert result.ok
    assert result.cache_disposition is CacheDisposition.BYPASS
    assert result.identity is None
    assert result.diagnostics[0].stage == "identity"
    assert cache.snapshot().bypasses == 1
    assert cache.snapshot().size == 0
    assert not hasattr(fq, "BoundedPipelineCache")
    assert not hasattr(fq, "CachedPipelineRunner")
