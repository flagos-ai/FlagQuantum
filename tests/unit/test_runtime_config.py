import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import flagquantum as fq
from flagquantum.core.runtime_config import (
    RuntimeConfig,
    get_runtime_config,
    runtime_config,
)

pytestmark = pytest.mark.unit


def test_runtime_config_manifest_is_versioned_and_round_trips():
    config = RuntimeConfig(
        device="cuda",
        complex_dtype="complex128",
        real_dtype="float64",
        jax_enable_x64=True,
    )
    manifest = config.to_manifest()
    assert manifest["schema"] == "flagquantum_runtime_config"
    assert manifest["version"] == "1.0"
    assert RuntimeConfig.from_manifest(manifest) == config
    with pytest.raises(ValueError, match="schema"):
        RuntimeConfig.from_manifest({**manifest, "schema": "unknown"})


def test_nested_scopes_restore_without_precision_leakage():
    original = get_runtime_config()
    with runtime_config(complex_dtype="complex128") as outer:
        assert get_runtime_config().complex_dtype == "complex128"
        with runtime_config(device="cuda") as inner:
            assert inner.device == "cuda"
            assert inner.complex_dtype == "complex128"
        assert get_runtime_config() == outer
    assert get_runtime_config() == original


def test_concurrent_threads_build_independent_precision_circuits():
    def build(dtype):
        with runtime_config(complex_dtype=dtype):
            circuit = fq.Circuit(1).h(0)
            return circuit.dtype, circuit.state().dtype

    with ThreadPoolExecutor(max_workers=2) as pool:
        single = pool.submit(build, "complex64")
        double = pool.submit(build, "complex128")
    assert single.result() == (torch.complex64, torch.complex64)
    assert double.result() == (torch.complex128, torch.complex128)


def test_async_tasks_keep_independent_runtime_precision():
    async def build(dtype):
        with runtime_config(complex_dtype=dtype):
            await asyncio.sleep(0)
            return get_runtime_config().complex_dtype, fq.Circuit(1).x(0).state().dtype

    async def gather():
        return await asyncio.gather(build("complex64"), build("complex128"))

    assert asyncio.run(gather()) == [
        ("complex64", torch.complex64),
        ("complex128", torch.complex128),
    ]


def test_circuit_ir_and_plan_carry_reconstructable_configuration():
    config = RuntimeConfig(
        complex_dtype="complex128", real_dtype="float64", jax_enable_x64=True
    )
    circuit = fq.Circuit(2, config=config).h(0).cx(0, 1)
    manifest = circuit.to_ir().metadata["runtime_config"]
    assert RuntimeConfig.from_manifest(manifest) == config
    assert circuit.plan().summary()["runtime_config"] == manifest
