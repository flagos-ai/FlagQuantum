"""Construct locked raw-builder helper lifetimes through native owners."""

import json
from copy import copy, deepcopy
from pathlib import Path
from typing import TypedDict, cast

import pytest

from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from tests.unit.test_physical_circuit_binding import observe

pytestmark = pytest.mark.unit


class _State(TypedDict):
    columns: dict[str, list[int]]
    observation: dict[str, object]


_DATA = cast(
    dict[str, _State],
    json.loads(
        (Path(__file__).with_name("data") / "physical_builder_results.json").read_text()
    ),
)


def _snapshot(builder: KmxCircuitBuilder, name: str) -> None:
    expected = _DATA[name]
    assert observe(builder.freeze()) == expected["observation"]
    cursor = copy(builder).create_classical_register(1, "probe")[0].value
    assert cursor == expected["columns"]["next"][0]
    probe = copy(builder)
    assert [probe.result_bit().bit.value for _ in expected["columns"]["free"]] == (
        expected["columns"]["free"][::-1]
    )
    assert probe.result_bit().bit.value == cursor


def test_complete_original_result_lifecycle_and_executed_observations() -> None:
    builder = KmxCircuitBuilder()
    builder.alloc_bits(2)
    with builder.result_bit() as dirty:
        assert dirty.bit.value == 2
        builder.append_kmx_line("BIT_STORE1 b2")
    _snapshot(builder, "dirty_released")
    with builder.result_bit(clean=True) as clean:
        assert clean.bit.value == 2
        builder.append_kmx_line("X q0 if b2")
    _snapshot(builder, "clean_released")
    with builder.result_bit() as first, builder.result_bit() as second:
        assert (first.bit.value, second.bit.value) == (2, 3)
        builder.append_kmx_line("BIT_STORE1 b3")
    _snapshot(builder, "nested_released")
    with builder.measure_result(1, condition=True) as measured:
        assert measured.bit.value == 2
        builder.append_kmx_line("Z q2")
    _snapshot(builder, "measured_released")
    with builder.result_bit() as detached:
        assert detached.disarm().value == 2
        builder.append_kmx_line("BIT_STORE0 b2")
    _snapshot(builder, "detached")
    with builder.result_bit() as remaining:
        assert remaining.bit.value == 3
        builder.append_kmx_line("DEBUG_PRINT b3")
    _snapshot(builder, "remaining")
    fresh = KmxCircuitBuilder()
    with fresh.result_bit(clean=True) as clean:
        assert clean.bit.value == 0
        fresh.append_kmx_line("X q0 if b0")
    _snapshot(fresh, "fresh_clean")


def test_helper_and_explicit_reuse_banks_share_only_the_id_cursor() -> None:
    builder = KmxCircuitBuilder()
    explicit = builder.alloc_bits(1)
    builder.free(explicit)
    with builder.result_bit() as helper:
        assert helper.bit.value == 1
        assert builder.alloc_bits(1) == explicit
    assert builder.create_classical_register(1, "reserved")[0].value == 2
    with builder.result_bit(clean=True) as reused:
        assert reused.bit.value == 1
    assert builder.freeze().to_kmx().endswith("BIT_STORE0 b1")


def test_exception_exit_and_disarm_retain_scope_and_bit_ownership() -> None:
    builder = KmxCircuitBuilder()
    with (
        pytest.raises(RuntimeError, match="body failure"),
        builder.measure_result(0, condition=True) as lease,
    ):
        builder.append_kmx_line("Z q1")
        raise RuntimeError("body failure")
    lease.close()
    assert builder.freeze().to_kmx() == (
        "HMR q0 b0\nPUSH_CONDITION if b0\nZ q1\nPOP_CONDITION"
    )
    with pytest.raises(ValueError, match="cannot be entered"):
        lease.__enter__()
    with pytest.raises(ValueError, match="cannot transfer"):
        lease.disarm()
    detached = builder.measure_result(1, condition=True)
    bit = detached.disarm()
    detached.close()
    builder.append_kmx_line("POP_CONDITION")
    builder.free((bit,))
    with builder.result_bit() as reused:
        assert reused.bit == bit
    builder.freeze()


def test_result_owner_copy_and_cleared_builder_refuse_explicitly() -> None:
    builder = KmxCircuitBuilder()
    lease = builder.result_bit()
    for clone in (copy, deepcopy):
        with pytest.raises(TypeError, match="one owner"):
            clone(lease)
    builder.clear()
    replacement = builder.result_bit()
    for operation in (lease.close, lease.disarm):
        with pytest.raises(ValueError, match="cleared builder"):
            operation()
    assert replacement.bit.value == 0
    replacement.close()


@pytest.mark.parametrize("wire", [-1, 500_000_000, True])
def test_bad_measurement_target_does_not_consume_a_result(wire: int) -> None:
    builder = KmxCircuitBuilder()
    with pytest.raises(ValueError):
        builder.measure_result(wire)
    assert builder.result_bit().bit.value == 0
    assert len(builder.freeze()) == 0
