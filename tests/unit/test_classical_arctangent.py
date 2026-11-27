"""Complete original ArcTan scalar callback beyond the bounded quantum lookup."""

import builtins
from collections.abc import Iterable
from itertools import product
from typing import Any

from tests._reference_dependencies import unavailable_reference

try:
    import numpy as np
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)
import pytest

from flagquantum.ecosystem.classical_arctangent import (
    source_arctangent_classical_values,
)
from flagquantum.simulation.classical_arctangent import normalized_arctangent_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (0, 1, 2, 8, 32, 33, 53, 64, 80, 128))
def test_full_source_callback_quantization_occupied_outputs_and_self_inverse(
    width: int,
) -> None:
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

    original = ArcTan(256, width)
    values = (0, 1, -1, 2, 3, 7, 255, 4095, 4096, 8191, 2**32, 2**53, 2**60, 2**200)
    targets = (0, (1 << width) - 1, (1 << (width + 2)) - 1, -3)
    for value, sign, target in product(values, (0, 1, 3, -1), targets):
        try:
            expected = original.apply(value, sign, target)
        except (TypeError, ValueError, OverflowError, AssertionError) as error:
            with pytest.raises(type(error)):
                source_arctangent_classical_values(value, sign, target, width)
            continue
        actual = source_arctangent_classical_values(value, sign, target, width)
        assert actual == expected
        assert source_arctangent_classical_values(*actual, width) == (
            value,
            sign,
            target,
        )


def test_binary64_endpoint_retains_original_wrap_instead_of_clipping() -> None:
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

    for value in (2**60, -(2**60)):
        output = normalized_arctangent_value(value)
        assert abs(output) == 1
        expected = ArcTan(64, 32).apply(value, 0, 0)
        assert isinstance(expected, tuple)
        assert expected[2] == 0
        assert source_arctangent_classical_values(value, 0, 0, 32) == expected


@pytest.mark.parametrize("value", (2**1024, float("nan")))
def test_original_numeric_failures_remain_visible(value: Any) -> None:
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

    try:
        ArcTan(2048, 8).apply(value, 0, 0)
    except (TypeError, OverflowError, AssertionError) as error:
        with pytest.raises(type(error)):
            source_arctangent_classical_values(value, 0, 0, 8)
    else:
        pytest.fail("The frozen failure witness unexpectedly succeeded")


def test_symbolic_width_stays_a_declaration_not_a_numeric_encoding() -> None:
    import sympy
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan

    width = sympy.Symbol("p")
    with pytest.raises(ValueError):
        ArcTan(4, width).apply(1, 0, 0)
    with pytest.raises(ValueError):
        source_arctangent_classical_values(1, 0, 0, width)


def test_callback_runs_without_source_sdk_and_math_needs_no_external_codec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.ecosystem.fixed_point as codec

    original_import = builtins.__import__

    def source_free(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise ImportError("Reference imports are blocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", source_free)
    assert source_arctangent_classical_values(1, 1, 7, 33) == (1, 0, 7 ^ (1 << 32))

    def missing_codec(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(codec, "import_module", missing_codec)
    assert normalized_arctangent_value(1) == np.float64(-0.5)
    with pytest.raises(ModuleNotFoundError, match="fxpmath"):
        source_arctangent_classical_values(1, 0, 0, 8)


@pytest.mark.parametrize("left,right", ((0, 0), (0, 3), (1, 0), (1, 2), (2, 3), (3, 3)))
def test_callback_operator_matches_append_every_small_branch_and_zero_work(
    left: int, right: int
) -> None:
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.lookup.arctan_lookup_component import (
        arctan_lookup_program,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.measurement_checks import forced_branch

    logical = left + 1 + right
    ports = tuple(
        port
        for port in (
            tuple(range(left)),
            (left,),
            tuple(range(left + 1, logical)),
        )
        if port
    )
    work = tuple(range(logical + 1, logical + max(0, left - 1) + 1))
    bound = bind_register_program(arctan_lookup_program(left, right), ports, clean=work)
    stride = 1 << len(work)
    raw = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 3 - 5 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + 1 + len(work)), dtype=torch.complex128)
    initial[::stride] = amplitudes
    expected = torch.zeros_like(initial)
    for word in range(1 << logical):
        selection = word >> (right + 1)
        sign = word >> right & 1
        target = word & ((1 << right) - 1)
        x, s, y = source_arctangent_classical_values(selection, sign, target, right)
        destination = x << (right + 1) | s << right | y
        for spectator in (0, 1):
            expected[((destination << 1) | spectator) * stride] = amplitudes[
                (word << 1) | spectator
            ]
    total = logical + 1 + len(work)
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(total, bound.circuit.instructions))
    events = sum(g.name == "measure" for g in bound.circuit.instructions)
    for history in product((0, 1), repeat=events):
        branch = forced_branch(initial, circuit.to_ir().instructions, history)
        torch.testing.assert_close(
            branch, expected / 2 ** (events / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(branch.reshape(-1, stride)[:, 1:]) == 0


@pytest.mark.parametrize("left,right", ((13, 1), (1, 33)))
def test_generic_owned_qrom_covers_source_operator_beyond_lookup_profile(
    left: int, right: int
) -> None:
    from qualtran.bloqs.arithmetic.trigonometric.arctan import ArcTan
    from qualtran.bloqs.data_loading.qrom import QROM

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.lookup.qrom_component import qrom_program
    from flagquantum.compiler.registers.lookup.qrom_table import QromTable
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from flagquantum.dynamic import DynamicCircuit
    from tests.unit.measurement_checks import certified_and_channel_map
    from tests.unit.sparse_register_execution import execute_sparse_register
    from tests.unit.test_multi_control_component import MeasurementSlots
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = ArcTan(left, right)
    data = tuple(
        source_arctangent_classical_values(x, 0, 0, right) for x in range(1 << left)
    )
    assert data == tuple(source.apply(x, 0, 0) for x in range(1 << left))
    signs = tuple(row[1] for row in data)
    values = tuple(row[2] for row in data)
    table = QromTable.from_data(signs, values, target_bitsizes=(1, right))
    logical = left + 1 + right
    ports = (tuple(range(left)), (left,), tuple(range(left + 1, logical)))
    work = tuple(range(logical + 1, logical + left))
    bound = bind_register_program(
        qrom_program(table, reserve_workspace=True), ports, clean=work
    )
    original_qrom = QROM(
        (np.array(signs), np.array(values)), (left,), (1, right), ((), ())
    )
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(
            original_qrom,
            dict(zip(("selection", "target0_", "target1_"), ports, strict=True)),
            work,
            MeasurementSlots(bound.circuit.instructions),
            cirq_style=True,
        ),
    )
    words: Iterable[int]
    if right == 1:
        words = range(1 << logical)
    else:
        # The emitted X/CX-only channel is affine with no phase. Its zero and
        # basis-vector images determine the complete wide occupied-target map.
        assert all(g.name in {"x", "cx"} for g in bound.circuit.instructions)
        words = (0, *(1 << bit for bit in range(logical)))
    before, after = [], []
    for word in words:
        x, s, y = word >> (right + 1), word >> right & 1, word & ((1 << right) - 1)
        output = (x << (right + 1)) | ((s ^ signs[x]) << right) | (y ^ values[x])
        before.append(
            tuple(word >> bit & 1 for bit in reversed(range(logical)))
            + (0,) * (len(work) + 1)
        )
        after.append(
            tuple(output >> bit & 1 for bit in reversed(range(logical)))
            + (0,) * (len(work) + 1)
        )
    events = sum(g.name == "measure" for g in bound.circuit.instructions)
    assert certified_and_channel_map(bound.circuit.instructions, tuple(before)) == (
        tuple(after),
        events,
    )
    initial: dict[int, complex] = {}
    expected: dict[int, complex] = {}
    total = logical + 1 + len(work)
    for i in (0, 1, len(before) // 2, len(before) - 1):
        for spectator in (0, 1):
            a, b = list(before[i]), list(after[i])
            a[logical] = b[logical] = spectator
            amplitude = complex(i + 1, 3 - i - spectator)
            initial[int("".join(map(str, a)), 2)] = amplitude
            expected[int("".join(map(str, b)), 2)] = amplitude
    circuit = DynamicCircuit(total).append(
        FieldOperation(total, bound.circuit.instructions)
    )
    norm = sum(abs(v) ** 2 for v in initial.values()) ** 0.5
    initial = {index: value / norm for index, value in initial.items()}
    expected = {index: value / norm for index, value in expected.items()}
    observed = execute_sparse_register(circuit.to_ir(), initial, seed=41)
    assert all(
        abs(observed.get(i, 0) - expected.get(i, 0)) < 3e-12
        for i in observed.keys() | expected.keys()
    )
