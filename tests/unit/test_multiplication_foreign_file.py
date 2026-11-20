"""Complete multiplication file SDK intake and native artifact boundaries."""

from itertools import product
from math import sqrt
from typing import Any

import pytest

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.core.ir import CircuitIR
from flagquantum.drawer.arithmetic_symbols import multiplication_wire_symbol
from flagquantum.drawer.component_score import _label
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.ecosystem.cirq._multiplication_arithmetic import multiplication_program
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_drawer_component_score import assert_source_geometry
from tests.unit.test_reciprocal_components import _reciprocal_image
from tests.unit.test_register_tensor import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _cases() -> tuple[tuple[str, Any], ...]:
    from qualtran.bloqs.arithmetic.multiplication import (
        InvertRealNumber,
        MultiplyTwoReals,
        PlusEqualProduct,
        Product,
        ScaleIntByReal,
        Square,
        SquareRealNumber,
        SumOfSquares,
    )

    return (
        ("product_accumulation", PlusEqualProduct(1, 1, 2)),
        ("integer_square", Square(1)),
        ("integer_product", Product(1, 1)),
        ("sum_of_squares", SumOfSquares(1, 2)),
        ("fixed_point_scaling", ScaleIntByReal(2, 1)),
        ("fixed_point_product", MultiplyTwoReals(2)),
        ("fixed_point_square", SquareRealNumber(3)),
        ("reciprocal", InvertRealNumber(2, 1)),
    )


def _image(name: str, source: Any, raw: int) -> int:
    values = {}
    cursor = raw
    for reg in reversed(tuple(source.signature)):
        values[reg.name] = cursor % (1 << reg.total_bits())
        cursor >>= reg.total_bits()
    if name == "product_accumulation":
        value = (values["result"] + values["a"] * values["b"]) % (
            1 << source.result_bitsize
        )
    elif name == "integer_square":
        value = values["a"] ** 2
    elif name == "integer_product":
        value = values["a"] * values["b"]
    elif name == "sum_of_squares":
        value = sum(
            ((values["input"] >> (i * source.bitsize)) % (1 << source.bitsize)) ** 2
            for i in range(source.k)
        )
    elif name in {"fixed_point_scaling", "fixed_point_product"}:
        left, right, width = (
            (values["real_in"], values["int_in"], source.i_bitsize)
            if name == "fixed_point_scaling"
            else (values["a"], values["b"], source.bitsize)
        )
        value = sum(
            left >> (j + 1) for j in range(width) if right & (1 << (width - 1 - j))
        )
    elif name == "fixed_point_square":
        n = source.bitsize
        value = sum(
            ((values["a"] >> (n - i)) & 1)
            * ((values["a"] >> (n - j)) & 1)
            * (1 << (n - i - j))
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i + j <= n
        )
    else:
        value = values["result"] ^ _reciprocal_image(
            values["a"], source.bitsize, source.num_frac
        )
    width = source.signature.get_right("result").total_bits()
    image = (raw >> width << width) | value
    assert isinstance(image, int)
    return image


@pytest.mark.parametrize("index", range(8))
def test_all_foreign_roots_native_channel_rebinding_and_serialization(
    index: int,
) -> None:
    name, source = _cases()[index]
    local = tuple(r.name for r in source.signature if r.side.value == 2)
    captured = import_register_emitter(source, local_ports=local)
    logical = sum(width for _, width in captured.layout)
    work, slots = captured.requirements()
    total = logical + work + 1
    # Exercise reverse physical placement and fresh measurement slots through
    # the existing named emitter. The final spectator stays entangled.
    mapping = tuple(reversed(range(logical + work)))
    offset = 0
    ports = {}
    for port, width in captured.layout:
        ports[port] = mapping[offset : offset + width]
        offset += width
    workspace = mapping[logical:]
    forward, backward = [
        captured(ports, workspace, tuple(range(7, 7 + slots)), inverse)
        for inverse in (False, True)
    ]
    circuit = (
        Circuit(total)
        .append(FieldOperation(total, forward))
        .append(FieldOperation(total, backward))
    )
    restored = CircuitIR.from_json(circuit.to_ir().to_json())
    assert restored.instructions == circuit.to_ir().instructions
    # Use the already accepted native channels as the image oracle, preserving
    # the original source value callback where it exists.
    valid_inputs = []
    input_regs = tuple(source.signature.lefts())
    for values in product(*(range(1 << (r.total_bits())) for r in input_regs)):
        raw = 0
        cursor = iter(values)
        for reg in source.signature:
            value = next(cursor) if reg.side.value & 1 else 0
            raw = (raw << reg.total_bits()) | value
        valid_inputs.append(raw)
    initial = {}
    expected = {}
    for raw, spectator in product(valid_inputs, (0, 1)):
        basis = spectator
        for index_wire, q in enumerate(mapping[:logical]):
            basis |= ((raw >> (logical - index_wire - 1)) & 1) << (total - q - 1)
        initial[basis] = complex(raw + 1, spectator - raw)
        image = _image(name, source, raw)
        target = spectator
        for index_wire, q in enumerate(mapping[:logical]):
            target |= ((image >> (logical - index_wire - 1)) & 1) << (total - q - 1)
        expected[target] = initial[basis]
    norm = sqrt(sum(abs(v) ** 2 for v in initial.values()))
    initial = {i: v / norm for i, v in initial.items()}
    expected = {i: v / norm for i, v in expected.items()}
    for seed in (2, 31):
        produced = execute_sparse_register(
            CircuitIR(total, forward), initial, seed=seed
        )
        assert all(
            abs(produced.get(i, 0) - expected.get(i, 0)) < 1e-11
            for i in produced.keys() | expected.keys()
        )
        actual = execute_sparse_register(restored, initial, seed=seed)
        assert all(
            abs(actual.get(i, 0) - initial.get(i, 0)) < 1e-11
            for i in actual.keys() | initial.keys()
        )
    # Canonical SDK snapshots retain primitive channel certificates and clean
    # scratch for all basis inputs. The forward target can remain live.
    canonical = captured.snapshots[0]
    basis_words = tuple(
        tuple((raw << work) >> q & 1 for q in reversed(range(logical + work)))
        for raw in valid_inputs
    )
    images, events = certified_and_channel_map(canonical.instructions, basis_words)
    assert events == sum(g.name == "measure" for g in canonical.instructions)
    assert all(not any(row[logical:]) for row in images)
    for raw, image_bits in zip(valid_inputs, images, strict=True):
        expected_raw = _image(name, source, raw)
        assert image_bits[:logical] == tuple(
            expected_raw >> q & 1 for q in reversed(range(logical))
        )


@pytest.mark.parametrize("index", range(8))
def test_complete_source_wire_geometry(index: int) -> None:
    from qualtran._infra.composite_bloq import BloqBuilder
    from qualtran.drawing.musical_score import get_musical_score_data

    name, source = _cases()[index]
    signature = native_signature(source.signature)
    left = tuple(_label(r.name, idx) for r in signature.lefts() for idx in r.all_idxs())
    right = tuple(
        _label(r.name, idx) for r in signature.rights() for idx in r.all_idxs()
    )
    step = RegisterStep(
        name,
        left,
        right,
        input_registers=tuple(signature.lefts()),
        output_registers=tuple(signature.rights()),
    )
    score = layout_register_score(
        {
            _label(r.name, idx): int(r.bitsize)
            for r in signature
            for idx in r.all_idxs()
        },
        left,
        (step,),
        outputs=right,
        input_registers=tuple(signature.lefts()),
        output_registers=tuple(signature.rights()),
    )
    builder, inputs = BloqBuilder.from_signature(source.signature)
    original = builder.finalize(**builder.add_d(source, **inputs))
    assert_source_geometry(
        register_score_data(score, register_wire_symbol=multiplication_wire_symbol),
        get_musical_score_data(original),
        ordered_ports=True,
    )


def test_original_rebinding_and_serialization_disposition() -> None:
    from qualtran.bloqs.arithmetic import multiplication
    from qualtran.testing import BloqCheckResult, check_bloq_example_serializes

    accumulator = multiplication.PlusEqualProduct(1, 1, 2)
    assert accumulator.registers() == ([2], [2], [2, 2])
    with pytest.raises(NotImplementedError, match="Not needed"):
        accumulator.with_registers([2], [2], [2, 2])
    factories = (
        "_plus_equal_product",
        "_square",
        "_sum_of_squares",
        "_product",
        "_scale_int_by_real",
        "_multiply_two_reals",
        "_square_real_number",
        "_invert_real_number",
    )
    for factory in factories:
        result, detail = check_bloq_example_serializes(getattr(multiplication, factory))
        assert result is BloqCheckResult.PASS, (factory, detail)


def test_external_accumulator_retains_adjoint_flag() -> None:
    from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct

    from flagquantum.compiler.registers.arithmetic.product_accumulation_component import (
        product_accumulation_program,
    )

    for flag, inverse in product((False, True), repeat=2):
        assert multiplication_program(
            PlusEqualProduct(1, 1, 2, flag), adjoint=inverse
        ) == product_accumulation_program(1, 1, 2, adjoint=flag != inverse)
