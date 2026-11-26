"""Exact typed arithmetic inverses used by the original ECC stage closure."""

from collections import Counter
from importlib import import_module
from itertools import product

import pytest

from flagquantum.compiler.registers.comparison.linear_component import (
    linear_greater_program,
)
from flagquantum.compiler.registers.modular.modular_addition_component import (
    modular_addition_program,
)
from flagquantum.compiler.registers.modular.modular_doubling_component import (
    modular_doubling_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.register_types import montgomery_register_type
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_modular_subtraction_component import source_stream
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ("compare", "add", "double"))
@pytest.mark.parametrize("width,modulus", [(1, 1), (3, 3), (5, 13)])
def test_original_inverse_calls_stream_and_independent_cost(
    kind: str, width: int, modulus: int
) -> None:
    qt = import_module("qualtran")
    comparison = import_module("qualtran.bloqs.arithmetic.comparison")
    modular = import_module("qualtran.bloqs.mod_arithmetic")
    costs = import_module("qualtran.resource_counting")
    dtype = montgomery_register_type(width)
    if kind == "compare":
        graph, source = (
            linear_greater_program(width, adjoint=True),
            comparison.LinearDepthGreaterThan(width).adjoint(),
        )
    elif kind == "add":
        graph, source = (
            modular_addition_program(width, modulus, adjoint=True),
            modular.ModAdd(width, modulus).adjoint(),
        )
    else:
        graph, source = (
            modular_doubling_program(dtype, modulus, adjoint=True),
            modular.ModDbl(qt.QMontgomeryUInt(width), modulus).adjoint(),
        )
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    ports = {}
    offset = 0
    for register in graph.functions[0].registers or ():
        size = int(register.bitsize)
        ports[register.name] = tuple(range(offset, offset + size))
        offset += size
    bound = bind_register_program(
        graph,
        tuple(ports.values()),
        clean=tuple(range(offset, offset + 30)),
        input_ownership=tuple((w, False) for w in range(offset)),
        allow_workspace_outputs=True,
    )
    allocations = tuple(
        w for event in bound.events if event.kind == "allocate" for w in event.wires
    )
    expected = (
        source_stream(source, bound, ports)
        if kind == "add"
        else reference_arithmetic_gates(
            source,
            ports,
            allocations,
            MeasurementSlots(bound.circuit.instructions),
            expected_outputs=dict(zip(ports, bound.outputs, strict=True)),
            reuse_allocations=False,
            preserve_components=True,
        )
    )
    assert_source(bound.circuit.instructions, expected)
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = costs.get_cost_value(source, costs.QECGatesCost())
    assert counts["measure"] == model.measurement
    assert counts["t"] + counts["tdg"] == model.total_t_count()


@pytest.mark.parametrize("kind", ("compare", "add", "double"))
def test_complete_complex_inverse_columns_all_histories_and_work_zero(
    kind: str,
) -> None:
    width, modulus = 3, 3
    names: tuple[str, ...]
    sizes: tuple[int, ...]
    values: tuple[tuple[int, ...], ...]
    expected: tuple[tuple[int, ...], ...]
    if kind == "compare":
        graph = linear_greater_program(width, adjoint=True)
        names, sizes = ("a", "b", "target"), (3, 3, 1)
        values = tuple(product(range(8), range(8), range(2)))
        expected = tuple((a, b, target ^ (a > b)) for a, b, target in values)
    elif kind == "add":
        graph = modular_addition_program(width, modulus, adjoint=True)
        names, sizes = ("x", "y"), (3, 3)
        values = tuple(product(range(modulus), repeat=2))
        expected = tuple((x, (y - x) % modulus) for x, y in values)
    else:
        graph = modular_doubling_program(
            montgomery_register_type(width), modulus, adjoint=True
        )
        names, sizes = ("x",), (3,)
        values = tuple((x,) for x in range(modulus))
        expected = tuple((x * pow(2, -1, modulus) % modulus,) for (x,) in values)
    offset, ports = 0, {}
    for name, size in zip(names, sizes, strict=True):
        ports[name] = tuple(range(offset, offset + size))
        offset += size
    bound = bind_register_program(
        graph,
        tuple(ports.values()),
        clean=tuple(range(offset, offset + 30)),
        input_ownership=tuple((w, False) for w in range(offset)),
        allow_workspace_outputs=True,
    )
    used = (
        max((w for event in bound.events for w in event.wires), default=offset - 1) + 1
    )
    bound = bind_register_program(
        graph,
        tuple(ports.values()),
        clean=tuple(range(offset, used)),
        input_ownership=tuple((w, False) for w in range(offset)),
        allow_workspace_outputs=True,
    )

    def pack(words: tuple[int, ...], layout: tuple[tuple[int, ...], ...]) -> int:
        return sum(
            ((word >> (len(wires) - i - 1)) & 1) << (used - w - 1)
            for word, wires in zip(words, layout, strict=True)
            for i, w in enumerate(wires)
        )

    raw_before = tuple(pack(words, tuple(ports.values())) for words in values)
    raw_after = tuple(pack(words, bound.outputs) for words in expected)
    basis = tuple(
        tuple((word >> (used - w - 1)) & 1 for w in range(used)) for word in raw_before
    )
    mapped, _ = certified_and_channel_map(bound.circuit.instructions, basis)
    assert (
        tuple(sum(v << (used - i - 1) for i, v in enumerate(row)) for row in mapped)
        == raw_after
    )
    amplitudes = tuple(complex(i + 1, 5 - i) for i in range(len(values)))
    norm = sum(abs(value) ** 2 for value in amplitudes) ** 0.5
    before = {
        word: value / norm for word, value in zip(raw_before, amplitudes, strict=True)
    }
    after = {
        word: value / norm for word, value in zip(raw_after, amplitudes, strict=True)
    }
    for seed in (0, 19):
        assert execute_sparse_register(
            bound.circuit, before, seed=seed
        ) == pytest.approx(after, abs=2e-12)


def test_signed_linear_inverse_matches_complete_original_complex_operator() -> None:
    import torch

    comparison = import_module("qualtran.bloqs.arithmetic.comparison")
    source = comparison.LinearDepthGreaterThan(3, signed=True).adjoint()
    graph = linear_greater_program(3, signed=True, adjoint=True)
    bound = bind_register_program(graph, ((0, 1, 2), (3, 4, 5), (6,)), clean=(7, 8))
    columns = tuple(
        tuple(((word << 2) >> (8 - i)) & 1 for i in range(9)) for word in range(128)
    )
    mapped, events = certified_and_channel_map(bound.circuit.instructions, columns)
    assert events == 2
    operator = torch.zeros((128, 128), dtype=torch.complex128)
    for index, bits in enumerate(mapped):
        assert bits[7:] == (0, 0)
        word = sum(bit << (6 - i) for i, bit in enumerate(bits[:7]))
        operator[word, index] = 1
    torch.testing.assert_close(
        operator,
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=2e-13,
        rtol=0,
    )
