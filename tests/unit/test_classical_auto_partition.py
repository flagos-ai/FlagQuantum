"""Original AutoPartition composition with existing numerical and gate consumers."""

from functools import partial
from itertools import product

import numpy as np
import pytest

from flagquantum.simulation.classical_auto_partition import (
    PackedValueBinding,
    auto_partition_classical_values,
)
from flagquantum.simulation.classical_partition import PartitionValuePort
from tests.unit.test_classical_comparison import same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def native_port(name, width):
    from flagquantum.simulation.integer_encoding import (
        integer_array_from_bits,
        integer_array_to_bits,
        integer_from_bits,
    )

    return PartitionValuePort(
        name,
        width,
        (),
        partial(integer_from_bits, width=width),
        partial(integer_array_from_bits, width=width),
        partial(integer_array_to_bits, width=width),
    )


def configuration(kind):
    from qualtran import QAny, QUInt, Register
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.bookkeeping.auto_partition import AutoPartition, Unused

    from flagquantum.compiler.registers.auto_partition import (
        AutoPartition as NativeAutoPartition,
    )
    from flagquantum.simulation.integer_encoding import (
        integer_from_bits,
        integer_to_bits,
    )

    groups = {
        "separate": (("left", (("a", 2),)), ("right", (("b", 2),))),
        "unused": (("packed", (("a", 2), (None, 1), ("b", 2))),),
        "left_only": (("packed", (("a", 2), ("b", 2))),),
        "spare": (("packed", (("a", 2), ("b", 2))), ("spare", ((None, 1),))),
    }[kind]
    bindings, source_parts = [], []
    for index, (outer, fields) in enumerate(groups):
        width = sum(n for _, n in fields)
        ports = tuple(
            native_port(name if name is not None else f"_unused{index}_{i}", n)
            for i, (name, n) in enumerate(fields)
        )
        bindings.append(
            PackedValueBinding(
                outer,
                ports,
                partial(integer_to_bits, width=width),
                partial(integer_from_bits, width=width),
            )
        )
        source_parts.append(
            (
                Register(outer, QAny(width)),
                [name if name is not None else Unused(n) for name, n in fields],
            )
        )
    left_only = kind == "left_only"
    source = AutoPartition(Add(QUInt(2), QUInt(2)), source_parts, left_only=left_only)
    return source, tuple(bindings), NativeAutoPartition(groups, left_only=left_only)


def add_child(values):
    from flagquantum.simulation.classical_addition import register_add_classical_values

    left, right = register_add_classical_values(values["a"], values["b"], 2)
    return {"a": left, "b": right}


@pytest.mark.parametrize("kind", ["separate", "unused", "left_only", "spare"])
def test_all_small_original_composite_classical_values_and_unused_preservation(kind):
    source, bindings, _ = configuration(kind)
    composite = source.decompose_bloq()
    widths = [
        sum(int(np.prod(port.shape + (port.width,))) for port in binding.ports)
        for binding in bindings
    ]
    for words in product(*(range(1 << n) for n in widths)):
        values = dict(zip((binding.name for binding in bindings), words, strict=True))
        expected = composite.call_classically(**values)
        actual = auto_partition_classical_values(
            values, bindings, ("a", "b"), add_child, left_only=kind == "left_only"
        )
        names = [reg.name for reg in source.signature.rights()]
        for name, value in zip(names, expected, strict=True):
            same_value(actual[name], value)
        assert set(actual) == set(names)


def test_partition_child_restore_order_and_real_component_replacement(monkeypatch):
    import flagquantum.simulation.classical_auto_partition as module

    _, bindings, _ = configuration("separate")
    events = []
    original_split, original_join = (
        module.partition_classical_value,
        module.unpartition_classical_values,
    )

    def split(value, encode, ports):
        events.append(("split", ports[0].name))
        return original_split(value, encode, ports)

    def join(values, ports, decode):
        events.append(("join", ports[0].name))
        return original_join(values, ports, decode)

    def child(values):
        events.append(("child", tuple(values)))
        return add_child(values)

    monkeypatch.setattr(module, "partition_classical_value", split)
    monkeypatch.setattr(module, "unpartition_classical_values", join)
    assert auto_partition_classical_values(
        {"left": 1, "right": 2}, bindings, ("a", "b"), child
    ) == {"left": 1, "right": 3}
    assert events == [
        ("split", "a"),
        ("split", "b"),
        ("child", ("a", "b")),
        ("join", "a"),
        ("join", "b"),
    ]
    assert auto_partition_classical_values(
        {"left": 1, "right": 2}, bindings, ("a", "b"), lambda values: values
    ) == {"left": 1, "right": 2}


def test_left_only_returns_child_outputs_without_repacking(monkeypatch):
    import flagquantum.simulation.classical_auto_partition as module

    _, bindings, _ = configuration("left_only")
    marker = {"a": 7, "b": 9}

    def forbidden(*args):
        pytest.fail("left-only path must not restore packed words")

    monkeypatch.setattr(module, "unpartition_classical_values", forbidden)
    assert (
        auto_partition_classical_values(
            {"packed": 3}, bindings, ("a", "b"), lambda values: marker, left_only=True
        )
        is marker
    )


def test_measured_add_original_stream_complete_complex_branches_clean_work_and_model():
    from dataclasses import asdict

    import torch
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler._component_trace.resource_model import (
        estimate_component_resources,
    )
    from flagquantum.compiler.registers.arithmetic import add_ir
    from flagquantum.compiler.registers.arithmetic_resource_model import source_add_cost
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source, bindings, layout = configuration("unused")
    inputs, unused = layout.bind({"packed": (0, 1, 2, 3, 4)})
    assert layout.restore(inputs, unused) == {"packed": (0, 1, 2, 3, 4)}
    gates = add_ir(inputs["a"], inputs["b"], (5,), (0,))
    expected_stream = reference_arithmetic_gates(
        source, {"packed": (0, 1, 2, 3, 4)}, (5,), {5: 0}
    )
    assert_source(gates, expected_stream)
    grid = torch.arange(1, 65, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, spectator in product(range(32), (0, 1)):
        output = auto_partition_classical_values(
            {"packed": word}, bindings, ("a", "b"), add_child
        )["packed"]
        initial[(word << 2) | spectator] = amplitudes[2 * word + spectator]
        expected[(output << 2) | spectator] = amplitudes[2 * word + spectator]
    for history in ((0,), (1,)):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(actual, expected / 2**0.5, rtol=0, atol=2e-12)
        residual = actual.reshape(32, 2, 2)[:, 1, :]
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), rtol=0, atol=2e-12
        )
    for legacy in (False, True):
        child = estimate_component_resources(source_add_cost(2, 2, legacy_shims=legacy))
        model = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        assert {
            name: value for name, value in asdict(child.gates).items() if value
        } == model.asdict()
        assert layout.logical_width(child.peak_logical_qubits) == get_cost_value(
            source, QubitCount()
        )
        assert model.total_t_count() == sum(g.name in ("t", "tdg") for g in gates) == 4
        assert model.measurement == sum(g.name == "measure" for g in gates) == 1
