"""Original modular source graphs, exact feedback and complete residue channels."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.modular.controlled_modular_addition_component import (
    controlled_modular_addition_program,
)
from flagquantum.compiler.registers.modular.modular_addition_component import (
    modular_addition_program,
)
from flagquantum.compiler.registers.modular.modular_constant_component import (
    controlled_modular_constant_program,
)
from flagquantum.compiler.registers.modular.scaled_modular_addition_component import (
    scaled_modular_addition_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    unsigned_register_type,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def fixture(
    kind: str, width: int, modulus: int, value: int = 1, cv: int = 1
) -> tuple[ProgramRecord, Any]:
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_addition import (
        CModAdd,
        CModAddK,
        CtrlScaleModAdd,
        ModAdd,
    )

    if kind == "mod":
        return modular_addition_program(width, modulus), ModAdd(width, modulus)
    if kind == "constant":
        return controlled_modular_constant_program(width, modulus, value), CModAddK(
            value, modulus, width
        )
    if kind == "scaled":
        return scaled_modular_addition_program(width, modulus, value), CtrlScaleModAdd(
            value, modulus, width
        )
    dtype = (
        montgomery_register_type(width, modulus)
        if kind == "montgomery"
        else unsigned_register_type(width)
    )
    source_dtype = (
        QMontgomeryUInt(width, modulus) if kind == "montgomery" else QUInt(width)
    )
    return controlled_modular_addition_program(
        dtype, modulus, control_value=cv
    ), CModAdd(source_dtype, modulus, cv)


def bind_case(
    record: ProgramRecord, *, spectator: bool = False
) -> tuple[BoundRegisterProgram, SourcePorts, int]:
    ports: SourcePorts = {}
    logical = 0
    for register in record.functions[0].registers or ():
        width = int(register.bitsize)
        ports[register.name] = tuple(range(logical, logical + width))
        logical += width
    pool = tuple(range(logical + spectator, logical + spectator + 4 * logical + 6))
    return (
        bind_register_program(record, tuple(ports.values()), clean=pool),
        ports,
        logical,
    )


def source_stream(
    source: Any, bound: BoundRegisterProgram, ports: SourcePorts
) -> tuple[Instruction, ...]:
    from qualtran.bloqs.mcmt import And
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAddK, CtrlScaleModAdd

    slots = MeasurementSlots(bound.circuit.instructions)
    if not isinstance(source, CtrlScaleModAdd):
        plan = tuple(w for e in bound.events if e.kind == "allocate" for w in e.wires)
        return reference_arithmetic_gates(
            source,
            ports,
            plan,
            slots,
            reuse_allocations=False,
            preserve_components=True,
        )
    cursor = 0

    def leaf(bloq: Any, old: SourcePorts, desired: SourcePorts) -> SourceResult | None:
        nonlocal cursor
        if isinstance(bloq, CModAddK):
            # A separate source expansion only determines this call's extent.
            # Actual physical names below come from this call's native lifetimes.
            dummy = tuple(range(100, 100 + 20 * int(bloq.bitsize) + 20))
            count = len(
                reference_arithmetic_gates(
                    bloq,
                    old,
                    dummy,
                    {wire: wire for wire in dummy},
                    reuse_allocations=False,
                    preserve_components=True,
                )
            )
        elif isinstance(bloq, And):
            count = 4 if bloq.uncompute else 13
        else:
            return None
        plan = tuple(
            w
            for event in bound.events
            if event.kind == "allocate"
            and event.instruction_offset is not None
            and cursor <= event.instruction_offset < cursor + count
            for w in event.wires
        )
        returned = dict(old)
        if isinstance(bloq, And):
            if bloq.uncompute:
                returned.pop("target")
            else:
                returned["target"] = plan[:1]
        expected = reference_arithmetic_gates(
            bloq.as_composite_bloq() if isinstance(bloq, And) else bloq,
            old,
            plan,
            slots,
            expected_outputs=returned,
            reuse_allocations=False,
            preserve_components=True,
        )
        assert len(expected) == count
        assert all(returned[key] == wires for key, wires in desired.items())
        cursor += count
        return expected, returned

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports and cursor == len(bound.circuit.instructions)
    return gates


@pytest.mark.parametrize(
    "width,modulus,value", [(1, 1, 0), (2, 2, 0), (2, 3, 1), (3, 5, 4), (8, 251, 173)]
)
@pytest.mark.parametrize(
    "kind,cv",
    [
        ("mod", 1),
        ("uint", 0),
        ("uint", 1),
        ("montgomery", 0),
        ("montgomery", 1),
        ("constant", 1),
        ("scaled", 1),
    ],
)
def test_complete_source_calls_gates_feedback_and_resources(
    width: int, modulus: int, value: int, kind: str, cv: int
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    record, source = fixture(kind, width, modulus, value, cv)
    compare_layers(
        record, source, {f.name for f in record.functions if f.name != "main"}
    )
    bound, ports, logical = bind_case(record)
    assert bound.outputs == tuple(ports.values())
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))
    counts = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == counts["reset"] == model.measurement
    active: set[int] = set()
    peak = 0
    for event in bound.events:
        if event.kind == "allocate":
            active.update(event.wires)
        else:
            active.difference_update(event.wires)
        peak = max(peak, len(active))
    assert not active
    assert logical + peak == get_cost_value(source, QubitCount())


def residue_labels(
    kind: str, width: int, modulus: int, value: int, cv: int
) -> tuple[list[int], list[int]]:
    before, after = [], []
    if kind == "constant":
        for ctrl, y in product((0, 1), range(modulus)):
            before.append((ctrl << width) | y)
            after.append((ctrl << width) | ((y + value) % modulus if ctrl else y))
        return before, after
    controls = (1,) if kind == "mod" else (0, 1)
    xs = (
        range(1 << width)
        if kind == "scaled"
        else range(modulus + (modulus < 1 << width))
    )
    for ctrl, x, y in product(controls, xs, range(modulus)):
        old = (x << width) | y
        new = (x << width) | (
            (y + x * (value if kind == "scaled" else 1)) % modulus if ctrl == cv else y
        )
        if kind != "mod":
            old |= ctrl << (2 * width)
            new |= ctrl << (2 * width)
        before.append(old)
        after.append(new)
    return before, after


@pytest.mark.parametrize(
    "kind,cv",
    [
        ("mod", 1),
        ("uint", 0),
        ("uint", 1),
        ("montgomery", 0),
        ("montgomery", 1),
        ("constant", 1),
        ("scaled", 1),
    ],
)
@pytest.mark.parametrize("width,modulus,value", [(1, 1, 0), (2, 3, 1), (3, 5, 4)])
def test_all_valid_columns_all_erasure_branches_and_clean_work(
    kind: str, cv: int, width: int, modulus: int, value: int
) -> None:
    record, _ = fixture(kind, width, modulus, value, cv)
    bound, _, logical = bind_case(record)
    n = bound.circuit.n_wires
    before, after = residue_labels(kind, width, modulus, value, cv)

    def bits(labels: list[int]) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple((label >> (logical - 1 - i)) & 1 for i in range(logical))
            + (0,) * (n - logical)
            for label in labels
        )

    mapped, events = certified_and_channel_map(bound.circuit.instructions, bits(before))
    assert mapped == bits(after)
    assert events == sum(g.name == "measure" for g in bound.circuit.instructions)


@pytest.mark.parametrize(
    "kind,cv", [("mod", 1), ("uint", 0), ("uint", 1), ("constant", 1), ("scaled", 1)]
)
@pytest.mark.parametrize("width", [1, 2])
def test_full_complex_amplitudes_spectator_raw_histories_and_append(
    kind: str, cv: int, width: int
) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    modulus, value = (1, 0) if width == 1 else (3, 1)
    record, _ = fixture(kind, width, modulus, value, cv)
    bound, _, logical = bind_case(record, spectator=True)
    before, after = residue_labels(kind, width, modulus, value, cv)
    n = max(w for g in bound.circuit.instructions for w in g.wires) + 1
    scratch = n - logical - 1
    labels = [((i << 1) | s) << scratch for i, s in product(before, (0, 1))]
    mapped = [((i << 1) | s) << scratch for i, s in product(after, (0, 1))]
    raw = torch.arange(1, len(labels) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 4 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[labels], expected[mapped] = amplitudes, amplitudes
    measured = sum(g.name == "measure" for g in bound.circuit.instructions)
    histories = (
        product((0, 1), repeat=measured)
        if measured <= 7
        else ((0,) * measured, (1,) * measured, tuple(i % 2 for i in range(measured)))
    )
    for history in histories:
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            expected / 2 ** (measured / 2),
            rtol=0,
            atol=3e-12,
        )
    circuit = DynamicCircuit(n, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(n, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=4, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), expected.expand(4, -1), rtol=0, atol=3e-12
    )


@pytest.mark.parametrize("kind", ["mod", "uint", "constant", "scaled"])
def test_source_full_ring_loader_failure_remains_explicit(kind: str) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_addition import (
        CModAdd,
        CModAddK,
        CtrlScaleModAdd,
        ModAdd,
    )

    sources = {
        "mod": ModAdd(2, 4),
        "uint": CModAdd(QUInt(2), 4),
        "constant": CModAddK(1, 4, 2),
        "scaled": CtrlScaleModAdd(1, 4, 2),
    }
    with pytest.raises(ValueError, match="Too-large"):
        sources[kind].decompose_bloq().flatten()
    with pytest.raises(ValueError):
        fixture(kind, 2, 4)


def test_encoded_out_of_domain_constant_keeps_source_gate_model_disagreement() -> None:
    record, source = fixture("constant", 2, 2, 3)
    bound, ports, logical = bind_case(record)
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))
    n = max(w for gate in bound.circuit.instructions for w in gate.wires) + 1
    old = (1, 0, 1, *(0,) * (n - logical))
    (actual,), events = certified_and_channel_map(bound.circuit.instructions, (old,))
    assert source.call_classically(ctrl=1, x=1) == (1, 0)
    assert actual == (1, 1, 0, *(0,) * (n - logical)) and events == 11
    from examples.algorithms.modular_addition_catalog import ModularAdditionExample
    from flagquantum.dynamic import DynamicCircuit

    with pytest.raises(ValueError, match="constant"):
        DynamicCircuit(13).append(ModularAdditionExample("CModAddK", 2, 2, 3))


@pytest.mark.parametrize("kind", ["ModAdd", "CModAdd", "CModAddK", "CtrlScaleModAdd"])
def test_existing_append_consumer_replaces_bounded_emit_path(
    kind: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.algorithms import modular_addition_catalog as catalog
    from flagquantum.dynamic import DynamicCircuit

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Bounded append must use its stored component graph")

    from importlib import import_module

    for module, name in (
        ("modular_addition", "modular_add_ir"),
        ("controlled_modular_addition", "controlled_modular_add_ir"),
        ("modular_constant_addition", "controlled_modular_add_constant_ir"),
        ("modular_scaled_addition", "controlled_modular_scaled_add_ir"),
    ):
        owner = import_module("flagquantum.compiler.registers.modular." + module)
        monkeypatch.setattr(owner, name, forbidden)
    example = catalog.ModularAdditionExample(kind, 2, 3, 1)
    work, _ = example.workspace_requirements()
    logical = sum(int(r.bitsize) for r in example.signature)
    circuit = DynamicCircuit(logical + work)
    circuit.append(example)
    native_kind = {
        "ModAdd": "mod",
        "CModAdd": "uint",
        "CModAddK": "constant",
        "CtrlScaleModAdd": "scaled",
    }[kind]
    graph, _ = fixture(native_kind, 2, 3)
    expected, _, _ = bind_case(graph)
    assert circuit.to_ir().instructions == expected.circuit.instructions


@pytest.mark.parametrize(
    "name", ["cmodadd_example", "cmod_add_k_small", "ctrl_scale_mod_add_small"]
)
def test_actual_concrete_source_factories_five_owned_modes(name: str) -> None:
    from qualtran.bloqs.mod_arithmetic import mod_addition

    from tests.unit.register_example_checks import (
        CHECKS,
        CheckStatus,
        RegisterExample,
        check_example,
    )

    source = getattr(mod_addition, "_" + name).make()
    if name == "cmodadd_example":
        graph = controlled_modular_addition_program(
            unsigned_register_type(int(source.dtype.bitsize)), int(source.mod)
        )
    elif name == "cmod_add_k_small":
        graph = controlled_modular_constant_program(
            int(source.bitsize), int(source.mod), int(source.k)
        )
    else:
        graph = scaled_modular_addition_program(
            int(source.bitsize), int(source.mod), int(source.k)
        )
    bound, ports, _ = bind_case(graph)
    expected = source_stream(source, bound, ports)
    example = RegisterExample(
        lambda: graph,
        lambda record: bind_case(record)[0].circuit,
        lambda: Counter(g.name for g in expected),
    )
    for kind in CHECKS:
        result = check_example(example, kind)
        assert result.status is CheckStatus.PASS, result.message


def test_complete_frozen_definition_dispositions() -> None:
    import ast
    from pathlib import Path

    from qualtran.bloqs.mod_arithmetic import mod_addition

    audit = (
        Path(__file__).parents[2] / "docs/development/MODULAR_ADDITION_FILE_AUDIT.md"
    ).read_text()
    count = 0
    for node in ast.parse(Path(mod_addition.__file__).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            assert f"`{node.name}`" in audit
            count += 1
            for method in node.body if isinstance(node, ast.ClassDef) else ():
                if isinstance(method, ast.FunctionDef):
                    assert f"`{node.name}.{method.name}`" in audit
                    count += 1
    assert count == 40


@pytest.mark.parametrize("scale", [-7, 0, 19])
def test_signed_scale_preserves_original_constants_and_zero_stages(scale: int) -> None:
    record, source = fixture("scaled", 3, 5, scale)
    bound, ports, _ = bind_case(record)
    compare_layers(
        record, source, {f.name for f in record.functions if f.name != "main"}
    )
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))


def test_target_at_modulus_retains_original_dirty_sign() -> None:
    record, source = fixture("mod", 2, 3)
    bound, ports, logical = bind_case(record)
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))
    assert source.on_classical_vals(x=0, y=3) == {"x": 0, "y": 0}
    with pytest.raises((ValueError, AssertionError)):
        source.decompose_bloq().call_classically(x=0, y=3)
    n = max(w for gate in bound.circuit.instructions for w in gate.wires) + 1
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    initial[3 << (n - logical)] = 1
    actual = forced_branch(initial, bound.circuit.instructions, (0,) * 7)
    sign = bound.events[-1].wires[0]
    expected = torch.zeros_like(initial)
    expected[1 << (n - 1 - sign)] = 2 ** (-7 / 2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)


@pytest.mark.parametrize("kind", ["ModAdd", "CModAdd", "CModAddK", "CtrlScaleModAdd"])
def test_existing_full_ring_append_keeps_original_rejection(kind: Any) -> None:
    from examples.algorithms.modular_addition_catalog import ModularAdditionExample
    from flagquantum.dynamic import DynamicCircuit

    example = ModularAdditionExample(kind, 2, 4, 1)
    logical = sum(int(r.bitsize) for r in example.signature)
    work, _ = example.workspace_requirements()
    with pytest.raises(
        ValueError, match="Modulus must be a positive integer fitting the operand"
    ):
        DynamicCircuit(logical + work).append(example)
