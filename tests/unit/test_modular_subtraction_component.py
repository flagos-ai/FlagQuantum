"""Source-shaped modular negation and subtraction compose shared typed owners."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.modular.controlled_modular_negation_component import (
    controlled_modular_negation_program,
)
from flagquantum.compiler.registers.modular.controlled_modular_subtraction_component import (
    controlled_modular_subtraction_program,
)
from flagquantum.compiler.registers.modular.modular_negation_component import (
    modular_negation_program,
)
from flagquantum.compiler.registers.modular.modular_subtraction_component import (
    modular_subtraction_program,
)
from flagquantum.compiler.registers.register_binding import BoundRegisterProgram
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
from tests.unit.test_modular_addition_component import bind_case
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def fixture(
    kind: str, width: int, modulus: int, montgomery: bool, cv: int = 1
) -> tuple[ProgramRecord, Any]:
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import (
        CModNeg,
        CModSub,
        ModNeg,
        ModSub,
    )

    dtype = (
        montgomery_register_type(width, modulus)
        if montgomery
        else unsigned_register_type(width)
    )
    original = QMontgomeryUInt(width, modulus) if montgomery else QUInt(width)
    if kind == "neg":
        return modular_negation_program(dtype, modulus), ModNeg(original, modulus)
    if kind == "cneg":
        return controlled_modular_negation_program(
            dtype, modulus, control_value=cv
        ), CModNeg(original, modulus, cv)
    if kind == "sub":
        return modular_subtraction_program(dtype, modulus), ModSub(original, modulus)
    return controlled_modular_subtraction_program(
        dtype, modulus, control_value=cv
    ), CModSub(original, modulus, cv)


def source_stream(
    source: Any,
    bound: BoundRegisterProgram,
    ports: SourcePorts,
    *,
    expand_modular: bool = False,
) -> tuple[Instruction, ...]:
    from qualtran.bloqs.bookkeeping import Allocate, Cast, Free, Join, Split
    from qualtran.bloqs.mcmt import And
    from qualtran.bloqs.mod_arithmetic.mod_addition import CModAdd, ModAdd

    cursor = 0
    slots = MeasurementSlots(bound.circuit.instructions)

    def leaf(bloq: Any, old: SourcePorts, desired: SourcePorts) -> SourceResult | None:
        nonlocal cursor
        base = bloq.subbloq if type(bloq).__name__ == "Adjoint" else bloq
        if expand_modular and isinstance(base, (ModAdd, CModAdd)):
            return None
        if bloq is source or isinstance(bloq, (Join, Split, Cast)):
            return None
        if isinstance(bloq, Free):
            return (), {}
        if isinstance(bloq, Allocate):
            event = next(
                e
                for e in bound.events
                if e.kind == "allocate" and e.instruction_offset == cursor
            )
            return (), {"reg": event.wires}
        atomic = bloq.as_composite_bloq()
        start = max(w for wires in old.values() for w in wires) + 100
        dummy = tuple(range(start, start + 1000))
        returned = {
            r.name: old[r.name] for r in bloq.signature.rights() if r.name in old
        }
        dry_outputs = dict(returned)
        if isinstance(bloq, And) and not bloq.uncompute:
            dry_outputs["target"] = dummy[:1]
        dry_slots = {
            w: w for w in (*dummy, *(w for wires in old.values() for w in wires))
        }
        dry = reference_arithmetic_gates(
            atomic,
            old,
            dummy,
            dry_slots,
            expected_outputs=dry_outputs,
            reuse_allocations=False,
            preserve_components=True,
        )
        plan = tuple(
            w
            for e in bound.events
            if e.kind == "allocate"
            and e.instruction_offset is not None
            and cursor <= e.instruction_offset < cursor + len(dry)
            for w in e.wires
        )
        returned = {
            r.name: old[r.name] for r in bloq.signature.rights() if r.name in old
        }
        if isinstance(bloq, And) and not bloq.uncompute:
            returned["target"] = plan[:1]
        expected = reference_arithmetic_gates(
            atomic,
            old,
            plan,
            slots,
            expected_outputs=returned,
            reuse_allocations=False,
            preserve_components=True,
        )
        assert len(expected) == len(dry)
        assert all(returned[k] == v for k, v in desired.items())
        cursor += len(expected)
        return expected, returned

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports and cursor == len(bound.circuit.instructions)
    return gates


@pytest.mark.parametrize("width,modulus", [(2, 1), (2, 2), (3, 5), (5, 13), (8, 251)])
@pytest.mark.parametrize("montgomery", [False, True])
@pytest.mark.parametrize(
    "kind,cv",
    [("neg", 1), ("cneg", 0), ("cneg", 1), ("sub", 1), ("csub", 0), ("csub", 1)],
)
def test_full_source_graph_primitives_feedback_and_resources(
    width: int, modulus: int, montgomery: bool, kind: str, cv: int
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    graph, source = fixture(kind, width, modulus, montgomery, cv)
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    bound, ports, logical = bind_case(graph)
    assert_source(bound.circuit.instructions, source_stream(source, bound, ports))
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["t"] + counts["tdg"] == model.total_t_count()
    assert counts["measure"] == model.measurement
    active: set[int] = set()
    peak = 0
    for e in bound.events:
        if e.kind == "allocate":
            active.update(e.wires)
        else:
            active.difference_update(e.wires)
        peak = max(peak, len(active))
    assert not active
    assert logical + peak == get_cost_value(source, QubitCount())


def residue_labels(
    kind: str, width: int, modulus: int, cv: int
) -> tuple[list[int], list[int]]:
    before, after = [], []
    controlled = kind.startswith("c")
    subtract = kind.endswith("sub")
    logical = width * (2 if subtract else 1)
    for ctrl in (0, 1) if controlled else (cv,):
        for x in range(modulus):
            for y in range(modulus) if subtract else (0,):
                old = (x << width) | y if subtract else x
                value = (
                    ((y - x) % modulus if subtract else (-x) % modulus)
                    if ctrl == cv
                    else (y if subtract else x)
                )
                new = (x << width) | value if subtract else value
                before.append((ctrl << logical) | old if controlled else old)
                after.append((ctrl << logical) | new if controlled else new)
    return before, after


@pytest.mark.parametrize(
    "kind,cv",
    [("neg", 1), ("cneg", 0), ("cneg", 1), ("sub", 1), ("csub", 0), ("csub", 1)],
)
@pytest.mark.parametrize("montgomery", [False, True])
@pytest.mark.parametrize("width,modulus", [(2, 2), (3, 5)])
def test_all_residue_columns_every_erasure_branch_and_clean_work(
    kind: str, cv: int, montgomery: bool, width: int, modulus: int
) -> None:
    graph, _ = fixture(kind, width, modulus, montgomery, cv)
    bound, _, logical = bind_case(graph)
    before, after = residue_labels(kind, width, modulus, cv)

    def bits(labels: list[int]) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple((i >> (logical - 1 - j)) & 1 for j in range(logical))
            + (0,) * (bound.circuit.n_wires - logical)
            for i in labels
        )

    mapped, events = certified_and_channel_map(bound.circuit.instructions, bits(before))
    assert mapped == bits(after)
    assert events == sum(g.name == "measure" for g in bound.circuit.instructions)


@pytest.mark.parametrize(
    "kind,cv",
    [("neg", 1), ("cneg", 0), ("cneg", 1), ("sub", 1), ("csub", 0), ("csub", 1)],
)
def test_full_complex_spectator_states_raw_histories_and_work_zero(
    kind: str, cv: int
) -> None:
    graph, _ = fixture(kind, 2, 2, False, cv)
    bound, _, logical = bind_case(graph, spectator=True)
    n = max(w for gate in bound.circuit.instructions for w in gate.wires) + 1
    scratch = n - logical - 1
    before, after = residue_labels(kind, 2, 2, cv)
    labels = [((i << 1) | s) << scratch for i, s in product(before, (0, 1))]
    mapped = [((i << 1) | s) << scratch for i, s in product(after, (0, 1))]
    values = torch.arange(1, len(labels) + 1, dtype=torch.float64)
    amps = torch.complex(values, 4 - 7 * values)
    amps /= amps.norm()
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[labels], expected[mapped] = amps, amps
    events = sum(g.name == "measure" for g in bound.circuit.instructions)
    histories = (
        product((0, 1), repeat=events)
        if events <= 7
        else ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events)))
    )
    for history in histories:
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            expected / 2 ** (events / 2),
            rtol=0,
            atol=3e-12,
        )


@pytest.mark.parametrize(
    "kind,cv",
    [
        ("ModNeg", 1),
        ("CModNeg", 0),
        ("CModNeg", 1),
        ("ModSub", 1),
        ("CModSub", 0),
        ("CModSub", 1),
    ],
)
def test_normal_append_replaces_forward_emitter_and_original_inverse_restores(
    kind: Any, cv: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from importlib import import_module

    from examples.algorithms.modular_subtraction_catalog import (
        ModularSubtractionExample,
    )
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Forward append must traverse its stored source graph")

    for module, name in (
        ("modular_negation", "modular_negate_ir"),
        ("controlled_modular_negation", "controlled_modular_negate_ir"),
        ("modular_subtraction", "modular_subtract_ir"),
        ("controlled_modular_subtraction", "controlled_modular_subtract_ir"),
    ):
        monkeypatch.setattr(
            import_module("flagquantum.compiler.registers.modular." + module),
            name,
            forbidden,
        )
    example = ModularSubtractionExample(kind, 2, 2, control_value=cv)
    logical = sum(int(r.bitsize) for r in example.signature)
    n = example.qubit_count
    native_kind = {
        "ModNeg": "neg",
        "CModNeg": "cneg",
        "ModSub": "sub",
        "CModSub": "csub",
    }[kind]
    before, after = residue_labels(native_kind, 2, 2, cv)
    values = torch.arange(1, len(before) + 1, dtype=torch.float64)
    amps = torch.complex(values, 5 - 3 * values)
    amps /= amps.norm()
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    initial[[v << (n - logical) for v in before]] = amps
    expected[[v << (n - logical) for v in after]] = amps
    circuit = DynamicCircuit(n, inputs=initial, dtype=torch.complex128)
    circuit.append(example)
    result = run_dynamic(circuit, shots=4, seed=19, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), expected.expand(4, -1), rtol=0, atol=3e-12
    )
    circuit.append(replace(example, adjoint=True))
    result = run_dynamic(circuit, shots=4, seed=19, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), initial.expand(4, -1), rtol=0, atol=3e-12
    )


@pytest.mark.parametrize("kind", ["neg", "cneg", "sub", "csub"])
def test_original_all_ones_correction_loader_failure(kind: str) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.mod_arithmetic.mod_subtraction import (
        CModNeg,
        CModSub,
        ModNeg,
        ModSub,
    )

    original = {"neg": ModNeg, "cneg": CModNeg, "sub": ModSub, "csub": CModSub}[kind](
        QUInt(2), 3
    )
    with pytest.raises(ValueError, match="Too-large"):
        original.decompose_bloq().flatten()
    with pytest.raises(ValueError, match="modulus\\+1"):
        fixture(kind, 2, 3, False)


@pytest.mark.parametrize("kind,cv", [("neg", 1), ("cneg", 0), ("cneg", 1)])
def test_original_exterior_negation_retains_dirty_flag(kind: str, cv: int) -> None:
    graph, original = fixture(kind, 2, 2, False, cv)
    bound, ports, logical = bind_case(graph)
    assert_source(bound.circuit.instructions, source_stream(original, bound, ports))
    arguments = {"x": 2} if kind == "neg" else {"ctrl": cv, "x": 2}
    assert original.on_classical_vals(**arguments) == arguments
    with pytest.raises((ValueError, AssertionError)):
        original.decompose_bloq().call_classically(**arguments)
    n = max(w for gate in bound.circuit.instructions for w in gate.wires) + 1
    data = 2 if kind == "neg" else (cv << 2) | 2
    bits = tuple((data >> (logical - 1 - i)) & 1 for i in range(logical)) + (0,) * (
        n - logical
    )
    (mapped,), _ = certified_and_channel_map(bound.circuit.instructions, (bits,))
    expected = [0] * n
    if kind == "cneg":
        expected[0] = cv
    expected[bound.events[-1].wires[0]] = 1
    assert mapped == tuple(expected)


def test_complete_original_definition_dispositions() -> None:
    import ast
    from pathlib import Path

    from qualtran.bloqs.mod_arithmetic import mod_subtraction

    audit = (
        Path(__file__).parents[2] / "docs/development/MODULAR_SUBTRACTION_FILE_AUDIT.md"
    ).read_text()
    count = 0
    for node in ast.parse(Path(mod_subtraction.__file__).read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            assert f"`{node.name}`" in audit
            count += 1
            for method in node.body if isinstance(node, ast.ClassDef) else ():
                if isinstance(method, ast.FunctionDef):
                    assert f"`{node.name}.{method.name}`" in audit
                    count += 1
    assert count == 28
