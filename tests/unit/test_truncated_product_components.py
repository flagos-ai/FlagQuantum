"""Stored D4/D5 products retain source ports, CAdd calls and exact erasure."""

from collections import Counter
from collections.abc import Callable
from itertools import product
from typing import Literal, cast

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.fixed_point.fixed_point_product_component import (
    fixed_point_product_program,
)
from flagquantum.compiler.registers.fixed_point.fixed_point_scaling import (
    fixed_point_product_ir,
    fixed_point_product_workspace_size,
    scale_integer_by_real_ir,
    scale_integer_workspace_size,
)
from flagquantum.compiler.registers.fixed_point.fixed_point_scaling_component import (
    scale_integer_by_real_program,
)
from flagquantum.compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import fixed_point_register_type
from flagquantum.core.ir.program.registers import RegisterType
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_fixed_point_scaling import truncated
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_weighted_columns_component import normalized_feedback

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Kind = Literal["scale", "product"]
truncate = cast(Callable[[int, int, int], int], truncated)
CASES: tuple[tuple[Kind, int, int], ...] = (
    ("scale", 1, 1),
    ("scale", 4, 1),
    ("scale", 2, 2),
    ("scale", 3, 2),
    ("scale", 4, 4),
    ("scale", 6, 4),
    ("product", 1, 1),
    ("product", 2, 2),
    ("product", 3, 3),
    ("product", 4, 4),
    ("product", 6, 6),
)


def graph(kind: Kind, n: int, m: int, adjoint: bool = False) -> ProgramRecord:
    return (
        scale_integer_by_real_program(n, m, adjoint=adjoint)
        if kind == "scale"
        else fixed_point_product_program(n, adjoint=adjoint)
    )


def layout(kind: Kind, n: int, m: int) -> tuple[tuple[int, ...], ...]:
    count = (
        scale_integer_workspace_size(n, m)
        if kind == "scale"
        else fixed_point_product_workspace_size(n)
    )
    start = 2 * n + m
    return (
        tuple(range(n)),
        tuple(range(n, n + m)),
        tuple(range(n + m, start)),
        tuple(range(start, start + 2 * count)),
    )


@pytest.mark.parametrize("kind,n,m", CASES)
@pytest.mark.parametrize("expanded", (False, True))
def test_original_ports_calls_primitive_operands_feedback_and_models(
    kind: Kind, n: int, m: int, expanded: bool
) -> None:
    from qualtran._infra.data_types import QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.bloqs.arithmetic.multiplication import (
        MultiplyTwoReals,
        ScaleIntByReal,
    )
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    original = ScaleIntByReal(n, m) if kind == "scale" else MultiplyTwoReals(n)
    a, b, output, work = layout(kind, n, m)
    first_width, omitted = (n - m, 0) if kind == "scale" else (1, 1)
    active = tuple(reversed(b))[omitted:]
    steps = tuple(enumerate(active[1:], first_width + 1))
    for adjoint in (False, True):
        a, b, output, work = layout(kind, n, m)
        module = graph(kind, n, m, adjoint)
        assert verify_program_record(module) == ()
        ports = module.functions[0].registers
        assert ports is not None
        assert [
            (
                r.name,
                r.dtype.quantum_bits,
                dict(r.dtype.parameters).get("fractional_width"),
                r.side.name,
            )
            for r in ports
        ] == [
            (
                r.name,
                r.bitsize,
                getattr(r.dtype, "num_frac", None),
                "LEFT" if adjoint and r.name == "result" else r.side.name,
            )
            for r in original.signature
        ]
        assert tuple(r.dtype for r in ports) == tuple(
            (
                fixed_point_register_type(r.bitsize, r.dtype.num_frac)
                if hasattr(r.dtype, "num_frac")
                else RegisterType("uint", r.bitsize)
            )
            for r in original.signature
        )
        functions = {f.name: f for f in module.functions}
        calls = []
        for op in module.functions[0].blocks[0].operations:
            callee = functions[op.callee or ""]
            registers = callee.registers or ()
            if tuple(r.name for r in registers) == ("ctrl", "a", "b"):
                calls.append(
                    (registers[1].dtype.quantum_bits, registers[2].dtype.quantum_bits)
                )
        assert calls == [(w, w + 1) for w, _ in (reversed(steps) if adjoint else steps)]
        if expanded:
            module, _ = expand_component_calls(module)
        bound = bind_register_program(
            module,
            (a, b, output) if adjoint else (a, b),
            clean=work if adjoint else (*output, *work),
            allow_workspace_outputs=not adjoint,
            input_ownership=tuple((wire, False) for wire in output) if adjoint else (),
        )
        assert bound.outputs[:2] == (a, b)
        if not adjoint:
            declared_pool = (*output, *work)
            if not expanded:
                assert bound.outputs[2] == output
            output = bound.outputs[2]
            assert len(output) == len(set(output)) == n
            assert set(output).issubset(declared_pool)
            work = tuple(wire for wire in declared_pool if wire not in output)
        else:
            assert bound.outputs == (a, b)
        pairs = (
            tuple(
                zip(
                    a[:first_width],
                    output[-first_width:] if first_width else (),
                    strict=True,
                )
            )
            if active
            else ()
        )
        allocations = tuple(
            wire
            for event in bound.events
            if event.kind == "allocate" and set(event.wires).isdisjoint(output)
            for wire in event.wires
        )
        reference: list[Instruction] = []
        cursor = 0
        for width, control in reversed(steps) if adjoint else steps:
            child_work = allocations[cursor : cursor + 2 * width]
            assert len(child_work) == len(set(child_work)) == 2 * width
            assert set(child_work).issubset(work)
            cursor += 2 * width
            source = CAdd(QUInt(width), QUInt(width + 1))
            reference.extend(
                reference_arithmetic_gates(
                    source.adjoint() if adjoint else source,
                    {"ctrl": (control,), "a": a[:width], "b": output[-width - 1 :]},
                    child_work,
                    dict(zip(child_work, range(2 * width), strict=True)),
                )
            )
        assert cursor == len(allocations)
        copy = tuple(
            gate
            for left, target in (reversed(pairs) if adjoint else pairs)
            for gate in (
                and_uncompute_ir(active[0], left, target, 0)
                if adjoint
                else and_compute_ir(active[0], left, target)
            )
        )
        expected = (*reference, *copy) if adjoint else (*copy, *reference)
        assert_source(
            normalized_feedback(bound.circuit.instructions),
            normalized_feedback(expected),
        )
        pool = len(work) // 2
        flat = (
            scale_integer_by_real_ir(
                a,
                b,
                output,
                work[:pool],
                work[pool:],
                tuple(range(2 * pool)),
                adjoint=adjoint,
            )
            if kind == "scale"
            else fixed_point_product_ir(
                a,
                b,
                output,
                work[:pool],
                work[pool:],
                tuple(range(2 * pool)),
                adjoint=adjoint,
            )
        )
        counts = Counter(g.name for g in bound.circuit.instructions)
        assert counts == Counter(g.name for g in flat)
        assert counts["t"] + counts["tdg"] == 8 * sum(w for w, _ in steps) + (
            0 if adjoint else 4 * len(pairs)
        )
        assert (
            counts["measure"]
            == counts["reset"]
            == 2 * sum(w for w, _ in steps) + (len(pairs) if adjoint else 0)
        )
        live = set(output) if adjoint else set()
        peak = 0
        for event in bound.events:
            if event.kind == "allocate":
                assert live.isdisjoint(event.wires)
                live.update(event.wires)
            elif event.kind == "free":
                assert set(event.wires).issubset(live)
                live.difference_update(event.wires)
            peak = max(peak, len(live))
        assert live == (set() if adjoint else set(output))
        assert peak <= n + len(work)
    assert get_cost_value(original, QECGatesCost()).toffoli == (
        n * (2 * m - 1) - m * m if kind == "scale" else n * n - n - 1
    )


@pytest.mark.parametrize(
    "kind,n,m",
    (
        ("scale", 2, 1),
        ("scale", 2, 2),
        ("scale", 3, 2),
        ("product", 1, 1),
        ("product", 2, 2),
        ("product", 3, 3),
    ),
)
@pytest.mark.parametrize("expanded", (False, True))
def test_full_complex_inputs_spectator_owned_image_and_zero_work(
    kind: Kind, n: int, m: int, expanded: bool
) -> None:
    a, b, output, work = layout(kind, n, m)
    size = n + m + n + len(work) + 1
    forward_module, inverse_module = graph(kind, n, m), graph(kind, n, m, True)
    if expanded:
        forward_module, _ = expand_component_calls(forward_module)
        inverse_module, _ = expand_component_calls(inverse_module)
    declared_pool = (*output, *work)
    forward = bind_register_program(
        forward_module, (a, b), clean=declared_pool, allow_workspace_outputs=True
    )
    output = forward.outputs[2]
    work = tuple(wire for wire in declared_pool if wire not in output)
    inverse = bind_register_program(
        inverse_module,
        (a, b, output),
        clean=work,
        input_ownership=tuple((wire, False) for wire in output),
    )
    before: dict[int, complex] = {}
    after: dict[int, complex] = {}
    for left, right, spectator in product(range(1 << n), range(1 << m), (0, 1)):
        image = truncate(left, right, m if kind == "scale" else n)
        prefix = ((left << m) | right) << (n + len(work) + 1)
        amplitude = complex(left + right + 1, left - 3 * right + spectator)
        before[prefix | spectator] = amplitude
        stored_image = sum(
            ((image >> (n - 1 - index)) & 1) << (size - wire - 1)
            for index, wire in enumerate(output)
        )
        after[prefix | stored_image | spectator] = amplitude
    norm = sum(abs(v) ** 2 for v in before.values()) ** 0.5
    before, after = (
        {i: v / norm for i, v in state.items()} for state in (before, after)
    )
    for bound, initial, expected in (
        (forward, before, after),
        (inverse, after, before),
    ):
        for seed in (2, 19):
            actual = execute_sparse_register(
                CircuitIR(size, bound.circuit.instructions), initial, seed=seed
            )
            assert all(
                abs(actual.get(i, 0) - expected.get(i, 0)) < 5e-12
                for i in actual.keys() | expected.keys()
            )
    if n <= 2:
        initial_vector = torch.zeros(1 << size, dtype=torch.complex128)
        expected_vector = torch.zeros_like(initial_vector)
        for i, coefficient in before.items():
            initial_vector[i] = torch.tensor(coefficient, dtype=torch.complex128)
        for i, coefficient in after.items():
            expected_vector[i] = torch.tensor(coefficient, dtype=torch.complex128)
        first_count = sum(g.name == "measure" for g in forward.circuit.instructions)
        second_count = sum(g.name == "measure" for g in inverse.circuit.instructions)
        for first in product((0, 1), repeat=first_count):
            state = forced_branch(initial_vector, forward.circuit.instructions, first)
            torch.testing.assert_close(
                state, expected_vector / 2 ** (first_count / 2), atol=5e-12, rtol=0
            )
            for second in product((0, 1), repeat=second_count):
                restored = forced_branch(state, inverse.circuit.instructions, second)
                torch.testing.assert_close(
                    restored,
                    initial_vector / 2 ** ((first_count + second_count) / 2),
                    atol=5e-12,
                    rtol=0,
                )


@pytest.mark.parametrize("kind", ("scale", "product"))
def test_normal_append_replaces_flat_roots(
    kind: Kind, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.compiler.registers.fixed_point import fixed_point_scaling

    def refuse(*args: object, **kwargs: object) -> object:
        raise AssertionError("Flat product roots must be replaceable")

    monkeypatch.setattr(fixed_point_scaling, "scale_integer_by_real_ir", refuse)
    monkeypatch.setattr(fixed_point_scaling, "fixed_point_product_ir", refuse)
    a, b, output, work = layout(kind, 3, 2 if kind == "scale" else 3)
    forward = bind_register_program(
        graph(kind, 3, len(b)),
        (a, b),
        clean=(*output, *work),
        allow_workspace_outputs=True,
    )
    inverse = bind_register_program(
        graph(kind, 3, len(b), True),
        (a, b, output),
        clean=work,
        input_ownership=tuple((wire, False) for wire in output),
    )
    size = 6 + len(b) + len(work) + 1
    state = torch.zeros(1 << size, dtype=torch.complex128)
    state[(7 << (len(b) + 3 + len(work) + 1)) | (3 << (3 + len(work) + 1)) | 1] = (
        torch.tensor((2 + 1j) / 5**0.5, dtype=torch.complex128)
    )
    circuit = DynamicCircuit(size, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(size, forward.circuit.instructions))
    circuit.append(FieldOperation(size, inverse.circuit.instructions))
    actual = run_dynamic(circuit, shots=4, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states.reshape(4, -1), state.expand(4, -1), atol=5e-12, rtol=0
    )


@pytest.mark.parametrize("n,m", ((True, 1), (0, 1), (1, 0), (1, 2), (-1, 1)))
def test_invalid_scaling_widths(n: int, m: int) -> None:
    with pytest.raises(ValueError):
        scale_integer_by_real_program(n, m)


@pytest.mark.parametrize("width", (True, 0, -1))
def test_invalid_product_width(width: int) -> None:
    with pytest.raises(ValueError):
        fixed_point_product_program(width)


def test_invalid_directions() -> None:
    with pytest.raises(ValueError, match="boolean"):
        scale_integer_by_real_program(2, 1, adjoint=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="boolean"):
        fixed_point_product_program(2, adjoint=1)  # type: ignore[arg-type]
