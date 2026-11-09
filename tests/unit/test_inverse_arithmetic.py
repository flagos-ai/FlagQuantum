"""Source conditional update graph and full coherent ring-arithmetic channels."""

from collections import Counter
from itertools import product
from typing import Any

import pytest

from flagquantum.compiler.registers.modular.inversion.inverse_arithmetic import (
    inverse_arithmetic_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.inverse_flag_checks import check_flag_channel
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(
    n: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
]:
    words = tuple(tuple(range(i * n, (i + 1) * n)) for i in range(4))
    return (
        *words,
        4 * n,
        4 * n + 1,
        tuple(range(4 * n + 2, 6 * n + 2)),
        tuple(range(40, 40 + 2 * n - 1)),
    )


def source_gates(n: int, adjoint: bool) -> tuple[Instruction, ...]:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep5

    u, v, r, s, b, f, w, bits = layout(n)
    control, loaded, carries = w[0], w[1 : n + 1], w[n + 1 :]
    slots = dict(zip((*loaded, *carries), bits, strict=True))
    slots[control] = bits[0]

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        underlying = getattr(bloq, "subbloq", bloq)
        name = type(underlying).__name__
        if name == "And":
            outputs = {"ctrl": inputs["ctrl"]}
            if "target" not in inputs:
                outputs["target"] = (control,)
            assert all(outputs[k] == value for k, value in desired.items())
            # Atomic source And exposes its unchanged Cirq generator directly.
            gates = reference_arithmetic_gates(
                bloq, {**inputs, **outputs}, (), slots, cirq_style=True
            )
            return gates, outputs
        if name == "CAdd":
            outputs = inputs
            allocations = (*loaded, *carries)
        elif name == "BitwiseNot":
            outputs, allocations = inputs, ()
        else:
            return None
        assert all(outputs[k] == value for k, value in desired.items())
        return (
            reference_arithmetic_gates(
                bloq,
                inputs,
                allocations,
                slots,
                expected_outputs=outputs,
                reuse_allocations=False,
            ),
            outputs,
        )

    source = _KaliskiIterationStep5(n)
    if adjoint:
        source = source.adjoint()
    inputs = dict(u=u, v=v, r=r, s=s, b=(b,), f=(f,))
    gates, outputs = walk_source_graph(source, inputs, inputs, leaf)
    assert outputs == inputs
    return gates


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("adjoint", [False, True])
def test_whole_source_component_gates_and_models(n: int, adjoint: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep5
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    gates = inverse_arithmetic_ir(*layout(n), adjoint=adjoint)
    assert_source(gates, source_gates(n, adjoint))
    source = _KaliskiIterationStep5(n)
    if adjoint:
        source = source.adjoint()
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 16 * n - 4
    assert counts["measure"] == counts["reset"] == model.measurement == 4 * n - 1
    assert get_cost_value(source, QubitCount()) == 6 * n + 2
    assert counts["x"] == 2 * n + 4
    if n == 1:
        assert model.clifford == -1


@pytest.mark.parametrize("n", [1, 2, 3])
def test_every_word_every_control_raw_branches_and_inverse(n: int) -> None:
    outputs = []
    mask = (1 << n) - 1
    for raw in range(1 << (4 * n + 2)):
        u, v, r, s = tuple(raw >> (2 + i * n) & mask for i in reversed(range(4)))
        branch, active = raw >> 1 & 1, raw & 1
        if active and not branch:
            v, s = (v - u) & mask, (s + r) & mask
        outputs.append(
            (u << (3 * n + 2))
            | (v << (2 * n + 2))
            | (r << (n + 2))
            | (s << 2)
            | (branch << 1)
            | active
        )
    check_flag_channel(
        inverse_arithmetic_ir(*layout(n)),
        inverse_arithmetic_ir(*layout(n), adjoint=True),
        tuple(outputs),
        4 * n + 2,
        2 * n,
        4 * n - 1,
        raw_branches=n <= 2,
    )


def test_eight_bit_edges_and_all_control_histories() -> None:
    n, width, q = 8, 50, 16
    values = (0, 1, 127, 128, 254, 255)
    before, after = [], []
    for u, v, r, s, branch, active in product(
        values, values, values, values, (0, 1), (0, 1)
    ):
        x = (u << 26) | (v << 18) | (r << 10) | (s << 2) | (branch << 1) | active
        if active and not branch:
            v, s = (v - u) % 256, (s + r) % 256
        y = (u << 26) | (v << 18) | (r << 10) | (s << 2) | (branch << 1) | active
        before.append(tuple((x << q) >> (width - 1 - i) & 1 for i in range(width)))
        after.append(tuple((y << q) >> (width - 1 - i) & 1 for i in range(width)))
    result, events = certified_and_channel_map(
        inverse_arithmetic_ir(*layout(n)), tuple(before)
    )
    assert result == tuple(after) and events == 31
    restored, _ = certified_and_channel_map(
        inverse_arithmetic_ir(*layout(n), adjoint=True), result
    )
    assert restored == tuple(before)


@pytest.mark.ftqc_reference
def test_source_classical_nonwrapping_domain_and_overflow_witness() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep5

    source = _KaliskiIterationStep5(2)
    for u, v, r, s in product(range(4), repeat=4):
        if v >= u and s + r < 4:
            assert source.on_classical_vals(u=u, v=v, r=r, s=s, b=0, f=1) == dict(
                u=u, v=v - u, r=r, s=s + r, b=0, f=1
            )
    assert source.on_classical_vals(u=1, v=0, r=1, s=3, b=0, f=1) == dict(
        u=1, v=-1, r=1, s=4, b=0, f=1
    )
    # The unchanged literal source gates wrap both words instead.
    raw = (1 << 8) | (1 << 4) | (3 << 2) | 1
    bits = tuple((raw << 4) >> (13 - i) & 1 for i in range(14))
    result, _ = certified_and_channel_map(source_gates(2, False), (bits,))
    expected = ((1 << 8) | (3 << 6) | (1 << 4) | 1) << 4
    assert result == (tuple(expected >> (13 - i) & 1 for i in range(14)),)


@pytest.mark.parametrize("bad", ["width", "adjoint", "alias", "work", "slots", "bits"])
def test_layout_rejections(bad: str) -> None:
    u, v, r, s, b, f, w, bits = layout(2)
    inverse = False
    if bad == "width":
        u = ()
    elif bad == "adjoint":
        inverse = 1
    elif bad == "alias":
        f = b
    elif bad == "work":
        w = w[:-1]
    elif bad == "slots":
        bits = bits[:-1]
    else:
        bits = (40, 40, 42)
    with pytest.raises(ValueError):
        inverse_arithmetic_ir(u, v, r, s, b, f, w, bits, adjoint=inverse)
