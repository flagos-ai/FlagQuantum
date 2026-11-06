"""Original scaling graph with explicit output aliases and promise-domain checks."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.inverse_scaling import (
    inverse_scaling_ir,
    inverse_scaling_outputs,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(1, 1), (2, 3), (3, 5), (3, 7), (5, 31), (8, 251)]


def layout(
    n: int,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
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
        4 * n + 2,
        4 * n + 3,
        tuple(range(4 * n + 4, 6 * n + 9)),
        tuple(range(40, 40 + n + 1)),
    )


def source_gates(n: int, p: int, adjoint: bool) -> tuple[Instruction, ...]:
    import cirq
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep6

    u, v, r, s, b, a, m, f, initial_work, bits = layout(n)
    result_r = (*r[1:], initial_work[0])
    result_work = (r[0], *initial_work[1:])
    r_input, work = (result_r, result_work) if adjoint else (r, initial_work)

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        underlying = getattr(bloq, "subbloq", bloq)
        name = type(underlying).__name__
        if name == "CNOT":
            return (Instruction("cx", (*inputs["ctrl"], *inputs["target"])),), inputs
        if name == "XGate":
            return (Instruction("x", inputs["q"]),), inputs
        if name == "TwoBitCSwap":
            names = {cirq.CNOT: "cx", cirq.H: "h", cirq.T: "t", cirq.T**-1: "tdg"}
            binding = {cirq.NamedQubit(key): value[0] for key, value in inputs.items()}
            gates = tuple(
                Instruction(names[op.gate], tuple(binding[q] for q in op.qubits))
                for op in underlying.to_clifford_t_circuit().all_operations()
            )
            return gates, inputs
        if name == "CSwap":
            return (
                reference_arithmetic_gates(
                    bloq, inputs, (), {}, preserve_components=True
                ),
                inputs,
            )
        if name != "ModDbl":
            return None
        high_or_low, sign = work[:2]
        constant, carry = work[2 : n + 4], work[n + 4 :]
        inverse = type(bloq).__name__ == "Adjoint"
        if inverse:
            allocations = (
                sign,
                high_or_low,
                *constant[: n + 1],
                *carry[:n],
                *constant,
                *carry,
            )
            outputs = {"x": (high_or_low, *inputs["x"][:-1])}
        else:
            allocations = (
                sign,
                high_or_low,
                *constant,
                *carry,
                *constant[: n + 1],
                *carry[:n],
            )
            outputs = {"x": (*inputs["x"][1:], high_or_low)}
        assert all(outputs[key] == value for key, value in desired.items())
        gates = reference_arithmetic_gates(
            bloq,
            inputs,
            allocations,
            dict(zip(carry, bits, strict=True)),
            expected_outputs=outputs,
            reuse_allocations=False,
            preserve_components=not inverse,
        )
        return gates, outputs

    source = _KaliskiIterationStep6(n, p)
    if adjoint:
        source = source.adjoint()
    inputs = dict(u=u, v=v, r=r_input, s=s, b=(b,), a=(a,), m=(m,), f=(f,))
    desired = dict(inputs, r=r if adjoint else result_r)
    gates, outputs = walk_source_graph(source, inputs, desired, leaf)
    assert outputs == desired
    return gates


@pytest.mark.parametrize("n,p", CASES)
@pytest.mark.parametrize("adjoint", [False, True])
def test_source_gates_ports_and_default_exact_models(
    n: int, p: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep6
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    u, v, r, s, b, a, m, f, w, bits = layout(n)
    original = (r, w)
    if adjoint:
        r, w = inverse_scaling_outputs(r, w)
    gates = inverse_scaling_ir(u, v, r, s, b, a, m, f, p, w, bits, adjoint=adjoint)
    assert_source(gates, source_gates(n, p, adjoint))
    result, clean = inverse_scaling_outputs(r, w, adjoint=adjoint)
    if adjoint:
        assert (result, clean) == original
    else:
        assert result == (*r[1:], w[0]) and clean == (r[0], *w[1:])
    source = _KaliskiIterationStep6(n, p)
    if adjoint:
        source = source.adjoint()
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert (
        counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7) == 29 * n - 3
    )
    assert model.total_t_count() == 20 * n
    assert counts["measure"] == counts["reset"] == model.measurement == 2 * n + 1
    assert get_cost_value(source, QubitCount()) == 6 * n + 9


def encode(
    ports: tuple[tuple[int, ...], ...], values: tuple[int, ...], width: int
) -> tuple[int, ...]:
    data = [0] * width
    for register, value in zip(ports, values, strict=True):
        for i, wire in enumerate(register):
            data[wire] = value >> (len(register) - 1 - i) & 1
    return tuple(data)


def scaling_columns(
    n: int, p: int, *, even_only: bool
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    u, v, r, s, b, a, m, f, w, _ = layout(n)
    width = 6 * n + 9
    result_r, _ = inverse_scaling_outputs(r, w)
    ports = (u, v, r, s, (b,), (a,), (m,), (f,))
    output_ports = (u, v, result_r, s, (b,), (a,), (m,), (f,))
    words = (
        range(1 << n)
        if n <= 3
        else tuple(
            sorted({0, 1, (1 << (n - 1)) - 1, 1 << (n - 1), (1 << n) - 2, (1 << n) - 1})
        )
    )
    residues = range(p) if n <= 3 else tuple(sorted({0, 1, p // 2, p - 1}))
    other_values = (
        range(1 << n)
        if n <= 3
        else tuple(sorted({0, 1, p // 2, p - 1, p, (1 << n) - 1}))
    )
    before, after = [], []
    for x, y, coeff, other, flags in product(
        words, words, residues, other_values, range(16)
    ):
        branch, parity, marker, active = (
            flags >> 3,
            flags >> 2 & 1,
            flags >> 1 & 1,
            flags & 1,
        )
        if even_only and active and y % 2:
            continue
        before.append(
            encode(ports, (x, y, coeff, other, branch, parity, marker, active), width)
        )
        branch ^= marker ^ parity
        if active:
            y = (y >> 1) | ((y & 1) << (n - 1))
        coeff = 2 * coeff % p
        if parity:
            x, y, coeff, other = y, x, other, coeff
        parity ^= int(other % 2 == 0)
        after.append(
            encode(
                output_ports,
                (x, y, coeff, other, branch, parity, marker, active),
                width,
            )
        )
    return tuple(before), tuple(after)


@pytest.mark.parametrize("n,p", CASES)
def test_residue_coefficient_full_other_word_all_histories_and_alias_restoration(
    n: int, p: int
) -> None:
    u, v, r, s, b, a, m, f, w, bits = layout(n)
    before, expected = scaling_columns(n, p, even_only=False)
    gates = inverse_scaling_ir(u, v, r, s, b, a, m, f, p, w, bits)
    actual, events = certified_and_channel_map(gates, before)
    assert actual == expected and events == 2 * n + 1
    result, clean = inverse_scaling_outputs(r, w)
    inverse = inverse_scaling_ir(
        u, v, result, s, b, a, m, f, p, clean, bits, adjoint=True
    )
    restored, back_events = certified_and_channel_map(inverse, actual)
    assert restored == before and back_events == events


@pytest.mark.parametrize("n,p", [(1, 1), (2, 3)])
def test_small_complete_complex_all_raw_branches_and_clean_roles(
    n: int, p: int
) -> None:
    before, after = scaling_columns(n, p, even_only=False)
    width = 6 * n + 9
    state = torch.zeros(1 << (width + 1), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    coeff = torch.arange(2 * len(before), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    for i, (x, y) in enumerate(zip(before, after, strict=True)):
        xv = sum(bit << (width - 1 - j) for j, bit in enumerate(x))
        yv = sum(bit << (width - 1 - j) for j, bit in enumerate(y))
        for spectator in (0, 1):
            state[(xv << 1) | spectator] = amps[2 * i + spectator]
            expected[(yv << 1) | spectator] = amps[2 * i + spectator]
    u, v, r, s, b, a, m, f, w, bits = layout(n)
    gates = inverse_scaling_ir(u, v, r, s, b, a, m, f, p, w, bits)
    result, clean = inverse_scaling_outputs(r, w)
    inverse = inverse_scaling_ir(
        u, v, result, s, b, a, m, f, p, clean, bits, adjoint=True
    )
    events = 2 * n + 1
    for branch in product((0, 1), repeat=events):
        actual = forced_branch(state, gates, branch)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), rtol=0, atol=1e-12
        )
        torch.testing.assert_close(
            forced_branch(actual, inverse, branch),
            state / 2**events,
            rtol=0,
            atol=1e-12,
        )


def test_odd_active_word_is_rotation_not_classical_division() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep6

    n, p = 2, 3
    u, v, r, s, b, a, m, f, w, bits = layout(n)
    row = dict(u=0, v=1, r=0, s=0, b=0, a=0, m=0, f=1)
    assert _KaliskiIterationStep6(n, p).on_classical_vals(**row)["v"] == 0
    before = encode((u, v, r, s, (b,), (a,), (m,), (f,)), (0, 1, 0, 0, 0, 0, 0, 1), 21)
    result, _ = certified_and_channel_map(
        inverse_scaling_ir(u, v, r, s, b, a, m, f, p, w, bits), (before,)
    )
    assert tuple(result[0][wire] for wire in v) == (1, 0)


@pytest.mark.parametrize(
    "bad", ["width", "adjoint", "alias", "work", "slots", "modulus"]
)
def test_invalid_layouts_and_port_mode(bad: str) -> None:
    u, v, r, s, b, a, m, f, w, bits = layout(2)
    inverse, modulus = False, 3
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
        modulus = 2
    with pytest.raises(ValueError):
        inverse_scaling_ir(u, v, r, s, b, a, m, f, modulus, w, bits, adjoint=inverse)


def test_invalid_output_role_mode() -> None:
    _, _, r, _, _, _, _, _, w, _ = layout(2)
    with pytest.raises(ValueError, match="boolean"):
        inverse_scaling_outputs(r, w, adjoint=1)


def test_original_classical_agreement_on_even_residue_domain() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep6

    u, v, r, s, b, a, m, f, w, _ = layout(2)
    result_r, _ = inverse_scaling_outputs(r, w)
    input_ports = dict(u=u, v=v, r=r, s=s, b=(b,), a=(a,), m=(m,), f=(f,))
    output_ports = dict(input_ports, r=result_r)
    before, after = scaling_columns(2, 3, even_only=True)
    source = _KaliskiIterationStep6(2, 3)
    for x, y in zip(before, after, strict=True):
        values = {
            name: sum(x[wire] << (len(wires) - 1 - i) for i, wire in enumerate(wires))
            for name, wires in input_ports.items()
        }
        expected = {
            name: sum(y[wire] << (len(wires) - 1 - i) for i, wire in enumerate(wires))
            for name, wires in output_ports.items()
        }
        assert source.on_classical_vals(**values) == expected

    # The original full iteration reaches s=p; this is not only a padded toy input.
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIteration

    iteration = _KaliskiIteration(3, 5)
    state = dict(u=5, v=1, r=0, s=1, f=1)
    right_names = tuple(reg.name for reg in iteration.signature.rights())
    witnessed = False
    for _ in range(6):
        values = dict(state, m=0, is_terminal=0)
        expected = iteration.on_classical_vals(**values)
        assert (
            dict(
                zip(
                    right_names,
                    iteration.decompose_bloq().call_classically(**values),
                    strict=True,
                )
            )
            == expected
        )
        witnessed |= expected["s"] == 5
        state = {key: expected[key] for key in ("u", "v", "r", "s", "f")}
    assert witnessed


def test_out_of_residue_three_source_contracts_are_distinct() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep6

    source = _KaliskiIterationStep6(2, 1)
    values = dict(u=0, v=0, r=2, s=0, b=0, a=0, m=0, f=0)
    assert source.on_classical_vals(**values)["r"] == 0
    composite = source.decompose_bloq().call_classically(**values)
    assert composite[2] == 2
    u, v, r, s, b, a, m, f, w, bits = layout(2)
    ports = (u, v, r, s, (b,), (a,), (m,), (f,))
    before = encode(ports, (0, 0, 2, 0, 0, 0, 0, 0), 21)
    gates = inverse_scaling_ir(u, v, r, s, b, a, m, f, 1, w, bits)
    assert_source(gates, source_gates(2, 1, False))
    actual, _ = certified_and_channel_map(gates, (before,))
    result_r, clean = inverse_scaling_outputs(r, w)
    assert tuple(actual[0][wire] for wire in result_r) == (1, 1)
    assert not any(actual[0][wire] for wire in clean)
