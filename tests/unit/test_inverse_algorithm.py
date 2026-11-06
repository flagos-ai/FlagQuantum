"""Original multi-round inverse graph, retained history and full complex channels."""

from collections import Counter
from math import gcd

import pytest

from flagquantum.compiler.registers.inverse_algorithm import (
    inverse_algorithm_ir,
    inverse_algorithm_outputs,
)
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.inverse_algorithm_source import walk_inverse_source
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(n: int) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    return (
        *(tuple(range(i * n, (i + 1) * n)) for i in range(4)),
        tuple(range(4 * n, 6 * n)),
        6 * n,
        tuple(range(6 * n + 1, 8 * n + 1)),
        tuple(range(8 * n + 1, 10 * n + 8)),
        tuple(range(40, 40 + max(2 * n - 1, n + 1))),
    )


def pack(
    values: tuple[int, ...], ports: tuple[tuple[int, ...], ...], width: int
) -> int:
    raw = 0
    for value, wires in zip(values, ports, strict=True):
        for i, wire in enumerate(wires):
            raw |= (value >> (len(wires) - 1 - i) & 1) << (width - 1 - wire)
    return raw


def original_algorithm(n: int, p: int, adjoint: bool) -> tuple[Instruction, ...]:
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiModInverseImpl

    u, v, r, s, m, f, t, work, bits = layout(n)
    rr, ww = inverse_algorithm_outputs(r, work)
    input_r, pool = (rr, ww) if adjoint else (r, work)
    source = _KaliskiModInverseImpl(n, p)
    if adjoint:
        source = source.adjoint()

    inputs = dict(u=u, v=v, r=input_r, s=s, m=m, f=(f,), terminal_condition=t)
    desired = dict(inputs, r=r if adjoint else rr)
    gates, outputs, pool, freed = walk_inverse_source(
        source, inputs, desired, n, p, adjoint, pool, bits
    )
    assert not freed
    assert outputs == desired
    assert pool == (work if adjoint else ww)
    return gates


@pytest.mark.parametrize("n,p", [(2, 1), (3, 5), (4, 13), (8, 251)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_full_source_stream_ports_and_cost_models(
    n: int, p: int, adjoint: bool
) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiModInverseImpl
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    u, v, r, s, m, f, t, work, bits = layout(n)
    if adjoint:
        r, work = inverse_algorithm_outputs(r, work)
    gates = inverse_algorithm_ir(u, v, r, s, m, f, t, p, work, bits, adjoint=adjoint)
    assert_source(gates, original_algorithm(n, p, adjoint))
    source = _KaliskiModInverseImpl(n, p)
    if adjoint:
        source = source.adjoint()
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert (
        counts["t"] + counts["tdg"]
        == model.total_t_count(ts_per_cswap=7)
        == 2 * n * (67 * n + 13) + 4 * (n - 1)
    )
    assert model.total_t_count() == 2 * n * (52 * n + 16) + 4 * (n - 1)
    assert (
        counts["measure"]
        == counts["reset"]
        == model.measurement
        == 2 * n * (8 * n + 5) + n - 1
    )
    assert get_cost_value(source, QubitCount()) == 10 * n + 8


def cases(n: int, p: int) -> tuple[tuple[int, ...], ...]:
    from qualtran.bloqs.mod_arithmetic.mod_division import KaliskiModInverse

    source = KaliskiModInverse(n, p)
    out = []
    for x in range(p):
        if x and gcd(x, p) != 1:
            continue
        result = source.on_classical_vals(x=x)
        value, junk = int(result["x"]), int(result["junk"])
        assert value == (pow(x, -1, p) * pow(2, 2 * n, p)) % p if x else value == 0
        out.append((x, value, junk >> (2 * n), junk & ((1 << (2 * n)) - 1)))
    return tuple(out)


@pytest.mark.parametrize("n,p", [(2, 1), (3, 3), (3, 5), (4, 9), (4, 13), (8, 251)])
def test_all_units_zero_full_history_channels_and_inverse(n: int, p: int) -> None:
    u, v, r, s, m, f, t, work, bits = layout(n)
    rr, ww = inverse_algorithm_outputs(r, work)
    width = 10 * n + 8
    ports = (u, v, r, s, m, (f,), t)
    outputs = (u, v, rr, s, m, (f,), t)
    before, expected = [], []
    for x, y, history, terminal in cases(n, p):
        for collection, values, bindings in (
            (before, (0, x, 0, 0, 0, 0, 0), ports),
            (expected, (0, 0, y, 0, history, 0, terminal), outputs),
        ):
            raw = pack(values, bindings, width)
            collection.append(tuple(raw >> (width - 1 - i) & 1 for i in range(width)))
    gates = inverse_algorithm_ir(u, v, r, s, m, f, t, p, work, bits)
    actual, events = certified_and_channel_map(gates, tuple(before))
    assert actual == tuple(expected) and events == 2 * n * (8 * n + 5) + n - 1
    inverse = inverse_algorithm_ir(u, v, rr, s, m, f, t, p, ww, bits, adjoint=True)
    restored, back = certified_and_channel_map(inverse, actual)
    assert restored == tuple(before) and back == events
    assert inverse_algorithm_outputs(rr, ww, adjoint=True) == (r, work)


def assert_complex_maps(
    actual: dict[int, complex], expected: dict[int, complex]
) -> None:
    """Compare every occupied key, including floating cancellation residuals."""
    keys = actual.keys() | expected.keys()
    assert {key: actual.get(key, 0j) for key in keys} == pytest.approx(
        {key: expected.get(key, 0j) for key in keys}, rel=0, abs=2e-12
    )


@pytest.mark.parametrize("n,p", [(3, 5), (4, 9), (8, 251)])
def test_primitive_sparse_complex_vectors_and_source_replacement(
    n: int, p: int
) -> None:
    u, v, r, s, m, f, t, work, bits = layout(n)
    rr, ww = inverse_algorithm_outputs(r, work)
    width = 10 * n + 8
    data = cases(n, p)
    state, expected = {}, {}
    norm = sum(abs(complex(i + 1, 2 * i - 3)) ** 2 for i in range(len(data))) ** 0.5
    for i, (x, y, history, terminal) in enumerate(data):
        amp = complex(i + 1, 2 * i - 3) / norm
        state[pack((0, x, 0, 0, 0, 0, 0), (u, v, r, s, m, (f,), t), width)] = amp
        expected[
            pack((0, 0, y, 0, history, 0, terminal), (u, v, rr, s, m, (f,), t), width)
        ] = amp
    gates = inverse_algorithm_ir(u, v, r, s, m, f, t, p, work, bits)
    inverse = inverse_algorithm_ir(u, v, rr, s, m, f, t, p, ww, bits, adjoint=True)
    for seed in (0, 7, 19):
        actual = execute_sparse_register(
            CircuitIR(width, gates), state.copy(), seed=seed
        )
        assert_complex_maps(actual, expected)
        restored = execute_sparse_register(
            CircuitIR(width, inverse), actual, seed=seed + 1
        )
        assert_complex_maps(restored, state)
    # Original executable source replaces native emission without changing inputs.
    source_gates = original_algorithm(n, p, False)
    assert_complex_maps(
        execute_sparse_register(CircuitIR(width, source_gates), state.copy(), seed=31),
        expected,
    )


@pytest.mark.parametrize("n", [2, 3, 5])
def test_original_all_ones_modulus_load_failure(n: int) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiModInverseImpl
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    p = (1 << n) - 1
    with pytest.raises(ValueError, match="Too-large classical value"):
        get_cost_value(_KaliskiModInverseImpl(n, p), QECGatesCost())
    u, v, r, s, m, f, t, work, bits = layout(n)
    from qualtran import QMontgomeryUInt
    from qualtran.bloqs.arithmetic import AddK

    with pytest.raises(ValueError, match="Too-large classical value"):
        reference_arithmetic_gates(
            AddK(QMontgomeryUInt(n), p + 1),
            {"x": r},
            work[: 2 * n - 1],
            dict(zip(work[n : 2 * n - 1], bits[: n - 1], strict=True)),
        )
    with pytest.raises(ValueError, match=r"p\+1 fitting"):
        inverse_algorithm_ir(u, v, r, s, m, f, t, p, work, bits)


@pytest.mark.parametrize(
    "bad",
    ["words", "markers", "terminals", "adjoint", "alias", "work", "bits", "modulus"],
)
def test_invalid_layout_rejected(bad: str) -> None:
    u, v, r, s, m, f, t, work, bits = layout(3)
    p, adjoint = 5, False
    if bad == "words":
        u = u[:-1]
    elif bad == "markers":
        m = m[:-1]
    elif bad == "terminals":
        t = t[:-1]
    elif bad == "adjoint":
        adjoint = 1
    elif bad == "alias":
        f = m[0]
    elif bad == "work":
        work = work[:-1]
    elif bad == "bits":
        bits = (40,) * len(bits)
    else:
        p = 4
    with pytest.raises(ValueError):
        inverse_algorithm_ir(u, v, r, s, m, f, t, p, work, bits, adjoint=adjoint)


def test_empty_output_roles_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        inverse_algorithm_outputs((), ())


@pytest.mark.parametrize("x", [3, 6])
def test_nonunit_inputs_do_not_claim_zero_logical_cleanup(x: int) -> None:
    from qualtran.bloqs.mod_arithmetic.mod_division import KaliskiModInverse

    n, p = 4, 9
    with pytest.raises(AssertionError):
        KaliskiModInverse(n, p).on_classical_vals(x=x)
    u, v, r, s, m, f, t, work, bits = layout(n)
    width = 48
    raw = pack((0, x, 0, 0, 0, 0, 0), (u, v, r, s, m, (f,), t), width)
    before = tuple(raw >> (width - 1 - i) & 1 for i in range(width))
    actual, _ = certified_and_channel_map(
        inverse_algorithm_ir(u, v, r, s, m, f, t, p, work, bits), (before,)
    )
    assert any(actual[0][wire] for wire in (*u, *v, *s, f))
