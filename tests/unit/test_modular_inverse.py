"""Original public inverse's inlined graph, logical junk and physical roles."""

from collections import Counter
from typing import Any

import pytest

from flagquantum.compiler.registers.modular.modular_inverse import (
    modular_inverse_ir,
    modular_inverse_outputs,
)
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.inverse_algorithm_source import walk_inverse_source
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_inverse_algorithm import assert_complex_maps, cases, pack
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(n: int) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(range(n)),
        tuple(range(n, 5 * n)),
        tuple(range(5 * n, 10 * n + 8)),
        tuple(range(40, 40 + max(2 * n - 1, n + 1))),
    )


def original_public(n: int, p: int, adjoint: bool) -> tuple[Instruction, ...]:
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.mod_arithmetic.mod_division import KaliskiModInverse

    target, junk, work, bits = layout(n)
    result, returned = modular_inverse_outputs(target, work)
    x, w = (result, returned) if adjoint else (target, work)
    u, r, s, f, pool = (
        w[:n],
        w[n : 2 * n],
        w[2 * n : 3 * n],
        w[3 * n : 3 * n + 1],
        w[3 * n + 1 :],
    )
    source = KaliskiModInverse(n, p, uncompute=adjoint)
    cb = source.decompose_bloq()
    entries = tuple(cb.iter_bloqsoqs())
    by_instance = {instance: inputs for instance, inputs, _ in entries}
    first = next(
        inputs
        for instance, inputs, _ in entries
        if type(getattr(instance.bloq, "subbloq", instance.bloq)).__name__
        == "_KaliskiIteration"
    )

    def allocated_ancestor(value: Any) -> Any:
        """Follow actual same-port source edges before the first iteration."""
        soq = np.asarray(value, dtype=object).item()
        visited = set()
        while type(soq.binst.bloq).__name__ != "Allocate":
            assert soq not in visited
            visited.add(soq)
            parent_inputs = by_instance[soq.binst]
            assert soq.reg.name in parent_inputs
            soq = np.asarray(parent_inputs[soq.reg.name], dtype=object).item()
        return soq

    bindings = {}
    for name, wires in (("u", u), ("v" if adjoint else "r", r), ("s", s), ("f", f)):
        soq = allocated_ancestor(first[name])
        assert soq not in bindings and soq.reg.total_bits() == len(wires)
        bindings[soq] = wires
    if not adjoint:
        for name, wires in (("m", junk[: 2 * n]), ("is_terminal", junk[2 * n :])):
            soq = allocated_ancestor(first[name])
            assert soq not in bindings and soq.reg.total_bits() == len(wires)
            bindings[soq] = wires
    allocated_nodes = {
        instance
        for instance, _, _ in entries
        if type(instance.bloq).__name__ == "Allocate"
    }
    assert {soq.binst for soq in bindings} == allocated_nodes
    clean_after = (
        (work[:n], work[n : 2 * n], work[2 * n : 3 * n], work[3 * n : 3 * n + 1])
        if adjoint
        else (u, target, s, f)
    )
    frees = (*clean_after, junk[: 2 * n], junk[2 * n :]) if adjoint else clean_after
    inputs = dict(x=x, junk=junk) if adjoint else dict(x=x)
    desired = dict(x=target) if adjoint else dict(x=result, junk=junk)
    gates, outputs, actual_pool, freed = walk_inverse_source(
        source,
        inputs,
        desired,
        n,
        p,
        adjoint,
        pool,
        bits,
        bindings=bindings,
        allowed_frees=frees,
    )
    assert outputs == desired and Counter(freed) == Counter(frees)
    assert actual_pool == (work[3 * n + 1 :] if adjoint else returned[3 * n + 1 :])
    return gates


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", [(2, 1), (3, 5), (4, 9), (8, 251)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_compute_uncompute_graph_ports_and_models(
    n: int, p: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import KaliskiModInverse
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    x, junk, work, bits = layout(n)
    if adjoint:
        x, work = modular_inverse_outputs(x, work)
    gates = modular_inverse_ir(x, junk, p, work, bits, adjoint=adjoint)
    assert_source(gates, original_public(n, p, adjoint))
    model = get_cost_value(KaliskiModInverse(n, p, uncompute=adjoint), QECGatesCost())
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
    assert (
        get_cost_value(KaliskiModInverse(n, p, uncompute=adjoint), QubitCount())
        == 10 * n + 8
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", [(2, 1), (3, 3), (3, 5), (4, 9), (4, 13), (8, 251)])
def test_all_units_zero_junk_and_clean_work_complete_history_channel(
    n: int, p: int
) -> None:
    x, junk, work, bits = layout(n)
    result, returned = modular_inverse_outputs(x, work)
    width = 10 * n + 8
    before, expected = [], []
    for value, y, history, terminal in cases(n, p):
        for collection, values, ports in (
            (before, (value, 0), (x, junk)),
            (expected, (y, (history << (2 * n)) | terminal), (result, junk)),
        ):
            raw = pack(values, ports, width)
            collection.append(tuple(raw >> (width - 1 - i) & 1 for i in range(width)))
    gates = modular_inverse_ir(x, junk, p, work, bits)
    actual, events = certified_and_channel_map(gates, tuple(before))
    assert actual == tuple(expected) and events == 2 * n * (8 * n + 5) + n - 1
    reverse = modular_inverse_ir(result, junk, p, returned, bits, adjoint=True)
    restored, back = certified_and_channel_map(reverse, actual)
    assert restored == tuple(before) and back == events
    assert all(
        restored_row[wire] == 0 for restored_row in restored for wire in (*junk, *work)
    )
    assert modular_inverse_outputs(result, returned, adjoint=True) == (x, work)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", [(3, 5), (4, 9), (8, 251)])
def test_full_complex_spectator_sparse_primitive_channels(n: int, p: int) -> None:
    x, junk, work, bits = layout(n)
    y, returned = modular_inverse_outputs(x, work)
    width = 10 * n + 9
    spectator = width - 1
    data = cases(n, p)
    state, expected = {}, {}
    norm = (
        sum(
            abs(complex(2 * i + q + 1, 3 * i - q - 4)) ** 2
            for i in range(len(data))
            for q in (0, 1)
        )
        ** 0.5
    )
    for i, (value, out, history, terminal) in enumerate(data):
        for q in (0, 1):
            amp = complex(2 * i + q + 1, 3 * i - q - 4) / norm
            state[pack((value, 0, q), (x, junk, (spectator,)), width)] = amp
            expected[
                pack(
                    (out, (history << (2 * n)) | terminal, q),
                    (y, junk, (spectator,)),
                    width,
                )
            ] = amp
    gates = modular_inverse_ir(x, junk, p, work, bits)
    reverse = modular_inverse_ir(y, junk, p, returned, bits, adjoint=True)
    for seed in (2, 11, 23):
        actual = execute_sparse_register(
            CircuitIR(width, gates), state.copy(), seed=seed
        )
        assert_complex_maps(actual, expected)
        restored = execute_sparse_register(
            CircuitIR(width, reverse), actual, seed=seed + 2
        )
        assert_complex_maps(restored, state)
    # Replace each native direction by the corresponding original public graph.
    actual = execute_sparse_register(
        CircuitIR(width, original_public(n, p, False)), state.copy(), seed=31
    )
    assert_complex_maps(actual, expected)
    assert_complex_maps(
        execute_sparse_register(
            CircuitIR(width, original_public(n, p, True)), actual, seed=37
        ),
        state,
    )


@pytest.mark.parametrize(
    "bad", ["n", "work", "adjoint", "junk", "alias", "classical", "modulus"]
)
def test_layout_and_source_boundaries_rejected(bad: str) -> None:
    x, junk, w, bits = layout(3)
    p, reverse = 5, False
    if bad == "n":
        x = x[:1]
    elif bad == "work":
        w = w[:-1]
    elif bad == "adjoint":
        reverse = 1
    elif bad == "junk":
        junk = junk[:-1]
    elif bad == "alias":
        junk = (x[0], *junk[1:])
    elif bad == "classical":
        bits = (40,) * len(bits)
    else:
        p = 7
    with pytest.raises(ValueError):
        modular_inverse_ir(x, junk, p, w, bits, adjoint=reverse)


@pytest.mark.ftqc_reference
def test_uncompute_requires_matching_original_history() -> None:
    x, junk, work, bits = layout(3)
    result, returned = modular_inverse_outputs(x, work)
    assert next(row for row in cases(3, 5) if row[0] == 1) == (1, 4, 36, 4)
    # The inverse's logical history cannot be replaced by fresh zero scratch.
    raw = pack((4, 0), (result, junk), 38)
    before = tuple(raw >> (37 - i) & 1 for i in range(38))
    actual, _ = certified_and_channel_map(
        modular_inverse_ir(result, junk, 5, returned, bits, adjoint=True), (before,)
    )
    assert any(actual[0][wire] for wire in (*junk, *work))
    expected = pack((1, 0), (x, junk), 38)
    assert actual[0] != tuple(expected >> (37 - i) & 1 for i in range(38))
