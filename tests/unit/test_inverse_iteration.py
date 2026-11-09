"""Original six-layer graph composition, reachable promises and exact channels."""

from collections import Counter
from typing import Any

import pytest
import torch

from examples.algorithms.register_inverse_updates import pack
from flagquantum.compiler.registers.modular.inversion.inverse_iteration import (
    inverse_iteration_ir,
    inverse_iteration_outputs,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_inverse_arithmetic import layout as arithmetic_layout
from tests.unit.test_inverse_arithmetic import source_gates as arithmetic_source
from tests.unit.test_inverse_ordering import layout as ordering_layout
from tests.unit.test_inverse_ordering import original_ordering_gates
from tests.unit.test_inverse_scaling import layout as scaling_layout
from tests.unit.test_inverse_scaling import source_gates as scaling_source
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(n: int) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
]:
    return (
        *(tuple(range(i * n, (i + 1) * n)) for i in range(4)),
        4 * n,
        4 * n + 1,
        4 * n + 2,
        tuple(range(4 * n + 3, 6 * n + 10)),
        tuple(range(40, 40 + max(2 * n - 1, n + 1))),
    )


def original_iteration(n: int, p: int, adjoint: bool) -> tuple[Instruction, ...]:
    """Walk the unchanged whole graph, binding allocations by actual input edges."""
    pytest.importorskip("qualtran")
    np = pytest.importorskip("numpy")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIteration

    u, v, r, s, m, f, t, work, bits = layout(n)
    result_r, result_work = inverse_iteration_outputs(r, work)
    input_r, input_work = (result_r, result_work) if adjoint else (r, work)
    a, b = work[:2]
    pool = work[2:]
    initial_pool = input_work[2:]
    source = _KaliskiIteration(n, p)
    if adjoint:
        source = source.adjoint()
    bindings = {}
    for instance, inputs, _ in source.decompose_bloq().iter_bloqsoqs():
        child = getattr(instance.bloq, "subbloq", instance.bloq)
        if type(child).__name__ == f"_KaliskiIterationStep{6 if adjoint else 2}":
            for key, wire in (("a", a), ("b", b)):
                soq = np.asarray(inputs[key], dtype=object).item()
                assert type(soq.binst.bloq).__name__ == "Allocate"
                bindings[soq] = (wire,)
    assert len(bindings) == 2

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        original = getattr(bloq, "subbloq", bloq)
        name = type(original).__name__
        if name == "Allocate":
            assert not inputs and len(desired["reg"]) == 1
            return (), desired
        if name == "Free":
            assert inputs["reg"] in ((a,), (b,)) and not desired
            return (), {}
        if not name.startswith("_KaliskiIterationStep"):
            return None
        step = int(name[-1])
        reverse = type(bloq).__name__ == "Adjoint"
        assert reverse == adjoint
        if step in (1, 2, 4):
            if step == 1:
                allocations, slots = pool[:n], dict(
                    zip(pool[:n], bits[:n], strict=True)
                )
            elif step == 2:
                allocations, slots = pool[:2], dict(
                    zip(pool[:2], bits[:2], strict=True)
                )
            else:
                allocations, slots = (), {}
            gates = reference_arithmetic_gates(
                bloq, inputs, allocations, slots, preserve_components=step == 4
            )
            return gates, inputs
        if step == 3:
            cu, cv, cb, ca, cm, cf, cw, cbit = ordering_layout(n)
            canonical = dict(u=cu, v=cv, b=(cb,), a=(ca,), m=(cm,), f=(cf,))
            stream = original_ordering_gates(n, reverse)
            actual_work = pool[: n + 4]
        elif step == 5:
            cu, cv, cr, cs, cb, cf, cw, cbit = arithmetic_layout(n)
            canonical = dict(u=cu, v=cv, r=cr, s=cs, b=(cb,), f=(cf,))
            stream = arithmetic_source(n, reverse)
            actual_work = pool[: 2 * n]
        else:
            assert step == 6
            cu, cv, cr, cs, cb, ca, cm, cf, cw, cbit = scaling_layout(n)
            canonical = dict(u=cu, v=cv, r=cr, s=cs, b=(cb,), a=(ca,), m=(cm,), f=(cf,))
            if reverse:
                canonical["r"] = (*cr[1:], cw[0])
                cw = (cr[0], *cw[1:])
            stream = scaling_source(n, p, reverse)
            actual_work = initial_pool
        qmap = dict(zip(cw, actual_work, strict=True))
        for key, wires in canonical.items():
            qmap.update(zip(wires, inputs[key], strict=True))
        cmap = dict(zip(cbit, bits[: len(cbit)], strict=True))
        gates = []
        for gate in stream:
            metadata = dict(gate.metadata)
            if "classical_bit" in metadata:
                metadata["classical_bit"] = cmap[metadata["classical_bit"]]
            if "conditions" in metadata:
                metadata["conditions"] = tuple(
                    (cmap[bit], val) for bit, val in metadata["conditions"]
                )
            gates.append(
                Instruction(
                    gate.name,
                    tuple(qmap[w] for w in gate.wires),
                    gate.params,
                    gate.matrix,
                    metadata,
                )
            )
        outputs = dict(inputs)
        if step == 6:
            output_r = cr if reverse else (*cr[1:], cw[0])
            outputs["r"] = tuple(qmap[w] for w in output_r)
        assert all(outputs[key] == val for key, val in desired.items())
        return tuple(gates), outputs

    inputs = dict(u=u, v=v, r=input_r, s=s, m=(m,), f=(f,), is_terminal=(t,))
    desired = dict(inputs, r=r if adjoint else result_r)
    gates, outputs = walk_source_graph(source, inputs, desired, leaf, bindings=bindings)
    assert outputs == desired
    return gates


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", [(2, 3), (3, 5), (5, 31), (8, 251)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_full_graph_stream_roles_and_models(
    n: int, p: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIteration
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    u, v, r, s, m, f, t, work, bits = layout(n)
    if adjoint:
        r, work = inverse_iteration_outputs(r, work)
    gates = inverse_iteration_ir(u, v, r, s, m, f, t, p, work, bits, adjoint=adjoint)
    assert_source(gates, original_iteration(n, p, adjoint))
    source = _KaliskiIteration(n, p)
    if adjoint:
        source = source.adjoint()
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert (
        counts["t"] + counts["tdg"]
        == model.total_t_count(ts_per_cswap=7)
        == 67 * n + 13
    )
    assert model.total_t_count() == 52 * n + 16
    assert counts["measure"] == counts["reset"] == model.measurement == 8 * n + 5
    assert get_cost_value(source, QubitCount()) == 6 * n + 10


def reachable(n: int, p: int, xs: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIteration

    source = _KaliskiIteration(n, p)
    cases = set()
    for x in xs:
        values = dict(u=p, v=x, r=0, s=1, m=0, f=1, is_terminal=0)
        for _ in range(2 * n):
            values.update(m=0, is_terminal=0)
            cases.add(
                tuple(
                    int(values[k])
                    for k in ("u", "v", "r", "s", "m", "f", "is_terminal")
                )
            )
            values = source.on_classical_vals(**values)
    return tuple(sorted(cases))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n,p", [(2, 3), (3, 5), (3, 7), (4, 13), (8, 251)])
def test_reachable_complete_channels_and_reverse(n: int, p: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIteration

    u, v, r, s, m, f, t, work, bits = layout(n)
    rr, ww = inverse_iteration_outputs(r, work)
    ports = (u, v, r, s, (m,), (f,), (t,))
    outports = (u, v, rr, s, (m,), (f,), (t,))
    width = 6 * n + 10
    source = _KaliskiIteration(n, p)
    cases = reachable(n, p, tuple(range(p)))
    initial = tuple(
        tuple(pack(vals, ports, width) >> (width - 1 - i) & 1 for i in range(width))
        for vals in cases
    )
    expected = []
    for vals in cases:
        output = source.on_classical_vals(
            **dict(
                zip(("u", "v", "r", "s", "m", "f", "is_terminal"), vals, strict=True)
            )
        )
        values = tuple(
            int(output[k]) for k in ("u", "v", "r", "s", "m", "f", "is_terminal")
        )
        raw = pack(values, outports, width)
        expected.append(tuple(raw >> (width - 1 - i) & 1 for i in range(width)))
    gates = inverse_iteration_ir(u, v, r, s, m, f, t, p, work, bits)
    actual, events = certified_and_channel_map(gates, initial)
    assert actual == tuple(expected) and events == 8 * n + 5
    reverse = inverse_iteration_ir(u, v, rr, s, m, f, t, p, ww, bits, adjoint=True)
    restored, back = certified_and_channel_map(reverse, actual)
    assert restored == initial and back == events
    assert inverse_iteration_outputs(rr, ww, adjoint=True) == (r, work)


@pytest.mark.ftqc_reference
def test_raw_complex_selected_full_histories() -> None:
    pytest.importorskip("qualtran")
    n, p = 2, 3
    u, v, r, s, m, f, t, work, bits = layout(n)
    rr, ww = inverse_iteration_outputs(r, work)
    width = 6 * n + 10
    cases = reachable(n, p, (0, 1, 2))
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIteration

    source = _KaliskiIteration(n, p)
    state = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    amps = torch.complex(
        torch.arange(1, len(cases) + 1, dtype=torch.float64),
        torch.arange(len(cases), dtype=torch.float64) - 3,
    )
    amps /= amps.norm()
    for values, amp in zip(cases, amps, strict=True):
        output = source.on_classical_vals(
            **dict(
                zip(("u", "v", "r", "s", "m", "f", "is_terminal"), values, strict=True)
            )
        )
        out = tuple(
            int(output[k]) for k in ("u", "v", "r", "s", "m", "f", "is_terminal")
        )
        state[pack(values, (u, v, r, s, (m,), (f,), (t,)), width)] = amp
        expected[pack(out, (u, v, rr, s, (m,), (f,), (t,)), width)] = amp
    gates = inverse_iteration_ir(u, v, r, s, m, f, t, p, work, bits)
    reverse = inverse_iteration_ir(u, v, rr, s, m, f, t, p, ww, bits, adjoint=True)
    events = 21
    # All 2**21 histories are certified above; raw vectors sample three histories.
    for history in ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events))):
        actual = forced_branch(state, gates, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), rtol=0, atol=1e-12
        )
        torch.testing.assert_close(
            forced_branch(actual, reverse, history),
            state / 2**events,
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "bad", ["short", "words", "alias", "work", "bits", "adjoint", "modulus"]
)
def test_invalid_layout_rejected(bad: str) -> None:
    u, v, r, s, m, f, t, work, bits = layout(2)
    adjoint = False
    p = 3
    if bad == "short":
        u = u[:1]
    elif bad == "words":
        s = s[:1]
    elif bad == "alias":
        t = m
    elif bad == "work":
        work = work[:-1]
    elif bad == "bits":
        bits = (40, 40, 41)
    elif bad == "adjoint":
        adjoint = 1
    else:
        p = 2
    with pytest.raises(ValueError):
        inverse_iteration_ir(u, v, r, s, m, f, t, p, work, bits, adjoint=adjoint)


@pytest.mark.ftqc_reference
def test_off_domain_source_free_is_not_a_reset() -> None:
    """Inactive even s is not a reachable terminal state and leaves original a=1."""
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIteration

    u, v, r, s, m, f, t, work, bits = layout(2)
    values = (1, 0, 0, 0, 0, 0, 0)
    width = 22
    raw = pack(values, (u, v, r, s, (m,), (f,), (t,)), width)
    initial = tuple(raw >> (width - 1 - i) & 1 for i in range(width))
    gates = inverse_iteration_ir(u, v, r, s, m, f, t, 3, work, bits)
    assert_source(gates, original_iteration(2, 3, False))
    actual, _ = certified_and_channel_map(gates, (initial,))
    _, returned = inverse_iteration_outputs(r, work)
    assert tuple(actual[0][i] for i in returned) == (1,) + (0,) * 10
    assert (
        _KaliskiIteration(2, 3).on_classical_vals(
            u=1, v=0, r=0, s=0, m=0, f=0, is_terminal=0
        )["r"]
        == 0
    )


def test_output_binding_rejections() -> None:
    _, _, r, _, _, _, _, work, _ = layout(2)
    with pytest.raises(ValueError, match="boolean"):
        inverse_iteration_outputs(r, work, adjoint=1)
    with pytest.raises(ValueError, match="n>=2"):
        inverse_iteration_outputs(r[:1], work)
