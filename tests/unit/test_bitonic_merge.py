"""Original bitonic merge gates/ports, coherent histories and shared clean work."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.bitonic_merge import (
    bitonic_merge_history_count,
    bitonic_merge_ir,
)
from flagquantum.compiler.registers.comparison_layout import (
    comparison_network_requirements,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(1, 1), (1, 2), (1, 8), (2, 1), (2, 2), (4, 1), (4, 3)]


def layout(
    k: int, n: int, count: int
) -> tuple[
    tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]:
    q, c = comparison_network_requirements(n, count)
    return (
        tuple(tuple(range(i * n, (i + 1) * n)) for i in range(k)),
        tuple(range(k * n, k * n + count)),
        tuple(range(k * n + count, k * n + count + q)),
        tuple(range(40, 40 + c)),
    )


def reference_network(
    source: Any,
    words: tuple[tuple[int, ...], ...],
    history: tuple[int, ...],
    work: tuple[int, ...],
    bits: tuple[int, ...],
    *,
    merge: bool,
    adjoint: bool,
) -> tuple[Instruction, ...]:
    """Read original hierarchical Soquet connections, retaining child lifetimes.

    Global flattening schedules independent source branches greedily and changes
    temporary birth order. Walk original component graphs instead; only wire
    names are bound here. All leaf gates come from unchanged Comparator objects.
    """
    import numpy as np
    from qualtran import LeftDangle, Soquet

    from flagquantum.compiler.registers.ordering import greater_ir

    def soqs(value: Any) -> tuple[Any, ...]:
        return tuple(np.asarray(value, dtype=object).flat)

    def walk(
        bloq: Any,
        inputs: dict[str, tuple[int, ...]],
        desired: dict[str, tuple[int, ...]],
    ) -> tuple[tuple[Instruction, ...], dict[str, tuple[int, ...]]]:
        underlying = getattr(bloq, "subbloq", bloq)
        if type(underlying).__name__ == "Comparator":
            outputs = {
                reg.name: inputs[reg.name]
                for reg in bloq.signature.rights()
                if reg.name in inputs
            }
            outputs.update(desired)
            marker = inputs["out"] if "out" in inputs else outputs["out"]
            binding = greater_ir(inputs["a"], inputs["b"], marker[0], work, bits)
            measured = {
                g.wires[0]: g.metadata["classical_bit"]
                for g in binding
                if g.name == "measure"
            }
            allocations = work if "out" in inputs else (*marker, *work)
            gates = reference_arithmetic_gates(
                bloq.decompose_bloq(),
                inputs,
                allocations,
                measured,
                reuse_allocations=False,
                preserve_components=True,
                expected_outputs=outputs,
            )
            return gates, outputs

        cb = bloq.decompose_bloq()
        mapping: dict[Any, tuple[int, ...]] = {}

        def bind(
            register: Any, values: tuple[Any, ...], wires: tuple[int, ...]
        ) -> None:
            width = int(register.bitsize)
            assert len(wires) == len(values) * width
            for i, soq in enumerate(values):
                word = wires[i * width : (i + 1) * width]
                assert soq not in mapping or mapping[soq] == word
                mapping[soq] = word

        for register in cb.signature.lefts():
            bind(
                register,
                tuple(Soquet(LeftDangle, register, idx) for idx in register.all_idxs()),
                inputs[register.name],
            )
        final = cb.final_soqs()
        for register in cb.signature.rights():
            if register.name in desired:
                bind(register, soqs(final[register.name]), desired[register.name])
        gates = []
        for instance, old_inputs, old_outputs in cb.iter_bloqsoqs():
            child_inputs = {
                name: tuple(wire for soq in soqs(value) for wire in mapping[soq])
                for name, value in old_inputs.items()
            }
            child_outputs = {}
            right_registers = tuple(instance.bloq.signature.rights())
            for register, value in zip(right_registers, old_outputs, strict=True):
                values = soqs(value)
                if all(soq in mapping for soq in values):
                    child_outputs[register.name] = tuple(
                        wire for soq in values for wire in mapping[soq]
                    )
            child_gates, actual_outputs = walk(
                instance.bloq, child_inputs, child_outputs
            )
            gates.extend(child_gates)
            for register, value in zip(right_registers, old_outputs, strict=True):
                bind(register, soqs(value), actual_outputs[register.name])
        outputs = {
            name: tuple(wire for soq in soqs(value) for wire in mapping[soq])
            for name, value in final.items()
        }
        assert all(outputs[name] == wires for name, wires in desired.items())
        return tuple(gates), outputs

    root = source.adjoint() if adjoint else source
    flat = tuple(wire for word in words for wire in word)
    if merge:
        half = len(flat) // 2
        inputs = (
            {"result": flat, "junk": history}
            if adjoint
            else {"xs": flat[:half], "ys": flat[half:]}
        )
        outputs = (
            {"xs": flat[:half], "ys": flat[half:]}
            if adjoint
            else {"result": flat, "junk": history}
        )
    else:
        inputs = {"xs": flat} | ({"junk": history} if adjoint else {})
        outputs = {"xs": flat} | ({} if adjoint else {"junk": history})
    gates, actual_outputs = walk(root, inputs, outputs)
    assert actual_outputs == outputs
    return gates


def pack(values: tuple[int, ...], n: int, histories: tuple[int, ...], work: int) -> int:
    value = 0
    for word in values:
        value = (value << n) | word
    for bit in histories:
        value = (value << 1) | bit
    return value << work


def input_words(k: int, n: int, *, merge: bool) -> tuple[tuple[int, ...], ...]:
    if k * n <= 8:
        labels = tuple(product(range(1 << n), repeat=k))
    else:
        labels = (
            tuple([0] * k),
            tuple([(1 << n) - 1] * k),
            tuple(i % (1 << n) for i in range(k)),
            tuple((k - 1 - i) % (1 << n) for i in range(k)),
        )
    if not merge:
        return labels
    half = k // 2
    if k * n > 8:
        return tuple(tuple(sorted(v[:half])) + tuple(sorted(v[half:])) for v in labels)
    return tuple(
        v
        for v in labels
        if tuple(sorted(v[:half])) == v[:half] and tuple(sorted(v[half:])) == v[half:]
    ) or (tuple([0] * k),)


@pytest.mark.parametrize("half,n", CASES)
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_gate_chain_ports_and_models(half: int, n: int, adjoint: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    count = bitonic_merge_history_count(half)
    words, h, w, bits = layout(2 * half, n, count)
    gates = bitonic_merge_ir(words, h, w, bits, adjoint=adjoint)
    parent = BitonicMerge(half, n)
    assert_source(
        gates, reference_network(parent, words, h, w, bits, merge=True, adjoint=adjoint)
    )
    source = parent.adjoint() if adjoint else parent
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert count == parent.num_comparisons
    assert (
        counts["t"] + counts["tdg"]
        == model.total_t_count(ts_per_cswap=7)
        == count * (15 * n - 4)
    )
    assert model.total_t_count() == count * (12 * n - 4)
    assert counts["measure"] == model.measurement == count * (2 * n - 1)
    assert get_cost_value(source, QubitCount()) == 2 * half * n + len(h) + len(w)


@pytest.mark.parametrize("half,n", CASES)
def test_merge_valid_basis_all_histories_clean_work_and_restore(
    half: int, n: int
) -> None:
    count = bitonic_merge_history_count(half)
    words, h, w, bits = layout(2 * half, n, count)
    labels = input_words(2 * half, n, merge=True)
    width = 2 * half * n + count + len(w)
    before = tuple(
        tuple(
            (pack(v, n, (0,) * count, len(w)) >> (width - 1 - i)) & 1
            for i in range(width)
        )
        for v in labels
    )
    actual, m = certified_and_channel_map(bitonic_merge_ir(words, h, w, bits), before)
    assert m == count * (2 * n - 1)
    for values, out in zip(labels, actual, strict=True):
        observed = tuple(
            sum(out[i * n + j] << (n - 1 - j) for j in range(n))
            for i in range(2 * half)
        )
        assert observed == tuple(sorted(values)) and not any(out[-len(w) :])
    restored, m2 = certified_and_channel_map(
        bitonic_merge_ir(words, h, w, bits, adjoint=True), actual
    )
    assert restored == before and m2 == m


def raw_channels(
    source: Any,
    words: tuple[tuple[int, ...], ...],
    h: tuple[int, ...],
    w: tuple[int, ...],
    bits: tuple[int, ...],
    *,
    merge: bool,
    adjoint: bool,
) -> None:
    import numpy as np

    n = len(words[0])
    k = len(words)
    count = len(h)
    width = k * n + count + len(w) + 1
    labels = input_words(k, n, merge=merge)
    coeff = torch.arange(2 * len(labels), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for i, values in enumerate(labels):
        if merge:
            result, hist = source.call_classically(
                xs=np.array(values[: k // 2]), ys=np.array(values[k // 2 :])
            )
        else:
            result, hist = source.call_classically(xs=np.array(values))
        assert tuple(result) == tuple(sorted(values))
        before = pack(values, n, (0,) * count, len(w))
        after = pack(tuple(map(int, result)), n, tuple(map(int, hist)), len(w))
        if adjoint:
            before, after = after, before
        for spectator in (0, 1):
            state[(before << 1) | spectator] = amps[2 * i + spectator]
            expected[(after << 1) | spectator] = amps[2 * i + spectator]
    if merge:
        emit = bitonic_merge_ir
    else:
        from flagquantum.compiler.registers.bitonic_sort import bitonic_sort_ir

        emit = bitonic_sort_ir
    gates = emit(words, h, w, bits, adjoint=adjoint)
    m = count * (2 * n - 1)
    for branch in product((0, 1), repeat=m):
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


@pytest.mark.parametrize("half,n", [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_all_small_raw_complex_branches_against_source_values(
    half: int, n: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge

    words, h, w, bits = layout(2 * half, n, bitonic_merge_history_count(half))
    raw_channels(BitonicMerge(half, n), words, h, w, bits, merge=True, adjoint=adjoint)


def test_bounds_and_unsorted_halves_are_not_globally_sorted() -> None:
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge

    for half in (0, 3, True):
        with pytest.raises(ValueError, match="power of two"):
            bitonic_merge_history_count(half)
    with pytest.raises(DecomposeNotImplementedError):
        BitonicMerge(3, 2).decompose_bloq()
    words, h, w, bits = layout(4, 2, 4)
    values = (1, 2, 3, 0)
    expected, hist = BitonicMerge(2, 2).call_classically(
        xs=np.array(values[:2]), ys=np.array(values[2:])
    )
    assert tuple(expected) != (0, 1, 2, 3)
    width = 8 + len(h) + len(w)
    x = pack(values, 2, (0,) * 4, len(w))
    actual, _ = certified_and_channel_map(
        bitonic_merge_ir(words, h, w, bits),
        (tuple((x >> (width - 1 - i)) & 1 for i in range(width)),),
    )
    y = pack(tuple(map(int, expected)), 2, tuple(map(int, hist)), len(w))
    assert actual == (tuple((y >> (width - 1 - i)) & 1 for i in range(width)),)
    with pytest.raises(ValueError, match="equal-length"):
        bitonic_merge_ir(words[:3], h, w, bits)
