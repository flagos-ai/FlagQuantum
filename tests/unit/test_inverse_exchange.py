"""Original paired controlled swaps with exact phases and separated cost models."""

from collections import Counter
from itertools import product

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.modular.inversion.inverse_exchange import (
    inverse_exchange_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def words(n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(range(1 + i * n, 1 + (i + 1) * n)) for i in range(4))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_whole_layer_gates_and_two_model_weights(
    n: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mod_arithmetic.mod_division import _KaliskiIterationStep4
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    u, v, r, s = words(n)
    source = _KaliskiIterationStep4(n)
    if adjoint:
        source = source.adjoint()
    gates = inverse_exchange_ir(0, u, v, r, s, adjoint=adjoint)
    expected = reference_arithmetic_gates(
        source, dict(u=u, v=v, r=r, s=s, a=(0,)), (), {}, preserve_components=True
    )
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7) == 14 * n
    assert model.total_t_count() == 8 * n
    assert counts["cx"] == 16 * n and counts["h"] == 4 * n
    assert counts["measure"] == counts["reset"] == model.measurement == 0
    assert get_cost_value(source, QubitCount()) == 4 * n + 1


@pytest.mark.parametrize("n", [1, 2, 3, 8])
def test_all_small_words_all_controls_and_restore(n: int) -> None:
    width = 4 * n + 1
    values = range(1 << n) if n <= 3 else (0, 1, 127, 128, 254, 255)
    before, after = [], []
    for c, u, v, r, s in product((0, 1), values, values, values, values):
        x = (c << (4 * n)) | (u << (3 * n)) | (v << (2 * n)) | (r << n) | s
        if c:
            u, v, r, s = v, u, s, r
        y = (c << (4 * n)) | (u << (3 * n)) | (v << (2 * n)) | (r << n) | s
        before.append(tuple(x >> (width - 1 - i) & 1 for i in range(width)))
        after.append(tuple(y >> (width - 1 - i) & 1 for i in range(width)))
    gates = inverse_exchange_ir(0, *words(n))
    actual, events = certified_and_channel_map(gates, tuple(before))
    assert actual == tuple(after) and events == 0
    restored, _ = certified_and_channel_map(
        inverse_exchange_ir(0, *words(n), adjoint=True), actual
    )
    assert restored == tuple(before)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.integration
def test_complete_complex_words_control_and_entangled_spectator(
    n: int, adjoint: bool
) -> None:
    width = 4 * n + 2
    data = torch.arange(1 << width, dtype=torch.int64)
    control = data >> (4 * n + 1)
    words_bits = data >> 1 & ((1 << (4 * n)) - 1)
    mask = (1 << n) - 1
    u, v, r, s = (
        (words_bits >> (3 * n)) & mask,
        (words_bits >> (2 * n)) & mask,
        (words_bits >> n) & mask,
        words_bits & mask,
    )
    exchanged = (v << (3 * n)) | (u << (2 * n)) | (s << n) | r
    out = (
        (control << (4 * n + 1))
        | (torch.where(control.bool(), exchanged, words_bits) << 1)
        | (data & 1)
    )
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    expected = torch.zeros_like(amps)
    expected[out] = amps
    gates = inverse_exchange_ir(0, *words(n), adjoint=adjoint)
    circuit = Circuit(width, inputs=amps, dtype=torch.complex128).append(
        FieldOperation(width, gates)
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    inverse = inverse_exchange_ir(0, *words(n), adjoint=not adjoint)
    torch.testing.assert_close(
        circuit.append(FieldOperation(width, inverse)).state().flatten(),
        amps,
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("bad", ["empty", "width", "alias", "wire", "adjoint"])
def test_invalid_physical_layouts(bad: str) -> None:
    u, v, r, s = words(2)
    inverse = False
    if bad == "empty":
        u = ()
    elif bad == "width":
        s = s[:-1]
    elif bad == "alias":
        r = u
    elif bad == "wire":
        r = (-1, r[1])
    else:
        inverse = 1
    with pytest.raises(ValueError):
        inverse_exchange_ir(0, u, v, r, s, adjoint=inverse)
