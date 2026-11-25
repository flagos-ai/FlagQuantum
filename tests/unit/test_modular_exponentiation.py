"""Check the pinned RSA modular exponentiation schedule and amplitudes."""

from collections import Counter
from importlib.metadata import version
from itertools import pairwise

import pytest

from flagquantum.algorithms.primitives.modular_exponentiation import (
    ModularExponentiation,
)
from flagquantum.compiler.registers.layout.classical_storage import (
    bind_classical_scratch,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_elliptic_curve_addition_slope import pack
from tests.unit.test_legacy_modular_exponentiation_factories import (
    _resolve,
    original_modexp_stream,
)
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_symbolic_source_and_original_factoring_factory() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp
    from sympy import Symbol, symbols

    assert version("qualtran") == "0.7.0"
    g, p, e, n = symbols("g N n_e n_x")
    native = ModularExponentiation(g, p, e, n)
    source = ModExp(g, p, e, n)
    assert [(r.name, r.bitsize, r.side.name) for r in source.signature] == [
        (r.name, r.dtype.num_qubits, r.side.name) for r in native.source_signature
    ]
    assert native.source_label == source.wire_symbol(None).text
    assert native.source_call_graph().leaf_counts == {
        str(child): count for child, count in source.bloq_counts().items()
    }
    exponent = Symbol("b")
    assert native.classical_values(exponent) == source.call_classically(
        exponent=exponent
    )
    formal = ModularExponentiation.for_shor(p)
    ref = ModExp.make_for_shor(p)
    assert (
        formal.base,
        formal.modulus,
        formal.exponent_width,
        formal.output_width,
    ) == (ref.base, ref.mod, ref.exp_bitsize, ref.x_bitsize)

    native_rng = np.random.RandomState(52)
    source_rng = np.random.RandomState(52)
    for _ in range(100):
        left = int(native_rng.randint(1, 128))
        right = int(source_rng.randint(1, 128))
        assert left == right
        actual = ModularExponentiation.for_shor(91, random_state=native_rng)
        original = ModExp.make_for_shor(91, rs=source_rng)
        assert (
            actual.base,
            actual.modulus,
            actual.exponent_width,
            actual.output_width,
        ) == (original.base, original.mod, original.exp_bitsize, original.x_bitsize)
        assert actual.classical_values(
            left
        ) == original.decompose_bloq().call_classically(exponent=right)


@pytest.mark.parametrize(
    "base,p,e,n,constants",
    [(2, 3, 3, 2, (2, 1, 1)), (4, 15, 4, 4, (4, 1, 1, 1)), (5, 6, 3, 3, (5, 1, 1))],
)
def test_original_order_operands_and_full_emitted_channels(
    base: int,
    p: int,
    e: int,
    n: int,
    constants: tuple[int, ...],
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import IntState
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp
    from qualtran.bloqs.mod_arithmetic import CModMulK
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    model = ModularExponentiation(base, p, e, n)
    source = ModExp(base, p, e, n)
    original = source.decompose_bloq()
    entries = sorted(original.iter_bloqsoqs(), key=lambda row: row[0].i)
    assert [
        (node.bloq.val, node.bloq.bitsize)
        for node, _, _ in entries
        if isinstance(node.bloq, IntState)
    ] == [(1, n)]
    body = [
        (node.bloq, ports, outs)
        for node, ports, outs in entries
        if isinstance(node.bloq, CModMulK)
    ]
    assert tuple(node.k for node, _, _ in body) == constants
    assert tuple(ports["ctrl"].idx for _, ports, _ in body) == tuple(
        (i,) for i in reversed(range(e))
    )
    initial = next(
        outs[0] for node, _, outs in entries if isinstance(node.bloq, IntState)
    )
    assert body[0][1]["x"] == initial
    assert all(right[1]["x"] == left[2][1] for left, right in pairwise(body))
    assert original.final_soqs()["x"] == body[-1][2][1]

    q, c = model.workspace_requirements()
    width = e + n + q
    wires = tuple(reversed(range(width)))
    exponent, target, work = wires[:e], wires[e : e + n], wires[e + n :]
    bits = tuple(range(71, 71 + 3 * c, 3))
    bound = model.make_on(
        exponent=exponent, target=target, workspace=work, classical_bits=bits
    )
    ir = DynamicCircuit(width).append(bound).to_ir()
    record = _resolve(source)
    reference = bind_register_program(
        record,
        (exponent,),
        clean=(*target, *work),
        allow_workspace_outputs=True,
    )
    reference = bind_classical_scratch(reference, bits)
    assert reference.outputs == (exponent, target)
    expected = original_modexp_stream(record, source, reference)
    assert_source(ir.instructions, expected)
    counts = Counter(g.name for g in ir.instructions)
    cost = get_cost_value(source, QECGatesCost())
    assert (
        counts["t"] + counts["tdg"]
        == cost.total_t_count(ts_per_cswap=7)
        == e * (40 * n * n + 23 * n)
    )
    assert counts["measure"] == cost.measurement == e * 2 * n * (5 * n + 2)
    assert width == get_cost_value(source, QubitCount())
    words = {"exponent": exponent, "x": target}
    before, after = [], []
    for value in range(1 << e):
        predicted = original.on_classical_vals(exponent=value)
        assert model.classical_values(value) == (value, predicted["x"])
        before.append(pack({"exponent": value}, words, width))
        after.append(pack(predicted, words, width))
    initial_bits = tuple(
        tuple(v >> (width - i - 1) & 1 for i in range(width)) for v in before
    )
    output, events = certified_and_channel_map(ir.instructions, initial_bits)
    assert events == counts["measure"]
    assert [
        sum(v << (width - i - 1) for i, v in enumerate(row)) for row in output
    ] == after
    weights = tuple(complex(i + 1, 2 - i) for i in range(len(before)))
    norm = sum(abs(v) ** 2 for v in weights) ** 0.5
    init = {j: v / norm for j, v in zip(before, weights, strict=True)}
    want = {j: v / norm for j, v in zip(after, weights, strict=True)}
    for seed in (11, 34):
        actual = execute_sparse_register(ir, init.copy(), seed=seed)
        assert {j: v for j, v in actual.items() if abs(v) > 1e-12} == pytest.approx(
            want, abs=1e-12
        )


def test_source_factory_modulus_fit_and_empty_exponent_boundaries() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp

    source = ModExp.make_for_shor(8, g=3)
    native = ModularExponentiation.for_shor(8, 3)
    assert source.x_bitsize == native.output_width == 3
    assert native.classical_values(3) == source.call_classically(exponent=3)
    with pytest.raises(ValueError, match="Too-large classical"):
        source.decompose_bloq().flatten()
    with pytest.raises(ValueError, match="fit the output"):
        native.workspace_requirements()
    assert ModularExponentiation(3, 8, 2, 4).workspace_requirements()[0] > 0
    empty = ModularExponentiation(2, 3, 0, 2)
    native_bound = empty.make_on(
        exponent=(), target=(1, 0), workspace=(), classical_bits=()
    )
    ir = DynamicCircuit(2).append(native_bound).to_ir()
    assert ir.instructions == (Instruction("x", (0,)),)
    assert empty.classical_values(0) == (0, 1)
    with pytest.raises(ValueError, match="invalid literal"):
        ModExp(2, 3, 0, 2).decompose_bloq().on_classical_vals(exponent=0)
    with pytest.raises(ValueError, match="coprime"):
        ModularExponentiation(6, 15, 2, 4)
