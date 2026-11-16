"""Check the locked fixed and windowed elliptic-curve addition paths."""

from collections import Counter
from importlib.metadata import version
from itertools import product

import pytest

from flagquantum.algorithms.primitives.elliptic_curve_fixed_point import (
    EllipticCurveFixedPointAddition,
    fixed_point_addition_signature,
)
from flagquantum.algorithms.primitives.elliptic_curve_point import ECPoint
from flagquantum.algorithms.primitives.elliptic_curve_window_addition import (
    EllipticCurveWindowAddition,
    windowed_point_addition_source_graph,
    windowed_point_classical_values,
    windowed_point_table,
)
from flagquantum.compiler.registers.lookup.qroam_resource_model import (
    source_qroam_clean_cost,
    source_qroam_measurement_erase_cost,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_elliptic_curve_addition_slope import pack
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_source_qroam_clean import source_clean_graph

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_raw_fixed_point_signature_and_original_callback() -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as ReferencePoint
    from sympy import symbols

    assert version("qualtran") == "0.7.0"
    n, p, u, v, k = symbols("n p u v k")
    symbolic = ECAddR(n, ReferencePoint(u, v, p, 1))
    native = EllipticCurveFixedPointAddition(n, ECPoint(u, v, p, 1))
    assert [(r.name, r.bitsize, r.shape) for r in symbolic.signature] == [
        (r.name, r.dtype.num_qubits, r.shape) for r in native.source_signature
    ]
    assert fixed_point_addition_signature(n, k).get_left("ctrl").shape_symbolic == (k,)
    source = ECAddR(3, ReferencePoint(0, 1, 3, 1))
    native = EllipticCurveFixedPointAddition(3, ECPoint(0, 1, 3, 1))
    for c, (x, y) in product((0, 1), ((0, 0), (0, 1), (0, 2), (1, 0))):
        values = source.on_classical_vals(c, x, y)
        assert native.classical_values(c, x, y) == tuple(
            values[name] for name in ("ctrl", "x", "y")
        )
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()


def layout(
    op: EllipticCurveWindowAddition, *, reverse: bool = False
) -> tuple[CircuitIR, dict[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...]]:
    n, k = op.width, op.window
    q, c = op.workspace_requirements()
    width = k + 2 * n + q
    wires = tuple(range(width))
    if reverse:
        wires = wires[::-1]
    ports = {"ctrl": wires[:k], "x": wires[k : k + n], "y": wires[k + n : k + 2 * n]}
    work = wires[k + 2 * n :]
    slots = tuple(101 + 2 * i for i in range(c))
    ir = (
        DynamicCircuit(width)
        .append(op.make_on(**ports, workspace=work, classical_bits=slots))
        .to_ir()
    )
    return ir, ports, work, slots


@pytest.mark.parametrize("k", [1, 5])
def test_source_lookup_operands_tables_and_declared_adjoint(k: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as ReferencePoint
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    n, p = 3, 3
    native = EllipticCurveWindowAddition(n, ECPoint(0, 1, p, 1), k)
    source = ECWindowAddR(n, ReferencePoint(0, 1, p, 1), k, 1)
    assert [(r.name, r.bitsize, r.shape) for r in source.signature] == [
        (r.name, r.dtype.num_qubits, r.shape) for r in native.source_signature
    ]
    rows = native.table
    assert tuple(tuple(data.tolist()) for data in source.qrom.data) == rows.datasets
    assert source.qrom.log_block_sizes == rows.log_block_sizes == (int(k == 5),)
    concrete_example = windowed_point_table(16, 4, ECPoint(2, 2, 7, 3))
    assert concrete_example.datasets == tuple(
        tuple(data.tolist())
        for data in ECWindowAddR(16, ReferencePoint(2, 2, 7, 3), 4).qrom.data
    )
    ir, ports, work, slots = layout(native)
    cells = rows.block_shape[0]
    count = 3 * n * cells
    q = rows.workspace_size(0, (k,))
    outputs = {"selection": ports["ctrl"]}
    for i in range(3):
        start = i * cells * n
        outputs[f"target{i}_"] = work[start : start + n]
        if cells > 1:
            outputs[f"junk_target{i}_"] = work[start + n : start + cells * n]
    expected, _, _ = source_clean_graph(
        source.qrom,
        {"selection": ports["ctrl"]},
        outputs,
        work[count : count + q],
        slots[:q],
    )
    assert_source(ir.instructions[: len(expected)], expected)
    loads = source_qroam_clean_cost(rows, 0, (k,), (n, n, n))
    assert loads.qubits == get_cost_value(source.qrom, QubitCount())
    erasure = source_qroam_measurement_erase_cost(rows, 0, (n, n, n)).wrapper
    locked = source.qrom.adjoint()
    assert erasure.gates.toffoli == get_cost_value(locked, QECGatesCost()).toffoli
    with pytest.raises(DecomposeNotImplementedError):
        locked.qroam_clean_adjoint_bloq.decompose_bloq()
    assert {
        op.metadata["classical_bit"] for op in ir.instructions if op.name == "measure"
    } >= set(slots[-count:])


@pytest.mark.parametrize("k", [1, 5])
def test_coherent_window_all_addresses_and_curve_inputs(k: int) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as ReferencePoint

    n, p = 3, 3
    native = EllipticCurveWindowAddition(n, ECPoint(0, 1, p, 1), k)
    source = ECWindowAddR(n, ReferencePoint(0, 1, p, 1), k)
    ir, ports, _, _ = layout(native, reverse=True)
    inputs, outputs = {}, {}
    cases = tuple(product(range(1 << k), ((0, 0), (0, 2), (0, 1), (2, 0))))
    norm = sum(abs(complex(i + 1, 2 - i)) ** 2 for i in range(len(cases))) ** 0.5
    for i, (address, (x, y)) in enumerate(cases):
        bits = np.array([address >> (k - j - 1) & 1 for j in range(k)])
        values = source.decompose_bloq().on_classical_vals(ctrl=bits, x=x, y=y)
        assert np.array_equal(values["ctrl"], bits)
        assert native.classical_values(address, x, y) == (
            address,
            values["x"],
            values["y"],
        )
        amp = complex(i + 1, 2 - i) / norm
        inputs[pack(dict(ctrl=address, x=x, y=y), ports, ir.n_wires)] = amp
        outputs[
            pack(dict(ctrl=address, x=values["x"], y=values["y"]), ports, ir.n_wires)
        ] = amp
    for seed in (17, 31):
        observed = execute_sparse_register(ir, inputs.copy(), seed=seed)
        assert {
            key: value for key, value in observed.items() if abs(value) > 1e-12
        } == pytest.approx(outputs, abs=1e-12)


def test_lookup_bank_cleanup_preserves_underlying_curve_flag() -> None:
    op = EllipticCurveWindowAddition(4, ECPoint(0, 1, 5, 1), 1)
    ir, ports, work, _ = layout(op)
    data = dict(ctrl=0, x=2, y=1)
    before = pack(data, ports, ir.n_wires)
    flag = work[3 * op.width]
    expected = before | (1 << (ir.n_wires - flag - 1))
    out = execute_sparse_register(ir, {before: 1j}, seed=19)
    assert {
        key: value for key, value in out.items() if abs(value) > 1e-12
    } == pytest.approx({expected: 1j}, abs=1e-12)
    assert Counter(g.name for g in ir.instructions)["measure"] == 2164


@pytest.mark.parametrize(
    "n,k,controls",
    [
        (n, k, (0, 1, 5) if n == 4 else (0, 1, 5, 8))
        for n in (4, 7, 8)
        for k in range(1, n + 1)
        if not n % k
    ],
)
def test_original_fast_and_slow_callback_matrix(
    n: int, k: int, controls: tuple[int, ...]
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as ReferencePoint

    p = 17
    encoding = QMontgomeryUInt(n, p)
    for (a, b), (x, y), ctrl in product(
        ((15, 13), (0, 0)), ((15, 13), (5, 8)), controls
    ):
        address = ctrl % (1 << k)
        bits = np.array(QUInt(k).to_bits(address))
        values = dict(
            ctrl=bits,
            x=encoding.uint_to_montgomery(x),
            y=encoding.uint_to_montgomery(y),
        )
        source = ECWindowAddR(n, ReferencePoint(a, b, p), k, k)
        expected = source.on_classical_vals(**values)
        assert np.array_equal(expected["ctrl"], bits)
        assert windowed_point_classical_values(
            n, ECPoint(a, b, p), address, values["x"], values["y"]
        ) == (address, expected["x"], expected["y"])


@pytest.mark.parametrize("a,b", [(15, 13), (0, 0)])
def test_all_original_declared_window_children(a: int, b: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.cryptography.ecc.ec_add_r import ECWindowAddR
    from qualtran.bloqs.cryptography.ecc.ec_point import ECPoint as ReferencePoint
    from qualtran.resource_counting import SympySymbolAllocator

    native = windowed_point_addition_source_graph()
    for n in (5, 6, 7):
        for k in range(1, n + 1):
            if n % k:
                continue
            source = ECWindowAddR(n, ReferencePoint(a, b, 17), k)
            names = {
                type(child).__name__: count
                for child, count in source.build_call_graph(
                    SympySymbolAllocator()
                ).items()
            }
            assert native.leaf_counts == names
            assert native.direct_calls == {
                ("ECWindowAddR", name): count for name, count in names.items()
            }
            table = windowed_point_table(n, k, ECPoint(a, b, 17))
            assert table.datasets == tuple(
                tuple(array.tolist()) for array in source.qrom.data
            )
            assert table.log_block_sizes == source.qrom.log_block_sizes
