"""Raw triangular arithmetic, source protocol boundaries and native XOR consumers."""

from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_contiguous_index import (
    contiguous_index_classical_values,
)
from tests.unit.test_classical_comparison import compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PAIRS = [
    (0, 0),
    (0, 1),
    (1, 1),
    (5, 2),
    (-1, 2),
    (2, -1),
    (-3, -4),
    (2**80, 2**70),
    (np.int8(2), np.int8(100)),
    (np.uint8(1), np.uint8(255)),
    (np.array([0, 1]), np.array([1, 2])),
    (np.array([[0], [2]]), np.array([1, 2])),
    (sympy.Integer(2), sympy.Integer(3)),
    (sympy.Symbol("m"), sympy.Symbol("n")),
    (1.5, 2.5),
    (np.bool_(True), np.bool_(True)),
]


@pytest.mark.parametrize(
    "width,result_width",
    [
        (1, 1),
        (3, 6),
        (0, 0),
        (-1, -2),
        (np.int64(3), np.int64(6)),
        (sympy.Symbol("n"), sympy.Symbol("s")),
    ],
)
@pytest.mark.parametrize("mu,nu", PAIRS)
def test_source_raw_values_types_overflow_and_unused_width_metadata(
    width, result_width, mu, nu
):
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import ToContiguousIndex

    original = ToContiguousIndex(width, result_width)
    compare_calls(
        lambda m, n: tuple(original.on_classical_vals(m, n).values()),
        contiguous_index_classical_values,
        (mu, nu),
    )


def test_raw_operator_sequence_and_operand_identity_match_original_without_integer_casts():
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import ToContiguousIndex

    class Number:
        def __init__(self, label, value, trace):
            self.label, self.value, self.trace = label, value, trace

        def operation(self, other, symbol):
            label = other.label if isinstance(other, Number) else str(other)
            value = other.value if isinstance(other, Number) else other
            self.trace.append((symbol, self.label, label))
            result = {
                "+": lambda: self.value + value,
                "*": lambda: self.value * value,
                "//": lambda: self.value // value,
            }[symbol]()
            return Number(f"({self.label}{symbol}{label})", result, self.trace)

        def __add__(self, other):
            return self.operation(other, "+")

        def __mul__(self, other):
            return self.operation(other, "*")

        def __floordiv__(self, other):
            return self.operation(other, "//")

        def __int__(self):
            pytest.fail("The source raw expression does not normalize operands")

    traces = [[], []]
    outcomes = []
    for native, trace in zip((False, True), traces, strict=True):
        mu, nu = Number("mu", 2, trace), Number("nu", 3, trace)
        result = (
            contiguous_index_classical_values(mu, nu)
            if native
            else tuple(ToContiguousIndex(3, 6).on_classical_vals(mu, nu).values())
        )
        assert result[0] is mu and result[1] is nu
        outcomes.append(result[2].value)
    assert outcomes == [8, 8]
    assert (
        traces[0]
        == traces[1]
        == [
            ("+", "nu", "1"),
            ("*", "nu", "(nu+1)"),
            ("//", "(nu*(nu+1))", "2"),
            ("+", "((nu*(nu+1))//2)", "mu"),
        ]
    )


def test_source_missing_decomposition_and_through_signature_error_remain_explicit():
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import ToContiguousIndex

    original = ToContiguousIndex(2, 3)
    with pytest.raises(DecomposeNotImplementedError):
        original.decompose_bloq()
    with pytest.raises(TypeError, match="unexpected keyword argument 's'"):
        original.call_classically(mu=0, nu=0, s=0)
    assert (
        original.on_classical_vals(0, 2)["s"]
        == contiguous_index_classical_values(0, 2)[2]
        == 3
    )


@pytest.mark.parametrize("columns", [False, True])
def test_occupied_target_all_complex_states_all_measurements_and_work_cleanup(columns):
    import torch

    from flagquantum.compiler.registers.sorting.contiguous_index import (
        contiguous_index_columns_ir,
        contiguous_index_columns_workspace_size,
        contiguous_index_ir,
        contiguous_index_requirements,
    )
    from tests.unit.measurement_checks import forced_branch

    q = (
        contiguous_index_columns_workspace_size(1)
        if columns
        else contiguous_index_requirements(1)[0]
    )
    work = tuple(range(4, 4 + q))
    gates = (
        contiguous_index_columns_ir((0,), (1,), (2, 3), work)
        if columns
        else contiguous_index_ir((0,), (1,), (2, 3), work, tuple(range(40, 41)))
    )
    total = 4 + q + 1
    grid = torch.arange(1, 33, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for data, spectator in product(range(16), (0, 1)):
        mu, nu, target = data >> 3, (data >> 2) & 1, data & 3
        raw = contiguous_index_classical_values(mu, nu)[2]
        updated = (data & ~3) | (target ^ (raw & 3))
        initial[(data << (q + 1)) | spectator] = amplitudes[2 * data + spectator]
        expected[(updated << (q + 1)) | spectator] = amplitudes[2 * data + spectator]
    measurements = sum(g.name == "measure" for g in gates)
    for history in product((0, 1), repeat=measurements):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            actual, expected / 2 ** (measurements / 2), rtol=0, atol=2e-12
        )
        residual = actual.reshape(16, 1 << q, 2)[:, 1:, :]
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), rtol=0, atol=2e-12
        )


def test_source_budget_and_emitted_report_keep_measurement_and_width_bases_separate():
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import ToContiguousIndex
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.sorting.contiguous_index import (
        contiguous_index_columns_ir,
        source_contiguous_index_cost,
    )
    from flagquantum.compiler.resource_reporting import resource_model_report
    from flagquantum.core.ir import CircuitIR

    original = ToContiguousIndex(1, 2)
    model = source_contiguous_index_cost(1, 2)
    ir = CircuitIR(6, contiguous_index_columns_ir((0,), (1,), (2, 3), (4,)))
    report = resource_model_report(model, emitted_ir=ir)
    source_gates = get_cost_value(original, QECGatesCost())
    assert report["logical_gates"]["toffoli"] == source_gates.toffoli == 1
    assert report["t_equivalents"] == source_gates.total_t_count() == 4
    assert report["peak_logical_qubits"] == get_cost_value(original, QubitCount()) == 4
    assert report["logical_gates"]["measurement"] == 0
    assert report["emitted_ir"]["t_count"] == 4
    assert report["emitted_ir"]["unconditional"]["measure"] == 1
    assert report["emitted_ir"]["declared_qubits"] == 6
    assert report["execution_measurements"] is None
