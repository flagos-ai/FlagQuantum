"""Source coefficient rescaling, alias gates and complete measured states."""

from itertools import product
from math import sqrt

import numpy as np
import pytest
import torch

from flagquantum.algorithms.primitives.linear_combination_preparation import (
    LinearCombinationPreparation,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_alias_preparation import expected_state, layout

pytestmark = pytest.mark.unit


def source_preparation(coefficients, normalizations, bits):
    from attrs import evolve
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary

    return LinearCombination(
        tuple(evolve(Unitary(XGate()), alpha=a) for a in normalizations),
        coefficients,
        bits,
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("bits", [1, 3, 8])
@pytest.mark.parametrize("count", [2, 3, 4, 5, 8, 9])
def test_original_coefficient_and_alias_rows(count, bits):
    coefficients = tuple((-1.0) ** i * (i + 0.125) for i in range(count))
    normalizations = tuple((i % 3 + 1) / 4 for i in range(count))
    native = LinearCombinationPreparation(coefficients, normalizations, bits)
    source = source_preparation(coefficients, normalizations, bits)
    assert native.rescaled_coefficients == tuple(source.rescaled_lambd)
    prepare = source.prepare.prepare
    assert native.table.alternates == tuple(prepare.alt)
    assert native.table.keep == tuple(prepare.keep)
    assert native.table.mu == prepare.mu
    assert native.prepare.selection_width == source.prepare.selection_bitsize
    assert native.prepare.junk_width == source.prepare.junk_bitsize
    assert native.prepare.normalization == 1


@pytest.mark.parametrize("coefficients", [(1.0, -3.0), (0.0, 1.0, -2.0)])
def test_all_histories_full_complex_state_and_clean_work(coefficients):
    native = LinearCombinationPreparation(coefficients, (1.0,) * len(coefficients), 1)
    bound, sparse = layout(native.table)
    selection, sigma, alt, keep, flag, work, bits = bound
    junk = (*sigma, *alt, *keep, flag)
    prepared = expected_state(native.table, bound, sparse)
    initial = torch.zeros_like(prepared)
    initial[0] = 1
    size = initial.numel().bit_length() - 1
    for inverse, state, expected in (
        (False, initial, prepared),
        (True, prepared, initial),
    ):
        circuit = DynamicCircuit(size)
        circuit.append(
            native.make_on(
                selection=selection,
                junk=junk,
                workspace=work,
                classical_bits=bits,
                adjoint=inverse,
            )
        )
        gates = tuple(circuit.to_ir().instructions)
        measurements = sum(g.name == "measure" for g in gates)
        assert measurements <= 6
        # Work is included in every amplitude comparison, not traced out.
        for history in product((0, 1), repeat=measurements):
            torch.testing.assert_close(
                forced_branch(state, gates, history),
                expected / sqrt(2**measurements),
                atol=3e-12,
                rtol=0,
            )


@pytest.mark.ftqc_reference
def test_original_prepare_matrix_including_absolute_phase():
    native = LinearCombinationPreparation((1.0, -3.0), (1.0, 1.0), 1)
    source = source_preparation(native.coefficients, native.normalizations, 1)
    bound, sparse = layout(native.table)
    expected = expected_state(native.table, bound, sparse)
    work = len(bound[-2])
    logical = np.asarray(source.prepare.tensor_contract())[:, 0]
    torch.testing.assert_close(
        expected.reshape(-1, 1 << work)[:, 0],
        torch.as_tensor(logical),
        atol=3e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "coefficients,alphas,bits",
    [
        ((1,), (1,), 1),
        ((1, 2), (1,), 1),
        ((0, 0), (1, 1), 1),
        ((1, 2), (0, 0), 1),
        ((1, 2), (1, -1), 1),
        ((float("nan"), 1), (1, 1), 1),
        ((1, 2), (1, 1), 0),
        ((1, 2), (1, 1), True),
    ],
)
def test_invalid_parameters(coefficients, alphas, bits):
    with pytest.raises(ValueError):
        LinearCombinationPreparation(coefficients, alphas, bits)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("precision", [1, 8])
@pytest.mark.parametrize("coefficients", [(1.0, -3.0), (0.0, 1.0, -2.0)])
def test_source_gate_operands_feedback_and_model(coefficients, precision):
    from collections import Counter

    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler._component_trace.resource_model import (
        estimate_component_resources,
    )
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    native = LinearCombinationPreparation(
        coefficients, (1.0,) * len(coefficients), precision
    )
    source = source_preparation(coefficients, native.normalizations, precision).prepare
    bound, _ = layout(native.table)
    selection, sigma, alt, keep, flag, work, bits = bound
    junk = (*sigma, *alt, *keep, flag)
    for inverse in (False, True):
        gates = native.make_on(
            selection=selection,
            junk=junk,
            workspace=work,
            classical_bits=bits,
            adjoint=inverse,
        ).instructions
        measured = {
            g.wires[0]: g.metadata["classical_bit"]
            for g in gates
            if g.name == "measure"
        }
        block = source.adjoint() if inverse else source
        original = reference_arithmetic_gates(
            block.decompose_bloq(),
            {"selection": selection, "junk": junk},
            work,
            measured,
            cirq_style=True,
            preserve_components=True,
        )
        assert_source(gates, original)
    model = estimate_component_resources(native.resource_model())
    reference = get_cost_value(source, QECGatesCost())
    assert model.peak_logical_qubits == get_cost_value(source, QubitCount())
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(model.gates, category) == getattr(reference, category)
    assert model.gates.total_t_count() == reference.total_t_count()
    counts = Counter(
        g.name
        for g in native.make_on(
            selection=selection, junk=junk, workspace=work, classical_bits=bits
        ).instructions
    )
    assert counts["measure"] == model.gates.measurement == reference.measurement
    # Exact Fredkin decomposition costs seven T; the abstract default costs four.
    assert counts["t"] + counts["tdg"] == reference.total_t_count(
        ts_per_cswap=7, ts_per_rotation=0
    )


def test_model_does_not_construct_or_lower_a_circuit(monkeypatch):
    from flagquantum.algorithms.primitives import alias_preparation
    from flagquantum.qec.resources.estimation import estimate_ftqc_resources

    def forbidden(*args, **kwargs):
        raise AssertionError("Model requested numerical emission")

    monkeypatch.setattr(alias_preparation, "alias_preparation_ir", forbidden)
    result = estimate_ftqc_resources(
        LinearCombinationPreparation((1.0, -3.0), (1.0, 1.0), 8)
    )
    assert result.emitted is None and result.logical.gates.measurement > 0


def test_runnable_example(capsys):
    from examples.algorithms.linear_combination_preparation import main

    main()
    assert "full complex inverse restored" in capsys.readouterr().out


@pytest.mark.ftqc_reference
def test_original_integer_array_normalization_boundary():
    # Preserve the original NumPy in-place division dtype failure.
    original = source_preparation((1, -3), (1, 1), 1)
    native = LinearCombinationPreparation((1, -3), (1, 1), 1)
    for read in (lambda: original.rescaled_lambd, lambda: native.rescaled_coefficients):
        with pytest.raises(TypeError, match="Cannot cast"):
            read()
