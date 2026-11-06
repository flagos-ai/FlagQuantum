"""Signed THC alias signal, branch cleanup, and original resource accounting."""

from dataclasses import asdict
from itertools import product
from math import sqrt

import numpy as np
import pytest

from flagquantum.algorithms.primitives.thc_chemistry_data import thc_chemistry_data
from flagquantum.algorithms.primitives.thc_chemistry_prepare import PrepareTHCChemistry
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.partition import partition_register
from flagquantum.dynamic import DynamicCircuit
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


def _tensors(num_mu):
    zeta = np.array([[1.0, -2.0, 2.0], [-2.0, 4.0, 1.0], [2.0, 1.0, -3.0]])
    return np.array([6.0, -8.0]), np.ones((num_mu, 2)), zeta[:num_mu, :num_mu]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m,bits,log", [(2, 1, 0), (2, 1, 1), (4, 4, 1), (6, 2, 2)])
def test_source_prepare_cost_and_unchanged_original_body_flag(m, bits, log):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.thc.prepare import PrepareTHC, UniformSuperpositionTHC
    from qualtran.bloqs.mcmt import MultiControlX
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    if m == 2:
        tensors = _tensors(m)
    else:
        rng = np.random.default_rng(29)
        raw = rng.normal(size=(m, m))
        tensors = rng.normal(size=2), rng.normal(size=(m, 2)), (raw + raw.T) / 2
    data = thc_chemistry_data(*tensors, probability_bits=bits, log_block_size=log)
    native = PrepareTHCChemistry(data)
    source = PrepareTHC.from_hamiltonian_coeffs(
        *tensors, num_bits_state_prep=bits, log_block_size=log
    )
    assert {k: v for k, v in asdict(native.source_gate_cost()).items() if v} == (
        get_cost_value(source, QECGatesCost()).asdict()
    )
    if log == 0:
        with pytest.raises(RuntimeError, match="junk_target0_"):
            get_cost_value(source, QubitCount())
        with pytest.raises(ValueError, match="junk_target0_"):
            native.source_resource_model()
    else:
        assert (*native.selection_layout, *native.junk_layout) == tuple(
            (reg.name, reg.total_bits()) for reg in source.signature
        )
        graph = source.decompose_bloq()
        (controls,) = [
            inputs["controls"]
            for child, inputs, _ in graph.iter_bloqsoqs()
            if isinstance(child.bloq, MultiControlX)
        ]
        assert isinstance(controls[0].binst.bloq, UniformSuperpositionTHC)
        assert graph.final_soqs()["nu_eq_mp1"].binst.bloq.cvs == (0, 1)
        assert_source_model(native.source_resource_model(), source)


def _loaded_words(data, address, exchange_matrices):
    columns = data.address_datasets
    if not data.log_block_size:
        return {
            name: column[address]
            for (name, _), column in zip(data.target_layout, columns, strict=True)
        }, 1 + 0j

    # These fixtures use two cells per block. Read the complete complex
    # exchange column from the unchanged source leaf for each bank.
    values = {}
    phase = 1 + 0j
    cell = address & 1
    start = address - cell
    for (name, width), column in zip(data.target_layout, columns, strict=True):
        left = column[start]
        right = column[start + 1] if start + 1 < len(column) else 0
        initial = (cell << (2 * width)) | (left << width) | right
        vector = exchange_matrices[width][:, initial]
        destination = int(np.argmax(np.abs(vector)))
        assert abs(vector[destination]) == pytest.approx(1)
        phase *= complex(vector[destination])
        mask = (1 << width) - 1
        values[name] = (destination >> width) & mask
        values[f"junk_{name}"] = destination & mask
    return values, phase


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m,log", [(1, 0), (2, 0), (2, 1), (3, 1)])
def test_complete_thc_prepare_signed_amplitudes_flags_and_inverse(m, log):
    pytest.importorskip("qualtran")
    cirq = pytest.importorskip("cirq")
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox

    data = thc_chemistry_data(*_tensors(m), probability_bits=1, log_block_size=log)
    prepare = PrepareTHCChemistry(data)
    layout = (*prepare.selection_layout, *prepare.junk_layout)
    logical = sum(width for _, width in layout)
    work, slots = prepare.requirements()
    width = logical + work + 1
    ports = partition_register(tuple(range(logical)), layout)
    scratch = tuple(range(logical, logical + work))
    bits = tuple(range(30, 30 + slots))
    circuit = DynamicCircuit(width).append(
        prepare.make_on(ports, workspace=scratch, classical_bits=bits)
    )
    inverse = DynamicCircuit(width).append(
        prepare.make_on(ports, workspace=scratch, classical_bits=bits, adjoint=True)
    )
    ir, reverse_ir = circuit.to_ir(), inverse.to_ir()
    counts = count_emitted_instructions(ir)
    events = counts.unconditional["measure"]
    assert events == count_emitted_instructions(reverse_ir).unconditional["measure"]
    assert counts.t_count > 0

    matrices = {}
    if log:
        for _, word_width in data.target_layout:
            wires = cirq.LineQubit.range(1 + 2 * word_width)
            matrices[word_width] = cirq.unitary(
                cirq.Circuit(cirq.decompose(CSwapApprox(word_width).on(*wires)))
            )

    initial = {0: (1 + 1j) / 2, 1: (1 - 1j) / 2}
    scale = sqrt(len(data.source_rows) * 2**data.probability_bits * 8)
    expected = {}
    flag_changes = 0
    for (mu, nu), sigma, a, b, mn in product(
        data.source_rows, range(2**data.probability_bits), (0, 1), (0, 1), (0, 1)
    ):
        address = nu * (nu + 1) // 2 + mu
        values, phase = _loaded_words(data, address, matrices)
        choose_alt = values["keep"] <= sigma
        sign = values["alt_theta"] if choose_alt else values["theta"]
        original_body = nu == m
        if choose_alt:
            mu, values["alt_mu"] = values["alt_mu"], mu
            nu, values["alt_nu"] = values["alt_nu"], nu
        body = nu == m
        flag_changes += int(body != original_body)
        swap = int(a and not body)
        if swap:
            mu, nu = nu, mu
        values.update(
            mu=mu,
            nu=nu,
            sigma=sigma,
            plus_a=a,
            plus_b=b,
            plus_mn=mn,
            nu_eq_mp1=int(body),
            extra_ctrl=swap,
            succ=1,
            s=address,
        )
        word = 0
        for name, size in layout:
            word = (word << size) | values.get(name, 0)
        for spectator, coefficient in initial.items():
            expected[(word << (work + 1)) | spectator] = (
                phase * (-1) ** sign * coefficient / scale
            )
    assert flag_changes > 0
    assert sum(abs(value) ** 2 for value in expected.values()) == pytest.approx(1)
    for pattern in ((0,), (1,), (0, 1), (1, 0)):
        history = (pattern * events)[:events]
        actual = _branch(
            ir.instructions, width, history, initial, normalize_measurements=True
        )
        assert actual == pytest.approx(expected, abs=2e-12)
        restored = _branch(
            reverse_ir.instructions,
            width,
            history,
            expected,
            normalize_measurements=True,
        )
        assert restored == pytest.approx(initial, abs=2e-12)
