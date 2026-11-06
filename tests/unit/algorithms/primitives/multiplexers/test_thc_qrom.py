"""Check the full 8-bit circuit without a dense statevector."""

from collections import Counter
from dataclasses import replace
from math import sqrt, tau

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectTHC
from flagquantum.compiler.registers.qrom import controlled_qrom_ir
from flagquantum.core.classical_control import (
    validate_classical_control,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.sparse_register_execution import execute_sparse_register

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "blocked, measurement_erasure, seed, upstream_add",
    [
        (False, False, 0, False),
        (False, True, 0, False),
        (False, True, 7, False),
        (True, False, 0, False),
        (True, True, 7, False),
        (False, True, 11, True),
    ],
)
def test_select_thc_qrom_phase_gradient(
    blocked: bool, measurement_erasure: bool, seed: int, upstream_add: bool
) -> None:
    op = SelectTHC.make_on(
        mu=(0, 1),
        nu=(2, 3),
        succ=4,
        nu_eq_mp1=5,
        plus_mn=6,
        plus_a=7,
        plus_b=8,
        sys_a=(9, 10),
        sys_b=(11, 12),
        networks=(((0, 1, 0.2),), ((0, 1, 0.5),)),
        one_body_networks=((), ((0, 1, -0.4),)),
        workspace=tuple(range(13, 33)),
    )
    # Success, failure, one-body, both spins, and a padded address in superposition.
    inputs = np.zeros(8192, dtype=np.complex128)
    for index, word in enumerate(
        (
            "0001100000101",
            "0100100100110",
            "0100110010110",
            "1100100000110",
            "0100000100101",
            "0001101011001",
        )
    ):
        inputs[int(word, 2)] = (1j) ** index / sqrt(6)

    def quantized(table):
        step = tau / 256
        return tuple(
            tuple(
                (left, right, round(angle / step) * step) for left, right, angle in net
            )
            for net in table
        )

    exact = replace(op, workspace=())
    rounded = replace(
        exact,
        networks=quantized(op.networks),
        one_body_networks=quantized(op.one_body_networks),
    )
    workspace = (28 if blocked else 20) + (6 if upstream_add else 0)
    builder = DynamicCircuit if measurement_erasure else fq.Circuit
    ir = (
        builder(13 + workspace, dtype=torch.complex128)
        .append(
            replace(
                op,
                workspace=tuple(range(13, 13 + workspace)),
                measurement_erasure=measurement_erasure,
            )
        )
        .to_ir()
    )
    assert validate_classical_control(ir.instructions, n_qubits=ir.n_wires) == (
        (16 if blocked else 8) + (7 if upstream_add else 0)
        if measurement_erasure
        else 0
    )
    counts = Counter(item.name for item in ir.instructions)
    assert counts["measure"] == (
        (64 if blocked else 32) + (28 if upstream_add else 0)
        if measurement_erasure
        else 0
    )
    assert counts["ccx"] == (
        97 if blocked else (113 if measurement_erasure else 129)
    ) - (64 if upstream_add else 0)
    result = execute_sparse_register(
        ir,
        {int(i) << workspace: complex(inputs[i]) for i in np.flatnonzero(inputs)},
        seed=seed,
    )
    logical = np.zeros_like(inputs)
    leakage = 0.0
    for basis, value in result.items():
        if basis & ((1 << workspace) - 1):
            leakage += abs(value) ** 2
        else:
            logical[basis >> workspace] = value
    expected, unrounded = (
        fq.Circuit(13, dtype=torch.complex128, inputs=torch.from_numpy(inputs))
        .append(reference)
        .state()
        .detach()
        .numpy()
        .ravel()
        for reference in (rounded, exact)
    )
    np.testing.assert_allclose(logical, expected, rtol=0, atol=1e-12)
    assert leakage < 1e-24
    # At most two reflections, each containing a rotation and its adjoint.
    assert np.linalg.norm(logical - unrounded) <= 4 * np.pi / 256
    print("8-bit amplitude error", np.max(np.abs(logical - expected)))
    print("Unquantized amplitude error", np.max(np.abs(logical - unrounded)))
    print("Workspace leakage probability", leakage)
    print("Emitted gates", dict(Counter(op.name for op in ir.instructions)))


def test_paired_qrom_erasure_on_native_dynamic_runtime() -> None:
    # Three words give a partial final block; the unused address loads zero.
    words = (0b110, 0b001, 0b101)
    circuit = DynamicCircuit(8, dtype=torch.complex128)
    circuit.h(0).h(1).h(2).s(1)
    expected = circuit.state().flatten()
    circuit.compose(controlled_qrom_ir(2, 3, words))
    circuit.compose(controlled_qrom_ir(2, 3, words, measurement_bits=(0, 1, 2)))
    result = run_dynamic(circuit, shots=16, seed=21, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(16, -1),
        expected.expand(16, -1),
        rtol=0,
        atol=1e-12,
    )


def test_controlled_adder_preserves_complex_superposition() -> None:
    from flagquantum.compiler.registers.arithmetic import (
        measurement_add_or_subtract_ir,
    )

    # A dense, complex superposition exercises every 3-bit operand and both
    # control values, including preservation of relative phase across outcomes.
    rng = np.random.default_rng(70)
    state = rng.normal(size=128) + 1j * rng.normal(size=128)
    state /= np.linalg.norm(state)
    inputs = np.zeros(512, dtype=np.complex128)
    inputs[::4] = state
    expected_state = np.zeros_like(inputs)
    for index, value in enumerate(state):
        control, a, b = index >> 6, (index >> 3) & 7, index & 7
        destination = (control << 6) | (a << 3) | ((b + (-a if control else a)) & 7)
        expected_state[destination << 2] = value
    adder = measurement_add_or_subtract_ir(0, (1, 2, 3), (4, 5, 6), (7, 8), (0, 1))
    circuit = DynamicCircuit.from_ir(
        CircuitIR(9, adder), dtype=torch.complex128, inputs=torch.from_numpy(inputs)
    )
    result = run_dynamic(circuit, shots=8, seed=11, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1),
        torch.from_numpy(expected_state).expand(8, -1),
        atol=1e-12,
        rtol=0,
    )
