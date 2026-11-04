"""Check the full 8-bit circuit without a dense statevector."""

from cmath import exp
from collections import Counter, defaultdict
from dataclasses import replace
from math import sqrt, tau

import numpy as np
import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectTHC
from flagquantum.compiler.qrom import controlled_qrom_ir
from flagquantum.core.classical_control import (
    instruction_condition_clauses,
    validate_classical_control,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def _execute_sparse(
    ir: CircuitIR, state: dict[int, complex], *, seed: int
) -> dict[int, complex]:
    """Execute the emitted gate set directly on occupied computational states."""
    rng = np.random.default_rng(seed)
    outcomes: dict[int, int] = {}
    for op in ir.instructions:
        clauses = instruction_condition_clauses(op)
        if clauses and not any(
            all(outcomes[bit] == expected for bit, expected in clause)
            for clause in clauses
        ):
            continue
        masks = tuple(1 << (ir.n_wires - q - 1) for q in op.wires)
        name = op.name
        if name == "h":
            (mask,) = masks
            updated: defaultdict[int, complex] = defaultdict(complex)
            for basis, value in state.items():
                updated[basis & ~mask] += value / sqrt(2)
                updated[basis | mask] += (-value if basis & mask else value) / sqrt(2)
            state = dict(updated)
        elif name == "measure":
            (mask,) = masks
            outcome = int(rng.integers(2))
            outcomes[op.metadata["classical_bit"]] = outcome
            state = {
                basis: value
                for basis, value in state.items()
                if bool(basis & mask) == outcome
            }
            probability = sum(abs(value) ** 2 for value in state.values())
            assert probability == pytest.approx(0.5, abs=1e-12)
            state = {basis: value / sqrt(probability) for basis, value in state.items()}
        elif name in ("phase", "s", "sdg", "z", "cz"):
            mask = sum(masks)
            phases = {"s": 1j, "sdg": -1j, "z": -1, "cz": -1}
            phase = phases[name] if name in phases else exp(1j * op.params["theta"])
            state = {
                basis: phase * value if basis & mask == mask else value
                for basis, value in state.items()
            }
        elif name in ("x", "cx", "ccx"):
            control, target = sum(masks[:-1]), masks[-1]
            state = {
                basis ^ target if basis & control == control else basis: value
                for basis, value in state.items()
            }
        elif name == "cswap":
            control, left, right = masks
            state = {
                (
                    basis ^ (left | right)
                    if basis & control and bool(basis & left) != bool(basis & right)
                    else basis
                ): value
                for basis, value in state.items()
            }
        else:
            raise AssertionError(f"Unexpected gate {name}")
    return state


@pytest.mark.parametrize(
    "blocked, measurement_erasure, seed",
    [
        (False, False, 0),
        (False, True, 0),
        (False, True, 7),
        (True, False, 0),
        (True, True, 7),
    ],
)
def test_select_thc_qrom_phase_gradient(
    blocked: bool, measurement_erasure: bool, seed: int
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
    workspace = 28 if blocked else 20
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
        (16 if blocked else 8) if measurement_erasure else 0
    )
    counts = Counter(item.name for item in ir.instructions)
    assert counts["measure"] == ((64 if blocked else 32) if measurement_erasure else 0)
    assert counts["ccx"] == (97 if blocked else (113 if measurement_erasure else 129))
    result = _execute_sparse(
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


def test_qualtran_v070_component_decompositions() -> None:
    from importlib.metadata import version
    from itertools import product

    cirq = pytest.importorskip("cirq")
    pytest.importorskip("qualtran")
    assert version("qualtran") == "0.7.0"
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.controlled_add_or_subtract import (
        ControlledAddOrSubtract,
    )
    from qualtran.bloqs.data_loading import QROM, QROAMClean
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState

    from flagquantum.compiler.phase_gradient import (
        controlled_add_or_subtract_ir,
        phase_gradient_state_ir,
    )

    words = (6, 1, 5, 0)
    for blocked in (False, True):
        options = {"log_block_sizes": (1,)} if blocked else {}
        upstream = (
            (QROAMClean if blocked else QROM)
            .build_from_data(
                np.asarray(words), num_controls=1, target_bitsizes=(3,), **options
            )
            .decompose_bloq()
        )
        ir = controlled_qrom_ir(2, 3, words, blocked=blocked)
        for enable, address in product(range(2), range(4)):
            kwargs = {"control": enable, "selection": address}
            if not blocked:
                kwargs["target0_"] = 0
            reference = upstream.on_classical_vals(**kwargs)
            result = _execute_sparse(
                ir, {(enable * 4 + address) << (ir.n_wires - 3): 1}, seed=0
            )
            word = int(reference["target0_"])
            if blocked:
                word = (word << 3) | int(reference["junk_target0_"][0])
            assert result == {
                ((enable * 4 + address) << (ir.n_wires - 3)) | (word << 2): 1
            }

    add = ControlledAddOrSubtract(QUInt(3), QUInt(3), add_when_ctrl_is_on=False)
    upstream_add = add.decompose_bloq()
    ir = CircuitIR(8, controlled_add_or_subtract_ir(0, (1, 2, 3), (4, 5, 6), 7))
    for control, a, b in product(range(2), range(8), range(8)):
        reference = upstream_add.on_classical_vals(ctrl=control, a=a, b=b)
        result = _execute_sparse(ir, {(control << 7) | (a << 4) | (b << 1): 1}, seed=0)
        assert result == {(control << 7) | (a << 4) | (int(reference["b"]) << 1): 1}

    wires = tuple(cirq.LineQubit.range(8))
    phase_state = PhaseGradientState(8, exponent=1).on_registers(phase_grad=wires)
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(cirq.Circuit(cirq.decompose(phase_state)), qubit_order=wires)
        .final_state_vector
    )
    ir = CircuitIR(8, phase_gradient_state_ir(tuple(range(8))))
    result = _execute_sparse(ir, {0: 1}, seed=0)
    np.testing.assert_allclose(
        [result[i] for i in range(256)], expected, atol=1e-12, rtol=0
    )
