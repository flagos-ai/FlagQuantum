"""Check the full 8-bit circuit without a dense statevector."""

from cmath import exp
from collections import defaultdict
from dataclasses import replace
from math import sqrt, tau

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectTHC
from flagquantum.compiler.registers.qrom import controlled_qrom_ir
from flagquantum.compiler.resource_estimation import count_instruction_resources
from flagquantum.core.classical_control import (
    instruction_condition_clauses,
    validate_classical_control,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.unit


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
            state = {
                basis: value for basis, value in updated.items() if abs(value) > 1e-16
            }
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
        elif name == "reset":
            (mask,) = masks
            state = {basis & ~mask: value for basis, value in state.items()}
        elif name in ("phase", "s", "sdg", "z", "cz", "t", "tdg"):
            mask = sum(masks)
            phases = {
                "s": 1j,
                "sdg": -1j,
                "z": -1,
                "cz": -1,
                "t": exp(1j * tau / 8),
                "tdg": exp(-1j * tau / 8),
            }
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
    "blocked, measurement_erasure, seed, upstream_add",
    [
        (False, False, 0, False),
        (False, True, 0, False),
        (False, True, 7, False),
        (True, False, 0, False),
        (True, True, 7, False),
        (False, True, 11, True),
        (True, True, 11, True),
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
        (16 if blocked else 11) + (7 if upstream_add else 0)
        if measurement_erasure
        else 0
    )
    resources = count_instruction_resources(ir, logical_qubits=13)
    assert resources.workspace_qubits == workspace
    assert bool(resources.conditional_counts) == measurement_erasure
    assert resources.fixed_counts.get("measure", 0) == (
        (64 if blocked else 46) + (28 if upstream_add else 0)
        if measurement_erasure
        else 0
    )
    assert resources.fixed_counts["ccx"] == (
        97 if blocked else (81 if measurement_erasure else 121)
    ) - (64 if upstream_add else 0)
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
    print("Emitted resources", resources)


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


@pytest.mark.parametrize(
    "width,words",
    [
        (3, (8, 20, 0, 0, 0, 240, 0, 0)),
        (3, (248, 236, 0, 0, 0, 16, 0, 0)),
        (2, (8, 20, 0, 0)),
        (2, (248, 236, 0, 0)),
    ],
)
def test_measured_qrom_matches_qualtran_thc_angle_tables(
    width: int, words: tuple[int, ...]
) -> None:
    qualtran = pytest.importorskip("qualtran")
    from importlib.metadata import version

    if version(qualtran.__name__) != "0.7.0":
        pytest.skip("Gate sequence comparison pins Qualtran 0.7.0")
    import cirq
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.data_loading import QROM
    from qualtran.cirq_interop import BloqAsCirqGate

    register_width = 1 + width + 8
    wires = cirq.LineQubit.range(register_width)
    context = cirq.DecompositionContext(cirq.SimpleQubitManager())
    qrom = QROM.build_from_data(np.array(words), target_bitsizes=(8,), num_controls=1)
    gate_names = {
        cirq.X: "x",
        cirq.H: "h",
        cirq.S: "s",
        cirq.T: "t",
        cirq.T**-1: "tdg",
        cirq.CNOT: "cx",
        cirq.CZ: "cz",
    }
    gate_names[BloqAsCirqGate(CNOT())] = "cx"
    upstream = cirq.decompose(
        qrom.decompose_from_registers(
            context=context,
            control=np.asarray(wires[:1]),
            selection=np.asarray(wires[1 : 1 + width]),
            target0_=np.asarray(wires[1 + width :]),
        ),
        context=context,
        keep=lambda op: (
            op.gate in gate_names
            or cirq.is_measurement(op)
            or isinstance(op.gate, cirq.ResetChannel)
            or isinstance(op, cirq.ClassicallyControlledOperation)
        ),
    )
    labels = {wire: index for index, wire in enumerate(wires)}
    bits = tuple(range(40, 40 + width))
    measured = {}
    expected = []
    for op in upstream:
        metadata = {}
        if isinstance(op, cirq.ClassicallyControlledOperation):
            (key,) = cirq.control_keys(op)
            metadata = {"conditions": ((measured[key], 1),)}
            op = op.without_classical_controls()
        operands = tuple(labels.setdefault(wire, len(labels)) for wire in op.qubits)
        if cirq.is_measurement(op):
            bit = bits[operands[0] - register_width]
            measured[cirq.measurement_key_obj(op)] = bit
            metadata = {"is_dynamic": True, "classical_bit": bit}
            name = "measure"
        elif isinstance(op.gate, cirq.ResetChannel):
            name = "reset"
            metadata = {"is_dynamic": True}
        else:
            name = gate_names[op.gate]
        expected.append((name, operands, metadata))

    ir = controlled_qrom_ir(width, 8, words, traversal_bits=bits)
    assert [
        (op.name, op.wires, dict(op.metadata)) for op in ir.instructions
    ] == expected

    # The padded addresses keep their zero words, including in superposition.
    source = {
        label << (ir.n_wires - width - 1): (1j) ** label / sqrt(2 ** (width + 1))
        for label in range(2 ** (width + 1))
    }
    expected_state = {
        basis | ((words[label % (1 << width)] if label >> width else 0) << width): value
        for label, (basis, value) in enumerate(source.items())
    }
    assert _execute_sparse(ir, source, seed=11) == pytest.approx(
        expected_state, abs=1e-12
    )
