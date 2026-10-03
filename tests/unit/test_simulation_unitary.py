from __future__ import annotations

import math
import time

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from flagquantum.simulation.unitary import MAX_UNITARY_WIRES, get_unitary

pytestmark = pytest.mark.unit

_ATOL = 1e-6
# complex64 round-off for a kernel that composes at most a few gates.


def _cnot(dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=dtype
    )


_SINGLE_WIRE_REFERENCES = {
    "x": lambda: torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
    "h": lambda: torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2),
    "y": lambda: torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
    "z": lambda: torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64),
    "s": lambda: torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64),
    "t": lambda: torch.tensor(
        [[1, 0], [0, (1 + 1j) / math.sqrt(2)]], dtype=torch.complex64
    ),
}


@pytest.mark.parametrize("gate", sorted(_SINGLE_WIRE_REFERENCES))
def test_single_wire_fixed_gates_match_their_matrices(gate: str) -> None:
    unitary = get_unitary(getattr(fq.Circuit(1), gate)(0))

    assert torch.allclose(unitary, _SINGLE_WIRE_REFERENCES[gate](), atol=_ATOL)


@pytest.mark.parametrize("gate", ("rx", "ry", "rz"))
@pytest.mark.parametrize("theta", (0.0, 0.7, -2.5, math.pi))
def test_single_wire_rotations_match_their_matrices(gate: str, theta: float) -> None:
    """The rotation angle reaches the matrix unchanged, sign included."""

    half = theta / 2
    cosine, sine = math.cos(half), math.sin(half)
    reference = {
        "rx": torch.tensor(
            [[cosine, -1j * sine], [-1j * sine, cosine]], dtype=torch.complex64
        ),
        "ry": torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.complex64),
        "rz": torch.tensor(
            [
                [complex(cosine, -sine), 0],
                [0, complex(cosine, sine)],
            ],
            dtype=torch.complex64,
        ),
    }[gate]

    unitary = get_unitary(getattr(fq.Circuit(1), gate)(0, theta))

    assert torch.allclose(unitary, reference, atol=_ATOL)


def test_the_two_wire_bell_circuit_matches_h_tensor_identity_then_cnot() -> None:
    identity = torch.eye(2, dtype=torch.complex64)
    hadamard = _SINGLE_WIRE_REFERENCES["h"]()

    unitary = get_unitary(fq.Circuit(2).h(0).cx(0, 1))

    assert torch.allclose(unitary, _cnot() @ torch.kron(hadamard, identity), atol=_ATOL)


def test_wire_zero_is_the_most_significant_amplitude_bit() -> None:
    """A low-bit-first ordering would put the phase on basis 1 instead of basis 2."""

    unitary = get_unitary(fq.Circuit(2).rz(0, 0.7))
    phase = complex(math.cos(0.35), -math.sin(0.35))

    diagonal = unitary.diagonal()
    assert torch.allclose(diagonal[:2], torch.full((2,), phase), atol=_ATOL)
    assert torch.allclose(diagonal[2:], torch.full((2,), phase.conjugate()), atol=_ATOL)


def test_control_and_target_wires_are_not_interchangeable() -> None:
    """``cx(0, 1)`` and ``cx(1, 0)`` differ, which a transposed result would hide."""

    cnot = _cnot()
    identities = torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=torch.complex64,
    )

    forward = get_unitary(fq.Circuit(2).cx(0, 1))
    reverse = get_unitary(fq.Circuit(2).cx(1, 0))

    assert not torch.allclose(forward, reverse, atol=_ATOL)
    assert torch.allclose(forward, cnot, atol=_ATOL)
    assert torch.allclose(reverse, identities @ cnot @ identities, atol=_ATOL)


def test_a_reversed_control_flips_only_the_state_where_its_own_control_is_set() -> None:
    """Witness the semantics directly, since the matrix alone is easy to misread."""

    reverse = get_unitary(fq.Circuit(2).cx(1, 0), dtype=torch.complex128)
    expected_target = {0: 0, 1: 3, 2: 2, 3: 1}

    for basis, target in expected_target.items():
        column = reverse[:, basis]
        assert int(column.abs().argmax()) == target
        assert torch.isclose(column.abs().sum(), torch.tensor(1.0, dtype=torch.float64))


def test_composition_multiplies_in_program_order() -> None:
    """U(c2 after c1) is U(c2) @ U(c1), so a reversed order would fail here."""

    first = fq.Circuit(2).h(0).t(1).cx(0, 1)
    second = fq.Circuit(2).rz(1, 0.4).cx(1, 0).h(0)
    combined = fq.Circuit(2).h(0).t(1).cx(0, 1).rz(1, 0.4).cx(1, 0).h(0)

    assert torch.allclose(
        get_unitary(combined),
        get_unitary(second) @ get_unitary(first),
        atol=_ATOL,
    )


@pytest.mark.parametrize("n_wires", (1, 2, 3, 4))
def test_the_result_is_unitary(n_wires: int) -> None:
    circuit = fq.Circuit(n_wires)
    for wire in range(n_wires - 1):
        circuit.cx(wire, wire + 1)
    circuit.h(0).rz(n_wires - 1, 0.9).t(n_wires - 1)

    unitary = get_unitary(circuit, dtype=torch.complex128)
    dimension = 1 << n_wires

    assert torch.allclose(
        unitary.conj().T @ unitary,
        torch.eye(dimension, dtype=torch.complex128),
        atol=1e-12,
    )


def test_the_columns_are_the_program_applied_to_the_basis_states() -> None:
    """The matrix must agree with ``Circuit.state`` rather than with a second kernel."""

    circuit = fq.Circuit(3).h(0).cx(0, 1).rz(2, 0.31).ccx(0, 1, 2)
    unitary = get_unitary(circuit, dtype=torch.complex128)

    for index in range(8):
        basis = torch.zeros(8, dtype=torch.complex128)
        basis[index] = 1
        source = fq.Circuit.from_ir(
            circuit.to_ir(), dtype=torch.complex128, inputs=basis
        )
        assert torch.allclose(
            unitary[:, index], source.state()[0], atol=1e-12
        ), f"column {index}"


def test_an_arbitrary_matrix_gate_is_materialized() -> None:
    """A custom unitary is a gate like any other, so it belongs in the matrix."""

    square_root_x = torch.tensor(
        [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=torch.complex64
    )
    circuit = fq.Circuit(1).any(0, unitary=square_root_x)

    assert torch.allclose(get_unitary(circuit), square_root_x, atol=_ATOL)


def test_a_canonical_ir_is_accepted_like_a_circuit() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    assert torch.allclose(get_unitary(circuit), get_unitary(circuit.to_ir()), atol=0)


def test_the_default_dtype_is_the_programs_own() -> None:
    assert get_unitary(fq.Circuit(1).h(0)).dtype is torch.complex64
    assert (
        get_unitary(fq.Circuit(1, dtype=torch.complex128).h(0)).dtype
        is torch.complex128
    )


def test_an_explicit_dtype_overrides_the_program() -> None:
    """A double-precision request must not be rounded through the single-precision path."""

    single = get_unitary(fq.Circuit(1).rz(0, 0.7))
    double = get_unitary(fq.Circuit(1).rz(0, 0.7), dtype=torch.complex128)
    reference = get_unitary(
        fq.Circuit(1, dtype=torch.complex128).rz(0, 0.7), dtype=torch.complex128
    )

    assert single.dtype is torch.complex64
    assert double.dtype is torch.complex128
    assert torch.allclose(double, reference, atol=1e-15)


def test_a_real_dtype_is_refused() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_unitary(fq.Circuit(1).h(0), dtype=torch.float32)

    assert "complex" in str(excinfo.value)


def test_the_device_argument_reaches_the_kernel() -> None:
    unitary = get_unitary(fq.Circuit(1).h(0), device="cpu")

    assert unitary.device.type == "cpu"


def test_the_result_is_on_the_requested_device() -> None:
    """A device argument that only reached the tensor constructor would show here."""

    unitary = get_unitary(fq.Circuit(2).h(0), device=torch.device("cpu"))

    assert unitary.device == torch.device("cpu")
    assert unitary.shape == (4, 4)


@pytest.mark.parametrize("n_wires", (MAX_UNITARY_WIRES + 1, MAX_UNITARY_WIRES + 4))
def test_a_program_over_the_ceiling_is_refused_before_anything_is_allocated(
    n_wires: int,
) -> None:
    """The refusal must cost no allocation, so a wide program cannot OOM first."""

    circuit = fq.Circuit(n_wires).h(0)
    started = time.perf_counter()

    with pytest.raises(ValueError) as excinfo:
        get_unitary(circuit)

    elapsed = time.perf_counter() - started
    assert str(MAX_UNITARY_WIRES) in str(excinfo.value)
    assert str(n_wires) in str(excinfo.value)
    assert elapsed < 1.0


def test_exactly_the_ceiling_is_accepted() -> None:
    """The limit is a ceiling, not an off-by-one: one wire below it must work."""

    assert get_unitary(fq.Circuit(MAX_UNITARY_WIRES).h(0), dtype=torch.complex64).shape[
        0
    ] == (1 << MAX_UNITARY_WIRES)


@pytest.mark.parametrize("kind", ("channel", "dynamic", "condition"))
def test_a_non_unitary_instruction_is_refused_by_index_and_opcode(kind: str) -> None:
    metadata = {
        "channel": {"is_channel": True},
        "dynamic": {"is_dynamic": True},
        "condition": {"condition_clauses": ({"wire": 0, "value": 1},)},
    }[kind]
    ir = CircuitIR(
        n_wires=2,
        instructions=(
            Instruction(name="h", wires=(0,)),
            Instruction(name="x", wires=(1,), metadata=metadata),
        ),
        dtype="complex64",
    )

    with pytest.raises(ValueError) as excinfo:
        get_unitary(ir)

    message = str(excinfo.value)
    assert "instruction 1" in message
    assert "'x'" in message


def test_a_channel_named_instruction_is_refused_by_index_and_opcode() -> None:
    ir = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                name="bit_flip",
                wires=(0,),
                params={"probability": 0.1},
                metadata={"is_channel": True},
            ),
        ),
        dtype="complex64",
    )

    with pytest.raises(ValueError) as excinfo:
        get_unitary(ir)

    assert "bit_flip" in str(excinfo.value)


def test_a_reset_instruction_is_refused() -> None:
    ir = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(name="reset", wires=(0,), metadata={"is_dynamic": True}),
        ),
        dtype="complex64",
    )

    with pytest.raises(ValueError) as excinfo:
        get_unitary(ir)

    assert "reset" in str(excinfo.value)


def test_a_measurement_request_is_refused() -> None:
    """Reading qubits discards coherence, so the program is no longer a unitary."""

    ir = CircuitIR(
        n_wires=2,
        instructions=(Instruction(name="h", wires=(0,)),),
        dtype="complex64",
        measurements=(MeasurementNode(kind="sample", wires=(0,), shots=64),),
    )

    with pytest.raises(ValueError) as excinfo:
        get_unitary(ir)

    assert "measurement" in str(excinfo.value)


def test_an_observable_request_is_refused() -> None:
    ir = CircuitIR(
        n_wires=2,
        instructions=(Instruction(name="h", wires=(0,)),),
        dtype="complex64",
        observables=(ObservableNode(name="z", wires=(0,)),),
    )

    with pytest.raises(ValueError) as excinfo:
        get_unitary(ir)

    assert "observable" in str(excinfo.value)


def test_an_unbound_parameter_is_refused_with_its_name() -> None:
    circuit = fq.Circuit(2).rx(0, fq.Parameter("theta"))

    with pytest.raises(ValueError) as excinfo:
        get_unitary(circuit)

    assert "theta" in str(excinfo.value)
    assert "bind_parameters" in str(excinfo.value)


def test_binding_the_parameter_produces_the_bound_matrix() -> None:
    unbound = fq.Circuit(2).rx(0, fq.Parameter("theta"))
    bound = unbound.bind_parameters({"theta": 0.7})

    assert torch.allclose(
        get_unitary(bound), get_unitary(fq.Circuit(2).rx(0, 0.7)), atol=_ATOL
    )


def test_an_empty_program_is_the_identity() -> None:
    assert torch.allclose(
        get_unitary(fq.Circuit(3)), torch.eye(8, dtype=torch.complex64), atol=0
    )


def test_the_ceiling_is_exported_so_a_caller_can_check_before_allocating() -> None:
    import flagquantum.simulation as simulation

    assert simulation.MAX_UNITARY_WIRES == MAX_UNITARY_WIRES
    assert MAX_UNITARY_WIRES == 12
    assert simulation.get_unitary is get_unitary
