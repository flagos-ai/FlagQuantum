"""Differential reference for the CPU statevector kernels.

Every case applies one gate program twice. Once through the shipped engine,
``Circuit.state()``, which routes the program through fusion, the batched
rotation kernels, the diagonal kernel and the permutation kernels. Once through
the naive reference below, which walks basis amplitudes with Python integer bit
arithmetic and knows nothing about the engine. The two share no code, so
agreement is evidence about the kernels rather than about a shared helper.

Comparison strength is declared per case instead of assumed globally. A program
built only from permutations is compared with ``torch.equal``: an integer
remapping of amplitudes introduces no floating-point error, so any difference is
a defect. Only a program carrying analytic rotations is allowed a tolerance, and
that tolerance is derived from the dtype and the number of rotation
applications rather than chosen to make the current implementation pass.

These cases also assert which engine path each program took. A differential net
that still passes after the path it was written to cover silently stopped
engaging would report agreement about nothing, so the routing is pinned next to
the agreement.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit

SQRT_HALF = 1.0 / math.sqrt(2.0)

EXACT = "exact"
WITHIN_ACCUMULATED_ERROR = "within_accumulated_error"

# One dtype epsilon, by the number of rotation applications, times a margin that
# covers a different summation order. A kernel that drops a term or reads the
# wrong wire is off by orders of magnitude more than this, so the margin does not
# blunt the test.
_DTYPE_EPSILON = {torch.complex64: 1.2e-7, torch.complex128: 2.3e-16}
_TOLERANCE_MARGIN = 32


@dataclass(frozen=True)
class Operation:
    """One gate application, in the form both references consume."""

    name: str
    wires: tuple[int, ...]
    params: tuple[float, ...] = ()


_PARAMETER_NAMES: dict[str, tuple[str, ...]] = {
    "p": ("theta",),
    "u1": ("theta",),
    "u2": ("phi", "lbd"),
    "u3": ("theta", "phi", "lbd"),
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
}

_ROTATION_GATES = frozenset(_PARAMETER_NAMES)
_PERMUTATION_GATES = frozenset({"x", "y", "z", "cx", "swap"})

_SINGLE_QUBIT_MATRIX_GATES = (
    "h",
    "x",
    "y",
    "z",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "p",
    "rx",
    "ry",
    "rz",
    "u1",
    "u2",
    "u3",
)

_GATE_PARAMETERS: dict[str, tuple[float, ...]] = {
    "p": (0.7,),
    "u1": (0.7,),
    "u2": (0.4, 0.9),
    "u3": (0.6, 0.4, 0.9),
    "rx": (0.9,),
    "ry": (-1.3,),
    "rz": (2.1,),
}


def single_qubit_gate_names() -> tuple[str, ...]:
    return _SINGLE_QUBIT_MATRIX_GATES


def operation(
    name: str, *wires: int, params: tuple[float, ...] | None = None
) -> Operation:
    """Build an operation, defaulting to this file's fixed gate parameters."""
    if params is None:
        params = _GATE_PARAMETERS.get(name, ())
    return Operation(name=name, wires=tuple(wires), params=tuple(params))


def _single_qubit_matrix(name: str, params: Sequence[float]) -> torch.Tensor:
    """The gate matrix, written from its definition rather than imported."""
    if name in {"p", "u1"}:
        (theta,) = params
        phase = cmath.exp(1j * theta)
        return torch.tensor([[1, 0], [0, phase]], dtype=torch.complex128)
    if name == "u2":
        phi, lbd = params
        return _u3_matrix(math.pi / 2, phi, lbd)
    if name == "u3":
        return _u3_matrix(*params)
    if name == "rx":
        (theta,) = params
        cosine, sine = math.cos(theta / 2), math.sin(theta / 2)
        return torch.tensor(
            [[cosine, -1j * sine], [-1j * sine, cosine]], dtype=torch.complex128
        )
    if name == "ry":
        (theta,) = params
        cosine, sine = math.cos(theta / 2), math.sin(theta / 2)
        return torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.complex128)
    if name == "rz":
        (theta,) = params
        return torch.tensor(
            [
                [cmath.exp(-1j * theta / 2), 0],
                [0, cmath.exp(1j * theta / 2)],
            ],
            dtype=torch.complex128,
        )
    fixed = {
        "h": [[SQRT_HALF, SQRT_HALF], [SQRT_HALF, -SQRT_HALF]],
        "x": [[0, 1], [1, 0]],
        "y": [[0, -1j], [1j, 0]],
        "z": [[1, 0], [0, -1]],
        "s": [[1, 0], [0, 1j]],
        "sdg": [[1, 0], [0, -1j]],
        "t": [[1, 0], [0, cmath.exp(1j * math.pi / 4)]],
        "tdg": [[1, 0], [0, cmath.exp(-1j * math.pi / 4)]],
        "sx": [
            [0.5 * (1 + 1j), 0.5 * (1 - 1j)],
            [0.5 * (1 - 1j), 0.5 * (1 + 1j)],
        ],
    }
    if name not in fixed:
        raise ValueError(f"no reference matrix for gate {name!r}")
    return torch.tensor(fixed[name], dtype=torch.complex128)


def _u3_matrix(theta: float, phi: float, lbd: float) -> torch.Tensor:
    cosine, sine = math.cos(theta / 2), math.sin(theta / 2)
    return torch.tensor(
        [
            [cosine, -cmath.exp(1j * lbd) * sine],
            [cmath.exp(1j * phi) * sine, cmath.exp(1j * (phi + lbd)) * cosine],
        ],
        dtype=torch.complex128,
    )


def _bit(index: int, wire: int, n_wires: int) -> int:
    return (index >> (n_wires - 1 - wire)) & 1


def _mask(wire: int, n_wires: int) -> int:
    return 1 << (n_wires - 1 - wire)


def _with_bit(index: int, wire: int, value: int, n_wires: int) -> int:
    mask = _mask(wire, n_wires)
    return (index | mask) if value else (index & ~mask)


def reference_state(
    program: Sequence[Operation],
    *,
    n_wires: int,
    dtype: torch.dtype,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply ``program`` to ``inputs`` one amplitude at a time."""
    amplitudes = inputs.to(dtype=dtype).clone()
    dimension = 1 << n_wires
    for op in program:
        if op.name == "cx":
            control, target = op.wires
            amplitudes = _reference_cx(amplitudes, control, target, n_wires=n_wires)
        elif op.name == "swap":
            first, second = op.wires
            amplitudes = _reference_swap(amplitudes, first, second, n_wires=n_wires)
        elif op.name in _SINGLE_QUBIT_MATRIX_GATES:
            matrix = _single_qubit_matrix(op.name, op.params).to(dtype)
            amplitudes = _reference_single_qubit(
                amplitudes, matrix, op.wires[0], n_wires=n_wires
            )
        else:
            raise ValueError(f"no reference application for gate {op.name!r}")
        assert amplitudes.shape[-1] == dimension
    return amplitudes


def _reference_single_qubit(
    amplitudes: torch.Tensor,
    matrix: torch.Tensor,
    wire: int,
    *,
    n_wires: int,
) -> torch.Tensor:
    output = torch.zeros_like(amplitudes)
    for batch in range(amplitudes.shape[0]):
        for index in range(1 << n_wires):
            column = _bit(index, wire, n_wires)
            for row in (0, 1):
                output[batch, _with_bit(index, wire, row, n_wires)] += (
                    matrix[row, column] * amplitudes[batch, index]
                )
    return output


def _reference_cx(
    amplitudes: torch.Tensor,
    control: int,
    target: int,
    *,
    n_wires: int,
) -> torch.Tensor:
    control_mask, target_mask = _mask(control, n_wires), _mask(target, n_wires)
    output = torch.zeros_like(amplitudes)
    for index in range(1 << n_wires):
        source = index ^ (target_mask if index & control_mask else 0)
        output[:, source] += amplitudes[:, index]
    return output


def _reference_swap(
    amplitudes: torch.Tensor,
    first: int,
    second: int,
    *,
    n_wires: int,
) -> torch.Tensor:
    first_mask, second_mask = _mask(first, n_wires), _mask(second, n_wires)
    output = torch.zeros_like(amplitudes)
    for index in range(1 << n_wires):
        high, low = bool(index & first_mask), bool(index & second_mask)
        source = index
        if high != low:
            source = (index ^ first_mask) ^ second_mask
        output[:, source] += amplitudes[:, index]
    return output


def normalized_input(
    *,
    n_wires: int,
    bsz: int = 1,
    dtype: torch.dtype = torch.complex128,
    seed: int = 20261,
) -> torch.Tensor:
    """A deterministic non-trivial input, so both bit values of every wire occur."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(bsz, 1 << n_wires, dtype=torch.complex128, generator=generator)
    return (raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True)).to(dtype)


def engine_state(
    program: Sequence[Operation],
    *,
    n_wires: int,
    dtype: torch.dtype,
    inputs: torch.Tensor,
    bsz: int | None = None,
) -> tuple[torch.Tensor, fq.Circuit]:
    """Run ``program`` through the shipped engine and return its state."""
    circuit = fq.Circuit(
        n_wires,
        bsz=inputs.shape[0] if bsz is None else bsz,
        device="cpu",
        dtype=dtype,
        inputs=inputs,
    )
    for op in program:
        getattr(circuit, op.name)(*op.wires, *op.params)
    return circuit.state(), circuit


def tolerance(program: Sequence[Operation], dtype: torch.dtype) -> float:
    """Absolute tolerance derived from the dtype and the rotation count."""
    rotations = sum(1 for op in program if op.name in _ROTATION_GATES)
    return _TOLERANCE_MARGIN * _DTYPE_EPSILON[dtype] * max(1, rotations)


def assert_matches_reference(
    program: Sequence[Operation],
    *,
    n_wires: int,
    dtype: torch.dtype = torch.complex128,
    inputs: torch.Tensor | None = None,
    strength: str = WITHIN_ACCUMULATED_ERROR,
) -> tuple[torch.Tensor, fq.Circuit]:
    """Compare the engine against the reference at the declared strength."""
    assert strength in {EXACT, WITHIN_ACCUMULATED_ERROR}
    if inputs is None:
        inputs = normalized_input(n_wires=n_wires, dtype=dtype)
    expected = reference_state(program, n_wires=n_wires, dtype=dtype, inputs=inputs)
    actual, circuit = engine_state(program, n_wires=n_wires, dtype=dtype, inputs=inputs)
    assert actual.dtype == dtype
    if strength == EXACT:
        assert torch.equal(
            actual, expected
        ), "a permutation-only program must reproduce amplitudes bit for bit"
    else:
        difference = float(torch.max(torch.abs(actual - expected)).item())
        assert difference <= tolerance(program, dtype), (
            f"maximum amplitude error {difference:.3e} exceeds the tolerance "
            f"{tolerance(program, dtype):.3e}"
        )
    return actual, circuit


def path_statistics(circuit: fq.Circuit) -> dict[str, Any]:
    return dict(circuit._last_statevector_runtime)


@pytest.mark.parametrize("gate", single_qubit_gate_names())
@pytest.mark.parametrize("wire", (0, 1, 2, 3))
def test_every_single_qubit_gate_on_every_wire_matches_the_reference(
    gate: str, wire: int
) -> None:
    """Matrix gates at every wire position, including the last wire."""
    program = [operation(gate, wire)]
    assert_matches_reference(program, n_wires=4)


@pytest.mark.parametrize(
    ("control", "target"),
    ((0, 1), (1, 2), (0, 3), (3, 0), (2, 0), (1, 3)),
)
def test_cx_variants_match_the_reference_exactly(control: int, target: int) -> None:
    """Adjacent, distant, and control-above-target CX are all pure permutations."""
    program = [operation("h", 0), operation("cx", control, target), operation("h", 2)]
    _, circuit = assert_matches_reference(program, n_wires=4, strength=EXACT)
    assert path_statistics(circuit)["permutation_gates"] >= 1


@pytest.mark.parametrize(("first", "second"), ((0, 1), (0, 3), (3, 0), (2, 3)))
def test_swap_matches_the_reference_exactly(first: int, second: int) -> None:
    program = [operation("h", 1), operation("swap", first, second)]
    assert_matches_reference(program, n_wires=4, strength=EXACT)


def _cx_sequence(length: int, *, n_wires: int, ladder: bool = True) -> list[Operation]:
    """A CX sequence whose controls are also targets of the same sequence."""
    steps: list[Operation] = []
    for index in range(length):
        if ladder:
            control = index % (n_wires - 1)
            steps.append(operation("cx", control, control + 1))
        else:
            # Control of one step is the target of the previous step, the case a
            # per-step mask cannot express.
            control = (index + 1) % n_wires
            target = index % n_wires
            if control == target:
                control = (control + 1) % n_wires
            steps.append(operation("cx", control, target))
    return steps


@pytest.mark.parametrize("length", (1, 2, 8))
def test_cx_sequences_match_the_reference_exactly(length: int) -> None:
    """A sequence is one affine permutation on the amplitudes, not one per gate.

    The planner only collects a sequence step at length two or more, so a single
    CX stays a plain permutation gate. That boundary is pinned here because the
    CPU sequence kernel is written against it.
    """
    program = [operation("h", 0), *_cx_sequence(length, n_wires=4), operation("x", 3)]
    _, circuit = assert_matches_reference(program, n_wires=4, strength=EXACT)
    statistics = path_statistics(circuit)
    assert statistics["permutation_gates"] >= length
    expected_regions = 1 if length >= 2 else 0
    assert statistics["triton_cx_sequence_regions"] == expected_regions


@pytest.mark.parametrize("length", (2, 5, 8))
def test_cx_sequences_with_overlapping_controls_match_the_reference_exactly(
    length: int,
) -> None:
    program = _cx_sequence(length, n_wires=4, ladder=False)
    assert_matches_reference(program, n_wires=4, strength=EXACT)


def test_repeated_pairs_of_the_same_cx_match_the_reference_exactly() -> None:
    """``cx(0, 1); cx(1, 0); cx(0, 1)`` is the standard maximally entangling idiom."""
    program = [
        operation("h", 0),
        operation("cx", 0, 1),
        operation("cx", 1, 0),
        operation("cx", 0, 1),
    ]
    assert_matches_reference(program, n_wires=2, strength=EXACT)


def test_batched_states_match_the_reference_exactly_for_permutations() -> None:
    program = [
        operation("x", 0),
        *_cx_sequence(6, n_wires=4),
        operation("swap", 0, 2),
    ]
    inputs = normalized_input(n_wires=4, bsz=3)
    assert_matches_reference(program, n_wires=4, inputs=inputs, strength=EXACT)


def test_batched_states_match_the_reference_for_rotations() -> None:
    program = [
        operation("ry", 0, params=(0.3,)),
        operation("cx", 0, 1),
        operation("rz", 1, params=(-0.8,)),
        operation("cx", 1, 2),
        operation("rx", 2, params=(1.1,)),
    ]
    inputs = normalized_input(n_wires=4, bsz=3, seed=99)
    actual, _ = assert_matches_reference(program, n_wires=4, inputs=inputs)
    assert actual.shape == (3, 16)
    for batch in range(3):
        norm = float(torch.linalg.vector_norm(actual[batch]).item())
        assert norm == pytest.approx(1.0, abs=1e-9)


def test_fused_rotation_regions_match_gate_by_gate_application() -> None:
    """Fusion must preserve the program, and it must actually engage here."""
    program: list[Operation] = []
    for layer in range(3):
        for wire in range(3):
            program.append(operation("ry", wire, params=(0.2 + layer * 0.1,)))
            program.append(operation("rz", wire, params=(-0.5 + wire * 0.2,)))
    _, circuit = assert_matches_reference(program, n_wires=3)
    statistics = path_statistics(circuit)
    assert statistics["fused_gate_regions"] >= 1
    assert statistics["fused_gate_count"] >= 2
    assert statistics["statevector_apply_count"] < len(program)


def test_diagonal_gates_match_the_reference_and_use_the_elementwise_kernel() -> None:
    program = [
        operation("h", 0),
        operation("rz", 1, params=(0.9,)),
        operation("p", 2, params=(1.4,)),
        operation("t", 3),
    ]
    _, circuit = assert_matches_reference(program, n_wires=4)
    statistics = path_statistics(circuit)
    assert statistics["diagonal_elementwise_gates"] >= 1


@pytest.mark.parametrize(
    ("name", "gates"),
    (
        ("hadamard_then_t", ("h", "t")),
        ("sx_then_t", ("sx", "t")),
        ("u3_pair", ("u3", "u3")),
        ("hadamard_then_rz", ("h", "rz")),
        ("phase_then_t", ("p", "t")),
    ),
)
def test_fused_regions_without_a_batched_rotation_pattern_match_the_reference(
    name: str, gates: tuple[str, ...]
) -> None:
    """The composition path, not the batched rotation path, orders these gates.

    A region whose gate names do not form a recognised rotation pattern is
    composed by multiplying its matrices together, so the multiplication order
    is the whole correctness argument for it. ``h`` and ``t`` do not commute, so
    a reversed product differs from the program by far more than rounding.
    """
    program: list[Operation] = []
    for wire in range(3):
        for gate in gates:
            program.append(operation(gate, wire))
    _, circuit = assert_matches_reference(program, n_wires=3)
    statistics = path_statistics(circuit)
    assert statistics["fused_gate_regions"] >= 1
    assert statistics["fused_gate_count"] >= len(gates)
    assert statistics["batched_rotation_sequence_regions"] == 0
    assert statistics["batched_rx_ry_rz_regions"] == 0


def test_two_gate_names_that_do_not_commute_are_composed_in_execution_order() -> None:
    """``h`` then ``t`` is not ``t`` then ``h``, and the engine must agree."""
    forward = [operation("h", 0), operation("t", 0)]
    reversed_program = [operation("t", 0), operation("h", 0)]
    inputs = normalized_input(n_wires=2)
    first = reference_state(forward, n_wires=2, dtype=torch.complex128, inputs=inputs)
    second = reference_state(
        reversed_program, n_wires=2, dtype=torch.complex128, inputs=inputs
    )
    # The two orders must differ, or this test would prove nothing about order.
    assert not torch.allclose(first, second, atol=1e-3)
    actual, circuit = assert_matches_reference(forward, n_wires=2, inputs=inputs)
    assert not torch.allclose(actual, second, atol=1e-3)
    assert path_statistics(circuit)["fused_gate_regions"] >= 1


def test_mixed_rotation_and_cx_layers_match_the_reference() -> None:
    program: list[Operation] = [operation("h", wire) for wire in range(4)]
    for layer in range(2):
        for wire in range(4):
            program.append(operation("ry", wire, params=(0.15 + layer * 0.05,)))
        for control in range(3):
            program.append(operation("cx", control, control + 1))
    program.append(operation("swap", 0, 3))
    actual, _ = assert_matches_reference(program, n_wires=4)
    norm = float(torch.linalg.vector_norm(actual).item())
    assert norm == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "dtype", (torch.complex64, torch.complex128), ids=("complex64", "complex128")
)
def test_dtype_is_preserved_and_its_precision_is_used(dtype: torch.dtype) -> None:
    """A complex128 request must not be silently computed or returned as float32."""
    program = [
        operation("ry", 0, params=(0.37,)),
        operation("cx", 0, 1),
        operation("u3", 1, params=(0.61, 0.42, 0.93)),
    ]
    inputs = normalized_input(n_wires=2, dtype=dtype)
    expected = reference_state(program, n_wires=2, dtype=dtype, inputs=inputs)
    actual, _ = engine_state(program, n_wires=2, dtype=dtype, inputs=inputs)
    assert actual.dtype == dtype
    difference = float(torch.max(torch.abs(actual - expected)).item())
    assert difference <= tolerance(program, dtype)


def test_a_fresh_simulation_agrees_with_the_cached_state() -> None:
    """Cache reuse must not change the answer, which the kernel work relies on."""
    program = [
        operation("h", 0),
        operation("ry", 1, params=(0.8,)),
        operation("cx", 0, 1),
    ]
    circuit = fq.Circuit(2, device="cpu", dtype=torch.complex128)
    for op in program:
        getattr(circuit, op.name)(*op.wires, *op.params)
    first = circuit.state()
    assert torch.equal(circuit.state(), first)
    assert torch.equal(circuit.state(refresh=True), first)


def test_mutating_a_circuit_invalidates_its_cached_state() -> None:
    """Appending a gate must not leave a stale state behind."""
    circuit = fq.Circuit(2, device="cpu", dtype=torch.complex128)
    circuit.h(0)
    before = circuit.state().clone()
    assert_matches_reference(
        [operation("h", 0)], n_wires=2, inputs=normalized_input(n_wires=2)
    )
    circuit.cx(0, 1)
    after = circuit.state()
    assert not torch.equal(before, after)
    # |00> -> (|00> + |11>)/sqrt(2), so only the two correlated amplitudes survive.
    assert float(after[0, 0].real.item()) == pytest.approx(SQRT_HALF, abs=1e-12)
    assert float(after[0, 3].real.item()) == pytest.approx(SQRT_HALF, abs=1e-12)
    assert float(after[0, 1].abs().item()) == pytest.approx(0.0, abs=1e-12)
    assert float(after[0, 2].abs().item()) == pytest.approx(0.0, abs=1e-12)


# Gradients. A fused region replaces several gate applications with one matrix
# product, and a permutation kernel replaces a copy with an index gather. Both
# are legitimate only if the backward pass still reflects the forward program,
# which is what these check. ``torch.autograd.gradcheck`` compares the analytic
# backward against central differences of the forward pass, so it is an
# independent check rather than a second use of the same graph.


def _differentiable_state(params: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """A small program whose parameters enter through rotations on both wires."""
    circuit = fq.Circuit(2, device="cpu", dtype=dtype)
    circuit.ry(0, params[0])
    circuit.cx(0, 1)
    circuit.rz(1, params[1])
    circuit.ry(0, params[2])
    state = circuit.state()
    return torch.cat([state.real.reshape(-1), state.imag.reshape(-1)])


def _differentiable_state_with_permutations(
    params: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Three wires, a CX ladder, and a non-adjacent SWAP around one rotation."""
    circuit = fq.Circuit(3, device="cpu", dtype=dtype)
    for wire in range(3):
        circuit.ry(wire, params[wire])
    for wire in range(2):
        circuit.cx(wire, wire + 1)
    circuit.rz(0, params[3])
    circuit.swap(0, 2)
    circuit.cx(2, 0)
    state = circuit.state()
    return torch.cat([state.real.reshape(-1), state.imag.reshape(-1)])


@pytest.mark.parametrize(
    ("name", "function", "values"),
    (
        ("rotations_and_cx", _differentiable_state, (0.31, -0.72, 1.05)),
        (
            "permutations_between_rotations",
            _differentiable_state_with_permutations,
            (0.2, -0.4, 0.6, -0.3),
        ),
    ),
)
def test_gradients_through_the_engine_match_central_differences(
    name: str, function: Any, values: tuple[float, ...]
) -> None:
    params = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda tensor: function(tensor, torch.complex128),
        (params,),
        eps=1e-6,
        atol=1e-8,
        rtol=1e-5,
    )


def test_gradients_are_available_at_the_default_dtype_with_a_declared_loose_bound() -> (
    None
):
    """complex64 backward is checked against complex128, not against the same graph.

    Finite differences at float32 precision cannot resolve a tight bound, so the
    comparison is against the analytic gradient of the same program in double
    precision. A kernel that silently detaches a fused region produces no
    gradient at all and fails before the tolerance matters.
    """
    values = (0.31, -0.72, 1.05)
    single = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    double = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    _differentiable_state(single, torch.complex64).sum().backward()
    _differentiable_state(double, torch.complex128).sum().backward()
    assert single.grad is not None and double.grad is not None
    assert torch.allclose(
        single.grad.to(torch.float64), double.grad, atol=1e-5, rtol=1e-4
    )


def test_reusing_the_cached_state_preserves_the_gradient() -> None:
    """Backward must not depend on whether the forward pass was cached."""
    fresh = torch.tensor([0.31, -0.72, 1.05], dtype=torch.float64, requires_grad=True)
    cached = fresh.detach().clone().requires_grad_(True)
    _differentiable_state(fresh, torch.complex128).sum().backward()

    circuit = fq.Circuit(2, device="cpu", dtype=torch.complex128)
    circuit.ry(0, cached[0])
    circuit.cx(0, 1)
    circuit.rz(1, cached[1])
    circuit.ry(0, cached[2])
    state = circuit.state()
    assert torch.equal(circuit.state(), state)
    torch.cat([state.real.reshape(-1), state.imag.reshape(-1)]).sum().backward()

    assert fresh.grad is not None and cached.grad is not None
    assert torch.equal(fresh.grad, cached.grad)


def test_gradients_with_a_batch_are_checked_against_central_differences() -> None:
    """The batched permutation path must carry a correct backward too."""

    def function(params: torch.Tensor) -> torch.Tensor:
        inputs = normalized_input(n_wires=3, bsz=2, dtype=torch.complex128, seed=5150)
        circuit = fq.Circuit(
            3, bsz=2, device="cpu", dtype=torch.complex128, inputs=inputs
        )
        circuit.ry(0, params[0])
        circuit.cx(2, 0)
        circuit.cx(1, 2)
        circuit.rz(1, params[1])
        state = circuit.state()
        return torch.cat([state.real.reshape(-1), state.imag.reshape(-1)])

    params = torch.tensor([0.44, -1.1], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(function, (params,), eps=1e-6, atol=1e-8, rtol=1e-5)
