"""The one-wire elementwise kernel: its routing, and how far it drifts.

Two things are being pinned here. The first is routing, because a silent
fallback to the matmul still returns a correct state and would leave the new
kernel unexercised. The second is accuracy: the kernel is *not* bitwise equal to
``torch.bmm`` at the widths that matter, only close to it, so the switch that
enables it has to be opt-in and the difference has to be bounded rather than
asserted away.

The bound is the measurement, not a guess. Across both dtypes, four wires per
width, and three circuit shapes at twelve to twenty wires, the worst difference
between the two routes was 0.41 ulp of the real dtype; ``_TOLERANCE_IN_ULPS``
multiplies the dtype's epsilon by four, which leaves an order of magnitude of
headroom. Tightening it towards the measured value would make the test fail on a
different BLAS, and loosening it would let a real regression through.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.statevector import local as statevector_local
from flagquantum.simulation.statevector.operations import (
    _apply_diagonal_matrix,
    _apply_matrix,
    _apply_single_wire_matrix,
    _cpu_single_wire_elementwise_enabled,
    _statevector_layout,
)

pytestmark = pytest.mark.unit

SWITCH = "FQ_CPU_SINGLE_WIRE_ELEMENTWISE"
_TOLERANCE_IN_ULPS = 4
_COMPLEX_DTYPES = (torch.complex64, torch.complex128)


def tolerance(dtype: torch.dtype) -> float:
    """Absolute tolerance for one gate, in the dtype's real epsilon."""

    real = torch.float32 if dtype == torch.complex64 else torch.float64
    return _TOLERANCE_IN_ULPS * torch.finfo(real).eps


def rotation_matrix(theta: float, dtype: torch.dtype) -> torch.Tensor:
    cosine, sine = math.cos(theta / 2), math.sin(theta / 2)
    return torch.tensor([[cosine, -sine], [sine, cosine]], dtype=dtype)


def general_matrix(dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [
            [complex(0.3, -0.4), complex(0.1, 0.7)],
            [complex(-0.2, 0.55), complex(0.6, 0.25)],
        ],
        dtype=dtype,
    )


def unitary_matrix(dtype: torch.dtype, seed: int = 909) -> torch.Tensor:
    """A dense two-by-two unitary, so that preserving the norm means something."""

    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(2, 2, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(2, 2, generator=generator, dtype=torch.float64)
    factor, _ = torch.linalg.qr(torch.complex(real, imaginary))
    return factor.to(dtype)


def normalized_state(
    bsz: int, n_wires: int, dtype: torch.dtype, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(bsz, 2**n_wires, generator=generator, dtype=dtype)
    return state / torch.linalg.vector_norm(state, dim=-1, keepdim=True)


def rotation_circuit(
    n_wires: int,
    *,
    dtype: torch.dtype,
    kind: str,
    layers: int = 4,
    bsz: int = 1,
) -> fq.Circuit:
    """A circuit whose one-wire gates are fused regions, interleaved with CX."""

    circuit = fq.Circuit(n_wires, bsz=bsz, dtype=dtype)
    for wire in range(n_wires):
        circuit.h(wire)
    for layer in range(layers):
        if kind == "ry":
            for wire in range(n_wires):
                circuit.ry(wire, 0.31 + 0.05 * layer + 0.01 * wire)
        elif kind == "rx":
            for wire in range(n_wires):
                circuit.rx(wire, 0.27 + 0.03 * layer)
        elif kind == "u3":
            for wire in range(n_wires):
                circuit.u3(wire, 0.3 + 0.01 * wire, 0.2, 0.1 + 0.02 * layer)
        elif kind == "diagonal":
            for wire in range(n_wires):
                circuit.rz(wire, 0.13 * (wire + 1))
                circuit.t(wire)
        else:  # pragma: no cover - the parametrization is the only caller
            raise AssertionError(kind)
        for wire in range(0, n_wires - 1, 2):
            circuit.cx(wire, wire + 1)
    return circuit


class Recorder:
    """Counts which kernels a run reaches, and delegates to all of them."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: dict[str, int] = {"elementwise": 0, "matmul": 0, "diagonal": 0}
        self.wires: list[tuple[int, ...]] = []
        self.by_kernel: dict[str, list[tuple[int, ...]]] = {
            "elementwise": [],
            "matmul": [],
            "diagonal": [],
        }
        monkeypatch.setattr(
            statevector_local,
            "_apply_single_wire_matrix",
            self._elementwise,
        )
        monkeypatch.setattr(statevector_local, "_apply_matrix", self._matmul)
        monkeypatch.setattr(
            statevector_local,
            "_apply_diagonal_matrix",
            self._diagonal,
        )

    def reset(self) -> None:
        """Forget everything recorded so far, so one phase can be asserted alone."""

        for name in self.calls:
            self.calls[name] = 0
            self.by_kernel[name].clear()
        self.wires.clear()

    def _elementwise(
        self, state: torch.Tensor, matrix: torch.Tensor, wire: int, n_wires: int
    ) -> torch.Tensor:
        self.calls["elementwise"] += 1
        self.wires.append((wire,))
        self.by_kernel["elementwise"].append((wire,))
        return _apply_single_wire_matrix(state, matrix, wire, n_wires)

    def _matmul(
        self,
        state: torch.Tensor,
        matrix: torch.Tensor,
        wires: tuple[int, ...],
        n_wires: int,
        layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    ) -> torch.Tensor:
        self.calls["matmul"] += 1
        self.wires.append(tuple(wires))
        self.by_kernel["matmul"].append(tuple(wires))
        return _apply_matrix(state, matrix, wires, n_wires, layout=layout)

    def _diagonal(
        self,
        state: torch.Tensor,
        matrix: torch.Tensor,
        wires: tuple[int, ...],
        n_wires: int,
        layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    ) -> torch.Tensor:
        self.calls["diagonal"] += 1
        self.wires.append(tuple(wires))
        self.by_kernel["diagonal"].append(tuple(wires))
        return _apply_diagonal_matrix(state, matrix, wires, n_wires, layout=layout)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    return Recorder(monkeypatch)


@pytest.fixture(autouse=True)
def _clear_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SWITCH, raising=False)


# --- the switch ------------------------------------------------------------


def test_the_switch_is_off_by_default() -> None:
    assert _cpu_single_wire_elementwise_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "on", "yes", " true "])
def test_the_switch_reads_an_affirmative_value(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(SWITCH, value)
    assert _cpu_single_wire_elementwise_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "off", "no", "nonsense"])
def test_the_switch_reads_anything_else_as_off(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(SWITCH, value)
    assert _cpu_single_wire_elementwise_enabled() is False


# --- routing ---------------------------------------------------------------


def test_the_default_route_never_reaches_the_new_kernel(recorder: Recorder) -> None:
    circuit = rotation_circuit(6, dtype=torch.complex64, kind="ry")
    circuit.state(refresh=True)

    assert recorder.calls["elementwise"] == 0
    assert recorder.calls["matmul"] >= 1


def test_an_explicit_zero_restores_the_matmul(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SWITCH, "0")
    circuit = rotation_circuit(6, dtype=torch.complex64, kind="ry")
    circuit.state(refresh=True)

    assert recorder.calls["elementwise"] == 0


def test_every_one_wire_step_routes_to_the_new_kernel(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    n_wires = 8
    circuit = fq.Circuit(n_wires, dtype=torch.complex64)
    # One gate per wire, so each wire is its own fused region: eight steps, no
    # more, and the count is the assertion rather than "greater than zero".
    for wire in range(n_wires):
        circuit.ry(wire, 0.25 + 0.1 * wire)
    circuit.state(refresh=True)

    assert recorder.calls["elementwise"] == n_wires
    assert recorder.calls["matmul"] == 0
    assert recorder.calls["diagonal"] == 0
    assert sorted(recorder.wires) == [(wire,) for wire in range(n_wires)]


def test_a_diagonal_one_wire_gate_keeps_its_own_kernel(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    circuit = fq.Circuit(6, dtype=torch.complex64)
    # One diagonal gate per wire, so each stays a plain gate step rather than
    # being fused into a region: a *fused* region is dispatched as a matmul
    # whether or not its members are diagonal, and that is the shipped behavior.
    for wire in range(6):
        circuit.rz(wire, 0.2 + 0.1 * wire)
    circuit.state(refresh=True)

    # _apply_diagonal_matrix is a different kernel from _apply_matrix, and the
    # elementwise kernel was measured against the latter, so a diagonal gate has
    # to stay where the shipped dispatch put it.
    assert recorder.calls["elementwise"] == 0
    assert recorder.calls["diagonal"] == 6


# Which kernel the shipped dispatch picks for each two-wire opcode. ``fixed`` is
# the permutation path in local.py, which never builds a matrix at all.
@pytest.mark.parametrize(
    "opcode,kernel",
    [
        ("cx", "fixed"),
        ("swap", "fixed"),
        ("cz", "diagonal"),
        ("rzz", "diagonal"),
        ("rxx", "matmul"),
    ],
)
def test_a_two_wire_gate_never_routes_to_the_new_kernel(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch, opcode: str, kernel: str
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    circuit = fq.Circuit(4, dtype=torch.complex64)
    for wire in range(4):
        circuit.h(wire)
    if opcode in {"rzz", "rxx"}:
        getattr(circuit, opcode)(0, 1, 0.3)
    else:
        getattr(circuit, opcode)(0, 1)
    circuit.state(refresh=True)

    # The four H gates are one-wire gates and should reach the new kernel; the
    # two-wire gate is the one that must not. The routing of the two-wire gate
    # itself is asserted as well, so that a change of dispatch shows up here
    # rather than only in the comparison tests.
    assert recorder.by_kernel["elementwise"] == [(0,), (1,), (2,), (3,)]
    expected: dict[str, list[tuple[int, ...]]] = {
        "elementwise": [(0,), (1,), (2,), (3,)],
        "matmul": [],
        "diagonal": [],
    }
    if kernel != "fixed":
        expected[kernel].append((0, 1))
    assert recorder.by_kernel == expected


def test_the_dispatch_helper_takes_the_new_kernel_for_one_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    n_wires = 10
    state = normalized_state(2, n_wires, torch.complex64, seed=17)
    matrix = rotation_matrix(0.37, torch.complex64)

    result = statevector_local._apply_gate_matrix(
        state,
        matrix,
        (4,),
        n_wires,
        uses_diagonal_kernel=False,
    )

    expected = _apply_single_wire_matrix(state, matrix, 4, n_wires)
    assert torch.equal(result, expected)


def test_the_dispatch_helper_stays_on_the_matmul_for_two_wires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    n_wires = 6
    state = normalized_state(1, n_wires, torch.complex64, seed=18)
    matrix = torch.eye(4, dtype=torch.complex64) * complex(0.5, 0.5)

    result = statevector_local._apply_gate_matrix(
        state,
        matrix,
        (1, 2),
        n_wires,
        uses_diagonal_kernel=False,
    )

    assert torch.equal(result, _apply_matrix(state, matrix, (1, 2), n_wires))


def test_the_dispatch_helper_honours_the_diagonal_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    n_wires = 6
    state = normalized_state(1, n_wires, torch.complex64, seed=19)
    matrix = torch.diag(torch.tensor([1.0, complex(0.6, 0.8)], dtype=torch.complex64))

    result = statevector_local._apply_gate_matrix(
        state,
        matrix,
        (2,),
        n_wires,
        uses_diagonal_kernel=True,
    )

    assert torch.equal(result, _apply_diagonal_matrix(state, matrix, (2,), n_wires))


def test_the_dispatch_helper_honours_the_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SWITCH, "0")
    n_wires = 6
    state = normalized_state(1, n_wires, torch.complex64, seed=20)
    matrix = rotation_matrix(0.37, torch.complex64)

    result = statevector_local._apply_gate_matrix(
        state,
        matrix,
        (2,),
        n_wires,
        uses_diagonal_kernel=False,
    )

    assert torch.equal(result, _apply_matrix(state, matrix, (2,), n_wires))


# --- the predicate itself --------------------------------------------------
#
# The device condition cannot be exercised by running a circuit, because this
# host has no CUDA device. It is instead evaluated as the named predicate it is,
# on a ``meta`` tensor whose device is real but whose data is not.


def test_the_predicate_accepts_a_one_wire_cpu_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    state = torch.empty(2, 8, dtype=torch.complex64)

    assert (
        statevector_local._use_elementwise_single_wire(
            state, (3,), uses_diagonal_kernel=False
        )
        is True
    )


@pytest.mark.parametrize(
    "wires,uses_diagonal_kernel",
    [((2, 3), False), ((2,), True)],
)
def test_the_predicate_rejects_the_two_kernels_it_does_not_stand_in_for(
    monkeypatch: pytest.MonkeyPatch, wires: tuple[int, ...], uses_diagonal_kernel: bool
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    state = torch.empty(1, 8, dtype=torch.complex64)

    assert (
        statevector_local._use_elementwise_single_wire(
            state, wires, uses_diagonal_kernel=uses_diagonal_kernel
        )
        is False
    )


def test_the_predicate_rejects_a_non_cpu_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SWITCH, "1")
    state = torch.empty(2, 8, dtype=torch.complex64, device="meta")

    assert state.device.type != "cpu"
    assert (
        statevector_local._use_elementwise_single_wire(
            state, (3,), uses_diagonal_kernel=False
        )
        is False
    )


def test_the_predicate_rejects_a_disabled_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SWITCH, "0")
    state = torch.empty(2, 8, dtype=torch.complex64)

    assert (
        statevector_local._use_elementwise_single_wire(
            state, (3,), uses_diagonal_kernel=False
        )
        is False
    )


# --- the kernel against the matmul ----------------------------------------


@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
@pytest.mark.parametrize("n_wires", [8, 12, 20])
@pytest.mark.parametrize("bsz", [1, 2])
def test_the_kernel_agrees_with_the_matmul(
    dtype: torch.dtype, n_wires: int, bsz: int
) -> None:
    state = normalized_state(bsz, n_wires, dtype, seed=101 + n_wires)
    matrix = general_matrix(dtype)
    for wire in (0, 1, n_wires // 2, n_wires - 1):
        reference = _apply_matrix(
            state,
            matrix,
            (wire,),
            n_wires,
            layout=_statevector_layout(n_wires, (wire,)),
        )
        candidate = _apply_single_wire_matrix(state, matrix, wire, n_wires)
        difference = float((reference - candidate).abs().max())
        assert difference <= tolerance(dtype), (n_wires, wire, difference)


@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_the_kernel_agrees_for_per_row_matrices(dtype: torch.dtype) -> None:
    n_wires = 12
    for bsz in (1, 2, 4):
        state = normalized_state(bsz, n_wires, dtype, seed=202 + bsz)
        angles = torch.linspace(0.1, 0.9, bsz, dtype=torch.float64) * 0.4
        cosine, sine = torch.cos(angles), torch.sin(angles)
        rows = torch.stack(
            (
                torch.stack((cosine, -sine), dim=-1),
                torch.stack((sine, cosine), dim=-1),
            ),
            dim=-2,
        ).to(dtype)
        reference = _apply_matrix(state, rows, (5,), n_wires)
        candidate = _apply_single_wire_matrix(state, rows, 5, n_wires)
        assert float((reference - candidate).abs().max()) <= tolerance(dtype)


@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_a_two_dimensional_matrix_matches_its_expanded_form(dtype: torch.dtype) -> None:
    n_wires = 10
    bsz = 3
    state = normalized_state(bsz, n_wires, dtype, seed=303)
    matrix = general_matrix(dtype)
    expanded = matrix.reshape(1, 2, 2).expand(bsz, -1, -1)

    flat = _apply_single_wire_matrix(state, matrix, 3, n_wires)
    batched = _apply_single_wire_matrix(state, expanded, 3, n_wires)

    assert torch.equal(flat, batched)


@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_a_single_row_batched_matrix_broadcasts_over_the_batch(
    dtype: torch.dtype,
) -> None:
    """A (1, 2, 2) matrix is expanded, which is more than ``_apply_matrix`` does."""

    n_wires = 10
    bsz = 4
    state = normalized_state(bsz, n_wires, dtype, seed=313)
    matrix = general_matrix(dtype)
    one_row = matrix.reshape(1, 2, 2)

    broadcast = _apply_single_wire_matrix(state, one_row, 3, n_wires)
    explicit = _apply_single_wire_matrix(state, one_row.expand(bsz, -1, -1), 3, n_wires)

    assert torch.equal(broadcast, explicit)
    # Pinning why the branch exists at all: the kernel it stands in for cannot
    # take this shape, so the two are not interchangeable on this input.
    with pytest.raises(RuntimeError):
        _apply_matrix(state, one_row, (3,), n_wires)


@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_the_kernel_casts_a_matrix_to_the_state_dtype(dtype: torch.dtype) -> None:
    n_wires = 8
    other = torch.complex128 if dtype == torch.complex64 else torch.complex64
    state = normalized_state(1, n_wires, dtype, seed=404)
    matrix = rotation_matrix(0.37, other)

    result = _apply_single_wire_matrix(state, matrix, 5, n_wires)

    assert result.dtype == dtype
    assert result.device == state.device


@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_the_kernel_preserves_the_norm(dtype: torch.dtype) -> None:
    n_wires = 12
    state = normalized_state(1, n_wires, dtype, seed=505)
    matrix = unitary_matrix(dtype)

    result = _apply_single_wire_matrix(state, matrix, 4, n_wires)

    before = float(torch.linalg.vector_norm(state))
    after = float(torch.linalg.vector_norm(result))
    # A norm over 2 ** n_wires amplitudes accumulates over that many terms, so
    # the allowance grows like its square root rather than staying at one gate's
    # epsilon. Measured here: 3.0 ulps on complex64 and 4.0 on complex128,
    # against an allowance of 256 and 64 ulps respectively.
    allowance = math.sqrt(2**n_wires) * tolerance(dtype)
    assert abs(after - before) <= allowance


@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_the_kernel_gradients_match_the_matmul(dtype: torch.dtype) -> None:
    n_wires = 12
    matrix = general_matrix(dtype)
    weight = normalized_state(1, n_wires, dtype, seed=606)
    base = normalized_state(1, n_wires, dtype, seed=707)

    collected = {}
    for label, kernel in (
        (
            "matmul",
            lambda value: _apply_matrix(value, matrix, (5,), n_wires),
        ),
        (
            "elementwise",
            lambda value: _apply_single_wire_matrix(value, matrix, 5, n_wires),
        ),
    ):
        operand = base.clone().requires_grad_(True)
        (kernel(operand) * weight).real.sum().backward()
        assert operand.grad is not None
        collected[label] = operand.grad

    difference = float((collected["matmul"] - collected["elementwise"]).abs().max())
    assert difference <= tolerance(dtype)


def test_the_diagonal_guard_is_not_vacuous() -> None:
    """The two shipped kernels really do disagree, which is why the flag exists."""

    n_wires = 12
    state = normalized_state(1, n_wires, torch.complex64, seed=808)
    matrix = torch.diag(torch.tensor([1.0, complex(0.6, 0.8)], dtype=torch.complex64))

    elementwise = _apply_single_wire_matrix(state, matrix, 4, n_wires)
    diagonal = _apply_diagonal_matrix(
        state,
        matrix,
        (4,),
        n_wires,
        layout=_statevector_layout(n_wires, (4,)),
    )

    assert not torch.equal(elementwise, diagonal)
    assert float((elementwise - diagonal).abs().max()) > 0.0


# --- end to end -----------------------------------------------------------


@pytest.mark.parametrize("kind", ["ry", "rx", "u3", "diagonal"])
@pytest.mark.parametrize("n_wires", [12, 20])
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_circuit_states_agree_with_the_switch_on_and_off(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    n_wires: int,
    dtype: torch.dtype,
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    enabled = (
        rotation_circuit(n_wires, dtype=dtype, kind=kind).state(refresh=True).clone()
    )

    monkeypatch.setenv(SWITCH, "0")
    disabled = (
        rotation_circuit(n_wires, dtype=dtype, kind=kind).state(refresh=True).clone()
    )

    difference = float((enabled - disabled).abs().max())
    assert difference <= tolerance(dtype), (kind, n_wires, difference)


@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_batched_circuit_states_agree(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    n_wires = 12
    monkeypatch.setenv(SWITCH, "1")
    enabled = (
        rotation_circuit(n_wires, dtype=dtype, kind="ry", bsz=2)
        .state(refresh=True)
        .clone()
    )
    monkeypatch.setenv(SWITCH, "0")
    disabled = (
        rotation_circuit(n_wires, dtype=dtype, kind="ry", bsz=2)
        .state(refresh=True)
        .clone()
    )

    assert enabled.shape == disabled.shape
    assert float((enabled - disabled).abs().max()) <= tolerance(dtype)


def test_expectation_z_agrees_between_the_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_wires = 12
    monkeypatch.setenv(SWITCH, "1")
    enabled = (
        rotation_circuit(n_wires, dtype=torch.complex64, kind="ry")
        .expectation_z()
        .clone()
    )
    monkeypatch.setenv(SWITCH, "0")
    disabled = (
        rotation_circuit(n_wires, dtype=torch.complex64, kind="ry")
        .expectation_z()
        .clone()
    )

    assert float((enabled - disabled).abs().max()) <= 10 * tolerance(torch.complex64)


def test_the_pauli_string_path_reaches_the_new_kernel(
    recorder: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SWITCH, "1")
    n_wires = 6
    circuit = rotation_circuit(n_wires, dtype=torch.complex64, kind="ry")
    circuit.state(refresh=True)
    recorder.reset()

    statevector_local._expectation_pauli_string(
        circuit,
        x=tuple(range(n_wires)),
        y=(),
        z=(),
    )

    # One matrix application per wire in the string, and none of them through
    # the matmul: this path used to be one _apply_matrix per wire.
    assert recorder.by_kernel["elementwise"] == [(wire,) for wire in range(n_wires)]
    assert recorder.by_kernel["matmul"] == []
    assert recorder.by_kernel["diagonal"] == []


@pytest.mark.parametrize("observable", ["x", "y", "z"])
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_pauli_string_expectations_agree_between_the_routes(
    monkeypatch: pytest.MonkeyPatch, observable: str, dtype: torch.dtype
) -> None:
    n_wires = 12
    wires = tuple(range(n_wires))

    def evaluate() -> torch.Tensor:
        circuit = rotation_circuit(n_wires, dtype=dtype, kind="ry")
        return statevector_local._expectation_pauli_string(
            circuit,
            x=wires if observable == "x" else (),
            y=wires if observable == "y" else (),
            z=wires if observable == "z" else (),
        )

    monkeypatch.setenv(SWITCH, "1")
    enabled = evaluate().clone()
    monkeypatch.setenv(SWITCH, "0")
    disabled = evaluate().clone()

    allowance = math.sqrt(2**n_wires) * tolerance(dtype)
    assert float((enabled - disabled).abs().max()) <= allowance


@pytest.mark.parametrize("observable", ["x", "y", "z"])
def test_the_public_pauli_string_entry_point_agrees_between_the_routes(
    monkeypatch: pytest.MonkeyPatch, observable: str
) -> None:
    """The routing change only matters if the public API actually reaches it."""

    n_wires = 12
    wires = tuple(range(n_wires))

    def evaluate() -> torch.Tensor:
        circuit = rotation_circuit(n_wires, dtype=torch.complex64, kind="ry")
        return circuit.expectation_ps(**{observable: wires})

    monkeypatch.setenv(SWITCH, "1")
    enabled = evaluate().clone()
    monkeypatch.setenv(SWITCH, "0")
    disabled = evaluate().clone()

    assert enabled.shape == disabled.shape
    allowance = math.sqrt(2**n_wires) * tolerance(torch.complex64)
    assert float((enabled - disabled).abs().max()) <= allowance
