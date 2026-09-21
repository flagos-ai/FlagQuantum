"""Route and equivalence tests for the CPU CX-sequence gather kernel.

A CX sequence is a permutation of basis indices, so it can be applied as one
gather instead of one pass over the state per gate. Whether that is correct does
not depend on how fast it is, and whether it is *used* is not visible in the
amplitudes: a program that silently fell back to the per-gate loop would still
return the right state. These tests therefore assert both, and the routing
assertions are what make the equivalence assertions mean something.

The kernel replaces several tensor operations with one, so the pieces that can
go wrong are: the direction of the permutation (the gather needs the inverse map,
and a single CX hides that because it is its own inverse), the bit order used
when the table is expanded, the dtype of the index, and the batch dimension. Each
of those has a case below that fails if it is wrong.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.statevector import local as statevector_local
from flagquantum.simulation.statevector import operations as statevector_operations

pytestmark = pytest.mark.unit

SQRT_HALF = 2.0**-0.5


@pytest.fixture(autouse=True)
def _clear_gather_cache() -> Iterator[None]:
    """Keep the content-addressed table cache from leaking between tests."""

    statevector_operations.clear_statevector_cx_index_cache()
    yield
    statevector_operations.clear_statevector_cx_index_cache()


def normalized_input(*, n_wires: int, bsz: int = 1, dtype: torch.dtype) -> torch.Tensor:
    """A reproducible normalized state with no structure a permutation could keep."""

    generator = torch.Generator().manual_seed(90210)
    real = torch.randn(bsz, 2**n_wires, generator=generator)
    imag = torch.randn(bsz, 2**n_wires, generator=generator)
    state = torch.complex(real, imag).to(dtype)
    return state / torch.linalg.vector_norm(state, dim=-1, keepdim=True)


def per_gate_loop(
    state: torch.Tensor, controls: Sequence[int], targets: Sequence[int], n_wires: int
) -> torch.Tensor:
    """The loop the gather replaces, spelled out here rather than imported."""

    current = state
    for control, target in zip(controls, targets, strict=True):
        current = statevector_operations._apply_cx_permutation(
            current, (control, target), n_wires
        )
    return current


def forward_map(
    controls: Sequence[int], targets: Sequence[int], n_wires: int
) -> list[int]:
    """Where the sequence sends each basis index, one index at a time."""

    images = []
    for index in range(1 << n_wires):
        value = index
        for control, target in zip(controls, targets, strict=True):
            if (value >> (n_wires - 1 - control)) & 1:
                value ^= 1 << (n_wires - 1 - target)
        images.append(value)
    return images


def gather_directly(
    state: torch.Tensor, controls: Sequence[int], targets: Sequence[int], n_wires: int
) -> torch.Tensor:
    """The gather kernel itself, bypassing the program planner."""

    return statevector_operations._apply_cx_sequence_gather(
        state, tuple(controls), tuple(targets), n_wires
    )


def programs(n_wires: int) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """CX sequences whose control/target structure differs in a way that matters.

    Two constraints hold for every entry, and both matter. Every wire stays
    inside ``range(n_wires)``, so one set can be reused at several widths. And no
    pair has ``control == target``: that is not a CX, the shipped per-gate kernel
    mis-handles it because un-binding the control axis shifts the target axis
    beneath it, and the fuser never emits one.
    """

    top = n_wires - 1
    mid = min(3, top)
    adjacent = tuple(range(top))
    third = 2 % n_wires
    sequences = {
        "adjacent_chain": (adjacent, tuple(wire + 1 for wire in adjacent)),
        "reversed_chain": (adjacent[::-1], tuple(wire + 1 for wire in adjacent[::-1])),
        "fan_out_from_zero": ((0,) * top, tuple(range(1, n_wires))),
        "control_after_target": (tuple(range(1, n_wires)), adjacent),
        "repeated_pair": ((0, 1) * 4, (1, 0) * 4),
        "overlapping_triples": (
            (0, 1, 0, mid, 1, 0, mid, 1),
            (1, mid, third, 1, 0, mid, 0, third),
        ),
        "far_apart": ((0, top, 1), (top, 0, third)),
    }
    for name, (controls, targets) in sequences.items():
        for control, target in zip(controls, targets, strict=True):
            assert control != target, f"{name} uses a wire as its own control"
            assert max(control, target) < n_wires, f"{name} leaves the register"
    return sequences


def cx_chain(n_wires: int, length: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """A CX chain of exactly ``length`` gates, for threshold-sensitive tests."""

    controls = tuple(wire % (n_wires - 1) for wire in range(length))
    return controls, tuple(control + 1 for control in controls)


def brute_force_inverse(
    controls: Sequence[int], targets: Sequence[int], n_wires: int
) -> list[int]:
    """The inverse permutation, one basis index at a time, with Python integers.

    This is the independent definition the table has to agree with: walk every
    basis index through the sequence, then invert the resulting mapping. It is
    slow and obvious on purpose.
    """

    forward = []
    for index in range(1 << n_wires):
        value = index
        for control, target in zip(controls, targets, strict=True):
            if (value >> (n_wires - 1 - control)) & 1:
                value ^= 1 << (n_wires - 1 - target)
        forward.append(value)
    inverse = [0] * (1 << n_wires)
    for source, destination in enumerate(forward):
        inverse[destination] = source
    return inverse


def circuit_state(
    *,
    n_wires: int,
    controls: Sequence[int],
    targets: Sequence[int],
    bsz: int = 1,
    dtype: torch.dtype = torch.complex64,
) -> tuple[torch.Tensor, Any]:
    """Run a CX sequence through the public engine and return its state."""

    circuit = fq.Circuit(n_wires, bsz=bsz, device="cpu", dtype=dtype)
    circuit._inputs = normalized_input(n_wires=n_wires, bsz=bsz, dtype=dtype)
    for control, target in zip(controls, targets, strict=True):
        circuit.cx(control, target)
    return circuit.state(refresh=True), circuit


def recorded_gathers(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[tuple[int, ...], int]]:
    """Record every gather the engine actually performs, with its width."""

    calls: list[tuple[tuple[int, ...], int]] = []
    original = statevector_local._apply_cx_sequence_gather

    def recording(
        state: torch.Tensor, controls: Any, targets: Any, n_wires: int
    ) -> torch.Tensor:
        calls.append((tuple(controls), n_wires))
        return original(state, controls, targets, n_wires)

    monkeypatch.setattr(statevector_local, "_apply_cx_sequence_gather", recording)
    return calls


# --------------------------------------------------------------------------- #
# Routing: the fast path is actually taken, and only where it is allowed to be
# --------------------------------------------------------------------------- #


def test_a_long_cx_sequence_is_applied_by_one_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eight gates is the measured break-even point for building a table."""

    controls, targets = cx_chain(5, 8)
    calls = recorded_gathers(monkeypatch)
    _, circuit = circuit_state(n_wires=5, controls=controls, targets=targets)

    statistics = dict(circuit._last_statevector_runtime)
    assert statistics["triton_cx_sequence_regions"] == 1
    assert len(calls) == 1
    assert len(calls[0][0]) == 8


def test_a_sequence_below_the_threshold_stays_on_the_per_gate_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One gate below the threshold, and the table is not built at all."""

    controls, targets = cx_chain(5, 7)
    calls = recorded_gathers(monkeypatch)
    _, circuit = circuit_state(n_wires=5, controls=controls, targets=targets)

    assert dict(circuit._last_statevector_runtime)["triton_cx_sequence_regions"] == 1
    assert calls == []


def test_the_switch_restores_the_per_gate_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-out has to actually opt out, not merely change a number."""

    controls, targets = cx_chain(5, 8)
    monkeypatch.setenv("FQ_CPU_CX_SEQUENCE_GATHER", "0")
    calls = recorded_gathers(monkeypatch)
    _, circuit = circuit_state(n_wires=5, controls=controls, targets=targets)

    assert calls == []
    assert dict(circuit._last_statevector_runtime)["triton_cx_sequence_regions"] == 1


def test_the_threshold_is_the_documented_one() -> None:
    """Pin the constant so a silent retune cannot pass unnoticed."""

    assert statevector_operations._CX_SEQUENCE_GATHER_MINIMUM_LENGTH == 8
    assert statevector_operations._cpu_cx_sequence_gather_enabled() is True


def test_the_table_is_not_built_for_a_sequence_that_will_not_use_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached table costs memory even when no program reads it."""

    controls, targets = cx_chain(5, 7)
    _, circuit = circuit_state(n_wires=5, controls=controls, targets=targets)
    assert dict(circuit._last_statevector_runtime)["triton_cx_sequence_regions"] == 1
    assert statevector_operations._cx_sequence_index_cache_bytes() == 0


# --------------------------------------------------------------------------- #
# Equivalence: the gather is the same permutation the loop performs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(programs(6)))
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_the_gather_reproduces_the_per_gate_loop_exactly(
    name: str, dtype: torch.dtype
) -> None:
    """A permutation moves amplitudes; it cannot round them."""

    controls, targets = programs(6)[name]
    state = normalized_input(n_wires=6, dtype=dtype)

    gathered = gather_directly(state, controls, targets, 6)
    looped = per_gate_loop(state, controls, targets, 6)

    assert torch.equal(gathered, looped)


@pytest.mark.parametrize("n_wires", [3, 5, 8])
def test_the_index_table_is_the_inverse_permutation(n_wires: int) -> None:
    """Compare against a definition that shares no code with the kernel."""

    for controls, targets in programs(n_wires).values():
        table = statevector_operations._cx_sequence_permutation_index(
            controls,
            targets,
            n_wires,
            device=torch.device("cpu"),
            dtype=torch.complex64,
        )
        assert table.tolist() == brute_force_inverse(controls, targets, n_wires)


def test_the_table_is_the_inverse_and_not_the_forward_map() -> None:
    """A single CX is its own inverse, so two of them can hide the convention.

    Most short sequences are involutions, and for those the forward and inverse
    maps coincide - a kernel that built the wrong one would still pass. This uses
    ``cx(0, 1)`` followed by ``cx(1, 0)``, which is not an involution, and
    asserts first that the two maps really do differ so the comparison cannot
    quietly become vacuous.
    """

    n_wires = 4
    controls, targets = (0, 1), (1, 0)
    state = normalized_input(n_wires=n_wires, dtype=torch.complex64)

    inverse = statevector_operations._cx_sequence_permutation_index(
        controls, targets, n_wires, device=torch.device("cpu"), dtype=torch.complex64
    )
    images = forward_map(controls, targets, n_wires)
    # A gather table entry is where a value is read *from*, so using the forward
    # map as the table is exactly the mistake this test is looking for.
    forward_as_table = torch.tensor(images, dtype=torch.int32)
    is_involution = [images[destination] for destination in images] == list(
        range(1 << n_wires)
    )

    assert not is_involution, "this pair must not be an involution"
    assert not torch.equal(inverse, forward_as_table), (
        "for an involution the two conventions coincide, so this case would not "
        "distinguish them"
    )

    gathered = torch.index_select(state, 1, inverse)
    assert torch.equal(gathered, per_gate_loop(state, controls, targets, n_wires))
    assert not torch.equal(torch.index_select(state, 1, forward_as_table), gathered)


def test_two_sequences_with_opposite_effects_are_told_apart() -> None:
    """A table built for the wrong sequence would still be a permutation."""

    n_wires = 5
    first = ((0, 1, 2, 3, 0, 1, 2, 3), (1, 2, 3, 4, 1, 2, 3, 4))
    second = (first[0][::-1], first[1][::-1])
    state = normalized_input(n_wires=n_wires, dtype=torch.complex64)

    forward_result = gather_directly(state, *first, n_wires)
    reverse_result = gather_directly(state, *second, n_wires)

    assert torch.equal(forward_result, per_gate_loop(state, *first, n_wires))
    assert torch.equal(reverse_result, per_gate_loop(state, *second, n_wires))
    assert not torch.equal(forward_result, reverse_result)


def test_the_gather_carries_every_batch_row() -> None:
    """The permutation acts on the amplitude axis, not on the batch axis."""

    n_wires = 6
    bsz = 3
    controls, targets = programs(n_wires)["overlapping_triples"]
    state = normalized_input(n_wires=n_wires, bsz=bsz, dtype=torch.complex64)

    gathered = gather_directly(state, controls, targets, n_wires)

    assert gathered.shape == state.shape
    assert torch.equal(gathered, per_gate_loop(state, controls, targets, n_wires))
    for row in range(bsz):
        assert torch.equal(
            gathered[row],
            per_gate_loop(state[row : row + 1], controls, targets, n_wires)[0],
        )
    assert not torch.equal(gathered[0], gathered[1])


def test_the_gather_agrees_through_the_public_engine_with_the_switch_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end, on a width where a chain fuses into one region."""

    n_wires = 12
    controls = tuple(range(n_wires - 1))
    targets = tuple(wire + 1 for wire in controls)

    gathered, gathered_circuit = circuit_state(
        n_wires=n_wires, controls=controls, targets=targets
    )
    monkeypatch.setenv("FQ_CPU_CX_SEQUENCE_GATHER", "0")
    statevector_operations.clear_statevector_cx_index_cache()
    looped, looped_circuit = circuit_state(
        n_wires=n_wires, controls=controls, targets=targets
    )

    assert (
        dict(gathered_circuit._last_statevector_runtime)["triton_cx_sequence_regions"]
        == 1
    )
    assert (
        dict(looped_circuit._last_statevector_runtime)["triton_cx_sequence_regions"]
        == 1
    )
    assert torch.equal(gathered, looped)


def test_a_mixed_circuit_agrees_with_the_switch_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-qubit gates between CX fragments split the chain into regions.

    The pure-chain numbers do not carry over to this shape: fusion regions get
    shorter, so the number of tables built goes up while the length of each goes
    down. Both paths still have to agree bit for bit.
    """

    n_wires = 10

    def build() -> tuple[torch.Tensor, Any]:
        circuit = fq.Circuit(n_wires, device="cpu", dtype=torch.complex64)
        circuit._inputs = normalized_input(
            n_wires=n_wires, dtype=torch.complex64
        ).reshape(-1)
        for layer in range(4):
            for wire in range(n_wires):
                circuit.h(wire)
                circuit.rz(wire, 0.3 + 0.1 * wire)
            for wire in range(0, n_wires - 1, 2):
                circuit.cx(wire, wire + 1)
        return circuit.state(refresh=True), circuit

    gathered, gathered_circuit = build()
    gathered_statistics = dict(gathered_circuit._last_statevector_runtime)
    monkeypatch.setenv("FQ_CPU_CX_SEQUENCE_GATHER", "0")
    statevector_operations.clear_statevector_cx_index_cache()
    looped, looped_circuit = build()
    looped_statistics = dict(looped_circuit._last_statevector_runtime)

    assert gathered_statistics["triton_cx_sequence_regions"] > 0
    assert (
        gathered_statistics["triton_cx_sequence_regions"]
        == looped_statistics["triton_cx_sequence_regions"]
    )
    assert (
        gathered_statistics["permutation_gates"]
        == looped_statistics["permutation_gates"]
    )
    assert torch.equal(gathered, looped)


@pytest.mark.parametrize("bsz", [1, 2])
def test_the_dtype_of_the_state_is_preserved(bsz: int) -> None:
    """A gather must not silently upcast or downcast the amplitudes."""

    n_wires = 5
    controls, targets = programs(n_wires)["adjacent_chain"]
    for dtype in (torch.complex64, torch.complex128):
        state = normalized_input(n_wires=n_wires, bsz=bsz, dtype=dtype)
        gathered = gather_directly(state, controls, targets, n_wires)
        assert gathered.dtype is dtype
        assert torch.equal(gathered, per_gate_loop(state, controls, targets, n_wires))


# --------------------------------------------------------------------------- #
# The index itself: dtype, memory, cache behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("n_wires", "expected"),
    [(3, torch.int32), (20, torch.int32), (31, torch.int32), (32, torch.int64)],
)
def test_the_index_dtype_grows_only_where_it_has_to(
    n_wires: int, expected: torch.dtype
) -> None:
    """An int32 index halves the table and still addresses every amplitude."""

    assert statevector_operations._cx_sequence_index_dtype(n_wires) is expected


def test_the_cached_table_is_reused_instead_of_rebuilt() -> None:
    """The second run must not pay the build again, or the threshold is wrong."""

    n_wires = 6
    controls, targets = programs(n_wires)["overlapping_triples"]
    table = statevector_operations._cx_sequence_permutation_index(
        controls, targets, n_wires, device=torch.device("cpu"), dtype=torch.complex64
    )
    first_bytes = statevector_operations._cx_sequence_index_cache_bytes()

    again = statevector_operations._cx_sequence_permutation_index(
        controls, targets, n_wires, device=torch.device("cpu"), dtype=torch.complex64
    )

    assert again is table
    assert statevector_operations._cx_sequence_index_cache_bytes() == first_bytes


def test_the_table_cache_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Content addressing means nothing evicts on its own, so a budget must."""

    monkeypatch.setattr(statevector_operations, "_CX_SEQUENCE_INDEX_CACHE_BYTES", 4096)
    n_wires = 6
    per_table = (1 << n_wires) * 4
    assert per_table == 256

    for index in range(40):
        controls = (index % (n_wires - 1), index % (n_wires - 1))
        targets = ((index + 1) % n_wires, (index + 2) % n_wires)
        statevector_operations._cx_sequence_permutation_index(
            controls,
            targets,
            n_wires,
            device=torch.device("cpu"),
            dtype=torch.complex64,
        )

    assert statevector_operations._cx_sequence_index_cache_bytes() <= 4096 + per_table
    assert len(statevector_operations._CX_SEQUENCE_INDEX_CACHE) >= 1


def test_the_table_is_one_component_per_amplitude() -> None:
    """The memory rule the threshold relies on: the table is half a state."""

    n_wires = 20
    state_bytes = (1 << n_wires) * 8
    table = statevector_operations._cx_sequence_permutation_index(
        (0, 1),
        (1, 2),
        n_wires,
        device=torch.device("cpu"),
        dtype=torch.complex64,
    )

    assert table.numel() * table.element_size() == state_bytes // 2


def test_two_identical_sequences_share_one_table() -> None:
    """The key is the content, so equal sequences cannot duplicate memory."""

    n_wires = 7
    controls, targets = programs(n_wires)["repeated_pair"]
    first = statevector_operations._cx_sequence_permutation_index(
        controls, targets, n_wires, device=torch.device("cpu"), dtype=torch.complex64
    )
    second = statevector_operations._cx_sequence_permutation_index(
        tuple(controls),
        tuple(targets),
        n_wires,
        device=torch.device("cpu"),
        dtype=torch.complex64,
    )

    assert second is first


# --------------------------------------------------------------------------- #
# Gradients and mutation
# --------------------------------------------------------------------------- #


def test_gradients_through_the_gather_match_the_per_gate_loop() -> None:
    """A gather is legitimate only if the backward pass reflects it."""

    n_wires = 5
    controls, targets = programs(n_wires)["overlapping_triples"]
    base = normalized_input(n_wires=n_wires, dtype=torch.complex128)
    weight = normalized_input(n_wires=n_wires, dtype=torch.complex128).detach()

    gathered_input = base.clone().requires_grad_(True)
    (
        gather_directly(gathered_input, controls, targets, n_wires) * weight
    ).real.sum().backward()

    looped_input = base.clone().requires_grad_(True)
    (
        per_gate_loop(looped_input, controls, targets, n_wires) * weight
    ).real.sum().backward()

    assert gathered_input.grad is not None
    assert torch.equal(gathered_input.grad, looped_input.grad)


def test_gradcheck_accepts_the_gather() -> None:
    """Analytic backward against central differences, on the kernel alone.

    ``gradcheck`` differentiates a real-valued function of a real tensor, so the
    amplitudes are handed in as interleaved real/imaginary pairs and read back
    the same way. That is the layout ``view_as_real`` produces, not a convention
    invented here.
    """

    n_wires = 3
    controls, targets = programs(n_wires)["overlapping_triples"]

    def forward(components: torch.Tensor) -> torch.Tensor:
        state = torch.view_as_complex(components.reshape(1, -1, 2))
        gathered = gather_directly(state, controls, targets, n_wires)
        return torch.view_as_real(gathered).reshape(-1)

    base = (
        torch.view_as_real(normalized_input(n_wires=n_wires, dtype=torch.complex128))
        .reshape(-1)
        .clone()
        .requires_grad_(True)
    )
    assert torch.autograd.gradcheck(forward, (base,), eps=1e-6, atol=1e-8, rtol=1e-5)


def test_a_mutated_circuit_rebuilds_its_program_around_the_gather() -> None:
    """The table is keyed by content, so a mutation cannot leave it stale."""

    n_wires = 6
    controls, targets = cx_chain(n_wires, 10)
    reference = normalized_input(n_wires=n_wires, dtype=torch.complex64)
    circuit = fq.Circuit(n_wires, device="cpu", dtype=torch.complex64)
    circuit._inputs = reference.clone().reshape(-1)
    for control, target in zip(controls, targets, strict=True):
        circuit.cx(control, target)

    once = per_gate_loop(reference, controls, targets, n_wires)
    assert torch.equal(circuit.state(refresh=True), once)

    for control, target in zip(controls, targets, strict=True):
        circuit.cx(control, target)

    twice = per_gate_loop(once, controls, targets, n_wires)
    assert torch.equal(circuit.state(refresh=True), twice)
    assert not torch.equal(once, twice)


def test_the_fuser_never_emits_a_wire_as_its_own_control() -> None:
    """The kernel assumes a CX actually flips a target; the planner must agree.

    ``control == target`` is not a CX, and the shipped per-gate kernel mis-handles
    it, so if fusion could ever produce one the gather would be compared against
    behaviour that is not defined. It cannot, and this pins that.
    """

    n_wires = 5
    controls, targets = cx_chain(n_wires, 12)
    circuit = fq.Circuit(n_wires, device="cpu", dtype=torch.complex64)
    for control, target in zip(controls, targets, strict=True):
        circuit.cx(control, target)

    program = statevector_operations._compile_statevector_program(
        circuit._instructions, n_wires, enable_triton_loop=False
    )
    sequences = [
        step
        for step in program
        if isinstance(step, statevector_operations._StatevectorCXSequenceStep)
    ]

    assert sequences, "the chain should fuse into a CX sequence"
    for step in sequences:
        for control, target in zip(step.controls, step.targets, strict=True):
            assert control != target


def test_every_distinct_control_target_pair_is_allowed() -> None:
    """Including the non-adjacent and the reversed pairs."""

    n_wires = 4
    state = normalized_input(n_wires=n_wires, dtype=torch.complex64)
    for control, target in itertools.product(range(n_wires), repeat=2):
        if control == target:
            continue
        controls, targets = (control,) * 3, (target,) * 3
        assert torch.equal(
            gather_directly(state, controls, targets, n_wires),
            per_gate_loop(state, controls, targets, n_wires),
        )
