"""Does every reachable executor apply a matrix the caller attached?

`Circuit.gate(..., matrix=...)` is accepted for every declared opcode, and
`gate_matrix` returns that matrix in preference to the operator the name
denotes. Three executors nevertheless selected a kernel from the *name* before
reading the operand, so `gate("x", [0], matrix=M)` applied `X`, not `M`, and
`gate("z", [0], matrix=M)` applied `Z`. The program said one thing and three
executors did another.

Each test here hands an opcode a seeded unitary of its own size whose image of
`|0...0>` is far from the image of the operator the name denotes, and compares
the public route against the operand's own action. A kernel that substitutes the
named operator fails by a large margin rather than by a rounding difference, so
the fixture cannot pass by accident.

The negative controls matter as much as the positive ones. Every guard is
`instruction.matrix is None`, so a change that deleted the fast paths instead of
guarding them would leave the operand tests green and the fast paths gone; the
tests marked `no_matrix` assert that the name-keyed kernels are still reached
when no operand is attached.
"""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, canonical_opcode
from flagquantum.runtime.dynamic import run_dynamic
from flagquantum.runtime.dynamic.circuit import DynamicCircuit

pytestmark = pytest.mark.unit

#: A tolerance wide enough for a float32 statevector and far narrower than the
#: distance between an operand and the named operator it is confused with.
_ATOL = 1.0e-5


def _seeded_unitary(dimension: int, seed: int) -> torch.Tensor:
    """A deterministic unitary whose first column is not a basis vector."""

    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw)
    return (q * torch.sign(torch.diagonal(r))).to(torch.complex128)


def _operand(name: str) -> tuple[torch.Tensor, dict[str, float]]:
    """The operand and the parameters for one declared non-channel opcode."""

    schema = OPERATOR_SCHEMAS[name]
    arity = schema.arity
    matrix = _seeded_unitary(2**arity, seed=arity * 100 + len(name))
    params = dict.fromkeys(schema.parameters, 0.3)
    return matrix, params


def _action(matrix: torch.Tensor) -> torch.Tensor:
    """What the operand does to the all-zero computational basis state."""

    basis = torch.zeros(matrix.shape[0], dtype=torch.complex128)
    basis[0] = 1
    return matrix @ basis


def _declared_unitary_opcodes() -> list[str]:
    return sorted(
        name for name, schema in OPERATOR_SCHEMAS.items() if not schema.channel
    )


def _amplitudes(state: torch.Tensor, arity: int, n_qubits: int) -> torch.Tensor:
    """The amplitudes the operand addresses, with the other qubits left at `|0>`.

    The opcode acts on the lowest-numbered `arity` qubits. Wire 0 is the highest
    order index bit, so the sub-state the operand addresses sits at the offsets
    that are multiples of `2 ** (n_qubits - arity)`.
    """

    flat = torch.as_tensor(state).to(torch.complex128).reshape(-1)
    stride = 2 ** (n_qubits - arity)
    return torch.stack([flat[index * stride] for index in range(2**arity)])


def _assert_applies_its_operand(
    got: torch.Tensor, matrix: torch.Tensor, *, arity: int, n_qubits: int
) -> None:
    """Compare the addressed sub-state with the operand's action, up to scale.

    Both sides are normalised because a route may carry the operand's action
    under a global scalar -- the product-state route does, since the rest of the
    register supplies an overall amplitude -- and a global phase is not something
    the IR records. Normalisation cannot hide the defect under test: the named
    operator's action is a different ray, not a multiple of the operand's.
    """

    expected = _action(matrix)
    observed = _amplitudes(got, arity, n_qubits)
    observed = observed / torch.linalg.vector_norm(observed)
    assert torch.allclose(observed, expected, atol=_ATOL), (
        f"the executor applied {[f'{complex(value):+.4f}' for value in observed]} "
        f"but the carried operand applies "
        f"{[f'{complex(value):+.4f}' for value in expected]}"
    )


@pytest.mark.parametrize("name", _declared_unitary_opcodes())
def test_the_statevector_executor_applies_a_carried_matrix(name: str) -> None:
    """The default executor reads the operand before it reads the name."""

    matrix, params = _operand(name)
    arity = OPERATOR_SCHEMAS[name].arity
    circuit = fq.Circuit(arity)
    circuit.gate(name, list(range(arity)), params=params, matrix=matrix)

    _assert_applies_its_operand(circuit.state(), matrix, arity=arity, n_qubits=arity)


@pytest.mark.parametrize("name", _declared_unitary_opcodes())
def test_the_dynamic_executor_applies_a_carried_matrix(name: str) -> None:
    """The dynamic executor reads the operand before it reads the name."""

    matrix, params = _operand(name)
    arity = OPERATOR_SCHEMAS[name].arity
    circuit = DynamicCircuit(arity)
    circuit.gate(name, list(range(arity)), params=params, matrix=matrix)

    result = run_dynamic(circuit, shots=1, seed=7)

    _assert_applies_its_operand(
        result.final_states[0], matrix, arity=arity, n_qubits=arity
    )


@pytest.mark.parametrize("mode", ["density_matrix", "mps"])
@pytest.mark.parametrize("name", _declared_unitary_opcodes())
def test_the_other_run_modes_apply_a_carried_matrix(name: str, mode: str) -> None:
    """The density-matrix and MPS executors read the operand too.

    Their state is not exposed as amplitudes, so the comparison is on
    probabilities, which is phase-free and still separates the operand from the
    named operator by a wide margin. The two modes are different executors from
    the default one, which is why they are not covered by the statevector test.
    """

    matrix, params = _operand(name)
    arity = OPERATOR_SCHEMAS[name].arity
    circuit = fq.Circuit(arity)
    circuit.gate(name, list(range(arity)), params=params, matrix=matrix)

    result = fq.run(
        circuit,
        options=fq.ExecutionOptions(mode=mode, shots=None),
        outputs=[fq.probabilities()],
    )

    observed = torch.as_tensor(result.probabilities[0]).to(torch.float64)
    expected = _action(matrix).abs() ** 2
    assert torch.allclose(observed, expected, atol=_ATOL), (
        f"mode {mode!r} produces {observed.tolist()} but the carried operand "
        f"produces {expected.tolist()}"
    )


#: The preparation register. The gate under test sits on qubit 0 and touches
#: nothing else, so the rest of the register only supplies an overall amplitude
#: and the addressed sub-state is the operand's action up to that scalar.
_PRODUCT_STATE_QUBITS = 16
_PRODUCT_STATE_ROUNDS = 2


def _product_state_circuit(name: str, matrix: torch.Tensor | None) -> fq.Circuit:
    """A sixteen-qubit circuit deep enough to select the product-state route.

    `product_state_execution_is_beneficial` declines the route below sixteen
    qubits and on shallow programs, so a smaller or flatter circuit would test
    the dense loop and never reach the guarded branch.
    """

    circuit = fq.Circuit(_PRODUCT_STATE_QUBITS)
    for _ in range(_PRODUCT_STATE_ROUNDS):
        for qubit in range(2, _PRODUCT_STATE_QUBITS, 2):
            circuit.h(qubit)
        circuit.cx(2, 3)
    if matrix is None:
        circuit.gate(name, [0], params={})
    else:
        circuit.gate(name, [0], matrix=matrix)
    return circuit


def _run_counting_fast_paths(
    name: str, matrix: torch.Tensor | None
) -> tuple[torch.Tensor, int, int]:
    """Run one circuit, counting name-keyed kernel calls for the tested opcode.

    The preparation uses the same opcode in some cases -- `h` above all -- so the
    count alone does not say what happened to the tested instruction. The caller
    compares the two counts of the same circuit with and without the operand,
    where the only difference is the instruction under test.
    """

    from flagquantum.simulation.statevector import local as statevector_local
    from flagquantum.simulation.statevector import product_state

    counts = {"execute": 0, "fast": 0}
    original_execute = statevector_local.execute_product_state_program
    original_permutation = product_state._apply_fixed_permutation
    original_clifford = product_state._apply_fixed_clifford_gate

    def counted_execute(*args, **kwargs):
        counts["execute"] += 1
        return original_execute(*args, **kwargs)

    def counted(original):
        def wrapper(state, gate_name, *args, **kwargs):
            if canonical_opcode(gate_name) == canonical_opcode(name):
                counts["fast"] += 1
            return original(state, gate_name, *args, **kwargs)

        return wrapper

    statevector_local.execute_product_state_program = counted_execute
    product_state._apply_fixed_permutation = counted(original_permutation)
    product_state._apply_fixed_clifford_gate = counted(original_clifford)
    try:
        state = _product_state_circuit(name, matrix).state()
    finally:
        statevector_local.execute_product_state_program = original_execute
        product_state._apply_fixed_permutation = original_permutation
        product_state._apply_fixed_clifford_gate = original_clifford
    return state, counts["execute"], counts["fast"]


@pytest.mark.parametrize("name", ["x", "y", "z", "h", "s", "sdg"])
def test_the_product_state_route_applies_a_carried_matrix(name: str) -> None:
    """The product-state route must not substitute the named operator.

    This route is reachable from the public entry point, not only from a unit
    test: above sixteen qubits `Circuit.state()` hands the program to
    `execute_product_state_program`, which then dispatched on the name alone.
    """

    matrix, _ = _operand(name)
    _, plain_execute, plain_fast = _run_counting_fast_paths(name, None)
    bearing_state, bearing_execute, bearing_fast = _run_counting_fast_paths(
        name, matrix
    )

    assert plain_execute == 1 and bearing_execute == 1, (
        "the product-state route was not selected, so this test does not reach "
        "the branch it is written for"
    )
    assert bearing_fast == plain_fast - 1, (
        f"the same circuit reached a name-keyed kernel {plain_fast} times for a "
        f"plain {name!r} and {bearing_fast} times with an operand attached; the "
        "difference has to be exactly the one instruction under test, and a "
        "matrix-bearing instruction must reach none"
    )

    _assert_applies_its_operand(
        bearing_state, matrix, arity=1, n_qubits=_PRODUCT_STATE_QUBITS
    )


@pytest.mark.parametrize("name", ["x", "y", "z", "h", "s", "sdg"])
def test_the_product_state_fast_paths_are_guarded_not_deleted(name: str) -> None:
    """The name-keyed kernels still run when no operand is attached.

    A change that deleted the fast paths instead of guarding them would satisfy
    the operand test above and cost the route the whole reason it exists.
    """

    _, execute, fast = _run_counting_fast_paths(name, None)

    assert execute == 1, "the product-state route was not selected"
    assert fast >= 1, (
        f"the product-state route no longer reaches a name-keyed kernel for a "
        f"plain {name!r}, so the fast path was deleted rather than guarded"
    )


def test_the_diagonal_kernel_is_still_used_without_a_matrix() -> None:
    """The dynamic executor keeps its diagonal kernel for a plain `z`."""

    from flagquantum.runtime.dynamic import execution as dynamic_execution

    selected: list[str] = []
    original = dynamic_execution._apply_diagonal_matrix

    def counted(*args, **kwargs):
        selected.append("diagonal")
        return original(*args, **kwargs)

    dynamic_execution._apply_diagonal_matrix = counted
    try:
        plain = DynamicCircuit(1)
        plain.z(0)
        run_dynamic(plain, shots=1, seed=5)
        assert selected == [
            "diagonal"
        ], "a plain z no longer reaches the diagonal kernel"

        selected.clear()
        bearing = DynamicCircuit(1)
        bearing.gate("z", [0], matrix=_seeded_unitary(2, seed=41))
        run_dynamic(bearing, shots=1, seed=5)
        assert selected == [], (
            "a matrix-bearing z still reaches the diagonal kernel, which reads "
            "only the diagonal entries of the operand"
        )
    finally:
        dynamic_execution._apply_diagonal_matrix = original
