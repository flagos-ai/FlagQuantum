"""Source graph and dynamic execution checks for three-coordinate exchange."""

from collections import Counter
from dataclasses import asdict
from itertools import product

import pytest
import torch

from flagquantum.algorithms.first_quantization import (
    ControlledMultiplexedCSwap3D,
    MultiplexedCSwap3D,
    momentum_exchange_signature,
    source_controlled_multiplexed_cswap_3d_cost,
    source_multiplexed_cswap_3d_cost,
)
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import ValidationError
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_first_quantization_resource_model import (
    assert_first_quantization_signature,
)
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_relative_controlled_swap import (
    assert_controlled_exchange_branches,
    assert_source_dependencies,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("eta", "bits", "permuted"),
    [(2, 1, False), (3, 1, False), (2, 2, True), (5, 3, True)],
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_pinned_momentum_exchange_graph(
    eta: int, bits: int, permuted: bool, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        MultiplexedCSwap3D as SourceSwap,
    )
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    selection_width = (eta - 1).bit_length()
    data_width = selection_width + 3 * bits * (eta + 1)
    total = data_width + selection_width - 1
    wires = iter(reversed(range(total)) if permuted else range(total))

    def take(count: int) -> tuple[int, ...]:
        return tuple(next(wires) for _ in range(count))

    selection = take(selection_width)
    targets = tuple(tuple(take(bits) for _ in range(3)) for _ in range(eta))
    junk = tuple(take(bits) for _ in range(3))
    work = take(selection_width - 1)
    classical = tuple(range(40, 40 + len(work)))
    args = dict(
        selection=selection,
        targets=targets,
        junk=junk,
        workspace=work,
        classical_bits=classical,
    )
    op = MultiplexedCSwap3D.make_on(**args, adjoint=adjoint)
    circuit = DynamicCircuit(total).append(op)
    actual = circuit.to_ir().instructions

    base = SourceSwap(bits, eta)
    source = base.adjoint() if adjoint else base
    signature = momentum_exchange_signature(eta, bits)
    assert_first_quantization_signature(
        signature.adjoint() if adjoint else signature, source.signature
    )
    ports = {
        "sel": selection,
        "targets": tuple(wire for row in targets for word in row for wire in word),
        "junk": tuple(wire for word in junk for wire in word),
    }
    child_calls = 0

    def leaf(
        bloq: object, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        nonlocal child_calls
        child = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if not isinstance(child, MultiplexedCSwap):
            return None
        child_calls += 1
        assert isinstance(bloq, Adjoint) == adjoint
        assert child.target_bitsize == 3 * bits
        assert inputs == {
            "sel": selection,
            "targets": ports["targets"],
            "output": ports["junk"],
        }
        assert all(inputs[name] == wires for name, wires in desired.items())
        return (
            reference_arithmetic_gates(
                bloq,
                inputs,
                work,
                dict(zip(work, classical, strict=True)),
                cirq_style=not adjoint,
                preserve_components=True,
            ),
            inputs,
        )

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports and child_calls == 1
    assert_source_dependencies(actual, expected)
    cost = get_cost_value(source, QECGatesCost())
    counts = Counter(gate.name for gate in actual)
    report = estimate_ftqc_resources(source_multiplexed_cswap_3d_cost(eta, bits))
    assert {
        name: n for name, n in asdict(report.logical.gates).items() if n
    } == cost.asdict()
    assert (
        report.logical.peak_logical_qubits
        == get_cost_value(source, QubitCount())
        == total
    )
    assert counts["t"] + counts["tdg"] == cost.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == counts["reset"] == cost.measurement
    if not permuted:
        inverse = MultiplexedCSwap3D.make_on(**args, adjoint=not adjoint).emit()
        assert_controlled_exchange_branches(actual, expected, inverse, data_width, work)


def test_append_exchanges_nonzero_junk_and_clears_work() -> None:
    state = torch.zeros((2, 1 << 15), dtype=torch.complex128)
    expected = torch.zeros_like(state)

    def index(select: int, particle: tuple[int, int, int], junk: int) -> int:
        value = select
        for word in (*particle, junk):
            value = (value << 3) | word
        return value << 1

    state[0, index(1, (1, 2, 3), 7)] = 1j
    expected[0, index(1, (1, 7, 3), 2)] = 1j
    state[1, index(2, (4, 6, 0), 5)] = -1
    expected[1, index(2, (4, 6, 5), 0)] = -1
    op = MultiplexedCSwap3D.make_on(
        selection=(0, 1),
        targets=(((2,), (3,), (4,)), ((5,), (6,), (7,)), ((8,), (9,), (10,))),
        junk=((11,), (12,), (13,)),
        workspace=(14,),
        classical_bits=(9,),
    )
    # Use two independent bound states through the actual append and runtime path.
    for initial, final in zip(state, expected, strict=True):
        circuit = DynamicCircuit(15, inputs=initial, dtype=torch.complex128)
        assert circuit.append(op) is circuit
        result = run_dynamic(circuit, shots=4, seed=7, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states, final.expand(4, -1), atol=2e-12, rtol=0
        )


@pytest.mark.ftqc_reference
def test_source_single_particle_has_no_zero_selection_decomposition() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_and_prepare import (
        MultiplexedCSwap3D as SourceSwap,
    )
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap

    (child,) = (
        node.bloq
        for node in SourceSwap(1, 1).decompose_bloq().bloq_instances
        if isinstance(node.bloq, MultiplexedCSwap)
    )
    with pytest.raises(
        NotImplementedError, match="Selection register must not be empty"
    ):
        child.decompose_bloq()
    with pytest.raises(ValidationError, match="at least two particles"):
        MultiplexedCSwap3D.make_on(
            selection=(), targets=(((0,), (1,), (2,)),), junk=((3,), (4,), (5,))
        ).emit()
    with pytest.raises(ValueError, match="at least two particles"):
        source_multiplexed_cswap_3d_cost(1, 1)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("eta", "p", "n", "cvs", "permuted"),
    [
        (2, 1, 2, (0,), False),
        (2, 1, 2, (0, 1), False),
        (3, 1, 2, (1,), True),
        (5, 2, 3, (1, 0), True),
    ],
)
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("explicit_activation", [False, True])
def test_pinned_controlled_momentum_exchange_graph(
    eta, p, n, cvs, permuted, adjoint, explicit_activation
):
    pytest.importorskip("qualtran")
    from qualtran import Adjoint
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare import (
        ControlledMultiplexedCSwap3D as SourceSwap,
    )
    from qualtran.bloqs.swap_network.multiplexed_cswap import MultiplexedCSwap
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    address_width = (eta - 1).bit_length()
    data_width = len(cvs) + address_width + 3 * (eta * p + n)
    total = data_width + address_width + len(cvs) - 1
    wires = iter(reversed(range(total)) if permuted else range(total))

    def take(size):
        return tuple(next(wires) for _ in range(size))

    controls = take(len(cvs))
    selection = take(address_width)
    targets = tuple(tuple(take(p) for _ in range(3)) for _ in range(eta))
    junk = tuple(take(n) for _ in range(3))
    workspace = take(total - data_width)
    slots = tuple(range(40, 40 + len(workspace)))
    args = dict(
        controls=controls,
        cvs=cvs,
        selection=selection,
        targets=targets,
        junk=junk,
        workspace=workspace,
        classical_bits=slots,
    )
    values = cvs if explicit_activation else None
    op = ControlledMultiplexedCSwap3D.make_on(
        **args, adjoint=adjoint, control_values=values
    )
    ir = DynamicCircuit(total).append(op).to_ir()
    actual = ir.instructions
    base = SourceSwap(p, n, eta, cvs)
    source = base.adjoint() if adjoint else base
    ports = {
        "ctrl": controls,
        "sel": selection,
        "targets": tuple(wire for row in targets for word in row for wire in word),
        "junk": tuple(wire for word in junk for wire in word),
    }
    child_calls = 0

    def leaf(
        bloq: object, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        nonlocal child_calls
        child = bloq.subbloq if isinstance(bloq, Adjoint) else bloq
        if not isinstance(child, MultiplexedCSwap):
            return None
        child_calls += 1
        assert isinstance(bloq, Adjoint) == adjoint
        assert child.target_bitsize == 3 * p
        assert inputs == {
            "ctrl": controls,
            "sel": selection,
            "targets": ports["targets"],
            "output": tuple(wire for word in junk for wire in word[:p]),
        }
        assert all(inputs[name] == bound for name, bound in desired.items())
        return (
            reference_arithmetic_gates(
                bloq,
                inputs,
                workspace,
                dict(zip(workspace, slots, strict=True)),
                cirq_style=not adjoint,
                preserve_components=True,
            ),
            inputs,
        )

    expected, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports and child_calls == 1
    flips = tuple(
        Instruction("x", (wire,))
        for wire, value in zip(controls, cvs, strict=True)
        if explicit_activation and value == 0
    )
    expected = (*flips, *expected, *flips)
    assert_source_dependencies(actual, expected)
    tails = {wire for word in junk for wire in word[p:]}
    assert not any(tails.intersection(gate.wires) for gate in actual)
    cost = get_cost_value(source, QECGatesCost())
    report = estimate_ftqc_resources(op.resource_model(), emitted_ir=ir)
    expected_counts = cost.asdict()
    if flips:
        expected_counts["clifford"] = expected_counts.get("clifford", 0) + 2 * len(
            flips
        )
    assert {
        key: value for key, value in asdict(report.logical.gates).items() if value
    } == expected_counts
    assert (
        report.logical.peak_logical_qubits
        == get_cost_value(source, QubitCount())
        == total
    )
    counts = Counter(gate.name for gate in actual)
    assert report.emitted is not None
    assert report.emitted.unconditional.get("x", 0) == counts["x"]
    assert counts["t"] + counts["tdg"] == cost.total_t_count(ts_per_cswap=7)
    assert counts["measure"] == counts["reset"] == cost.measurement
    if not permuted:
        inverse = ControlledMultiplexedCSwap3D.make_on(
            **args, adjoint=not adjoint, control_values=values
        ).emit()
        assert_controlled_exchange_branches(
            actual, expected, inverse, data_width, workspace
        )


def test_projectile_swap_executes_all_one_activation_and_preserves_suffixes() -> None:
    op = ControlledMultiplexedCSwap3D.make_on(
        controls=(0,),
        cvs=0,
        selection=(1,),
        targets=(((2,), (3,), (4,)), ((5,), (6,), (7,))),
        junk=((8, 9), (10, 11), (12, 13)),
        workspace=(14,),
        classical_bits=(6,),
    )
    # Control zero leaves the state untouched. Control one exchanges the
    # addressed 101 with the projectile prefixes 010, retaining tails 110.
    state = torch.zeros(1 << 15, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    inactive = int("010001010111000", 2)
    active = int("110001010111000", 2)
    exchanged = int("110000101101100", 2)
    state[inactive] = 0.6
    state[active] = 0.8j
    expected[inactive] = 0.6
    expected[exchanged] = 0.8j
    circuit = DynamicCircuit(15, inputs=state, dtype=torch.complex128).append(op)
    result = run_dynamic(circuit, shots=4, seed=7, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states, expected.expand(4, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize("pattern", [(0,), (0, 1), (1, 0), (0, 0)])
def test_explicit_momentum_activation_matches_complex_permutation(
    pattern: tuple[int, ...],
) -> None:
    control_count = len(pattern)
    width = 13 + 2 * control_count
    controls = tuple(range(control_count))
    workspace = tuple(range(width - control_count, width))
    binding = dict(
        controls=controls,
        cvs=(1,) * control_count,
        selection=(control_count,),
        targets=tuple(
            tuple((control_count + 1 + 3 * row + axis,) for axis in range(3))
            for row in range(2)
        ),
        junk=tuple(
            (control_count + 7 + 2 * axis, control_count + 8 + 2 * axis)
            for axis in range(3)
        ),
        workspace=workspace,
        classical_bits=tuple(range(14, 14 + control_count)),
    )
    base = ControlledMultiplexedCSwap3D.make_on(**binding).emit()
    op = ControlledMultiplexedCSwap3D.make_on(**binding, control_values=pattern)
    circuit = DynamicCircuit(width).append(op)
    gates = circuit.to_ir().instructions
    added = Counter(gate.name for gate in gates) - Counter(gate.name for gate in base)
    assert added == {"x": 2 * pattern.count(0)}

    state = torch.zeros(1 << width, dtype=torch.complex128)
    ideal = torch.zeros_like(state)
    for control in product((0, 1), repeat=control_count):
        for address in (0, 1):
            rows = ["000", "101"]
            junk = ["01", "11", "00"]
            prefix = "".join(map(str, control)) + str(address)
            work = "0" * control_count
            before = prefix + "".join(rows + junk) + work
            if control == pattern:
                old = rows[address]
                rows[address] = "".join(word[0] for word in junk)
                junk = [bit + word[1] for bit, word in zip(old, junk, strict=True)]
            after = prefix + "".join(rows + junk) + work
            amplitude = complex(1 + address, (-1) ** sum(control))
            state[int(before, 2)] = amplitude
            ideal[int(after, 2)] = amplitude

    inverse = ControlledMultiplexedCSwap3D.make_on(
        **binding, control_values=pattern, adjoint=True
    )
    undo = circuit.append(inverse).to_ir().instructions[len(gates) :]
    for history in product((0, 1), repeat=control_count):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            ideal / 2 ** (control_count / 2),
            atol=2e-12,
            rtol=0,
        )
        torch.testing.assert_close(
            forced_branch(ideal, undo, history),
            state / 2 ** (control_count / 2),
            atol=2e-12,
            rtol=0,
        )


def test_projectile_swap_rejects_invalid_idle_wires() -> None:
    from dataclasses import replace

    op = ControlledMultiplexedCSwap3D.make_on(
        controls=(0,),
        cvs=(1,),
        selection=(1,),
        targets=(((2,), (3,), (4,)), ((5,), (6,), (7,))),
        junk=((8, 9), (10, 11), (12, 15)),
        workspace=(14,),
        classical_bits=(6,),
    )
    with pytest.raises(ValidationError, match="exceeds circuit width"):
        DynamicCircuit(15).append(op)
    with pytest.raises(ValueError, match="distinct"):
        replace(op, junk=((8, 0), (10, 11), (12, 13))).emit()
    with pytest.raises(ValidationError, match="binary label"):
        replace(op, cvs=(2,)).emit()
    for values in ((), (0, 1), (2,), (False,)):
        invalid = replace(op, control_values=values)
        with pytest.raises(ValidationError, match="binary activation value"):
            invalid.emit()
        with pytest.raises(ValidationError, match="binary activation value"):
            invalid.resource_model()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("eta", "p", "n", "cvs", "exception", "message"),
    [
        (1, 1, 2, (1,), NotImplementedError, "Selection register must not be empty"),
        (2, 1, 2, (), ValueError, "iteration dimensions are 0"),
        (2, 2, 1, (1,), ValueError, "register dtypes are not consistent"),
    ],
)
def test_projectile_swap_source_invalid_shapes(eta, p, n, cvs, exception, message):
    pytest.importorskip("qualtran")
    from qualtran import BloqError
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile.select_and_prepare import (
        ControlledMultiplexedCSwap3D as SourceSwap,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    with pytest.raises((exception, BloqError), match=message):
        get_cost_value(SourceSwap(p, n, eta, cvs), QECGatesCost())
    with pytest.raises(ValueError, match="Controlled momentum exchange"):
        source_controlled_multiplexed_cswap_3d_cost(eta, p, n, len(cvs))
