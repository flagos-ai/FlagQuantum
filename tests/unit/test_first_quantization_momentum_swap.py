"""Source graph and dynamic execution checks for three-coordinate exchange."""

from collections import Counter
from dataclasses import asdict

import pytest
import torch

from flagquantum.algorithms.first_quantization import (
    MultiplexedCSwap3D,
    source_multiplexed_cswap_3d_cost,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import ValidationError
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
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
