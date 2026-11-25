"""Complete source-file parity for Luis–Perina interim and amplified windows."""

from collections import Counter
from hashlib import sha1
from itertools import product
from math import sqrt
from pathlib import Path

import numpy as np
import pytest
import sympy
import torch

from flagquantum.algorithms.primitives.luis_perina_state import (
    LuisPerinaInterim,
    LuisPerinaState,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.measurement_checks import forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_controlled_unitary_synthesis import assert_source_stream

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (0, -1, True))
def test_historical_cost_requires_a_resolved_positive_width(width: int) -> None:
    with pytest.raises(ValueError, match="positive width"):
        LuisPerinaState(width).source_t_complexity()


@pytest.mark.parametrize(
    "n", [1, 3, 5, 7, 9, 11, 13, sympy.Symbol("n", positive=True, integer=True)]
)
def test_complete_original_historical_cost_assertions(n: int | sympy.Expr) -> None:
    from qualtran.bloqs.phase_estimation.lp_resource_state import (
        LPResourceState,
        LPRSInterimPrep,
    )

    actual = LuisPerinaState(n).source_t_complexity()
    original = LPResourceState(n).t_complexity()
    for field in ("t", "rotations", "clifford"):
        assert sympy.simplify(actual[field] - getattr(original, field)) == 0
    assert sympy.simplify(actual["t"] - (4 * n + 4)) == 0
    interim = legacy_gate_budget(LuisPerinaInterim(n).source_gate_cost())
    source_interim = LPRSInterimPrep(n).t_complexity()
    for field in ("t", "rotations", "clifford"):
        assert sympy.simplify(interim[field] - getattr(source_interim, field)) == 0
    if isinstance(n, int):
        assert actual == {
            "t": 4 * n + 4,
            "rotations": 6 * n + 6,
            "clifford": 26 * n + 33,
        }
        assert interim == {"t": 0, "rotations": 2 * n + 1, "clifford": 3 * n + 2}


@pytest.mark.parametrize("n", range(1, 14, 2))
def test_full_original_width_preparation_state_all_branch_certificate(n: int) -> None:
    from qualtran.bloqs.phase_estimation.lp_resource_state_test import (
        get_resource_state,
    )

    native = LuisPerinaState(n)
    work, slots = native.requirements()
    total = 1 + n + work
    phase = tuple(range(1, n + 1))
    workspace = tuple(range(n + 1, total))
    op = native.bind().make_on(
        selection=phase,
        junk=(),
        workspace=workspace,
        classical_bits=tuple(range(slots)),
    )
    gates = DynamicCircuit(total, dtype=torch.complex128).append(op).to_ir()
    # The certificate proves that every raw measurement history has the same
    # normalized operator. Numeric cleanup of one branch then proves cleanup
    # for the prepared input in every history, including entangled spectators.
    events = certify_erasure_histories(
        gates.instructions,
        n_wires=total,
        zero_wires=tuple(range(1, total)),
        check_clean_output=False,
    )
    assert events == slots == n + 1
    counts = Counter(g.name for g in gates.instructions)
    assert counts["t"] + counts["tdg"] == 4 * (n + 1)
    assert counts["measure"] == counts["reset"] == events
    spectator = (sqrt(0.3), 1j * sqrt(0.7))
    initial = {0: spectator[0], 1 << (total - 1): spectator[1]}
    actual = execute_sparse_register(gates, initial, seed=20261125 + n)
    values = get_resource_state(n)
    expected = {
        ((s << n) | word) << work: complex(coefficient * value)
        for s, coefficient in enumerate(spectator)
        for word, value in enumerate(values)
    }
    assert (
        max(
            abs(actual.get(word, 0) - expected.get(word, 0))
            for word in actual.keys() | expected.keys()
        )
        < 2e-12
    )
    # Ry/H cancellation is floating point. Bound the full residual probability,
    # including retained near-zero entries, rather than requiring bitwise zeros.
    assert (
        sum(
            abs(value) ** 2
            for word, value in actual.items()
            if word & ((1 << work) - 1)
        )
        < 1e-28
    )
    assert sum(abs(value) ** 2 for value in actual.values()) == pytest.approx(
        1, abs=2e-12
    )


@pytest.fixture(scope="module", autouse=True)
def frozen_complete_sources() -> None:
    from qualtran.bloqs.phase_estimation import lp_resource_state

    root = Path(lp_resource_state.__file__).parent
    for name, blob in (
        ("lp_resource_state.py", "55466291aba7d239ad4992b3aa3e09b44e33afe2"),
        ("lp_resource_state_test.py", "41b84e5cb8df63838ddbd843d2e4202bd1ad6c18"),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 9, 11, 13])
def test_complete_interim_stream_operands_both_directions_and_complex_state(
    n: int,
) -> None:
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPRSInterimPrep
    from qualtran.simulation.tensor import initialize_from_zero

    native = LuisPerinaInterim(n)
    original = LPRSInterimPrep(n)
    captured = import_register_emitter(original)
    m, anc = tuple(reversed(range(n))), n
    for adjoint in (False, True):
        op = native.make_on(m=m, anc=anc, adjoint=adjoint)
        source = captured({"m": m, "anc": (anc,)}, (), (), adjoint)
        assert_source_stream(
            CircuitIR(n + 1, op.instructions), CircuitIR(n + 1, source)
        )
    forward = native.make_on(m=tuple(range(n)), anc=anc)
    inverse = native.make_on(m=tuple(range(n)), anc=anc, adjoint=True)
    expected = initialize_from_zero(original).tensor_contract()
    circuit = Circuit(n + 1, dtype=torch.complex128).append(forward)
    np.testing.assert_allclose(
        circuit.state().detach().numpy().flatten(), expected, atol=1e-13, rtol=0
    )
    rng = np.random.default_rng(176 + n)
    coherent = rng.normal(size=2 ** (n + 1)) + 1j * rng.normal(size=2 ** (n + 1))
    coherent /= np.linalg.norm(coherent)
    circuit = Circuit(n + 1, inputs=torch.tensor(coherent), dtype=torch.complex128)
    circuit.append(forward)
    if n <= 5:
        np.testing.assert_allclose(
            circuit.state().detach().numpy().flatten(),
            original.tensor_contract() @ coherent,
            atol=1e-13,
            rtol=0,
        )
        np.testing.assert_allclose(
            get_unitary(
                CircuitIR(n + 1, forward.instructions), dtype=torch.complex128
            ).numpy(),
            original.tensor_contract(),
            atol=1e-13,
            rtol=0,
        )
    circuit.append(inverse)
    np.testing.assert_allclose(
        circuit.state().detach().numpy().flatten(), coherent, atol=1e-13, rtol=0
    )


@pytest.mark.parametrize("n", [1, 2, 3])
def test_complete_amplification_all_measurement_branches_and_exact_inverse(
    n: int,
) -> None:
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState

    native = LuisPerinaState(n)
    original = LPResourceState(n)
    work, slots = native.requirements()
    assert (work, slots) == (n + 4, n + 1)
    # Keep a coherent spectator ahead of the output and scratch partitions.
    phase = tuple(range(1, n + 1))
    workspace = tuple(range(n + 1, n + 1 + work))
    bound = native.bind()
    op = bound.make_on(
        selection=phase,
        junk=(),
        workspace=workspace,
        classical_bits=tuple(range(slots)),
    )
    circuit = DynamicCircuit(1 + n + work, dtype=torch.complex128).append(op)
    gates = circuit.to_ir().instructions
    assert CircuitIR.from_json(circuit.to_ir().to_json()).instructions == gates
    events = sum(g.name == "measure" for g in gates)
    assert events == slots
    assert Counter(g.name for g in gates) == {
        "ry": 3,
        "h": 6 * n + 9,
        "phase": 6 * n + 1,
        "cx": 12 * n + 6,
        "rz": 3,
        "cz": n + 3,
        "i": 2 * n + 4,
        "x": 4 * n + 12,
        "t": 2 * n + 2,
        "tdg": 2 * n + 2,
        "s": n + 1,
        "measure": n + 1,
        "reset": n + 1,
    }
    spectator = torch.tensor([sqrt(0.3), 1j * sqrt(0.7)], dtype=torch.complex128)
    logical = torch.zeros(2**n, dtype=torch.complex128)
    logical[0] = 1
    clean = torch.zeros(2**work, dtype=torch.complex128)
    clean[0] = 1
    initial = torch.kron(torch.kron(spectator, logical), clean)
    expected = torch.kron(
        torch.kron(spectator, torch.tensor(original.tensor_contract())), clean
    )
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / sqrt(2**events),
            atol=1e-12,
            rtol=0,
        )
    reverse = bound.emit(phase, (), workspace, tuple(range(slots)), adjoint=True)
    # Both directions have independent projective histories and restore every work bit.
    for history in product((0, 1), repeat=2 * events):
        torch.testing.assert_close(
            forced_branch(initial, (*gates, *reverse), history),
            initial / (2**events),
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "n", [1, 3, 5, 7, 9, 11, 13, sympy.Symbol("n", positive=True, integer=True)]
)
def test_complete_declared_resources_and_direct_graphs(n: int | sympy.Expr) -> None:
    from qualtran.bloqs.phase_estimation.lp_resource_state import (
        LPResourceState,
        LPRSInterimPrep,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    for native, original in (
        (LuisPerinaInterim(n), LPRSInterimPrep(n)),
        (LuisPerinaState(n), LPResourceState(n)),
    ):
        actual = native.source_gate_cost()
        expected = get_cost_value(original, QECGatesCost())
        for field in (
            "t",
            "toffoli",
            "cswap",
            "and_bloq",
            "clifford",
            "rotation",
            "measurement",
        ):
            assert (
                sympy.simplify(getattr(actual, field) - getattr(expected, field)) == 0
            )
    graph = LuisPerinaInterim(n).source_graph()
    assert graph.leaf_counts[("H", 0)] == n + 2
    direct = LuisPerinaState(n).source_graph().leaf_counts
    assert direct[("LuisPerinaInterim", n)] == 2
    assert direct[("LuisPerinaInterim†", n)] == 1
    assert direct[("ReflectionZero(i)", n + 2)] == 1
    assert sorted(direct.values()) == [1, 1, 1, 1, 2, 2, 2]
    assert (
        sum(count for (name, _), count in graph.leaf_counts.items() if name == "CRz")
        == n
    )


def test_original_factories_signatures_and_symbolic_physical_failure() -> None:
    from qualtran.bloqs.phase_estimation.lp_resource_state import (
        LPResourceState,
        LPRSInterimPrep,
    )

    for eps in (0.01, 0.1, sympy.Symbol("eps", positive=True)):
        assert (
            LuisPerinaState.from_standard_deviation_eps(eps).bitsize
            == LPResourceState.from_standard_deviation_eps(eps).bitsize
        )
    n = sympy.Symbol("n", positive=True, integer=True)
    for native, source in (
        (LuisPerinaState(n), LPResourceState(n)),
        (LuisPerinaInterim(n), LPRSInterimPrep(n)),
    ):
        assert [(r.name, r.bitsize, r.side.name) for r in native.signature] == [
            (r.name, r.bitsize, r.side.name) for r in source.signature
        ]
    with pytest.raises(ValueError, match="concrete positive"):
        LuisPerinaState(n).bind()
    with pytest.raises(ValueError, match="concrete positive"):
        LuisPerinaInterim(n).make_on(m=(0,), anc=1)
    with pytest.raises(ZeroDivisionError):
        LuisPerinaState.from_standard_deviation_eps(0)


def test_binding_rejects_overlaps_and_incomplete_owned_storage() -> None:
    with pytest.raises(ValueError):
        LuisPerinaInterim(2).make_on(m=(0, 0), anc=1)
    with pytest.raises(ValueError):
        LuisPerinaInterim(2).make_on(m=(0,), anc=1)
    with pytest.raises(ValueError, match="work and slots"):
        LuisPerinaState(1).bind().emit((0,), (), (), ())


def test_complete_direct_call_mapping_and_interim_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran import Adjoint
    from qualtran.bloqs.basic_gates import CZ, Ry, XGate
    from qualtran.bloqs.phase_estimation.lp_resource_state import (
        LPResourceState,
        LPRSInterimPrep,
    )
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )
    from qualtran.resource_counting import SympySymbolAllocator

    from flagquantum.algorithms.primitives._block_binding import _BoundBlockOperation

    n = 2
    original = LPResourceState(n)
    declared = original.build_call_graph(SympySymbolAllocator())
    mapped: dict[tuple[str, int | float | sympy.Expr], int | sympy.Expr] = {}
    for child, count in declared.items():
        if isinstance(child, LPRSInterimPrep):
            key = ("LuisPerinaInterim", child.bitsize)
        elif isinstance(child, Adjoint):
            assert isinstance(child.subbloq, LPRSInterimPrep)
            key = ("LuisPerinaInterim†", child.subbloq.bitsize)
        elif isinstance(child, Ry):
            key = ("Ry", child.angle)
        elif isinstance(child, ReflectionUsingPrepare):
            assert child.global_phase == 1j
            key = (
                "ReflectionZero(i)",
                sum(r.total_bits() for r in child.selection_registers),
            )
        elif isinstance(child, XGate):
            key = ("X", 0)
        else:
            assert isinstance(child, CZ)
            key = ("CZ", 0)
        mapped[key] = count
    assert LuisPerinaState(n).source_graph().leaf_counts == mapped
    native = LuisPerinaState(n)
    work, slots = native.requirements()
    workspace = tuple(range(n, n + work))
    expected = native.bind().emit(tuple(range(n)), (), workspace, tuple(range(slots)))
    capture = import_register_emitter(LPRSInterimPrep(n))

    def source_interim(
        self: LuisPerinaInterim, *, m: tuple[int, ...], anc: int, adjoint: bool = False
    ) -> _BoundBlockOperation:
        assert self.bitsize == n
        stream = capture({"m": m, "anc": (anc,)}, (), (), adjoint)
        return _BoundBlockOperation(stream, max((*m, anc)) + 1)

    monkeypatch.setattr(LuisPerinaInterim, "make_on", source_interim)
    actual = native.bind().emit(tuple(range(n)), (), workspace, tuple(range(slots)))
    assert actual == expected


@pytest.mark.parametrize("n", [1, 2])
def test_original_nonreal_reflection_adjoint_defect_is_retained_as_evidence(
    n: int,
) -> None:
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState

    original = LPResourceState(n)
    forward = original.tensor_contract()
    declared_inverse = original.adjoint().tensor_contract()
    np.testing.assert_allclose(declared_inverse, -forward.conj(), atol=1e-13, rtol=0)
    assert np.max(np.abs(declared_inverse - forward.conj())) > 1
