"""Compare the bound Hamiltonian GQSP path with its locked source."""

from collections import Counter
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any, no_type_check

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.hamiltonian_gqsp import (
    HamiltonianGQSP,
    hamiltonian_gqsp_source_graph,
)
from flagquantum.algorithms.primitives.qubitization_walk import QubitizationWalk
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.phase.su2_rotation import su2_rotation_ir
from flagquantum.core.ir import Instruction

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@no_type_check
def _source(time: float, precision: float) -> Any:
    import cirq
    from attrs import frozen
    from qualtran import QBit, Register
    from qualtran.bloqs.basic_gates import Hadamard, XGate
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        SelectBlockEncoding as SourceBlock,
    )
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        HamiltonianSimulationByGQSP,
    )
    from qualtran.bloqs.multiplexers.select_base import SelectOracle
    from qualtran.bloqs.qubitization import QubitizationWalkOperator
    from qualtran.bloqs.state_preparation.prepare_base import PrepareOracle

    @frozen
    class Prepare(PrepareOracle):  # type: ignore[misc]
        @property
        def selection_registers(self):
            return (Register("selection", QBit()),)

        @property
        def junk_registers(self):
            return (Register("junk", QBit()),)

        @property
        def l1_norm_of_coeffs(self):
            return 2.0

        def build_composite_bloq(self, bb, **soqs):
            return {
                "selection": bb.add(Hadamard(), q=soqs["selection"]),
                "junk": bb.add(XGate(), q=soqs["junk"]),
            }

    @frozen
    class Select(SelectOracle):  # type: ignore[misc]
        @property
        def control_registers(self):
            return ()

        @property
        def selection_registers(self):
            return (Register("selection", QBit()),)

        @property
        def target_registers(self):
            return (Register("target", QBit()),)

        def decompose_from_registers(self, *, context, **quregs):
            yield cirq.CNOT(*quregs["selection"], *quregs["target"])

        def adjoint(self):
            return self

    return HamiltonianSimulationByGQSP(
        QubitizationWalkOperator(SourceBlock(Select(), Prepare())),
        t=time,
        precision=precision,
    )


@pytest.mark.parametrize("time,precision", [(0.75, 1e-5), (-0.75, 1e-7)])
def test_locked_coefficients_walk_calls_and_complete_state(
    time: float, precision: float
) -> None:
    import cirq
    import kickmix
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import SU2RotationGate
    from qualtran.bloqs.bookkeeping import Always
    from qualtran.cirq_interop import BloqAsCirqGate
    from qualtran.resource_counting import SympySymbolAllocator

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    original = _source(time, precision)

    def prepare(
        bound: Mapping[str, tuple[int, ...]],
        _workspace: tuple[int, ...],
        _bits: tuple[int, ...],
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        stages = (Instruction("h", bound["selection"]), Instruction("x", bound["junk"]))
        return stages[::-1] if inverse else stages

    def select(
        bound: Mapping[str, tuple[int, ...]],
        _workspace: tuple[int, ...],
        _bits: tuple[int, ...],
        _inverse: bool,
    ) -> tuple[Instruction, ...]:
        return (Instruction("cx", (*bound["selection"], *bound["target"])),)

    def controlled_select(
        value: int,
        bound: Mapping[str, tuple[int, ...]],
        _workspace: tuple[int, ...],
        _bits: tuple[int, ...],
        _inverse: bool,
    ) -> tuple[Instruction, ...]:
        assert value == 1
        return (
            Instruction(
                "ccx", (*bound["control"], *bound["selection"], *bound["target"])
            ),
        )

    selection_layout, target_layout = (("selection", 1),), (("target", 1),)
    prep = BlackBoxPrepare(selection_layout, (("junk", 1),), prepare, 2.0)
    walk = QubitizationWalk(
        SelectBlockEncoding(selection_layout, target_layout, select, prep)
    )
    calls: list[
        tuple[int, tuple[tuple[str, tuple[int, ...]], ...], tuple[int, ...], int, bool]
    ] = []
    walk_blocks: list[tuple[Instruction, ...]] = []

    def emit(
        signal: int,
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        value: int | None,
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        assert value in (0, 1)
        calls.append((signal, tuple(ports.items()), workspace, value, inverse))
        block = walk.make_on(
            selection={"selection": ports["selection"]},
            target=ports["target"],
            signal_junk=workspace,
            reflection_workspace=(),
            adjoint=inverse,
            controls=((signal,),),
            control_registers=(ControlRegister(1, (value,)),),
            controlled_select=controlled_select,
        ).instructions
        walk_blocks.append(block)
        return block

    model = HamiltonianGQSP(
        (*selection_layout, *target_layout), emit, prep, time, precision
    )
    assert model.degree == original.degree
    np.testing.assert_array_equal(model.coefficients, original.approx_cos)
    np.testing.assert_allclose(model.gqsp.q, original.gqsp.Q, atol=1e-13, rtol=0)
    assert model.gqsp.precision == original.gqsp.precision
    allocated_ports = (("selection", (1,)), ("target", (0,)))
    program = model.make_on(
        signal=2, target=(1, 0), prepare_junk=(3,), walk_workspace=(4,)
    )
    gates = program.instructions
    assert gates[:2] == (Instruction("h", (1,)), Instruction("x", (3,)))
    assert gates[-2:] == tuple(reversed(gates[:2]))
    assert calls == [
        (2, allocated_ports, (4,), value, inverse)
        for value, inverse in [(1, True)] * model.degree + [(0, False)] * model.degree
    ]
    theta, phi, initial_phase = original.gqsp._qsp_phases
    expected_body = list(
        su2_rotation_ir(2, float(theta[0]), float(phi[0]), float(initial_phase))
    )
    for block, t, f in zip(walk_blocks, theta[1:], phi[1:], strict=True):
        expected_body.extend(block)
        expected_body.extend(su2_rotation_ir(2, float(t), float(f), 0.0))
    assert gates[2:-2] == tuple(expected_body)
    original_layers = sorted(
        original.decompose_bloq().bloq_instances, key=lambda node: node.i
    )
    assert [type(node.bloq).__name__ for node in original_layers] == [
        "Allocate",
        "Always",
        "GeneralizedQSP",
        "Always",
        "Free",
    ]
    source_costs = original.build_call_graph(SympySymbolAllocator())
    source_walk = original.walk_operator
    labels = {
        Always(source_walk.prepare): "Always(PREPARE)",
        Always(source_walk.prepare.adjoint()): "Always(PREPARE†)",
        source_walk.controlled(CtrlSpec(cvs=0)): "ZeroControlledWalk",
        source_walk.adjoint().controlled(): "PositiveControlledWalk†",
    }
    assert hamiltonian_gqsp_source_graph(2.0, time, precision).leaf_counts == {
        "SU2Rotation" if isinstance(child, SU2RotationGate) else labels[child]: count
        for child, count in source_costs.items()
    }
    assert Counter(g.name for g in gates)["measure"] == 0
    source_matrix = cirq.unitary(BloqAsCirqGate(original))
    # The native physical order is target, selection, signal, outer junk,
    # reflection junk. The reference declares signal, selection, target.
    src_order = (2, 1, 0, 3, 4)
    columns = []
    for basis in range(8):
        source_ket = torch.zeros((8, 4), dtype=torch.complex128)
        source_ket[basis, 0] = 1
        initial = source_ket.reshape((2,) * 5).permute(src_order).reshape(-1)
        output = Circuit(5, inputs=initial, dtype=initial.dtype).append(program).state()
        amplitudes = output.reshape((2,) * 5).permute(src_order).reshape(8, 4)
        columns.append(amplitudes[:, 0].numpy())
        torch.testing.assert_close(
            amplitudes[:, 1:], torch.zeros_like(amplitudes[:, 1:]), atol=1e-13, rtol=0
        )
    actual = np.stack(columns, axis=-1)
    np.testing.assert_allclose(actual, source_matrix, atol=2e-12, rtol=0)
    ham = np.eye(2) + np.array([[0, 1], [1, 0]])
    np.testing.assert_allclose(
        actual[:2, :2], expm(-1j * ham * time), atol=8 * precision, rtol=0
    )


@pytest.mark.parametrize("symbolic", [False, True])
def test_hubbard_example_declares_the_five_exact_costs(symbolic: bool) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import SU2RotationGate
    from qualtran.bloqs.bookkeeping import Always
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        _hubbard_time_evolution_by_gqsp,
        _symbolic_hamsim_by_gqsp,
    )
    from qualtran.resource_counting import SympySymbolAllocator

    factory = _symbolic_hamsim_by_gqsp if symbolic else _hubbard_time_evolution_by_gqsp
    original = factory.make()
    walk = original.walk_operator
    keys = {
        Always(walk.prepare): "Always(PREPARE)",
        Always(walk.prepare.adjoint()): "Always(PREPARE†)",
        walk.controlled(CtrlSpec(cvs=0)): "ZeroControlledWalk",
        walk.adjoint().controlled(): "PositiveControlledWalk†",
    }
    counts = original.build_call_graph(SympySymbolAllocator())
    graph = hamiltonian_gqsp_source_graph(
        original.alpha, original.t, original.precision
    )
    assert graph.leaf_counts == {
        "SU2Rotation" if isinstance(child, SU2RotationGate) else keys[child]: count
        for child, count in counts.items()
    }


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("time", [2, 3, 5, 10])
@pytest.mark.parametrize("precision", [1e-5, 1e-7])
def test_source_random_unitary_matrix_grid(
    width: int, time: float, precision: float, pytestconfig: pytest.Config
) -> None:
    pytestconfig.addinivalue_line(
        "markers", "notebook: frozen Qualtran notebook fixture"
    )
    pytestconfig.addinivalue_line("markers", "slow: frozen Qualtran slow fixture")
    import cirq
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        random_qubitization_walk_operator,
    )
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        HamiltonianSimulationByGQSP,
    )
    from qualtran.bloqs.qsp.generalized_qsp import GeneralizedQSP as SourceGQSP
    from scipy.linalg import block_diag

    random = np.random.RandomState(42)
    reference_walk, _ = random_qubitization_walk_operator(1, 1, random_state=random)
    original = HamiltonianSimulationByGQSP(reference_walk, t=time, precision=precision)
    identity = np.eye(1 << width)
    matrix = identity

    def emit(
        signal: int,
        ports: Mapping[str, tuple[int, ...]],
        _workspace: tuple[int, ...],
        value: int | None,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        unitary = matrix.conj().T if adjoint else matrix
        if value is None:
            return (Instruction("matrix", ports["q"], matrix=unitary),)
        controlled = (
            block_diag(unitary, identity)
            if value == 0
            else block_diag(identity, unitary)
        )
        return (Instruction("matrix", (signal, *ports["q"]), matrix=controlled),)

    prep = BlackBoxPrepare((), (), lambda *_args: (), original.alpha)
    model = HamiltonianGQSP((("q", width),), emit, prep, time, precision)
    np.testing.assert_array_equal(model.coefficients, original.approx_cos)
    size = 1 << (width + 1)
    for _ in range(5):
        gate = MatrixGate.random(width, random_state=random)
        matrix = cirq.unitary(gate)
        source = SourceGQSP(
            gate,
            original.gqsp.P,
            original.gqsp.Q,
            negative_power=original.degree,
        )
        expected = cirq.unitary(source)
        program = model.make_on(signal=0, target=tuple(range(1, width + 1)))
        columns = [
            Circuit(width + 1, inputs=column, dtype=column.dtype)
            .append(program)
            .state()
            .reshape(-1)
            .numpy()
            for column in torch.eye(size, dtype=torch.complex128)
        ]
        np.testing.assert_allclose(
            np.stack(columns, axis=-1), expected, atol=8e-12, rtol=0
        )
