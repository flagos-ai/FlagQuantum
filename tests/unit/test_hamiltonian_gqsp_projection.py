"""Run the complete locked random Hamiltonian projection grid through append."""

from collections.abc import Mapping
from typing import Any

import cirq
import numpy as np
import pytest
import torch
from scipy.linalg import block_diag, expm

from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.hamiltonian_gqsp import HamiltonianGQSP
from flagquantum.algorithms.primitives.qubitization_walk import QubitizationWalk
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.core.ir import Instruction

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _bind_walk(
    source: Any, width: int, time: float, precision: float
) -> HamiltonianGQSP:
    prepare_matrix = np.asarray(source.prepare.U.matrix)
    select_matrix = block_diag(
        *(
            cirq.unitary(pauli)
            for pauli in source.block_encoding.select.select_unitaries
        )
    )
    controlled_matrix = block_diag(np.eye(len(select_matrix)), select_matrix)

    def prepare(
        ports: Mapping[str, tuple[int, ...]],
        _work: tuple[int, ...],
        _bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        matrix = prepare_matrix.conj().T if adjoint else prepare_matrix
        return (Instruction("matrix", ports["selection"], matrix=matrix),)

    def select(
        ports: Mapping[str, tuple[int, ...]],
        _work: tuple[int, ...],
        _bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        matrix = select_matrix.conj().T if adjoint else select_matrix
        return (
            Instruction(
                "matrix", (*ports["selection"], *ports["target"]), matrix=matrix
            ),
        )

    def controlled_select(
        value: int,
        ports: Mapping[str, tuple[int, ...]],
        _work: tuple[int, ...],
        _bits: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert value == 1
        matrix = controlled_matrix.conj().T if adjoint else controlled_matrix
        return (
            Instruction(
                "matrix",
                (*ports["control"], *ports["selection"], *ports["target"]),
                matrix=matrix,
            ),
        )

    selection, target = (("selection", 1),), (("target", width),)
    prepared = BlackBoxPrepare(selection, (), prepare, 1.0)
    walk = QubitizationWalk(SelectBlockEncoding(selection, target, select, prepared))

    def emit(
        signal: int,
        ports: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        value: int | None,
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        assert value in (0, 1) and not workspace
        return walk.make_on(
            selection={"selection": ports["selection"]},
            target=ports["target"],
            reflection_workspace=(),
            controls=((signal,),),
            control_registers=(ControlRegister(1, (value,)),),
            controlled_select=controlled_select,
            adjoint=adjoint,
        ).instructions

    return HamiltonianGQSP((*selection, *target), emit, prepared, time, precision)


@pytest.mark.parametrize("width", [1, 2])
@pytest.mark.parametrize("time", [2, 5])
@pytest.mark.parametrize("precision", [1e-5, 1e-7, 1e-9])
def test_random_hamiltonian_complete_unitary_and_projected_evolution(
    width: int, time: float, precision: float, pytestconfig: pytest.Config
) -> None:
    pytestconfig.addinivalue_line(
        "markers", "notebook: frozen Qualtran notebook fixture"
    )
    pytestconfig.addinivalue_line("markers", "slow: frozen Qualtran slow fixture")
    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        random_qubitization_walk_operator,
    )
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        HamiltonianSimulationByGQSP,
    )
    from qualtran.cirq_interop import BloqAsCirqGate

    random = np.random.RandomState(42)
    for _ in range(5):
        source_walk, hamiltonian = random_qubitization_walk_operator(
            1, width, random_state=random
        )
        source = HamiltonianSimulationByGQSP(source_walk, t=time, precision=precision)
        native = _bind_walk(source_walk, width, time, precision)
        n = width + 2
        size = 1 << width
        bound = native.make_on(signal=0, target=tuple(range(1, n)))
        columns = [
            Circuit(n, inputs=column, dtype=column.dtype)
            .append(bound)
            .state()
            .reshape(-1)
            .numpy()
            for column in torch.eye(1 << n, dtype=torch.complex128)
        ]
        actual = np.stack(columns, axis=1)
        np.testing.assert_allclose(
            actual, cirq.unitary(BloqAsCirqGate(source)), atol=2e-11, rtol=0
        )
        np.testing.assert_allclose(
            actual[:size, :size],
            expm(-1j * hamiltonian.matrix() * time),
            atol=1e-4,
            rtol=0,
        )
