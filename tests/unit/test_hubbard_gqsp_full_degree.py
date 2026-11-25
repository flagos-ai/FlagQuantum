# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source: Qualtran v0.7.0 bcb17bb25360ed68ce3bc084645db3e929799389.

"""Exact original Hubbard GQSP degree and its nonzero freed-junk counterexample."""

from collections import Counter
from collections.abc import Callable
from math import sqrt
from typing import Any
from unittest.mock import patch

import pytest
import torch

from flagquantum.algorithms.primitives import hubbard_hamiltonian_gqsp as hubbard_module
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.algorithms.primitives.generalized_qsp import UnitaryEmitter
from flagquantum.algorithms.primitives.hamiltonian_gqsp import (
    HamiltonianGQSP,
    hamiltonian_gqsp_source_graph,
)
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.compiler.registers.layout.partition import (
    RegisterLayout,
    partition_register,
)
from flagquantum.compiler.registers.phase.su2_rotation import su2_rotation_ir
from flagquantum.compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import certified_and_monomial_map
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_sources_and_native_reference() -> None:
    from hashlib import sha1
    from importlib.metadata import version
    from pathlib import Path

    import kickmix
    import qualtran

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and callable(kickmix.Simulator)
    root = Path(qualtran.__file__).parent / "bloqs"
    for name, blob in (
        ("bookkeeping/free.py", "8a0c73db71e1d8ffa31b2c7a15d246f02f8f4f63"),
        (
            "hamiltonian_simulation/hamiltonian_simulation_by_gqsp.py",
            "21c3ee41f15e45391411ff665b70b7d3e8dec878",
        ),
        ("qsp/generalized_qsp.py", "5848dcb4785697563fca6173e499015e504b233b"),
        (
            "chemistry/hubbard_model/qubitization/prepare_hubbard.py",
            "cfdbf20b0baf5ef1f5fa5a7e21826d7f51f5aec5",
        ),
        (
            "chemistry/hubbard_model/qubitization/select_hubbard.py",
            "6d912f2ae1f63730cc296cfc35f68eb9ccbabb84",
        ),
        (
            "reflections/reflection_using_prepare.py",
            "4a8467fde126bd596be69286d88c198726a4557a",
        ),
        (
            "phase_estimation/text_book_qpe.py",
            "9349ca32908fbba054cab61f095852f880960b0f",
        ),
        (
            "phase_estimation/text_book_qpe_test.py",
            "39a0d095da29abccf3f9c6d7bf8d48d54a8a8f59",
        ),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def monomial_columns(
    gates: tuple[Instruction, ...], logical: int, work: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    basis = tuple(
        tuple((value >> (logical - 1 - i)) & 1 for i in range(logical)) + (0,) * work
        for value in range(1 << logical)
    )
    outputs, phases, events = certified_and_monomial_map(gates, basis)
    assert all(not any(bits[logical:]) for bits in outputs)
    permutation = torch.tensor(
        [
            sum(bit << (logical - 1 - i) for i, bit in enumerate(bits[:logical]))
            for bits in outputs
        ]
    )
    assert len(set(permutation.tolist())) == 1 << logical
    torch.testing.assert_close(
        phases.abs(), torch.ones_like(phases.real), atol=2e-14, rtol=0
    )
    return permutation, phases, events


def source_emitter_ir(source: Any, adjoint: bool = False) -> tuple[CircuitIR, int]:
    emitter = import_register_emitter(source)
    work, slots = emitter.requirements()
    ports: dict[str, tuple[int, ...]] = {}
    logical = 0
    for register in source.signature:
        width = int(register.total_bits())
        ports[register.name] = tuple(range(logical, logical + width))
        logical += width
    gates = emitter(
        ports, tuple(range(logical, logical + work)), tuple(range(slots)), adjoint
    )
    return CircuitIR(logical + work, gates), logical


def apply_preparation(state: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return (
        (matrix @ state.permute(1, 3, 0, 2).reshape(1024, -1))
        .reshape(256, 4, 2, 256)
        .permute(2, 0, 3, 1)
    )


def apply_walk(
    state: torch.Tensor,
    reflection: torch.Tensor,
    permutation: torch.Tensor,
    phases: torch.Tensor,
    value: int,
    adjoint: bool,
) -> torch.Tensor:
    result = state.clone()
    active = state[value]
    if adjoint:
        reflected = (reflection @ active.reshape(256, -1)).reshape(65536, 4)
        result[value] = (phases.conj()[:, None] * reflected[permutation]).reshape_as(
            active
        )
    else:
        selected = torch.empty_like(active.reshape(65536, 4))
        selected[permutation] = phases[:, None] * active.reshape(65536, 4)
        result[value] = (reflection @ selected.reshape(256, -1)).reshape_as(active)
    return result


def test_original_degree127_source_channels_and_freed_outer_junk(
    record_property: Callable[[str, object], None],
) -> None:
    pytest.importorskip("cirq")
    import cirq
    from qualtran.bloqs.bookkeeping import Always, Free
    from qualtran.bloqs.chemistry.hubbard_model.qubitization import (
        get_walk_operator_for_hubbard_model,
    )
    from qualtran.bloqs.hamiltonian_simulation.hamiltonian_simulation_by_gqsp import (
        HamiltonianSimulationByGQSP,
    )
    from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE
    from qualtran.testing import assert_valid_bloq_decomposition

    from tests.unit.test_textbook_phase_estimation import test_frozen_complete_sources

    test_frozen_complete_sources()
    original = HamiltonianSimulationByGQSP(
        get_walk_operator_for_hubbard_model(2, 2, 1, 1), t=5, precision=1e-7
    )
    assert original.degree == 127 and original.alpha == 20
    frees = [
        instance.bloq
        for instance, _, _ in original.decompose_bloq().iter_bloqsoqs()
        if isinstance(instance.bloq, Free)
    ]
    assert sum(free.dtype.num_qubits for free in frees) == 2
    assert all(not free.dirty for free in frees)
    assert_valid_bloq_decomposition(TextbookQPE(original, RectangularWindowState(3)))
    prepare = PrepareHubbard(2, 1, 1)
    ports = partition_register(tuple(range(10)), prepare.layout)
    native_gates = prepare._emit(ports, (10, 11), (0, 1), False)
    preparation_matrix = logical_operator(CircuitIR(12, native_gates), 10)
    source_ir, logical = source_emitter_ir(Always(original.walk_operator.prepare))
    assert logical == 10
    source_preparation = logical_operator(source_ir, logical)
    torch.testing.assert_close(
        preparation_matrix, source_preparation, atol=8e-12, rtol=0
    )
    source_inverse_ir, logical = source_emitter_ir(
        Always(original.walk_operator.prepare), True
    )
    source_inverse = logical_operator(source_inverse_ir, logical)
    torch.testing.assert_close(
        source_inverse, source_preparation.mH, atol=8e-12, rtol=0
    )
    native_inverse = logical_operator(
        CircuitIR(12, prepare._emit(ports, (10, 11), (0, 1), True)), 10
    )
    torch.testing.assert_close(
        native_inverse, preparation_matrix.mH, atol=8e-12, rtol=0
    )
    full_reflection_matrix = 2 * preparation_matrix[:, :4] @ preparation_matrix[
        :, :4
    ].mH - torch.eye(1024, dtype=torch.complex128)
    source_full_reflection = 2 * source_preparation[:, :4] @ source_preparation[
        :, :4
    ].mH - torch.eye(1024, dtype=torch.complex128)
    clean_reflection = full_reflection_matrix[:, ::4].clone()
    clean_reflection[::4] = 0
    assert float(clean_reflection.abs().max()) < 8e-12
    reflection_matrix = full_reflection_matrix[::4, ::4]
    source_reflection = source_full_reflection[::4, ::4]
    select = SelectHubbard(2)
    work, slots = select.requirements()
    native_select = select._emit(
        partition_register(tuple(range(16)), select.layout),
        tuple(range(16, 16 + work)),
        tuple(range(slots)),
    )
    permutation, phases, select_events = monomial_columns(native_select, 16, work)
    source_ir, logical = source_emitter_ir(original.walk_operator.block_encoding.select)
    source_permutation, source_phases, source_events = monomial_columns(
        source_ir.instructions, logical, source_ir.n_wires - logical
    )
    assert select_events == source_events == 15
    assert torch.equal(permutation, source_permutation)
    torch.testing.assert_close(phases, source_phases, atol=8e-12, rtol=0)

    # Every actual controlled SELECT direction preserves its inactive sector.
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(1 << 16)
    inverse_phases = phases[inverse_permutation].conj()
    for value in (0, 1):
        controlled = SelectHubbard(2, controlled=True, control_value=value)
        w, s = controlled.requirements()
        for adjoint in (False, True):
            gates = controlled._emit(
                partition_register(tuple(range(17)), controlled.layout),
                tuple(range(17, 17 + w)),
                tuple(range(s)),
                adjoint=adjoint,
            )
            actual_perm, actual_phases, _ = monomial_columns(gates, 17, w)
            expected_perm = torch.arange(1 << 17)
            expected_phases = torch.ones(1 << 17, dtype=torch.complex128)
            sector = slice(value * (1 << 16), (value + 1) * (1 << 16))
            expected_perm[sector] = (
                inverse_permutation if adjoint else permutation
            ) + value * (1 << 16)
            expected_phases[sector] = inverse_phases if adjoint else phases
            assert torch.equal(actual_perm, expected_perm)
            torch.testing.assert_close(
                actual_phases, expected_phases, atol=8e-12, rtol=0
            )

    owners: list[HamiltonianGQSP] = []

    def observe(
        layout: RegisterLayout,
        walk: UnitaryEmitter,
        prepared: BlackBoxPrepare,
        time: float,
        precision: float,
    ) -> HamiltonianGQSP:
        owner = HamiltonianGQSP(layout, walk, prepared, time, precision)
        owners.append(owner)
        return owner

    operation = hubbard_module.HubbardHamiltonianGQSP(prepare, 5, 1e-7)
    with patch.object(hubbard_module, "HamiltonianGQSP", side_effect=observe):
        bound = operation.make_on(
            signal=0,
            target=tuple(range(1, 17)),
            prepare_temp=(17, 18),
            reflection_temp=(19, 20),
            workspace=tuple(range(21, 28)),
            classical_bits=tuple(range(7)),
        )
    assert (
        tuple(DynamicCircuit(28).append(bound).to_ir().instructions)
        == bound.instructions
    )
    owner = owners[0]
    actual_ports = partition_register(tuple(range(1, 17)), owner.walk_layout)
    reflection_ports = {**actual_ports, "temp": (19, 20)}
    reflection_pre = prepare._emit(reflection_ports, (21, 22), (0, 1), False)
    reflection_inverse = prepare._emit(reflection_ports, (21, 22), (0, 1), True)
    center = prepare_reflection_ir(
        tuple(range(1, 9)),
        tuple(range(9, 16)),
        tuple(range(7)),
        control=0,
        global_phase=-1,
    )
    center_perm, center_phases, center_events = monomial_columns(center, 9, 7)
    assert torch.equal(center_perm, torch.arange(512))
    center_expected = torch.ones(512, dtype=torch.complex128)
    center_expected[257:] = -1
    torch.testing.assert_close(center_phases, center_expected, atol=2e-14, rtol=0)
    assert center_events == 7
    full_reflection = prepare_reflection_ir(
        tuple(range(1, 9)),
        tuple(range(21, 28)),
        tuple(range(7)),
        prepare=reflection_pre,
        unprepare=reflection_inverse,
        control=0,
        global_phase=-1,
    )
    controlled = SelectHubbard(2, controlled=True)
    cw, cs = controlled.requirements()
    walk_blocks = {}
    for adjoint in (False, True):
        select_block = controlled._emit(
            {"control": (0,), **actual_ports},
            tuple(range(21, 21 + cw)),
            tuple(range(cs)),
            adjoint=adjoint,
        )
        for value in (0, 1):
            middle = (
                (*full_reflection, *select_block)
                if adjoint
                else (*select_block, *full_reflection)
            )
            flips = () if value else (Instruction("x", (0,)),)
            expected_walk = (*flips, *middle, *flips)
            assert (
                owner.walk(0, actual_ports, tuple(range(19, 28)), value, adjoint)
                == expected_walk
            )
            walk_blocks[value, adjoint] = expected_walk
    theta, phi, lambd = owner.gqsp._phases
    qubits = cirq.LineQubit.range(17)
    pytest.importorskip("numpy")
    import numpy as np

    cursor = 1
    quregs = {}
    for name, width in owner.walk_layout:
        quregs[name] = np.array(qubits[cursor : cursor + width])
        cursor += width
    stages = tuple(
        cirq.flatten_op_tree(
            original.gqsp.decompose_from_registers(
                context=cirq.DecompositionContext(cirq.SimpleQubitManager()),
                signal=np.array([qubits[0]]),
                **quregs,
            )
        )
    )
    assert len(stages) == 509
    state = torch.zeros((2, 256, 256, 4), dtype=torch.complex128)
    state[0, 0, 0, 0] = 1 / sqrt(3)
    state[0, 0, 129, 0] = torch.tensor(1j / sqrt(3), dtype=torch.complex128)
    state[0, 0, 36, 0] = -1 / sqrt(3)
    actual = apply_preparation(state, preparation_matrix)
    expected = apply_preparation(state, source_preparation)
    stream = list(owner.prepare.emit(tuple(range(1, 9)), (17, 18), (21, 22), (0, 1)))
    max_rotation_error = 0.0
    for index, rotation in enumerate(original.gqsp.signal_rotations):
        assert stages[2 * index] == rotation.on(qubits[0])
        if index:
            adjoint = index <= 127
            value = 1 if adjoint else 0
            source_walk = (
                original.walk_operator.adjoint() if adjoint else original.walk_operator
            )
            assert stages[2 * index - 1] == source_walk.on_registers(
                **quregs
            ).controlled_by(qubits[0], control_values=[value])
            stream.extend(walk_blocks[value, adjoint])
            actual = apply_walk(
                actual, reflection_matrix, permutation, phases, value, adjoint
            )
            expected = apply_walk(
                expected,
                source_reflection,
                source_permutation,
                source_phases,
                value,
                adjoint,
            )
        rotation_ir = su2_rotation_ir(
            0, theta[index], phi[index], lambd if index == 0 else 0
        )
        stream.extend(rotation_ir)
        native_matrix = logical_operator(CircuitIR(1, rotation_ir), 1)
        source_matrix = torch.as_tensor(
            rotation.rotation_matrix, dtype=torch.complex128
        )
        max_rotation_error = max(
            max_rotation_error, float((native_matrix - source_matrix).abs().max())
        )
        torch.testing.assert_close(native_matrix, source_matrix, atol=8e-12, rtol=0)
        actual = (native_matrix @ actual.reshape(2, -1)).reshape_as(actual)
        expected = (source_matrix @ expected.reshape(2, -1)).reshape_as(expected)
    stream.extend(
        owner.prepare.emit(tuple(range(1, 9)), (17, 18), (21, 22), (0, 1), adjoint=True)
    )
    assert tuple(stream) == bound.instructions
    actual = apply_preparation(actual, native_inverse)
    expected = apply_preparation(expected, source_inverse)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
    assert float(actual.abs().square().sum()) == pytest.approx(1, abs=2e-11)
    leaked = float(actual[:, :, :, 1:].abs().square().sum())
    assert leaked == pytest.approx(0.3289881219704135, abs=2e-11)
    with pytest.raises(ValueError):
        operation.make_on(
            signal=0,
            target=tuple(range(1, 17)),
            prepare_temp=(),
            reflection_temp=(19, 20),
            workspace=tuple(range(21, 28)),
            classical_bits=tuple(range(7)),
        )
    from qualtran.resource_counting import (
        QECGatesCost,
        QubitCount,
        SympySymbolAllocator,
        get_cost_value,
    )

    from tests.unit.test_hubbard_gqsp_notebook import _key

    native_graph = hamiltonian_gqsp_source_graph(20, 5, 1e-7)
    root = "HamiltonianSimulationByGQSP"
    assert native_graph.direct_calls == {
        (root, _key(child, original)): weight
        for child, weight in original.build_call_graph(SympySymbolAllocator()).items()
    }
    record_property(
        "source_qec_model", get_cost_value(original, QECGatesCost()).asdict()
    )
    record_property("source_model_peak", get_cost_value(original, QubitCount()))
    record_property("degree", 127)
    record_property("full_complex_error", float((actual - expected).abs().max()))
    record_property("max_signal_rotation_error", max_rotation_error)
    record_property("freed_outer_junk_probability", leaked)
    record_property(
        "zero_junk_projection_norm", float(actual[:, :, :, 0].abs().square().sum())
    )
    record_property(
        "all_raw_histories_exponent",
        Counter(g.name for g in bound.instructions)["measure"],
    )
    record_property("source_accepted_decomposition", True)
    record_property(
        "native_emitted_instruction_inventory",
        dict(Counter(g.name for g in bound.instructions)),
    )
