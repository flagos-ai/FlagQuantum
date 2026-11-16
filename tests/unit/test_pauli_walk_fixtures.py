"""Compare uniform and randomized Pauli fixtures against frozen Qualtran."""

from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.algorithms.primitives.qubitization_walk import QubitizationWalk
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler.registers.lookup.bounded_pauli_select import (
    source_bounded_pauli_select_cost,
)
from flagquantum.compiler.registers.lookup.unary_iteration import (
    unary_iteration_workspace,
)
from flagquantum.compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_requirements,
)
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
    uniform_preparation_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.pauli_walk_fixtures import (
    RandomPauliFixture,
    fixture_hamiltonian,
    uniform_prepare,
    uniform_ring_walk,
    uniform_ring_words,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def check_walk_channel(
    walk: QubitizationWalk,
    source: Any,
    target_width: int,
    *,
    prepare_work: int = 0,
    prepare_slots: int = 0,
) -> None:
    width = walk.block_encoding.signal_state.selection_width
    logical = width + target_width
    selector_work = unary_iteration_workspace(width, None)
    reflector_work, reflector_slots = prepare_reflection_requirements(
        width, controlled=False
    )
    scratch = tuple(
        range(logical, logical + max(selector_work, reflector_work, prepare_work + 1))
    )
    selection = {"selection": tuple(range(width))}
    target = tuple(range(width, logical))
    values = torch.arange(1, (1 << logical) + 1, dtype=torch.float64)
    amplitudes = torch.complex(values, 3 - values)
    amplitudes /= amplitudes.norm()
    vacuum = torch.zeros(1 << len(scratch), dtype=amplitudes.dtype)
    vacuum[0] = 1
    initial = torch.kron(amplitudes, vacuum)

    for adjoint in (False, True):
        bound = walk.make_on(
            selection=selection,
            target=target,
            select_workspace=scratch[:selector_work],
            select_classical_bits=tuple(range(selector_work)),
            prepare_workspace=scratch[1 : 1 + prepare_work],
            prepare_classical_bits=tuple(range(prepare_slots)),
            reflection_workspace=scratch[:reflector_work],
            reflection_classical_bits=tuple(range(reflector_slots)),
            adjoint=adjoint,
        )
        circuit = DynamicCircuit(logical + len(scratch)).append(bound)
        instructions = circuit.to_ir().instructions
        from_source = source.adjoint() if adjoint else source
        tensor = torch.as_tensor(
            from_source.tensor_contract().copy(), dtype=amplitudes.dtype
        )
        expected = torch.kron(tensor @ amplitudes, vacuum)
        measured = sum(g.name == "measure" for g in instructions)
        histories = (
            product((0, 1), repeat=measured)
            if measured <= 4
            else (
                (0,) * measured,
                (1,) * measured,
                tuple(i % 2 for i in range(measured)),
            )
        )
        for history in histories:
            actual = forced_branch(initial, instructions, history)
            torch.testing.assert_close(
                actual, expected * 2 ** (-measured / 2), atol=8e-12, rtol=0
            )


@pytest.mark.parametrize("sites", [2, 3, 4])
def test_uniform_ring_terms_model_and_channel(sites: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.qubitization_walk_test import (
        get_uniform_pauli_qubitized_walk,
    )

    hamiltonian, source = get_uniform_pauli_qubitized_walk(sites)
    native = uniform_ring_walk(sites)
    words = uniform_ring_words(sites)
    assert words == tuple(
        str(pauli)[1:] for pauli in source.block_encoding.select.select_unitaries
    )
    assert native.alpha == source.sum_of_lcu_coefficients == len(hamiltonian) == sites
    np.testing.assert_array_equal(
        fixture_hamiltonian(words, (1 + 0j,) * sites).matrix().numpy(),
        hamiltonian.matrix(),
    )
    width, work, slots = uniform_preparation_requirements(sites)
    selector = SelectPauliLCU(width, sites, words)
    model = source_bounded_pauli_select_cost(
        selector._instruction(
            tuple(range(width)), tuple(range(width, width + sites)), ()
        )
    )
    prep = source_uniform_preparation_cost(sites)
    assert_source_model(native.resource_model(model, prep, prep), source)

    address = tuple(range(width))
    target = tuple(range(width, width + sites))
    select_work = tuple(
        range(width + sites, width + sites + unary_iteration_workspace(width, None))
    )
    readouts = tuple(range(len(select_work)))
    assert isinstance(native.block_encoding, SelectBlockEncoding)
    native_select = native.block_encoding.emit(address, target, select_work, readouts)
    source_select = reference_arithmetic_gates(
        source.block_encoding.select,
        {"selection": address, "target": target},
        select_work,
        dict(zip(select_work, readouts, strict=True)),
        cirq_style=True,
        keep_pauli_leaves=True,
    )
    assert_source(native_select, source_select)
    check_walk_channel(native, source, sites, prepare_work=work, prepare_slots=slots)


def test_uniform_ring_rejects_repeated_single_site() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.for_testing.qubitization_walk_test import (
        PrepareUniformSuperpositionTest,
        get_uniform_pauli_qubitized_walk,
    )

    with pytest.raises(AssertionError):
        get_uniform_pauli_qubitized_walk(1)
    with pytest.raises(ValueError, match="not Hermitian"):
        uniform_ring_walk(1)
    controlled = PrepareUniformSuperpositionTest(3, 1)
    assert not controlled.junk_registers and tuple(
        r.name for r in controlled.signature
    ) == ("selection",)
    with pytest.raises(ValueError, match="ctrl not in qubit_regs"):
        controlled.decompose_bloq()

    wrapper = uniform_prepare(3, 1)
    assert wrapper.selection_layout == (("selection", 2),) and not wrapper.junk_layout
    assert wrapper.normalization == controlled.l1_norm_of_coeffs == 0
    with pytest.raises(ValueError, match="no ctrl register"):
        wrapper.emit((0, 1), (), (), ())


@pytest.mark.parametrize("width", [1, 2])
def test_random_pauli_seed_and_bit_order(width: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        random_qubitization_walk_operator,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source_rng, native_rng = np.random.RandomState(42), np.random.RandomState(42)
    source, hamiltonian = random_qubitization_walk_operator(
        width, 2, random_state=source_rng
    )
    native = RandomPauliFixture.random(width, 2, native_rng)
    prep = source.block_encoding.prepare
    selection = source.block_encoding.select
    np.testing.assert_array_equal(native.prepare.unitary(), prep.U.matrix)
    np.testing.assert_array_equal(native.alphas, prep.alphas)
    assert native.words == tuple(str(pauli)[1:] for pauli in selection.select_unitaries)
    assert native_rng.randint(1 << 20) == source_rng.randint(1 << 20)
    assert native.walk().alpha == source.sum_of_lcu_coefficients == 1.0
    column = native.prepare.unitary()[:, 0]
    source_select = selection.tensor_contract().reshape(1 << width, 4, 1 << width, 4)
    projected = np.einsum("a,abij,i->bj", column.conj(), source_select, column)
    np.testing.assert_allclose(
        native.hamiltonian().matrix().numpy(), hamiltonian.matrix(), atol=1e-14, rtol=0
    )
    difference = np.max(np.abs(projected - hamiltonian.matrix()))
    assert difference == pytest.approx(
        0.0 if width == 1 else 0.382220521869599, abs=1e-14
    )
    with pytest.raises(DecomposeNotImplementedError, match="MatrixGate"):
        get_cost_value(source, QECGatesCost())
    check_walk_channel(native.walk(), source, 2)


@pytest.mark.parametrize("control", [None, 0, 1])
def test_random_select_controlled_source_matrix(control: int | None) -> None:
    pytest.importorskip("qualtran")
    from attrs import evolve
    from qualtran.bloqs.for_testing.random_select_and_prepare import (
        TestPauliSelectOracle,
    )

    source = TestPauliSelectOracle.random(2, 2, random_state=np.random.RandomState(31))
    native = RandomPauliFixture.random(2, 2, np.random.RandomState(42))
    words = tuple(str(word)[1:] for word in source.select_unitaries)
    fixture = RandomPauliFixture(native.prepare, words, 2)
    source = evolve(source, control_val=control)
    offset = int(control is not None)
    selector = fixture.selector(control)
    selection = (offset, offset + 1)
    target = (offset + 2, offset + 3)
    values = torch.arange(1, (1 << (4 + offset)) + 1, dtype=torch.float64)
    initial = torch.complex(values, 1 - values)
    initial /= initial.norm()
    from flagquantum import Circuit

    circuit = Circuit(4 + offset, inputs=initial, dtype=initial.dtype)
    circuit.append(
        selector.make_on(
            selection=selection[::-1],
            target=target,
            control=(0,) if offset else (),
        )
    )
    reference = torch.as_tensor(source.tensor_contract().copy(), dtype=initial.dtype)
    torch.testing.assert_close(
        circuit.state().flatten(), reference @ initial, atol=2e-14, rtol=0
    )
