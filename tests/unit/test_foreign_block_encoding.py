"""Actual external blocks are snapshotted before native LCU composition."""

from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.ecosystem.cirq._block_encoding import import_block_encoding
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def original_models(
    source: Any,
) -> tuple[ComponentCost, ComponentCost, ComponentCost, ComponentCost]:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    controlled = source.controlled()
    models = tuple(
        ComponentCost(
            "Independent locked declaration",
            int(get_cost_value(s, QubitCount())),
            GateCost(**get_cost_value(s, QECGatesCost()).asdict()),
        )
        for s in (source, source.adjoint(), controlled, controlled.adjoint())
    )
    return models[0], models[1], models[2], models[3]


@pytest.mark.parametrize("label", ("h", "t"))
@pytest.mark.parametrize("auxiliary", (False, True))
def test_full_original_matrices_for_all_four_interfaces(
    label: str, auxiliary: bool
) -> None:
    from attrs import evolve
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.unitary import Unitary

    source = evolve(
        Unitary(Hadamard() if label == "h" else TGate()),
        ancilla_bitsize=int(auxiliary),
        resource_bitsize=int(auxiliary),
        alpha=2.0,
        epsilon=0.01,
    )
    block = import_block_encoding(source, models=original_models(source))
    assert (block.alpha, block.epsilon) == (source.alpha, source.epsilon)
    for controlled, adjoint in product((False, True), repeat=2):
        original = source.controlled() if controlled else source
        if adjoint:
            original = original.adjoint()
        assert_source_model(
            block.resource_model(controlled=controlled, adjoint=adjoint), original
        )
        offset = int(controlled)
        total = offset + 1 + 2 * int(auxiliary)
        work, bits = block.requirements(controlled=controlled)
        op = block.make_on(
            system=(offset,),
            ancilla=(offset + 1,) if auxiliary else (),
            resource=(offset + 2,) if auxiliary else (),
            control=(0,) if controlled else (),
            workspace=tuple(range(total, total + work)),
            classical_bits=tuple(range(bits)),
            adjoint=adjoint,
        )
        assert (work, bits) == (0, 0)
        actual = get_unitary(Circuit(total).append(op).to_ir(), dtype=torch.complex128)
        torch.testing.assert_close(
            actual,
            torch.as_tensor(original.tensor_contract(), dtype=torch.complex128),
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.parametrize("controlled,adjoint", tuple(product((False, True), repeat=2)))
def test_foreign_children_replace_existing_lcu_factories(
    controlled: bool, adjoint: bool
) -> None:
    from tests.unit.test_linear_combination_block_encoding import fixture

    native, base, source, previous, ports, work, bits, costs, _, options = fixture(
        2, 1, True, controlled, adjoint
    )
    from attrs import evolve
    from qualtran.bloqs.basic_gates import Hadamard, TGate
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination

    children = tuple(
        evolve(child, U=gate)
        for child, gate in zip(
            base._block_encodings, (Hadamard(), TGate()), strict=True
        )
    )
    base = LinearCombination(children, native.coefficients, native.coefficient_bits)
    source = evolve(base, is_controlled=controlled)
    if adjoint:
        source = source.adjoint()
    imported = tuple(
        import_block_encoding(child, models=original_models(child))
        for child in children
    )
    replacement = replace(native, block_encodings=imported)
    op = replacement.make_on(tuple(child.bind_child for child in imported), **options)
    circuit = Circuit(previous.required_qubits).append(op)
    logical = sum(len(port) for port in ports.values())
    reserve = circuit.n_qubits - logical
    values = torch.arange(1 << logical, dtype=torch.float64)
    state = torch.complex(values + 1, values % 5 - 2)
    state /= state.norm()
    expected = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128) @ state
    clean = torch.zeros(1 << reserve, dtype=torch.complex128)
    clean[0] = 1
    gates = circuit.to_ir().instructions
    events = sum(g.name == "measure" for g in gates)
    for outcomes in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(torch.kron(state, clean), gates, outcomes),
            torch.kron(expected, clean) / (2 ** (events / 2)),
            atol=3e-12,
            rtol=0,
        )


def test_symbolic_and_nonidentity_signal_objects_fail_before_capture() -> None:
    from types import SimpleNamespace

    from qualtran.bloqs.block_encoding.sparse_matrix import (
        _sparse_matrix_symb_block_encoding,
    )

    model = ComponentCost("Not executed", 1)
    with pytest.raises(ValueError, match="concrete nonnegative widths"):
        import_block_encoding(_sparse_matrix_symb_block_encoding(), models=(model,) * 4)
    with pytest.raises(ValueError, match="identity signal state"):
        import_block_encoding(
            SimpleNamespace(
                system_bitsize=1,
                ancilla_bitsize=1,
                resource_bitsize=0,
                signal_state=SimpleNamespace(prepare=object()),
            ),
            models=(model,) * 4,
        )
