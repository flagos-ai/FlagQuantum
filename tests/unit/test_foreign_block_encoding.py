"""Actual external blocks are snapshotted before native LCU composition."""

from dataclasses import replace
from hashlib import sha1
from itertools import product
from pathlib import Path
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.ecosystem.cirq._block_encoding import import_block_encoding
from flagquantum.simulation.unitary import get_unitary
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_linear_combination_catalog import (
    coefficient_gates as coefficient_gates,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_dense_sources() -> None:
    import qualtran._infra.controlled as controlled
    from qualtran.bloqs.block_encoding import linear_combination
    from qualtran.bloqs.for_testing import matrix_gate

    for module, blob in (
        (controlled, "6fabe760e793f1b39f825d76f80e92b6e5fa5c28"),
        (matrix_gate, "7665416abde2cc9f8dec35a112758cadcf460658"),
        (linear_combination, "a3d8aa7fafa7a6f3fcaf095826751209427cd255"),
    ):
        raw = Path(module.__file__).read_bytes()
        assert sha1(b"blob %d\0" % len(raw) + raw).hexdigest() == blob


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


@pytest.mark.parametrize("width", (1, 2))
def test_original_dense_source_leaf_is_not_synthesized_or_given_fake_cost(
    width: int,
) -> None:
    import numpy as np
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.bloqs.for_testing.matrix_gate import MatrixGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = Unitary(MatrixGate.random(width, random_state=np.random.RandomState(1234)))
    block = import_block_encoding(source)
    assert block.models is None
    for controlled, adjoint in product((False, True), repeat=2):
        original = source.controlled() if controlled else source
        if adjoint:
            original = original.adjoint()
        n = width + int(controlled)
        assert get_cost_value(original, QubitCount()) == n
        with pytest.raises(DecomposeNotImplementedError, match="MatrixGate"):
            get_cost_value(original, QECGatesCost())
        op = block.make_on(
            system=tuple(range(int(controlled), n)),
            control=(0,) if controlled else (),
            adjoint=adjoint,
        )
        gates = Circuit(n).append(op).to_ir().instructions
        assert len(gates) == 1 and gates[0].matrix is not None
        expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
        torch.testing.assert_close(
            get_unitary(Circuit(n).append(op).to_ir(), dtype=torch.complex128),
            expected,
            atol=2e-12,
            rtol=0,
        )
        with pytest.raises(ValueError, match="No independent resource model"):
            block.resource_model(controlled=controlled, adjoint=adjoint)


@pytest.mark.parametrize("row", range(10))
@pytest.mark.parametrize("adjoint", (False, True))
def test_all_original_random_dense_families_replace_native_lcu_leaves(
    row: int,
    adjoint: bool,
    coefficient_gates: Any,
) -> None:
    from qualtran.bloqs.block_encoding.unitary import Unitary

    from flagquantum.compiler.registers.control.indexed_component import (
        indexed_component_workspace,
    )
    from tests.unit.sparse_register_execution import execute_sparse_register
    from tests.unit.test_linear_combination_catalog import _bind_coefficient_fixture

    _, dense = coefficient_gates
    gates, coefficients = dense[row]
    native, _, previous, work = _bind_coefficient_fixture(
        gates, coefficients, 9, inverse=adjoint
    )
    blocks = tuple(import_block_encoding(Unitary(gate)) for gate in gates)
    replacement = replace(native, block_encodings=blocks)
    n = replacement.system_bitsize
    end = n + replacement.ancilla_bitsize
    through = end + replacement.resource_bitsize
    prep, prep_bits = replacement.coefficient_preparation.workspace_requirements
    traversal = indexed_component_workspace((replacement.select.selection_bitsize,), 0)
    operation = replacement.make_on(
        tuple(block.bind_child for block in blocks),
        system=tuple(range(n)),
        ancilla=tuple(range(n, end)),
        resource=tuple(range(end, through)),
        prepare_workspace=work[:prep],
        prepare_classical_bits=tuple(range(prep_bits)),
        traversal_workspace=work[:traversal],
        traversal_classical_bits=tuple(range(traversal)),
        workspaces=((),) * len(blocks),
        classical_bits=((),) * len(blocks),
        adjoint=adjoint,
    )
    actual = Circuit(previous.n_qubits).append(operation).to_ir()
    expected = previous.to_ir()
    assert actual.n_wires == expected.n_wires
    assert len(actual.instructions) == len(expected.instructions)
    for left, right in zip(actual.instructions, expected.instructions, strict=True):
        assert (left.name, left.wires, left.params, left.metadata) == (
            right.name,
            right.wires,
            right.params,
            right.metadata,
        )
        if right.matrix is not None:
            torch.testing.assert_close(left.matrix, right.matrix, atol=1e-14, rtol=0)
        else:
            assert left.matrix is None
    # Every family is compared instruction by instruction above. Reuse the
    # accepted catalogue's numerical evidence; keep one coherent directional
    # replacement check here instead of interpreting every identical stream twice.
    if row != 0:
        return
    initial = {
        i << (actual.n_wires - n): complex(i + 1, i % 3 - 1) for i in range(1 << n)
    }
    norm = sum(abs(value) ** 2 for value in initial.values()) ** 0.5
    initial = {basis: value / norm for basis, value in initial.items()}
    for seed in (19, 73):
        observed = execute_sparse_register(actual, initial.copy(), seed=seed)
        reference = execute_sparse_register(expected, initial.copy(), seed=seed)
        assert observed == pytest.approx(reference, abs=2e-12, rel=0)
        assert all((basis & ((1 << len(work)) - 1)) == 0 for basis in observed)
