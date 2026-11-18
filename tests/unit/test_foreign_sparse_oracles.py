"""Actual foreign oracle objects bind through owned snapshots to Hermitian blocks."""

from itertools import product

import pytest
import torch

from examples.algorithms.custom_sparse_oracles import custom_hermitian_block
from flagquantum import Circuit
from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
    HermitianSparseMatrixBlockEncoding,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    GateCost,
    estimate_component_resources,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._sparse_oracles import (
    import_column_oracle,
    import_sqrt_entry_oracle,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import assert_source_wire_streams
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_hermitian_custom_oracles import frozen_source as frozen_source
from tests.unit.test_hermitian_custom_oracles import source_oracles

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("controlled", (False, True))
def test_full_source_block_and_entry_signature_metadata(
    symbolic: bool, controlled: bool
) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import TopLeftRowColumnOracle
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        SparseMatrixHermitian,
        UniformSqrtEntryOracle,
    )
    from sympy import Symbol

    from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
        source_hermitian_sparse_signature,
        source_sqrt_entry_signature,
    )
    from tests.unit.test_first_quantization_resource_model import (
        assert_first_quantization_signature,
    )

    width = Symbol("n", positive=True, integer=True) if symbolic else 2
    entry = UniformSqrtEntryOracle(width, 0.3)
    block = SparseMatrixHermitian(TopLeftRowColumnOracle(width), entry, 0)
    source = block.controlled() if controlled else block
    assert_first_quantization_signature(
        source_hermitian_sparse_signature(width, controlled=controlled),
        source.signature,
    )
    assert_first_quantization_signature(
        source_sqrt_entry_signature(width), entry.signature
    )
    assert str(custom_hermitian_block(2)) == str(block)


@pytest.mark.parametrize("width", (1, 2))
@pytest.mark.parametrize("controlled", (False, True))
def test_external_custom_objects_full_stream_model_branches_and_recovery(
    width: int, controlled: bool
) -> None:
    source, original = source_oracles(width), custom_hermitian_block(width)
    column, entry = original.column_oracle, original.entry_oracle
    imported_column = import_column_oracle(
        source.col_oracle,
        models=(column.resource_model(), column.resource_model(adjoint=True)),
    )
    imported_entry = import_sqrt_entry_oracle(
        source.entry_oracle,
        models=(entry.resource_model(), entry.resource_model(adjoint=True)),
    )
    assert imported_column.binding_swaps == imported_entry.binding_swaps == (0, 0)
    native = HermitianSparseMatrixBlockEncoding(imported_column, imported_entry)
    assert native.requirements() == (1, 1)
    reference = source.controlled() if controlled else source
    assert_source_model(native.resource_model(controlled=controlled), reference)
    shift = int(controlled)
    through = 2 * width + 2 + shift
    args = dict(
        system=tuple(range(shift, shift + width)),
        ancilla=tuple(range(shift + width, through)),
        control=0 if controlled else None,
        workspace=(through,),
        classical_bits=(1,),
    )
    op = native.make_on(**args)
    assert_source_wire_streams(op.instructions, original.make_on(**args).instructions)
    matrix = torch.as_tensor(reference.tensor_contract(), dtype=torch.complex128)
    values = torch.arange(1 << (through + 1), dtype=torch.float64)
    state = torch.complex(values + 1, values % 7 - 3).reshape(1 << through, 2)
    state /= state.norm()

    def extend(vector: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((1 << through, 2, 2, 2), dtype=torch.complex128)
        out[:, 0, :, 1] = vector
        return out.flatten()

    initial, expected = extend(state), extend((matrix @ state).flip(1))
    prefix = Instruction(
        "measure", (through + 2,), metadata={"is_dynamic": True, "classical_bit": 0}
    )
    suffix = Instruction("x", (through + 1,), metadata={"conditions": ((0, 1),)})
    for outcomes in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(initial, (prefix, *op.instructions, suffix), (1, *outcomes)),
            expected / 2,
            atol=5e-12,
            rtol=0,
        )
    inverse = native.make_on(**args, adjoint=True)
    for outcomes in product((0, 1), repeat=4):
        torch.testing.assert_close(
            forced_branch(initial, (*op.instructions, *inverse.instructions), outcomes),
            initial / 4,
            atol=7e-12,
            rtol=0,
        )


@pytest.mark.parametrize("width", (1, 2))
def test_original_uniform_objects_preserve_matrix_and_source_model(width: int) -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import TopLeftRowColumnOracle
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        SparseMatrixHermitian,
        UniformSqrtEntryOracle,
    )

    from flagquantum.algorithms.primitives.hermitian_sparse_matrix_block_encoding import (
        UniformSqrtEntryOracle as NativeEntry,
    )
    from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
        TopLeftRowColumnOracle as NativeColumn,
    )
    from flagquantum.simulation.unitary import get_unitary

    source_column, source_entry = (
        TopLeftRowColumnOracle(width),
        UniformSqrtEntryOracle(width, 0.3),
    )
    column, entry = NativeColumn(width), NativeEntry(width, 0.3)
    imported = HermitianSparseMatrixBlockEncoding(
        import_column_oracle(
            source_column,
            models=(column.resource_model(), column.resource_model(adjoint=True)),
        ),
        import_sqrt_entry_oracle(
            source_entry,
            models=(entry.resource_model(), entry.resource_model(adjoint=True)),
        ),
    )
    assert imported.requirements() == (0, 0)
    source = SparseMatrixHermitian(source_column, source_entry, 0)
    assert_source_model(imported.resource_model(), source)
    op = imported.make_on(
        system=tuple(range(width)), ancilla=tuple(range(width, 2 * width + 2))
    )
    torch.testing.assert_close(
        get_unitary(CircuitIR(2 * width + 2, op.instructions), dtype=torch.complex128),
        torch.as_tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=4e-12,
        rtol=0,
    )


def test_original_banded_objects_all_leaf_branches_and_directional_storage() -> None:
    from qualtran.bloqs.block_encoding.sparse_matrix import (
        SymmetricBandedRowColumnOracle,
    )

    from flagquantum.algorithms.primitives.sparse_matrix_oracles import (
        SymmetricBandedRowColumnOracle as NativeColumn,
    )

    source, model = SymmetricBandedRowColumnOracle(2, 1), NativeColumn(2, 1)
    imported = import_column_oracle(
        source, models=(model.resource_model(), model.resource_model(adjoint=True))
    )
    assert imported.requirements() == (3, 2)
    assert model.requirements() == (
        3,
        1,
    )  # Snapshot import retains both recorded events.
    values = torch.arange(32, dtype=torch.float64)
    state = torch.complex(values + 1, values % 5 - 2).reshape(4, 4, 2)
    state[3] = 0  # Only declared ordinals 0,1,2 belong to the source input domain.
    state /= state.norm()
    output = torch.zeros_like(state)
    for ordinal in range(3):
        for index in range(4):
            output[(ordinal + index - 1) % 4, index] = state[ordinal, index]

    def extend(value: torch.Tensor) -> torch.Tensor:
        result = torch.zeros((16, 8, 2), dtype=torch.complex128)
        result[:, 0, :] = value.reshape(16, 2)
        return result.flatten()

    for adjoint, incoming, expected in ((False, state, output), (True, output, state)):
        assert_source_model(
            imported.resource_model(adjoint=adjoint),
            source.adjoint() if adjoint else source,
        )
        gates = imported.emit((0, 1), (2, 3), (4, 5, 6), (7, 8), adjoint=adjoint)
        for outcomes in product((0, 1), repeat=2):
            torch.testing.assert_close(
                forced_branch(extend(incoming), gates, outcomes),
                extend(expected) / 2,
                atol=5e-12,
                rtol=0,
            )


def test_original_same_name_sided_gate_wrapper_failure_is_not_patched() -> None:
    import cirq
    import numpy as np

    column = source_oracles(1).col_oracle
    manager = cirq.GreedyQubitManager(prefix="work")
    left, index = manager.qalloc(2)
    with pytest.raises(ValueError, match="RIGHT register"):
        column.as_cirq_op(
            qubit_manager=manager,
            l=np.asarray([left]),
            i=np.asarray([index]),
        )


@pytest.mark.parametrize("new_output", (False, True))
def test_returned_wire_aliases_and_allocated_outputs_are_restored_and_priced(
    new_output: bool,
) -> None:
    import cirq
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        UniformSqrtEntryOracle,
    )

    class ForeignEntry:
        system_bitsize = 1
        epsilon = 0.0
        signature = UniformSqrtEntryOracle(1, 0.3).signature

        def adjoint(self):
            return self

        def as_cirq_op(self, qubit_manager, **ports):
            if new_output:
                fresh = qubit_manager.qalloc(1)[0]
                return cirq.SWAP(ports["q"][0], fresh), ports | {"q": [fresh]}
            return None, ports | {"i": ports["j"], "j": ports["i"]}

    body = ComponentCost(
        "ExternalBody", 4 if new_output else 3, GateCost(clifford=int(new_output))
    )
    entry = import_sqrt_entry_oracle(ForeignEntry(), models=(body, body))
    assert entry.binding_swaps == (1, 1)
    assert entry.requirements() == ((1, 0) if new_output else (0, 0))
    model = estimate_component_resources(entry.resource_model())
    assert model.gates.clifford == 1 + int(new_output)
    gates = entry.emit(0, (1,), (2,), (3,) if new_output else (), ())
    values = torch.arange(8, dtype=torch.float64)
    state = torch.complex(values + 1, values % 3 - 1)
    state /= state.norm()
    if new_output:
        initial = torch.stack((state, torch.zeros_like(state)), dim=1).flatten()
        expected = initial
    else:
        initial = state
        expected = state.reshape(2, 2, 2).permute(0, 2, 1).flatten()
    actual = (
        Circuit(4 if new_output else 3, inputs=initial, dtype=initial.dtype)
        .compose(CircuitIR(4 if new_output else 3, gates))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=0)


@pytest.mark.parametrize("fault", ("borrow", "external", "alias"))
def test_unknown_or_dirty_storage_is_not_assumed_clean(fault: str) -> None:
    import cirq
    from qualtran.bloqs.block_encoding.sparse_matrix_hermitian import (
        UniformSqrtEntryOracle,
    )

    class ForeignEntry:
        system_bitsize = 1
        epsilon = 0.0
        signature = UniformSqrtEntryOracle(1, 0.3).signature

        def adjoint(self):
            return self

        def as_cirq_op(self, qubit_manager, **ports):
            if fault == "borrow":
                qubit_manager.qborrow(1)
            if fault == "alias":
                return None, ports | {"j": ports["i"]}
            return cirq.X(cirq.NamedQubit("unowned")), ports

    model = ComponentCost("External", 3)
    with pytest.raises(ValueError):
        import_sqrt_entry_oracle(ForeignEntry(), models=(model, model))
