"""Audit complete walk metadata through its existing native execution path."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha1
from pathlib import Path

import pytest
import torch
from sympy import Symbol

from flagquantum import Circuit
from flagquantum.algorithms.primitives.identity_prepare import IdentityPrepare
from flagquantum.algorithms.primitives.ising_walk import ising_walk
from flagquantum.algorithms.primitives.lcu_block_encoding import LCUBlockEncoding
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.algorithms.primitives.qubitization_walk import (
    QubitizationWalk,
    source_walk_signature,
)
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_relative_controlled_swap import assert_source_dependencies

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def complete_frozen_sources() -> None:
    from qualtran.bloqs import qubitization
    from qualtran.bloqs.qubitization import qubitization_walk_operator

    assert (
        qubitization.QubitizationWalkOperator
        is qubitization_walk_operator.QubitizationWalkOperator
    )
    root = Path(qubitization_walk_operator.__file__).parent
    for name, blob in (
        ("__init__.py", "d259f1657c3ca0e9d3ac08ed554667b9be79cd57"),
        ("qubitization_walk_operator.py", "117515efb1e42c58b0fab451c454fbd756585f74"),
        (
            "qubitization_walk_operator_test.py",
            "a9dfd4398e3ec279e9a2a08a99d31c6ca22bc52b",
        ),
    ):
        raw = (root / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("lcu", (False, True))
def test_original_preparation_reflection_signature_and_complete_operators(
    lcu: bool,
) -> None:
    import cirq
    from qualtran import QAny
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        LCUBlockEncoding as SourceLCU,
    )
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        SelectBlockEncoding as SourceSelect,
    )
    from qualtran.bloqs.multiplexers.select_pauli_lcu import (
        SelectPauliLCU as SourcePauli,
    )
    from qualtran.bloqs.qubitization.qubitization_walk_operator import (
        QubitizationWalkOperator,
    )
    from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity

    original_select = SourcePauli(
        1, 1, (cirq.DensePauliString("X"), cirq.DensePauliString("Z"))
    )
    original_prepare = PrepareIdentity(original_select.selection_registers)
    source = QubitizationWalkOperator(
        (SourceLCU if lcu else SourceSelect)(original_select, original_prepare)
    )
    selection = Register("selection", bounded_unsigned_register_type(1, 2))
    target = Register("target", opaque_register_type(1))
    prepare = IdentityPrepare((selection,)).bind()
    pauli = SelectPauliLCU(1, 1, ("X", "Z"))

    def emit(
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        adjoint: bool,
    ) -> tuple[Instruction, ...]:
        return pauli.make_bounded_on(
            selection=ports["selection"],
            target=ports["target"],
            workspace=work,
            classical_bits=slots,
        ).emit()

    block = SelectBlockEncoding((("selection", 1),), (("target", 1),), emit, prepare)
    native = QubitizationWalk(LCUBlockEncoding(block) if lcu else block)
    assert native.prepare is prepare
    assert native.reflection.global_phase == source.reflect.global_phase == -1
    assert (
        native.reflection.selection_registers
        == native.block_encoding.signal_state.selection_registers
    )
    assert native.wire_symbols() == tuple(
        cirq.circuit_diagram_info(source).wire_symbols
    )
    typed = source_walk_signature((selection,), (selection,), (target,), ())
    assert [(r.name, r.bitsize) for r in typed] == [
        (r.name, r.bitsize) for r in source.signature
    ]
    assert [(r.name, r.bitsize) for r in native.signature] == [
        (r.name, r.bitsize) for r in typed
    ]
    assert list(typed)[0].dtype.encoding == "uint"
    assert list(native.signature)[0].dtype.encoding == "opaque"
    assert source.prepare is original_prepare
    if not lcu:
        unequal = QubitizationWalkOperator(
            SourceSelect(
                original_select,
                PrepareIdentity((SourceRegister("selection", QAny(1)),)),
            )
        )
        with pytest.raises(ValueError, match="duplicate registers"):
            _ = unequal.selection_registers
        with pytest.raises(ValueError, match="unequal same-name"):
            source_walk_signature(
                (selection,),
                (Register("selection", opaque_register_type(1)),),
                (target,),
                (),
            )

    select_cost = pauli.make_bounded_on(
        selection=(0,), target=(1,), workspace=(), classical_bits=()
    ).resource_model()
    identity_cost = ComponentCost("PrepareIdentity", 1)
    assert_source_model(
        native.resource_model(select_cost, identity_cost, identity_cost), source
    )
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        4, 4
    )
    logical = torch.tensor(
        [complex(i + 1, 5 - 2 * i) for i in range(8)], dtype=torch.complex128
    )
    logical /= logical.norm()
    for column in range(4):
        state = torch.zeros(8, dtype=torch.complex128)
        state[column] = 1
        initial = torch.zeros(16, dtype=torch.complex128)
        initial[::2] = state
        circuit = Circuit(4, inputs=initial, dtype=torch.complex128)
        operation = native.make_on(
            selection={"selection": (1,)}, target=(2,), reflection_workspace=(3,)
        )
        circuit.append(operation)
        expected = torch.zeros_like(initial)
        expected[:8:2] = matrix[:, column]
        torch.testing.assert_close(
            circuit.state().flatten(), expected, rtol=0, atol=2e-12
        )
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = logical
    circuit = Circuit(4, inputs=initial, dtype=torch.complex128).append(operation)
    expected = torch.zeros_like(initial)
    expected[::2] = (
        torch.kron(torch.eye(2, dtype=torch.complex128), matrix.contiguous()) @ logical
    )
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    assert (
        CircuitIR.from_json(circuit.to_ir().to_json()).instructions
        == circuit.to_ir().instructions
    )
    circuit.append(
        native.make_on(
            selection={"selection": (1,)},
            target=(2,),
            reflection_workspace=(3,),
            adjoint=True,
        )
    )
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=3e-12)

    # Replace the complete child with a frozen ecosystem capture, keeping consumers.
    captured = import_register_emitter(original_select)
    assert captured.requirements() == (0, 0)
    replaced_block = replace(block, select=captured)
    replacement = QubitizationWalk(
        LCUBlockEncoding(replaced_block) if lcu else replaced_block
    )
    imported = Circuit(4, inputs=initial, dtype=torch.complex128).append(
        replacement.make_on(
            selection={"selection": (1,)}, target=(2,), reflection_workspace=(3,)
        )
    )
    torch.testing.assert_close(imported.state().flatten(), expected, rtol=0, atol=3e-12)
    assert not any(g.name == "measure" for g in imported.to_ir().instructions)
    assert_source_dependencies(imported.to_ir().instructions, operation.instructions)
    assert Counter(g.name for g in operation.instructions) == {
        "i": 2,
        "x": 6,
        "cx": 1,
        "cz": 2,
        "phase": 1,
    }


def test_selection_union_preserves_full_metadata_and_rejects_unequal_names() -> None:
    n = Symbol("n", integer=True, positive=True)
    word = Register("signal", opaque_register_type(n), shape=(2,))
    other = Register("address", opaque_register_type(3))
    target = Register("system", opaque_register_type(1))
    signature = source_walk_signature((other, word), (word,), (target,), ())
    assert tuple(signature) == (other, word, target)
    with pytest.raises(ValueError, match="unequal same-name"):
        source_walk_signature(
            (word,), (Register("signal", opaque_register_type(n)),), (), ()
        )
    with pytest.raises(ValueError, match="unequal same-name"):
        source_walk_signature(
            (Register("a", opaque_register_type(2)),),
            (Register("a", bounded_unsigned_register_type(2, 3)),),
            (),
            (),
        )


def test_complete_ising_factory_metadata_and_original_doc_examples() -> None:
    from qualtran.bloqs.qubitization import qubitization_walk_operator as reference

    source = reference._walk_op.make()
    native, _ = ising_walk(4, 0.2)
    assert native.walk.alpha == source.sum_of_lcu_coefficients
    assert {r.name: r.bitsize for r in native.walk.signature} == {
        r.name: r.bitsize for r in source.signature
    }
    assert len(native.walk.wire_symbols()) == source.signature.n_qubits()
    assert len(reference._QUBITIZATION_WALK_DOC.examples) == 3
