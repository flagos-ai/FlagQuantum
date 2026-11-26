"""Keep original QROAM logical refusals distinct from bounded SDK matrix views."""

from importlib import import_module
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.ecosystem.cirq._register_graph import _leaf_program
from tests.unit.test_cirq_decomposition_file import locked_source as locked_source
from tests.unit.test_legacy_arithmetic_factories import _library

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("polarity", (0, 1))
@pytest.mark.parametrize("blocked", (False, True))
def test_original_logical_refusals_and_sdk_control_decomposition_cycle(
    polarity: int,
    blocked: bool,
    locked_source: Any,
) -> None:
    import cirq
    import numpy as np

    upstream = import_module("qualtran")
    qroam = import_module("qualtran.bloqs.data_loading.qroam_clean")
    controls = import_module("qualtran._infra.controlled")
    child = qroam.QROAMClean.build_from_data(
        np.array([0, 1, 1, 0]), target_bitsizes=(1,), log_block_sizes=(int(blocked),)
    )
    source = controls.Controlled(child, upstream.CtrlSpec(cvs=polarity))
    with pytest.raises(upstream.DecomposeTypeError, match="non-thru"):
        source.decompose_bloq()
    with pytest.raises(ValueError, match="non-thru"):
        source._tensor_data()
    operation = source.on(*cirq.LineQubit.range(cirq.num_qubits(source)))
    assert cirq.decompose_once(operation, default=None) is None
    pure_sdk = cirq.ControlledGate(child, control_values=(polarity,)).on(
        *operation.qubits
    )
    (again,) = cirq.decompose_once(pure_sdk)
    if polarity:
        assert again == operation  # A cycle, not an independent gate recipe.
    else:
        assert type(again.gate).__name__ == "ControlledViaAnd"
        first, positive, last = cirq.decompose_once(again)
        assert first == last == cirq.X(operation.qubits[0])
        assert positive.gate == controls.Controlled(child, upstream.CtrlSpec())
        assert cirq.decompose_once(positive, default=None) is None


@pytest.mark.parametrize("polarity", (0, 1))
@pytest.mark.parametrize("datasets", (1, 2))
def test_small_original_unitary_fallback_preserves_right_outputs_and_all_complex_columns(
    polarity: int,
    datasets: int,
    locked_source: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cirq
    import numpy as np

    upstream = import_module("qualtran")
    qroam = import_module("qualtran.bloqs.data_loading.qroam_clean")
    controls = import_module("qualtran._infra.controlled")
    child = qroam.QROAMClean.build_from_data(
        *(np.array(values) for values in ((0, 1), (1, 0))[:datasets]),
        target_bitsizes=(1,) * datasets,
        log_block_sizes=(0,),
    )
    source = controls.Controlled(child, upstream.CtrlSpec(cvs=polarity))
    registers = _library(source, 0).entries[0].declaration.registers
    assert registers is not None
    record = _leaf_program(source, "main", registers)
    program = record if isinstance(record, ProgramRecord) else ProgramRecord((record,))
    assert verify_program_record(program) == ()
    width = 2 + datasets
    bound = bind_register_program(
        program,
        ((0,), (1,)),
        clean=tuple(range(2, width)),
        borrowed=(width,),
        allow_workspace_outputs=True,
    )
    assert tuple(op.name for op in bound.circuit.instructions) == (
        ("any",) if polarity else ("x", "any", "x")
    )
    (gate,) = (op for op in bound.circuit.instructions if op.name == "any")
    assert gate.matrix is not None
    original = torch.tensor(cirq.unitary(source), dtype=torch.complex128)
    # The source SDK wraps the positive atom in X gates for a zero control.
    atom = torch.tensor(
        cirq.unitary(controls.Controlled(child, upstream.CtrlSpec())),
        dtype=torch.complex128,
    )
    torch.testing.assert_close(gate.matrix, atom, atol=0, rtol=0)
    expected = torch.kron(original, torch.eye(2, dtype=torch.complex128))
    assert bound.circuit.n_wires == width + 1
    with monkeypatch.context() as patch:
        patch.setitem(__import__("sys").modules, "qualtran", None)
        patch.setitem(__import__("sys").modules, "cirq", None)
        ir = CircuitIR.from_json(bound.circuit.to_json())
        for column in range(8):
            state = torch.zeros(1 << (width + 1), dtype=torch.complex128)
            position = ((column >> 1) << (datasets + 1)) | (column & 1)
            state[position] = 1
            circuit = Circuit(width + 1, inputs=state, dtype=state.dtype).append(
                FieldOperation(width + 1, ir.instructions)
            )
            torch.testing.assert_close(
                circuit.state().flatten(), expected[:, position], atol=4e-14, rtol=0
            )
    # One materialized unitary, no measured-erasure primitive claim or inferred cost.
    assert not any(op.name in ("measure", "reset") for op in ir.instructions)


def test_original_nonblocked_four_row_unitary_recurses_without_a_gate_decomposition(
    locked_source: Any,
) -> None:
    import cirq
    import numpy as np

    child = import_module(
        "qualtran.bloqs.data_loading.qroam_clean"
    ).QROAMClean.build_from_data(
        np.array([0, 1, 1, 0]), target_bitsizes=(1,), log_block_sizes=(0,)
    )
    source = child.controlled()
    assert cirq.num_qubits(source) == 4
    with pytest.raises(RecursionError):
        cirq.unitary(source)
