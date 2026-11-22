"""Complete Fourier source graphs, joint amplitudes and clean native execution."""

from dataclasses import replace
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms import rotations_ui_export as native
from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.fourier_decomposition import fourier_child
from flagquantum.algorithms.primitives.approximate_qft import (
    approximate_qft_ir,
    approximate_qft_workspace_size,
)
from flagquantum.algorithms.primitives.fermionic_fourier import (
    two_bit_fermionic_fourier_ir,
    two_bit_fermionic_fourier_program,
)
from flagquantum.algorithms.primitives.qft import _qft_circuit
from flagquantum.algorithms.primitives.qft_phase_gradient import (
    qft_phase_gradient_ir,
    qft_phase_gradient_requirements,
)
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.phase.approximate_qft_component import (
    approximate_qft_program,
)
from flagquantum.compiler.registers.phase.gradient_qft_component import (
    gradient_qft_program,
)
from flagquantum.compiler.registers.phase.textbook_qft_component import (
    textbook_qft_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import Operation, ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_field_product_decomposition import source_walk
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_rotations_ui_export import originals as originals

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOTS = (
    "two_bit_ffft",
    "qft_text_book",
    "qft_phase_gradient_small",
    "approximate_qft_small",
    "approximate_qft_from_epsilon",
)


@pytest.mark.parametrize(
    "identifier,exact", tuple(zip(ROOTS, (1, 6, 4, 1, 1), strict=True))
)
def test_complete_files_recursive_calls_and_original_drawings(
    identifier: str, exact: int, originals: tuple[Any, dict[str, Any]]
) -> None:
    rows = source_walk(identifier, originals[1][identifier], catalog=native)
    assert sum(r["disposition"] == "exact_graph" for r in rows) == exact
    assert all(r.get("drawing", "source_exact") == "source_exact" for r in rows)


def test_original_symbolic_root_remains_atomic(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    from qualtran import DecomposeTypeError

    with pytest.raises(DecomposeTypeError):
        originals[1]["symbolic_qft"].decompose_bloq()
    assert native.decomposition("symbolic_qft") is None


def _case(kind: str, width: int, reverse: bool) -> tuple[Any, ProgramRecord, int]:
    from qualtran.bloqs.qft.approximate_qft import ApproximateQFT
    from qualtran.bloqs.qft.qft_phase_gradient import QFTPhaseGradient
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook

    if kind == "textbook":
        return (
            QFTTextBook(width, reverse),
            textbook_qft_program(width, with_reverse=reverse),
            0,
        )
    if kind == "gradient":
        return (
            QFTPhaseGradient(width, reverse),
            gradient_qft_program(width, with_reverse=reverse),
            qft_phase_gradient_requirements(width)[0],
        )
    return (
        ApproximateQFT(width, width, reverse),
        approximate_qft_program(width, width, with_reverse=reverse),
        approximate_qft_workspace_size(width, width),
    )


@pytest.mark.parametrize(
    "kind,width,reverse",
    tuple(product(("textbook", "gradient", "approximate"), (2, 3), (False, True))),
)
def test_complete_joint_source_operator_all_histories_and_exact_recovery(
    kind: str, width: int, reverse: bool
) -> None:
    import cirq

    source, record, inverse_work = _case(kind, width, reverse)
    expected = torch.tensor(cirq.unitary(source), dtype=torch.complex128)
    signature = Signature(record.functions[0].registers or ())
    logical = sum(p.bitsize for p in signature.lefts())
    contiguous = tuple(range(logical))
    inputs = (
        (contiguous[:width], contiguous[width:])
        if kind != "textbook"
        else (contiguous,)
    )
    provisional = bind_register_program(
        record, inputs, clean=tuple(range(logical, logical + 32))
    )
    used = set(contiguous)
    for event in provisional.events:
        used.update(event.wires)
    work = max(max(used) + 1 - logical, inverse_work)
    bound = bind_register_program(
        record, inputs, clean=tuple(range(logical, logical + work))
    )
    assert copy_component_program(record) == record
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=1e-11, rtol=0
    )
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    total = logical + work + 1
    wires = tuple(reversed(range(logical)))
    inputs = (wires[:width], wires[width:]) if kind != "textbook" else (wires,)
    clean = tuple(range(logical + 1, total))
    forward = bind_register_program(record, inputs, clean=clean)
    if kind == "gradient":
        slots = qft_phase_gradient_requirements(width)[1]
        inverse = qft_phase_gradient_ir(
            wires[:width],
            wires[width:],
            clean[:inverse_work],
            tuple(range(slots)),
            with_reverse=reverse,
            adjoint=True,
        )
    elif kind == "approximate":
        inverse = approximate_qft_ir(
            wires[:width],
            wires[width:],
            clean[:inverse_work],
            tuple(range(inverse_work)),
            with_reverse=reverse,
            adjoint=True,
        )
    else:
        ir = _qft_circuit(width, with_reverse=reverse).adjoint().to_ir()
        assert isinstance(ir, CircuitIR)
        inverse = tuple(
            replace(g, wires=tuple(wires[w] for w in g.wires)) for g in ir.instructions
        )
    raw = torch.arange(1, 2 ** (logical + 1) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, raw.flip(0))
    amplitudes /= amplitudes.norm()
    order = (*wires, logical, *clean)
    restore = tuple(order.index(w) for w in range(total))
    initial_view = torch.zeros((2**logical, 2, 2**work), dtype=torch.complex128)
    initial_view[:, :, 0] = amplitudes.reshape(2**logical, 2)
    initial = initial_view.reshape((2,) * total).permute(restore).flatten()
    target_view = torch.zeros_like(initial_view)
    target_view[:, :, 0] = expected @ amplitudes.reshape(2**logical, 2)
    target = target_view.reshape((2,) * total).permute(restore).flatten()
    gates = forward.circuit.instructions
    for instructions, start, finish in (
        (gates, initial, target),
        (inverse, target, initial),
    ):
        count = sum(g.name == "measure" for g in instructions)
        for history in product((0, 1), repeat=count):
            branch = forced_branch(start, instructions, history)
            torch.testing.assert_close(
                branch, finish / 2 ** (count / 2), atol=1e-11, rtol=0
            )
            dirty = (
                branch.reshape((2,) * total).permute(order).reshape(-1, 2**work)[:, 1:]
            )
            torch.testing.assert_close(
                dirty, torch.zeros_like(dirty), atol=1e-11, rtol=0
            )
    circuit = DynamicCircuit(total, inputs=initial, dtype=initial.dtype)
    circuit.append(FieldOperation(total, gates))
    result = run_dynamic(circuit, shots=4, seed=103, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(4, -1), target.expand(4, -1), atol=1e-11, rtol=0
    )


def test_fermionic_complete_operator_and_original_ignored_adjoint() -> None:
    from qualtran.bloqs.qft.two_bit_ffft import TwoBitFFFT

    source = TwoBitFFFT(2, 3)
    original_inverse = TwoBitFFFT(2, 3, is_adjoint=True)
    assert source.decompose_bloq() == original_inverse.decompose_bloq()
    expected = torch.tensor(
        source.decompose_bloq().tensor_contract(), dtype=torch.complex128
    )
    matrices = []
    for inverse in (False, True):
        record = two_bit_fermionic_fourier_program(2, 3, adjoint=inverse)
        bound = bind_register_program(record, ((0,), (1,)))
        matrices.append(logical_operator(bound.circuit, 2))
        assert bound.circuit.instructions == two_bit_fermionic_fourier_ir(
            0, 1, 2, 3, adjoint=inverse
        )
        assert bound.events == ()
    torch.testing.assert_close(matrices[0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(matrices[1], expected.mH, atol=1e-12, rtol=0)
    assert not torch.allclose(matrices[1], expected)


@pytest.mark.parametrize("identifier", ("qft_text_book", "qft_phase_gradient_small"))
def test_child_qualification_rejects_mutated_complete_definitions(
    identifier: str,
) -> None:
    record = native.decomposition(identifier)
    assert record is not None
    changed = replace(
        record,
        functions=tuple(
            replace(
                f,
                blocks=tuple(
                    replace(
                        b,
                        operations=tuple(_alter_gate(op) for op in b.operations),
                    )
                    for b in f.blocks
                ),
            )
            for f in record.functions
        ),
    )
    found = 0
    for f in record.functions[1:]:
        if fourier_child(record, f.name, native.attributes(identifier)) is not None:
            assert fourier_child(changed, f.name, native.attributes(identifier)) is None
            found += 1
    assert found


def _alter_gate(operation: Operation) -> Operation:
    if operation.opcode == "cx":
        return replace(operation, operands=operation.operands[::-1])
    if operation.opcode == "h":
        return replace(operation, opcode="z")
    return operation
