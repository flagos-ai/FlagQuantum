"""Import local external gates while retaining the original QROM array program."""

from collections import Counter
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import BlackBoxPrepare
from flagquantum.compiler.optimization.local_unitary_synthesis import local_unitary_ir
from flagquantum.compiler.registers.lookup.source_qrom import source_qrom_workspace_size
from flagquantum.compiler.registers.phase.programmable_rotation_array import (
    _rotation_batches,
    programmable_rotation_array_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_programmable_rotation_array_file import (
    source_module as source_module,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("synthesize", [False, True])
@pytest.mark.parametrize(
    "word,kappa,custom,opaque",
    [
        (1, 1, False, False),
        (1, 2, False, False),
        (2, 1, False, False),
        (2, 2, False, False),
        (1, 2, True, False),
        (1, 2, False, True),
        (2, 2, False, True),
    ],
)
def test_external_local_gate_array_full_operators_and_owned_replacement(
    source_module: Any,
    word: int,
    kappa: int,
    custom: bool,
    opaque: bool,
    synthesize: bool,
) -> None:
    import cirq
    from qualtran.bloqs.rotations import programmable_rotation_gate_array_test as cases

    angles = ((1, 2, 3), (3, 0, 1))
    rotation: cirq.Gate = cirq.MatrixGate(
        np.exp(0.19j) * cirq.testing.random_unitary(1 << word, random_state=17)
    )
    interleaver: cirq.Gate = cirq.MatrixGate(
        np.exp(-0.27j) * cirq.testing.random_unitary(1 << word, random_state=23)
    )
    if opaque:

        class UnitaryHook(cirq.Gate):
            def __init__(self, data: Any) -> None:
                self._data = np.array(data, dtype=np.complex128, copy=True)

            def _num_qubits_(self) -> int:
                return len(self._data).bit_length() - 1

            def _unitary_(self) -> Any:
                return self._data.copy()

            def __pow__(self, exponent: float) -> Any:
                return UnitaryHook(
                    cirq.unitary(cirq.MatrixGate(self._data) ** exponent)
                )

        rotation = UnitaryHook(cirq.unitary(rotation))
        interleaver = UnitaryHook(cirq.unitary(interleaver))
    source = (
        cases.construct_custom_prga(*angles, kappa=kappa, rotation_gate=rotation)
        if custom
        else source_module.ProgrammableRotationGateArray(
            *angles,
            kappa=kappa,
            rotation_gate=rotation,
            interleaved_unitaries=(interleaver,),
        )
    )
    if kappa == 2 and not custom and not opaque:
        from qualtran import DecomposeTypeError
        from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

        assert get_cost_value(source, QubitCount()) == (6 if word == 1 else 8)
        if word == 1:
            model = get_cost_value(source, QECGatesCost())
            assert (
                model.and_bloq,
                model.clifford,
                model.rotation,
                model.measurement,
            ) == (10, 24, 17, 10)
        else:
            with pytest.raises(DecomposeTypeError, match=r"call graph for C\[And\]"):
                get_cost_value(source, QECGatesCost())
    original = import_register_emitter(source)
    emitter = replace(
        original,
        snapshots=(
            CircuitIR.from_json(original.snapshots[0].to_json()),
            CircuitIR.from_json(original.snapshots[1].to_json()),
        ),
    )
    ports = tuple((r.name, int(r.total_bits())) for r in source.signature)
    logical = sum(bits for _, bits in ports)
    bound = BlackBoxPrepare(ports, (), emitter)
    source_ops = []
    for adjoint in (False, True):
        work, slots = emitter.requirements()
        ir = (
            Circuit(logical + work)
            .append(
                bound.make_on(
                    selection=tuple(range(logical)),
                    junk=(),
                    workspace=tuple(range(logical, logical + work)),
                    classical_bits=tuple(range(slots)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert isinstance(ir, CircuitIR)
        source_ops.append(logical_operator(ir, logical))
    selection = (0, 1)
    load = tuple(range(2, 2 + kappa))
    target = tuple(range(2 + kappa, 2 + kappa + word))
    extra = (logical - 1,) if custom else ()
    # Conversion stops at each explicit local matrix gate, never at the whole
    # source algorithm. The compiler receives only native instruction payloads.
    powers = {
        power: np.array(cirq.unitary(rotation ** (2.0 ** (-1 - power))))
        for power in (0, 1)
    }
    controlled = {}
    for power, matrix in powers.items():
        local = np.eye(2 * len(matrix), dtype=np.complex128)
        local[len(matrix) :, len(matrix) :] = matrix
        controlled[power] = local.tolist()

    def rotate(
        control: int, targets: tuple[int, ...], power: int
    ) -> tuple[Instruction, ...]:
        if synthesize:
            return local_unitary_ir(powers[power], targets, control=control)
        return (Instruction("any", (control, *targets), matrix=controlled[power]),)

    interleaved: tuple[tuple[Instruction, ...], ...] = (
        ((Instruction("cx", (*target, *extra)),),)
        if custom
        else ((Instruction("any", target, matrix=cirq.unitary(interleaver).tolist()),),)
    )
    if synthesize and not custom:
        interleaved = (local_unitary_ir(np.asarray(cirq.unitary(interleaver)), target),)
    work = max(
        source_qrom_workspace_size(0, (2,), (delta,))
        for delta, _ in _rotation_batches(angles, kappa)
    )
    gates = programmable_rotation_array_ir(
        selection,
        load,
        target,
        angles,
        tuple(range(logical, logical + work)),
        tuple(range(work)),
        rotation=rotate,
        interleaved=interleaved,
        interleaved_target=extra,
    )
    native_ir = CircuitIR.from_json(CircuitIR(logical + work, gates).to_json())
    assert (
        Circuit(logical + work)
        .append(FieldOperation(logical + work, gates))
        .to_ir()
        .to_dict()["instructions"]
        == native_ir.to_dict()["instructions"]
    )
    native_op = logical_operator(native_ir, logical)
    torch.testing.assert_close(source_ops[0], native_op, atol=1e-11, rtol=0)
    torch.testing.assert_close(
        source_ops[1] @ native_op,
        torch.eye(1 << logical, dtype=torch.complex128),
        atol=1e-11,
        rtol=0,
    )
    if synthesize:
        assert all(g.matrix is None and g.name != "any" for g in native_ir.instructions)
        assert (
            Counter(g.name for g in native_ir.instructions)["measure"]
            == Counter(g.name for g in original.snapshots[0].instructions)["measure"]
        )
    else:
        assert Counter(g.name for g in original.snapshots[0].instructions) == Counter(
            g.name for g in native_ir.instructions
        )
        assert (
            max(len(g.wires) for g in native_ir.instructions if g.name == "any")
            == word + 1
        )
    # Independently assemble each per-address target operator with zero angle
    # load. This also proves relative phases across selection sectors.
    target_dimension = 1 << (word + len(extra))
    for address in range(3):
        local = np.eye(target_dimension, dtype=np.complex128)
        for stage, row in enumerate(angles):
            for power in (1, 0):
                if (row[address] >> power) & 1:
                    matrix = powers[power]
                    if extra:
                        matrix = np.kron(matrix, np.eye(2))
                    local = matrix @ local
            if stage == 0:
                matrix = (
                    cirq.unitary(cirq.CNOT) if custom else cirq.unitary(interleaver)
                )
                local = matrix @ local
        offset = address << (kappa + word + len(extra))
        columns = tuple(offset + i for i in range(target_dimension))
        expected = torch.zeros((1 << logical, target_dimension), dtype=torch.complex128)
        expected[list(columns), :] = torch.tensor(local, dtype=torch.complex128)
        torch.testing.assert_close(
            native_op[:, list(columns)], expected, atol=1e-11, rtol=0
        )


def test_provider_decomposition_precedes_unitary_protocol(source_module: Any) -> None:
    import cirq

    class DecomposableGate(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def _unitary_(self) -> Any:
            raise AssertionError("Do not contract a gate with a declared decomposition")

        def _decompose_(self, qubits: Any) -> Any:
            yield cirq.H(qubits[0])

    source = source_module.ProgrammableRotationGateArray(
        (1, 2, 3),
        (3, 0, 1),
        kappa=2,
        rotation_gate=cirq.Z,
        interleaved_unitaries=(DecomposableGate(),),
    )
    snapshots = import_register_emitter(source).snapshots
    assert all(not any(g.name == "any" for g in ir.instructions) for ir in snapshots)
    forward = logical_operator(snapshots[0], 5)
    backward = logical_operator(snapshots[1], 5)
    torch.testing.assert_close(
        backward @ forward, torch.eye(32, dtype=torch.complex128), atol=1e-11, rtol=0
    )
