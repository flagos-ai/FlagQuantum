"""Execute a source indexed callable returning a general one-qubit unitary."""

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from itertools import product
from math import sqrt
from typing import Any

import pytest
import torch

from flagquantum import Circuit
from flagquantum.algorithms.primitives.black_box_prepare import (
    BlackBoxPrepare,
    _identity_prepare,
)
from flagquantum.algorithms.primitives.select_block_encoding import SelectBlockEncoding
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests._reference_dependencies import unavailable_reference
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_target_position_file import source as source

try:
    import cirq

    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("controls", (0, 1, 2))
def test_matrix_callable_full_operator_all_branches_and_clean_work(
    source: Any, controls: int, record_property: Callable[[str, object], None]
) -> None:
    from qualtran import BQUInt, QAny, Register
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    gates = (
        cirq.rx(0.3),
        cirq.MatrixGate(cirq.testing.random_unitary(2, random_state=1084)),
    )

    def nth_gate(index: int) -> cirq.Gate:
        return gates[index]

    original = source.ApplyGateToLthQubit(
        Register("selection", BQUInt(1, 2)),
        nth_gate,
        control_regs=() if not controls else (Register("enable", QAny(controls)),),
    )
    captured = import_register_emitter(original)
    emitter = replace(
        captured,
        snapshots=(
            CircuitIR.from_json(captured.snapshots[0].to_json()),
            CircuitIR.from_json(captured.snapshots[1].to_json()),
        ),
    )
    selection, targets = emitter.layout[:-1], emitter.layout[-1:]
    operation = SelectBlockEncoding(
        selection, targets, emitter, BlackBoxPrepare(selection, (), _identity_prepare)
    )
    logical = controls + 3
    work, slots = emitter.requirements()
    for adjoint in (False, True):
        directed = original.adjoint() if adjoint else original
        matrix = torch.as_tensor(directed.tensor_contract(), dtype=torch.complex128)
        ir = (
            Circuit(logical + work, dtype=torch.complex128)
            .append(
                operation.make_on(
                    selection=tuple(range(controls + 1)),
                    target=(controls + 1, controls + 2),
                    workspace=tuple(range(logical, logical + work)),
                    classical_bits=tuple(range(slots)),
                    adjoint=adjoint,
                )
            )
            .to_ir()
        )
        assert ir.instructions == emitter.snapshots[int(adjoint)].instructions
        restored = CircuitIR.from_json(ir.to_json())
        assert restored.to_json() == ir.to_json()
        ir = restored
        counts = Counter(gate.name for gate in ir.instructions)
        record_property(f"primitive_{adjoint}", dict(counts))
        record_property(
            f"source_model_{adjoint}", get_cost_value(directed, QECGatesCost()).asdict()
        )
        record_property(
            f"source_peak_{adjoint}", get_cost_value(directed, QubitCount())
        )
        for column in range(1 << logical):
            initial = torch.zeros(1 << ir.n_wires, dtype=torch.complex128)
            initial[column << work] = 1
            expected = torch.zeros_like(initial)
            expected[:: 1 << work] = matrix[:, column] / sqrt(1 << counts["measure"])
            for history in product((0, 1), repeat=counts["measure"]):
                torch.testing.assert_close(
                    forced_branch(initial, ir.instructions, history),
                    expected,
                    atol=4e-12,
                    rtol=0,
                )
